"""Utilities for making TensorBoard history transactional with training state."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import os
import socket
import tempfile
import time
from typing import Optional


_EVENT_FILE_PREFIX = "events.out.tfevents."


@dataclass(frozen=True)
class TensorBoardTrimResult:
    event_files: int
    kept_events: int
    removed_events: int


def _normalize_epoch_summary_step(event, epoch_log_step_mode: str, num_update_steps_per_epoch: Optional[int]):
    """Migrate version-1 loss/epoch records from global-step to epoch-step."""

    if (
        epoch_log_step_mode != "global_step"
        or not num_update_steps_per_epoch
        or not event.HasField("summary")
        or not event.summary.value
        or any(value.tag != "loss/epoch" for value in event.summary.value)
    ):
        return event

    epoch_step, remainder = divmod(int(event.step), int(num_update_steps_per_epoch))
    if epoch_step <= 0 or remainder != 0:
        return event

    normalized_event = type(event)()
    normalized_event.CopyFrom(event)
    normalized_event.step = epoch_step
    return normalized_event


def trim_tensorboard_log_to_checkpoint(
    logging_dir: str,
    tracker_name: str,
    global_step: int,
    checkpoint_saved_at: Optional[float],
    epoch_log_step_mode: str = "epoch",
    num_update_steps_per_epoch: Optional[int] = None,
) -> TensorBoardTrimResult:
    """Physically remove TensorBoard events newer than a training state.

    TensorBoard's ``purge_step`` only records a session marker and leaves all
    orphaned records in the event files. Repeatedly resuming an older state can
    therefore still produce overlapping curves with some TensorBoard reload
    modes. This function compacts the main tracker run before a new writer is
    opened, retaining only events committed by the checkpoint.
    """

    run_dir = os.path.join(os.path.abspath(os.path.expanduser(logging_dir)), tracker_name)
    if not os.path.isdir(run_dir):
        return TensorBoardTrimResult(0, 0, 0)

    event_files = sorted(
        os.path.join(run_dir, name)
        for name in os.listdir(run_dir)
        if name.startswith(_EVENT_FILE_PREFIX) and os.path.isfile(os.path.join(run_dir, name))
    )
    if not event_files:
        return TensorBoardTrimResult(0, 0, 0)

    try:
        from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
        from tensorboard.summary.writer.record_writer import RecordWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard is required to compact resumed logs. Install tensorboard or disable TensorBoard logging."
        ) from exc

    kept_events = 0
    removed_events = 0
    migrated_events = 0
    wrote_file_version = False
    temp_fd, temp_path = tempfile.mkstemp(prefix=".musubi-tensorboard-", suffix=".tmp", dir=run_dir)
    os.close(temp_fd)

    try:
        # A state may have been reached more than once (for example, two
        # separate epoch-7 resumes both wrote steps 71..80). First locate the
        # last eligible record for every tag+step so compaction keeps the
        # branch that actually corresponds to the newest selected state.
        latest_summary_record: dict[tuple[int, str], int] = {}
        record_index = 0
        for event_file in event_files:
            for event in LegacyEventFileLoader(event_file).Load():
                record_index += 1
                if event.HasField("file_version") or event.HasField("session_log"):
                    continue
                if int(event.step) > int(global_step):
                    continue
                if checkpoint_saved_at is not None and float(event.wall_time) > float(checkpoint_saved_at):
                    continue
                event = _normalize_epoch_summary_step(event, epoch_log_step_mode, num_update_steps_per_epoch)
                if event.HasField("summary"):
                    for value in event.summary.value:
                        latest_summary_record[(int(event.step), value.tag)] = record_index

        with open(temp_path, "wb") as compacted_file:
            writer = RecordWriter(compacted_file)
            record_index = 0
            for event_file in event_files:
                for event in LegacyEventFileLoader(event_file).Load():
                    record_index += 1
                    # The compacted stream needs one header, not one per old
                    # writer session. SessionLog markers are obsolete after
                    # physical compaction and can trigger another purge.
                    if event.HasField("file_version"):
                        if wrote_file_version:
                            removed_events += 1
                            continue
                        wrote_file_version = True
                    elif event.HasField("session_log"):
                        removed_events += 1
                        continue
                    elif int(event.step) > int(global_step):
                        removed_events += 1
                        continue
                    elif checkpoint_saved_at is not None and float(event.wall_time) > float(checkpoint_saved_at):
                        removed_events += 1
                        continue

                    original_event = event
                    event = _normalize_epoch_summary_step(event, epoch_log_step_mode, num_update_steps_per_epoch)
                    if event is not original_event:
                        migrated_events += 1

                    event_to_write = event
                    if event.HasField("summary") and event.summary.value:
                        values_to_keep = [
                            value
                            for value in event.summary.value
                            if latest_summary_record.get((int(event.step), value.tag)) == record_index
                        ]
                        if not values_to_keep:
                            removed_events += 1
                            continue
                        if len(values_to_keep) != len(event.summary.value):
                            event_to_write = type(event)()
                            event_to_write.CopyFrom(event)
                            del event_to_write.summary.value[:]
                            event_to_write.summary.value.extend(values_to_keep)

                    writer.write(event_to_write.SerializeToString())
                    kept_events += 1
            compacted_file.flush()
            os.fsync(compacted_file.fileno())

        # Do not rewrite a clean, single event file unnecessarily.
        if removed_events == 0 and migrated_events == 0 and len(event_files) == 1:
            os.remove(temp_path)
            return TensorBoardTrimResult(1, kept_events, 0)

        backup_paths: list[tuple[str, str]] = []
        try:
            for index, source_path in enumerate(event_files):
                backup_path = os.path.join(run_dir, f".musubi-tensorboard-backup-{os.getpid()}-{index}.tmp")
                os.replace(source_path, backup_path)
                backup_paths.append((source_path, backup_path))

            # TensorBoard normally tails only the lexicographically newest
            # event file in a run. Give the immutable compacted history an
            # earlier timestamp than the SummaryWriter that is opened next;
            # otherwise both can be created in the same second and the
            # ``musubi-compacted`` suffix sorts after SummaryWriter's ``.0``,
            # making TensorBoard watch the static file instead of new logs.
            compacted_timestamp = max(0, int(time.time()) - 1)
            compacted_name = (
                f"{_EVENT_FILE_PREFIX}{compacted_timestamp:010d}.{socket.gethostname()}.{os.getpid()}.musubi-compacted"
            )
            compacted_path = os.path.join(run_dir, compacted_name)
            os.replace(temp_path, compacted_path)
        except Exception:
            # Restore every original file if publishing the compacted stream
            # fails. This keeps log cleanup transactional.
            for source_path, backup_path in reversed(backup_paths):
                if os.path.exists(backup_path):
                    os.replace(backup_path, source_path)
            raise
        else:
            for _, backup_path in backup_paths:
                if os.path.exists(backup_path):
                    os.remove(backup_path)

        return TensorBoardTrimResult(len(event_files), kept_events, removed_events)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main() -> None:
    """Compact an existing TensorBoard run without starting training."""

    parser = argparse.ArgumentParser(description="Remove TensorBoard events newer than a Musubi training state.")
    parser.add_argument("--state", required=True, help="Path to the resumable state directory")
    parser.add_argument("--logging_dir", default=None, help="Override the logging directory stored in the state")
    parser.add_argument("--tracker_name", default=None, help="Override the tracker name stored in the state")
    args = parser.parse_args()

    from musubi_tuner.training.training_state import TRAINING_STATE_FILE, TrainingProgressState

    state_dir = os.path.abspath(os.path.expanduser(args.state))
    state = TrainingProgressState.read_json(state_dir)
    if state is None:
        raise ValueError(f"No {TRAINING_STATE_FILE} found in state directory: {state_dir}")

    logging_dir = args.logging_dir or state.logging_dir
    if not logging_dir:
        raise ValueError("No logging directory is stored in the state; pass --logging_dir explicitly.")

    checkpoint_saved_at = state.checkpoint_saved_at
    if checkpoint_saved_at is None:
        checkpoint_saved_at = os.path.getmtime(os.path.join(state_dir, TRAINING_STATE_FILE))

    result = trim_tensorboard_log_to_checkpoint(
        logging_dir,
        args.tracker_name or state.tracker_name or "network_train",
        state.global_step,
        checkpoint_saved_at,
        state.epoch_log_step_mode,
        state.num_update_steps_per_epoch,
    )
    print(
        f"TensorBoard compacted to step {state.global_step}: "
        f"{result.event_files} event file(s), {result.kept_events} event(s) kept, "
        f"{result.removed_events} orphaned event(s) removed."
    )


if __name__ == "__main__":
    main()
