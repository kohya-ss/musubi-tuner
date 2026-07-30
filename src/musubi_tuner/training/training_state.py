"""Serializable application-level state for seamless training resume."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from typing import Any, Optional


TRAINING_STATE_FILE = "musubi_training_state.json"
TRAINING_STATE_VERSION = 2


@dataclass
class TrainingProgressState:
    """State that Accelerate does not track for the Musubi training loop.

    ``epoch`` is zero-based and ``step_in_epoch`` is the number of local
    dataloader batches already consumed in that epoch. Together they identify
    the next batch to train.
    """

    version: int = TRAINING_STATE_VERSION
    global_step: int = 0
    epoch: int = 0
    step_in_epoch: int = 0
    max_train_steps: int = 0
    num_batches_per_epoch: int = 0
    num_update_steps_per_epoch: int = 0
    gradient_accumulation_steps: int = 1
    num_processes: int = 1
    seed: int = 0
    logging_dir: Optional[str] = None
    log_with: Optional[str] = None
    tracker_name: Optional[str] = None
    wandb_run_id: Optional[str] = None
    session_id: Optional[int] = None
    training_started_at: Optional[float] = None
    checkpoint_saved_at: Optional[float] = None
    epoch_log_step_mode: str = "epoch"
    timestep_range_pool: list[tuple[float, float]] = field(default_factory=list)
    loss_list: list[float] = field(default_factory=list)
    loss_total: float = 0.0

    # Runtime-only marker; deliberately excluded from state_dict/JSON.
    loaded: bool = field(default=False, init=False, repr=False)

    def state_dict(self) -> dict[str, Any]:
        state = asdict(self)
        state.pop("loaded", None)
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        version = int(state.get("version", 0))
        if version > TRAINING_STATE_VERSION:
            raise ValueError(f"Training state version {version} is newer than supported version {TRAINING_STATE_VERSION}.")

        for key in self.state_dict():
            if key in state:
                setattr(self, key, state[key])

        # Version-1 seamless states were created while loss/epoch used the
        # optimization step as its TensorBoard x-axis. Mark them for a
        # one-time migration when the event history is compacted on resume.
        if "epoch_log_step_mode" not in state:
            self.epoch_log_step_mode = "global_step"

        self.timestep_range_pool = [tuple(item) for item in self.timestep_range_pool]
        self.loss_list = [float(item) for item in self.loss_list]
        self.loss_total = float(self.loss_total)
        self.version = TRAINING_STATE_VERSION
        self.loaded = True

    def write_json(self, state_dir: str) -> str:
        os.makedirs(state_dir, exist_ok=True)
        path = os.path.join(state_dir, TRAINING_STATE_FILE)
        temporary_path = path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(self.state_dict(), file, ensure_ascii=False, indent=2)
        os.replace(temporary_path, path)
        return path

    @classmethod
    def read_json(cls, state_dir: str) -> Optional["TrainingProgressState"]:
        path = os.path.join(state_dir, TRAINING_STATE_FILE)
        if not os.path.isfile(path):
            return None

        with open(path, "r", encoding="utf-8") as file:
            state_dict = json.load(file)

        state = cls()
        state.load_state_dict(state_dict)
        return state

    @classmethod
    def has_metadata(cls, state_dir: Optional[str]) -> bool:
        return bool(state_dir) and os.path.isfile(os.path.join(state_dir, TRAINING_STATE_FILE))


def configure_resume_tracker_init_kwargs(init_kwargs: dict, state: TrainingProgressState) -> dict:
    """Configure tracker rollback so logs cannot remain ahead of a checkpoint.

    ``global_step`` is the last fully committed optimization/logging step in a
    seamless state. TensorBoard therefore purges from the next step, while W&B
    rewinds the saved run to the committed step before accepting new history.
    """

    if not state.loaded:
        return init_kwargs

    if state.log_with in ("tensorboard", "all"):
        tensorboard_kwargs = init_kwargs.setdefault("tensorboard", {})
        tensorboard_kwargs["purge_step"] = state.global_step + 1

    if state.log_with in ("wandb", "all") and state.wandb_run_id:
        wandb_kwargs = init_kwargs.setdefault("wandb", {})
        # W&B does not allow resume/resume_from/fork_from together. Rewind is
        # required here: plain resume would keep any history newer than the
        # checkpoint and reject replacement steps as non-monotonic.
        wandb_kwargs.pop("id", None)
        wandb_kwargs.pop("resume", None)
        wandb_kwargs.pop("fork_from", None)
        wandb_kwargs["resume_from"] = f"{state.wandb_run_id}?_step={state.global_step}"

    return init_kwargs
