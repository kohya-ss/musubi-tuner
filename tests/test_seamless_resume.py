import argparse
import os
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.image_video_dataset import _sorted_glob
from musubi_tuner.training.accelerator_setup import EpochSeededRandomSampler, resolve_logging_dir
from musubi_tuner.training.tensorboard_logs import trim_tensorboard_log_to_checkpoint
from musubi_tuner.training.training_state import (
    TRAINING_STATE_FILE,
    TrainingProgressState,
    configure_resume_tracker_init_kwargs,
)


class TrainingProgressStateTest(unittest.TestCase):
    def test_json_round_trip_preserves_resume_and_logging_state(self):
        state = TrainingProgressState(
            global_step=17,
            epoch=2,
            step_in_epoch=5,
            max_train_steps=100,
            num_batches_per_epoch=12,
            num_update_steps_per_epoch=6,
            gradient_accumulation_steps=2,
            num_processes=1,
            seed=42,
            logging_dir=os.path.abspath("logs/run"),
            log_with="all",
            tracker_name="network_train",
            wandb_run_id="wandb-id",
            session_id=123,
            training_started_at=456.0,
            checkpoint_saved_at=789.0,
            timestep_range_pool=[(0.1, 0.2)],
            loss_list=[1.0, 0.5],
            loss_total=1.5,
        )

        with tempfile.TemporaryDirectory() as state_dir:
            state.write_json(state_dir)
            self.assertTrue(os.path.isfile(os.path.join(state_dir, TRAINING_STATE_FILE)))
            restored = TrainingProgressState.read_json(state_dir)

        self.assertIsNotNone(restored)
        self.assertTrue(restored.loaded)
        self.assertEqual(restored.state_dict(), state.state_dict())
        self.assertEqual(restored.timestep_range_pool, [(0.1, 0.2)])

    def test_resolve_logging_dir_reuses_resume_directory(self):
        with tempfile.TemporaryDirectory() as root:
            state_dir = os.path.join(root, "run-step00000010-state")
            log_dir = os.path.join(root, "logs", "original")
            TrainingProgressState(logging_dir=log_dir).write_json(state_dir)
            args = argparse.Namespace(
                resume=state_dir,
                resume_from_huggingface=False,
                logging_dir=os.path.join(root, "logs"),
                log_prefix="new_",
            )

            self.assertEqual(resolve_logging_dir(args), os.path.abspath(log_dir))

    def test_version_one_state_marks_epoch_logs_for_migration(self):
        state = TrainingProgressState()
        state.load_state_dict({"version": 1, "global_step": 80, "epoch": 8})

        self.assertEqual(state.epoch_log_step_mode, "global_step")
        self.assertEqual(state.version, 2)

    def test_resume_rolls_trackers_back_to_checkpoint_boundary(self):
        state = TrainingProgressState(global_step=2000, log_with="all", wandb_run_id="wandb-id")
        state.loaded = True
        init_kwargs = {
            "tensorboard": {"flush_secs": 10, "purge_step": 9999},
            "wandb": {"name": "run", "id": "old-id", "resume": "must"},
        }

        configure_resume_tracker_init_kwargs(init_kwargs, state)

        self.assertEqual(init_kwargs["tensorboard"]["purge_step"], 2001)
        self.assertEqual(init_kwargs["tensorboard"]["flush_secs"], 10)
        self.assertEqual(init_kwargs["wandb"]["resume_from"], "wandb-id?_step=2000")
        self.assertEqual(init_kwargs["wandb"]["name"], "run")
        self.assertNotIn("id", init_kwargs["wandb"])
        self.assertNotIn("resume", init_kwargs["wandb"])

    def test_tensorboard_events_are_physically_trimmed_to_state(self):
        try:
            from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
            from tensorboard.compat.proto import event_pb2, summary_pb2
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
        except ImportError:
            self.skipTest("tensorboard is not installed")

        with tempfile.TemporaryDirectory() as logging_dir:
            run_dir = os.path.join(logging_dir, "network_train")
            writer = EventFileWriter(run_dir)

            def add_scalar(step, wall_time, value, tag="loss/current"):
                summary = summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag, simple_value=value)])
                writer.add_event(event_pb2.Event(step=step, wall_time=wall_time, summary=summary))

            add_scalar(10, 100.0, 1.0)
            add_scalar(20, 200.0, 2.0)
            add_scalar(30, 220.0, 3.0)  # newer step
            add_scalar(5, 300.0, 4.0)  # low step written after the checkpoint
            add_scalar(20, 205.0, 2.5)  # retry: latest value for the same tag+step wins
            add_scalar(2, 205.0, 0.75, tag="loss/epoch")  # valid epoch-axis summary
            add_scalar(3, 220.0, 0.5, tag="loss/epoch")  # orphaned epoch-axis summary
            writer.add_event(
                event_pb2.Event(
                    step=21,
                    wall_time=230.0,
                    session_log=event_pb2.SessionLog(status=event_pb2.SessionLog.START),
                )
            )
            writer.close()

            result = trim_tensorboard_log_to_checkpoint(logging_dir, "network_train", 20, 210.0)

            # The writer opened after resume must sort after the immutable
            # compacted history, or TensorBoard may tail the wrong file.
            resumed_writer = EventFileWriter(run_dir)
            resumed_writer.close()
            event_names = sorted(name for name in os.listdir(run_dir) if name.startswith("events.out.tfevents."))
            self.assertTrue(any(name.endswith(".musubi-compacted") for name in event_names))
            self.assertFalse(event_names[-1].endswith(".musubi-compacted"))

            remaining_steps = []
            remaining_values = []
            has_session_log = False
            for name in os.listdir(run_dir):
                if not name.startswith("events.out.tfevents."):
                    continue
                for event in LegacyEventFileLoader(os.path.join(run_dir, name)).Load():
                    if event.HasField("summary"):
                        remaining_steps.append(event.step)
                        remaining_values.extend(value.simple_value for value in event.summary.value)
                    has_session_log = has_session_log or event.HasField("session_log")

            self.assertEqual(remaining_steps, [10, 20, 2])
            self.assertEqual(remaining_values, [1.0, 2.5, 0.75])
            self.assertFalse(has_session_log)
            self.assertEqual(result.event_files, 1)
            self.assertGreaterEqual(result.removed_events, 4)

    def test_version_one_epoch_logs_are_migrated_to_epoch_axis(self):
        try:
            from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
            from tensorboard.compat.proto import event_pb2, summary_pb2
            from tensorboard.summary.writer.event_file_writer import EventFileWriter
        except ImportError:
            self.skipTest("tensorboard is not installed")

        with tempfile.TemporaryDirectory() as logging_dir:
            run_dir = os.path.join(logging_dir, "network_train")
            writer = EventFileWriter(run_dir)
            for global_step, loss in ((10, 1.0), (20, 0.5)):
                summary = summary_pb2.Summary(
                    value=[summary_pb2.Summary.Value(tag="loss/epoch", simple_value=loss)]
                )
                writer.add_event(event_pb2.Event(step=global_step, wall_time=100.0 + global_step, summary=summary))
            writer.close()

            trim_tensorboard_log_to_checkpoint(
                logging_dir,
                "network_train",
                20,
                125.0,
                epoch_log_step_mode="global_step",
                num_update_steps_per_epoch=10,
            )

            remaining_steps = []
            for name in os.listdir(run_dir):
                if name.startswith("events.out.tfevents."):
                    for event in LegacyEventFileLoader(os.path.join(run_dir, name)).Load():
                        if event.HasField("summary"):
                            remaining_steps.append(event.step)

            self.assertEqual(remaining_steps, [1, 2])


class DeterministicDataOrderTest(unittest.TestCase):
    def test_cache_file_enumeration_is_stable(self):
        with mock.patch(
            "musubi_tuner.dataset.image_video_dataset.glob.glob",
            return_value=["cache/003.safetensors", "cache/001.safetensors", "cache/002.safetensors"],
        ):
            self.assertEqual(
                _sorted_glob("cache/*.safetensors"),
                ["cache/001.safetensors", "cache/002.safetensors", "cache/003.safetensors"],
            )

    def test_epoch_sampler_is_reproducible_without_consuming_torch_rng(self):
        shared_epoch = SimpleNamespace(value=3)
        data = list(range(20))
        torch.manual_seed(1234)
        rng_before = torch.get_rng_state().clone()

        first = list(EpochSeededRandomSampler(data, 99, shared_epoch))
        second = list(EpochSeededRandomSampler(data, 99, shared_epoch))

        self.assertEqual(first, second)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        shared_epoch.value = 4
        self.assertNotEqual(first, list(EpochSeededRandomSampler(data, 99, shared_epoch)))

    def test_mid_epoch_resume_uses_the_uninterrupted_sampler_suffix(self):
        shared_epoch = SimpleNamespace(value=8)
        data = list(range(20))
        uninterrupted = list(EpochSeededRandomSampler(data, 1024, shared_epoch))
        reconstructed = list(EpochSeededRandomSampler(data, 1024, shared_epoch))

        self.assertEqual(reconstructed[7:], uninterrupted[7:])

    def test_bucket_shuffle_uses_the_supplied_rng_only(self):
        def make_manager():
            return BucketBatchManager({(16, 16): list(range(8))}, batch_size=2, num_timestep_buckets=3)

        random.seed(1234)
        global_state = random.getstate()
        first = make_manager()
        first.shuffle(random.Random(77))

        self.assertEqual(global_state, random.getstate())

        second = make_manager()
        second.shuffle(random.Random(77))
        self.assertEqual(first.buckets, second.buckets)
        self.assertEqual(first.bucket_batch_indices, second.bucket_batch_indices)
        self.assertEqual(first.timestep_pool, second.timestep_pool)


if __name__ == "__main__":
    unittest.main()
