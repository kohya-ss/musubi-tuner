from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import torch
import torch.distributed as dist
from accelerate import Accelerator

import musubi_tuner.training.full_finetune as full_finetune
from musubi_tuner.training.full_finetune import save_state_all_ranks
from test_full_finetune_runtime import TinyFullTrainer, make_args


GLOO_UNAVAILABLE = not dist.is_available() or not dist.is_gloo_available()
GLOO_UNAVAILABLE_REASON = "PyTorch distributed Gloo backend is unavailable"


def _run_distributed_workers(worker: str, output_dir: Path) -> None:
    init_method = (output_dir / "gloo-init").resolve().as_uri()
    processes = []
    for rank in range(2):
        env = os.environ.copy()
        env.update(
            {
                "LOCAL_RANK": str(rank),
                "LOCAL_WORLD_SIZE": "2",
                "OMP_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
                "RANK": str(rank),
                "WORLD_SIZE": "2",
            }
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--distributed-worker",
            worker,
            "--output-dir",
            str(output_dir),
            "--rank",
            str(rank),
            "--init-method",
            init_method,
        ]
        processes.append(
            subprocess.Popen(
                command,
                cwd=Path(__file__).parents[1],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )

    try:
        results = [process.communicate(timeout=120) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
    failures = [
        f"rank {rank} exited {process.returncode}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        for rank, (process, (stdout, stderr)) in enumerate(zip(processes, results))
        if process.returncode != 0
    ]
    assert not failures, "distributed workers failed\n" + "\n".join(failures)


@pytest.mark.skipif(GLOO_UNAVAILABLE, reason=GLOO_UNAVAILABLE_REASON)
def test_two_process_full_model_update_keeps_rank_weights_equal(tmp_path):
    _run_distributed_workers("ddp-update", tmp_path)

    rank_weights = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
    assert torch.equal(rank_weights[0]["linear.weight"], rank_weights[1]["linear.weight"])
    assert torch.equal(rank_weights[0]["linear.bias"], rank_weights[1]["linear.bias"])
    assert not torch.equal(rank_weights[0]["linear.weight"], torch.full((1, 1), 0.25))


@pytest.mark.skipif(GLOO_UNAVAILABLE, reason=GLOO_UNAVAILABLE_REASON)
def test_state_retention_waits_for_delayed_rank_and_second_barrier(tmp_path):
    _run_distributed_workers("delayed-state", tmp_path)

    events = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in tmp_path.glob("event-*.json")}
    save_events = {f"event-save_state-rank{rank}" for rank in range(2)}
    assert save_events | {"event-retention-rank0"} <= events.keys()

    second_barrier_entries = {f"event-barrier_2_enter-rank{rank}" for rank in range(2)}
    assert second_barrier_entries <= events.keys()
    for rank in range(2):
        assert events[f"event-barrier_2_exit-rank{rank}"]["order"] < events[f"event-continue-rank{rank}"]["order"]


@pytest.mark.skipif(GLOO_UNAVAILABLE, reason=GLOO_UNAVAILABLE_REASON)
@pytest.mark.parametrize(
    ("worker", "main_exception_type", "main_message", "operation"),
    [
        ("retention-error", "ValueError", "retention exploded", "state retention"),
        ("checkpoint-error", "OSError", "checkpoint export exploded", "checkpoint export"),
    ],
)
def test_main_rank_action_error_stops_both_processes_at_the_same_operation(
    tmp_path,
    worker,
    main_exception_type,
    main_message,
    operation,
):
    _run_distributed_workers(worker, tmp_path)

    results = [json.loads((tmp_path / f"failure-{worker}-rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    assert all(not result["continued"] for result in results)
    assert all(result["reached_post_action_barrier"] for result in results)
    assert results[0]["exception_type"] == main_exception_type
    assert results[0]["exception_message"] == main_message
    assert results[1]["exception_type"] == "RuntimeError"
    assert operation in results[1]["exception_message"]
    assert "main process" in results[1]["exception_message"]


@pytest.mark.skipif(GLOO_UNAVAILABLE, reason=GLOO_UNAVAILABLE_REASON)
def test_checkpoint_retention_delete_error_stops_both_processes_at_the_same_operation(tmp_path):
    worker = "checkpoint-retention-error"
    _run_distributed_workers(worker, tmp_path)

    results = [json.loads((tmp_path / f"retention-{worker}-rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    assert all(not result["continued"] for result in results)
    assert all(result["reached_post_action_barrier"] for result in results)
    assert results[0]["exception_type"] == "OSError"
    assert results[0]["exception_message"] == "checkpoint delete exploded"
    assert results[1]["exception_type"] == "RuntimeError"
    assert "checkpoint retention" in results[1]["exception_message"]
    assert "main process" in results[1]["exception_message"]


@pytest.mark.skipif(GLOO_UNAVAILABLE, reason=GLOO_UNAVAILABLE_REASON)
@pytest.mark.parametrize(
    ("worker", "expected_checkpoint"),
    [
        ("checkpoint-retention-step", full_finetune.train_utils.get_step_ckpt_name("tiny", 0)),
        ("checkpoint-retention-epoch", full_finetune.train_utils.get_epoch_ckpt_name("tiny", 0)),
    ],
)
def test_two_process_step_and_epoch_checkpoint_retention_deletes_once(
    tmp_path,
    worker,
    expected_checkpoint,
):
    _run_distributed_workers(worker, tmp_path)

    results = [json.loads((tmp_path / f"retention-{worker}-rank{rank}.json").read_text(encoding="utf-8")) for rank in range(2)]
    assert all(result["continued"] for result in results)
    assert all(result["reached_post_action_barrier"] for result in results)
    removal_records = [json.loads(path.read_text(encoding="utf-8")) for path in tmp_path.glob(f"remove-{worker}-*.json")]
    assert removal_records == [{"rank": 0, "checkpoint": expected_checkpoint}]


def _record_event(output_dir: Path, label: str, rank: int, order: int) -> None:
    payload = {"label": label, "rank": rank, "time_ns": time.monotonic_ns(), "order": order}
    (output_dir / f"event-{label}-rank{rank}.json").write_text(json.dumps(payload), encoding="utf-8")


def _run_ddp_update(output_dir: Path, rank: int, init_method: str) -> None:
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1, mixed_precision="no")
    assert accelerator.process_index == rank
    trainer = TinyFullTrainer()
    full_finetune.prepare_accelerator = lambda _args: accelerator
    full_finetune.clean_memory_on_device = lambda _device: None
    full_finetune.sai_model_spec.build_metadata = lambda *args, **kwargs: {
        "modelspec.architecture": "tiny",
        "is_lora": str(kwargs["is_lora"]),
    }

    trainer.train(
        make_args(
            output_dir,
            blocks_to_swap=0,
            compile=False,
            sample_prompts=None,
            max_train_steps=1,
        )
    )
    torch.save(trainer.raw_model.state_dict(), output_dir / f"rank-{rank}.pt")


class _DelayedStateAccelerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.rank = dist.get_rank()
        self.is_main_process = self.rank == 0
        self.device = torch.device("cpu")
        self.num_processes = 2
        self.barrier_index = 0
        self.event_order = 0

    def record(self, label: str) -> None:
        self.event_order += 1
        _record_event(self.output_dir, label, self.rank, self.event_order)

    def save_state(self, _state_dir) -> None:
        if self.rank == 1:
            time.sleep(0.75)
        self.record("save_state")

    def wait_for_everyone(self) -> None:
        self.barrier_index += 1
        label = f"barrier_{self.barrier_index}"
        self.record(f"{label}_enter")
        dist.barrier()
        self.record(f"{label}_exit")

    def reduce(self, tensor, reduction="sum"):
        assert reduction == "sum"
        self.record("reduce_enter")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        self.record("reduce_exit")
        return tensor

    def print(self, *_args, **_kwargs):
        pass


def _run_delayed_state(output_dir: Path, rank: int, init_method: str) -> None:
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        accelerator = _DelayedStateAccelerator(output_dir)

        def retention_callback():
            save_events = [output_dir / f"event-save_state-rank{worker_rank}.json" for worker_rank in range(2)]
            if not all(path.exists() for path in save_events):
                raise AssertionError("retention started before both ranks completed save_state")
            accelerator.record("retention")

        save_state_all_ranks(accelerator, argparse.Namespace(), output_dir / "state", retention_callback)
        second_barrier_entries = [output_dir / f"event-barrier_2_enter-rank{worker_rank}.json" for worker_rank in range(2)]
        if not all(path.exists() for path in second_barrier_entries):
            raise AssertionError("training continued before both ranks entered the second barrier")
        accelerator.record("continue")
    finally:
        dist.destroy_process_group()


def _run_main_action_error(output_dir: Path, rank: int, init_method: str, worker: str) -> None:
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        accelerator = _DelayedStateAccelerator(output_dir)
        continued = False
        caught = None
        try:
            if worker == "retention-error":

                def fail_retention():
                    raise ValueError("retention exploded")

                save_state_all_ranks(
                    accelerator,
                    argparse.Namespace(),
                    output_dir / "state",
                    fail_retention,
                )
            else:
                trainer = TinyFullTrainer()
                args = make_args(output_dir, blocks_to_swap=0, compile=False, sample_prompts=None)
                raw_model = torch.nn.Linear(1, 1)
                full_finetune.sai_model_spec.build_metadata = lambda *args, **kwargs: {}

                def fail_checkpoint_export(*_args, **_kwargs):
                    raise OSError("checkpoint export exploded")

                full_finetune.save_file = fail_checkpoint_export
                trainer._save_full_finetune_model(
                    accelerator,
                    args,
                    raw_model,
                    raw_model,
                    "sync-test.safetensors",
                    {},
                    {},
                    1,
                    1,
                    torch.float32,
                )
            continued = True
        except Exception as error:
            caught = error

        dist.barrier()
        result = {
            "continued": continued,
            "exception_type": type(caught).__name__ if caught is not None else None,
            "exception_message": str(caught) if caught is not None else None,
            "reached_post_action_barrier": True,
        }
        (output_dir / f"failure-{worker}-rank{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def _run_checkpoint_retention(output_dir: Path, rank: int, init_method: str, worker: str) -> None:
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2)
    try:
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1, mixed_precision="no")
        assert accelerator.process_index == rank
        accelerator.wait_for_everyone = lambda: None
        accelerator.end_training = lambda: None
        trainer = TinyFullTrainer()
        full_finetune.prepare_accelerator = lambda _args: accelerator
        full_finetune.clean_memory_on_device = lambda _device: None
        full_finetune.sai_model_spec.build_metadata = lambda *args, **kwargs: {
            "modelspec.architecture": "tiny",
            "is_lora": str(kwargs["is_lora"]),
        }
        trainer._save_full_finetune_model = lambda *_args, **_kwargs: None
        removal_count = 0

        def remove_checkpoint(_args, ckpt_name):
            nonlocal removal_count
            if worker == "checkpoint-retention-error":
                raise OSError("checkpoint delete exploded")
            removal_count += 1
            record_path = output_dir / f"remove-{worker}-rank{rank}-{removal_count}.json"
            record_path.write_text(json.dumps({"rank": rank, "checkpoint": ckpt_name}), encoding="utf-8")

        trainer._remove_checkpoint = remove_checkpoint
        retention_args = {
            "save_every_n_steps": 1,
            "save_last_n_steps": 1,
        }
        if worker == "checkpoint-retention-epoch":
            retention_args = {
                "save_every_n_epochs": 1,
                "save_last_n_epochs": 1,
            }

        continued = False
        caught = None
        try:
            trainer.train(
                make_args(
                    output_dir,
                    blocks_to_swap=0,
                    compile=False,
                    sample_prompts=None,
                    max_train_steps=2,
                    **retention_args,
                )
            )
            continued = True
        except Exception as error:
            caught = error

        if dist.is_initialized():
            dist.barrier()
        result = {
            "continued": continued,
            "exception_type": type(caught).__name__ if caught is not None else None,
            "exception_message": str(caught) if caught is not None else None,
            "reached_post_action_barrier": True,
        }
        (output_dir / f"retention-{worker}-rank{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _worker_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distributed-worker",
        choices=(
            "ddp-update",
            "delayed-state",
            "retention-error",
            "checkpoint-error",
            "checkpoint-retention-error",
            "checkpoint-retention-step",
            "checkpoint-retention-epoch",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--init-method", required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.distributed_worker == "ddp-update":
        _run_ddp_update(args.output_dir, args.rank, args.init_method)
    elif args.distributed_worker == "delayed-state":
        _run_delayed_state(args.output_dir, args.rank, args.init_method)
    elif args.distributed_worker.startswith("checkpoint-retention-"):
        _run_checkpoint_retention(args.output_dir, args.rank, args.init_method, args.distributed_worker)
    else:
        _run_main_action_error(args.output_dir, args.rank, args.init_method, args.distributed_worker)


if __name__ == "__main__":
    _worker_main()
