"""Real two-process Gloo contracts; these are not multi-GPU kernel tests."""

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import toml
import torch
from accelerate.state import AcceleratorState
from safetensors.torch import load_file

from musubi_tuner.training import dlssnr_trainer as trainer
from musubi_tuner.training.dlssnr_parser import setup_parser
from test_dlssnr_buckets import write_pairs
from test_dlssnr_training import SmallNR, small_inject


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize(
    "platform,device,backend,init_method",
    [
        ("nt", "cuda", "gloo", "env://?use_libuv=False"),
        ("nt", "cpu", "gloo", "env://?use_libuv=False"),
        ("posix", "cuda", "nccl", None),
        ("posix", "cpu", "gloo", None),
    ],
)
def test_launched_workers_use_platform_backend_even_with_one_process(
    monkeypatch, world_size, platform, device, backend, init_method
):
    from musubi_tuner.training import dlssnr_services as services

    for name, value in {
        "WORLD_SIZE": str(world_size),
        "LOCAL_WORLD_SIZE": str(world_size),
        "RANK": "0",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "ACCELERATE_USE_CPU": str(device == "cpu").lower(),
        "ACCELERATE_MIXED_PRECISION": "no",
        "ACCELERATE_USE_DEEPSPEED": "false",
        "ACCELERATE_USE_FSDP": "false",
        "ACCELERATE_USE_MEGATRON_LM": "false",
        "ACCELERATE_USE_SAGEMAKER": "false",
        "OMP_NUM_THREADS": "1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(services, "os", SimpleNamespace(name=platform, environ=os.environ))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)

    class InitializationCaptured(Exception):
        pass

    captured = {}

    def capture_init(**kwargs):
        captured.update(kwargs)
        raise InitializationCaptured

    # Keep Accelerate's backend selection real; stop before any GPU or collective work.
    monkeypatch.setattr(torch.distributed, "init_process_group", capture_init)
    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        with pytest.raises(InitializationCaptured):
            services.create_accelerator({"device": device, "gradient_accumulation_steps": 1})
        assert captured["backend"] == backend
        assert captured.get("init_method") == init_method
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)


def _args(root, name, *, lora=False, distributed=False, dropout=0.0, resume=False):
    args = setup_parser(lora=lora).parse_args(
        [
            "--dataset_config",
            str(root / "dataset.toml"),
            "--development_smoke",
            "--device",
            "cpu",
            "--seed",
            "4",
            "--optimizer_type",
            "SGD",
            "--optimizer_args",
            "momentum=0.4",
            "--learning_rate",
            "0.001",
            "--max_train_steps",
            "2",
            "--gradient_accumulation_steps",
            "2" if distributed else "4",
            "--output_dir",
            str(root / "output"),
            "--output_name",
            name,
            "--save_every_n_steps",
            "1",
            "--save_state",
        ]
    )
    if lora:
        args.network_dropout = dropout
    if (root / "fp8").exists():
        args.numerics_profile, args.fp8_base, args.fp8_scaled = "train_experimental", True, True
    if (root / "temporal").exists():
        args.training_mode = "temporal"
        args.sequence_length, args.burn_in, args.tbptt_length = 3, 1, 2
        args.loss_temporal = 0.1
    if resume:
        args.resume = root / "output" / name / "state-step000001"
    return args


def _data(root, *, temporal=False):
    sizes = [(64, 48), (64, 48), (64, 48), (48, 64), (48, 64)]
    write_pairs(root, sizes, frames=3 if temporal else 1)
    if temporal:
        (root / "temporal").touch()
    for index, (width, height) in enumerate(sizes):
        mask = np.ones((1, height, width), np.float32)
        mask[:, : index * 8] = 0
        np.save(root / f"loss{index}.npy", mask)
    (root / "dataset.toml").write_text(
        toml.dumps(
            {
                "general": {"resolution": 64, "batch_size": 2, "enable_bucket": True, "bucket_no_upscale": True},
                "datasets": [{"train_manifest": "pairs.jsonl"}],
            }
        ),
        encoding="utf-8",
    )


def _launch(root, *, lora=False, dropout=0.0, resume=False, failure=False):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    processes, logs = [], []
    try:
        for rank in range(2):
            path = root / f"worker{rank}.log"
            handle = path.open("w", encoding="utf-8")
            logs.append(handle)
            env = {
                **os.environ,
                "WORLD_SIZE": "2",
                "LOCAL_WORLD_SIZE": "2",
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(port),
                "USE_LIBUV": "0",
                "ACCELERATE_USE_CPU": "true",
                "ACCELERATE_MIXED_PRECISION": "no",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(root),
                str(rank),
                str(int(lora)),
                str(dropout),
                str(int(resume)),
                str(int(failure)),
            ]
            processes.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT))
        codes = [process.wait(timeout=120) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=20)
        for handle in logs:
            handle.close()
    # Windows c10d emits native-codepage socket warnings even when Python uses UTF-8.
    outputs = [(root / f"worker{rank}.log").read_text(encoding="utf-8", errors="replace") for rank in range(2)]
    return codes, "\n".join(outputs)


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("temporal", [False, True])
def test_two_ranks_match_global_pixel_weighted_batch_and_keep_bucket_tails(tmp_path, monkeypatch, lora, temporal):
    from musubi_tuner.networks import lora_dlssnr

    _data(tmp_path, temporal=temporal)
    codes, logs = _launch(tmp_path, lora=lora)
    assert codes == [0, 0], logs
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    # Public buckets sort width first: (48,64), then the (64,48) full and tail batches.
    assert ranks[0]["indices"] == [[3, 4], [2], [0, 1], [3, 4]]
    assert ranks[1]["indices"] == [[0, 1], [3, 4], [2], [0, 1]]
    folder = tmp_path / "output/ddp"
    records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    assert [record["consumed_samples"] for record in records] == [7, 14]
    assert [record["update"] for record in records] == [1, 2]
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert state["identity"]["world_size"] == 2
    assert [item["rank"] for item in state["rank_states"]] == [0, 1]
    assert not torch.equal(state["rank_states"][0]["rng"]["torch"], state["rank_states"][1]["rng"]["torch"])
    monkeypatch.setattr(trainer, "NRModel", SmallNR)
    monkeypatch.setattr(lora_dlssnr, "inject", small_inject)
    AcceleratorState._reset_state(reset_partial_state=True)
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    try:
        train(_args(tmp_path, "reference", lora=lora))
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
    filename = "adapter.safetensors" if lora else "model.safetensors"
    torch.testing.assert_close(
        load_file(folder / "final" / filename), load_file(tmp_path / "output/reference/final" / filename), rtol=1e-5, atol=2e-7
    )


def test_distributed_dropout_resume_restores_each_rank_and_rejects_world_change(tmp_path, monkeypatch):
    from musubi_tuner.networks import lora_dlssnr

    _data(tmp_path)
    codes, logs = _launch(tmp_path, lora=True, dropout=0.3)
    assert codes == [0, 0], logs
    folder = tmp_path / "output/ddp"
    expected = {name: value.clone() for name, value in load_file(folder / "final/adapter.safetensors").items()}
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert state["scheduler"]["last_epoch"] == state["global_update"] == 2
    codes, logs = _launch(tmp_path, lora=True, dropout=0.3, resume=True)
    assert codes == [0, 0], logs
    torch.testing.assert_close(load_file(folder / "final/adapter.safetensors"), expected, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    for rank in range(2):
        torch.testing.assert_close(
            resumed["rank_states"][rank]["rng"]["torch"], state["rank_states"][rank]["rng"]["torch"], rtol=0, atol=0
        )
    monkeypatch.setattr(trainer, "NRModel", SmallNR)
    monkeypatch.setattr(lora_dlssnr, "inject", small_inject)
    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        with pytest.raises(ValueError, match="identity|world"):
            trainer.train_lora_from_args(_args(tmp_path, "ddp", lora=True, distributed=True, dropout=0.3, resume=True))
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)


def test_rank_zero_output_failure_is_reported_to_both_ranks_without_hanging(tmp_path):
    _data(tmp_path)
    (tmp_path / "output/ddp").mkdir(parents=True)
    sentinel = tmp_path / "output/ddp/existing.txt"
    sentinel.write_text("preserve me", encoding="utf-8")
    codes, logs = _launch(tmp_path, failure=True)
    assert codes == [0, 0], logs
    for rank in range(2):
        message = json.loads((tmp_path / f"rank{rank}.json").read_text())["error"]
        assert "already exists" in message and "rank 0" in message
    assert sentinel.read_text() == "preserve me"


def test_fp8_frozen_buffers_support_ddp_resume_without_fp8_collectives(tmp_path):
    _data(tmp_path)
    (tmp_path / "fp8").touch()
    codes, logs = _launch(tmp_path, lora=True, dropout=0.3)
    assert codes == [0, 0], logs
    folder = tmp_path / "output/ddp"
    expected = {name: value.clone() for name, value in load_file(folder / "final/adapter.safetensors").items()}
    codes, logs = _launch(tmp_path, lora=True, dropout=0.3, resume=True)
    assert codes == [0, 0], logs
    torch.testing.assert_close(load_file(folder / "final/adapter.safetensors"), expected, rtol=0, atol=0)


def _worker(root, rank, lora, dropout, resume, failure):
    from musubi_tuner.networks import lora_dlssnr

    trainer.NRModel = SmallNR
    lora_dlssnr.inject = small_inject
    if (root / "fp8").exists():
        from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject

        trainer.NRModel = TinyFP8NR
        lora_dlssnr.inject = tiny_fp8_inject
    original = trainer._microbatch
    indices = []

    def observed(dataset, index, config, plan):
        indices.append(plan.indices(index))
        return original(dataset, index, config, plan)

    trainer._microbatch = observed
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    try:
        train(_args(root, "ddp", lora=lora, distributed=True, dropout=dropout, resume=resume))
    except Exception as error:
        if not failure:
            raise
        result = {"error": str(error)}
    else:
        assert not failure, "expected the existing output directory to fail"
        result = {"indices": indices, "rng": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
    (root / f"rank{rank}.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__" and sys.argv[1] == "--worker":
    _worker(
        Path(sys.argv[2]),
        int(sys.argv[3]),
        bool(int(sys.argv[4])),
        float(sys.argv[5]),
        bool(int(sys.argv[6])),
        bool(int(sys.argv[7])),
    )
