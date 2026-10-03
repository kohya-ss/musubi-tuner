"""Opt-in real-weight CUDA smoke, separate from numerical/native quality acceptance."""

import os

import pytest
import torch
from accelerate.state import AcceleratorState

from musubi_tuner.dlssnr.checkpoint import load_source, unpack_record
from musubi_tuner.dlssnr.dataset import NRBatchPlan, collate_single_frames, load_single_frame_manifest
from musubi_tuner.dlssnr.model import NRModel
from musubi_tuner.dlssnr.numerics import fp32_execution
from musubi_tuner.dlssnr.pipeline import forward_frame
from musubi_tuner.networks.lora_dlssnr import base_target_sha256, inject, merge_adapter
from musubi_tuner.training.dlssnr_services import assert_finite_gradients
from musubi_tuner.training.dlssnr_trainer import NRTrainModule, build_optimizer, clear_lane15_state
from test_dlssnr_buckets import write_pairs
from test_dlssnr_packing import _source_dir


pytestmark = pytest.mark.skipif(
    os.environ.get("DLSSNR_RUN_CUDA_SMOKE") != "1" or not torch.cuda.is_available() or _source_dir() is None,
    reason="set DLSSNR_RUN_CUDA_SMOKE=1 with CUDA and the original 310.8.0 source weights available",
)


@pytest.fixture(scope="module")
def canonical_weights():
    source = load_source(_source_dir())
    tensors = {}
    for record in source.records:
        tensors.update(
            {
                name: torch.from_numpy(value)
                for name, value in unpack_record(record, source.blobs[record.name]).items()
                if not name.startswith("opaque.")
            }
        )
    return tensors


@pytest.mark.parametrize("lora", [False, True])
def test_original_weights_complete_a_finite_cuda_update(canonical_weights, lora):
    with fp32_execution():
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(42)
        model = NRModel()
        model.load_state_dict(canonical_weights, strict=True)
        model.cuda()
        network = inject(model, {"profile": "vit_only", "rank": 16, "alpha": 16, "dropout": 0.0}) if lora else None
        if lora:
            identity = base_target_sha256(model, network.target_names)
            optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=0)
        else:
            optimizer = build_optimizer(model, 1e-5, {"priors": 0.1, "scales": 0.1}, 0.0)
        module = NRTrainModule(model, {"pre": 1.0, "out": 1.0, "edge": 0.05}, network=network)
        assert all(parameter.device.type == "cuda" for parameter in module.parameters())
        source = torch.linspace(0.2, 0.8, 48 * 48, device="cuda").reshape(1, 1, 48, 48).expand(1, 3, 48, 48)
        controls = torch.zeros(1, 5, 48, 48, device="cuda")
        controls[:, 1] = 0.5
        batch = {"source": source, "target": (source * 0.9).clone(), "controls": controls}
        before = model.blocks["70"].head.rgb.weight.detach().clone()
        loss, metrics = module(batch, [123])
        loss.backward()
        assert_finite_gradients(module)
        if lora:
            assert all(
                adapter.lora_up.grad is not None and torch.isfinite(adapter.lora_up.grad).all() for adapter in network.adapters
            )
            assert all(torch.count_nonzero(adapter.lora_up.grad) > 0 for adapter in network.adapters)
        model.enforce_lane15()
        optimizer.step()
        clear_lane15_state(model, optimizer)
        optimizer.zero_grad(set_to_none=True)
        if lora:
            assert base_target_sha256(model, network.target_names) == identity
            fresh = NRModel().cuda()
            fresh.load_state_dict(merge_adapter(model.state_dict(), network))
            with torch.no_grad():
                original = forward_frame(model, source, controls, [123])
                merged = forward_frame(fresh, source, controls, [123])
            torch.testing.assert_close(merged["raw_head"], original["raw_head"], rtol=2e-4, atol=2e-4)
        else:
            assert not torch.equal(before, model.blocks["70"].head.rgb.weight)
        assert torch.count_nonzero(model.blocks["0"].input_adapter.weight[:, 15]) == 0
        print({"lora": lora, "loss": metrics["loss"], "peak_cuda_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1)})


@pytest.mark.parametrize("lora", [False, True])
def test_original_weights_update_across_landscape_and_portrait_buckets(tmp_path, canonical_weights, lora):
    path = write_pairs(tmp_path, [(832, 480), (480, 832)])
    data = load_single_frame_manifest(path, 320, 320, enable_bucket=True)
    assert data.bucket_sizes == [(416, 240), (240, 416)]
    plan = NRBatchPlan(data, 2)
    with fp32_execution():
        torch.manual_seed(42)
        model = NRModel().cuda()
        model.load_state_dict(canonical_weights, strict=True)
        network = inject(model, {"profile": "vit_only", "rank": 4, "alpha": 4, "dropout": 0.0}) if lora else None
        optimizer = (
            torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=0)
            if lora
            else build_optimizer(model, 1e-5, {"priors": 0.1, "scales": 0.1}, 0.0)
        )
        module = NRTrainModule(model, {"pre": 1.0, "out": 1.0, "edge": 0.05}, network=network)
        before = model.blocks["70"].head.rgb.weight.detach().clone()
        batches = []
        for index in range(len(plan)):
            batch = {
                name: value.cuda() for name, value in collate_single_frames([data[item] for item in plan.indices(index)]).items()
            }
            batches.append(batch)
            loss, _ = module(batch, [123])
            loss.backward()
            assert_finite_gradients(module)
            model.enforce_lane15()
            optimizer.step()
            clear_lane15_state(model, optimizer)
            optimizer.zero_grad(set_to_none=True)
        if lora:
            assert all(torch.count_nonzero(adapter.lora_up) > 0 for adapter in network.adapters)
            torch.testing.assert_close(model.blocks["70"].head.rgb.weight, before, rtol=0, atol=0)
            merged = NRModel().cuda()
            merged.load_state_dict(merge_adapter(model.state_dict(), network))
            with torch.no_grad():
                for batch in batches:
                    actual = forward_frame(model, batch["source"], batch["controls"], [123])
                    reference = forward_frame(merged, batch["source"], batch["controls"], [123])
                    torch.testing.assert_close(actual["raw_head"], reference["raw_head"], rtol=2e-4, atol=2e-4)
        else:
            assert not torch.equal(model.blocks["70"].head.rgb.weight, before)


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.parametrize("lora", [False, True])
def test_original_weights_checkpointed_amp_updates_are_finite(tmp_path, canonical_weights, precision, lora, backend="native"):
    from musubi_tuner.dlssnr.config import build_train_config
    from musubi_tuner.dlssnr.runtime import configure_model_runtime, runtime_policy
    from musubi_tuner.training.dlssnr_services import create_accelerator, optimizer_update
    from test_dlssnr_config import make_args

    AcceleratorState._reset_state(reset_partial_state=True)
    args = make_args(
        tmp_path,
        ["--numerics_profile", "train_experimental", "--mixed_precision", precision, "--gradient_checkpointing"],
        lora=lora,
    )
    args.attention_backend = backend
    config = build_train_config(args, lora=lora)
    accelerator = create_accelerator(config["training"], config["precision"])
    try:
        with fp32_execution():
            model = NRModel()
            model.load_state_dict(canonical_weights, strict=True)
            configure_model_runtime(model, runtime_policy(config), training=True)
            network = inject(model, config["lora"]) if lora else None
            optimizer = (
                torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=0)
                if lora
                else build_optimizer(model, 1e-5, {"priors": 0.1, "scales": 0.1}, 0.0)
            )
            module = NRTrainModule(model, config["loss"], network=network)
            wrapped, optimizer = accelerator.prepare(module, optimizer)
            source = torch.linspace(0.2, 0.8, 48 * 48, device="cuda").reshape(1, 1, 48, 48).expand(1, 3, 48, 48)
            controls = torch.zeros(1, 5, 48, 48, device="cuda")
            controls[:, 1] = 0.5
            batch = {"source": source, "target": source * 0.9, "controls": controls}
            trainable = next(network.parameters()) if lora else model.blocks["70"].head.rgb.weight
            before = trainable.detach().clone()
            for attempt in range(17):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = wrapped(batch, [123])
                assert loss.dtype == torch.float32 and torch.isfinite(loss)
                accelerator.backward(loss)
                if optimizer_update(accelerator, module, optimizer, 1.0):
                    break
            else:
                pytest.fail("real NR AMP gradients did not recover after 16 scaler retries")
            assert_finite_gradients(module)
            assert all(parameter.dtype == torch.float32 for parameter in module.parameters())
            if lora:
                assert any(torch.count_nonzero(adapter.lora_up) for adapter in network.adapters)
            else:
                assert not torch.equal(before, trainable)
            print(
                {
                    "precision": precision,
                    "lora": lora,
                    "backend": backend,
                    "overflow_retries": attempt,
                    "loss": float(loss.detach()),
                }
            )
    finally:
        accelerator.end_training()
        accelerator.free_memory()
        AcceleratorState._reset_state(reset_partial_state=True)


@pytest.mark.parametrize("lora", [False, True])
def test_original_weights_sdpa_bf16_full_and_lora_updates(tmp_path, canonical_weights, lora):
    test_original_weights_checkpointed_amp_updates_are_finite(tmp_path, canonical_weights, "bf16", lora, backend="sdpa")


@pytest.mark.parametrize("scaled", [False, True])
def test_original_weights_fp8_lora_cuda_update_and_materialized_merge(tmp_path, canonical_weights, scaled):
    from musubi_tuner.dlssnr.config import build_train_config
    from musubi_tuner.dlssnr.fp8 import materialize_state_dict, quantize_frozen_base
    from musubi_tuner.dlssnr.runtime import configure_model_runtime, runtime_policy
    from test_dlssnr_config import make_args

    options = ["--numerics_profile", "train_experimental", "--mixed_precision", "bf16", "--gradient_checkpointing", "--fp8_base"]
    if scaled:
        options.append("--fp8_scaled")
    config = build_train_config(make_args(tmp_path, options, lora=True), lora=True)
    policy = runtime_policy(config)
    with fp32_execution():
        model = NRModel()
        model.load_state_dict(canonical_weights, strict=True)
        configure_model_runtime(model, policy, training=True)
        network = inject(model, config["lora"])
        network.base_quantization = quantize_frozen_base(model, scaled=scaled)
        model.cuda()
        network.cuda()
        assert model.blocks["31"].ffn.fc1.weight.dtype == torch.float8_e4m3fn
        assert model.blocks["31"].ffn.fc1.weight.device.type == "cuda"
        source = torch.linspace(0.2, 0.8, 48 * 48, device="cuda").reshape(1, 1, 48, 48).expand(1, 3, 48, 48)
        controls = torch.zeros(1, 5, 48, 48, device="cuda")
        controls[:, 1] = 0.5
        module = NRTrainModule(model, config["loss"], network=network)
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4, weight_decay=0)
        loss, _ = module({"source": source, "target": source * 0.9, "controls": controls}, [123])
        loss.backward()
        assert_finite_gradients(module)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert any(torch.count_nonzero(adapter.lora_up) for adapter in network.adapters)
        merged = NRModel().cuda()
        merged.load_state_dict(merge_adapter(materialize_state_dict(model), network), strict=True)
        configure_model_runtime(merged, {**policy, "fp8_base": False, "fp8_scaled": False}, training=False)
        model.eval()
        merged.eval()
        with torch.no_grad():
            original = forward_frame(model, source, controls, [123])
            reference = forward_frame(merged, source, controls, [123])
        torch.testing.assert_close(reference["raw_head"], original["raw_head"], rtol=2e-4, atol=2e-4)
