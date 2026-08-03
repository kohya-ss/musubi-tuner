import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch

from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.model import MiniMaxH3Config, MiniMaxH3Model
from musubi_tuner.minimax_h3.packing import H3VideoGeometry, build_h3_layout
from musubi_tuner.minimax_h3_train_network import (
    MiniMaxH3NetworkTrainer,
    minimax_h3_setup_parser,
    validate_h3_dataset_batches,
)
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.networks import lora_minimax_h3
from musubi_tuner.training.trainer_base import DiTOutput


BUCKET = (32, 32, 5)


def _write_item(
    root: Path,
    name: str,
    *,
    task: str = "t2va",
    target_shape: tuple[int, int, int] = (2, 4, 4),
    token_tags: tuple[int, ...] = (1, 0, 1),
    references: tuple[tuple[str, tuple[int, ...]], ...] = (),
) -> ItemInfo:
    latent_path = root / f"{name}_00000-005_32x32_mmh3.safetensors"
    text_path = root / f"{name}_00000-005_mmh3_te.safetensors"
    frames, height, width = target_shape
    latent_tensors = {
        f"latents_{frames}x{height}x{width}_bfloat16": torch.zeros(24, frames, height, width, dtype=torch.bfloat16),
        "latents_audio_32x2x8_bfloat16": torch.zeros(32, 2, 8, dtype=torch.bfloat16),
    }
    reference_kinds = []
    for role, shape in references:
        if role.endswith("_audio"):
            latent_tensors[f"latents_{role}_32x2x{shape[-1]}_bfloat16"] = torch.zeros(shape, dtype=torch.bfloat16)
        else:
            latent_tensors[f"latents_{role}_{'x'.join(map(str, shape[1:]))}_bfloat16"] = torch.zeros(shape, dtype=torch.bfloat16)
        if role.startswith("ref_"):
            index = int(role.split("_")[1])
            while len(reference_kinds) <= index:
                reference_kinds.append(None)
            kind = role.rsplit("_", 1)[1]
            if kind == "audio" and reference_kinds[index] == "video":
                reference_kinds[index] = "video+audio"
            elif kind == "video" and reference_kinds[index] == "audio":
                reference_kinds[index] = "video+audio"
            else:
                reference_kinds[index] = kind
    save_file(
        latent_tensors,
        str(latent_path),
        metadata={"task": task, "reference_kinds": json.dumps(reference_kinds, separators=(",", ":"))},
    )
    tags = torch.tensor(token_tags, dtype=torch.int64)
    save_file(
        {
            "varlen_mmh3_hidden_states_bfloat16": torch.zeros(len(token_tags), 5120, dtype=torch.bfloat16),
            "varlen_mmh3_token_tags_int64": tags,
        },
        str(text_path),
        metadata={"task": task},
    )
    item = ItemInfo(name, "", (32, 32), BUCKET, frame_count=5, latent_cache_path=str(latent_path))
    item.text_encoder_output_cache_path = str(text_path)
    return item


def _dataset_group(items, *, batch_size: int = 2, num_timestep_buckets: int | None = None):
    manager = BucketBatchManager({BUCKET: list(items)}, batch_size, num_timestep_buckets=num_timestep_buckets)
    return SimpleNamespace(datasets=[SimpleNamespace(batch_manager=manager)])


def test_preflight_scans_every_item_in_a_bucket_not_only_the_initial_partition(tmp_path):
    first = _write_item(tmp_path, "first")
    second = _write_item(tmp_path, "second")
    incompatible_third = _write_item(tmp_path, "third", token_tags=(1, 0, 1, 1))

    with pytest.raises(ValueError) as error:
        validate_h3_dataset_batches(_dataset_group([first, second, incompatible_third]))

    message = str(error.value)
    assert "dataset 0" in message
    assert str(BUCKET) in message
    assert str(first.latent_cache_path) in message
    assert str(incompatible_third.text_encoder_output_cache_path) in message
    assert "text_length" in message


@pytest.mark.parametrize(
    ("mutated", "conflicting_field"),
    [
        ({"token_tags": (1, 1, 1)}, "token_tags"),
        ({"target_shape": (2, 4, 6)}, "packed_rows"),
        ({"target_shape": (2, 2, 8)}, "rotary_inputs"),
        (
            {
                "task": "ref2va",
                "references": (
                    ("ref_000_image", (24, 1, 4, 4)),
                    ("ref_001_audio", (32, 2, 8)),
                ),
            },
            "task",
        ),
    ],
)
def test_preflight_reports_structural_conflicts(tmp_path, mutated, conflicting_field):
    baseline = _write_item(tmp_path, "baseline")
    changed = _write_item(tmp_path, "changed", **mutated)

    with pytest.raises(ValueError, match=conflicting_field):
        validate_h3_dataset_batches(_dataset_group([baseline, changed]))


def test_preflight_accepts_a_structurally_compatible_replicated_bucket(tmp_path):
    items = [_write_item(tmp_path, name) for name in ("first", "second", "third")]

    validate_h3_dataset_batches(_dataset_group(items))


def test_preflight_rejects_a_per_sample_timestep_pool_for_replicated_batches(tmp_path):
    items = [_write_item(tmp_path, name) for name in ("first", "second")]

    with pytest.raises(ValueError, match=r"dataset 0.*bucket.*num_timestep_buckets"):
        validate_h3_dataset_batches(_dataset_group(items, num_timestep_buckets=4))


def test_preflight_checks_the_authoritative_task_even_for_single_item_buckets(tmp_path):
    item = _write_item(tmp_path, "single", task="t2va")

    with pytest.raises(ValueError, match=r"dataset 0.*task.*ref2va.*t2va"):
        validate_h3_dataset_batches(_dataset_group([item], batch_size=1), expected_task="ref2va")


def test_preflight_checks_every_items_task_when_dataset_batch_size_is_one(tmp_path):
    matching = _write_item(tmp_path, "matching", task="t2va")
    mismatched = _write_item(
        tmp_path,
        "mismatched",
        task="ref2va",
        references=(("ref_000_image", (24, 1, 4, 4)),),
    )

    with pytest.raises(ValueError) as error:
        validate_h3_dataset_batches(_dataset_group([matching, mismatched], batch_size=1), expected_task="t2va")

    message = str(error.value)
    assert "dataset 0" in message
    assert "--task t2va" in message
    assert "cache task ref2va" in message
    assert str(mismatched.latent_cache_path) in message


class _Accelerator:
    device = torch.device("cpu")

    @staticmethod
    def autocast():
        return nullcontext()


class _RecordingTransformer:
    def __init__(self, video_prediction: float = 2.0, audio_prediction: float = -1.0):
        self.video_prediction = video_prediction
        self.audio_prediction = audio_prediction
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            video=torch.full_like(kwargs["video_latents"], self.video_prediction),
            audio=torch.full_like(kwargs["audio_latents"], self.audio_prediction),
        )


def _trainer_args(**overrides):
    values = {
        "timestep_sampling": "uniform",
        "weighting_scheme": "none",
        "discrete_flow_shift": 1.0,
        "h3_shift_video": 12.0,
        "h3_shift_audio": 3.0,
        "h3_visual_cond_clean": 0.999,
        "h3_audio_cond_clean": 1.0,
        "min_timestep": None,
        "max_timestep": None,
        "blocks_to_swap": 0,
        "sample_prompts": None,
        "gradient_checkpointing": False,
        "task": "t2va",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _training_batch(batch_size: int = 2, *, text_length: int = 3):
    return {
        "latents_audio": torch.full((batch_size, 32, 2, 8), 4.0),
        "mmh3_hidden_states": [torch.full((text_length, 12), float(index)) for index in range(batch_size)],
        "mmh3_token_tags": [torch.tensor([1, 0, 1][:text_length], dtype=torch.int64) for _ in range(batch_size)],
        "timesteps": None,
    }


def test_h3_parser_defaults_to_the_only_supported_training_coordinates():
    parser = minimax_h3_setup_parser(argparse.ArgumentParser())

    args = parser.parse_args(["--task", "t2va"])

    assert args.timestep_sampling == "uniform"
    assert args.weighting_scheme == "none"
    assert args.discrete_flow_shift == 1.0
    assert args.h3_shift_video == 12.0
    assert args.h3_shift_audio == 3.0
    assert args.h3_visual_cond_clean == 0.999
    assert args.h3_audio_cond_clean == 1.0
    assert args.network_module == "networks.lora_minimax_h3"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"timestep_sampling": "sigma"}, "timestep_sampling"),
        ({"weighting_scheme": "sigma_sqrt"}, "weighting_scheme"),
        ({"discrete_flow_shift": 1.1}, "discrete_flow_shift"),
        ({"h3_shift_video": 0.0}, "h3_shift_video"),
        ({"h3_shift_audio": 101.0}, "h3_shift_audio"),
        ({"h3_visual_cond_clean": -0.1}, "h3_visual_cond_clean"),
        ({"h3_audio_cond_clean": 1.1}, "h3_audio_cond_clean"),
        ({"blocks_to_swap": 49}, "blocks_to_swap"),
    ],
)
def test_h3_trainer_rejects_training_knobs_with_the_wrong_coordinate_contract(override, message):
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(ValueError, match=message):
        trainer.handle_model_specific_args(_trainer_args(**override))


def test_process_batch_uses_one_shared_base_time_and_independent_audio_noise(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer()
    batch = _training_batch()
    video_latents = torch.full((2, 24, 2, 4, 4), 5.0)
    video_noise = torch.full_like(video_latents, -2.0)
    real_randn_like = torch.randn_like

    def fixed_audio_noise(tensor, *positional, **kwargs):
        if tuple(tensor.shape) == (2, 32, 2, 8):
            return torch.full_like(tensor, 3.0)
        return real_randn_like(tensor, *positional, **kwargs)

    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", fixed_audio_noise)

    loss, metrics = trainer.process_batch(
        args,
        _Accelerator(),
        transformer,
        None,
        batch,
        video_latents,
        video_noise,
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )

    call = transformer.calls[0]
    assert call["model_t_video"].shape == torch.Size([])
    assert call["model_t_audio"].shape == torch.Size([])
    assert call["model_t_video"].item() == pytest.approx(0.2)
    assert call["model_t_audio"].item() == pytest.approx(0.5)
    assert torch.allclose(call["video_latents"], torch.full_like(video_latents, -0.6))
    assert torch.allclose(call["audio_latents"], torch.full_like(batch["latents_audio"], 3.5))
    video_target = video_latents - video_noise
    audio_target = batch["latents_audio"] - 3.0
    expected_video_loss = torch.nn.functional.mse_loss(torch.full_like(video_target, 2.0), video_target)
    expected_audio_loss = torch.nn.functional.mse_loss(torch.full_like(audio_target, -1.0), audio_target)
    assert loss == pytest.approx((expected_video_loss + expected_audio_loss).item())
    assert metrics["loss/video"] == pytest.approx(expected_video_loss.item())
    assert metrics["loss/audio"] == pytest.approx(expected_audio_loss.item())


def test_process_batch_preserves_the_released_fp32_audio_cache_dtype(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer()
    batch = _training_batch(batch_size=1)
    batch["latents_audio"] = batch["latents_audio"].to(torch.float32)
    video_latents = torch.zeros(1, 24, 2, 4, 4, dtype=torch.float16)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))

    trainer.process_batch(
        args,
        _Accelerator(),
        transformer,
        None,
        batch,
        video_latents,
        torch.zeros_like(video_latents),
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )

    assert transformer.calls[0]["video_latents"].dtype == torch.float16
    assert transformer.calls[0]["audio_latents"].dtype == torch.float32


def _cpu_noise(shape, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def test_condition_noise_restarts_per_role_uses_audio_seed_plus_one_and_changes_per_step(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(task="ref2va", h3_visual_cond_clean=0.5, h3_audio_cond_clean=0.5)
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer()
    batch = _training_batch()
    batch["latents_ref_000_image"] = torch.zeros(2, 24, 1, 4, 4)
    batch["latents_ref_001_audio"] = torch.zeros(2, 32, 2, 8)
    video_latents = torch.zeros(2, 24, 2, 4, 4)
    seeds = iter((torch.tensor([100, 200]), torch.tensor([300, 400])))
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: next(seeds))
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))

    for step in range(2):
        trainer.process_batch(
            args,
            _Accelerator(),
            transformer,
            None,
            batch,
            video_latents,
            torch.zeros_like(video_latents),
            None,
            torch.bfloat16,
            torch.float32,
            None,
            step,
        )

    first_call, second_call = transformer.calls
    first_visual = first_call["visual_condition_latents"][0]
    first_audio = first_call["audio_condition_latents"][0]
    assert torch.equal(first_visual[0], 0.5 * _cpu_noise((24, 1, 4, 4), 100))
    assert torch.equal(first_visual[1], 0.5 * _cpu_noise((24, 1, 4, 4), 200))
    assert torch.equal(first_audio[0], 0.5 * _cpu_noise((32, 2, 8), 101))
    assert torch.equal(first_audio[1], 0.5 * _cpu_noise((32, 2, 8), 201))
    assert not torch.equal(first_visual, second_call["visual_condition_latents"][0])
    assert not torch.equal(first_audio, second_call["audio_condition_latents"][0])


def test_runtime_rejects_mismatched_token_tags_and_per_sample_timestep_values():
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    video_latents = torch.zeros(2, 24, 2, 4, 4)
    batch = _training_batch()
    batch["mmh3_token_tags"][1] = torch.tensor([1, 1, 1])

    with pytest.raises(ValueError, match="token tags"):
        trainer.process_batch(
            args,
            _Accelerator(),
            _RecordingTransformer(),
            None,
            batch,
            video_latents,
            torch.zeros_like(video_latents),
            None,
            torch.bfloat16,
            torch.float32,
            None,
            0,
        )

    batch = _training_batch()
    batch["timesteps"] = [0.1, 0.2]
    with pytest.raises(ValueError, match="per-sample timestep"):
        trainer.process_batch(
            args,
            _Accelerator(),
            _RecordingTransformer(),
            None,
            batch,
            video_latents,
            torch.zeros_like(video_latents),
            None,
            torch.bfloat16,
            torch.float32,
            None,
            0,
        )


def test_runtime_rejects_a_batch_from_a_different_authoritative_task():
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(task="ref2va")
    trainer.handle_model_specific_args(args)
    video_latents = torch.zeros(1, 24, 2, 4, 4)

    with pytest.raises(ValueError, match=r"--task ref2va.*T2VA"):
        trainer.process_batch(
            args,
            _Accelerator(),
            _RecordingTransformer(),
            None,
            _training_batch(batch_size=1),
            video_latents,
            torch.zeros_like(video_latents),
            None,
            torch.bfloat16,
            torch.float32,
            None,
            0,
        )


def test_t2va_does_not_advance_the_condition_seed_stream(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: pytest.fail("T2VA without conditions must not draw a condition seed"),
    )
    video_latents = torch.zeros(1, 24, 2, 4, 4)

    trainer.process_batch(
        args,
        _Accelerator(),
        _RecordingTransformer(),
        None,
        _training_batch(batch_size=1),
        video_latents,
        torch.zeros_like(video_latents),
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )


def test_compute_loss_is_plain_video_plus_audio_mean_mse():
    trainer = MiniMaxH3NetworkTrainer()
    output = DiTOutput(
        pred=torch.tensor([1.0, 5.0]),
        target=torch.tensor([3.0, 1.0]),
        extra={"audio_pred": torch.tensor([0.0, 2.0]), "audio_target": torch.tensor([2.0, 2.0])},
    )

    loss, metrics = trainer.compute_loss(
        _trainer_args(),
        output,
        torch.tensor(0.25),
        object(),
        torch.bfloat16,
        torch.float32,
        7,
    )

    assert loss.item() == pytest.approx(12.0)
    assert metrics == {"loss/video": pytest.approx(10.0), "loss/audio": pytest.approx(2.0)}


def test_h3_training_metadata_records_task_scheduler_and_target_policy():
    trainer = MiniMaxH3NetworkTrainer()

    metadata = trainer.extra_metadata(_trainer_args(task="fl2va"))

    assert metadata == {
        "ss_minimax_h3_task": "fl2va",
        "ss_minimax_h3_base_family": "fl2va",
        "ss_minimax_h3_shift_video": 12.0,
        "ss_minimax_h3_shift_audio": 3.0,
        "ss_minimax_h3_visual_cond_clean": 0.999,
        "ss_minimax_h3_audio_cond_clean": 1.0,
        "ss_minimax_h3_target_modules": "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2",
        "ss_minimax_h3_latent_cache_version": "1",
        "ss_minimax_h3_text_cache_version": "1",
    }


def _tiny_model(num_layers: int = 2):
    config = MiniMaxH3Config(
        hidden_size=16,
        num_layers=num_layers,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_hidden_size=24,
        text_dim=12,
        timestep_input_dim=4,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        rope_inv_freq_len=1,
    )
    return MiniMaxH3Model(config, dtype=torch.float32)


def test_default_h3_lora_policy_targets_only_four_projections_in_main_blocks():
    model = _tiny_model(num_layers=2)

    network = lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, None, model)

    assert {module.lora_name for module in network.unet_loras} == {
        "lora_unet_blocks_0_attn_qkv_proj",
        "lora_unet_blocks_0_attn_out_proj",
        "lora_unet_blocks_0_mlp_fc1",
        "lora_unet_blocks_0_mlp_fc2",
        "lora_unet_blocks_1_attn_qkv_proj",
        "lora_unet_blocks_1_attn_out_proj",
        "lora_unet_blocks_1_mlp_fc1",
        "lora_unet_blocks_1_mlp_fc2",
    }


def test_h3_lora_gets_gradients_with_checkpointing_and_block_swap(monkeypatch):
    class _Offloader:
        def __init__(self, blocks, device):
            self.blocks = blocks
            self.device = device

        def prepare_block_devices_before_forward(self, blocks):
            for block in blocks:
                block.to(self.device)

        def wait_for_block(self, index):
            return None

        def submit_move_blocks_forward(self, blocks, index):
            return None

        def set_forward_only(self, value):
            return None

    monkeypatch.setattr(
        "musubi_tuner.minimax_h3.model.create_offloader",
        lambda block_type, blocks, num_blocks, blocks_to_swap, config: _Offloader(blocks, config.device),
    )
    model = _tiny_model(num_layers=3)
    model.requires_grad_(False)
    model.enable_gradient_checkpointing()
    model.train()
    model.enable_block_swap(1, BlockSwapConfig(device=torch.device("cpu"), supports_backward=True))
    model.move_to_device_except_swap_blocks(torch.device("cpu"))
    model.prepare_block_swap_before_forward()
    network = lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.prepare_optimizer_params(unet_lr=1e-4)
    layout = build_h3_layout(
        task="t2va",
        text_length=3,
        target_video=H3VideoGeometry(2, 4, 4),
        target_audio_frames=8,
    )

    output = model(
        video_latents=torch.randn(1, 24, 2, 4, 4),
        audio_latents=torch.randn(1, 32, 2, 8),
        text_hidden_states=torch.randn(1, 3, 12),
        text_token_tags=torch.tensor([1, 0, 1]),
        layout=layout,
        model_t_video=torch.tensor(0.25),
        model_t_audio=torch.tensor(0.75),
    )
    (output.video.square().mean() + output.audio.square().mean()).backward()

    gradients = [parameter.grad for parameter in network.parameters()]
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in gradients)
