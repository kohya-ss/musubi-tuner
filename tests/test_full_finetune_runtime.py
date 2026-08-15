from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from accelerate.utils import DistributedType
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

import musubi_tuner.training.full_finetune as full_finetune
from musubi_tuner.training.full_finetune import FullFineTuningTrainerMixin, save_state_all_ranks
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer


class TinyDataset(torch.utils.data.Dataset):
    num_train_items = 2
    batch_size = 1

    def __init__(self):
        self.datasets = [self]
        self.max_train_steps = None

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"latents": torch.tensor([[float(index + 1)]])}

    def get_metadata(self):
        return {"name": "tiny-cached-dataset"}

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scale=1.0)
        self.linear = nn.Linear(1, 1)
        nn.init.constant_(self.linear.weight, 0.25)
        nn.init.constant_(self.linear.bias, 0.5)
        self.gradient_checkpointing_cpu_offload = None
        self.swap_config = None
        self.move_except_swap_calls = 0
        self.prepare_swap_calls = 0
        self.swap_training_calls = 0

    def forward(self, value):
        return self.linear(value)

    def enable_gradient_checkpointing(self, cpu_offload):
        self.gradient_checkpointing_cpu_offload = cpu_offload

    def enable_block_swap(self, blocks_to_swap, config):
        self.blocks_to_swap = blocks_to_swap
        self.swap_config = config

    def move_to_device_except_swap_blocks(self, _device):
        self.move_except_swap_calls += 1

    def prepare_block_swap_before_forward(self):
        self.prepare_swap_calls += 1

    def switch_block_swap_for_inference(self):
        pass

    def switch_block_swap_for_training(self):
        self.swap_training_calls += 1


class PreparedWrapper(nn.Module):
    """A prepared model that deliberately does not expose architecture config."""

    def __init__(self, raw_model, trainer):
        super().__init__()
        self.wrapped = raw_model
        self.trainer = trainer

    def forward(self, *args, **kwargs):
        self.trainer.forward_wrapper_called = True
        return self.wrapped(*args, **kwargs)


class FakeAccelerator:
    def __init__(self, trainer):
        self.trainer = trainer
        self.device = torch.device("cpu")
        self.mixed_precision = "no"
        self.num_processes = 1
        self.distributed_type = DistributedType.NO
        self.is_main_process = True
        self.is_local_main_process = True
        self.sync_gradients = True
        self.trackers = []
        self.model_prepare_calls = 0
        self.model_device_placement = None
        self.accumulate_model = None

    def prepare(self, *objects, device_placement=None):
        if len(objects) == 1 and isinstance(objects[0], nn.Module):
            self.model_prepare_calls += 1
            self.model_device_placement = device_placement
            self.trainer.compile_called_before_prepare = self.trainer.compile_called
            return PreparedWrapper(objects[0], self.trainer)
        assert not any(isinstance(obj, nn.Module) for obj in objects), "model was prepared a second time"
        return objects if len(objects) > 1 else objects[0]

    def unwrap_model(self, model, keep_fp32_wrapper=True):
        del keep_fp32_wrapper
        return model.wrapped if isinstance(model, PreparedWrapper) else model

    @contextmanager
    def accumulate(self, model):
        self.accumulate_model = model
        yield

    @contextmanager
    def autocast(self):
        yield

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        torch.nn.utils.clip_grad_norm_(list(parameters), max_norm)

    def print(self, *_args, **_kwargs):
        pass

    def wait_for_everyone(self):
        pass

    def reduce(self, tensor, reduction="sum"):
        assert reduction == "sum"
        return tensor

    def init_trackers(self, *_args, **_kwargs):
        pass

    def log(self, *_args, **_kwargs):
        pass

    def end_training(self):
        pass

    def register_for_checkpointing(self, _obj):
        pass


class TinyFullTrainer(FullFineTuningTrainerMixin, NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.forward_wrapper_called = False
        self.compile_called = False
        self.compile_called_before_prepare = False
        self.after_sample_called = False
        self.optimizer_mode_events = []

    @property
    def architecture(self):
        return "tiny"

    @property
    def architecture_full_name(self):
        return "tiny-transformer"

    def handle_model_specific_args(self, _args):
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0

    def _build_dataset(self, _args):
        dataset = TinyDataset()
        return dataset, lambda items: items[0], SimpleNamespace(value=0)

    def process_sample_prompts(self, _args, _accelerator, _sample_prompts):
        return [{"prompt": "tiny"}]

    def load_vae(self, _args, vae_dtype, vae_path):
        del vae_dtype, vae_path
        self.sample_vae = nn.Linear(1, 1)
        return self.sample_vae

    def load_full_finetune_transformer(
        self,
        accelerator,
        args,
        dit_path,
        attn_mode,
        split_attn,
        loading_device,
        trainable_dtype,
    ):
        del accelerator, args, dit_path, attn_mode, split_attn, loading_device
        self.loader_dtype = trainable_dtype
        self.loaded_model = TinyTransformer().to(dtype=trainable_dtype)
        self.initial_weight = self.loaded_model.linear.weight.detach().clone()
        return self.loaded_model

    def compile_transformer(self, _args, transformer):
        self.compile_called = True
        return transformer

    def get_optimizer(self, args, trainable_params):
        params = [param for group in trainable_params for param in group["params"]]
        self.optimizer_param_ids = {id(param) for param in params}
        optimizer = torch.optim.SGD(trainable_params, lr=args.learning_rate)
        return (
            "SGD",
            "",
            optimizer,
            lambda: self.optimizer_mode_events.append("train"),
            lambda: self.optimizer_mode_events.append("eval"),
        )

    def get_lr_scheduler(self, _args, optimizer, _num_processes):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)

    def scale_shift_latents(self, latents):
        return latents

    def call_dit(
        self,
        args,
        accelerator,
        transformer_arg,
        latents,
        batch,
        noise,
        noisy_model_input,
        timesteps,
        network_dtype,
        **kwargs,
    ):
        del args, latents, batch, noise, timesteps, network_dtype, kwargs
        raw_model = accelerator.unwrap_model(transformer_arg, keep_fp32_wrapper=False)
        prediction = transformer_arg(noisy_model_input * raw_model.config.scale)
        return DiTOutput(pred=prediction, target=torch.zeros_like(prediction))

    def process_batch(
        self,
        args,
        accelerator,
        transformer,
        network,
        batch,
        latents,
        noise,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        vae,
        global_step,
    ):
        del network, noise_scheduler, dit_dtype, vae, global_step
        output = self.call_dit(
            args,
            accelerator,
            transformer,
            latents,
            batch,
            noise,
            latents,
            torch.zeros(latents.shape[0]),
            network_dtype,
        )
        return torch.nn.functional.mse_loss(output.pred, output.target), {}

    def on_after_sample_images(self, *args, **kwargs):
        del args, kwargs
        self.after_sample_called = True

    def sample_images(self, *_args, **_kwargs):
        self.optimizer_mode_events.append("sample")

    def on_post_save(
        self,
        args,
        _accelerator,
        _network,
        _transformer,
        ckpt_name,
        _save_dtype,
        _metadata,
        _force_sync_upload,
    ):
        checkpoint_path = str(args.output_dir) + "/" + ckpt_name
        self.saved_state_dict = load_file(checkpoint_path)
        with safe_open(checkpoint_path, framework="pt") as checkpoint:
            self.saved_metadata = checkpoint.metadata()


def make_args(tmp_path, **overrides):
    defaults = {
        "dataset_config": "unused.toml",
        "dit": "unused.safetensors",
        "vae": "unused-vae.safetensors",
        "network_weights": None,
        "network_module": None,
        "network_dim": None,
        "network_alpha": 1,
        "network_dropout": None,
        "network_args": None,
        "dim_from_weights": False,
        "scale_weight_norms": None,
        "base_weights": None,
        "base_weights_multiplier": None,
        "fp8_base": False,
        "fp8_scaled": False,
        "block_swap_h2d_only": False,
        "block_swap_ring_size": 2,
        "blocks_to_swap": 1,
        "use_pinned_memory_for_block_swap": False,
        "block_swap_optimizer_patch_params": False,
        "fused_backward_pass": False,
        "optimizer_type": "SGD",
        "optimizer_args": None,
        "gradient_accumulation_steps": 1,
        "save_precision": None,
        "full_bf16": False,
        "full_fp16": False,
        "mixed_precision": "no",
        "gradient_checkpointing": True,
        "gradient_checkpointing_cpu_offload": False,
        "compile": True,
        "cuda_allow_tf32": False,
        "cuda_cudnn_benchmark": False,
        "sage_attn": False,
        "disable_numpy_memmap": False,
        "show_timesteps": None,
        "num_timestep_buckets": None,
        "seed": 123,
        "sdpa": True,
        "flash_attn": False,
        "flash3": False,
        "xformers": False,
        "split_attn": False,
        "vae_dtype": "float32",
        "sample_prompts": "unused-prompts.txt",
        "sample_at_first": False,
        "sample_every_n_steps": None,
        "sample_every_n_epochs": None,
        "max_data_loader_n_workers": 0,
        "persistent_data_loader_workers": False,
        "max_train_steps": 1,
        "max_train_epochs": None,
        "learning_rate": 0.1,
        "max_grad_norm": 1.0,
        "lr_warmup_steps": 0,
        "lr_scheduler": "constant",
        "weighting_scheme": "none",
        "logit_mean": 0.0,
        "logit_std": 1.0,
        "mode_scale": 1.29,
        "guidance_scale": 1.0,
        "timestep_sampling": "uniform",
        "sigmoid_scale": 1.0,
        "discrete_flow_shift": 1.0,
        "output_dir": str(tmp_path),
        "output_name": "tiny",
        "training_comment": None,
        "no_metadata": False,
        "metadata_title": None,
        "metadata_reso": None,
        "metadata_author": None,
        "metadata_description": None,
        "metadata_license": None,
        "metadata_tags": None,
        "metadata_arch": None,
        "min_timestep": None,
        "max_timestep": None,
        "wandb_run_name": None,
        "log_tracker_config": None,
        "log_tracker_name": None,
        "save_every_n_steps": None,
        "save_every_n_epochs": None,
        "save_last_n_steps": None,
        "save_last_n_epochs": None,
        "save_last_n_steps_state": None,
        "save_last_n_epochs_state": None,
        "save_state": False,
        "save_state_on_train_end": False,
        "save_state_to_huggingface": False,
        "resume": None,
        "resume_from_huggingface": False,
        "huggingface_repo_id": None,
        "mem_eff_save": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def install_runtime_fakes(monkeypatch, trainer):
    accelerator = FakeAccelerator(trainer)
    monkeypatch.setattr(full_finetune, "prepare_accelerator", lambda _args: accelerator)
    monkeypatch.setattr(full_finetune, "clean_memory_on_device", lambda _device: None)
    monkeypatch.setattr(
        full_finetune.sai_model_spec,
        "build_metadata",
        lambda *args, **kwargs: {"modelspec.architecture": "tiny", "is_lora": str(kwargs["is_lora"])},
    )

    return accelerator


def test_full_model_lifecycle_uses_prepared_forward_and_exports_only_transformer(tmp_path, monkeypatch):
    trainer = TinyFullTrainer()
    accelerator = install_runtime_fakes(monkeypatch, trainer)

    trainer.train(make_args(tmp_path, sample_at_first=True))

    assert trainer.loader_dtype is torch.float32
    assert trainer.forward_wrapper_called
    assert trainer.compile_called_before_prepare
    assert accelerator.model_prepare_calls == 1
    assert accelerator.model_device_placement == [False]
    assert accelerator.accumulate_model is trainer.forward_model
    assert trainer.optimizer_param_ids == {id(param) for param in trainer.raw_model.parameters()}
    assert trainer.optimizer_param_ids.isdisjoint({id(param) for param in trainer.sample_vae.parameters()})
    assert all(not param.requires_grad for param in trainer.sample_vae.parameters())
    assert not torch.equal(trainer.initial_weight, trainer.raw_model.linear.weight.detach())
    assert trainer.raw_model.swap_config.supports_backward is True
    assert trainer.raw_model.move_except_swap_calls == 2
    assert trainer.raw_model.prepare_swap_calls == 1
    assert trainer.saved_state_dict.keys() == {"linear.weight", "linear.bias"}
    assert trainer.saved_metadata["ss_training_type"] == "full-finetune"
    assert trainer.saved_metadata["ss_full_finetune"] == "True"
    assert trainer.saved_metadata["ss_fp8_base"] == "False"
    assert trainer.saved_metadata["is_lora"] == "False"
    assert "modelspec.architecture" not in trainer.full_metadata
    assert "ss_steps" not in trainer.full_metadata
    assert trainer.optimizer_mode_events[:4] == ["train", "eval", "sample", "train"]


def test_resume_is_rejected_before_accelerator_or_model_loading(tmp_path, monkeypatch):
    trainer = TinyFullTrainer()

    def fail_prepare(_args):
        raise AssertionError("accelerator should not be prepared for unsupported resume")

    monkeypatch.setattr(full_finetune, "prepare_accelerator", fail_prepare)

    with pytest.raises(NotImplementedError, match="--resume"):
        trainer.train(make_args(tmp_path, resume="state-dir"))

    assert not hasattr(trainer, "loaded_model")


def test_sampling_failure_restores_mode_rng_swap_and_runs_after_hook(tmp_path, monkeypatch):
    trainer = TinyFullTrainer()
    install_runtime_fakes(monkeypatch, trainer)

    def fail_sample(accelerator, _args, _epoch, _steps, _vae, transformer, _sample_parameters, _dit_dtype):
        trainer.rng_at_sample_entry = torch.get_rng_state().clone()
        raw_model = accelerator.unwrap_model(transformer, keep_fp32_wrapper=False)
        raw_model.eval()
        raw_model.switch_block_swap_for_inference()
        torch.rand(8)
        raise RuntimeError("sample failed")

    trainer.sample_images = fail_sample

    with pytest.raises(RuntimeError, match="sample failed"):
        trainer.train(make_args(tmp_path, sample_at_first=True))

    assert trainer.after_sample_called
    assert trainer.raw_model.training
    assert trainer.raw_model.swap_training_calls == 1
    assert torch.equal(torch.get_rng_state(), trainer.rng_at_sample_entry)
    assert trainer.optimizer_mode_events[-1] == "eval"


def test_full_loss_boundary_casts_target_before_architecture_loss_and_backward():
    class ArchitectureSpecificLoss:
        def compute_loss(
            self,
            args,
            output,
            timesteps,
            noise_scheduler,
            dit_dtype,
            network_dtype,
            global_step,
        ):
            del timesteps, noise_scheduler, dit_dtype, network_dtype, global_step
            self.architecture_target_dtype = output.target.dtype
            loss = torch.nn.functional.mse_loss(output.pred, output.target, reduction="mean")
            return loss * args.architecture_loss_scale, {"loss/architecture": args.architecture_loss_scale}

    class FullArchitectureTrainer(FullFineTuningTrainerMixin, ArchitectureSpecificLoss):
        pass

    trainer = FullArchitectureTrainer()
    prediction = torch.tensor([1.0], dtype=torch.bfloat16, requires_grad=True)
    output = DiTOutput(pred=prediction, target=torch.zeros(1, dtype=torch.float32))

    loss, metrics = trainer.compute_loss(
        SimpleNamespace(architecture_loss_scale=3.0),
        output,
        torch.tensor([1.0]),
        None,
        torch.bfloat16,
        torch.bfloat16,
        0,
    )
    assert metrics == {"loss/architecture": 3.0}
    loss.backward()

    assert trainer.architecture_target_dtype is torch.bfloat16
    assert prediction.grad is not None
    assert prediction.grad.dtype is torch.bfloat16


def test_negative_block_swap_is_rejected_before_full_model_lifecycle(tmp_path, monkeypatch):
    trainer = TinyFullTrainer()
    install_runtime_fakes(monkeypatch, trainer)

    with pytest.raises(ValueError, match="--blocks_to_swap"):
        trainer.train(make_args(tmp_path, blocks_to_swap=-1, sample_prompts=None))

    assert not hasattr(trainer, "loaded_model")


def test_attention_backend_is_validated_before_dataset_and_sampling_setup(tmp_path, monkeypatch):
    trainer = TinyFullTrainer()
    install_runtime_fakes(monkeypatch, trainer)
    dataset_calls = []

    def fail_if_dataset_is_built(_args):
        dataset_calls.append(True)
        raise AssertionError("dataset must not be built before attention validation")

    monkeypatch.setattr(trainer, "_build_dataset", fail_if_dataset_is_built)

    with pytest.raises(ValueError, match="either --sdpa"):
        trainer.train(
            make_args(
                tmp_path,
                sample_prompts=None,
                sdpa=False,
                flash_attn=False,
                flash3=False,
                sage_attn=False,
                xformers=False,
            )
        )

    assert dataset_calls == []
    assert not hasattr(trainer, "loaded_model")


@pytest.mark.parametrize(
    "distributed_type",
    [
        DistributedType.DEEPSPEED,
        DistributedType.FSDP,
        DistributedType.TP,
        DistributedType.MEGATRON_LM,
        DistributedType.XLA,
    ],
)
def test_non_replica_backend_is_rejected_before_dataset_or_model_allocation(
    tmp_path,
    monkeypatch,
    distributed_type,
):
    trainer = TinyFullTrainer()
    accelerator = install_runtime_fakes(monkeypatch, trainer)
    accelerator.distributed_type = distributed_type
    dataset_calls = []

    def fail_if_dataset_is_built(_args):
        dataset_calls.append(True)
        raise AssertionError("dataset allocation must not start for an unsupported distributed backend")

    monkeypatch.setattr(trainer, "_build_dataset", fail_if_dataset_is_built)

    with pytest.raises(ValueError) as error:
        trainer.train(make_args(tmp_path, sample_prompts=None, blocks_to_swap=0))

    assert distributed_type.value in str(error.value)
    assert "DDP" in str(error.value)
    assert dataset_calls == []
    assert not hasattr(trainer, "loaded_model")


@pytest.mark.parametrize(
    ("retention_kind", "overrides", "expected_checkpoint"),
    [
        (
            "step",
            {"max_train_steps": 2, "save_every_n_steps": 1, "save_last_n_steps": 1},
            full_finetune.train_utils.get_step_ckpt_name("tiny", 0),
        ),
        (
            "epoch",
            {"max_train_steps": 3, "save_every_n_epochs": 1, "save_last_n_epochs": 1},
            full_finetune.train_utils.get_epoch_ckpt_name("tiny", 0),
        ),
    ],
)
def test_training_routes_step_and_epoch_checkpoint_retention_through_all_rank_method(
    tmp_path,
    monkeypatch,
    retention_kind,
    overrides,
    expected_checkpoint,
):
    trainer = TinyFullTrainer()
    accelerator = install_runtime_fakes(monkeypatch, trainer)
    removal_calls = []

    def record_all_rank_removal(call_accelerator, _args, ckpt_name):
        removal_calls.append((call_accelerator, ckpt_name))

    monkeypatch.setattr(trainer, "_remove_checkpoint_all_ranks", record_all_rank_removal, raising=False)

    trainer.train(make_args(tmp_path, sample_prompts=None, blocks_to_swap=0, **overrides))

    assert removal_calls == [(accelerator, expected_checkpoint)], retention_kind


@pytest.mark.parametrize("is_main_process", [False, True])
def test_save_state_all_ranks_brackets_main_rank_retention(is_main_process):
    events = []

    def reduce(tensor, reduction="sum"):
        assert reduction == "sum"
        events.append(("reduce", int(tensor.item())))
        return tensor

    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        is_main_process=is_main_process,
        reduce=reduce,
        save_state=lambda state_dir: events.append(("save", state_dir)),
        wait_for_everyone=lambda: events.append(("barrier", None)),
    )

    save_state_all_ranks(accelerator, SimpleNamespace(), "state-dir", lambda: events.append(("retention", None)))

    expected = [("save", "state-dir"), ("barrier", None)]
    if is_main_process:
        expected.append(("retention", None))
    expected.append(("reduce", 0))
    expected.append(("barrier", None))
    assert events == expected


def test_save_state_all_ranks_reaches_second_barrier_when_retention_fails():
    events = []
    retention_error = RuntimeError("retention failed")

    def reduce(tensor, reduction="sum"):
        assert reduction == "sum"
        events.append(("reduce", int(tensor.item())))
        return tensor

    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        is_main_process=True,
        reduce=reduce,
        save_state=lambda state_dir: events.append(("save", state_dir)),
        wait_for_everyone=lambda: events.append(("barrier", None)),
    )

    def fail_retention():
        events.append(("retention", None))
        raise retention_error

    with pytest.raises(RuntimeError, match="retention failed") as caught:
        save_state_all_ranks(accelerator, SimpleNamespace(), "state-dir", fail_retention)

    assert caught.value is retention_error
    assert events == [
        ("save", "state-dir"),
        ("barrier", None),
        ("retention", None),
        ("reduce", 1),
        ("barrier", None),
    ]


def test_save_state_all_ranks_propagates_main_failure_to_a_non_main_rank():
    events = []

    def reduce(tensor, reduction="sum"):
        assert reduction == "sum"
        events.append(("reduce", int(tensor.item())))
        return torch.ones_like(tensor)

    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        is_main_process=False,
        reduce=reduce,
        save_state=lambda state_dir: events.append(("save", state_dir)),
        wait_for_everyone=lambda: events.append(("barrier", None)),
    )

    with pytest.raises(RuntimeError, match=r"state retention.*state-dir.*main process"):
        save_state_all_ranks(
            accelerator,
            SimpleNamespace(),
            "state-dir",
            lambda: events.append(("retention", None)),
        )

    assert events == [
        ("save", "state-dir"),
        ("barrier", None),
        ("reduce", 0),
        ("barrier", None),
    ]
