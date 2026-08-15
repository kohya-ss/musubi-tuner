from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest
from accelerate import Accelerator

import musubi_tuner.training.full_finetune as full_finetune
import musubi_tuner.training.trainer_base as trainer_base
from test_full_finetune_runtime import TinyDataset, TinyFullTrainer, install_runtime_fakes, make_args


class ResumeDataset(TinyDataset):
    num_train_items = 5

    def __len__(self):
        return 5


class ResumeTrainer(TinyFullTrainer):
    interrupt_after_state_save = False

    def _build_dataset(self, _args):
        dataset = ResumeDataset()
        return dataset, lambda items: items[0], SimpleNamespace(value=0)

    def get_lr_scheduler(self, args, optimizer, num_processes):
        scheduler = super().get_lr_scheduler(args, optimizer, num_processes)
        self.scheduler = scheduler
        return scheduler

    def _save_step_state(self, accelerator, args, global_step):
        super()._save_step_state(accelerator, args, global_step)
        if self.interrupt_after_state_save:
            raise TrainingInterrupted(self)


class TrainingInterrupted(Exception):
    pass


def _run_training(
    monkeypatch,
    output_dir,
    *,
    max_train_steps,
    resume=None,
    save_at_step=None,
    interrupt_after_state_save=False,
    save_state_on_train_end=False,
    enable_tracker=False,
    seed=123,
):
    trainer = ResumeTrainer()
    trainer.interrupt_after_state_save = interrupt_after_state_save
    trainer.skipped_batches = []
    trainer.tracker_logs = []
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=1, mixed_precision="no")
    if enable_tracker:
        accelerator.trackers.append(SimpleNamespace(finish=lambda: None))
        accelerator.init_trackers = lambda *_args, **_kwargs: None
        accelerator.log = lambda values, step=None: trainer.tracker_logs.append((values, step))
    skip_first_batches = accelerator.skip_first_batches

    def record_skip(dataloader, num_batches):
        trainer.skipped_batches.append(num_batches)
        return skip_first_batches(dataloader, num_batches)

    accelerator.skip_first_batches = record_skip
    monkeypatch.setattr(full_finetune, "prepare_accelerator", lambda _args: accelerator)
    monkeypatch.setattr(full_finetune, "clean_memory_on_device", lambda _device: None)
    monkeypatch.setattr(
        full_finetune.sai_model_spec,
        "build_metadata",
        lambda *args, **kwargs: {"modelspec.architecture": "tiny", "is_lora": str(kwargs["is_lora"])},
    )

    trainer.run_args = make_args(
        output_dir,
        blocks_to_swap=0,
        compile=False,
        sample_prompts=None,
        max_train_steps=max_train_steps,
        save_every_n_steps=save_at_step,
        save_state=save_at_step is not None,
        save_state_on_train_end=save_state_on_train_end,
        resume=None if resume is None else str(resume),
        seed=seed,
    )
    trainer.train(trainer.run_args)
    return trainer


def test_interrupted_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    uninterrupted = _run_training(monkeypatch, tmp_path / "uninterrupted", max_train_steps=4)

    interrupted_dir = tmp_path / "interrupted"
    with pytest.raises(TrainingInterrupted):
        _run_training(
            monkeypatch,
            interrupted_dir,
            max_train_steps=4,
            save_at_step=2,
            interrupt_after_state_save=True,
        )
    state_dir = interrupted_dir / "tiny-step00000002-state"
    resumed = _run_training(monkeypatch, tmp_path / "resumed", max_train_steps=4, resume=state_dir)

    uninterrupted_state = uninterrupted.raw_model.state_dict()
    resumed_state = resumed.raw_model.state_dict()
    assert uninterrupted_state.keys() == resumed_state.keys()
    assert all(torch.equal(uninterrupted_state[key], resumed_state[key]) for key in uninterrupted_state)
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
    assert resumed.training_progress == uninterrupted.training_progress
    assert resumed.skipped_batches == [2]
    assert resumed.training_progress.global_step == 4
    assert resumed.training_progress.epoch == 0
    assert resumed.training_progress.next_batch == 4


def test_same_max_resume_does_not_execute_an_extra_batch(tmp_path, monkeypatch):
    interrupted_dir = tmp_path / "same-max-interrupted"
    with pytest.raises(TrainingInterrupted) as interruption:
        _run_training(
            monkeypatch,
            interrupted_dir,
            max_train_steps=2,
            save_at_step=2,
            interrupt_after_state_save=True,
        )
    interrupted = interruption.value.args[0]

    resumed = _run_training(
        monkeypatch,
        tmp_path / "same-max-resumed",
        max_train_steps=2,
        resume=interrupted_dir / "tiny-step00000002-state",
    )

    assert resumed.training_progress.global_step == 2
    assert resumed.training_progress.epoch == 0
    assert resumed.training_progress.next_batch == 2
    assert resumed.skipped_batches == []
    assert not resumed.forward_wrapper_called
    assert resumed.scheduler.state_dict() == interrupted.scheduler.state_dict()
    assert all(
        torch.equal(interrupted.raw_model.state_dict()[key], resumed.raw_model.state_dict()[key])
        for key in interrupted.raw_model.state_dict()
    )


def test_same_max_resume_finalizes_an_exact_epoch_boundary(tmp_path, monkeypatch):
    uninterrupted = _run_training(monkeypatch, tmp_path / "boundary-uninterrupted", max_train_steps=5)
    interrupted_dir = tmp_path / "boundary-interrupted"
    with pytest.raises(TrainingInterrupted) as interruption:
        _run_training(
            monkeypatch,
            interrupted_dir,
            max_train_steps=5,
            save_at_step=5,
            interrupt_after_state_save=True,
        )
    interrupted = interruption.value.args[0]
    assert interrupted.training_progress.global_step == 5
    assert interrupted.training_progress.epoch == 0
    assert interrupted.training_progress.next_batch == 5

    resumed = _run_training(
        monkeypatch,
        tmp_path / "boundary-resumed",
        max_train_steps=5,
        resume=interrupted_dir / "tiny-step00000005-state",
        enable_tracker=True,
    )

    assert resumed.training_progress.global_step == 5
    assert resumed.training_progress.epoch == 1
    assert resumed.training_progress.next_batch == 0
    assert resumed.skipped_batches == []
    assert not resumed.forward_wrapper_called
    assert resumed.tracker_logs == [({}, 5)]
    assert resumed.optimizer_mode_events == uninterrupted.optimizer_mode_events
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
    assert all(
        torch.equal(uninterrupted.raw_model.state_dict()[key], resumed.raw_model.state_dict()[key])
        for key in uninterrupted.raw_model.state_dict()
    )


def test_extending_final_mid_epoch_state_matches_uninterrupted_training(tmp_path, monkeypatch):
    uninterrupted = _run_training(monkeypatch, tmp_path / "extended-uninterrupted", max_train_steps=4)
    short_dir = tmp_path / "short-final"
    short = _run_training(
        monkeypatch,
        short_dir,
        max_train_steps=2,
        save_state_on_train_end=True,
    )

    assert short.training_progress.global_step == 2
    assert short.training_progress.epoch == 0
    assert short.training_progress.next_batch == 2

    resumed = _run_training(
        monkeypatch,
        tmp_path / "extended-resumed",
        max_train_steps=4,
        resume=short_dir / "tiny-state",
    )

    assert resumed.training_progress == uninterrupted.training_progress
    assert resumed.training_progress.global_step == 4
    assert resumed.training_progress.epoch == 0
    assert resumed.training_progress.next_batch == 4
    assert resumed.skipped_batches == [2]
    assert resumed.scheduler.state_dict() == uninterrupted.scheduler.state_dict()
    assert all(
        torch.equal(uninterrupted.raw_model.state_dict()[key], resumed.raw_model.state_dict()[key])
        for key in uninterrupted.raw_model.state_dict()
    )


def test_resume_requires_registered_training_progress_file(tmp_path, monkeypatch):
    state_dir = tmp_path / "legacy-state"
    state_dir.mkdir()
    trainer = TinyFullTrainer()
    install_runtime_fakes(monkeypatch, trainer)

    with pytest.raises(RuntimeError, match="TrainingProgress"):
        trainer.train(make_args(tmp_path, resume=str(state_dir)))


def test_seed_none_resume_restores_checkpointed_effective_sampler_seed(tmp_path, monkeypatch):
    interrupted_dir = tmp_path / "interrupted-none-seed"
    with pytest.raises(TrainingInterrupted) as interruption:
        _run_training(
            monkeypatch,
            interrupted_dir,
            max_train_steps=4,
            save_at_step=2,
            interrupt_after_state_save=True,
            seed=None,
        )
    interrupted = interruption.value.args[0]
    effective_seed = interrupted.run_args.seed
    assert effective_seed is not None

    uninterrupted = _run_training(
        monkeypatch,
        tmp_path / "uninterrupted-none-seed",
        max_train_steps=4,
        seed=effective_seed,
    )
    monkeypatch.setattr(trainer_base.random, "randint", lambda *_args: (effective_seed + 1) % (2**32))
    resumed = _run_training(
        monkeypatch,
        tmp_path / "resumed-none-seed",
        max_train_steps=4,
        resume=interrupted_dir / "tiny-step00000002-state",
        seed=None,
    )

    assert resumed.run_args.seed == effective_seed
    assert resumed.training_progress.sampler_seed == effective_seed
    assert resumed.training_progress == uninterrupted.training_progress
    assert all(
        torch.equal(uninterrupted.raw_model.state_dict()[key], resumed.raw_model.state_dict()[key])
        for key in uninterrupted.raw_model.state_dict()
    )


@pytest.mark.parametrize(
    ("remote_files", "preflight_passes"),
    [
        (["states/tiny/custom_checkpoint_0.pkl", "states/tiny/model.safetensors"], True),
        (["states/tiny/model.safetensors"], False),
    ],
)
def test_huggingface_resume_preflights_training_progress_before_allocation(
    tmp_path,
    monkeypatch,
    remote_files,
    preflight_passes,
):
    listed = []

    def list_dir(**kwargs):
        listed.append(kwargs)
        return [SimpleNamespace(rfilename=filename) for filename in remote_files]

    monkeypatch.setattr(full_finetune.huggingface_utils, "list_dir", list_dir)
    trainer = TinyFullTrainer()

    class PreflightPassed(Exception):
        pass

    def stop_after_preflight(_args):
        raise PreflightPassed

    monkeypatch.setattr(trainer, "_validate_args_and_init", stop_after_preflight)
    args = make_args(
        tmp_path,
        resume="owner/repo/states/tiny:review:model",
        resume_from_huggingface=True,
        huggingface_token="secret",
    )

    if preflight_passes:
        with pytest.raises(PreflightPassed):
            trainer.train(args)
    else:
        with pytest.raises(RuntimeError, match="TrainingProgress"):
            trainer.train(args)

    assert listed == [
        {
            "repo_id": "owner/repo",
            "subfolder": "states/tiny",
            "revision": "review",
            "token": "secret",
            "repo_type": "model",
        }
    ]
    assert not hasattr(trainer, "loaded_model")
