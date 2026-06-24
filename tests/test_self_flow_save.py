import torch
from safetensors.torch import load_file

from musubi_tuner.flux_2_train_network_self_flow import Flux2SelfFlowNetworkTrainer

from .test_self_flow_call_dit import FakeAccelerator, make_args


class SavingStubNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_w = torch.nn.Parameter(torch.ones(4))

    def save_weights(self, file, dtype, metadata):
        from safetensors.torch import save_file

        state = {k: v.detach().to(dtype) if dtype else v.detach() for k, v in self.state_dict().items()}
        save_file(state, file, metadata={k: str(v) for k, v in (metadata or {}).items()})


class PrintingAccelerator(FakeAccelerator):
    def print(self, *a, **k):
        pass


def test_companion_files_written(tmp_path):
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(output_dir=str(tmp_path), huggingface_repo_id=None)
    net = SavingStubNetwork()
    trainer.rep_proj = torch.nn.Linear(4, 4)
    trainer.ema_lora_state = {"lora_w": torch.full((4,), 2.0)}

    trainer.on_post_save(
        args,
        PrintingAccelerator(),
        net,
        None,
        ckpt_name="lora-000010.safetensors",
        save_dtype=torch.float32,
        metadata={},
        force_sync_upload=False,
    )

    ema_path = tmp_path / "lora-000010-ema.safetensors"
    proj_path = tmp_path / "lora-000010-proj.safetensors"
    assert ema_path.exists() and proj_path.exists()
    assert torch.equal(load_file(str(ema_path))["lora_w"], torch.full((4,), 2.0))
    # student weights restored after the EMA swap-save
    assert torch.equal(net.lora_w.detach(), torch.ones(4))


def test_noop_without_self_flow(tmp_path):
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(self_flow=False, output_dir=str(tmp_path))
    trainer.on_post_save(args, PrintingAccelerator(), SavingStubNetwork(), None, "x.safetensors", None, {}, False)
    assert list(tmp_path.iterdir()) == []
