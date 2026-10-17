"""Checked optimizer-boundary checkpoints for both full and adapter training."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch

from musubi_tuner.dlssnr.artifacts import write_json
from musubi_tuner.dlssnr.identity import file_sha256
from musubi_tuner.training.dlssnr_services import capture_rng, coordinated_call, gather_rank_values, restore_rng


def save_state(folder, optimizer, identity, update, cursor, *, accelerator=None, scheduler=None, ema=None):
    state = {
        "rank": accelerator.process_index if accelerator is not None else 0,
        "rng": capture_rng(accelerator.device if accelerator is not None else None),
        "scaler": accelerator.scaler.state_dict() if accelerator is not None and accelerator.scaler is not None else None,
    }
    rank_states = gather_rank_values(accelerator, state)
    coordinated_call(
        accelerator, lambda: _write_state(folder, optimizer, identity, update, cursor, rank_states, scheduler, ema), main_only=True
    )


def _write_state(folder, optimizer, identity, update, cursor, rank_states, scheduler, ema=None):
    folder = Path(folder)
    payload = {
        "schema": "dlssnr_train_state_v3",
        "identity": identity,
        "global_update": update,
        "consumed_samples": cursor,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rank_states": rank_states,
    }
    if ema is not None:
        if ema.num_updates != update:
            raise ValueError("EMA update count does not match the checkpoint boundary")
        payload["ema"] = ema.state_dict()
    with tempfile.NamedTemporaryFile(dir=folder, delete=False, suffix=".pt") as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, folder / "trainer_state.pt")
    finally:
        temporary.unlink(missing_ok=True)
    files = {
        path.name: file_sha256(path) for path in folder.iterdir() if path.is_file() and path.name != "checkpoint_manifest.json"
    }
    write_json(folder / "checkpoint_manifest.json", {"schema": "dlssnr_checkpoint_v3", "files": files})


def read_state(folder, identity):
    folder = Path(folder)
    manifest_file = folder / "checkpoint_manifest.json"
    if not manifest_file.is_file():
        raise ValueError("not a complete v3 training state; old weights can only be used as a warm start")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    if manifest.get("schema") != "dlssnr_checkpoint_v3" or "trainer_state.pt" not in files:
        raise ValueError("invalid or old training state manifest; v3 is required, use old weights only as a warm start")
    if not ({"adapter.safetensors", "model.safetensors"} & files.keys()):
        raise ValueError("training state contains no model/adapter")
    for name, digest in files.items():
        if Path(name).name != name or ":" in name or not (folder / name).is_file() or file_sha256(folder / name) != digest:
            raise ValueError(f"incomplete or modified training state file: {name}")
    payload = torch.load(folder / "trainer_state.pt", map_location="cpu", weights_only=True)
    if payload.get("schema") != "dlssnr_train_state_v3" or payload.get("identity") != identity:
        raise ValueError("resume identity mismatch: effective config, data, base, implementation or runtime changed")
    for name in ("global_update", "consumed_samples"):
        if type(payload.get(name)) is not int or payload[name] < 0:
            raise ValueError(f"invalid resume counter {name}")
    states = payload.get("rank_states")
    world_size = identity.get("world_size", 1)
    if (
        not isinstance(states, list)
        or len(states) != world_size
        or any(not isinstance(state, dict) or state.get("rank") != rank for rank, state in enumerate(states))
    ):
        raise ValueError("invalid per-rank training state")
    return payload


def restore_state(payload, optimizer, *, accelerator=None, scheduler=None, ema=None):
    if (ema is None) != (payload.get("ema") is None):
        raise ValueError("resume EMA state does not match the requested training configuration")
    if ema is not None:
        ema.load_state_dict(payload["ema"], expected_updates=payload["global_update"])
    if (scheduler is None) != (payload.get("scheduler") is None):
        raise ValueError("resume scheduler state does not match the requested training configuration")
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler"])
    rank = accelerator.process_index if accelerator is not None else 0
    state = payload["rank_states"][rank]
    scaler = accelerator.scaler if accelerator is not None else None
    if (scaler is None) != (state["scaler"] is None):
        raise ValueError("resume scaler does not match the requested precision")
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
