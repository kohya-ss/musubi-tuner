#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profiling stuff written by AI to help track whats using up our VRAM
Created on Tue Aug 19 18:11:14 2025

@author: blyss
"""

import inspect
import json
from collections import defaultdict
from typing import Dict, Optional

import torch
import torch.nn as nn


def _fmt_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    i = 0
    f = float(n)
    while f >= 1024 and i < len(units) - 1:
        f /= 1024
        i += 1
    return f"{f:.2f} {units[i]}"


def tensor_nbytes(t: torch.Tensor) -> int:
    if not isinstance(t, torch.Tensor):
        return 0
    # element_size() returns bytes per element for the dtype
    return t.numel() * t.element_size()


def summarize_tensors(tensors: Dict[str, torch.Tensor]) -> str:
    by_dtype = defaultdict(int)
    by_device = defaultdict(int)
    lines = []
    total = 0
    for name, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            continue
        nbytes = tensor_nbytes(t)
        total += nbytes
        by_dtype[str(t.dtype)] += nbytes
        by_device[str(t.device)] += nbytes
        lines.append(f"  • {name:<20} {list(t.shape)} {t.dtype} {t.device} = {_fmt_bytes(nbytes)}")
    lines.sort(
        key=lambda s: int(s.split()[-2].replace(".", "").replace("KiB", "").replace("MiB", "").replace("GiB", "")), reverse=True
    )
    head = [f"Tracked tensors total: {_fmt_bytes(total)}"]
    if by_dtype:
        head.append("By dtype: " + ", ".join(f"{k}: {_fmt_bytes(v)}" for k, v in by_dtype.items()))
    if by_device:
        head.append("By device: " + ", ".join(f"{k}: {_fmt_bytes(v)}" for k, v in by_device.items()))
    return "\n".join(head + lines)


def summarize_parameters(model: nn.Module, topk: int = 15) -> str:
    by_key = defaultdict(int)  # (device,dtype) -> bytes
    entries = []
    total = 0
    for name, p in model.named_parameters(recurse=True):
        if p.device.type == "meta":
            continue
        nbytes = tensor_nbytes(p)
        total += nbytes
        key = (str(p.device), str(p.dtype))
        by_key[key] += nbytes
        entries.append((nbytes, name, list(p.shape), p.dtype, p.device))
    entries.sort(reverse=True, key=lambda x: x[0])
    lines = [f"Parameters total: {_fmt_bytes(total)}"]
    if by_key:
        lines.append("By device/dtype: " + ", ".join(f"{dev}/{dt}: {_fmt_bytes(sz)}" for (dev, dt), sz in by_key.items()))
    lines.append(f"Top {min(topk, len(entries))} largest parameters:")
    for nbytes, name, shape, dtype, device in entries[:topk]:
        lines.append(f"  • {name:<40} {shape} {dtype} {device} = {_fmt_bytes(nbytes)}")
    return "\n".join(lines)


def _cuda_stats_block() -> str:
    # Works only on CUDA
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    max_alloc = torch.cuda.max_memory_allocated()
    max_rsrv = torch.cuda.max_memory_reserved()
    lines = [
        f"CUDA memory: allocated={_fmt_bytes(allocated)} reserved={_fmt_bytes(reserved)}",
        f"CUDA peaks : max_allocated={_fmt_bytes(max_alloc)} max_reserved={_fmt_bytes(max_rsrv)}",
    ]
    # Deeper stats where available
    try:
        stats = torch.cuda.memory_stats()
        active = stats.get("active_bytes.all.current", 0)
        inactive = stats.get("inactive_split_bytes.all.current", 0)
        lines.append(f"CUDA pools : active={_fmt_bytes(active)} inactive={_fmt_bytes(inactive)}")
    except Exception:
        pass
    return "\n".join(lines)


def _mps_stats_block() -> str:
    # Best-effort for MPS; APIs vary by PyTorch version
    lines = ["MPS memory:"]
    try:
        current = torch.mps.current_allocated_memory()
        driver = torch.mps.driver_allocated_memory()
        lines.append(f"  current_allocated={_fmt_bytes(current)} driver_allocated={_fmt_bytes(driver)}")
    except Exception:
        lines.append("  (MPS stats unavailable in this PyTorch build)")
    return "\n".join(lines)


def dump_cuda_memory_snapshot(path: str) -> None:
    """
    Dumps torch.cuda.memory_snapshot() to a JSON file for later offline analysis.
    Heavy but very informative (alloc blocks, sizes, streams, traces).
    """
    snap = torch.cuda.memory_snapshot()
    with open(path, "w") as f:
        json.dump(snap, f)
    print(f"[vram_probe] Wrote CUDA memory snapshot to {path}")


def vram_probe(
    tag: Optional[str] = None,
    model: Optional[nn.Module] = None,
    tracked: Optional[Dict[str, torch.Tensor]] = None,
    dump_snapshot_path: Optional[str] = None,
) -> None:
    """
    Print a compact VRAM summary. Call this anywhere (e.g., each denoise step).
    Args:
      tag: A label for this probe point (e.g., "step 12 pre-attn").
      model: If provided, parameter sizes by dtype/device + top-N largest.
      tracked: dict name->tensor for activations you care about (x, e, q, k, v, latents, etc.).
      dump_snapshot_path: If set (CUDA only), writes a JSON memory snapshot to this path.
    """
    if tag is None:
        tag = inspect.stack()[1][3]
    print(f"\n==== VRAM PROBE: {tag} ====")

    # Device overview
    devs = set()
    for obj in [tracked.values() if tracked else []]:
        for t in obj:
            if isinstance(t, torch.Tensor):
                devs.add(t.device.type)
    if model is not None:
        for p in model.parameters():
            if isinstance(p, torch.Tensor):
                devs.add(p.device.type)

    # CUDA block (if available)
    if torch.cuda.is_available():
        print(_cuda_stats_block())

    # MPS block (best-effort)
    if "mps" in devs or (hasattr(torch, "mps") and torch.backends.mps.is_available()):
        print(_mps_stats_block())

    # Parameters
    if model is not None:
        try:
            print(summarize_parameters(model))
        except Exception as e:
            print(f"(parameter summary failed: {e})")

    # Tracked activations (user-specified)
    if tracked:
        print("Tracked activations:")
        try:
            print(summarize_tensors(tracked))
        except Exception as e:
            print(f"(tracked summary failed: {e})")

    # Optional CUDA memory snapshot
    if dump_snapshot_path and torch.cuda.is_available():
        try:
            dump_cuda_memory_snapshot(dump_snapshot_path)
        except Exception as e:
            print(f"(snapshot failed: {e})")

    print("==== end probe ====\n")
