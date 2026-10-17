"""
Self-contained Memory Management, Headroom Guard, and OOM-Recovery Module
for Musubi-Tuner training and inference.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from typing import Any, Callable, Generator, Iterable, TypeVar

import psutil
import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ---------------------------------------------------------------------------
# 1. Environment & Allocator Setup
# ---------------------------------------------------------------------------

def setup_memory_environment() -> None:
    """
    Configures PyTorch CUDA allocator environment variables.
    MUST be called before torch.cuda initializes any context.
    """
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    flags = []

    # expandable_segments: True eliminates virtual memory address fragmentation
    if "expandable_segments" not in alloc_conf:
        flags.append("expandable_segments:True")

    # Keep small block splitting efficient
    if "garbage_collection_threshold" not in alloc_conf:
        flags.append("garbage_collection_threshold:0.8")

    if flags:
        new_conf = ",".join(filter(None, [alloc_conf, *flags]))
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf
        logger.info(f"[MemoryManager] Configured PYTORCH_CUDA_ALLOC_CONF='{new_conf}'")


# ---------------------------------------------------------------------------
# 2. Status & Metric Probes
# ---------------------------------------------------------------------------

def get_vram_info(device: torch.device | str | int | None = None) -> dict[str, float]:
    """Returns total, free, allocated, and reserved VRAM in GB."""
    if not torch.cuda.is_available():
        return {"total_gb": 0.0, "free_gb": 0.0, "allocated_gb": 0.0, "reserved_gb": 0.0}

    device = torch.device(device or torch.cuda.current_device())
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)

    return {
        "total_gb": total_bytes / (1024 ** 3),
        "free_gb": free_bytes / (1024 ** 3),
        "allocated_gb": allocated_bytes / (1024 ** 3),
        "reserved_gb": reserved_bytes / (1024 ** 3),
    }


def get_ram_info() -> dict[str, float]:
    """Returns system RAM metrics in GB."""
    vm = psutil.virtual_memory()
    return {
        "total_gb": vm.total / (1024 ** 3),
        "available_gb": vm.available / (1024 ** 3),
        "used_percent": vm.percent,
    }


# ---------------------------------------------------------------------------
# 3. Flushes & Headroom Checks (Wait-and-Clear)
# ---------------------------------------------------------------------------

def flush_memory(device: torch.device | str | int | None = None, deep: bool = False) -> None:
    """
    Synchronizes streams, runs Python cyclic GC, and purges PyTorch's cache.
    If deep=False, it only flushes if VRAM reservation is above 90% to avoid driver stalls.
    """
    if torch.cuda.is_available():
        device = torch.device(device or torch.cuda.current_device())
        
        # If deep or if VRAM is running out of headroom, proactively empty cache
        if deep:
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        else:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            reserved_bytes = torch.cuda.memory_reserved(device)
            # If driver has less than 1.5 GB free and PyTorch reserved over 85%, release unused blocks
            if free_bytes < (1.5 * 1024**3) and (reserved_bytes / total_bytes) > 0.85:
                torch.cuda.empty_cache()

    gc.collect()


def ensure_memory_headroom(
    min_ram_gb: float = 3.0,
    min_vram_gb: float = 1.0,
    max_wait_sec: float = 20.0,
    poll_interval: float = 0.5,
    device: torch.device | str | int | None = None,
    strict: bool = False,
) -> bool:
    """
    Checks if enough free RAM and VRAM are available. If not, it pauses,
    actively flushes caches and GC, and polls until memory recovers or times out.

    Returns True if adequate headroom is found, False if timed out (or raises if strict=True).
    """
    start_time = time.time()
    waited = False

    while True:
        flush_memory(device=device, deep=False)

        ram = get_ram_info()
        vram = get_vram_info(device=device)

        ram_ok = ram["available_gb"] >= min_ram_gb
        vram_ok = vram["free_gb"] >= min_vram_gb

        if ram_ok and vram_ok:
            if waited:
                logger.info(
                    f"[MemoryManager] Memory recovered: RAM free={ram['available_gb']:.2f}GB, "
                    f"VRAM free={vram['free_gb']:.2f}GB (waited {time.time() - start_time:.1f}s)"
                )
            return True

        elapsed = time.time() - start_time
        if elapsed >= max_wait_sec:
            msg = (
                f"[MemoryManager] Headroom warning: Timed out waiting after {elapsed:.1f}s. "
                f"RAM available: {ram['available_gb']:.2f}GB (needed {min_ram_gb:.2f}GB), "
                f"VRAM free: {vram['free_gb']:.2f}GB (needed {min_vram_gb:.2f}GB)."
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)
            return False

        if not waited:
            logger.warning(
                f"[MemoryManager] Low memory detected (RAM free: {ram['available_gb']:.2f}GB, "
                f"VRAM free: {vram['free_gb']:.2f}GB). Pausing to allow system recovery..."
            )
            waited = True

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# 4. OOM-Catching Retry Wrapper
# ---------------------------------------------------------------------------

def is_oom_error(exception: BaseException) -> bool:
    """Detects standard and custom CUDA / host OOM exceptions."""
    if isinstance(exception, (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError)):
        return True
    if isinstance(exception, RuntimeError):
        msg = str(exception).lower()
        oom_markers = (
            "out of memory",
            "cuda error: out of memory",
            "tried to allocate",
            "cuda_error_out_of_memory",
        )
        return any(marker in msg for marker in oom_markers)
    return False


def execute_with_retry(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = 2,
    backoff_sec: float = 1.5,
    on_retry: Callable[[int, BaseException], None] | None = None,
    device: torch.device | str | int | None = None,
    **kwargs: Any,
) -> T:
    """
    Executes `fn(*args, **kwargs)`. If an OOM is encountered:
      1. Synchronizes CUDA and flushes all caches and GC.
      2. Waits with linear backoff (giving async streams and host memory time to purge).
      3. Invokes `on_retry` callback if provided (e.g., to adjust checkpointing or log states).
      4. Retries up to `max_retries` before reraising.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not is_oom_error(exc):
                raise exc

            if attempt == max_retries:
                logger.error(
                    f"[MemoryManager] Step failed permanently after {max_retries} OOM retry attempts."
                )
                raise exc

            logger.warning(
                f"[MemoryManager] OOM caught on attempt {attempt + 1}/{max_retries + 1}: {exc}. "
                f"Flushing memory and backing off for {backoff_sec * (attempt + 1):.1f}s..."
            )

            # Emergency memory purge
            flush_memory(device=device, deep=True)

            if on_retry is not None:
                try:
                    on_retry(attempt + 1, exc)
                except Exception as cb_err:
                    logger.warning(f"[MemoryManager] on_retry callback failed: {cb_err}")

            time.sleep(backoff_sec * (attempt + 1))


# ---------------------------------------------------------------------------
# 5. Safe DataLoader Pacing Iterator
# ---------------------------------------------------------------------------

def pace_dataloader(
    dataloader: Iterable[T],
    min_ram_gb: float = 3.5,
    min_vram_gb: float = 1.0,
    device: torch.device | str | int | None = None,
) -> Generator[T, None, None]:
    """
    Wraps a DataLoader/generator to ensure host RAM and VRAM are healthy
    before fetching each batch, preventing prefetch buffer buildup.
    """
    for batch in dataloader:
        ensure_memory_headroom(
            min_ram_gb=min_ram_gb,
            min_vram_gb=min_vram_gb,
            max_wait_sec=30.0,
            poll_interval=0.5,
            device=device,
            strict=False,
        )
        yield batch