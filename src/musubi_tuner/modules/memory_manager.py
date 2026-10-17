"""
Self-contained Memory Management, Headroom Guard, and OOM-Recovery Module
for Musubi-Tuner training and inference.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, TypeVar

import psutil
import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")

def get_local_vmm_dll() -> str | None:
    """Looks for the VMM allocator binary strictly in this module's folder."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "vmm_alloc_win_amd64.dll",
        here / "vmm_alloc_linux_x86_64.so",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def get_local_ram_dll() -> str | None:
    """Looks for the RAM allocator binary strictly in this module's folder."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "ram_alloc_win_amd64.dll",
        here / "ram_alloc_linux_x86_64.so",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return None


def install_mmgp_ram_allocator(cache_gb: float = 2.0, pressure_gb: float = 8.0) -> ctypes.CDLL | None:
    """Installs the MMGP CPU RAM allocator to stop Windows mimalloc from hoarding system RAM."""
    dll_path = get_local_ram_dll()
    if not dll_path:
        here = Path(__file__).resolve().parent
        print(f"[MemoryManager] ram_alloc binary not found in {here}. CPU RAM allocator skipped.")
        return None
    try:
        lib = ctypes.CDLL(dll_path)
        lib.ra_install.argtypes = (ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
        lib.ra_install.restype = ctypes.c_int
        min_bytes = 1 * 1024 * 1024  # 1 MB
        cache_bytes = int(cache_gb * (1024 ** 3))
        pressure_bytes = int(pressure_gb * (1024 ** 3))
        if lib.ra_install(min_bytes, cache_bytes, pressure_bytes) == 0:
            print(f"[MemoryManager] Successfully installed MMGP CPU RAM Allocator (cache={cache_gb}GB, pressure={pressure_gb}GB)")
            return lib
        else:
            print("[MemoryManager] ra_install returned non-zero. CPU RAM allocator skipped.")
    except Exception as e:
        print(f"[MemoryManager] Could not load MMGP RAM allocator ({e}).")
    return None


def setup_memory_environment(
    allocator_type: str = "mmgp_spill_ram",
    vram_limit_gb: float | None = None,
    ram_cache_gb: float = 2.0,
    ram_pressure_gb: float = 4.0,
) -> None:
    """
    Configures the memory allocator. MUST be called before torch.cuda initializes.
    Options for allocator_type:
      - 'mmgp': MMGP VMM Allocator (strict VRAM, no spilling)
      - 'mmgp_ram': MMGP VMM Allocator + MMGP CPU RAM Allocator
      - 'mmgp_spill': MMGP VMM Allocator with pinned Host-RAM spilling enabled
      - 'mmgp_spill_ram': MMGP VMM Allocator with spilling + MMGP CPU RAM Allocator
      - 'pytorch': PyTorch native allocator with expandable_segments:True
      - 'default': Stock PyTorch allocator
    """
    allocator_type = (allocator_type or "mmgp_spill_ram").lower().strip()

    is_mmgp = allocator_type in ("mmgp", "mmgp_ram", "mmgp_spill", "mmgp_spill_ram")
    use_spill = allocator_type in ("mmgp_spill", "mmgp_spill_ram")
    use_ram = allocator_type in ("mmgp_ram", "mmgp_spill_ram")

    if is_mmgp:
        vmm_path = get_local_vmm_dll()
        if vmm_path:
            try:
                vmm_lib = ctypes.CDLL(vmm_path)
                vmm_lib.vmm_check.argtypes = (ctypes.c_int,)
                status = vmm_lib.vmm_check(0)
                if status == 0:
                    custom_alloc = torch.cuda.memory.CUDAPluggableAllocator(
                        vmm_path, "vmm_alloc", "vmm_free"
                    )
                    torch.cuda.memory.change_current_allocator(custom_alloc)

                    spill_flag = 3 if use_spill else 0
                    vmm_lib.vmm_configure.argtypes = (ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int)
                    vmm_lib.vmm_configure(256 << 20, 32 << 20, spill_flag)

                    if vram_limit_gb is not None and vram_limit_gb > 0:
                        vmm_lib.vmm_set_vram_limit.argtypes = (ctypes.c_int64,)
                        vmm_lib.vmm_set_vram_limit(int(vram_limit_gb * (1024 ** 3)))
                        print(f"[MemoryManager] Set MMGP VRAM limit to {vram_limit_gb:.2f} GB")

                    print(f"[MemoryManager] Successfully installed MMGP VMM Allocator (spill={use_spill})")

                    if use_ram:
                        ram_lib = install_mmgp_ram_allocator(cache_gb=ram_cache_gb, pressure_gb=ram_pressure_gb)
                        if ram_lib is not None and hasattr(vmm_lib, "vmm_set_ram_release") and hasattr(ram_lib, "ra_release"):
                            vmm_lib.vmm_set_ram_release.argtypes = (ctypes.c_void_p,)
                            vmm_lib.vmm_set_ram_release(ctypes.cast(ram_lib.ra_release, ctypes.c_void_p))
                    return
                else:
                    print(f"[MemoryManager] MMGP vmm_check failed with status {status}. Falling back to PyTorch native.")
            except Exception as e:
                print(f"[MemoryManager] Could not load MMGP VMM DLL ({e}). Falling back to PyTorch native.")
        else:
            here = Path(__file__).resolve().parent
            print(f"[MemoryManager] MMGP vmm binary not found in {here}. Falling back to PyTorch native.")
        allocator_type = "pytorch"

    if allocator_type == "pytorch":
        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        flags = []
        if "expandable_segments" not in alloc_conf:
            flags.append("expandable_segments:True")
        if "garbage_collection_threshold" not in alloc_conf:
            flags.append("garbage_collection_threshold:0.75")
        if flags:
            new_conf = ",".join(filter(None, [alloc_conf, *flags]))
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf
            print(f"[MemoryManager] Configured native PYTORCH_CUDA_ALLOC_CONF='{new_conf}'")
    else:
        print("[MemoryManager] Using default stock PyTorch allocator.")

def get_vram_info(device: torch.device | str | int | None = None) -> dict[str, float]:
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
    vm = psutil.virtual_memory()
    return {
        "total_gb": vm.total / (1024 ** 3),
        "available_gb": vm.available / (1024 ** 3),
        "used_percent": vm.percent,
    }


def flush_memory(device: torch.device | str | int | None = None, deep: bool = False) -> None:
    if torch.cuda.is_available():
        device = torch.device(device or torch.cuda.current_device())
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
    start_time = time.time()
    waited = False
    while True:
        flush_memory(device=device, deep=False)
        ram = get_ram_info()
        vram = get_vram_info(device=device)

        if ram["available_gb"] >= min_ram_gb and vram["free_gb"] >= min_vram_gb:
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

def is_oom_error(exception: BaseException) -> bool:
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
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not is_oom_error(exc):
                raise exc
            if attempt == max_retries:
                logger.error(f"[MemoryManager] Step failed permanently after {max_retries} OOM retry attempts.")
                raise exc
            logger.warning(
                f"[MemoryManager] OOM caught on attempt {attempt + 1}/{max_retries + 1}: {exc}. "
                f"Flushing memory and backing off for {backoff_sec * (attempt + 1):.1f}s..."
            )
            flush_memory(device=device, deep=True)
            if on_retry is not None:
                try:
                    on_retry(attempt + 1, exc)
                except Exception as cb_err:
                    logger.warning(f"[MemoryManager] on_retry callback failed: {cb_err}")
            time.sleep(backoff_sec * (attempt + 1))

def pace_dataloader(
    dataloader: Iterable[T],
    min_ram_gb: float = 3.5,
    min_vram_gb: float = 1.0,
    device: torch.device | str | int | None = None,
) -> Generator[T, None, None]:
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