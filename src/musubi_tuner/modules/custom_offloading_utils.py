from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import time
from typing import Optional
import torch
import torch.nn as nn


# Keep these functions here for portability, and private to avoid confusion with the ones in device_utils.py
def _clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def _synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    not tested
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    # device to cpu
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    _synchronize_device(device)

    # cpu to device
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    _synchronize_device(device)


def weighs_to_device(layer: nn.Module, device: torch.device):
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None and module.__class__.__name__.endswith("Linear"):
            module.weight.data = module.weight.data.to(device, non_blocking=device.type != "cpu")


@dataclass
class BlockSwapConfig:
    """
    Construction policy for a block-swap offloader, assembled once by the training/inference script and
    passed through each architecture's ``enable_block_swap`` to ``create_offloader``.

    It holds everything the offloader constructor needs that is *not* architecture-specific. The
    architecture-specific arguments (the block-type label, the block list, and its counts) are supplied by
    ``enable_block_swap`` per block list. Adding a new offloader knob (or a whole new offloader type) means
    adding a field here and a branch in ``create_offloader`` -- the per-architecture ``enable_block_swap``
    signatures stay ``(blocks_to_swap, config)`` and never change.
    """

    device: torch.device
    supports_backward: bool
    use_pinned_memory: bool = False
    h2d_only: bool = False  # frozen-base (LoRA / LoHa / LoKr) only: H2D-only streaming, no device->host copy
    ring_size: int = 2  # (h2d_only) number of GPU ring buffers for streamed blocks; 2 = double buffering
    debug: bool = False

    @classmethod
    def from_args(cls, args, device: torch.device, supports_backward: bool) -> "BlockSwapConfig":
        """Build from a parsed-args namespace, tolerating scripts whose parser lacks the optional knobs."""
        h2d_only = getattr(args, "block_swap_h2d_only", False)

        # H2D-only streams frozen weights into a reused GPU ring buffer and loads them with an in-place copy_,
        # which bumps the autograd version of the saved weight view. Gradient checkpointing re-reads the weight
        # at recompute time (a short, safe window), so the version stays consistent; without it the backward
        # pass fails with an opaque "modified by an inplace operation ... version N expected M" error that does
        # not point at the offloader. Fail early with an actionable message instead. (Inference / forward-only
        # has no backward, so it is unaffected.)
        if h2d_only and supports_backward and not getattr(args, "gradient_checkpointing", False):
            raise ValueError(
                "--block_swap_h2d_only requires --gradient_checkpointing for training. H2D-only block swap streams"
                " frozen weights through a reused GPU ring buffer, which advances the autograd version of weights"
                " saved for backward; gradient checkpointing re-reads them at recompute time and avoids this."
                " / --block_swap_h2d_only は学習時に --gradient_checkpointing が必須です（リングバッファの上書きで"
                "backward 用に保存された重みの version が進むため。gradient checkpointing は再計算時に読み直すので回避できます）。"
            )

        ring_size = getattr(args, "block_swap_ring_size", 2)
        if ring_size < 1:
            raise ValueError("--block_swap_ring_size must be >= 1")

        return cls(
            device=device,
            supports_backward=supports_backward,
            use_pinned_memory=getattr(args, "use_pinned_memory_for_block_swap", False),
            h2d_only=h2d_only,
            ring_size=ring_size,
        )


def create_offloader(block_type: str, blocks: list[nn.Module], num_blocks: int, blocks_to_swap: int, config: BlockSwapConfig):
    """
    Create the block-swap offloader for one block list. This is the single place that selects the offloader
    implementation from ``config``; ``enable_block_swap`` only supplies the per-block-list arguments, so a new
    offloader type plugs in here without touching any architecture.
    """
    if config.h2d_only:
        # H2D-only streaming for frozen-base (LoRA) training: keep a CPU master, copy Host->Device only.
        return LoRAStreamOffloader(
            block_type,
            blocks,
            num_blocks,
            blocks_to_swap,
            config.supports_backward,
            config.device,
            ring_size=config.ring_size,
            use_pinned_memory=config.use_pinned_memory,
            debug=config.debug,
        )
    return ModelOffloader(
        block_type,
        blocks,
        num_blocks,
        blocks_to_swap,
        config.supports_backward,
        config.device,
        config.use_pinned_memory,
        debug=config.debug,
    )


class Offloader:
    """
    common offloading class
    """

    def __init__(
        self,
        block_type: str,
        num_blocks: int,
        blocks_to_swap: int,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ):
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.use_pinned_memory = use_pinned_memory

        # check if debug is enabled from os environment variable
        if not debug:
            import os

            debug = os.getenv("MUSUBI_TUNER_OFFLOADER_DEBUG", "0") == "1"

        self.debug = debug
        self.debug_block_count = 0

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"
        self.stream = torch.cuda.Stream(device=device) if self.cuda_available else None

        # Staging buffers for cuda offloading without large pinned memory. These are pinned memory buffers to speed up the transfer between CPU and GPU
        # We create one staging buffer per transfer direction (A: GPU to CPU, B: CPU to GPU)
        self.staging_buffer_a = None
        self.staging_buffer_b = None

        # Pinned buffer for cuda offloading with pinned memory. We need only one pinned buffer per layer transfer
        self.pinned_buffer = None

    def swap_weight_devices_cuda(self, device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__

        debug_print = False
        if self.debug:
            debug_print = self.debug_block_count % 10 == 0
            self.debug_block_count += 1

        class Timer:
            def __init__(self, enabled=False):
                self.enabled = enabled
                self.totals = defaultdict(float)
                self.start_time = time.perf_counter()

            @contextmanager
            def section(self, name):
                if not self.enabled:
                    yield
                    return
                t0 = time.perf_counter()
                try:
                    yield
                finally:
                    self.totals[name] += time.perf_counter() - t0

        T = Timer(enabled=debug_print)

        weight_swap_jobs = []

        # This is not working for all cases (e.g. SD3), so we need to find the corresponding modules. kept here for reference:
        # for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        #     print(module_to_cpu.__class__, module_to_cuda.__class__)
        #     if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
        #         weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

        with T.section("find modules"):
            modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
            for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
                if (
                    hasattr(module_to_cuda, "weight")
                    and module_to_cuda.weight is not None
                    and module_to_cuda.__class__.__name__.endswith("Linear")
                ):
                    module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
                    if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                        weight_swap_jobs.append(
                            (module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data)
                        )
                    else:
                        if module_to_cuda.weight.data.device.type != device.type:
                            module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

        with T.section("synchronize before swap"):
            torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value by ensuring offloading layer's calculation is done

        if not self.use_pinned_memory:
            # Minimize using pinned memory for lower shared GPU RAM usage
            stream = self.stream
            with torch.cuda.stream(stream):
                if self.staging_buffer_a is None:
                    # Create staging buffer as pinned memory (as shared GPU ram). We specify device for correct pinning on multi-GPU systems
                    self.staging_buffer_a = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                    self.staging_buffer_b = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]

                # Copy weights to staging buffers and record events
                event_b = None
                for sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                    self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
                ):
                    # CUDA to staging buffer A, non-blocking copy
                    event_a = torch.cuda.Event()
                    with T.section("cuda to staging A"):
                        sbuf_a.copy_(cuda_data_view.data, non_blocking=True)
                        event_a.record(stream)

                    # Wait for staging buffer B to be ready
                    if event_b is not None:
                        with T.section("wait staging B"):
                            event_b.synchronize()  # synchronize is needed to wait CPU process. wait_event does not work here because it waits on GPU side only

                    # CPU to staging buffer B, CPU to pinned CPU, synchronous copy. Can overlap with CUDA to staging buffer A
                    with T.section("cpu to staging B"):
                        # Making this multithreaded does not help, and 'non_blocking=True' does not help either.
                        sbuf_b.copy_(module_to_cuda.weight.data)  # BOTTLENECK

                    # Wait for staging buffer A to be ready, and CUDA data view can be reused
                    with T.section("wait staging A"):
                        event_a.synchronize()

                    # Staging buffer B to CUDA, non-blocking copy.
                    event_b = torch.cuda.Event()
                    with T.section("staging B to CUDA"):
                        cuda_data_view.copy_(sbuf_b, non_blocking=True)
                        event_b.record(stream)

                    # Staging buffer A to CPU, synchronous copy. Can overlap with staging buffer B to CUDA
                    with T.section("staging A to CPU"):
                        cpu_data_view.copy_(sbuf_a)  # BOTTLENECK

            for sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
            ):
                # Update references
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = cpu_data_view

            sync_event = event_b  # final sync event for CPU to CUDA copy

        else:
            # Use pinned memory for faster transfer between CPU and GPU, but it requires more memory
            if self.pinned_buffer is None:
                with torch.cuda.stream(self.stream):
                    # Create pinned buffer as pinned memory (as shared GPU ram). We specify device for correct pinning on multi-GPU systems
                    self.pinned_buffer = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                self.stream.synchronize()
            released_pinned_buffer = []

            events = [torch.cuda.Event() for _ in weight_swap_jobs]  # Waiting events for GPU to CPU non-blocking copy

            # Copy weights to CPU
            for event, module_pin_buf, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                events, self.pinned_buffer, weight_swap_jobs
            ):
                # CUDA to CPU, non-blocking copy
                with torch.cuda.stream(self.stream):
                    with T.section("cuda to cpu"):
                        module_pin_buf.copy_(cuda_data_view, non_blocking=True)
                        event.record(self.stream)

            # CPU to CUDA
            for event, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(events, weight_swap_jobs):
                with torch.cuda.stream(self.stream):
                    # Wait for cuda_data_view to be ready
                    with T.section("wait cpu"):
                        self.stream.wait_event(event)

                    # CPU to CUDA, non-blocking copy
                    with T.section("cpu to cuda"):
                        cuda_data_view.copy_(cpu_data_view, non_blocking=True)

            # Update references
            for module_pin_buf, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                self.pinned_buffer, weight_swap_jobs
            ):
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = module_pin_buf
                released_pinned_buffer.append(cpu_data_view)  # CPU data view can be reused as pinned buffer

            # Reuse released pinned buffers
            if not released_pinned_buffer[0].is_pinned():
                # In first time, we need to create pinned buffers because offloaded weights are not pinned yet
                with torch.cuda.stream(self.stream):
                    released_pinned_buffer = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
            self.pinned_buffer = released_pinned_buffer

            sync_event = self.stream.record_event()

        if debug_print:
            print(f"[{self.block_type}] Weight swap timing at {self.debug_block_count - 1}:")
            for name, total in T.totals.items():
                print(f"  {name}: {total * 1000:.2f}ms")
            print(
                f"Overall time: {(time.perf_counter() - T.start_time) * 1000:.2f}ms, total time in sections: {sum(T.totals.values()) * 1000:.2f}ms"
            )
        # print(
        #     f"[{self.block_type}] Swapped weights in {time.perf_counter() - start_time:.2f}s. Count of modules swapped: {len(weight_swap_jobs)}"
        # )

        return sync_event

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            sync_event = self.swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)
            sync_event = None
        return sync_event

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(
                    f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}"
                )

            dev = self.device.index if self.device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(dev)

            sync_event = self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(
                    f"[{self.block_type}] Moved blocks {bidx_to_cpu} to CPU and {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'} in {time.perf_counter() - start_time:.2f}s"
                )
            return bidx_to_cpu, bidx_to_cuda, sync_event

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda, sync_event = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.cuda_available and sync_event is not None:
            # this does not wait CPU side, so the log below should be immediate when pinned memory is used
            torch.cuda.current_stream().wait_event(sync_event)

        if self.debug:
            print(f"[{self.block_type}] Waited for block {block_idx}: {time.perf_counter() - start_time:.2f}s")


class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        use_pinned_memory: bool = False,
        debug: bool = False,
    ):
        super().__init__(block_type, num_blocks, blocks_to_swap, device, use_pinned_memory, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward  # forward only offloading: can be changed to True for inference

        if self.supports_backward:
            # register backward hooks
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        # switching must wait for all pending transfers
        for block_idx in list(self.futures.keys()):
            self._wait_blocks_move(block_idx)

        self.forward_only = forward_only

    def __del__(self):
        if self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(self, blocks: list[nn.Module], block_index: int) -> Optional[callable]:
        # -1 for 0-based index
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # create  hook
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward")

        for b in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weighs_to_device(b, self.device)  # make sure weights are on device

        cpu_device = torch.device("cpu")
        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            b.to(self.device)  # move block to device first. this makes sure that buffers (non weights) are on the device
            weighs_to_device(b, cpu_device)  # make sure weights are on cpu

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        # check if blocks_to_swap is enabled
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if not self.forward_only:
            # if backward is enabled, we do not swap blocks in forward pass more than blocks_to_swap, because it should be on GPU
            if block_idx >= self.blocks_to_swap:
                return
            block_idx_to_cpu = block_idx
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
            block_idx_to_cuda = block_idx_to_cuda % self.num_blocks  # this does nothing for backward offloading
            self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            return

        # We use two strategies here for forward-only offloading:
        # 1. If blocks_to_swap is less than half of num_blocks, we swap the num_blocks blocks without wrapping around.
        #   This reduces the number of swaps, so it is especially useful for small blocks_to_swap or lightweight models like Qwen-Image
        # 2. If blocks_to_swap is more than half of num_blocks, we swap the blocks with wrapping around.
        #   This is the common strategy used in most offloading implementations. It transfers all blocks in a wrapping manner.
        #   This is useful for large blocks_to_swap or heavyweight models like Wan/HunyuanVideo, where the transfer time is less significant compared to computation time.

        # current block to swap out (to CPU)
        block_idx_to_cpu = block_idx

        if self.blocks_to_swap < (self.num_blocks // 2):
            # strategy 1: no wrap around
            # If the current block is in the middle blocks that are not swapped, do nothing
            if self.blocks_to_swap <= block_idx < self.num_blocks - self.blocks_to_swap:
                return
            if block_idx < self.blocks_to_swap:
                # move the next block to cuda
                block_idx_to_cuda = (self.num_blocks - self.blocks_to_swap + block_idx) % self.num_blocks
            else:
                # move the previous block to cuda
                block_idx_to_cuda = block_idx - (self.num_blocks - self.blocks_to_swap)
        else:
            # strategy 2: with wrap around
            block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
            block_idx_to_cuda = block_idx_to_cuda % self.num_blocks  # this works for forward-only offloading

        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)


class _DirectCopier:
    """
    Transfer engine for ``LoRAStreamOffloader`` when the masters are *pinned*: copies a flat pinned host
    buffer straight into a flat device buffer on a private copy stream (async Host->Device only).

    This is the seam that isolates all CUDA stream/event primitives from the residency logic. The residency
    side only ever calls ``submit`` / ``wait``; ``_StagedCopier`` (pageable master + worker thread) and a
    future non-CUDA device shim plug in behind the same interface without the residency code changing. Keep
    every ``torch.cuda.*`` transfer primitive inside copiers like this one.
    """

    def __init__(self, device: torch.device, debug: bool = False):
        self.device = device
        self.debug = debug
        self.copy_stream = torch.cuda.Stream(device=device)
        self._events = {}  # key -> cuda.Event recording H2D completion
        self._xfer = {}  # key -> (ev_a, ev_b) transfer-timing events (debug only)

    def submit(self, key, dst_flat: torch.Tensor, src_flat: torch.Tensor, gate_event=None):
        """Begin the H2D copy of ``src_flat`` into ``dst_flat``, gated behind ``gate_event`` (compute done with dst)."""
        ev_a = ev_b = None
        with torch.cuda.stream(self.copy_stream):
            if gate_event is not None:
                self.copy_stream.wait_event(gate_event)  # do not overwrite a buffer still used by compute
            if self.debug:
                ev_a = torch.cuda.Event(enable_timing=True)
                ev_b = torch.cuda.Event(enable_timing=True)
                ev_a.record(self.copy_stream)
            dst_flat.copy_(src_flat, non_blocking=True)  # src is pinned -> async H2D
            if self.debug:
                ev_b.record(self.copy_stream)
                self._xfer[key] = (ev_a, ev_b)
        self._events[key] = self.copy_stream.record_event()

    def wait(self, key):
        """Return the H2D-completion event for ``key`` (None if nothing is in flight for it)."""
        return self._events.get(key)

    def pop_xfer_timing(self, key):
        """Pop the (start, end) transfer-timing events recorded for ``key`` (debug only)."""
        return self._xfer.pop(key, None)

    def sync(self):
        self.copy_stream.synchronize()

    def reset(self):
        self._events.clear()
        self._xfer.clear()


class _StagedCopier:
    """
    Transfer engine for ``LoRAStreamOffloader`` when the masters are kept in *pageable* host memory: each
    H2D is staged through a small pool of pinned buffers filled by a background worker thread. Same
    ``submit`` / ``wait`` interface as ``_DirectCopier``, so the residency code is identical.

    Why: pinning every streaming master (what ``_DirectCopier`` needs) can cost tens of GB of non-pageable
    "shared GPU memory" (notably on Windows). Here the masters stay pageable and only ``num_staging`` pinned
    buffers exist. A pageable->device copy is, inside the CUDA driver, already a synchronous serial
    pageable->pinned->device transfer; doing it ourselves on a worker thread lets the pageable->pinned
    memcpy overlap the previous block's pinned->device DMA and -- crucially -- keeps it off the main thread
    so compute kernel launches are never blocked. That is the whole speed difference versus the old
    synchronous pageable path this replaces.

    Hazards handled:
      - a staging buffer is reused only after the H2D that consumed it has completed (one consume-event per
        buffer; the worker waits on it before overwriting);
      - ``wait`` blocks the caller until the worker has actually enqueued the H2D and recorded its
        completion event, so the residency layer can ``current_stream().wait_event`` it as usual.
    """

    def __init__(self, device: torch.device, num_staging: int = 2, debug: bool = False):
        self.device = device
        self.num_staging = max(1, num_staging)
        self.debug = debug
        self.copy_stream = torch.cuda.Stream(device=device)
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._futures = {}  # key -> Future -> (load_event, (ev_a, ev_b))
        self._xfer = {}  # key -> (ev_a, ev_b) transfer-timing events (debug only)
        self._staging = None  # list[pinned uint8 tensor] (lazily sized to the flat block)
        self._staging_free = None  # list[cuda.Event | None]: the H2D that last consumed each buffer
        self._rr = 0  # round-robin staging index (touched only on the worker thread)

    def _ensure_staging(self, nbytes: int):
        if self._staging is None:
            self._staging = [torch.empty(nbytes, dtype=torch.uint8).pin_memory(device=self.device) for _ in range(self.num_staging)]
            self._staging_free = [None] * self.num_staging

    def _run(self, dst_flat: torch.Tensor, src_flat: torch.Tensor, gate_event):
        # runs on the worker thread: pageable->pinned memcpy, then enqueue the async pinned->device H2D
        idx = self._rr
        self._rr = (idx + 1) % self.num_staging
        staging = self._staging[idx]

        free_ev = self._staging_free[idx]
        if free_ev is not None:
            free_ev.synchronize()  # CPU(worker) waits until the prior H2D released this buffer -- off main thread

        staging.copy_(src_flat)  # pageable -> pinned, plain CPU memcpy

        ev_a = ev_b = None
        with torch.cuda.stream(self.copy_stream):
            if gate_event is not None:
                self.copy_stream.wait_event(gate_event)  # do not overwrite a dst still used by compute
            if self.debug:
                ev_a = torch.cuda.Event(enable_timing=True)
                ev_b = torch.cuda.Event(enable_timing=True)
                ev_a.record(self.copy_stream)
            dst_flat.copy_(staging, non_blocking=True)  # pinned -> device, async
            if self.debug:
                ev_b.record(self.copy_stream)
            done = self.copy_stream.record_event()  # H2D complete == this staging buffer is free again
        self._staging_free[idx] = done
        return done, (ev_a, ev_b)

    def submit(self, key, dst_flat: torch.Tensor, src_flat: torch.Tensor, gate_event=None):
        self._ensure_staging(src_flat.numel())  # flat uint8 -> numel == byte size
        self._futures[key] = self._pool.submit(self._run, dst_flat, src_flat, gate_event)

    def wait(self, key):
        fut = self._futures.get(key)
        if fut is None:
            return None
        load_event, xfer = fut.result()  # block until the worker enqueued the H2D (tiny if it ran ahead)
        if self.debug and xfer[0] is not None:
            self._xfer[key] = xfer
        return load_event

    def pop_xfer_timing(self, key):
        return self._xfer.pop(key, None)

    def _flush(self):
        for fut in list(self._futures.values()):
            fut.result()

    def sync(self):
        self._flush()
        self.copy_stream.synchronize()

    def reset(self):
        self._flush()
        self.copy_stream.synchronize()  # ensure all in-flight H2D are done before clearing staging state
        self._futures.clear()
        self._xfer.clear()
        if self._staging is not None:
            self._staging_free = [None] * self.num_staging

    def __del__(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)


class LoRAStreamOffloader:
    """
    H2D-only offloader for training where the base weights are frozen (e.g. LoRA / LoHa / LoKr).

    The classic block swap (``ModelOffloader``) *exchanges* a block: it copies one block back to the
    CPU (Device->Host) while copying the next one in (Host->Device). When the base weights never change,
    the CPU already holds an identical copy, so the D2H half is pure overhead. This offloader keeps a
    permanent (pinned) master copy of every streamed weight on the CPU and only ever copies Host->Device,
    removing the D2H transfer and the CPU-side staging memcpy entirely.

    Layout (N = num_blocks, S = blocks_to_swap, B = ring_size):
      - ``N - S`` blocks stay resident on the GPU permanently.
      - ``S`` "streaming" blocks are placed at even intervals; their ``Linear`` weights live on the CPU
        (pinned) as masters and are streamed into a small ring of ``B`` GPU buffers (default 2: one being
        computed on, one being prefetched). Non-Linear params / buffers of streaming blocks stay resident.

    Mechanics:
      - Because the streamed weights are frozen (no ``.grad`` to preserve), we swap the whole ``module.weight``
        *reference* between its CPU master Parameter and a preallocated GPU ring Parameter, instead of the
        ``.data`` storage trick used by ``ModelOffloader``. No per-step ``cudaMalloc``.
      - All swap weights of a block are packed into a single contiguous (byte) buffer -- one pinned CPU
        buffer per streaming block and one GPU buffer per ring slot -- and the individual weights are
        dtype/shape views into it. Loading a block is therefore a single H2D ``copy_`` instead of one
        per Linear, which removes the per-tensor launch/driver overhead on PCIe/CPU-bound systems.
      - Eviction is free: repoint ``module.weight`` back to the CPU master (no transfer). A ring slot is
        overwritten only when its block is consumed, so the last ``B`` streamed blocks of a pass stay
        resident. On the reverse pass they are exactly the first blocks needed, so the forward<->backward
        turn-around needs no reload.
      - Direction is implicit in the callback: ``submit_move_blocks_forward`` prefetches the block ``B``
        streaming-slots *ahead* (larger index); the backward hook prefetches ``B`` slots *behind* (smaller
        index). No explicit scheduler / no ``prepare_*_before_backward``.
      - ``wait_for_block`` self-heals: if a needed streaming block is not resident in its slot (cold start,
        or an inference pass boundary), it is loaded on demand.
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        ring_size: int = 2,
        use_pinned_memory: bool = True,
        debug: bool = False,
    ):
        self.block_type = block_type
        self._blocks = blocks
        self.num_blocks = num_blocks  # N
        self.blocks_to_swap = blocks_to_swap  # requested S
        self.device = device
        self.use_pinned_memory = use_pinned_memory
        self.supports_backward = supports_backward
        self.forward_only = not supports_backward

        import os

        if not debug:
            debug = os.getenv("MUSUBI_TUNER_OFFLOADER_DEBUG", "0") == "1"
        self.debug = debug
        self.debug_interval = int(os.getenv("MUSUBI_TUNER_OFFLOADER_DEBUG_INTERVAL", "10"))  # steps between prints

        assert device.type == "cuda", "LoRAStreamOffloader currently supports CUDA only"

        # ---- streaming placement: S evenly spaced block indices (midpoint formula -> distinct for S <= N) ----
        stream_idx = sorted({((2 * i + 1) * num_blocks) // (2 * blocks_to_swap) for i in range(blocks_to_swap)})
        self.stream_idx = stream_idx
        self.S = len(stream_idx)  # actual streaming count (>=1; dedup is a no-op unless S is close to N)
        self.rank = {b: k for k, b in enumerate(stream_idx)}  # block_idx -> position in stream_idx
        self.is_stream = [b in self.rank for b in range(num_blocks)]
        self.B = min(ring_size, self.S)  # clamp ring to number of streaming blocks

        # ---- finetuning guard: H2D-only must never lose weight updates ----
        for b in stream_idx:
            for m in self._swap_modules(blocks[b]):
                assert not m.weight.requires_grad, (
                    "LoRAStreamOffloader requires frozen base weights (LoRA / no full fine-tune). "
                    f"Found a trainable Linear weight in block {b}."
                )

        # ---- transfer engine: owns the copy stream and all H2D primitives ----
        # pinned masters -> direct async H2D; pageable masters -> stage through a worker thread + pinned pool
        # (low "shared GPU memory" footprint, e.g. on Windows). Both expose the same submit/wait interface.
        if use_pinned_memory:
            self.copier = _DirectCopier(device, debug=self.debug)
        else:
            self.copier = _StagedCopier(device, num_staging=self.B, debug=self.debug)

        # ---- runtime state (GPU buffers allocated lazily in prepare_block_devices_before_forward) ----
        self.cpu_master = {}  # block_idx -> [CPU (pinned) Parameter per swap weight] (views into cpu_flat)
        self.cpu_flat = {}  # block_idx -> flat (pinned) uint8 CPU tensor backing the masters
        self.ring_param = None  # [slot] -> [GPU nn.Parameter per swap weight] (views into ring_flat)
        self.ring_flat = None  # [slot] -> flat uint8 GPU tensor backing the ring params
        self._layout = None  # ([byte offset per swap weight], total bytes) shared by all streaming blocks
        self.in_slot = [None] * self.B  # slot -> block_idx currently bound to this slot (or None)
        self.free_event = [None] * self.B  # slot -> cuda.Event recording when compute finished using the slot
        self._module_cache = {}  # block_idx -> [modules with swap weights]

        self._wait_ctx = "fwd"  # context tag for wait/load timing ("fwd" while in the forward loop, "bwd" in hooks)
        if self.debug:
            self._dbg_reset()

        # ---- backward hooks: only where there is work (prefetch this block, or wait the previous one) ----
        if supports_backward:
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self._create_backward_hook(i)
                if hook is not None:
                    self.remove_handles.append(block.register_full_backward_hook(hook))

        print(
            f"LoRAStreamOffloader[{block_type}]: H2D-only block swap. "
            f"{self.S} streaming / {num_blocks} blocks, ring={self.B}, pinned={use_pinned_memory}. "
            f"streaming indices: {stream_idx}"
        )

    # ------------------------------------------------------------------ helpers

    def _swap_modules(self, block: nn.Module) -> list[nn.Module]:
        # same selection rule as ModelOffloader.swap_weight_devices_cuda: Linear layers with a weight
        mods = []
        for _, m in block.named_modules():
            if hasattr(m, "weight") and m.weight is not None and m.__class__.__name__.endswith("Linear"):
                mods.append(m)
        return mods

    def _modules(self, block_idx: int) -> list[nn.Module]:
        cached = self._module_cache.get(block_idx)
        if cached is None:
            cached = self._swap_modules(self._blocks[block_idx])
            self._module_cache[block_idx] = cached
        return cached

    def _bind(self, block_idx: int, params: list[nn.Parameter]):
        for m, p in zip(self._modules(block_idx), params):
            m.weight = p

    @staticmethod
    def _compute_layout(weights: list[torch.Tensor]) -> tuple[list[int], int]:
        """Byte offsets (aligned so ``view(dtype)`` is valid for any element size) and total size of the flat buffer."""
        align = 256
        offsets = []
        total = 0
        for w in weights:
            total = (total + align - 1) // align * align
            offsets.append(total)
            total += w.numel() * w.element_size()
        return offsets, total

    def _flat_views(self, flat: torch.Tensor, weights: list[torch.Tensor]) -> list[torch.Tensor]:
        """dtype/shape views into a flat uint8 buffer, one per swap weight, following ``self._layout``."""
        offsets, _ = self._layout
        return [flat[off : off + w.numel() * w.element_size()].view(w.dtype).view(w.shape) for off, w in zip(offsets, weights)]

    def _load(self, rank: int, slot: int, ctx: str = "fwd"):
        """Stream the streaming block at ``rank`` into ring ``slot`` (H2D only), repointing weights."""
        blk = self.stream_idx[rank]

        # already resident (e.g. ring survived a re-prepare): the slot content is still identical to the
        # immutable CPU master, so just rebind the weights to the ring -- no transfer needed
        if self.in_slot[slot] == blk:
            self._bind(blk, self.ring_param[slot])
            return

        # evict the slot's current owner: repoint its weights back to the CPU master (no transfer)
        prev = self.in_slot[slot]
        if prev is not None:
            self._bind(prev, self.cpu_master[prev])

        # single coalesced H2D copy of the whole block, gated behind the slot's free_event (compute done
        # with it). The ring params are views into ring_flat, so the copy bumps their autograd version
        # counter: safe under gradient checkpointing (weights are re-read at recompute time, and a slot is
        # only overwritten after its free_event); without checkpointing autograd raises a version error
        # instead of silently using stale-saved weights -- which the old per-tensor `.data.copy_` hid.
        self.copier.submit(blk, self.ring_flat[slot], self.cpu_flat[blk], self.free_event[slot])

        self._bind(blk, self.ring_param[slot])  # reference swap (no .data trick, no cudaMalloc)
        self.in_slot[slot] = blk

        if self.debug:
            c = self._cur[ctx]
            c["loads"] += 1
            xfer = self.copier.pop_xfer_timing(blk)
            if xfer is not None:
                c["xfer_ev"].append(xfer)

    # ------------------------------------------------------------------ public interface (mirrors ModelOffloader)

    def set_forward_only(self, forward_only: bool):
        self.copier.sync()
        self.forward_only = forward_only

    def __del__(self):
        if getattr(self, "supports_backward", False):
            for handle in getattr(self, "remove_handles", []):
                handle.remove()

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.S == 0:
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward (H2D-only)")

        first_time = not self.cpu_master  # device placement is only needed on the very first call
        cpu_device = torch.device("cpu")
        for i, b in enumerate(blocks):
            if not self.is_stream[i]:
                if first_time:
                    b.to(self.device)  # resident block: buffers + Linear weights on GPU permanently
                    weighs_to_device(b, self.device)
                continue

            if first_time:
                # move the whole block to device (buffers, norms, bias), then pull the swap weights back to
                # the CPU into one flat (pinned) buffer per block as the persistent masters
                b.to(self.device)
                mods = self._modules(i)
                weights = [m.weight.data for m in mods]
                if self._layout is None:
                    self._layout = self._compute_layout(weights)  # first streaming block defines the shared layout
                flat = torch.empty(self._layout[1], dtype=torch.uint8, device=cpu_device)
                if self.use_pinned_memory:
                    flat = flat.pin_memory(device=self.device)
                master = []
                for m, view in zip(mods, self._flat_views(flat, weights)):
                    view.copy_(m.weight.data)  # one-time D2H into the flat master
                    m.weight.data = view
                    master.append(m.weight)  # keep the original Parameter object as the persistent master
                self.cpu_flat[i] = flat
                self.cpu_master[i] = master
            else:
                # re-prepare (e.g. around in-training sampling): the non-swap parts are already on the device
                # and the CPU masters have stayed on the CPU throughout, so just repoint weights to the masters.
                # IMPORTANT: do NOT call b.to(device) here -- it would drag the CPU masters onto the GPU and
                # they would never be released (every swap block would end up resident).
                self._bind(i, self.cpu_master[i])

        # validate homogeneous streaming blocks (shared ring template)
        template = self.cpu_master[self.stream_idx[0]]
        for b in self.stream_idx[1:]:
            assert len(self.cpu_master[b]) == len(template), f"block {b} has a different number of swap weights"
            for p, t in zip(self.cpu_master[b], template):
                assert p.data.shape == t.data.shape and p.data.dtype == t.data.dtype, (
                    f"block {b} swap-weight shape/dtype differs from the streaming template"
                )

        # preallocate the GPU ring (once); copies happen into these flat buffers, never reallocated
        if self.ring_param is None:
            template_weights = [p.data for p in template]
            self.ring_flat = [torch.empty(self._layout[1], dtype=torch.uint8, device=self.device) for _ in range(self.B)]
            self.ring_param = [
                [nn.Parameter(view, requires_grad=False) for view in self._flat_views(flat, template_weights)]
                for flat in self.ring_flat
            ]

        # reset transfer state and preload the first B streaming blocks. in_slot deliberately survives:
        # after a completed backward pass the ring already holds ranks 0..B-1, so these loads become
        # no-op rebinds (see _load) instead of redundant cold-start transfers.
        self.free_event = [None] * self.B
        self.copier.reset()
        for k in range(self.B):
            self._load(k, k)

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.S == 0 or not self.is_stream[block_idx]:
            return
        ctx = self._wait_ctx
        if self.debug and ctx == "fwd" and block_idx == self.stream_idx[0]:
            self._dbg_boundary()  # a forward pass is starting -> finalize the previous step's stats
        j = self.rank[block_idx]
        slot = j % self.B
        if self.in_slot[slot] != block_idx:
            # self-healing: not resident (cold start / inference pass boundary / same-direction restart)
            if self.debug:
                self._cur[ctx]["self_heal"] += 1
            self._load(j, slot, ctx)
        ev = self.copier.wait(block_idx)
        if ev is None:
            return
        if not self.debug:
            torch.cuda.current_stream().wait_event(ev)  # compute waits for H2D (GPU-side, does not block CPU)
            return
        # debug: measure the actual GPU stall (idle time the compute stream spends on wait_event)
        s = torch.cuda.current_stream()
        c = self._cur[ctx]
        c["waits"] += 1
        if not ev.query():
            c["not_ready"] += 1  # H2D not finished when compute reached it -> a real stall (cheap, no sync)
        ev_a = torch.cuda.Event(enable_timing=True)
        ev_b = torch.cuda.Event(enable_timing=True)
        ev_a.record(s)
        s.wait_event(ev)
        ev_b.record(s)
        c["stall_ev"].append((ev_a, ev_b))
        # staged copier records its transfer-timing events on the worker thread, so they only become
        # available once wait() has joined it; the direct copier already popped them in _load (returns None)
        xfer = self.copier.pop_xfer_timing(block_idx)
        if xfer is not None:
            c["xfer_ev"].append(xfer)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        # consumed block_idx in the forward pass: free its slot and prefetch the block B streaming-slots ahead
        if self.S == 0 or not self.is_stream[block_idx]:
            return
        j = self.rank[block_idx]
        self.free_event[j % self.B] = torch.cuda.current_stream().record_event()
        if j + self.B < self.S:
            self._load(j + self.B, (j + self.B) % self.B, "fwd")  # same slot as rank j; evicts rank j
        elif self.forward_only and j == self.S - 1 and self.S > self.B:
            # inference: no backward pass will consume the retained tail, and the next forward pass starts
            # from rank 0 again. Wrap around: all tail slots are free once the last streaming block is
            # consumed, so preload the first B ranks now. The H2D overlaps with the remaining (resident)
            # blocks and the output layers, instead of stalling at the start of the next pass (self-heal).
            for k in range(self.B):
                self._load(k, k, "fwd")

    def _create_backward_hook(self, block_index: int):
        prefetch = self.is_stream[block_index]
        wait_prev = block_index - 1 >= 0 and self.is_stream[block_index - 1]
        if not prefetch and not wait_prev:
            return None

        def backward_hook(module, grad_input, grad_output):
            if prefetch:
                # consumed block_index in backward: free its slot and prefetch B streaming-slots behind
                j = self.rank[block_index]
                self.free_event[j % self.B] = torch.cuda.current_stream().record_event()
                if j - self.B >= 0:
                    self._load(j - self.B, (j - self.B) % self.B, "bwd")  # same slot as rank j; evicts rank j
            if wait_prev:
                self._wait_ctx = "bwd"
                self.wait_for_block(block_index - 1)  # ensure the next block to run is ready
                self._wait_ctx = "fwd"
            return None

        return backward_hook

    # ------------------------------------------------------------------ debug timing

    def _dbg_reset(self):
        self._dbg_empty = lambda: {"stall_ev": [], "xfer_ev": [], "waits": 0, "not_ready": 0, "self_heal": 0, "loads": 0}
        self._cur = {"fwd": self._dbg_empty(), "bwd": self._dbg_empty()}
        self._roll = {"fwd": defaultdict(float), "bwd": defaultdict(float)}
        self._step = 0

    def _dbg_boundary(self):
        # finalize the events accumulated for the step that just ended, then print every debug_interval steps
        if not any(self._cur[c]["waits"] or self._cur[c]["loads"] for c in ("fwd", "bwd")):
            return  # very first forward: nothing finished yet
        torch.cuda.synchronize()  # ensure timing events are complete (debug only)
        for ctx in ("fwd", "bwd"):
            c, r = self._cur[ctx], self._roll[ctx]
            r["stall_ms"] += sum(a.elapsed_time(b) for a, b in c["stall_ev"])
            r["xfer_ms"] += sum(a.elapsed_time(b) for a, b in c["xfer_ev"])
            for k in ("waits", "not_ready", "self_heal", "loads"):
                r[k] += c[k]
            self._cur[ctx] = self._dbg_empty()
        self._step += 1
        if self._step % self.debug_interval == 0:
            self._dbg_print()

    def _dbg_print(self):
        n = self.debug_interval
        print(f"[{self.block_type}] H2D-only timing (avg over {n} steps):")
        for ctx in ("fwd", "bwd"):
            r = self._roll[ctx]
            print(
                f"  {ctx}: stall {r['stall_ms'] / n:6.2f} ms/step, H2D {r['xfer_ms'] / n:6.2f} ms/step"
                f" | waits {r['waits'] / n:.1f}, not-ready {r['not_ready'] / n:.1f},"
                f" self-heal {r['self_heal'] / n:.1f}, loads {r['loads'] / n:.1f}"
            )
            self._roll[ctx] = defaultdict(float)
