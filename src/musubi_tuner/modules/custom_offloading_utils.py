from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
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


# Use pinned memory for faster transfer between CPU and GPU, but it requires more memory. Keep these functions here for portability
def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    # start_time = time.perf_counter()

    weight_swap_jobs = []

    # This is not working for all cases (e.g. SD3), so we need to find the corresponding modules
    # for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
    #     print(module_to_cpu.__class__, module_to_cuda.__class__)
    #     if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
    #         weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))
            else:
                if module_to_cuda.weight.data.device.type != device.type:
                    # print(
                    #     f"Module {module_to_cuda_name} not found in CPU model or shape mismatch, so not swapping and moving to device"
                    # )
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # cuda to cpu
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        stream.synchronize()

        # cpu to cuda
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

    stream.synchronize()
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    # print(f"Swapped weights in {time.perf_counter() - start_time:.2f}s")


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
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data = module.weight.data.to(device, non_blocking=device.type != "cpu")


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

        # Staging buffers for cuda offloading. These are pinned memory buffers to speed up the transfer between CPU and GPU
        # We create one staging buffer per transfer direction (A: GPU to CPU, B: CPU to GPU)
        self.staging_buffer_a = None
        self.staging_buffer_b = None

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
                if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
                    module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
                    if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                        weight_swap_jobs.append(
                            (module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data)
                        )
                    else:
                        if module_to_cuda.weight.data.device.type != device.type:
                            module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

        with T.section("synchronize before swap"):
            torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

        if not self.use_pinned_memory:
            # Minimize using pinned memory for lower shared GPU RAM usage
            stream = torch.cuda.Stream()
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
                            event_b.synchronize()

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

                with T.section("wait staging B"):
                    event_b.synchronize()  # Ensure the last staging buffer B copy is done

            with T.section("synchronize"):
                stream.synchronize()  # Synchronize staging buffer B to CUDA
                torch.cuda.current_stream().synchronize()  # This prevents the illegal loss value

            for sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
            ):
                # Update references
                module_to_cuda.weight.data = cuda_data_view
                module_to_cpu.weight.data = cpu_data_view

        else:
            # Use pinned memory for faster transfer between CPU and GPU, but it requires more memory
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                events_a = [torch.cuda.Event() for _ in weight_swap_jobs]  # Waiting events for GPU to CPU non-blocking copy

                # Copy weights to CPU
                for event, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(events_a, weight_swap_jobs):
                    # CUDA to CPU, non-blocking copy
                    with T.section("cuda to cpu"):
                        # event.record(stream)  # slower
                        stream.record_event(event)  # faster
                        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True).pin_memory(device=device)

                # CPU to CUDA
                for event, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(events_a, weight_swap_jobs):
                    # Wait for CPU data to be ready
                    with T.section("wait cpu"):
                        # event.synchronize() # slower
                        stream.wait_event(event)  # faster

                    # CPU to CUDA, non-blocking copy
                    with T.section("cpu to cuda"):
                        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
                        module_to_cuda.weight.data = cuda_data_view

            with T.section("synchronize"):
                stream.synchronize()
                torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

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

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            self.swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(
                    f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}"
                )

            self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(
                    f"[{self.block_type}] Moved blocks {bidx_to_cpu} to CPU and {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'} in {time.perf_counter() - start_time:.2f}s"
                )
            return bidx_to_cpu, bidx_to_cuda  # , event

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
        _, bidx_to_cuda = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

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

        # if backward is enabled, we do not swap blocks in forward pass more than blocks_to_swap, because it should be on GPU
        if not self.forward_only and block_idx >= self.blocks_to_swap:
            return

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        block_idx_to_cuda = block_idx_to_cuda % self.num_blocks  # this works for forward-only offloading
        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
