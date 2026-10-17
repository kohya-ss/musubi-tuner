from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import sys
from threading import Event
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musubi_tuner.modules.custom_offloading_utils import ModelOffloader


def test_prepare_waits_for_pending_transfer_before_placing_blocks():
    blocks = [nn.Linear(4, 4, bias=False) for _ in range(2)]
    offloader = ModelOffloader("test", blocks, 2, 1, False, torch.device("cpu"))
    release = Event()
    boundary = Event()
    placements = []

    def transfer():
        assert release.wait(10), "test did not release the transfer"
        return 0, 1, None

    future = offloader.thread_pool.submit(transfer)
    offloader.futures[1] = future
    original_result = future.result

    def wait_for_transfer(*args, **kwargs):
        boundary.set()
        return original_result(*args, **kwargs)

    def place(*args, **kwargs):
        placements.append(future.done())
        boundary.set()

    try:
        with patch.object(future, "result", side_effect=wait_for_transfer), patch.object(nn.Module, "to", side_effect=place):
            with ThreadPoolExecutor(max_workers=1) as executor:
                prepare = executor.submit(offloader.prepare_block_devices_before_forward, blocks)
                try:
                    assert boundary.wait(10), "prepare did not reach the transfer or placement boundary"
                    assert placements == [], "placement started before the pending transfer was joined"
                    assert not prepare.done()
                finally:
                    release.set()
                    prepare.result(timeout=10)
        assert placements == [True, True]
        assert offloader.futures == {}
    finally:
        release.set()
        offloader.thread_pool.shutdown(wait=True)


def test_prepare_propagates_transfer_failure_before_placing_blocks():
    blocks = [nn.Linear(4, 4) for _ in range(2)]
    offloader = ModelOffloader("test", blocks, 2, 1, False, torch.device("cpu"))
    failed = Future()
    failed.set_exception(RuntimeError("transfer failed"))
    offloader.futures[1] = failed
    try:
        with patch.object(nn.Module, "to") as place:
            with pytest.raises(RuntimeError, match="transfer failed"):
                offloader.prepare_block_devices_before_forward(blocks)
            place.assert_not_called()
    finally:
        offloader.thread_pool.shutdown(wait=True)


@pytest.mark.parametrize("blocks_to_swap", [None, 0])
def test_prepare_with_swapping_disabled_is_a_noop(blocks_to_swap):
    blocks = [nn.Linear(4, 4)]
    offloader = ModelOffloader("test", blocks, 1, blocks_to_swap, False, torch.device("cpu"))
    try:
        with patch.object(nn.Module, "to") as place, patch.object(offloader, "_wait_blocks_move") as wait:
            offloader.prepare_block_devices_before_forward(blocks)
            place.assert_not_called()
            wait.assert_not_called()
    finally:
        offloader.thread_pool.shutdown(wait=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("use_pinned_memory", [False, True])
@pytest.mark.parametrize("blocks_to_swap", [1, 3])
def test_cuda_reprepare_preserves_weights_and_outputs(use_pinned_memory, blocks_to_swap):
    device = torch.device("cuda:0")
    generator = torch.Generator().manual_seed(123)
    blocks = [nn.Linear(32, 32, bias=False) for _ in range(4)]
    for block in blocks:
        block.weight.data.copy_(torch.randn(32, 32, generator=generator) / 32)
    weights = [block.weight.detach().clone() for block in blocks]
    x = torch.randn(2, 32, generator=generator).to(device)
    reference = x
    for weight in weights:
        reference = torch.nn.functional.linear(reference, weight.to(device))
    offloader = ModelOffloader("test", blocks, 4, blocks_to_swap, False, device, use_pinned_memory)
    try:
        with torch.no_grad():
            for _ in range(5):
                offloader.prepare_block_devices_before_forward(blocks)
                assert offloader.futures == {}
                for index, (block, weight) in enumerate(zip(blocks, weights)):
                    expected_device = device if index < 4 - blocks_to_swap else torch.device("cpu")
                    assert block.weight.device == expected_device
                    torch.testing.assert_close(block.weight.cpu(), weight, rtol=0, atol=0)
                output = x
                for index, block in enumerate(blocks):
                    offloader.wait_for_block(index)
                    output = block(output)
                    offloader.submit_move_blocks_forward(blocks, index)
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                # The wrap-around preloads are deliberately left for the next prepare.
                assert offloader.futures
    finally:
        offloader.thread_pool.shutdown(wait=True)
        torch.cuda.synchronize(device)
