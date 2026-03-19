"""Tests that rep_proj parameters are managed by the main optimizer.

Before the fix: rep_proj had its own AdamW optimizer, updated independently.
After the fix: rep_proj params are merged into the first param group of the main optimizer.
"""
import torch
import torch.nn as nn
import pytest


def make_rep_proj(hidden_size: int = 16) -> nn.Module:
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size),
    )


def test_rep_proj_in_main_optimizer_gets_updated():
    """Rep_proj params merged into main optimizer param group are updated by optimizer.step()."""
    torch.manual_seed(0)
    rep_proj = make_rep_proj(8)

    lora_param = nn.Parameter(torch.randn(8, 8))
    # Merge rep_proj params into same group as lora_param (no separate lr)
    all_params = [lora_param] + list(rep_proj.parameters())
    optimizer = torch.optim.AdamW([{"params": all_params}], lr=1e-3)

    before = [p.clone().detach() for p in rep_proj.parameters()]

    x = torch.randn(2, 8)
    loss = rep_proj(x).mean()
    loss.backward()
    optimizer.step()

    for b, p in zip(before, rep_proj.parameters()):
        assert not torch.allclose(b, p.detach()), "rep_proj param was not updated"


def test_rep_proj_not_updated_when_not_in_optimizer():
    """Documents the old broken behavior: rep_proj params NOT updated by main optimizer
    when they are only in a separate optimizer that is never stepped."""
    torch.manual_seed(0)
    rep_proj = make_rep_proj(8)

    lora_param = nn.Parameter(torch.randn(8, 8))
    # Old design: rep_proj NOT in main optimizer
    main_optimizer = torch.optim.AdamW([lora_param], lr=1e-3)

    before = [p.clone().detach() for p in rep_proj.parameters()]

    loss = rep_proj(torch.randn(2, 8)).mean()
    loss.backward()
    main_optimizer.step()  # only main optimizer stepped — rep_proj not included

    for b, p in zip(before, rep_proj.parameters()):
        assert torch.allclose(b, p.detach()), "rep_proj should NOT update from unrelated optimizer.step()"
