"""Dual-GPU model-parallel path for Wan 2.2 LoRA training in musubi-tuner.

Activates when ``WAN_DUAL_GPU=true``. Distributes the vendored
:class:`~musubi_tuner.wan.modules.model.WanModel` transformer across two
CUDA devices with a single PCIe boundary mid-``blocks``, and routes
per-layer LoRA modules to the device of the transformer layer they wrap
(see the inline ``WAN_DUAL_GPU`` pin in ``networks/lora.py``). Default
behavior is unchanged when the env var is unset.

Companion to the validated FLUX.2 musubi port (``flux_2/flux2_dual_gpu.py``)
and the ai-toolkit Wan 2.2 port. The structural shape mirrors the FLUX.2
helper; the differences come from WanModel's API:

- ``WanModel.forward`` builds a single ``kwargs`` dict (``e``, ``seq_lens``,
  ``grid_sizes``, ``freqs``, ``context``, ``context_lens``) and reuses it
  for every ``block(x, **kwargs)`` call. We replace ``forward`` with a
  split-aware variant that crosses the PCIe boundary exactly once — both
  the activation ``x`` and the shared ``kwargs`` dict are bridged to
  cuda:1 at the split index, so every subsequent block sees cuda:1 inputs.
- ``WanModel.freqs`` is a plain tensor attribute, NOT a registered buffer
  (the class deliberately avoids ``register_buffer`` to keep its dtype
  stable across ``.to()``). ``module.to(device)`` therefore misses it —
  we move ``freqs`` (and the ``freqs_fhw`` cache) explicitly to cuda:0.
- ``move_to_device_except_swap_blocks`` is the single canonical place to
  intercept transformer-wide device placement, same as FLUX.2's
  ``Flux2.move_to_device_except_swap_blocks``.

A14B dual-expert support — handled at the swap site, see below.
Single-expert training (one ``--dit``, no ``--dit_high_noise``) and the
dual-expert path (``--dit`` + ``--dit_high_noise`` + ``--timestep_boundary``)
both work. The dual-expert case needs care: musubi swaps the full model
``state_dict`` between the high/low experts at ``swap_high_low_weights()``
via ``load_state_dict(..., assign=True)``. ``assign=True`` rebinds
parameter tensors, so a naive swap (a) collapses our cuda:0/cuda:1 split
back onto one device and (b) brings in the incoming expert's raw weights
straight out of ``load_wan_model`` — CPU-resident and with a per-module
dtype layout that diverges from the active model (which has been through
``accelerator.prepare`` + :func:`enable_wan_dual_gpu`). Left unhandled,
base modules end up with fp8 bias against bf16 input, raising
``RuntimeError: Input type (BFloat16) and bias type (Float8_e4m3fn)
should be the same`` on the first forward after the swap (validated
2026-05-14: an i2v-A14B run trained cleanly for 14/20 steps then crashed
the step the swap fired). ``WanNetworkTrainer.swap_high_low_weights``
fixes this for the ``WAN_DUAL_GPU`` path: after the ``assign=True`` load
it casts the incoming tensors back to the outgoing model's dtype layout
and re-invokes :func:`enable_wan_dual_gpu` (idempotent) to restore the
cuda:0/cuda:1 split and reinstall the split-aware forward.

Env vars:
    WAN_DUAL_GPU=true                      enable the dual-GPU path
    WAN_DUAL_GPU_SPLIT_AT=<int>            override blocks split index
                                           (default: num_blocks // 2)
"""
from __future__ import annotations

import logging
import os
import types
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from musubi_tuner.wan.modules.model import WanModel

logger = logging.getLogger(__name__)


def is_dual_gpu_enabled() -> bool:
    """True iff ``WAN_DUAL_GPU=true`` in the environment."""
    return os.getenv("WAN_DUAL_GPU", "false").lower() == "true"


def get_split_at(num_blocks: int) -> int:
    """Blocks split index. Override via ``WAN_DUAL_GPU_SPLIT_AT``."""
    override = os.getenv("WAN_DUAL_GPU_SPLIT_AT")
    if override is not None:
        return int(override)
    return num_blocks // 2


def enable_wan_dual_gpu(model: "WanModel") -> "WanModel":
    """Distribute the ``WanModel`` transformer across cuda:0 and cuda:1.

    Layout:
        cuda:0  — patch_embedding, text_embedding, time_embedding,
                  time_projection, img_emb (if present), blocks[:split_at],
                  head
        cuda:1  — blocks[split_at:]

    ``head`` stays on cuda:0 so the returned tensor (and the trainer's
    downstream loss ops, which run on the model's nominal device) match
    cuda:0. The split-aware forward installed by :func:`_install_split_forward`
    crosses the PCIe boundary once at ``blocks[split_at]`` and crosses back
    to cuda:0 before ``head``.

    Idempotent: safe to call again (e.g., after an A14B expert swap) — it
    re-applies the same placement and rebinds the same patched forward.

    Returns the (in-place modified) model.
    """
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"WAN_DUAL_GPU=true requires >=2 CUDA devices, found "
            f"{torch.cuda.device_count()}."
        )

    # Block-swap (CPU offload of idle blocks) and dual-GPU model-parallel are
    # mutually exclusive: the split forward below drops the offloader's
    # wait/submit calls. Fail loudly rather than silently ignoring --blocks_to_swap.
    if getattr(model, "blocks_to_swap", 0):
        raise RuntimeError(
            "WAN_DUAL_GPU and --blocks_to_swap are mutually exclusive: dual-GPU "
            "model-parallel already splits the transformer across two devices, "
            "while block-swap offloads idle blocks to CPU. Use one or the other."
        )

    num_blocks = len(model.blocks)
    split_at = get_split_at(num_blocks)
    if not 0 < split_at < num_blocks:
        raise RuntimeError(
            f"WAN_DUAL_GPU_SPLIT_AT={split_at} out of range "
            f"(model has {num_blocks} blocks)."
        )

    cuda0 = torch.device("cuda:0")
    cuda1 = torch.device("cuda:1")

    # Pre-block scaffolding + first half of blocks + output head land on
    # cuda:0; second half of blocks on cuda:1.
    model.patch_embedding.to(cuda0)
    model.text_embedding.to(cuda0)
    model.time_embedding.to(cuda0)
    model.time_projection.to(cuda0)
    if getattr(model, "img_emb", None) is not None:
        model.img_emb.to(cuda0)
    for block in model.blocks[:split_at]:
        block.to(cuda0)
    for block in model.blocks[split_at:]:
        block.to(cuda1)
    model.head.to(cuda0)

    # freqs is a plain tensor attribute, not a registered buffer (WanModel
    # deliberately avoids register_buffer to keep its dtype stable across
    # .to()) -- module.to(device) misses it, so move it explicitly. The
    # patchify / freq-prep step runs on cuda:0.
    if torch.is_tensor(model.freqs):
        model.freqs = model.freqs.to(cuda0)
    # freqs_fhw is a plain dict cache of per-grid-size freq tensors, also
    # not reached by module.to(). Move any cached entries to cuda:0.
    if isinstance(getattr(model, "freqs_fhw", None), dict):
        model.freqs_fhw = {
            k: (v.to(cuda0) if torch.is_tensor(v) else v)
            for k, v in model.freqs_fhw.items()
        }

    _install_split_forward(model, split_at)

    model._wan_dual_gpu_split_at = split_at  # type: ignore[attr-defined]
    logger.info(
        f"WAN_DUAL_GPU: distributed {num_blocks} blocks across cuda:0/cuda:1 "
        f"({split_at} on cuda:0, {num_blocks - split_at} on cuda:1); "
        f"scaffolding + head on cuda:0"
    )
    return model


def _install_split_forward(model: "WanModel", split_at: int) -> None:
    """Replace ``WanModel.forward`` with a split-aware variant.

    WanModel.forward builds one ``kwargs`` dict and reuses it for every
    ``block(x, **kwargs)`` call. We crosses the PCIe boundary exactly once
    at ``blocks[split_at]``: both the activation ``x`` and every tensor in
    the shared ``kwargs`` dict move to cuda:1, so all subsequent cuda:1
    blocks see cuda:1 inputs without per-block bridging. After the loop we
    move ``x`` back to its original input device for ``head`` + unpatchify.
    """
    from musubi_tuner.wan.modules.model import calculate_freqs_i, sinusoidal_embedding_1d
    from musubi_tuner.utils.device_utils import clean_memory_on_device

    cuda1 = torch.device("cuda:1")

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None, skip_block_indices=None, f_indices=None):
        # Mirror of WanModel.forward (see wan/modules/model.py) with a
        # single PCIe boundary inserted in the block loop. Kept in sync
        # structurally; only the block loop differs.
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            y = None

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        freqs_list = []
        for i, fhw in enumerate(grid_sizes):
            fhw = tuple(fhw.tolist())
            if f_indices is not None:
                fhw = tuple(list(fhw) + f_indices[i])
            if fhw not in self.freqs_fhw:
                c = self.dim // self.num_heads // 2
                self.freqs_fhw[fhw] = calculate_freqs_i(fhw, c, self.freqs, None if f_indices is None else f_indices[i])
            freqs_list.append(self.freqs_fhw[fhw])

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len, f"Sequence length exceeds maximum allowed length {seq_len}. Got {seq_lens.max()}"
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
            if self.model_version == "2.1" or self.force_v2_1_time_embedding:
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
                if self.model_version != "2.1":
                    e0 = e0.unsqueeze(1)
                    e = e.unsqueeze(1)
                    t = t.unsqueeze(1).expand(-1, seq_len)
            else:
                if t.dim() == 1:
                    t = t.unsqueeze(1).expand(-1, seq_len)
                bt = t.size(0)
                t = t.flatten()
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        if type(context) is list:
            context = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        context = self.text_embedding(context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)
            context = torch.concat([context_clip, context], dim=1)
            clip_fea = None
            context_clip = None

        # arguments -- a single dict reused for every block call
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs_list, context=context, context_lens=context_lens)

        if self.blocks_to_swap:
            clean_memory_on_device(device)

        input_device = x.device
        for block_idx, block in enumerate(self.blocks):
            is_block_skipped = skip_block_indices is not None and block_idx in skip_block_indices

            # Cross the PCIe boundary once at the split point. Bridge the
            # activation x and every tensor in the shared kwargs dict to
            # cuda:1; subsequent cuda:1 blocks reuse the bridged kwargs.
            if block_idx == split_at:
                x = x.to(cuda1)
                kwargs = _bridge_block_kwargs(kwargs, cuda1)

            if not is_block_skipped:
                x = block(x, **kwargs)

        if x.device != input_device:
            x = x.to(input_device)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    model.forward = types.MethodType(forward, model)


def _bridge_block_kwargs(kwargs: dict, device: torch.device) -> dict:
    """Move every tensor in the WanAttentionBlock kwargs dict to ``device``.

    ``freqs`` is a list of per-sample tensors; ``e`` / ``seq_lens`` /
    ``grid_sizes`` / ``context`` are tensors; ``context_lens`` is ``None``.
    ``Tensor.to`` is an identity no-op when the tensor is already on
    ``device``.
    """
    bridged = {}
    for key, value in kwargs.items():
        if torch.is_tensor(value):
            bridged[key] = value.to(device)
        elif isinstance(value, (list, tuple)):
            bridged[key] = type(value)(
                v.to(device) if torch.is_tensor(v) else v for v in value
            )
        else:
            bridged[key] = value
    return bridged
