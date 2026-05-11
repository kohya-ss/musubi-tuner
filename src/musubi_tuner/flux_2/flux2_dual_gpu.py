"""Dual-GPU model-parallel path for FLUX.2 LoRA training in musubi-tuner.

Activates when ``FLUX2_DUAL_GPU=true``. Distributes the FLUX.2 transformer
across two CUDA devices with a single PCIe boundary mid-``single_blocks``,
and routes per-layer LoRA modules to the device of the transformer layer
they wrap. Default behavior is unchanged when the env var is unset.

Companion to the validated ai-toolkit patch
(https://github.com/genno-whittlery/flux2-dual-gpu-lora). Musubi-tuner's
architecture is structurally cleaner for porting:

- Text encoders (Mistral 3 / Qwen-3) are pre-cached to disk via the
  separate ``flux_2_cache_text_encoder_outputs.py`` script, so the inner
  training loop never touches them. No TE-on-CPU runtime patch needed.
- ``LoRAModule.apply_to`` is a 3-line monkey-patch over ``forward``; a
  one-line snapshot of the wrapped layer's device plus a lazy forward
  shim is sufficient for per-LoRA device routing.
- ``Flux2.move_to_device_except_swap_blocks`` is the single canonical
  place to intercept transformer-wide device placement.

Env vars:
    FLUX2_DUAL_GPU=true                    enable the dual-GPU path
    FLUX2_DUAL_GPU_SPLIT_AT=24             override single_blocks split
                                           index (default: n_single // 2)
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from musubi_tuner.flux_2.flux2_models import Flux2
    from musubi_tuner.networks.lora import LoRANetwork, LoRAModule


def is_dual_gpu_enabled() -> bool:
    """True iff ``FLUX2_DUAL_GPU=true`` in the environment."""
    return os.getenv("FLUX2_DUAL_GPU", "false").lower() == "true"


def get_split_at(num_single_blocks: int) -> int:
    """Single-blocks split index. Override via ``FLUX2_DUAL_GPU_SPLIT_AT``."""
    override = os.getenv("FLUX2_DUAL_GPU_SPLIT_AT")
    if override is not None:
        return int(override)
    return num_single_blocks // 2


def assert_no_block_swap_in_dual_gpu(blocks_to_swap: Optional[int]) -> None:
    """Block-swap and dual-GPU are mutually exclusive.

    Block-swap pages whole blocks RAM↔single-GPU; dual-GPU splits blocks
    across two GPUs statically. The two strategies make different
    assumptions about block residence and don't compose.
    """
    if not is_dual_gpu_enabled():
        return
    if blocks_to_swap and blocks_to_swap > 0:
        raise RuntimeError(
            "FLUX2_DUAL_GPU=true is incompatible with --blocks_to_swap. "
            "The dual-GPU split already distributes blocks across two GPUs; "
            "block-swap pages blocks to system RAM, which fights the split. "
            "Disable --blocks_to_swap or unset FLUX2_DUAL_GPU."
        )


def distribute_flux2_transformer(
    model: "Flux2", dtype: Optional[torch.dtype] = None
) -> None:
    """Distribute the FLUX.2 transformer across cuda:0 and cuda:1.

    Layout:
        cuda:0  — img_in, txt_in, time_in, pe_embedder, guidance_in,
                  *_modulation*, all double_blocks, single_blocks[0:split_at]
        cuda:1  — single_blocks[split_at:], final_layer

    The forward path is updated by :func:`install_split_forward` to insert
    one ``.to(cuda:1)`` boundary at the split point and ``.to(cuda:0)``
    on the way back to ``final_layer`` (we keep final_layer on cuda:0 so
    the returned tensor matches ``vec.device`` for the caller).

    ``dtype`` is optional and defaults to None (preserve current dtype).
    Passing an explicit dtype will trigger conversion during the move,
    which can balloon memory if the model is already quantized
    (e.g., fp8 → bf16 conversion ~2× the per-block footprint).
    """
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"FLUX2_DUAL_GPU=true requires ≥2 CUDA devices, found "
            f"{torch.cuda.device_count()}."
        )

    split_at = get_split_at(model.num_single_blocks)
    if not 0 < split_at < model.num_single_blocks:
        raise RuntimeError(
            f"FLUX2_DUAL_GPU_SPLIT_AT={split_at} out of range "
            f"(model has {model.num_single_blocks} single blocks)."
        )

    cuda0 = torch.device("cuda:0")
    cuda1 = torch.device("cuda:1")

    def _move(mod, device):
        # Preserve existing dtype unless caller explicitly asked for one.
        # Passing dtype when the model is already quantized (fp8) would
        # expand weights back to bf16 during the move and balloon memory.
        if dtype is None:
            mod.to(device)
        else:
            mod.to(device, dtype=dtype)

    # Pre-blocks scaffolding + all double_blocks + first half single_blocks
    # land on cuda:0. Final layer also lives on cuda:0 (see forward).
    _move(model.img_in, cuda0)
    _move(model.txt_in, cuda0)
    _move(model.time_in, cuda0)
    model.pe_embedder.to(cuda0)
    if model.use_guidance_embed and not isinstance(model.guidance_in, nn.Identity):
        _move(model.guidance_in, cuda0)
    _move(model.double_stream_modulation_img, cuda0)
    _move(model.double_stream_modulation_txt, cuda0)
    _move(model.single_stream_modulation, cuda0)
    for block in model.double_blocks:
        _move(block, cuda0)
    for block in model.single_blocks[:split_at]:
        _move(block, cuda0)
    for block in model.single_blocks[split_at:]:
        _move(block, cuda1)
    # final_layer stays on cuda:0 — the forward moves img back before
    # calling it, and final_layer's output matches vec.device (cuda:0).
    _move(model.final_layer, cuda0)

    # Stash the split index for the patched forward.
    model._flux2_dual_gpu_split_at = split_at  # type: ignore[attr-defined]


def install_split_forward(model: "Flux2") -> None:
    """Replace ``Flux2.forward`` with a split-aware version.

    Inserts one ``.to(cuda:1)`` boundary at ``single_blocks[split_at]``
    and ``.to(cuda:0)`` after the loop so ``final_layer`` (and the
    returned tensor) live on cuda:0 — matching ``vec.device`` for the
    rest of the training step.

    Cross-PCIe traffic per forward: one activation tensor
    (~18 MB at 512², bf16) one way, plus the same on the way back
    during backward. At PCIe Gen5 x16 (~50 GB/s practical), this is
    ~1 ms — well below the per-step compute cost.
    """
    from einops import rearrange  # noqa: F401  (imported in flux2_models)
    from musubi_tuner.flux_2.flux2_models import (  # local to avoid cycles
        AttentionParams,
        timestep_embedding,
    )

    split_at: int = model._flux2_dual_gpu_split_at  # type: ignore[attr-defined]
    cuda1 = torch.device("cuda:1")
    cuda0 = torch.device("cuda:0")

    def forward(
        self,
        x: torch.Tensor,
        x_ids: torch.Tensor,
        timesteps: torch.Tensor,
        ctx: torch.Tensor,
        ctx_ids: torch.Tensor,
        guidance: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # accelerator.prepare(model, ...) under mixed_precision=bf16 wraps
        # model.forward in an autocast(bf16) context. We overwrite forward
        # below via types.MethodType, which shadows that wrapper — so we
        # have to re-enter autocast here ourselves. Autocast contexts nest
        # cleanly, so this is a no-op when one is already active outside.
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            num_txt_tokens = ctx.shape[1]

            timestep_emb = timestep_embedding(timesteps, 256)
            del timesteps
            vec = self.time_in(timestep_emb)
            del timestep_emb
            if self.use_guidance_embed:
                guidance_emb = timestep_embedding(guidance, 256)
                vec = vec + self.guidance_in(guidance_emb)
                del guidance_emb

            double_block_mod_img = self.double_stream_modulation_img(vec)
            double_block_mod_txt = self.double_stream_modulation_txt(vec)
            single_block_mod, _ = self.single_stream_modulation(vec)

            img = self.img_in(x)
            del x
            txt = self.txt_in(ctx)
            del ctx
            pe_x = self.pe_embedder(x_ids)
            del x_ids
            pe_ctx = self.pe_embedder(ctx_ids)
            del ctx_ids

            attn_params = AttentionParams.create_attention_params(
                self.attn_mode, self.split_attn
            )

            for block in self.double_blocks:
                img, txt = block(
                    img,
                    txt,
                    pe_x,
                    pe_ctx,
                    double_block_mod_img,
                    double_block_mod_txt,
                    attn_params,
                )

            del double_block_mod_img, double_block_mod_txt

            img = torch.cat((txt, img), dim=1)
            del txt
            pe = torch.cat((pe_ctx, pe_x), dim=2)
            del pe_ctx, pe_x

            for block_idx, block in enumerate(self.single_blocks):
                # Cross the PCIe boundary at the split point. Note that
                # single_block_mod is a 2-tuple (see SingleStreamBlock.forward
                # in flux2_models.py: `mod: tuple[Tensor, Tensor]`), so we
                # have to map .to() across both elements.
                if block_idx == split_at:
                    img = img.to(cuda1)
                    pe = pe.to(cuda1)
                    single_block_mod = tuple(t.to(cuda1) for t in single_block_mod)
                img = block(img, pe, single_block_mod, attn_params)

            del single_block_mod, pe

            # Move the result back to cuda:0 for final_layer + caller.
            img = img.to(cuda0)
            img = img[:, num_txt_tokens:, ...]
            img = self.final_layer(img, vec)
            return img

    # Bind as a method on the instance.
    import types
    model.forward = types.MethodType(forward, model)


def _get_wrapped_device(lora: "LoRAModule") -> Optional[torch.device]:
    """Read the wrapped module's CURRENT device.

    LoRAModule.apply_to stores ``self.org_forward = self.org_module.forward``
    — a bound method whose ``__self__`` is the wrapped layer. Reading the
    device through that reference each time gives us the *current* placement
    even after the model has been re-distributed (e.g., dual-GPU split).

    Falls back to the static ``_wrapped_device`` snapshot if the bound-method
    trick fails (e.g., user installed a callable that isn't a bound method).
    """
    bound = getattr(lora, "org_forward", None)
    wrapped_module = getattr(bound, "__self__", None)
    if wrapped_module is not None:
        try:
            return next(wrapped_module.parameters()).device
        except StopIteration:
            pass
    return getattr(lora, "_wrapped_device", None)


def route_loras_to_wrapped_devices(network: "LoRANetwork") -> None:
    """Place each LoRA module on the device of the layer it wraps.

    Reads the wrapped device DYNAMICALLY via the org_forward bound method —
    safe to call at any point in the training-setup lifecycle, including
    after the dual-GPU distribute has moved blocks to cuda:1.

    Also installs a per-LoRA forward shim as a safety net: if anything
    moves the LoRA off the wrapped layer's device after this point
    (e.g., ``accelerator.prepare(network)`` flattening everything onto
    cuda:0), the shim silently re-places it on the next forward.
    """
    loras = list(network.text_encoder_loras) + list(network.unet_loras)
    for lora in loras:
        wrapped_device = _get_wrapped_device(lora)
        if wrapped_device is None:
            continue
        lora.to(wrapped_device)
        _install_lora_forward_pin(lora)


def _install_lora_forward_pin(lora: "LoRAModule") -> None:
    """Wrap ``lora.forward`` to re-pin params to the wrapped layer's
    *current* device on each call.

    Looking up the device dynamically (vs. capturing it in the closure)
    is what makes this robust to a distribute-after-apply_to ordering:
    even if route_loras_to_wrapped_devices ran when the wrapped layer
    was on CPU, the next forward call will see it on cuda:1 (say) and
    move the LoRA accordingly.
    """
    orig_forward = lora.forward

    def pinned_forward(x: torch.Tensor) -> torch.Tensor:
        target = _get_wrapped_device(lora)
        if target is not None:
            try:
                current = next(lora.parameters()).device
            except StopIteration:
                current = target
            if current != target:
                lora.to(target)
        return orig_forward(x)

    lora.forward = pinned_forward  # type: ignore[method-assign]
