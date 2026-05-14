"""Dual-GPU model-parallel path for HunyuanVideo LoRA training in musubi-tuner.

Activates when ``HV_DUAL_GPU=true``. Distributes the
:class:`~musubi_tuner.hunyuan_model.models.HYVideoDiffusionTransformer` across
two CUDA devices with a single PCIe boundary mid-``single_blocks``, and routes
per-layer LoRA modules to the device of the transformer layer they wrap (see
the inline ``HV_DUAL_GPU`` pin in ``networks/lora.py``). Default behavior is
unchanged when the env var is unset.

Companion to the validated FLUX.2 and Wan 2.2 musubi ports
(``flux_2/flux2_dual_gpu.py``, ``wan/wan_dual_gpu.py``). HunyuanVideo's
double-stream + single-stream architecture is structurally closer to FLUX.2
than to Wan, so the split shape mirrors the FLUX.2 helper:

- ``HYVideoDiffusionTransformer.forward`` runs all ``double_blocks`` first,
  then ``x = torch.cat((img, txt), 1)``, then all ``single_blocks``. We
  replace ``forward`` with a split-aware variant that crosses the PCIe
  boundary exactly once, mid-``single_blocks`` at ``split_at``. All
  ``double_blocks`` run on cuda:0; the boundary bridges the merged ``x``
  plus the per-block tensor args to cuda:1.
- Unlike Wan's ``WanModel.forward`` (one reused ``kwargs`` dict),
  HunyuanVideo's ``forward`` rebuilds a fresh positional ARG LIST per block
  (``double_block_args`` / ``single_block_args``, ~10 entries each). So the
  bridge helper recurses over a list/tuple, not a dict.
- RoPE (``freqs_cos`` / ``freqs_sin`` -> ``freqs_cis`` tuple) is computed
  outside the model and passed as a forward arg — there is no plain-tensor
  attribute on the model to move (contrast WanModel.freqs).
- The two text encoders (llava_llama3 + clip_l) are pre-encoded, cached to
  disk, and deleted before the training loop — the inner loop never touches
  them, so no TE-on-CPU runtime patch is needed.
- ``HYVideoDiffusionTransformer.move_to_device_except_swap_blocks`` is the
  single canonical place to intercept transformer-wide device placement,
  same as FLUX.2 / Wan.

Env vars:
    HV_DUAL_GPU=true                       enable the dual-GPU path
    HV_DUAL_GPU_SPLIT_AT=<int>             override single_blocks split index
                                           (default: num_single_blocks // 2)
"""
from __future__ import annotations

import logging
import os
import types
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from musubi_tuner.hunyuan_model.models import HYVideoDiffusionTransformer

logger = logging.getLogger(__name__)


def is_dual_gpu_enabled() -> bool:
    """True iff ``HV_DUAL_GPU=true`` in the environment."""
    return os.getenv("HV_DUAL_GPU", "false").lower() == "true"


def get_split_at(num_single_blocks: int) -> int:
    """Single-blocks split index. Override via ``HV_DUAL_GPU_SPLIT_AT``."""
    override = os.getenv("HV_DUAL_GPU_SPLIT_AT")
    if override is not None:
        return int(override)
    return num_single_blocks // 2


def enable_hv_dual_gpu(model: "HYVideoDiffusionTransformer") -> "HYVideoDiffusionTransformer":
    """Distribute the HunyuanVideo transformer across cuda:0 and cuda:1.

    Layout:
        cuda:0  — img_in, txt_in, time_in, vector_in, guidance_in (if present),
                  ALL double_blocks, single_blocks[:split_at]
        cuda:1  — single_blocks[split_at:], final_layer

    The split-aware forward installed by :func:`_install_split_forward`
    crosses the PCIe boundary once mid-``single_blocks`` at ``split_at``, and
    crosses back to the input device before ``final_layer`` so the returned
    tensor matches the caller's device.

    Idempotent: safe to call again — it re-applies the same placement and
    rebinds the same patched forward.

    Returns the (in-place modified) model.
    """
    if torch.cuda.device_count() < 2:
        raise RuntimeError(
            f"HV_DUAL_GPU=true requires >=2 CUDA devices, found "
            f"{torch.cuda.device_count()}."
        )

    # Block-swap (CPU offload of idle blocks) and dual-GPU model-parallel are
    # mutually exclusive: the split forward below drops the offloader's
    # wait/submit calls. Fail loudly rather than silently ignoring --blocks_to_swap.
    if getattr(model, "blocks_to_swap", None):
        raise RuntimeError(
            "HV_DUAL_GPU and --blocks_to_swap are mutually exclusive: dual-GPU "
            "model-parallel already splits the transformer across two devices, "
            "while block-swap offloads idle blocks to CPU. Use one or the other."
        )

    num_double_blocks = len(model.double_blocks)
    num_single_blocks = len(model.single_blocks)
    split_at = get_split_at(num_single_blocks)
    if not 0 < split_at < num_single_blocks:
        raise RuntimeError(
            f"HV_DUAL_GPU_SPLIT_AT={split_at} out of range "
            f"(model has {num_single_blocks} single blocks)."
        )

    cuda0 = torch.device("cuda:0")
    cuda1 = torch.device("cuda:1")

    # Pre-block scaffolding + all double_blocks + first half of single_blocks
    # land on cuda:0; second half of single_blocks + final_layer on cuda:1.
    model.img_in.to(cuda0)
    model.txt_in.to(cuda0)
    model.time_in.to(cuda0)
    model.vector_in.to(cuda0)
    if getattr(model, "guidance_in", None) is not None:
        model.guidance_in.to(cuda0)
    for block in model.double_blocks:
        block.to(cuda0)
    for block in model.single_blocks[:split_at]:
        block.to(cuda0)
    for block in model.single_blocks[split_at:]:
        block.to(cuda1)
    # final_layer stays on cuda:1 — the forward bridges the result back to
    # the input device after the loop, before final_layer runs.
    model.final_layer.to(cuda1)

    _install_split_forward(model, split_at)

    model._hv_dual_gpu_split_at = split_at  # type: ignore[attr-defined]
    logger.info(
        f"HV_DUAL_GPU: distributed {num_double_blocks} double + "
        f"{num_single_blocks} single blocks across cuda:0/cuda:1 "
        f"(cuda:0: {num_double_blocks} double + {split_at} single; "
        f"cuda:1: {num_single_blocks - split_at} single + final_layer)"
    )
    return model


def _install_split_forward(model: "HYVideoDiffusionTransformer", split_at: int) -> None:
    """Replace ``HYVideoDiffusionTransformer.forward`` with a split-aware variant.

    Mirror of ``HYVideoDiffusionTransformer.forward`` (see
    ``hunyuan_model/models.py``) with a single PCIe boundary inserted in the
    ``single_blocks`` loop. All ``double_blocks`` run on cuda:0; at
    ``single_blocks[split_at]`` the merged activation ``x`` and every tensor
    in the per-block arg list move to cuda:1, so all subsequent cuda:1 blocks
    see cuda:1 inputs. After the loop the result moves back to the input
    device for ``final_layer`` + unpatchify.

    The block-swap (``offloader_*``) branches are deliberately dropped — they
    are mutually exclusive with dual-GPU (guarded in :func:`enable_hv_dual_gpu`).
    """
    from musubi_tuner.hunyuan_model.attention import get_cu_seqlens
    from musubi_tuner.utils.device_utils import clean_memory_on_device, synchronize_device

    cuda1 = torch.device("cuda:1")

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        text_states_2=None,
        freqs_cos=None,
        freqs_sin=None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
    ):
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        if self._enable_img_in_txt_in_offloading:
            self.img_in.to(x.device, non_blocking=True)
            self.txt_in.to(x.device, non_blocking=True)
            synchronize_device(x.device)

        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        if self._enable_img_in_txt_in_offloading:
            self.img_in.to(torch.device("cpu"), non_blocking=True)
            self.txt_in.to(torch.device("cpu"), non_blocking=True)
            synchronize_device(x.device)
            clean_memory_on_device(x.device)

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        attn_mask = total_len = None
        if self.split_attn or self.attn_mode == "torch":
            text_len = text_mask.sum(dim=1)  # (bs, )
            total_len = img_seq_len + text_len  # (bs, )
        if self.attn_mode == "torch" and not self.split_attn:
            bs = img.shape[0]
            attn_mask = torch.zeros((bs, 1, max_seqlen_q, max_seqlen_q), dtype=torch.bool, device=text_mask.device)
            for i in range(bs):
                attn_mask[i, :, : total_len[i], : total_len[i]] = True
            total_len = None  # means we don't use split_attn

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        input_device = img.device
        for block_idx, block in enumerate(self.double_blocks):
            double_block_args = [
                img,
                txt,
                vec,
                attn_mask,
                total_len,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
            ]

            img, txt = block(*double_block_args)

        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)

        if len(self.single_blocks) > 0:
            for block_idx, block in enumerate(self.single_blocks):
                # Cross the PCIe boundary once at the split point: bridge the
                # merged activation x and every tensor in the per-block arg
                # list to cuda:1. Subsequent cuda:1 blocks reuse cuda:1 args.
                if block_idx == split_at:
                    x = x.to(cuda1)
                    vec = vec.to(cuda1)

                single_block_args = _bridge_block_args(
                    [
                        x,
                        vec,
                        txt_seq_len,
                        attn_mask,
                        total_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        freqs_cis,
                    ],
                    x.device,
                )

                x = block(*single_block_args)

        img = x[:, :img_seq_len, ...]
        x = None
        if img.device != input_device:
            img = img.to(input_device)

        # ---------------------------- Final layer ------------------------------
        # final_layer lives on cuda:1; bring vec along (it may be on cuda:0 if
        # the split fired after block 0, but is on cuda:1 if split_at > 0 and
        # we re-pinned it above — re-pin defensively to final_layer's device).
        fl_device = next(self.final_layer.parameters()).device
        img = self.final_layer(img.to(fl_device), vec.to(fl_device))  # (N, T, patch_size ** 2 * out_channels)
        if img.device != input_device:
            img = img.to(input_device)

        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img

    model.forward = types.MethodType(forward, model)


def _bridge_block_args(args, device: torch.device):
    """Move every tensor in a HunyuanVideo per-block arg list to ``device``.

    HunyuanVideo's ``forward`` rebuilds a positional arg list per block (vs.
    Wan's single reused ``kwargs`` dict). Entries are a mix of tensors
    (``img``/``txt``/``x``, ``vec``, ``attn_mask``), the ``freqs_cis`` tuple
    of two tensors, ints (``txt_seq_len``, ``cu_seqlens_*``, ``max_seqlen_*``)
    and ``None``. Recurse over lists/tuples; move tensors; leave the rest
    alone. ``Tensor.to`` is an identity no-op when already on ``device``.
    """
    bridged = []
    for value in args:
        if torch.is_tensor(value):
            bridged.append(value.to(device))
        elif isinstance(value, (list, tuple)):
            bridged.append(_bridge_block_args(value, device))
        else:
            bridged.append(value)
    return type(args)(bridged) if isinstance(args, tuple) else bridged
