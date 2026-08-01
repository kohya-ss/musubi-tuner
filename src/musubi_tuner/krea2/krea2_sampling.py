"""Functional flow-matching sampler for K2 text-to-image and edit generation."""

import gc
import math

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image
from tqdm import tqdm


def roundup(value, multiple, name):
    """Round `value` up to the nearest multiple, logging when padding is applied."""
    aligned = ((value + multiple - 1) // multiple) * multiple
    if aligned != value:
        print(f"[sample] {name}={value} is not a multiple of {multiple}; padding to {aligned}")
    return aligned


def gather_valid_text(txt, mask):
    """Drop masked (invalid) text tokens so the valid ones form a contiguous prefix, then
    right-pad to the batch maximum.

    The Qwen3-VL conditioner pads the prompt to max_length and appends the template suffix,
    so its mask is [valid prompt, pad, valid suffix] — valid tokens are NOT a prefix. The
    shared attention (cu_seqlens / trim) assumes valid == leading prefix, so the interior
    padding must be removed first. Dropping it is lossless: text tokens get zero RoPE position
    and padding is masked out, so only the set/order of valid tokens matters.

    txt: (B, seq, L, D), mask: (B, seq) bool -> (B, max_valid, L, D), (B, max_valid) bool.
    """
    valid = [txt[i][mask[i]] for i in range(txt.shape[0])]  # list of (n_i, L, D)
    max_len = max(v.shape[0] for v in valid)
    out = txt.new_zeros(txt.shape[0], max_len, txt.shape[2], txt.shape[3])
    newmask = torch.zeros(txt.shape[0], max_len, device=txt.device, dtype=torch.bool)
    for i, v in enumerate(valid):
        out[i, : v.shape[0]] = v
        newmask[i, : v.shape[0]] = True
    return out, newmask


def load_reference_images(paths: list[str]) -> list[torch.Tensor]:
    """Load RGB reference images as CPU ``(C,H,W)`` float tensors in ``[0,1]``."""
    images = []
    for path in paths:
        with Image.open(path) as image:
            array = np.array(image.convert("RGB"), copy=True)
        images.append(torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0))
    return images


def resize_reference_image(
    image: torch.Tensor,
    max_pixels: int,
    *,
    snap: int = 1,
    mode: str = "bilinear",
    min_size: int | None = None,
    antialias: bool = True,
    allow_upscale: bool = False,
) -> torch.Tensor:
    """Resize a ``(C,H,W)`` image to a pixel-area budget, preserving aspect ratio."""
    if image.ndim != 3:
        raise ValueError(f"Expected a (C,H,W) reference image, got {tuple(image.shape)}")
    _, height, width = image.shape
    scale = math.sqrt(max_pixels / (height * width))
    if not allow_upscale:
        scale = min(1.0, scale)
    new_height = max(round(height * scale / snap) * snap, snap)
    new_width = max(round(width * scale / snap) * snap, snap)
    if min_size is not None:
        new_height = max(new_height, min_size)
        new_width = max(new_width, min_size)
    if (new_height, new_width) == (height, width):
        return image
    kwargs = {"mode": mode, "size": (new_height, new_width)}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
        kwargs["antialias"] = antialias
    return F.interpolate(image.unsqueeze(0).float(), **kwargs).squeeze(0).clamp_(0, 1)


def prepare_vlm_reference_images(images: list[torch.Tensor], max_pixels: int = 384 * 384) -> list[torch.Tensor]:
    """Prepare low-resolution reference copies for Qwen3-VL conditioning."""
    return [resize_reference_image(image, max_pixels, mode="bicubic", min_size=28) for image in images]


@torch.no_grad()
def encode_reference_images(
    ae,
    images: list[torch.Tensor],
    device: torch.device | str,
    dtype: torch.dtype,
    max_pixels: int = 1024 * 1024,
    snap: int = 16,
) -> list[torch.Tensor]:
    """VAE-encode clean edit references for ``index_timestep_zero`` conditioning."""
    if not images:
        return []
    ae.to(device)
    ae.eval()
    latents = []
    for image in images:
        # ai-toolkit uses plain bilinear interpolation for the VAE reference
        # branch (antialias=False); the Qwen3-VL branch remains antialiased.
        pixels = resize_reference_image(image, max_pixels, snap=snap, antialias=False)
        # Musubi's Qwen-Image VAE encoder consumes pixels in [-1,1].
        pixels = pixels.unsqueeze(0).mul(2).sub(1).to(device=device, dtype=ae.dtype)
        # ai-toolkit samples the Qwen-Image VAE posterior for clean references.
        # The normal dataset-cache path keeps using posterior.mode(); this
        # stochastic path is scoped to Krea 2 Edit sampling only.
        latent = ae.encode_pixels_to_latents(pixels, sample_posterior=True)
        # Qwen-Image VAE emits (B,C,T,H,W); Krea 2 edit uses one-frame images.
        latents.append(latent[0, :, 0].to(device=device, dtype=dtype))
    ae.to("cpu")
    return latents


def pack_reference_latents(
    ref_latents: list[torch.Tensor],
    batch_size: int,
    patch: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patchify references and assign each one its own RoPE axis-0 index."""
    tokens = []
    positions = []
    for index, ref in enumerate(ref_latents):
        if ref.ndim == 3:
            ref = ref.unsqueeze(0)
        elif ref.ndim == 5:
            ref = rearrange(ref, "b c t h w -> (b t) c h w")
        if ref.ndim != 4:
            raise ValueError(f"Expected reference latent with 3, 4 or 5 dimensions, got {tuple(ref.shape)}")
        ref = ref.to(device=device, dtype=dtype)
        if ref.shape[0] == 1 and batch_size > 1:
            ref = ref.repeat(batch_size, 1, 1, 1)
        if ref.shape[0] != batch_size:
            raise ValueError(f"Reference batch size {ref.shape[0]} does not match sample batch size {batch_size}")

        pad_height = (-ref.shape[-2]) % patch
        pad_width = (-ref.shape[-1]) % patch
        if pad_height or pad_width:
            ref = F.pad(ref, (0, pad_width, 0, pad_height))
        ref_height, ref_width = ref.shape[-2] // patch, ref.shape[-1] // patch
        tokens.append(rearrange(ref, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch))

        ref_ids = torch.zeros(ref_height, ref_width, 3, device=device)
        ref_ids[..., 0] = index + 1
        ref_ids[..., 1] = torch.arange(ref_height, device=device)[:, None]
        ref_ids[..., 2] = torch.arange(ref_width, device=device)[None, :]
        positions.append(repeat(ref_ids, "h w three -> b (h w) three", b=batch_size, three=3))

    if not tokens:
        raise ValueError("At least one reference latent is required")
    return torch.cat(tokens, dim=1), torch.cat(positions, dim=1)


def append_reference_latents(img, pos, mask, ref_latents, patch):
    """Insert reference tokens between the noisy image tokens and text tokens."""
    ref_tokens, ref_pos = pack_reference_latents(ref_latents, img.shape[0], patch, img.device, img.dtype)
    image_len = img.shape[1]
    ref_mask = torch.ones(img.shape[0], ref_tokens.shape[1], device=img.device, dtype=torch.bool)
    img = torch.cat((img, ref_tokens), dim=1)
    pos = torch.cat((pos[:, :image_len], ref_pos, pos[:, image_len:]), dim=1)
    mask = torch.cat((mask[:, :image_len], ref_mask, mask[:, image_len:]), dim=1)
    return img, pos, mask, ref_tokens.shape[1]


def prepare(img, txtlen, patch, txtmask):
    """Patchify the latent and build the combined image+text position / mask tensors.

    Image tokens lead the sequence so each sample's valid tokens form a contiguous prefix
    ([img (all valid), text (valid prefix + padding)]), which the shared attention's
    varlen / cu_seqlens path requires. Returns (img_tokens, pos, mask).
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((imgmask, txtmask), dim=1)
    pos = torch.cat((imgpos, txtpos), dim=1)
    return img, pos, mask


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    """Resolution-aware flow-matching timestep schedule (t: 1 -> 0).

    `mu` is interpolated linearly in image-sequence length between (x1,y1) and
    (x2,y2), then used to time-shift a uniform 1->0 grid. Pass an explicit `mu`
    to pin a constant shift regardless of resolution (used by the distilled
    checkpoint, which was trained at a fixed mu=1.15).
    """
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


@torch.no_grad()
def encode_prompts(encoder, prompts, negative_prompts=None, *, cfg=True, images=None):
    """Encode prompts (and optional negatives) into gathered varlen text embeddings.

    Returns ``(txt, txtmask, untxt, untxtmask)``; the unconditional pair is ``None`` when
    ``cfg`` is False. Run this BEFORE loading the DiT so the (~8GB Qwen3-VL) encoder can be
    freed and not compete with the DiT for VRAM — on a 24GB card the encoder and the DiT do
    not fit at the same time. ``gather_valid_text`` drops the interior padding the encoder
    inserts between prompt and suffix so the valid tokens form a contiguous prefix.
    """
    txt, txtmask = encoder(prompts) if images is None else encoder(prompts, images=images)
    txt, txtmask = gather_valid_text(txt, txtmask)

    untxt = untxtmask = None
    if cfg:
        if negative_prompts is None:
            negative_prompts = [""] * len(prompts)
        untxt, untxtmask = encoder(negative_prompts)
        untxt, untxtmask = gather_valid_text(untxt, untxtmask)

    return txt, txtmask, untxt, untxtmask


@torch.no_grad()
def sample(
    model,
    ae,
    txt,
    txtmask,
    *,
    untxt=None,
    untxtmask=None,
    device="cuda",
    dtype=torch.bfloat16,
    width=1024,
    height=1024,
    steps=28,
    cfg_scale=5.5,
    seed=0,
    minres=256,
    maxres=1280,
    y1=0.5,
    y2=1.15,
    mu=None,
    ref_latents=None,
    kv_cache=False,
):
    """Denoise pre-encoded text embeddings to images: euler+CFG denoise -> decode.

    Takes the gathered text embeddings from ``encode_prompts`` (not the encoder), so the
    encoder can be freed before this runs. CFG is enabled when ``cfg_scale > 1`` and an
    unconditional embedding (``untxt``) was provided.

    The DiT (``model``) stays resident on its device for the whole call — it is never moved
    to CPU. The VAE is kept on CPU and moved to the latent's device only for the final decode,
    then moved back to CPU before returning. So the only VRAM the decode adds on top of the
    resident DiT is the VAE plus its transient activations; that headroom is expected to come
    from running the DiT under fp8 and/or block swap (moving the ~24GB DiT to CPU instead would
    only shift the pressure onto host RAM). Keeping the DiT in place lets the caller reuse it
    for the next prompt without reloading.
    """
    patch = model.config.patch

    # Qwen-Image VAE geometry (f8, 16 latent channels), read from the musubi
    # AutoencoderKLQwenImage so K2 shares the same VAE as the rest of musubi.
    compression = 2 ** len(ae.temperal_downsample)
    channels = ae.z_dim

    # The latent grid (dim // compression) is patchified in `patch`-sized blocks,
    # so width/height must be multiples of compression * patch. Pad up otherwise.
    align = compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    n = txt.shape[0]
    cfg = cfg_scale > 1.0 and untxt is not None

    # Text embeddings come from the (now-freed) encoder; make sure they are on the compute device.
    txt, txtmask = txt.to(device=device, dtype=dtype), txtmask.to(device)
    if cfg:
        untxt, untxtmask = untxt.to(device=device, dtype=dtype), untxtmask.to(device)

    if kv_cache and not ref_latents:
        raise ValueError("Krea 2 kv_cache sampling requires at least one reference latent")

    # Per-prompt seeded gaussian latent noise. Keep the ODE state in fp32 like
    # ai-toolkit; only the tensor presented to the DiT is cast to model dtype.
    noise = torch.cat(
        [
            torch.randn(
                1,
                channels,
                height // compression,
                width // compression,
                device="cpu",
                dtype=torch.float32,
                generator=torch.Generator(device="cpu").manual_seed(seed + i),
            )
            for i in range(n)
        ],
        dim=0,
    ).to(device)

    x, base_pos, base_mask = prepare(noise, txt.shape[1], patch, txtmask)
    target_seq_len = x.shape[1]
    pos, mask = base_pos, base_mask
    reflen = 0
    ref_tokens = None
    if ref_latents:
        model_img, pos, mask, reflen = append_reference_latents(x, pos, mask, ref_latents, patch)
        ref_tokens = model_img[:, target_seq_len:].to(dtype=dtype)
    unbase_pos = unbase_mask = None
    if cfg:
        _, unbase_pos, unbase_mask = prepare(noise, untxt.shape[1], patch, untxtmask)
        unpos, unmask = unbase_pos, unbase_mask
        if ref_latents:
            _, unpos, unmask, _ = append_reference_latents(x, unpos, unmask, ref_latents, patch)

    # min_res/max_res define the (x1,y1)-(x2,y2) interpolation endpoints for `mu`.
    x1 = (minres // (compression * patch)) ** 2
    x2 = (maxres // (compression * patch)) ** 2
    ts = timesteps(target_seq_len, steps, x1, x2, y1=y1, y2=y2, mu=mu)

    # Euler integration of the flow ODE with CFG. Run the DiT under autocast: with fp8 the
    # non-quantized layers (e.g. `first`) keep their checkpoint dtype (fp32), so without
    # autocast a bf16 activation hits "mat1 and mat2 must have the same dtype". This mirrors
    # how training wraps both call_dit and sample generation (trainer_base) in autocast; for
    # the non-fp8 (all-bf16) path it is effectively a no-op.
    img = x.float()
    ref_cache = None
    device_type = torch.device(device).type
    with torch.autocast(device_type=device_type, dtype=dtype):
        for tcurr, tprev in tqdm(zip(ts[:-1], ts[1:]), total=len(ts) - 1, desc="sampling"):
            t = torch.full((len(img),), tcurr, dtype=dtype, device=img.device)
            live_img = img.to(dtype=dtype)

            if kv_cache and ref_cache is not None:
                cond = model(
                    img=live_img,
                    context=txt,
                    t=t,
                    pos=base_pos,
                    mask=base_mask,
                    ref_kv_cache=ref_cache,
                )
            else:
                model_img = torch.cat((live_img, ref_tokens), dim=1) if ref_tokens is not None else live_img
                capture = [] if kv_cache else None
                cond_kwargs = dict(
                    img=model_img,
                    context=txt,
                    t=t,
                    pos=pos,
                    mask=mask,
                    reflen=reflen,
                )
                if kv_cache:
                    cond_kwargs.update(isolate_refs=True, ref_kv_capture=capture)
                cond = model(**cond_kwargs)
                if capture is not None:
                    if len(capture) != len(model.blocks):
                        raise RuntimeError(
                            f"Krea 2 reference K/V capture produced {len(capture)} entries for "
                            f"{len(model.blocks)} transformer blocks"
                        )
                    ref_cache = [(cached_k.detach(), cached_v.detach()) for cached_k, cached_v in capture]

            if cfg:
                if kv_cache:
                    uncond = model(
                        img=live_img,
                        context=untxt,
                        t=t,
                        pos=unbase_pos,
                        mask=unbase_mask,
                        ref_kv_cache=ref_cache,
                    )
                else:
                    model_img = torch.cat((live_img, ref_tokens), dim=1) if ref_tokens is not None else live_img
                    uncond = model(img=model_img, context=untxt, t=t, pos=unpos, mask=unmask, reflen=reflen)
                v = uncond + cfg_scale * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v.float()

    # Unpatchify back to a latent (add the VAE frame axis) and decode to pixels.
    img = rearrange(
        img,
        "b (h w) (c ph pw) -> b c 1 (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=height // (compression * patch),
        w=width // (compression * patch),
    )
    # decode_to_pixels denormalizes (*std + mean), decodes, drops the frame axis, returns [0, 1].
    # Move the VAE to the latent's device for decode (it is kept on CPU otherwise to save VRAM),
    # then move it back to CPU so the next generation starts with the decode VRAM freed. The DiT
    # stays put on its device; the decode is expected to fit alongside it via fp8 / block swap.
    ae = ae.to(img.device)
    pixels = ae.decode_to_pixels(img.to(torch.bfloat16))
    ae = ae.to("cpu")
    pixels = rearrange(pixels * 255.0, "b c h w -> b h w c").round().clamp_(0, 255).cpu().to(torch.uint8).numpy()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [Image.fromarray(pixels[i]) for i in range(len(pixels))]
