"""Functional flow-matching sampler for the K2 MMDiT (no Scheduler class).

Ported verbatim from references/Krea2/sampling.py.
"""

import gc
import math

import torch
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
def encode_prompts(encoder, prompts, negative_prompts=None, *, cfg=True):
    """Encode prompts (and optional negatives) into gathered varlen text embeddings.

    Returns ``(txt, txtmask, untxt, untxtmask)``; the unconditional pair is ``None`` when
    ``cfg`` is False. Run this BEFORE loading the DiT so the (~8GB Qwen3-VL) encoder can be
    freed and not compete with the DiT for VRAM — on a 24GB card the encoder and the DiT do
    not fit at the same time. ``gather_valid_text`` drops the interior padding the encoder
    inserts between prompt and suffix so the valid tokens form a contiguous prefix.
    """
    txt, txtmask = encoder(prompts)
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

    # Per-prompt seeded gaussian latent noise.
    noise = torch.cat(
        [
            torch.randn(
                1,
                channels,
                height // compression,
                width // compression,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            for i in range(n)
        ],
        dim=0,
    )

    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)
    if cfg:
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    # min_res/max_res define the (x1,y1)-(x2,y2) interpolation endpoints for `mu`.
    x1 = (minres // (compression * patch)) ** 2
    x2 = (maxres // (compression * patch)) ** 2
    ts = timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)

    # Euler integration of the flow ODE with CFG. Run the DiT under autocast: with fp8 the
    # non-quantized layers (e.g. `first`) keep their checkpoint dtype (fp32), so without
    # autocast a bf16 activation hits "mat1 and mat2 must have the same dtype". This mirrors
    # how training wraps both call_dit and sample generation (trainer_base) in autocast; for
    # the non-fp8 (all-bf16) path it is effectively a no-op.
    img = x
    device_type = torch.device(device).type
    with torch.autocast(device_type=device_type, dtype=dtype):
        for tcurr, tprev in tqdm(zip(ts[:-1], ts[1:]), total=len(ts) - 1, desc="sampling"):
            t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
            cond = model(img=img, context=txt, t=t, pos=pos, mask=mask)
            if cfg:
                uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
                v = uncond + cfg_scale * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v

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
    pixels = rearrange(pixels * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [Image.fromarray(pixels[i]) for i in range(len(pixels))]
