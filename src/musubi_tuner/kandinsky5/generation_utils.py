# This file includes code derived from:
# https://github.com/kandinskylab/kandinsky-5
# Copyright (c) 2025 Kandinsky Lab
# Licensed under the MIT License

import torch
from PIL import Image
from torch.distributed import all_gather
from tqdm import tqdm

from .models.utils import fast_sta_nabla
import torchvision.transforms.functional as F
from math import sqrt
from typing import Sequence, Union


def resize_image(image, max_area, divisibility=16):
    h, w = image.shape[2:]
    area = h * w
    k = min(1.0, sqrt(max_area / area))
    new_h = int(round((h * k) / divisibility) * divisibility)
    new_w = int(round((w * k) / divisibility) * divisibility)
    new_h = max(divisibility, new_h)
    new_w = max(divisibility, new_w)
    return F.resize(image, (new_h, new_w)), k


def _to_pil(image):
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image
    raise ValueError(f"unknown image type: {type(image)}")


def get_reference_latents(
    image: Union[str, Image.Image, Sequence],
    vae,
    device,
    max_area,
    divisibility,
    i2v_mode: str = "first",
):
    """
    Returns reference PIL (first element), stacked reference latents [N, H, W, C], and resize scale.
    Supports single image or list/tuple for first+last conditioning.
    """
    if isinstance(image, (list, tuple)):
        pil_images = [_to_pil(im) for im in image]
    else:
        pil_images = [_to_pil(image)]

    # resize target from the first image to keep spatial shape consistent across references
    image_tensor = F.pil_to_tensor(pil_images[0]).unsqueeze(0)
    image_tensor, k = resize_image(image_tensor, max_area=max_area, divisibility=divisibility)
    target_hw = image_tensor.shape[2:]

    latents = []
    target_dtype = getattr(vae, "dtype", torch.float16)
    for pil in pil_images:
        tensor = F.pil_to_tensor(pil).unsqueeze(0)
        tensor = F.resize(tensor, target_hw)
        tensor = tensor / 127.5 - 1.0
        with torch.no_grad():
            tensor = tensor.to(device=device, dtype=target_dtype).transpose(0, 1).unsqueeze(0)
            try:
                enc_out = vae.encode(tensor, opt_tiling=False)
            except TypeError:
                enc_out = vae.encode(tensor)
            lat_image = enc_out.latent_dist.sample().squeeze(0).permute(1, 2, 3, 0)
            lat_image = lat_image * vae.config.scaling_factor
            latents.append(lat_image)

    latents = torch.stack(latents, dim=0) if len(latents) > 1 else latents[0]

    # If caller requested first_last but only one image provided, duplicate to keep indices valid downstream.
    if i2v_mode == "first_last" and latents.dim() == 3:
        latents = torch.stack([latents, latents], dim=0)

    return pil_images[0], latents, k


def get_first_frame_from_image(image, vae, device, max_area, divisibility):
    """Backward-compatible helper: returns a single-frame latent and scale."""
    pil, latents, k = get_reference_latents(image, vae, device, max_area, divisibility, i2v_mode="first")
    if latents.dim() == 4 and latents.shape[0] > 1:
        latents = latents[0]
    return pil, latents, k


def get_sparse_params(conf, batch_embeds, device):
    assert conf.model.dit_params.patch_size[0] == 1
    T, H, W, _ = batch_embeds["visual"].shape
    T, H, W = (
        T // conf.model.dit_params.patch_size[0],
        H // conf.model.dit_params.patch_size[1],
        W // conf.model.dit_params.patch_size[2],
    )
    if conf.model.attention.type == "nabla":
        sta_mask = fast_sta_nabla(
            T, H // 8, W // 8, conf.model.attention.wT, conf.model.attention.wH, conf.model.attention.wW, device=device
        )
        sparse_params = {
            "sta_mask": sta_mask.unsqueeze_(0).unsqueeze_(0),
            "attention_type": conf.model.attention.type,
            "to_fractal": True,
            "P": conf.model.attention.P,
            "wT": conf.model.attention.wT,
            "wW": conf.model.attention.wW,
            "wH": conf.model.attention.wH,
            "add_sta": conf.model.attention.add_sta,
            "visual_shape": (T, H, W),
            "method": getattr(conf.model.attention, "method", "topcdf"),
        }
    else:
        sparse_params = None

    return sparse_params


def adaptive_mean_std_normalization(source, reference):
    source_mean = source.mean(dim=(1, 2, 3), keepdim=True)
    source_std = source.std(dim=(1, 2, 3), keepdim=True)
    # magic constants - limit changes in latents
    clump_mean_low = 0.05
    clump_mean_high = 0.1
    clump_std_low = 0.1
    clump_std_high = 0.25

    reference_mean = torch.clamp(reference.mean(), source_mean - clump_mean_low, source_mean + clump_mean_high)
    reference_std = torch.clamp(reference.std(), source_std - clump_std_low, source_std + clump_std_high)

    # normalization
    normalized = (source - source_mean) / source_std
    normalized = normalized * reference_std + reference_mean

    return normalized


def normalize_first_frame(latents, reference_frames=5, clump_values=False):
    latents_copy = latents.clone()
    samples = latents_copy

    if samples.shape[0] <= 1:
        return (latents, "Only one frame, no normalization needed")
    nFr = 4
    first_frames = samples[:nFr]
    reference_frames_data = samples[nFr : nFr + min(reference_frames, samples.shape[0] - 1)]

    # print("First frame stats - Mean:", first_frames.mean(dim=(1,2,3)), "Std: ", first_frames.std(dim=(1,2,3)))
    # print(f"Reference frames stats - Mean: {reference_frames_data.mean().item():.4f}, Std: {reference_frames_data.std().item():.4f}")

    normalized_first = adaptive_mean_std_normalization(first_frames, reference_frames_data)
    if clump_values:
        min_val = reference_frames_data.min()
        max_val = reference_frames_data.max()
        normalized_first = torch.clamp(normalized_first, min_val, max_val)

    samples[:nFr] = normalized_first

    return samples


@torch.no_grad()
def get_velocity(
    dit,
    x,
    t,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    conf,
    sparse_params=None,
    attention_mask=None,
    null_attention_mask=None,
):
    with torch._dynamo.utils.disable_cache_limit():
        pred_velocity = dit(
            x,
            text_embeds["text_embeds"],
            text_embeds["pooled_embed"],
            t * 1000,
            visual_rope_pos,
            text_rope_pos,
            scale_factor=conf.metrics.scale_factor,
            sparse_params=sparse_params,
            attention_mask=attention_mask,
        )
        if abs(guidance_weight - 1.0) > 1e-6:
            uncond_pred_velocity = dit(
                x,
                null_text_embeds["text_embeds"],
                null_text_embeds["pooled_embed"],
                t * 1000,
                visual_rope_pos,
                null_text_rope_pos,
                scale_factor=conf.metrics.scale_factor,
                sparse_params=sparse_params,
                attention_mask=null_attention_mask,
            )
            pred_velocity = uncond_pred_velocity + guidance_weight * (pred_velocity - uncond_pred_velocity)
    return pred_velocity


@torch.no_grad()
def decode_latents(latent_visual, vae, device="cuda", batch_size=1, num_frames=None):
    """Decode latent video to uint8 images. latent_visual: [B*F, H, W, C] -> [B, F, H, W, C]"""
    b_times_f, h, w, c = latent_visual.shape
    if num_frames is None:
        num_frames = b_times_f // batch_size
    latent_visual = latent_visual.reshape(batch_size, num_frames, h, w, c)

    # enforce BCHWT ordering and correct scaling factor
    latent_visual = latent_visual.to(device=device, dtype=vae.dtype)
    images = (latent_visual / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)  # B, C, F, H, W
    images = vae.decode(images).sample  # B, C, F, H, W
    images = images.permute(0, 2, 3, 4, 1)  # B, F, H, W, C
    images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)
    return images


@torch.no_grad()
def generate_sample_latents_only(
    shape,
    dit,
    text_embeds,
    pooled_embed,
    attention_mask,
    null_text_embeds=None,
    null_pooled_embed=None,
    null_attention_mask=None,
    first_frames=None,
    num_steps=25,
    guidance_weight=5.0,
    scheduler_scale=1,
    seed=6554,
    device="cuda",
    conf=None,
    progress=False,
    i2v_mode=None,  # unused; kept for call-site compatibility
):
    """Minimal sampler that returns latents only (no VAE decode)."""
    bs, duration, height, width, dim = shape

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)

    # Normalize text shapes; squeeze singleton batch to packed (S, D) when present, reshape/trim masks accordingly.
    if text_embeds.dim() == 3 and text_embeds.shape[0] == 1:
        text_embeds = text_embeds.squeeze(0)
        if attention_mask is not None and attention_mask.dim() > 1:
            attention_mask = attention_mask.reshape(1, -1)
    seq_len = text_embeds.shape[0] if text_embeds.dim() == 2 else text_embeds.shape[1]
    if attention_mask is None:
        attention_mask = torch.ones((1, seq_len), dtype=torch.bool, device=text_embeds.device)
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    # trim/pad mask to seq_len
    if attention_mask.shape[1] > seq_len:
        attention_mask = attention_mask[:, :seq_len]
    elif attention_mask.shape[1] < seq_len:
        pad = seq_len - attention_mask.shape[1]
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad), value=True)

    if null_text_embeds is None:
        null_text_embeds = torch.zeros_like(text_embeds)
    if null_text_embeds.dim() == 3 and null_text_embeds.shape[0] == 1:
        null_text_embeds = null_text_embeds.squeeze(0)
        if null_attention_mask is not None and null_attention_mask.dim() > 1:
            null_attention_mask = null_attention_mask.reshape(1, -1)
    null_seq_len = null_text_embeds.shape[0] if null_text_embeds.dim() == 2 else null_text_embeds.shape[1]
    if null_pooled_embed is None:
        null_pooled_embed = torch.zeros_like(pooled_embed)
    if null_attention_mask is None:
        null_attention_mask = attention_mask
    if null_attention_mask.dim() == 1:
        null_attention_mask = null_attention_mask.unsqueeze(0)
    if null_attention_mask.shape[1] > null_seq_len:
        null_attention_mask = null_attention_mask[:, :null_seq_len]
    elif null_attention_mask.shape[1] < null_seq_len:
        pad = null_seq_len - null_attention_mask.shape[1]
        null_attention_mask = torch.nn.functional.pad(null_attention_mask, (0, pad), value=True)

    attention_mask = attention_mask.to(device=device, dtype=torch.bool)
    null_attention_mask = null_attention_mask.to(device=device, dtype=torch.bool)
    text_embeds = text_embeds.to(device=device)
    null_text_embeds = null_text_embeds.to(device=device)
    pooled_embed = pooled_embed.to(device=device)
    null_pooled_embed = null_pooled_embed.to(device=device)

    text_dict = {"text_embeds": text_embeds, "pooled_embed": pooled_embed}
    null_text_dict = {"text_embeds": null_text_embeds, "pooled_embed": null_pooled_embed}

    # Shape/patch sanity guard: visual grid must be divisible by patch sizes
    ps_t, ps_h, ps_w = conf.model.dit_params.patch_size
    if (height % ps_h) != 0 or (width % ps_w) != 0 or (duration % ps_t) != 0:
        raise ValueError(
            f"Invalid visual shape for patch_size {ps_t, ps_h, ps_w}: frames={duration}, height={height}, width={width}"
        )

    visual_rope_pos = [
        torch.arange(duration, device=device),
        torch.arange(height // conf.model.dit_params.patch_size[1], device=device),
        torch.arange(width // conf.model.dit_params.patch_size[2], device=device),
    ]
    text_rope_pos = torch.arange(seq_len, device=device)
    null_text_rope_pos = torch.arange(null_seq_len, device=device)

    latents = generate(
        dit,
        device,
        img,
        num_steps,
        text_dict,
        null_text_dict,
        visual_rope_pos,
        text_rope_pos,
        null_text_rope_pos,
        guidance_weight,
        scheduler_scale,
        first_frames,
        conf,
        progress=progress,
        seed=seed,
        tp_mesh=None,
        attention_mask=attention_mask,
        null_attention_mask=null_attention_mask,
    )
    return latents


@torch.no_grad()
def generate(
    model,
    device,
    img,
    num_steps,
    text_embeds,
    null_text_embeds,
    visual_rope_pos,
    text_rope_pos,
    null_text_rope_pos,
    guidance_weight,
    scheduler_scale,
    first_frames,
    conf,
    progress=False,
    seed=6554,
    tp_mesh=None,
    attention_mask=None,
    null_attention_mask=None,
    first_frame_indices=None,
):
    sparse_params = get_sparse_params(conf, {"visual": img}, device)
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    timesteps = scheduler_scale * timesteps / (1 + (scheduler_scale - 1) * timesteps)

    if tp_mesh:
        tp_rank = tp_mesh["tensor_parallel"].get_local_rank()
        tp_world_size = tp_mesh["tensor_parallel"].size()
        img = torch.chunk(img, tp_world_size, dim=1)[tp_rank]

    for timestep, timestep_diff in tqdm(list(zip(timesteps[:-1], torch.diff(timesteps)))):
        time = timestep.unsqueeze(0)
        if model.visual_cond:
            visual_cond = torch.zeros_like(img)
            visual_cond_mask = torch.zeros([*img.shape[:-1], 1], dtype=img.dtype, device=img.device)
            if first_frames is not None:
                # Allow either a single frame (shape [..., H, W, C]) or multiple frames stacked on dim 0.
                ff = first_frames.to(device=visual_cond.device, dtype=visual_cond.dtype)
                if ff.dim() == img.dim() - 1:  # H, W, C
                    ff = ff.unsqueeze(0)
                indices = first_frame_indices or [0]
                if len(indices) > ff.shape[0]:
                    # If fewer frames provided than indices, repeat the last available frame.
                    ff = torch.cat([ff, ff[-1:].repeat(len(indices) - ff.shape[0], 1, 1, 1)], dim=0)
                for idx, frame_idx in enumerate(indices):
                    if 0 <= frame_idx < img.shape[0]:
                        img[frame_idx : frame_idx + 1] = ff[idx]
                        visual_cond_mask[frame_idx : frame_idx + 1] = 1
                        visual_cond[frame_idx : frame_idx + 1] = ff[idx]
            model_input = torch.cat([img, visual_cond, visual_cond_mask], dim=-1)
        else:
            model_input = img
        pred_velocity = get_velocity(
            model,
            model_input,
            time,
            text_embeds,
            null_text_embeds,
            visual_rope_pos,
            text_rope_pos,
            null_text_rope_pos,
            guidance_weight,
            conf,
            sparse_params=sparse_params,
            attention_mask=attention_mask,
            null_attention_mask=null_attention_mask,
        )
        img[..., : pred_velocity.shape[-1]] += timestep_diff * pred_velocity
        # NOTE: remove extra channels that can be added in Image Editing (I2I)
    return img[..., : pred_velocity.shape[-1]]


def resize_video(video, visual_size):
    height, width = video.shape[-2:]
    nearest_height, nearest_width = visual_size

    scale_factor = min(height / nearest_height, width / nearest_width)
    video = F.resize(video, (int(height / scale_factor), int(width / scale_factor)))

    height, width = video.shape[-2:]
    video = F.crop(
        video,
        (height - nearest_height) // 2,
        (width - nearest_width) // 2,
        nearest_height,
        nearest_width,
    )
    return video


def encode_video(data, vae, image_vae):  # batch, channels, time, h, w
    if image_vae:
        assert data.shape[2] == 1
        data = vae.encode(data[:, :, 0]).latent_dist.sample()[:, :, None]
    else:
        data = vae.encode(data)[0]
    data *= vae.config.scaling_factor
    return data.permute(0, 2, 3, 4, 1)  # batch, time, h, w, channels


def generate_sample(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    num_steps=25,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    text_embedder_device="cuda",
    progress=True,
    offload=False,
    tp_mesh=None,
):
    bs, duration, height, width, dim = shape

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)

    # Use the dedicated image-to-video prompt template for both text and negative text.
    type_of_content = "image2video"

    with torch.no_grad():
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode([caption], type_of_content=type_of_content)
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    if offload:
        text_embedder = text_embedder.to("cpu")

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    if offload:
        dit.to(device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                img,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                None,
                conf,
                seed=seed,
                progress=progress,
                tp_mesh=tp_mesh,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
            )

    if tp_mesh:
        tensor_list = [
            torch.zeros_like(latent_visual, device=latent_visual.device) for _ in range(tp_mesh["tensor_parallel"].size())
        ]
        all_gather(tensor_list, latent_visual.contiguous(), group=tp_mesh.get_group(mesh_dim="tensor_parallel"))
        latent_visual = torch.cat(tensor_list, dim=1)

    if offload:
        dit = dit.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()

    if offload:
        vae = vae.to(vae_device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    if offload:
        vae = vae.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()

    return images


def generate_sample_ti2i(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    num_steps=25,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    text_embedder_device="cuda",
    progress=True,
    offload=False,
    image_vae=False,
    image=None,
):
    bs, duration, height, width, dim = shape

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)

    if duration == 1:
        if image is None:
            type_of_content = "image"
        else:
            type_of_content = "image_edit"
    else:
        type_of_content = "video"

    if image is not None:
        image = [resize_video(image, (height * 8, width * 8))]

    if dit.instruct_type == "channel":
        if image is not None:
            if offload:
                vae.to(vae_device)
            edit_latent = [(i.to(device=vae_device, dtype=torch.bfloat16) / 127.5 - 1.0) for i in image]
            edit_latent = torch.cat([encode_video(i[:, :, None], vae, image_vae).squeeze(0) for i in edit_latent], 0)
            edit_latent = torch.cat([edit_latent, torch.ones_like(img[..., :1])], -1)
            if offload:
                vae.to("cpu")
        else:
            edit_latent = torch.cat([torch.zeros_like(img), torch.zeros_like(img[..., :1])], -1)
        img = torch.cat([img, edit_latent], dim=-1)

    with torch.no_grad():
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode(
            [caption], type_of_content=type_of_content, images=image
        )
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content, images=image
        )

    if offload:
        text_embedder = text_embedder.to("cpu")

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device, dtype=torch.bfloat16)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device, dtype=torch.bfloat16)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    if offload:
        dit.to(device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                img,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                None,
                conf,
                seed=seed,
                progress=progress,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
            )

    if offload:
        dit = dit.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()

    if offload:
        vae = vae.to(vae_device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            if image_vae:
                images = images[:, :, 0]
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    if offload:
        vae = vae.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()

    return images


def generate_sample_i2v(
    shape,
    caption,
    dit,
    vae,
    conf,
    text_embedder,
    images,
    num_steps=50,
    guidance_weight=5.0,
    scheduler_scale=1,
    negative_caption="",
    seed=6554,
    device="cuda",
    vae_device="cuda",
    progress=True,
    offload=False,
    tp_mesh=None,
    i2v_mode="first",
):
    text_embedder.embedder.mode = "i2v"
    bs, duration, height, width, dim = shape

    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    img = torch.randn(bs * duration, height, width, dim, device=device, generator=g, dtype=torch.bfloat16)

    if duration == 1:
        type_of_content = "image"
    else:
        type_of_content = "video"

    with torch.no_grad():
        bs_text_embed, text_cu_seqlens, attention_mask = text_embedder.encode([caption], type_of_content=type_of_content)
        bs_null_text_embed, null_text_cu_seqlens, null_attention_mask = text_embedder.encode(
            [negative_caption], type_of_content=type_of_content
        )

    if offload:
        text_embedder = text_embedder.to("cpu")

    for key in bs_text_embed:
        bs_text_embed[key] = bs_text_embed[key].to(device=device)
        bs_null_text_embed[key] = bs_null_text_embed[key].to(device=device)
    text_cu_seqlens = text_cu_seqlens.to(device=device)[-1].item()
    null_text_cu_seqlens = null_text_cu_seqlens.to(device=device)[-1].item()

    visual_rope_pos = [
        torch.arange(duration),
        torch.arange(shape[-3] // conf.model.dit_params.patch_size[1]),
        torch.arange(shape[-2] // conf.model.dit_params.patch_size[2]),
    ]
    text_rope_pos = torch.arange(text_cu_seqlens)
    null_text_rope_pos = torch.arange(null_text_cu_seqlens)

    if offload:
        dit.to(device, non_blocking=True)

    # Prepare conditioning frames and placement indices.
    first_frames = images
    first_frame_indices = [0]
    if i2v_mode == "first_last" and duration > 1 and images is not None:
        # Expect images shape [F,H,W,C]; if only one provided, duplicate it.
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.shape[0] == 1:
            images = torch.cat([images, images], dim=0)
        first_frames = images[:2]
        first_frame_indices = [0, duration - 1]

    if tp_mesh and first_frames is not None and first_frames.dim() > 3:
        tp_rank = tp_mesh["tensor_parallel"].get_local_rank()
        tp_world_size = tp_mesh["tensor_parallel"].size()
        first_frames = torch.chunk(first_frames, tp_world_size, dim=0)[tp_rank]

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent_visual = generate(
                dit,
                device,
                img,
                num_steps,
                bs_text_embed,
                bs_null_text_embed,
                visual_rope_pos,
                text_rope_pos,
                null_text_rope_pos,
                guidance_weight,
                scheduler_scale,
                first_frames,
                conf,
                seed=seed,
                progress=progress,
                tp_mesh=tp_mesh,
                attention_mask=attention_mask,
                null_attention_mask=null_attention_mask,
                first_frame_indices=first_frame_indices if first_frames is not None else None,
            )
    if tp_mesh:
        tensor_list = [
            torch.zeros_like(latent_visual, device=latent_visual.device) for _ in range(tp_mesh["tensor_parallel"].size())
        ]
        all_gather(tensor_list, latent_visual.contiguous(), group=tp_mesh.get_group(mesh_dim="tensor_parallel"))
        latent_visual = torch.cat(tensor_list, dim=1)

    if first_frames is not None:
        ff = first_frames.to(device=latent_visual.device, dtype=latent_visual.dtype)
        if ff.dim() == 3:
            ff = ff.unsqueeze(0)
        for idx, frame_idx in enumerate(first_frame_indices):
            if frame_idx < latent_visual.shape[0]:
                latent_visual[frame_idx : frame_idx + 1] = ff[min(idx, ff.shape[0] - 1)]
    latent_visual = normalize_first_frame(latent_visual)

    if offload:
        dit = dit.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()

    if offload:
        vae = vae.to(vae_device, non_blocking=True)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            images = latent_visual.reshape(
                bs,
                -1,
                latent_visual.shape[-3],
                latent_visual.shape[-2],
                latent_visual.shape[-1],
            )
            images = images.to(device=vae_device)
            images = (images / vae.config.scaling_factor).permute(0, 4, 1, 2, 3)
            images = vae.decode(images).sample
            images = ((images.clamp(-1.0, 1.0) + 1.0) * 127.5).to(torch.uint8)

    if offload:
        vae = vae.to("cpu", non_blocking=True)
    torch.cuda.empty_cache()

    return images
