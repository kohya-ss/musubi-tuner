# Copyright 2025 The MiniMax authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for Musubi from Hugging Face Diffusers PR #14355 at commit
# abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
# (schedulers/scheduling_minimax_h3.py and modular_pipelines/minimax_h3/denoise.py).
# ComfyUI is used only as an independent numerical reference.

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
from PIL import Image
import torch

from musubi_tuner.minimax_h3.media import AUDIO_SAMPLE_RATE, TARGET_FPS
from musubi_tuner.minimax_h3.packing import (
    AUDIO_CHANNELS,
    STEREO_CHANNELS,
    VIDEO_CHANNELS,
    H3PackedLayout,
    validate_clean_coefficient,
)

# released sampler defaults, shared by the generation CLI, the trainer's sampling flags and
# the function signatures below
DEFAULT_VIDEO_SHIFT = 12.0
DEFAULT_AUDIO_SHIFT = 3.0
DEFAULT_VISUAL_CONDITION_CLEAN = 0.999
DEFAULT_AUDIO_CONDITION_CLEAN = 1.0


@dataclass(frozen=True)
class H3SigmaSchedule:
    base: torch.Tensor
    video: torch.Tensor
    audio: torch.Tensor


@dataclass(frozen=True)
class H3SampleResult:
    video: torch.Tensor
    audio: torch.Tensor


@dataclass(frozen=True)
class H3DecodedAV:
    video: torch.Tensor
    audio: torch.Tensor
    fps: int = TARGET_FPS
    sample_rate: int = AUDIO_SAMPLE_RATE


def validate_shift(value: float, label: str) -> float:
    """The per-modality timestep shift must lie in [0.01,100]; the schedule builder, the trainer
    and the CLIs all validate through here."""
    value = float(value)
    if not 0.01 <= value <= 100.0:
        raise ValueError(f"MiniMax-H3 {label} must be in [0.01,100.0], got {value}")
    return value


def shift_sigma(base: torch.Tensor, shift: float) -> torch.Tensor:
    """The released H3 timestep shift ``s*u / (1 + (s-1)*u)`` of a base sigma in [0,1]; the
    sampler applies it to its schedule, training to each step's drawn base sigma."""
    return shift * base / (1.0 + (shift - 1.0) * base)


def build_shifted_schedule(
    steps: int,
    *,
    video_shift: float = DEFAULT_VIDEO_SHIFT,
    audio_shift: float = DEFAULT_AUDIO_SHIFT,
    device: torch.device | str | None = None,
) -> H3SigmaSchedule:
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError(f"MiniMax-H3 sampling steps must be a positive integer, got {steps}")
    video_shift = validate_shift(video_shift, "video shift")
    audio_shift = validate_shift(audio_shift, "audio shift")
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64, device=device)
    return H3SigmaSchedule(base=base, video=shift_sigma(base, video_shift), audio=shift_sigma(base, audio_shift))


def create_sampling_generator(seed: int) -> torch.Generator:
    """One seed drives one CPU noise stream, consumed in call order:
    initialize_target_latents (video, then audio), then augment_condition_latents
    (visual conditions, then audio conditions). Sequential draws from the shared
    stream keep every tensor's noise independent without per-purpose seed offsets."""
    return torch.Generator(device="cpu").manual_seed(int(seed))


def initialize_target_latents(
    *,
    video_shape: Sequence[int],
    audio_shape: Sequence[int],
    generator: torch.Generator,
    device: torch.device | str,
    video_dtype: torch.dtype = torch.float16,
    audio_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    video_shape = tuple(int(value) for value in video_shape)
    audio_shape = tuple(int(value) for value in audio_shape)
    if len(video_shape) != 5 or video_shape[1] != VIDEO_CHANNELS:
        raise ValueError(f"MiniMax-H3 target video noise shape must be [B,{VIDEO_CHANNELS},F,H,W], got {video_shape}")
    if len(audio_shape) != 4 or audio_shape[1:3] != (AUDIO_CHANNELS, STEREO_CHANNELS) or audio_shape[0] != video_shape[0]:
        raise ValueError(f"MiniMax-H3 target audio noise shape must be [B,{AUDIO_CHANNELS},{STEREO_CHANNELS},A], got {audio_shape}")
    video = torch.randn(video_shape, generator=generator, dtype=torch.float32, device="cpu").to(device=device, dtype=video_dtype)
    audio = torch.randn(audio_shape, generator=generator, dtype=torch.float32, device="cpu").to(device=device, dtype=audio_dtype)
    return video, audio


def _augment_condition_group(
    tensors: Sequence[torch.Tensor],
    *,
    generator: torch.Generator | None,
    clean: float,
    device: torch.device | str,
) -> tuple[torch.Tensor, ...]:
    moved = tuple(tensor.to(device) for tensor in tensors)
    if clean == 1.0:
        return moved
    augmented = []
    for tensor in moved:
        if generator is None:
            # training: fresh noise from the global RNG on the tensor's device, like the target noise
            noise = torch.randn(tuple(tensor.shape), dtype=torch.float32, device=tensor.device).to(tensor.dtype)
        else:
            noise = torch.randn(tuple(tensor.shape), generator=generator, dtype=torch.float32, device="cpu").to(
                device=tensor.device, dtype=tensor.dtype
            )
        augmented.append(clean * tensor + (1.0 - clean) * noise)
    return tuple(augmented)


def augment_condition_latents(
    visual_conditions: Sequence[torch.Tensor],
    audio_conditions: Sequence[torch.Tensor],
    *,
    generator: torch.Generator | None,
    visual_clean: float = DEFAULT_VISUAL_CONDITION_CLEAN,
    audio_clean: float = DEFAULT_AUDIO_CONDITION_CLEAN,
    device: torch.device | str,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Blend independent Gaussian noise into the condition latents: clean*x + (1-clean)*eps.

    Sampling passes the seed's CPU generator (see create_sampling_generator) so the noise is
    reproducible; training passes ``generator=None`` and draws from the global RNG per step.
    """
    visual_clean = validate_clean_coefficient(visual_clean, "visual condition clean coefficient")
    audio_clean = validate_clean_coefficient(audio_clean, "audio condition clean coefficient")
    return (
        _augment_condition_group(visual_conditions, generator=generator, clean=visual_clean, device=device),
        _augment_condition_group(audio_conditions, generator=generator, clean=audio_clean, device=device),
    )


@torch.no_grad()
def sample_joint_av(
    transformer,
    *,
    layout: H3PackedLayout,
    text_hidden_states: torch.Tensor,
    text_token_tags: torch.Tensor,
    initial_video: torch.Tensor,
    initial_audio: torch.Tensor,
    steps: int,
    video_shift: float = DEFAULT_VIDEO_SHIFT,
    audio_shift: float = DEFAULT_AUDIO_SHIFT,
    visual_condition_latents: Sequence[torch.Tensor] = (),
    audio_condition_latents: Sequence[torch.Tensor] = (),
    visual_condition_clean: float = DEFAULT_VISUAL_CONDITION_CLEAN,
    audio_condition_clean: float = DEFAULT_AUDIO_CONDITION_CLEAN,
    step_callback: Callable[[int, int], None] | None = None,
    x0_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> H3SampleResult:
    if initial_video.ndim != 5 or tuple(initial_video.shape[2:]) != (
        layout.target_video.frames,
        layout.target_video.height,
        layout.target_video.width,
    ):
        raise ValueError("MiniMax-H3 initial video noise does not match the packed layout")
    if initial_audio.ndim != 4 or tuple(initial_audio.shape[1:]) != (AUDIO_CHANNELS, STEREO_CHANNELS, layout.target_audio_frames):
        raise ValueError("MiniMax-H3 initial audio noise does not match the packed layout")
    if initial_video.shape[0] != initial_audio.shape[0]:
        raise ValueError("MiniMax-H3 initial video and audio batch sizes differ")
    if initial_video.shape[0] != 1:
        raise ValueError(f"MiniMax-H3 R1 requires batch_size=1, got {initial_video.shape[0]}")
    if text_hidden_states.shape[:2] != (initial_video.shape[0], layout.text_length):
        raise ValueError("MiniMax-H3 text hidden states do not match the sampling layout")
    if text_token_tags.shape != (initial_video.shape[0], layout.text_length):
        raise ValueError("MiniMax-H3 text token tags must preserve the [B,L] axes")

    schedule = build_shifted_schedule(
        steps,
        video_shift=video_shift,
        audio_shift=audio_shift,
        device=initial_video.device,
    )
    video = initial_video
    audio = initial_audio
    for index in range(steps):
        sigma_video = schedule.video[index].to(dtype=torch.float32)
        sigma_audio = schedule.audio[index].to(dtype=torch.float32)
        prediction = transformer(
            video_latents=video,
            audio_latents=audio,
            text_hidden_states=text_hidden_states,
            text_token_tags=text_token_tags,
            layout=layout,
            model_t_video=1.0 - sigma_video,
            model_t_audio=1.0 - sigma_audio,
            visual_condition_latents=visual_condition_latents,
            audio_condition_latents=audio_condition_latents,
            visual_condition_clean=visual_condition_clean,
            audio_condition_clean=audio_condition_clean,
        )
        if prediction.video.shape != video.shape or prediction.audio.shape != audio.shape:
            raise ValueError("MiniMax-H3 transformer predictions do not match the target latent shapes")
        if x0_callback is not None:
            # the model predicts the dataward velocity v = x0 - eps, so the step's clean
            # estimate is x0_hat = x_t + sigma * v
            x0_callback(
                index,
                video + sigma_video.to(video) * prediction.video,
                audio + sigma_audio.to(audio) * prediction.audio,
            )
        video_delta = (schedule.video[index] - schedule.video[index + 1]).to(video)
        audio_delta = (schedule.audio[index] - schedule.audio[index + 1]).to(audio)
        video = video + video_delta * prediction.video
        audio = audio + audio_delta * prediction.audio
        if step_callback is not None:
            step_callback(index + 1, steps)
    return H3SampleResult(video=video, audio=audio)


@torch.no_grad()
def sample_joint_av_latents(
    transformer,
    *,
    layout: H3PackedLayout,
    seed: int,
    text_hidden_states: torch.Tensor,
    text_token_tags: torch.Tensor,
    visual_conditions: Sequence[torch.Tensor] = (),
    audio_conditions: Sequence[torch.Tensor] = (),
    steps: int,
    video_shift: float = DEFAULT_VIDEO_SHIFT,
    audio_shift: float = DEFAULT_AUDIO_SHIFT,
    visual_condition_clean: float = DEFAULT_VISUAL_CONDITION_CLEAN,
    audio_condition_clean: float = DEFAULT_AUDIO_CONDITION_CLEAN,
    device: torch.device | str,
    step_callback: Callable[[int, int], None] | None = None,
    x0_callback: Callable[[int, torch.Tensor, torch.Tensor], None] | None = None,
) -> H3SampleResult:
    """One seeded generation on a prepared layout, shared by the generation CLI and the trainer's
    training-time samples: the target noise and the condition augmentation come from the seed's
    single CPU stream (see create_sampling_generator), the text rows run in bf16, and the sampled
    latents are returned on the CPU."""
    generator = create_sampling_generator(seed)
    initial_video, initial_audio = initialize_target_latents(
        video_shape=(1, VIDEO_CHANNELS, layout.target_video.frames, layout.target_video.height, layout.target_video.width),
        audio_shape=(1, AUDIO_CHANNELS, STEREO_CHANNELS, layout.target_audio_frames),
        generator=generator,
        device=device,
        video_dtype=torch.float32,
        audio_dtype=torch.float32,
    )
    visual_conditions, audio_conditions = augment_condition_latents(
        visual_conditions,
        audio_conditions,
        generator=generator,
        visual_clean=visual_condition_clean,
        audio_clean=audio_condition_clean,
        device=device,
    )
    sample = sample_joint_av(
        transformer,
        layout=layout,
        text_hidden_states=text_hidden_states.to(device=device, dtype=torch.bfloat16),
        text_token_tags=text_token_tags.to(device),
        initial_video=initial_video,
        initial_audio=initial_audio,
        steps=steps,
        video_shift=video_shift,
        audio_shift=audio_shift,
        visual_condition_latents=visual_conditions,
        audio_condition_latents=audio_conditions,
        visual_condition_clean=visual_condition_clean,
        audio_condition_clean=audio_condition_clean,
        step_callback=step_callback,
        x0_callback=x0_callback,
    )
    return H3SampleResult(video=sample.video.detach().cpu(), audio=sample.audio.detach().cpu())


def synchronize_decoded_av(
    decoded_video: torch.Tensor,
    decoded_audio: torch.Tensor,
    *,
    frame_count: int,
    fps: int = TARGET_FPS,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> H3DecodedAV:
    if frame_count <= 0 or fps <= 0 or sample_rate <= 0:
        raise ValueError("MiniMax-H3 decode frame count, fps, and sample rate must be positive")
    if decoded_video.ndim != 5 or decoded_video.shape[0] != 1 or decoded_video.shape[1] != 3:
        raise ValueError(f"MiniMax-H3 video VAE must decode [1,3,F,H,W], got {tuple(decoded_video.shape)}")
    if decoded_audio.ndim != 3 or decoded_audio.shape[:2] != (1, 2):
        raise ValueError(f"MiniMax-H3 audio VAE must decode [1,2,L], got {tuple(decoded_audio.shape)}")

    planned_audio_samples = round(Fraction(frame_count * sample_rate, fps))
    video_audio_samples = round(Fraction(decoded_video.shape[2] * sample_rate, fps))
    common_audio_samples = min(decoded_audio.shape[-1], planned_audio_samples, video_audio_samples)
    if common_audio_samples <= 0:
        raise ValueError("MiniMax-H3 decoded audio/video duration is empty")
    common_video_frames = min(
        frame_count,
        decoded_video.shape[2],
        max(1, round(Fraction(common_audio_samples * fps, sample_rate))),
    )

    video = decoded_video_to_uint8(decoded_video, frame_limit=common_video_frames)
    audio = decoded_audio[0, :, :common_audio_samples].detach().cpu().float().clamp(-1.0, 1.0).contiguous()
    return H3DecodedAV(video=video, audio=audio, fps=fps, sample_rate=sample_rate)


def decoded_video_to_uint8(decoded_video: torch.Tensor, *, frame_limit: int) -> torch.Tensor:
    """Convert a [1,3,F,H,W] video in [-1,1] to uint8 [F,H,W,3], trimmed to frame_limit frames."""
    if decoded_video.ndim != 5 or decoded_video.shape[0] != 1 or decoded_video.shape[1] != 3:
        raise ValueError(f"MiniMax-H3 decoded video must be [1,3,F,H,W], got {tuple(decoded_video.shape)}")
    if frame_limit <= 0:
        raise ValueError(f"MiniMax-H3 decoded video frame limit must be positive, got {frame_limit}")
    decoded_video = decoded_video[0, :, :frame_limit].detach().cpu()
    video_chunks = []
    for chunk in decoded_video.split(16, dim=1):
        chunk = chunk.float().clamp(-1.0, 1.0)
        video_chunks.append(((chunk + 1.0) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0))
    return torch.cat(video_chunks).contiguous()


def write_image(frame: torch.Tensor, output_path: str | Path) -> None:
    """Write one uint8 [H,W,3] frame as an image file; one-frame generation outputs."""
    if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != torch.uint8:
        raise ValueError(f"MiniMax-H3 image write needs uint8 [H,W,3], got {tuple(frame.shape)} {frame.dtype}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.cpu().numpy()).save(output_path)


def write_image_sequence(video: torch.Tensor, output_dir: str | Path) -> None:
    """Write uint8 [F,H,W,3] frames as zero-padded numbered PNGs into a directory."""
    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != torch.uint8:
        raise ValueError(f"MiniMax-H3 image-sequence write needs uint8 [F,H,W,3], got {tuple(video.shape)} {video.dtype}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(video):
        Image.fromarray(frame.cpu().numpy()).save(output_dir / f"{index:05d}.png")


def write_audio_wav(audio: torch.Tensor, output_path: str | Path, *, sample_rate: int) -> None:
    """Write stereo [-1,1] float [2,L] audio as a 16-bit PCM WAV; the audio track of image-sequence outputs."""
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError(f"MiniMax-H3 WAV write needs stereo [2,L], got {tuple(audio.shape)}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    interleaved = (audio.detach().cpu().float().clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).t().contiguous()
    with av.open(str(output_path), mode="w") as container:
        audio_stream = container.add_stream("pcm_s16le", rate=sample_rate)
        audio_stream.layout = "stereo"
        for start in range(0, interleaved.shape[0], 1024):
            chunk = interleaved[start : start + 1024]
            frame = av.AudioFrame.from_ndarray(chunk.reshape(1, -1).numpy(), format="s16", layout="stereo")
            frame.sample_rate = sample_rate
            frame.pts = start
            frame.time_base = Fraction(1, sample_rate)
            for packet in audio_stream.encode(frame):
                container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)


# Without an explicit rate control, PyAV encodes libx264 at its ~1 Mbps ABR default — far too
# low for 1 MP/24 fps outputs and enough to masquerade as generation artifacts (mushy lines).
# CRF keeps quality resolution- and content-independent; 16 is evaluation-grade. The silent
# trajectory dumps pass the same value to the shared save_videos_grid.
H3_VIDEO_CRF = 16


def mux_audio_video(
    video: torch.Tensor,
    audio: torch.Tensor,
    output_path: str | Path,
    *,
    fps: int,
    sample_rate: int,
) -> None:
    if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != torch.uint8:
        raise ValueError(f"MiniMax-H3 mux video must be uint8 [F,H,W,3], got {tuple(video.shape)} {video.dtype}")
    if audio.ndim != 2 or audio.shape[0] != 2:
        raise ValueError(f"MiniMax-H3 mux audio must be stereo [2,L], got {tuple(audio.shape)}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(output_path), mode="w") as container:
        video_stream = container.add_stream("libx264", rate=fps, options={"crf": str(H3_VIDEO_CRF)})
        video_stream.width = video.shape[2]
        video_stream.height = video.shape[1]
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("aac", rate=sample_rate)
        audio_stream.layout = "stereo"

        for pixels in video:
            frame = av.VideoFrame.from_ndarray(pixels.numpy(), format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)
        for packet in video_stream.encode():
            container.mux(packet)

        for start in range(0, audio.shape[1], 1024):
            samples = audio[:, start : start + 1024].numpy()
            frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
            frame.sample_rate = sample_rate
            frame.pts = start
            frame.time_base = Fraction(1, sample_rate)
            for packet in audio_stream.encode(frame):
                container.mux(packet)
        for packet in audio_stream.encode():
            container.mux(packet)


def write_joint_av(decoded: H3DecodedAV, output_path: str | Path) -> None:
    mux_audio_video(decoded.video, decoded.audio, Path(output_path), fps=decoded.fps, sample_rate=decoded.sample_rate)
