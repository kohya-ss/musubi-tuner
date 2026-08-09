import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import wave

import av
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.audio_utils import (
    AudioSource,
    AudioSpec,
    assemble_audio_chunks,
    audio_window_start,
    decode_audio,
    probe_audio,
    resolve_audio_source,
    slice_audio_window,
)
from musubi_tuner.dataset.cache_io import (
    AUDIO_PRESENT_KEY,
    append_audio_present_entry,
    validate_audio_present_entry,
)
from musubi_tuner.dataset.datasources import VideoJsonlDatasource
from musubi_tuner.dataset.image_video_dataset import VideoDataset
from musubi_tuner.dataset.media_utils import load_video, resample_frame_indices
from musubi_tuner.training.audio_loss import (
    add_audio_train_args,
    effective_audio_loss_weights,
)


SAMPLE_RATE = 32000


def _sine_stereo(num_samples: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(num_samples) / sample_rate
    left = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.25 * np.sin(2 * np.pi * 880.0 * t)
    return np.stack([left, right]).astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    data = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    interleaved = np.empty(data.shape[1] * 2, dtype=np.int16)
    interleaved[0::2] = data[0]
    interleaved[1::2] = data[1]
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(interleaved.tobytes())


def _write_video(path: Path, *, fps: int = 24, frames: int = 48, size: int = 64) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            image = np.full((size, size, 3), (index * 4) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_video_with_embedded_audio(
    path: Path, *, fps: int = 24, frames: int = 24, size: int = 64, pts_jitter: tuple[int, ...] = ()
) -> None:
    # audio is interleaved in per-frame chunks like real muxers produce; in containers with a
    # coarse timestamp grid (Matroska: 1 ms) this quantizes chunk timestamps, which decode_audio
    # must tolerate when reassembling the stream. pts_jitter offsets chunk timestamps (cycled
    # per chunk, in samples) the way wall-clock muxers do, without touching the samples.
    samples = (_sine_stereo(SAMPLE_RATE) * 32767.0).astype(np.int16)
    with av.open(str(path), mode="w") as container:
        video_stream = container.add_stream("mpeg4", rate=fps)
        video_stream.width = size
        video_stream.height = size
        video_stream.pix_fmt = "yuv420p"
        audio_stream = container.add_stream("pcm_s16le", rate=SAMPLE_RATE)
        audio_stream.layout = "stereo"

        def mux_audio_chunk(start: int, count: int, jitter: int = 0) -> None:
            chunk = samples[:, start : start + count]
            if chunk.shape[1] == 0:
                return
            interleaved = np.empty((1, chunk.shape[1] * 2), dtype=np.int16)
            interleaved[0, 0::2] = chunk[0]
            interleaved[0, 1::2] = chunk[1]
            audio_frame = av.AudioFrame.from_ndarray(interleaved, format="s16", layout="stereo")
            audio_frame.sample_rate = SAMPLE_RATE
            audio_frame.pts = start + jitter
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

        samples_per_frame = SAMPLE_RATE // fps
        audio_pos = 0
        for index in range(frames):
            image = np.full((size, size, 3), (index * 8) % 256, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in video_stream.encode(frame):
                container.mux(packet)
            mux_audio_chunk(audio_pos, samples_per_frame, pts_jitter[index % len(pts_jitter)] if pts_jitter else 0)
            audio_pos += samples_per_frame
        for packet in video_stream.encode():
            container.mux(packet)
        mux_audio_chunk(audio_pos, samples.shape[1] - audio_pos)
        for packet in audio_stream.encode():
            container.mux(packet)


def _spec(samples_per_frame: int = 1000) -> AudioSpec:
    return AudioSpec(sample_rate=SAMPLE_RATE, channels=2, samples_per_crop=lambda frames: frames * samples_per_frame)


def test_audio_spec_validation():
    with pytest.raises(ValueError, match="channels"):
        AudioSpec(sample_rate=SAMPLE_RATE, channels=3, samples_per_crop=lambda frames: frames)
    with pytest.raises(ValueError, match="sample rate"):
        AudioSpec(sample_rate=0, channels=2, samples_per_crop=lambda frames: frames)


def test_audio_window_start_matches_h3_formula():
    assert audio_window_start(0, 24, SAMPLE_RATE) == 0
    assert audio_window_start(24, 24, SAMPLE_RATE) == SAMPLE_RATE
    assert audio_window_start(1, 24, SAMPLE_RATE) == (SAMPLE_RATE + 12) // 24
    with pytest.raises(ValueError):
        audio_window_start(-1, 24, SAMPLE_RATE)


def test_assemble_audio_chunks_fills_small_gaps_and_trims_overlaps():
    first = torch.ones(2, 10)
    second = torch.full((2, 10), 2.0)

    contiguous = assemble_audio_chunks([(0, first), (10, second)], channels=2)
    assert contiguous.shape == (2, 20)

    gap = assemble_audio_chunks([(0, first), (12, second)], channels=2)
    assert gap.shape == (2, 22)
    assert torch.all(gap[:, 10:12] == 0)

    overlap = assemble_audio_chunks([(0, first), (9, second)], channels=2)
    assert overlap.shape == (2, 19)

    with pytest.raises(ValueError, match="discontinuous"):
        assemble_audio_chunks([(0, first), (15, second)], channels=2)


def test_assemble_audio_chunks_recovers_bounded_pts_jitter():
    chunks = [torch.full((2, 100), float(index)) for index in range(5)]
    jitter = (0, -20, 15, -5, 0)
    stamped = [(index * 100 + jitter[index], chunk) for index, chunk in enumerate(chunks)]

    # recovery disabled (the default): the wobble is a hard error as before
    with pytest.raises(ValueError, match="discontinuous"):
        assemble_audio_chunks(stamped, channels=2)

    recovered = assemble_audio_chunks(stamped, channels=2, pts_jitter_tolerance_samples=50)
    assert torch.equal(recovered, torch.cat(chunks, dim=1))


def test_assemble_audio_chunks_rejects_net_drift_and_excessive_jitter():
    chunks = [torch.ones(2, 100) for _ in range(3)]

    # a real 40-sample gap after the first chunk shifts all later timestamps permanently
    stamped = [(0, chunks[0]), (140, chunks[1]), (240, chunks[2])]
    with pytest.raises(ValueError, match="discontinuous"):
        assemble_audio_chunks(stamped, channels=2, pts_jitter_tolerance_samples=50)

    # oscillation beyond the jitter tolerance is not recovered either
    stamped = [(0, chunks[0]), (40, chunks[1]), (200, chunks[2])]
    with pytest.raises(ValueError, match="discontinuous"):
        assemble_audio_chunks(stamped, channels=2, pts_jitter_tolerance_samples=30)


def test_slice_audio_window_pads_within_tolerance_and_errors_beyond():
    waveform = torch.ones(2, 1000)

    exact = slice_audio_window(waveform, start_sample=0, sample_count=1000)
    assert exact.shape == (2, 1000)

    padded = slice_audio_window(waveform, start_sample=0, sample_count=1100, pad_tolerance=200)
    assert padded.shape == (2, 1100)
    assert torch.all(padded[:, 1000:] == 0)

    with pytest.raises(ValueError, match="materially short"):
        slice_audio_window(waveform, start_sample=0, sample_count=2000, pad_tolerance=200)

    with pytest.raises(ValueError, match="empty"):
        slice_audio_window(waveform, start_sample=1200, sample_count=100, pad_tolerance=200, require_exact=False)


def test_resolve_audio_source_prefers_sidecar_and_rejects_ambiguity(tmp_path: Path):
    video_path = tmp_path / "clip.mp4"
    _write_video(video_path, frames=4)

    assert resolve_audio_source(video_path) is None

    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path, _sine_stereo(SAMPLE_RATE // 4))
    source = resolve_audio_source(video_path)
    assert source == AudioSource(path=wav_path.resolve(), embedded=False)

    (tmp_path / "clip.mp3").write_bytes(b"junk")
    with pytest.raises(ValueError, match="Multiple same-stem audio sidecars"):
        resolve_audio_source(video_path)


def test_resolve_audio_source_explicit_and_embedded(tmp_path: Path):
    video_path = tmp_path / "clip.mp4"
    _write_video(video_path, frames=4)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_audio_source(video_path, tmp_path / "missing.wav")

    embedded_path = tmp_path / "embedded.mkv"
    _write_video_with_embedded_audio(embedded_path)
    assert probe_audio(embedded_path)
    source = resolve_audio_source(embedded_path)
    assert source == AudioSource(path=embedded_path.resolve(), embedded=True)


def test_decode_audio_tolerates_coarse_container_timestamps(tmp_path: Path):
    # Matroska quantizes chunk timestamps to 1 ms (up to 16 samples of jitter at 32 kHz);
    # reassembly must not report a discontinuous stream for interleaved chunked audio
    path = tmp_path / "embedded.mkv"
    _write_video_with_embedded_audio(path)

    waveform = decode_audio(AudioSource(path=path, embedded=True), sample_rate=SAMPLE_RATE, channels=2)

    assert waveform.shape[0] == 2
    assert abs(waveform.shape[1] - SAMPLE_RATE) <= SAMPLE_RATE // 1000  # within one timestamp tick


def test_decode_audio_recovers_wall_clock_pts_jitter(tmp_path: Path):
    # capture-style muxers stamp audio pts from a wall clock: timestamps oscillate around
    # the true sample positions (up to 10 ms here) while the samples stay contiguous
    path = tmp_path / "jitter.mkv"
    _write_video_with_embedded_audio(path, pts_jitter=(0, -320, 320, 0, -160, 160))

    waveform = decode_audio(AudioSource(path=path, embedded=True), sample_rate=SAMPLE_RATE, channels=2)

    samples = _sine_stereo(SAMPLE_RATE)
    assert waveform.shape[0] == 2
    assert abs(waveform.shape[1] - SAMPLE_RATE) <= SAMPLE_RATE // 1000  # within one timestamp tick
    length = min(waveform.shape[1], SAMPLE_RATE)
    assert torch.allclose(waveform[:, :length], torch.from_numpy(samples[:, :length]), atol=1e-3)


def test_decode_audio_roundtrips_wav(tmp_path: Path):
    samples = _sine_stereo(SAMPLE_RATE)
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path, samples)

    waveform = decode_audio(AudioSource(path=wav_path, embedded=False), sample_rate=SAMPLE_RATE, channels=2)
    assert waveform.shape == (2, SAMPLE_RATE)
    assert torch.allclose(waveform, torch.from_numpy(samples), atol=1e-3)


def test_resample_frame_indices_nearest_frame_selection():
    # 30 fps source resampled to 24 fps: nearest-source-frame per target tick
    timestamps = [index / 30 for index in range(21)]
    indices = resample_frame_indices(timestamps, source_frame_duration=1.0 / 30, target_fps=24)
    assert len(indices) == 17  # 0.7 seconds at 24 fps
    assert indices[0] == 0
    assert indices == sorted(indices)
    assert max(indices) <= 20

    # 12 fps source upsampled to 24 fps repeats frames
    timestamps = [index / 12 for index in range(7)]
    indices = resample_frame_indices(timestamps, source_frame_duration=1.0 / 12, target_fps=24)
    assert len(indices) == 14
    assert indices == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]


def test_load_video_timestamps_mode_resamples_to_target_fps(tmp_path: Path):
    video_path = tmp_path / "clip30.mp4"
    _write_video(video_path, fps=30, frames=21, size=64)

    video = load_video(str(video_path), target_fps=24, fps_resample_mode="timestamps")
    assert len(video) == 17  # 0.7 seconds at 24 fps
    assert video[0].shape == (64, 64, 3)

    with pytest.raises(ValueError, match="requires target_fps"):
        load_video(str(video_path), fps_resample_mode="timestamps")
    with pytest.raises(ValueError, match="does not use source_fps"):
        load_video(str(video_path), source_fps=30.0, target_fps=24, fps_resample_mode="timestamps")


def test_jsonl_datasource_resolves_explicit_audio_path(tmp_path: Path):
    video_path = tmp_path / "clip.mp4"
    _write_video(video_path, frames=4)
    wav_path = tmp_path / "narration.wav"
    _write_wav(wav_path, _sine_stereo(SAMPLE_RATE // 4))

    jsonl_path = tmp_path / "data.jsonl"
    record = {"video_path": str(video_path), "caption": "caption", "audio_path": str(wav_path)}
    jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    datasource = VideoJsonlDatasource(str(jsonl_path))
    datasource.set_audio_spec(_spec())
    assert datasource.audio_sources == [AudioSource(path=wav_path.resolve(), embedded=False)]


def test_jsonl_datasource_resolves_relative_paths_cwd_first_then_jsonl_directory(tmp_path: Path, monkeypatch):
    jsonl_dir = tmp_path / "ds"
    working_dir = tmp_path / "cwd"
    jsonl_dir.mkdir()
    working_dir.mkdir()
    monkeypatch.chdir(working_dir)

    _write_video(jsonl_dir / "clip.mp4", frames=4)
    _write_wav(jsonl_dir / "narration.wav", _sine_stereo(SAMPLE_RATE // 4))

    jsonl_path = jsonl_dir / "data.jsonl"
    record = {"video_path": "clip.mp4", "caption": "caption", "audio_path": "narration.wav"}
    jsonl_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    # absent from the working directory, paths fall back to the JSONL's own directory
    datasource = VideoJsonlDatasource(str(jsonl_path))
    assert datasource.data[0]["video_path"] == str(jsonl_dir / "clip.mp4")
    assert datasource.data[0]["audio_path"] == str(jsonl_dir / "narration.wav")

    # a working-directory match wins over the JSONL-directory match
    _write_video(working_dir / "clip.mp4", frames=4)
    datasource = VideoJsonlDatasource(str(jsonl_path))
    assert datasource.data[0]["video_path"] == str(working_dir / "clip.mp4")
    assert datasource.data[0]["audio_path"] == str(jsonl_dir / "narration.wav")

    # nonexistent relative paths are kept as-is
    jsonl_path.write_text(json.dumps({"video_path": "missing.mp4", "caption": "c"}) + "\n", encoding="utf-8")
    datasource = VideoJsonlDatasource(str(jsonl_path))
    assert datasource.data[0]["video_path"] == "missing.mp4"


def _make_video_dataset(directory: Path, audio_spec: AudioSpec) -> VideoDataset:
    return VideoDataset(
        resolution=(64, 64),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        target_frames=[5],
        frame_extraction="head",
        video_directory=str(directory),
        cache_directory=str(directory),
        architecture=ARCHITECTURE_MINIMAX_H3,
        audio_spec=audio_spec,
    )


def test_video_dataset_attaches_audio_window_to_items(tmp_path: Path):
    samples = _sine_stereo(SAMPLE_RATE * 2)
    _write_video(tmp_path / "clip.mp4", fps=24, frames=48)
    _write_wav(tmp_path / "clip.wav", samples)
    (tmp_path / "clip.txt").write_text("caption", encoding="utf-8")

    dataset = _make_video_dataset(tmp_path, _spec())
    batches = list(dataset.retrieve_latent_cache_batches(num_workers=1))

    assert len(batches) == 1
    _, items = batches[0]
    item = items[0]
    assert item.frame_count == 5
    assert item.frame_pos == 0
    assert item.datasource_index == 0
    assert item.audio_present is True
    assert item.audio_content.shape == (2, 5000)
    assert torch.allclose(item.audio_content, torch.from_numpy(samples[:, :5000]), atol=1e-3)


def test_video_dataset_uses_silence_placeholder_when_audio_is_missing(tmp_path: Path):
    _write_video(tmp_path / "clip.mp4", fps=24, frames=48)
    (tmp_path / "clip.txt").write_text("caption", encoding="utf-8")

    dataset = _make_video_dataset(tmp_path, _spec())
    batches = list(dataset.retrieve_latent_cache_batches(num_workers=1))

    item = batches[0][1][0]
    assert item.audio_present is False
    assert item.audio_content.shape == (2, 5000)
    assert torch.all(item.audio_content == 0)


def test_video_dataset_errors_on_materially_short_audio(tmp_path: Path):
    _write_video(tmp_path / "clip.mp4", fps=24, frames=48)
    _write_wav(tmp_path / "clip.wav", _sine_stereo(1000))
    (tmp_path / "clip.txt").write_text("caption", encoding="utf-8")

    dataset = _make_video_dataset(tmp_path, _spec())
    with pytest.raises(ValueError, match="materially short"):
        list(dataset.retrieve_latent_cache_batches(num_workers=1))


def test_add_audio_train_args_defaults():
    parser = argparse.ArgumentParser()
    add_audio_train_args(parser)
    args = parser.parse_args([])
    assert args.video_only is False
    assert args.audio_loss_weight == 1.0


def test_effective_audio_loss_weights_combines_policy_and_presence():
    audio_present = torch.tensor([1.0, 0.0])

    args = SimpleNamespace(video_only=False, audio_loss_weight=0.5)
    assert torch.equal(effective_audio_loss_weights(audio_present, args), torch.tensor([0.5, 0.0]))

    args = SimpleNamespace(video_only=True, audio_loss_weight=0.5)
    assert torch.equal(effective_audio_loss_weights(audio_present, args), torch.tensor([0.0, 0.0]))

    args = SimpleNamespace(video_only=False, audio_loss_weight=-1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        effective_audio_loss_weights(audio_present, args)

    args = SimpleNamespace(video_only=False, audio_loss_weight=1.0)
    with pytest.raises(ValueError, match="exactly 0.0 or 1.0"):
        effective_audio_loss_weights(torch.tensor([0.5]), args)


def test_audio_present_cache_entry_roundtrip():
    sd = {}
    append_audio_present_entry(sd, True)
    assert validate_audio_present_entry(sd) == 1.0

    append_audio_present_entry(sd, False)
    assert validate_audio_present_entry(sd) == 0.0

    sd[AUDIO_PRESENT_KEY] = torch.tensor(0.5, dtype=torch.float32)
    with pytest.raises(ValueError, match="exactly 0.0 or 1.0"):
        validate_audio_present_entry(sd)

    with pytest.raises(ValueError, match="scalar float32"):
        validate_audio_present_entry({})
