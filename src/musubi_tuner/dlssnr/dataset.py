"""Lazy paired clips with shared buckets and synchronized spatial transforms."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from musubi_tuner.dataset.architectures import ARCHITECTURE_DLSSNR
from musubi_tuner.dataset.bucket import BucketBatchManager, BucketSelector
from musubi_tuner.dlssnr.filenames import validate_filename
from musubi_tuner.dlssnr.controls import fixed_control_tensor, resolve_fixed_controls
from musubi_tuner.dlssnr.geometry import resolve_geometry
from musubi_tuner.dlssnr.identity import file_sha256, json_sha256


def _numeric(path: Path) -> torch.Tensor:
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray) or array.dtype.kind not in "biuf":
        raise ValueError(f"{path} must be a real numeric npy array")
    tensor = torch.from_numpy(np.array(array, dtype=np.float32, copy=True))
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{path} contains non-finite values")
    return tensor


def _proxy_image(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        tensor = _numeric(path)
    else:
        with Image.open(path) as image:
            if image.mode not in ("RGB", "RGBA", "L"):
                raise ValueError(f"{path}: unsupported image precision/mode {image.mode}; use FP32 CHW npy")
            array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if tensor.ndim != 3 or tensor.shape[0] != 3 or not ((tensor >= 0) & (tensor <= 1)).all():
        raise ValueError(f"{path} must be [3,H,W] srgb_proxy in [0,1]")
    return tensor


def _proxy_size(path: Path) -> tuple[int, int]:
    if path.suffix.lower() == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(array, np.ndarray) or array.ndim != 3 or array.shape[0] != 3 or array.dtype.kind not in "biuf":
            raise ValueError(f"{path} must be a real numeric [3,H,W] npy array")
        if min(array.shape[1:]) < 1:
            raise ValueError(f"{path} must have positive source dimensions")
        return int(array.shape[2]), int(array.shape[1])
    with Image.open(path) as image:
        return image.size


def _resize_paired_frame(frame, bucket):
    width, height = frame["source"].shape[-1], frame["source"].shape[-2]
    bucket_width, bucket_height = bucket
    scale = max(bucket_width / width, bucket_height / height)
    resized_width, resized_height = int(width * scale + 0.5), int(height * scale + 0.5)
    left, top = (resized_width - bucket_width) // 2, (resized_height - bucket_height) // 2
    for name in ("source", "target", "controls", "motion", "history_valid", "temporal_valid", "loss_mask"):
        value = frame[name]
        if value is None:
            continue
        binary = value.dtype == torch.bool
        if (resized_height, resized_width) != (height, width):
            options = (
                {"mode": "nearest-exact"} if binary else {"mode": "bilinear", "align_corners": False, "antialias": name != "motion"}
            )
            value = F.interpolate(value.float().unsqueeze(0), size=(resized_height, resized_width), **options)[0]
        value = value[:, top : top + bucket_height, left : left + bucket_width].contiguous()
        if name == "motion":
            # Same crop on both frames cancels; displacements use actual rounded resize ratios.
            value = value * value.new_tensor([resized_width / width, resized_height / height])[:, None, None]
        frame[name] = value.bool() if binary else value
        if not torch.isfinite(frame[name]).all():
            raise ValueError(f"frame {frame['frame_index']}: {name} contains non-finite values after bucket transform")
        if name in ("source", "target", "loss_mask"):
            # Bilinear accumulation can round a bounded value a few ulps past 1.
            frame[name] = frame[name].clamp(0, 1)
    return frame


def _controls(path: Path) -> torch.Tensor:
    tensor = _numeric(path)
    if tensor.ndim != 3 or tensor.shape[0] != 5:
        raise ValueError(f"{path} must be [5,H,W], got {tuple(tensor.shape)}")
    return tensor


def _mask(path: Path, height: int, width: int, *, binary: bool) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        tensor = _numeric(path)
    else:
        with Image.open(path) as image:
            if image.mode not in ("1", "L"):
                raise ValueError(f"{path} must be a single-channel mask")
            tensor = torch.from_numpy(np.array(image.convert("L"), dtype=np.float32) / 255.0)[None]
    if tuple(tensor.shape) != (1, height, width):
        raise ValueError(f"{path}: expected mask [1,{height},{width}], got {tuple(tensor.shape)}")
    valid = (tensor == 0) | (tensor == 1) if binary else (tensor >= 0) & (tensor <= 1)
    if not valid.all():
        raise ValueError(f"{path}: mask must be {'binary' if binary else 'in [0,1]'}")
    return tensor.bool() if binary else tensor


class NRDataset(Dataset):
    def __init__(
        self,
        path,
        width,
        height,
        sequence_length,
        *,
        single_frame=False,
        require_target=True,
        require_temporal_mask=True,
        enable_bucket=False,
        bucket_no_upscale=False,
        rows=None,
        fixed_controls=None,
    ):
        self.path = Path(path).resolve()
        self.width, self.height = width, height
        self.single_frame = single_frame
        self.require_target = require_target
        self.require_temporal_mask = require_temporal_mask
        self.enable_bucket = enable_bucket
        self.bucket_no_upscale = bucket_no_upscale if enable_bucket else False
        self.fixed_controls = resolve_fixed_controls(fixed_controls) if fixed_controls is not None else None
        self.bucket_sizes = []
        self.original_sizes = []
        selector = BucketSelector((width, height), True, self.bucket_no_upscale, ARCHITECTURE_DLSSNR) if enable_bucket else None
        self.rows = []
        self.sequence_ids = set()
        sample_ids = set()
        self.files = {self.path} if rows is None else set()
        source_rows = (
            (
                (number, json.loads(line))
                for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1)
                if line.strip()
            )
            if rows is None
            else enumerate(rows, 1)
        )
        for line_number, row in source_rows:
            row = copy.deepcopy(row)
            context = f"{self.path}:{line_number}"
            if not isinstance(row, dict):
                raise ValueError(f"{context}: sample must be an object")
            if self.fixed_controls is not None:
                row.setdefault("controls_encoding", "dlssnr_lanes_10_14_v1")
            if row.get("schema") != "dlssnr_pairs_v1":
                raise ValueError(f"{context}: schema must be dlssnr_pairs_v1")
            for key, value in (("source_encoding", "srgb_proxy"), ("controls_encoding", "dlssnr_lanes_10_14_v1")):
                if row.get(key) != value:
                    raise ValueError(f"{context}: {key} must be {value}")
            if require_target and row.get("target_encoding") != "srgb_proxy":
                raise ValueError(f"{context}: target_encoding must be srgb_proxy")
            sample_id = row.get("sample_id")
            validate_filename(sample_id, f"{context}: sample_id")
            # Enforce case-insensitive output uniqueness even on case-sensitive hosts.
            sample_key = sample_id.upper().casefold()
            if sample_key in sample_ids:
                raise ValueError(f"{context}: sample_id {sample_id!r} collides with another filename (case-insensitive)")
            sample_ids.add(sample_key)
            sequence_id = row.get("sequence_id")
            if not isinstance(sequence_id, str) or not sequence_id:
                raise ValueError(f"{context}: sequence_id is required")
            self.sequence_ids.add(sequence_id)
            crop_id = row.get("crop_id", 0)
            if type(crop_id) is not int or crop_id < 0:
                raise ValueError(f"{context}: crop_id must be a nonnegative integer")
            frames = row.get("frames", [])
            if not isinstance(frames, list) or not frames or (sequence_length is not None and len(frames) != sequence_length):
                raise ValueError(f"{context}: sequence_length requires {sequence_length} frames")
            previous_index = -1
            for index, frame in enumerate(frames):
                frame_index = frame.get("frame_index")
                if type(frame_index) is not int or frame_index <= previous_index:
                    raise ValueError(f"{context}: frame_index must be nonnegative and strictly increasing")
                previous_index = frame_index
                reset = frame.get("reset")
                if type(reset) is not bool or (index == 0 and not reset):
                    raise ValueError(f"{context}: reset must be boolean and the first frame must reset")
                required = {"input_path"}
                if self.fixed_controls is None:
                    required.add("controls_path")
                else:
                    frame.pop("controls_path", None)
                if require_target:
                    required.add("target_path")
                if not reset:
                    required.update(("motion_path", "history_valid_path"))
                    if row.get("motion_layout") not in ("chw", "hwc"):
                        raise ValueError(f"{context}: motion_layout must be chw or hwc")
                    if require_temporal_mask:
                        required.add("temporal_valid_path")
                if missing := required - frame.keys():
                    raise ValueError(f"{context}: missing {sorted(missing)}")
                for key in (
                    "input_path",
                    "target_path",
                    "controls_path",
                    "motion_path",
                    "history_valid_path",
                    "temporal_valid_path",
                    "loss_mask_path",
                ):
                    if key in frame:
                        if not isinstance(frame[key], str) or not frame[key]:
                            raise ValueError(f"{context}: {key} must be a path")
                        file = (self.path.parent / frame[key]).resolve()
                        if not file.is_file():
                            raise FileNotFoundError(f"{context}: missing {file}")
                        frame[key] = str(file)
                        self.files.add(file)
            if selector is not None:
                sizes = {_proxy_size(Path(frame["input_path"])) for frame in frames}
                if len(sizes) != 1:
                    raise ValueError(f"{context}: all frames in a clip must have the same source resolution")
                original_size = sizes.pop()
                bucket = selector.get_bucket_resolution(original_size)
                resolve_geometry(*bucket)
            else:
                original_size, bucket = None, (width, height)
            self.rows.append(row)
            self.original_sizes.append(original_size)
            self.bucket_sizes.append(bucket)
        if not self.rows:
            raise ValueError(f"{self.path} has no samples")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        frames = [self._load_frame(frame, row.get("motion_layout"), index) for frame in row["frames"]]
        metadata = {"sample_id": row["sample_id"], "sequence_id": row["sequence_id"], "crop_id": row.get("crop_id", 0)}
        return {**frames[0], **metadata} if self.single_frame else {"frames": frames, **metadata}

    def _load_frame(self, frame, motion_layout, sample_index):
        source = _proxy_image(Path(frame["input_path"]))
        target = _proxy_image(Path(frame["target_path"])) if "target_path" in frame else None
        controls = (
            _controls(Path(frame["controls_path"]))
            if self.fixed_controls is None
            else fixed_control_tensor(self.fixed_controls, source.shape[-1], source.shape[-2])
        )
        shape = self.original_sizes[sample_index][::-1] if self.enable_bucket else (self.height, self.width)
        if source.shape[-2:] != shape or controls.shape[-2:] != shape or (target is not None and target.shape != source.shape):
            expected = f"bucket {self.width}x{self.height}"
            if self.enable_bucket:
                bucket_width, bucket_height = self.bucket_sizes[sample_index]
                expected = f"original paired grid {shape[1]}x{shape[0]} before bucketing to {bucket_width}x{bucket_height}"
            raise ValueError(f"frame {frame['frame_index']}: source/target/controls do not match {expected}")
        motion = torch.zeros(2, *shape)
        history_valid = torch.zeros(1, *shape, dtype=torch.bool)
        if not frame["reset"]:
            motion = _numeric(Path(frame["motion_path"]))
            expected = (2, *shape) if motion_layout == "chw" else (*shape, 2)
            if tuple(motion.shape) != expected:
                raise ValueError(f"motion must have declared {motion_layout} shape {expected}")
            if motion_layout == "hwc":
                motion = motion.permute(2, 0, 1).contiguous()
            history_valid = _mask(Path(frame["history_valid_path"]), *shape, binary=True)
        temporal_valid = torch.zeros(1, *shape, dtype=torch.bool)
        if not frame["reset"] and "temporal_valid_path" in frame:
            temporal_valid = _mask(Path(frame["temporal_valid_path"]), *shape, binary=True)
        loss_mask = (
            _mask(Path(frame["loss_mask_path"]), *shape, binary=False) if "loss_mask_path" in frame else torch.ones(1, *shape)
        )
        loaded = {
            "source": source,
            "target": target,
            "controls": controls,
            "motion": motion,
            "history_valid": history_valid,
            "temporal_valid": temporal_valid,
            "loss_mask": loss_mask,
            "reset": frame["reset"],
            "frame_index": frame["frame_index"],
        }
        if self.enable_bucket:
            loaded = _resize_paired_frame(loaded, self.bucket_sizes[sample_index])
        if self.require_target and not loaded["loss_mask"].any():
            raise ValueError(f"frame {frame['frame_index']}: loss mask has no supervised pixels")
        return loaded

    def validate(self):
        for index in range(len(self)):
            for _ in self.iter_frames(index):
                pass

    def iter_frames(self, index):
        row = self.rows[index]
        for frame in row["frames"]:
            yield self._load_frame(frame, row.get("motion_layout"), index)

    def fingerprint(self):
        return json_sha256(
            {
                "resolution": [self.width, self.height],
                "enable_bucket": self.enable_bucket,
                "bucket_no_upscale": self.bucket_no_upscale,
                "bucket_sizes": self.bucket_sizes,
                "original_sizes": self.original_sizes,
                "fixed_controls": self.fixed_controls,
                "rows": self.rows,
                "files": {str(path): file_sha256(path) for path in sorted(self.files)},
            }
        )


class NRBatchPlan:
    """Deterministic same-bucket batches, including partial tails without duplication."""

    def __init__(self, dataset, batch_size):
        group_indices = getattr(dataset, "group_indices", [0] * len(dataset))
        batch_sizes = getattr(dataset, "batch_sizes", [batch_size])
        groups = [{} for _ in batch_sizes]
        for index, (bucket, row) in enumerate(zip(dataset.bucket_sizes, dataset.rows)):
            groups[group_indices[index]].setdefault((*bucket, len(row["frames"])), []).append(index)
        self.managers = [BucketBatchManager(buckets, size) for buckets, size in zip(groups, batch_sizes)]
        self.manager = self.managers[0]
        self.batches = [manager.get_batch_items(index) for manager in self.managers for index in range(len(manager))]
        self.prefix_counts = [0]
        for batch in self.batches:
            self.prefix_counts.append(self.prefix_counts[-1] + len(batch))

    def __len__(self):
        return len(self.batches)

    def indices(self, microbatch_index):
        return self.batches[microbatch_index % len(self)]

    def show_bucket_info(self):
        for manager in self.managers:
            manager.show_bucket_info()

    def sample_count(self, microbatches):
        epochs, offset = divmod(microbatches, len(self))
        return epochs * self.prefix_counts[-1] + self.prefix_counts[offset]

    def report(self):
        return {
            "policy": "dlssnr_same_bucket_batches_v1",
            "samples_per_epoch": self.prefix_counts[-1],
            "batches_per_epoch": len(self),
            "batch_size": self.manager.batch_size,
            "dataset_batch_sizes": [manager.batch_size for manager in self.managers],
            "order_sha256": json_sha256([self.indices(index) for index in range(len(self))]),
            "buckets": [
                {"resolution": list(key[:2]), "frames": key[2], "samples": len(manager.buckets[key]), "dataset_index": index}
                for index, manager in enumerate(self.managers)
                for key in manager.bucket_resos
            ],
        }


def load_single_frame_manifest(
    path, bucket_width, bucket_height, *, require_target=True, enable_bucket=False, bucket_no_upscale=False, fixed_controls=None
):
    return NRDataset(
        path,
        bucket_width,
        bucket_height,
        1,
        single_frame=True,
        require_target=require_target,
        require_temporal_mask=False,
        enable_bucket=enable_bucket,
        bucket_no_upscale=bucket_no_upscale,
        fixed_controls=fixed_controls,
    )


def load_temporal_manifest(
    path,
    bucket_width,
    bucket_height,
    sequence_length,
    *,
    require_target=True,
    require_temporal_mask=None,
    enable_bucket=False,
    bucket_no_upscale=False,
    fixed_controls=None,
):
    if require_temporal_mask is None:
        require_temporal_mask = require_target
    return NRDataset(
        path,
        bucket_width,
        bucket_height,
        sequence_length,
        require_target=require_target,
        require_temporal_mask=require_temporal_mask,
        enable_bucket=enable_bucket,
        bucket_no_upscale=bucket_no_upscale,
        fixed_controls=fixed_controls,
    )


def load_directory_pairs(config):
    from musubi_tuner.dataset.datasources import ImageDirectoryDatasource

    source = ImageDirectoryDatasource(config["image_directory"], control_directory=config["control_directory"])
    rows = []
    for target in sorted(source.image_paths):
        controls = source.control_paths.get(target, [])
        if len(controls) != 1:
            raise ValueError(f"{target}: paired NR data requires exactly one matching control image, found {len(controls)}")
        rows.append(
            {
                "schema": "dlssnr_pairs_v1",
                "sample_id": Path(target).stem,
                "sequence_id": "directory:" + json_sha256(str(Path(controls[0]).resolve())),
                "source_encoding": "srgb_proxy",
                "target_encoding": "srgb_proxy",
                "frames": [
                    {
                        "frame_index": 0,
                        "reset": True,
                        "input_path": str(Path(controls[0]).resolve()),
                        "target_path": str(Path(target).resolve()),
                    }
                ],
            }
        )
    return NRDataset(
        config["image_directory"],
        *config["bucket_size"],
        1,
        single_frame=True,
        require_temporal_mask=False,
        enable_bucket=config["enable_bucket"],
        bucket_no_upscale=config["bucket_no_upscale"],
        rows=rows,
        fixed_controls=config["fixed_controls"],
    )


class NRDatasetCollection(Dataset):
    """Combine paired sources without merging their batch-size or control policies."""

    def __init__(self, datasets, configs):
        self.datasets = datasets
        self.batch_sizes = [config["batch_size"] for config in configs]
        self.repeats = [config.get("num_repeats", 1) for config in configs]
        self.references, self.rows, self.bucket_sizes, self.group_indices = [], [], [], []
        self.sequence_ids = set().union(*(dataset.sequence_ids for dataset in datasets))
        for group, (dataset, repeats) in enumerate(zip(datasets, self.repeats)):
            for repeat in range(repeats):
                for index, row in enumerate(dataset.rows):
                    self.references.append((group, index))
                    self.rows.append({**row, "sample_id": f"d{group}-r{repeat}-{row['sample_id']}"})
                    self.bucket_sizes.append(dataset.bucket_sizes[index])
                    self.group_indices.append(group)

    def __len__(self):
        return len(self.references)

    def __getitem__(self, index):
        group, local_index = self.references[index]
        return {**self.datasets[group][local_index], "sample_id": self.rows[index]["sample_id"]}

    def validate(self):
        for dataset in self.datasets:
            dataset.validate()

    def fingerprint(self):
        return json_sha256({"sources": [dataset.fingerprint() for dataset in self.datasets], "repeats": self.repeats})


def collate_single_frames(samples):
    batch = {name: torch.stack([sample[name] for sample in samples]) for name in ("source", "target", "controls")}
    batch["loss_mask"] = torch.stack([sample.get("loss_mask", torch.ones_like(sample["source"][:1])) for sample in samples])
    return batch


def collate_clips(clips):
    fields = ("source", "target", "controls", "motion", "history_valid", "temporal_valid", "loss_mask")
    batch = {name: torch.stack([torch.stack([frame[name] for frame in clip["frames"]]) for clip in clips]) for name in fields}
    batch["reset"] = torch.tensor([[frame["reset"] for frame in clip["frames"]] for clip in clips], dtype=torch.bool)
    batch["sample_id"] = [clip["sample_id"] for clip in clips]
    return batch
