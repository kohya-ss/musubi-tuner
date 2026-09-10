"""How the MiniMax-H3 cache scripts see a dataset group.

The shared cache drivers hand the architecture callbacks one batch of ``ItemInfo`` at a time,
while MiniMax-H3 needs per-dataset context for every item: the H3 records built from the
datasource (references, teacher captions), the target audio files behind the audio windows,
and the control image paths behind the control pixels. ``plan_h3_datasets`` builds that
context once per dataset, and the items find theirs through ``ItemInfo.dataset_index`` (the
dataset's position in its ``DatasetGroup``) and ``ItemInfo.datasource_index`` (the record's
position in the datasource). Image and video items are told apart the way every other
architecture does it: image items carry no ``frame_count``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence

from safetensors import safe_open

from musubi_tuner.dataset.audio_utils import AudioSource
from musubi_tuner.dataset.image_video_dataset import BaseDataset, ImageDataset, ItemInfo, VideoDataset
from musubi_tuner.minimax_h3.media import H3Record, H3Task, h3_records_from_datasource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class H3DatasetPlan:
    index: int
    is_image: bool
    # aligned with the datasource indices (ItemInfo.datasource_index)
    records: list[H3Record]
    # video datasets: the target audio source of each record (None = silence placeholder);
    # None when the dataset was built without an audio spec (text caching)
    audio_sources: Optional[Sequence[Optional[AudioSource]]]
    # image datasets: {image path: control image paths in index order}, for cache fingerprints
    control_paths: Mapping[str, Sequence[str]]
    # image datasets: control images without fp_1f_clean_indices are untimed Ref2VA references
    controls_as_references: bool


def validate_h3_dataset(dataset: BaseDataset) -> None:
    # image datasets use control images as time-annotated fl2va conditions (validated in the
    # dataset layer); the shared control-VIDEO fields stay unsupported
    if isinstance(dataset, VideoDataset) and (dataset.control_directory is not None or dataset.has_control):
        raise ValueError("MiniMax-H3 does not use the shared control-video fields")


def validate_h3_image_dataset_task(dataset, task: H3Task, one_frame: bool, record_task: H3Task | None = None) -> None:
    """The one-frame task matrix, shared by both cache scripts: plain images cache as t2va;
    time-annotated control images (fp_1f_clean_indices) require fl2va; control images without
    indices are untimed references and require the records to be built as ref2va (``record_task``,
    the cache task itself or the subject-reference teacher's), like JSONL ``references``."""
    record_task = task if record_task is None else record_task
    if not one_frame:
        raise ValueError("MiniMax-H3 image datasets require --one_frame (experimental one-frame training)")
    if dataset.fp_1f_clean_indices is not None:
        if task != "fl2va":
            raise ValueError(
                "MiniMax-H3 image datasets with time-annotated control images (fp_1f_clean_indices) require --task fl2va"
            )
    elif dataset.has_control:
        if record_task != "ref2va":
            raise ValueError(
                "MiniMax-H3 image datasets with control images and no fp_1f_clean_indices use them as untimed references:"
                " cache with --task ref2va (or --teacher_conditions subject_ref), or add fp_1f_clean_indices for --task fl2va"
            )
    elif task == "fl2va":
        raise ValueError(
            "MiniMax-H3 --task fl2va requires image datasets with control images (plain image datasets cache with --task t2va)"
        )


def plan_h3_datasets(
    datasets: Sequence[BaseDataset],
    *,
    task: H3Task,
    one_frame: bool,
    record_task: H3Task | None = None,
) -> list[H3DatasetPlan]:
    """One plan per dataset of a DatasetGroup, in group order (``ItemInfo.dataset_index``).

    ``record_task`` is the task the records are parsed as when it differs from the cache task
    (the subject-reference teacher reads Ref2VA references for a T2VA student).
    """
    record_task = task if record_task is None else record_task
    plans = []
    for index, dataset in enumerate(datasets):
        if dataset.dataset_index != index:
            raise ValueError(
                f"MiniMax-H3 dataset {index} is not at its DatasetGroup position (dataset_index={dataset.dataset_index})"
            )
        validate_h3_dataset(dataset)
        controls_as_references = False
        control_paths: Mapping[str, Sequence[str]] = {}
        audio_sources = None
        if isinstance(dataset, ImageDataset):
            validate_h3_image_dataset_task(dataset, task, one_frame, record_task)
            control_paths = dataset.datasource.get_control_paths()
            controls_as_references = dataset.fp_1f_clean_indices is None
        elif isinstance(dataset, VideoDataset):
            audio_sources = dataset.datasource.audio_sources
        else:
            raise ValueError("MiniMax-H3 caching accepts only image and video datasets")
        records = h3_records_from_datasource(dataset.datasource, record_task, control_images_as_references=controls_as_references)
        plans.append(
            H3DatasetPlan(
                index=index,
                is_image=isinstance(dataset, ImageDataset),
                records=records,
                audio_sources=audio_sources,
                control_paths=control_paths,
                controls_as_references=controls_as_references,
            )
        )
    return plans


def item_plan(plans: Sequence[H3DatasetPlan], item: ItemInfo) -> H3DatasetPlan:
    """The plan of the dataset an item came from."""
    if item.dataset_index is None or not 0 <= item.dataset_index < len(plans):
        raise ValueError(f"MiniMax-H3 cache item is missing its dataset provenance: {item.item_key}")
    plan = plans[item.dataset_index]
    if plan.is_image != (item.frame_count is None):
        raise ValueError(f"MiniMax-H3 cache item does not match its dataset kind: {item.item_key}")
    return plan


def item_record(plan: H3DatasetPlan, item: ItemInfo) -> H3Record:
    """The H3 record of an item (its datasource record with the H3-specific fields parsed)."""
    return plan.records[item_record_index(item)]


def item_audio_source(plan: H3DatasetPlan, item: ItemInfo) -> Optional[AudioSource]:
    """The target audio file behind a video item's audio window, if any."""
    if plan.audio_sources is None:
        return None
    return plan.audio_sources[item_record_index(item)]


def item_record_index(item: ItemInfo) -> int:
    if item.datasource_index is None:
        raise ValueError(f"MiniMax-H3 cache item is missing its datasource provenance: {item.item_key}")
    return item.datasource_index


def item_crop_start(item: ItemInfo) -> int:
    """The start frame of a video item's crop (target-fps space)."""
    if item.frame_pos is None:
        raise ValueError(f"MiniMax-H3 cache item is missing its crop provenance: {item.item_key}")
    return item.frame_pos


def item_control_paths(plan: H3DatasetPlan, item: ItemInfo, expected_count: int) -> list[Path]:
    """The control image files behind a one-frame item's control pixels, resolved."""
    control_paths = plan.control_paths.get(item.item_key)
    if control_paths is None or len(control_paths) != expected_count:
        raise ValueError(f"MiniMax-H3 fl2va one-frame item is missing its control paths: {item.item_key}")
    return [Path(path).resolve() for path in control_paths]


def cache_metadata_matches(path: str | Path, expected: Mapping[str, str]) -> bool:
    """Whether a cache file carries every expected metadata value (the --skip_existing staleness check)."""
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            actual = handle.metadata() or {}
    except Exception as error:
        logger.warning("Unable to read MiniMax-H3 cache metadata from %s: %s", path, error)
        return False
    return all(actual.get(key) == value for key, value in expected.items())
