from __future__ import annotations

import json
import os
from typing import Optional, TYPE_CHECKING

import torch
from PIL import Image

from musubi_tuner.dataset.audio_utils import AudioSource, AudioSpec, decode_audio, resolve_audio_source
from musubi_tuner.dataset.media_utils import glob_images, glob_videos, load_video, VIDEO_EXTENSIONS

if TYPE_CHECKING:
    from musubi_tuner.dataset.bucket import BucketSelector

import logging

logger = logging.getLogger(__name__)


def _numbered_suffix_index(path_without_extension: str, prefix: str) -> int | None:
    marker = prefix + "_"
    if not path_without_extension.startswith(marker):
        return None
    suffix = path_without_extension[len(marker) :]
    return int(suffix) if suffix.isdigit() else None


class ContentDatasource:
    def __init__(self):
        self.caption_only = False  # set to True to only fetch caption for Text Encoder caching
        self.has_control = False

    def set_caption_only(self, caption_only: bool):
        self.caption_only = caption_only

    def is_indexable(self):
        return False

    def get_caption(self, idx: int) -> tuple[str, str]:
        """
        Returns caption. May not be called if is_indexable() returns False.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class ImageDatasource(ContentDatasource):
    def __init__(self):
        super().__init__()

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, list[Image.Image]]:
        """
        Returns image data as a tuple of image path, image, and caption for the given index.
        Key must be unique and valid as a file name.
        May not be called if is_indexable() returns False.
        """
        raise NotImplementedError

    def get_media_paths(self, image_path: str) -> tuple[list[str], list[str]]:
        """Return target and control paths contributing to one image sample."""
        raise NotImplementedError


class ImageDirectoryDatasource(ImageDatasource):
    def __init__(
        self,
        image_directory: str,
        caption_extension: Optional[str] = None,
        control_directory: Optional[str] = None,
        control_count_per_image: Optional[int] = None,
        multiple_target: bool = False,
    ):
        super().__init__()
        self.image_directory = image_directory
        self.caption_extension = caption_extension
        self.control_directory = control_directory
        self.control_count_per_image = control_count_per_image
        self.multiple_target = multiple_target
        self.current_idx = 0

        # glob images
        logger.info(f"glob images in {self.image_directory}")
        self.image_paths = glob_images(self.image_directory, caption_extension=self.caption_extension)
        self.caption_paths = {
            image_path: os.path.splitext(image_path)[0] + self.caption_extension
            for image_path in self.image_paths
            if self.caption_extension
        }
        self.caption_alias_paths: set[str] = set()
        if self.multiple_target and self.caption_extension:
            all_image_paths = glob_images(self.image_directory)
            image_paths = set(self.image_paths)
            for image_path in all_image_paths:
                image_path_no_ext = os.path.splitext(image_path)[0]
                suffix = image_path_no_ext.rsplit("_", 1)[-1]
                if suffix != "0" or image_path in image_paths:
                    continue
                base_no_ext = image_path_no_ext[: -(len(suffix) + 1)]
                caption_path = base_no_ext + self.caption_extension
                if os.path.exists(caption_path):
                    self.image_paths.append(image_path)
                    self.caption_paths[image_path] = caption_path
                    self.caption_alias_paths.add(image_path)
                    image_paths.add(image_path)
            self.image_paths.sort()
        logger.info(f"found {len(self.image_paths)} images")

        # check if multiple-target images exist
        self.target_paths: dict[str, list[str]] = {}  # image_path -> list of target image paths

        if self.multiple_target:
            # sort by length, longer first
            sorted_image_paths = sorted(self.image_paths, key=lambda p: len(os.path.basename(p)), reverse=True)

            all_image_paths = set(glob_images(self.image_directory))  # image1.jpg, image1_1.jpg, image1_2.jpg, ...
            multiple_target_candidates = all_image_paths - set(sorted_image_paths)  # those not in the images with captions

            if len(multiple_target_candidates) > 0:
                logger.info("checking for multiple-target images")
                for image_path in sorted_image_paths:
                    image_path_no_ext = os.path.splitext(image_path)[0]
                    match_prefix = image_path_no_ext
                    if image_path in self.caption_alias_paths:
                        match_prefix = os.path.splitext(self.caption_paths[image_path])[0]

                    # find matching multiple-target images
                    indexed_paths = [
                        (index, path)
                        for path in multiple_target_candidates
                        if (index := _numbered_suffix_index(os.path.splitext(path)[0], match_prefix)) is not None
                    ]

                    if indexed_paths:
                        # sort by the digits (`_0000`) suffix
                        indexed_paths.sort()
                        potential_paths = [path for _index, path in indexed_paths]
                        self.target_paths[image_path] = potential_paths

                        # remove to avoid duplicate matching
                        multiple_target_candidates.difference_update(potential_paths)

                # check the number of targets: all multiple-target images should have the same number of targets
                num_targets = 0
                for image_path, paths in self.target_paths.items():
                    if num_targets == 0:
                        num_targets = len(paths)
                    elif num_targets != len(paths):
                        logger.error(
                            f"All multiple-target images must have the same number of targets / 全ての複数ターゲット画像は同じ数のターゲットを持つ必要があります: {image_path}"
                        )
                        raise ValueError(
                            f"All multiple-target images must have the same number of targets / 全ての複数ターゲット画像は同じ数のターゲットを持つ必要があります: {image_path}"
                        )

                if num_targets == 0:
                    logger.error("no multiple-target images found, but multiple_target is set to True")
                    raise ValueError("no multiple-target images found, but multiple_target is set to True")

                logger.info(f"found multiple-target images, max targets per image: {num_targets}")

        # glob control images if specified
        if self.control_directory is not None:
            logger.info(f"glob control images in {self.control_directory}")
            self.has_control = True
            self.control_paths = {}

            # sort image paths for matching control images properly: longer names first
            image_paths_sorted = sorted(self.image_paths, key=lambda p: len(os.path.basename(p)), reverse=True)

            # glob control images first
            all_control_image_paths = set(glob_images(self.control_directory))

            for image_path in image_paths_sorted:
                image_basename = os.path.basename(image_path)
                image_basename_no_ext = os.path.splitext(image_basename)[0]
                control_match_basename = image_basename_no_ext
                if image_path in self.caption_alias_paths:
                    control_match_basename = os.path.splitext(os.path.basename(self.caption_paths[image_path]))[0]

                # find matching control images
                indexed_paths = []
                for path in all_control_image_paths:
                    basename_no_ext = os.path.splitext(os.path.basename(path))[0]
                    if basename_no_ext == control_match_basename:
                        indexed_paths.append((0, path))
                        continue
                    index = _numbered_suffix_index(basename_no_ext, control_match_basename)
                    if index is not None:
                        indexed_paths.append((index + 1, path))
                indexed_paths.sort()
                potential_paths = [path for _index, path in indexed_paths]

                # remove to avoid duplicate matching
                all_control_image_paths.difference_update(potential_paths)

                if potential_paths:
                    if control_count_per_image is not None and len(potential_paths) < control_count_per_image:
                        logger.error(
                            f"Not enough control images for {image_path}: found {len(potential_paths)}, expected {control_count_per_image}"
                        )
                        raise ValueError(
                            f"Not enough control images for {image_path}: found {len(potential_paths)}, expected {control_count_per_image}"
                        )

                    # take the first `control_count_per_image` paths
                    self.control_paths[image_path] = (
                        potential_paths[:control_count_per_image] if control_count_per_image is not None else potential_paths
                    )
            logger.info(
                f"found {len(self.control_paths)} matching control images for {'arbitrary' if control_count_per_image is None else control_count_per_image} images"
            )

            # log the distribution of number of control images
            count_of_num_control_images = {}
            for paths in self.control_paths.values():
                count = len(paths)
                if count not in count_of_num_control_images:
                    count_of_num_control_images[count] = 0
                count_of_num_control_images[count] += 1
            for count, num_images in count_of_num_control_images.items():
                logger.info(f"  {num_images} images have {count} control images")

            missing_controls = len(self.image_paths) - len(self.control_paths)
            if missing_controls > 0:
                missing_control_paths = set(self.image_paths) - set(self.control_paths.keys())
                logger.error(f"Could not find matching control images for {missing_controls} images: {missing_control_paths}")
                raise ValueError(f"Could not find matching control images for {missing_controls} images")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.image_paths)

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        image_path = self.image_paths[idx]
        image_paths = [image_path]
        if self.multiple_target:
            # load multiple-target images
            image_paths += self.target_paths.get(image_path, [])

        images = []
        for p in image_paths:
            img = Image.open(p)
            if img.mode != "RGB" and img.mode != "RGBA":
                img = img.convert("RGB")
            images.append(img)

        _, caption = self.get_caption(idx)

        controls = None
        if self.has_control:
            controls = []
            for control_path in self.control_paths[image_path]:
                control = Image.open(control_path)
                if control.mode != "RGB" and control.mode != "RGBA":
                    control = control.convert("RGB")
                controls.append(control)

        return image_path, images, caption, controls

    def get_media_paths(self, image_path: str) -> tuple[list[str], list[str]]:
        normalized = os.path.normcase(os.path.abspath(image_path))
        source_path = next(
            (path for path in self.image_paths if os.path.normcase(os.path.abspath(path)) == normalized),
            None,
        )
        if source_path is None:
            raise KeyError(f"Image path is not present in the datasource: {image_path}")
        targets = [source_path, *self.target_paths.get(source_path, [])]
        controls = self.control_paths.get(source_path, []) if self.has_control else []
        return targets, list(controls)

    def get_caption(self, idx: int) -> tuple[str, str]:
        image_path = self.image_paths[idx]
        caption_path = (
            self.caption_paths.get(image_path, os.path.splitext(image_path)[0] + self.caption_extension)
            if self.caption_extension
            else ""
        )
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return image_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        """
        Returns a fetcher function that returns image data.
        """
        if self.current_idx >= len(self.image_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)
        else:

            def create_image_fetcher(index):
                return lambda: self.get_image_data(index)

            fetcher = create_image_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class ImageJsonlDatasource(ImageDatasource):
    def __init__(self, image_jsonl_file: str, control_count_per_image: Optional[int] = None, multiple_target: bool = False):
        super().__init__()
        self.image_jsonl_file = image_jsonl_file
        self.control_count_per_image = control_count_per_image
        self.multiple_target = multiple_target
        self.current_idx = 0

        # load jsonl
        logger.info(f"load image jsonl from {self.image_jsonl_file}")
        self.data = []
        with open(self.image_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.error(f"failed to load json: {line} @ {self.image_jsonl_file}")
                    raise
                self.data.append(data)
        logger.info(f"loaded {len(self.data)} images")

        # Normalize control paths
        for item in self.data:
            if "control_path" in item:
                item["control_path_0"] = item.pop("control_path")

            # Ensure control paths are named consistently, from control_path_0000 to control_path_0, control_path_1, etc.
            control_path_keys = [key for key in item.keys() if key.startswith("control_path_")]
            control_path_keys.sort(key=lambda x: int(x.split("_")[-1]))
            for i, key in enumerate(control_path_keys):
                if key != f"control_path_{i}":
                    item[f"control_path_{i}"] = item.pop(key)

        # Check if there are control paths in the JSONL
        self.has_control = any("control_path_0" in item for item in self.data)
        if self.has_control:
            if self.control_count_per_image is None:
                logger.info(f"found {len(self.data)} images with arbitrary control images per image in JSONL data")
            else:
                missing_control_images = [
                    item["image_path"]
                    for item in self.data
                    if sum(f"control_path_{i}" not in item for i in range(self.control_count_per_image)) > 0
                ]
                if missing_control_images:
                    logger.error(f"Some images do not have control paths in JSONL data: {missing_control_images}")
                    raise ValueError(f"Some images do not have control paths in JSONL data: {missing_control_images}")
                logger.info(
                    f"found {len(self.data)} images with {self.control_count_per_image} control images per image in JSONL data"
                )

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_image_data(self, idx: int) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        data = self.data[idx]
        image_path = data.get("image_path", data.get("image_path_0"))
        image_paths = [image_path]
        if self.multiple_target:
            # load multiple-target images
            while True:
                next_index = len(image_paths)  # start from 1
                next_image_path = data.get("image_path_" + str(next_index), None)
                if next_image_path is None:
                    break
                if not os.path.exists(next_image_path):
                    raise ValueError(f"multiple-target image not found: {next_image_path}")

                image_paths.append(next_image_path)

        images = []
        for path in image_paths:
            img = Image.open(path)
            if img.mode != "RGB" and img.mode != "RGBA":
                img = img.convert("RGB")
            images.append(img)

        caption = data["caption"]

        controls = None
        if self.has_control:
            controls = []
            for i in range(self.control_count_per_image or 1000):  # arbitrary large number if control_count_per_image is None
                if f"control_path_{i}" not in data:
                    break
                control_path = data[f"control_path_{i}"]
                control = Image.open(control_path)
                if control.mode != "RGB" and control.mode != "RGBA":
                    control = control.convert("RGB")
                controls.append(control)

        return image_path, images, caption, controls

    def get_media_paths(self, image_path: str) -> tuple[list[str], list[str]]:
        normalized = os.path.normcase(os.path.abspath(image_path))
        for data in self.data:
            primary = data.get("image_path", data.get("image_path_0"))
            if primary is None or os.path.normcase(os.path.abspath(primary)) != normalized:
                continue
            targets = [primary]
            if self.multiple_target:
                index = 1
                while (path := data.get(f"image_path_{index}")) is not None:
                    targets.append(path)
                    index += 1
            controls = []
            index = 0
            while (path := data.get(f"control_path_{index}")) is not None:
                controls.append(path)
                index += 1
            return targets, controls
        raise KeyError(f"Image path is not present in the datasource: {image_path}")

    def get_caption(self, idx: int) -> tuple[str, str]:
        data = self.data[idx]
        image_path = data.get("image_path", data.get("image_path_0"))
        caption = data["caption"]
        return image_path, caption

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self) -> callable:
        if self.current_idx >= len(self.data):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:

            def create_fetcher(index):
                return lambda: self.get_image_data(index)

            fetcher = create_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class VideoDatasource(ContentDatasource):
    def __init__(self):
        super().__init__()

        # None means all frames
        self.start_frame = None
        self.end_frame = None

        self.bucket_selector = None

        self.source_fps = None
        self.target_fps = None

        # timestamp-based fps normalization (audio-capable architectures need deterministic fps)
        self.strict_target_fps: Optional[float] = None

        # audio support: set via set_audio_spec by audio-capable architectures
        self.audio_spec: Optional[AudioSpec] = None
        self.audio_sources: Optional[list[Optional[AudioSource]]] = None

    def __len__(self):
        raise NotImplementedError

    def get_video_data_from_path(
        self,
        video_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> list[Image.Image]:
        # this method can resize the video if bucket_selector is given to reduce the memory usage

        start_frame = start_frame if start_frame is not None else self.start_frame
        end_frame = end_frame if end_frame is not None else self.end_frame
        bucket_selector = bucket_selector if bucket_selector is not None else self.bucket_selector

        if self.strict_target_fps is not None:
            return load_video(
                video_path,
                start_frame,
                end_frame,
                bucket_selector,
                target_fps=self.strict_target_fps,
                fps_resample_mode="timestamps",
            )

        video = load_video(
            video_path, start_frame, end_frame, bucket_selector, source_fps=self.source_fps, target_fps=self.target_fps
        )
        return video

    def get_control_data_from_path(
        self,
        control_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> list[Image.Image]:
        start_frame = start_frame if start_frame is not None else self.start_frame
        end_frame = end_frame if end_frame is not None else self.end_frame
        bucket_selector = bucket_selector if bucket_selector is not None else self.bucket_selector

        if self.strict_target_fps is not None:
            return load_video(
                control_path,
                start_frame,
                end_frame,
                bucket_selector,
                target_fps=self.strict_target_fps,
                fps_resample_mode="timestamps",
            )

        control = load_video(
            control_path, start_frame, end_frame, bucket_selector, source_fps=self.source_fps, target_fps=self.target_fps
        )
        return control

    def set_start_and_end_frame(self, start_frame: Optional[int], end_frame: Optional[int]):
        self.start_frame = start_frame
        self.end_frame = end_frame

    def set_bucket_selector(self, bucket_selector: BucketSelector):
        self.bucket_selector = bucket_selector

    def set_source_and_target_fps(self, source_fps: Optional[float], target_fps: Optional[float]):
        self.source_fps = source_fps
        self.target_fps = target_fps

    def set_strict_target_fps(self, target_fps: Optional[float]):
        self.strict_target_fps = target_fps

    def set_audio_spec(self, audio_spec: Optional[AudioSpec]):
        """Enables audio for this datasource and eagerly resolves all audio sources (fail-fast)."""
        self.audio_spec = audio_spec
        self.audio_sources = None
        if audio_spec is None:
            return

        audio_sources = []
        missing = []
        for index in range(len(self)):
            video_path, explicit_path = self._audio_resolution_inputs(index)
            source = resolve_audio_source(video_path, explicit_path)
            audio_sources.append(source)
            if source is None:
                missing.append(video_path)
        self.audio_sources = audio_sources

        if missing:
            for video_path in missing[:10]:
                logger.warning(f"Video has no audio source; an unsupervised silence placeholder will be cached: {video_path}")
            logger.info(f"audio sources resolved: {len(audio_sources) - len(missing)} with audio, {len(missing)} without")

    def _audio_resolution_inputs(self, idx: int) -> tuple[str, Optional[str]]:
        """Returns (video_path, explicit_audio_path) for audio source resolution."""
        raise NotImplementedError

    def get_audio_waveform(self, idx: int) -> Optional[torch.Tensor]:
        """Decodes the full waveform [C, L] for the item, or None if it has no audio source."""
        if self.audio_spec is None or self.audio_sources is None:
            raise ValueError("Audio is not enabled for this datasource; call set_audio_spec first")
        source = self.audio_sources[idx]
        if source is None:
            return None
        return decode_audio(source, sample_rate=self.audio_spec.sample_rate, channels=self.audio_spec.channels)

    def _create_video_fetcher(self, index: int):
        if self.audio_spec is not None:

            def fetch():
                video_path, video, caption, control = self.get_video_data(index)
                waveform = self.get_audio_waveform(index)
                return video_path, video, caption, control, waveform

        else:

            def fetch():
                return self.get_video_data(index)

        # the datasource record index travels as a fetcher attribute so that ItemInfo can
        # reference the originating record without re-deriving it from item keys
        fetch.datasource_index = index
        return fetch

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class VideoDirectoryDatasource(VideoDatasource):
    def __init__(self, video_directory: str, caption_extension: Optional[str] = None, control_directory: Optional[str] = None):
        super().__init__()
        self.video_directory = video_directory
        self.caption_extension = caption_extension
        self.control_directory = control_directory
        self.current_idx = 0

        # glob videos
        logger.info(f"glob videos in {self.video_directory}")
        self.video_paths = glob_videos(self.video_directory)
        logger.info(f"found {len(self.video_paths)} videos")

        # glob control images if specified
        if self.control_directory is not None:
            logger.info(f"glob control videos in {self.control_directory}")
            self.has_control = True
            self.control_paths = {}
            for video_path in self.video_paths:
                video_basename = os.path.basename(video_path)
                # construct control path from video path
                # for example: video_path = "vid/video.mp4" -> control_path = "control/video.mp4"
                control_path = os.path.join(self.control_directory, video_basename)
                if os.path.exists(control_path):
                    self.control_paths[video_path] = control_path
                else:
                    # use the same base name for control path
                    base_name = os.path.splitext(video_basename)[0]

                    # directory with images. for example: video_path = "vid/video.mp4" -> control_path = "control/video"
                    potential_path = os.path.join(self.control_directory, base_name)  # no extension
                    if os.path.isdir(potential_path):
                        self.control_paths[video_path] = potential_path
                    else:
                        # another extension for control path
                        # for example: video_path = "vid/video.mp4" -> control_path = "control/video.mov"
                        for ext in VIDEO_EXTENSIONS:
                            potential_path = os.path.join(self.control_directory, base_name + ext)
                            if os.path.exists(potential_path):
                                self.control_paths[video_path] = potential_path
                                break

            logger.info(f"found {len(self.control_paths)} matching control videos/images")
            # check if all videos have matching control paths, if not, raise an error
            missing_controls = len(self.video_paths) - len(self.control_paths)
            if missing_controls > 0:
                # logger.warning(f"Could not find matching control videos/images for {missing_controls} videos")
                missing_controls_videos = [video_path for video_path in self.video_paths if video_path not in self.control_paths]
                logger.error(
                    f"Could not find matching control videos/images for {missing_controls} videos: {missing_controls_videos}"
                )
                raise ValueError(f"Could not find matching control videos/images for {missing_controls} videos")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.video_paths)

    def get_video_data(
        self,
        idx: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        video_path = self.video_paths[idx]
        video = self.get_video_data_from_path(video_path, start_frame, end_frame, bucket_selector)

        _, caption = self.get_caption(idx)

        control = None
        if self.control_directory is not None and video_path in self.control_paths:
            control_path = self.control_paths[video_path]
            control = self.get_control_data_from_path(control_path, start_frame, end_frame, bucket_selector)

        return video_path, video, caption, control

    def get_caption(self, idx: int) -> tuple[str, str]:
        video_path = self.video_paths[idx]
        caption_path = os.path.splitext(video_path)[0] + self.caption_extension if self.caption_extension else ""
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.read().strip()
        return video_path, caption

    def _audio_resolution_inputs(self, idx: int) -> tuple[str, Optional[str]]:
        return self.video_paths[idx], None

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.video_paths):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:
            fetcher = self._create_video_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher


class VideoJsonlDatasource(VideoDatasource):
    PATH_KEYS = ("video_path", "control_path", "audio_path")

    def __init__(self, video_jsonl_file: str):
        super().__init__()
        self.video_jsonl_file = video_jsonl_file
        self.current_idx = 0

        # load jsonl
        logger.info(f"load video jsonl from {self.video_jsonl_file}")
        self.data = []
        with open(self.video_jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                self.data.append(data)
        logger.info(f"loaded {len(self.data)} videos")

        # resolve relative paths against the working directory first (the historical behavior),
        # then against the JSONL's own directory. Whichever location matches is rewritten to an
        # absolute path so downstream consumers (e.g. MiniMax-H3 record building) never
        # re-resolve the value against a different base.
        base_directory = os.path.dirname(os.path.abspath(self.video_jsonl_file))
        for data in self.data:
            for key in VideoJsonlDatasource.PATH_KEYS:
                value = data.get(key)
                if not isinstance(value, str) or not value or os.path.isabs(value):
                    continue
                jsonl_candidate = os.path.join(base_directory, value)
                if os.path.exists(value):
                    data[key] = os.path.abspath(value)
                    if os.path.exists(jsonl_candidate) and not os.path.samefile(value, jsonl_candidate):
                        logger.warning(
                            f"{key} {value!r} exists both relative to the working directory and to the JSONL directory; "
                            f"using the working-directory match {data[key]}"
                        )
                elif os.path.exists(jsonl_candidate):
                    data[key] = jsonl_candidate

        # Check if there are control paths in the JSONL
        self.has_control = any("control_path" in item for item in self.data)
        if self.has_control:
            control_count = sum(1 for item in self.data if "control_path" in item)
            if control_count < len(self.data):
                missing_control_videos = [item["video_path"] for item in self.data if "control_path" not in item]
                logger.error(f"Some videos do not have control paths in JSONL data: {missing_control_videos}")
                raise ValueError(f"Some videos do not have control paths in JSONL data: {missing_control_videos}")
            logger.info(f"found {control_count} control videos/images in JSONL data")

    def is_indexable(self):
        return True

    def __len__(self):
        return len(self.data)

    def get_video_data(
        self,
        idx: int,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        bucket_selector: Optional[BucketSelector] = None,
    ) -> tuple[str, list[Image.Image], str, Optional[list[Image.Image]]]:
        data = self.data[idx]
        video_path = data["video_path"]
        video = self.get_video_data_from_path(video_path, start_frame, end_frame, bucket_selector)

        caption = data["caption"]

        control = None
        if "control_path" in data and data["control_path"]:
            control_path = data["control_path"]
            control = self.get_control_data_from_path(control_path, start_frame, end_frame, bucket_selector)

        return video_path, video, caption, control

    def get_caption(self, idx: int) -> tuple[str, str]:
        data = self.data[idx]
        video_path = data["video_path"]
        caption = data["caption"]
        return video_path, caption

    def _audio_resolution_inputs(self, idx: int) -> tuple[str, Optional[str]]:
        data = self.data[idx]
        return data["video_path"], data.get("audio_path")

    def __iter__(self):
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= len(self.data):
            raise StopIteration

        if self.caption_only:

            def create_caption_fetcher(index):
                return lambda: self.get_caption(index)

            fetcher = create_caption_fetcher(self.current_idx)

        else:
            fetcher = self._create_video_fetcher(self.current_idx)

        self.current_idx += 1
        return fetcher
