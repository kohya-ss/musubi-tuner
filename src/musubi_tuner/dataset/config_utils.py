import argparse
from dataclasses import (
    asdict,
    dataclass,
)
import functools
import random
from textwrap import dedent, indent
import json
from pathlib import Path

from typing import List, Optional, Sequence, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import Synchronized

SharedEpoch = Optional["Synchronized[int]"]


import toml
import voluptuous
from voluptuous import Any, ExactSequence, MultipleInvalid, Object, Schema

from musubi_tuner.dataset.image_video_dataset import DatasetGroup, ImageDataset, VideoDataset

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class BaseDatasetParams:
    resolution: Tuple[int, int] = (960, 544)
    enable_bucket: bool = False
    bucket_no_upscale: bool = False
    caption_extension: Optional[str] = None
    batch_size: int = 1
    num_repeats: int = 1
    cache_directory: Optional[str] = None
    debug_dataset: bool = False
    architecture: str = "no_default"  # short style like "hv" or "wan"


@dataclass
class ImageDatasetParams(BaseDatasetParams):
    image_directory: Optional[str] = None
    image_jsonl_file: Optional[str] = None
    control_directory: Optional[str] = None
    multiple_target: Optional[bool] = False

    # FramePack dependent parameters
    fp_latent_window_size: Optional[int] = 9
    fp_1f_clean_indices: Optional[Sequence[int]] = None
    fp_1f_target_index: Optional[int] = None
    fp_1f_no_post: Optional[bool] = False

    no_resize_control: Optional[bool] = False  # if True, control images are not resized to target resolution
    control_resolution: Optional[Tuple[int, int]] = None  # if set, control images are resized to this resolution


@dataclass
class VideoDatasetParams(BaseDatasetParams):
    video_directory: Optional[str] = None
    video_jsonl_file: Optional[str] = None
    control_directory: Optional[str] = None
    target_frames: Sequence[int] = (1,)
    frame_extraction: Optional[str] = "head"
    frame_stride: Optional[int] = 1
    frame_sample: Optional[int] = 1
    max_frames: Optional[int] = 129
    source_fps: Optional[float] = None

    # FramePack dependent parameters
    fp_latent_window_size: Optional[int] = 9


@dataclass
class DatasetBlueprint:
    is_image_dataset: bool
    params: Union[ImageDatasetParams, VideoDatasetParams]


@dataclass
class DatasetGroupBlueprint:
    datasets: Sequence[DatasetBlueprint]


@dataclass
class Blueprint:
    dataset_group: DatasetGroupBlueprint


class ConfigSanitizer:
    # @curry
    @staticmethod
    def __validate_and_convert_twodim(klass, value: Sequence) -> Tuple:
        Schema(ExactSequence([klass, klass]))(value)
        return tuple(value)

    # @curry
    @staticmethod
    def __validate_and_convert_scalar_or_twodim(klass, value: Union[float, Sequence]) -> Tuple:
        Schema(Any(klass, ExactSequence([klass, klass])))(value)
        try:
            Schema(klass)(value)
            return (value, value)
        except:
            return ConfigSanitizer.__validate_and_convert_twodim(klass, value)

    # datasets schema
    DATASET_ASCENDABLE_SCHEMA = {
        "caption_extension": str,
        "batch_size": int,
        "num_repeats": int,
        "resolution": functools.partial(__validate_and_convert_scalar_or_twodim.__func__, int),
        "enable_bucket": bool,
        "bucket_no_upscale": bool,
    }
    IMAGE_DATASET_DISTINCT_SCHEMA = {
        "image_directory": str,
        "image_jsonl_file": str,
        "cache_directory": str,
        "control_directory": str,
        "multiple_target": bool,
        "fp_latent_window_size": int,
        "fp_1f_clean_indices": [int],
        "fp_1f_target_index": int,
        "fp_1f_no_post": bool,
        "no_resize_control": bool,
        "control_resolution": functools.partial(__validate_and_convert_scalar_or_twodim.__func__, int),
    }
    VIDEO_DATASET_DISTINCT_SCHEMA = {
        "video_directory": str,
        "video_jsonl_file": str,
        "control_directory": str,
        "target_frames": [int],
        "frame_extraction": str,
        "frame_stride": int,
        "frame_sample": int,
        "max_frames": int,
        "cache_directory": str,
        "source_fps": float,
        "fp_latent_window_size": int,
    }

    # options handled by argparse but not handled by user config
    ARGPARSE_SPECIFIC_SCHEMA = {
        "debug_dataset": bool,
    }

    def __init__(self) -> None:
        self.image_dataset_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.IMAGE_DATASET_DISTINCT_SCHEMA,
        )
        self.video_dataset_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
            self.VIDEO_DATASET_DISTINCT_SCHEMA,
        )

        def validate_flex_dataset(dataset_config: dict):
            if "video_directory" in dataset_config or "video_jsonl_file" in dataset_config:
                return Schema(self.video_dataset_schema)(dataset_config)
            else:
                return Schema(self.image_dataset_schema)(dataset_config)

        self.dataset_schema = validate_flex_dataset

        self.general_schema = self.__merge_dict(
            self.DATASET_ASCENDABLE_SCHEMA,
        )
        self.user_config_validator = Schema(
            {
                "general": self.general_schema,
                "datasets": [self.dataset_schema],
            }
        )
        self.argparse_schema = self.__merge_dict(
            self.ARGPARSE_SPECIFIC_SCHEMA,
        )
        self.argparse_config_validator = Schema(Object(self.argparse_schema), extra=voluptuous.ALLOW_EXTRA)

    def sanitize_user_config(self, user_config: dict) -> dict:
        try:
            return self.user_config_validator(user_config)
        except MultipleInvalid:
            # TODO: clarify the error message
            logger.error("Invalid user config / ユーザ設定の形式が正しくないようです")
            raise

    # NOTE: In nature, argument parser result is not needed to be sanitize
    #   However this will help us to detect program bug
    def sanitize_argparse_namespace(self, argparse_namespace: argparse.Namespace) -> argparse.Namespace:
        try:
            return self.argparse_config_validator(argparse_namespace)
        except MultipleInvalid:
            # XXX: this should be a bug
            logger.error(
                "Invalid cmdline parsed arguments. This should be a bug. / コマンドラインのパース結果が正しくないようです。プログラムのバグの可能性が高いです。"
            )
            raise

    # NOTE: value would be overwritten by latter dict if there is already the same key
    @staticmethod
    def __merge_dict(*dict_list: dict) -> dict:
        merged = {}
        for schema in dict_list:
            # merged |= schema
            for k, v in schema.items():
                merged[k] = v
        return merged


class BlueprintGenerator:
    BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME = {}

    def __init__(self, sanitizer: ConfigSanitizer):
        self.sanitizer = sanitizer

    # runtime_params is for parameters which is only configurable on runtime, such as tokenizer
    def generate(self, user_config: dict, argparse_namespace: argparse.Namespace, **runtime_params) -> Blueprint:
        sanitized_user_config = self.sanitizer.sanitize_user_config(user_config)
        sanitized_argparse_namespace = self.sanitizer.sanitize_argparse_namespace(argparse_namespace)

        argparse_config = {k: v for k, v in vars(sanitized_argparse_namespace).items() if v is not None}
        general_config = sanitized_user_config.get("general", {})

        dataset_blueprints = []
        for dataset_config in sanitized_user_config.get("datasets", []):
            is_image_dataset = "image_directory" in dataset_config or "image_jsonl_file" in dataset_config
            if is_image_dataset:
                dataset_params_klass = ImageDatasetParams
            else:
                dataset_params_klass = VideoDatasetParams

            params = self.generate_params_by_fallbacks(
                dataset_params_klass, [dataset_config, general_config, argparse_config, runtime_params]
            )
            dataset_blueprints.append(DatasetBlueprint(is_image_dataset, params))

        dataset_group_blueprint = DatasetGroupBlueprint(dataset_blueprints)

        return Blueprint(dataset_group_blueprint)

    @staticmethod
    def generate_params_by_fallbacks(param_klass, fallbacks: Sequence[dict]):
        name_map = BlueprintGenerator.BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME
        search_value = BlueprintGenerator.search_value
        default_params = asdict(param_klass())
        param_names = default_params.keys()

        params = {name: search_value(name_map.get(name, name), fallbacks, default_params.get(name)) for name in param_names}

        return param_klass(**params)

    @staticmethod
    def search_value(key: str, fallbacks: Sequence[dict], default_value=None):
        for cand in fallbacks:
            value = cand.get(key)
            if value is not None:
                return value

        return default_value


# if training is True, it will return a dataset group for training, otherwise for caching
def generate_dataset_group_by_blueprint(
    dataset_group_blueprint: DatasetGroupBlueprint,
    training: bool = False,
    num_timestep_buckets: Optional[int] = None,
    shared_epoch: SharedEpoch = None,
) -> DatasetGroup:
    datasets: List[Union[ImageDataset, VideoDataset]] = []

    for dataset_blueprint in dataset_group_blueprint.datasets:
        if dataset_blueprint.is_image_dataset:
            dataset_klass = ImageDataset
        else:
            dataset_klass = VideoDataset

        dataset = dataset_klass(**asdict(dataset_blueprint.params))
        datasets.append(dataset)

    # assertion
    cache_directories = [dataset.cache_directory for dataset in datasets]
    num_of_unique_cache_directories = len(set(cache_directories))
    if num_of_unique_cache_directories != len(cache_directories):
        raise ValueError(
            "cache directory should be unique for each dataset (note that cache directory is image/video directory if not specified)"
            + " / cache directory は各データセットごとに異なる必要があります（指定されていない場合はimage/video directoryが使われるので注意）"
        )

    # print info
    info = ""
    for i, dataset in enumerate(datasets):
        is_image_dataset = isinstance(dataset, ImageDataset)
        info += dedent(
            f"""\
      [Dataset {i}]
        is_image_dataset: {is_image_dataset}
        resolution: {dataset.resolution}
        batch_size: {dataset.batch_size}
        num_repeats: {dataset.num_repeats}
        caption_extension: "{dataset.caption_extension}"
        enable_bucket: {dataset.enable_bucket}
        bucket_no_upscale: {dataset.bucket_no_upscale}
        cache_directory: "{dataset.cache_directory}"
        debug_dataset: {dataset.debug_dataset}
    """
        )

        if is_image_dataset:
            info += indent(
                dedent(
                    f"""\
        image_directory: "{dataset.image_directory}"
        image_jsonl_file: "{dataset.image_jsonl_file}"
        control_directory: "{dataset.control_directory}"
        multiple_target: {dataset.multiple_target}
        fp_latent_window_size: {dataset.fp_latent_window_size}
        fp_1f_clean_indices: {dataset.fp_1f_clean_indices}
        fp_1f_target_index: {dataset.fp_1f_target_index}
        fp_1f_no_post: {dataset.fp_1f_no_post}
        no_resize_control: {dataset.no_resize_control}
        control_resolution: {dataset.control_resolution}
    \n"""
                ),
                "    ",
            )
        else:
            info += indent(
                dedent(
                    f"""\
        video_directory: "{dataset.video_directory}"
        video_jsonl_file: "{dataset.video_jsonl_file}"
        control_directory: "{dataset.control_directory}"
        target_frames: {dataset.target_frames}
        frame_extraction: {dataset.frame_extraction}
        frame_stride: {dataset.frame_stride}
        frame_sample: {dataset.frame_sample}
        max_frames: {dataset.max_frames}
        source_fps: {dataset.source_fps}
        fp_latent_window_size: {dataset.fp_latent_window_size}
    \n"""
                ),
                "    ",
            )
    logger.info(f"{info}")

    # make buckets first because it determines the length of dataset
    # and set the same seed for all datasets
    seed = random.randint(0, 2**31)  # actual seed is seed + epoch_no
    for i, dataset in enumerate(datasets):
        # logger.info(f"[Dataset {i}]")
        dataset.set_seed(seed, shared_epoch)
        if training:
            dataset.prepare_for_training(num_timestep_buckets=num_timestep_buckets)

    return DatasetGroup(datasets)


def load_user_config(file: str) -> dict:
    file: Path = Path(file)
    if not file.is_file():
        raise ValueError(f"file not found / ファイルが見つかりません: {file}")

    if file.name.lower().endswith(".json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception:
            logger.error(
                f"Error on parsing JSON config file. Please check the format. / JSON 形式の設定ファイルの読み込みに失敗しました。文法が正しいか確認してください。: {file}"
            )
            raise
    elif file.name.lower().endswith(".toml"):
        try:
            config = toml.load(file)
        except Exception:
            logger.error(
                f"Error on parsing TOML config file. Please check the format. / TOML 形式の設定ファイルの読み込みに失敗しました。文法が正しいか確認してください。: {file}"
            )
            raise
    else:
        raise ValueError(f"not supported config file format / 対応していない設定ファイルの形式です: {file}")

    deprecated_key_map = {
        "flux_kontext_no_resize_control": "no_resize_control",
        "qwen_image_edit_no_resize_control": "no_resize_control",
        "qwen_image_edit_control_resolution": "control_resolution",
    }

    def normalize_deprecated_keys(section: dict, section_name: str) -> None:
        for old_key, new_key in deprecated_key_map.items():
            if old_key not in section:
                continue
            if new_key in section:
                logger.warning(
                    f"Deprecated config key '{old_key}' is ignored because '{new_key}' is already set in {section_name}."
                )
            else:
                section[new_key] = section[old_key]
                logger.warning(f"Deprecated config key '{old_key}' found in {section_name}; use '{new_key}' instead.")
            del section[old_key]

    general_config = config.get("general")
    if isinstance(general_config, dict):
        normalize_deprecated_keys(general_config, "general")

    datasets_config = config.get("datasets", [])
    if isinstance(datasets_config, list):
        for idx, dataset_config in enumerate(datasets_config):
            if isinstance(dataset_config, dict):
                normalize_deprecated_keys(dataset_config, f"datasets[{idx}]")

    return config


# for config test
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_config")
    config_args, remain = parser.parse_known_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_dataset", action="store_true")
    argparse_namespace = parser.parse_args(remain)

    logger.info("[argparse_namespace]")
    logger.info(f"{vars(argparse_namespace)}")

    user_config = load_user_config(config_args.dataset_config)

    logger.info("")
    logger.info("[user_config]")
    logger.info(f"{user_config}")

    sanitizer = ConfigSanitizer()
    sanitized_user_config = sanitizer.sanitize_user_config(user_config)

    logger.info("")
    logger.info("[sanitized_user_config]")
    logger.info(f"{sanitized_user_config}")

    blueprint = BlueprintGenerator(sanitizer).generate(user_config, argparse_namespace)

    logger.info("")
    logger.info("[blueprint]")
    logger.info(f"{blueprint}")

    dataset_group = generate_dataset_group_by_blueprint(blueprint.dataset_group)
