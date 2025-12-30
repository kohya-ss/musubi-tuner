import toml
import json
import logging
from dataclasses import asdict
from typing import List, Dict, Any, Union, Optional

from musubi_tuner.dataset.config_utils import (
    BaseDatasetParams,
    ImageDatasetParams,
    VideoDatasetParams,
    ConfigSanitizer
)

logger = logging.getLogger(__name__)

class DatasetStateManager:
    def __init__(self):
        self.datasets: List[Union[ImageDatasetParams, VideoDatasetParams]] = []
        self.general_params: Dict[str, Any] = {
            "resolution": [1328, 1328],
            "batch_size": 1,
            "num_repeats": 1,
            "enable_bucket": False,
            "bucket_no_upscale": False,
            "caption_extension": ".txt",
        }

    def add_dataset(self, dataset_type: str = "image") -> None:
        if dataset_type == "image":
            self.datasets.append(ImageDatasetParams())
        else:
            self.datasets.append(VideoDatasetParams())

    def remove_dataset(self, index: int) -> None:
        if 0 <= index < len(self.datasets):
            self.datasets.pop(index)

    def update_dataset(self, index: int, params: Dict[str, Any]) -> None:
        if 0 <= index < len(self.datasets):
            dataset = self.datasets[index]
            for key, value in params.items():
                if hasattr(dataset, key):
                    setattr(dataset, key, value)

    def update_general_params(self, params: Dict[str, Any]) -> None:
        for key, value in params.items():
            self.general_params[key] = value

    def to_toml_string(self, architecture: Optional[str] = None) -> str:
        gen = {k: v for k, v in self.general_params.items() if k != "cache_directory"}

        filtered_datasets = []
        for d in self.datasets:
            data = asdict(d)
            for list_field in ["resolution", "target_frames"]:
                if list_field in data and isinstance(data[list_field], (list, tuple)):
                    data[list_field] = [int(x) for x in data[list_field]]

            for junk in ["architecture", "debug_dataset"]:
                if junk in data:
                    del data[junk]
            filtered_datasets.append(data)

        if "resolution" in gen and isinstance(gen["resolution"], (list, tuple)):
            gen["resolution"] = [int(x) for x in gen["resolution"]]

        config = {
            "general": gen,
            "datasets": filtered_datasets
        }

        for ds in config["datasets"]:
            for k, v in gen.items():
                if k in ds:
                    ds_val = ds[k]
                    c_ds_val = list(ds_val) if isinstance(ds_val, (list, tuple)) else ds_val
                    c_gen_val = list(v) if isinstance(v, (list, tuple)) else v
                    
                    if c_ds_val == c_gen_val:
                        ds[k] = None

            if architecture:
                arch_fields = {
                    "FramePack": ["fp_latent_window_size", "fp_1f_clean_indices", "fp_1f_target_index", "fp_1f_no_post"],
                    "Flux.1 Kontext": ["flux_kontext_no_resize_control"],
                    "Qwen-Image": ["qwen_image_edit_no_resize_control", "qwen_image_edit_control_resolution"],
                    "Wan 2.1": [],
                    "Wan 2.2": [],
                    "HunyuanVideo": [],
                }
                for arch, fields in arch_fields.items():
                    if arch not in architecture:
                        for f in fields:
                            if f in ds:
                                ds[f] = None

        clean_config = self._clean_dict(config)
        
        return toml.dumps(clean_config)

    def _clean_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        clean = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, dict):
                nested = self._clean_dict(v)
                if nested:
                    clean[k] = nested
            elif isinstance(v, list):
                cleaned_list = []
                for i in v:
                    if isinstance(i, dict):
                        ni = self._clean_dict(i)
                        if ni: cleaned_list.append(ni)
                    else:
                        cleaned_list.append(i)
                if cleaned_list:
                    clean[k] = cleaned_list
            else:
                clean[k] = v
        return clean

    def from_toml_string(self, toml_str: str) -> None:
        try:
            config = toml.loads(toml_str)
            
            if "general" in config:
                self.update_general_params(config["general"])
            
            if "datasets" in config:
                self.datasets = []
                for ds_data in config["datasets"]:
                    if "video_directory" in ds_data or "video_jsonl_file" in ds_data:
                         ds = VideoDatasetParams(**{k: v for k, v in ds_data.items() if k in VideoDatasetParams.__dataclass_fields__})
                    else:
                         ds = ImageDatasetParams(**{k: v for k, v in ds_data.items() if k in ImageDatasetParams.__dataclass_fields__})
                    self.datasets.append(ds)

        except Exception as e:
            logger.error(f"Failed to parse TOML: {e}")
            raise e
