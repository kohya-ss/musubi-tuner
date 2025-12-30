import os
from typing import Dict, Any, Tuple, Optional

from musubi_tuner.gui.arg_loader import (
    load_config,
    get_model_config,
    get_model_resolution,
    get_model_vram_settings,
    get_model_training_base,
    get_model_paths,
)

class ConfigManager:
    def __init__(self):
        self._config = load_config()
        self.models = self._config.get("models", {})
    
    def get_resolution(self, model_name: str) -> Tuple[int, int]:
        return get_model_resolution(model_name)
    
    def get_batch_size(self, model_name: str, vram_size: str) -> int:
        if not vram_size:
            vram_size = "16"
        
        vram_conf = get_model_vram_settings(model_name, vram_size)
        return vram_conf.get("batch_size", 1)
    
    def get_preprocessing_paths(self, model_name: str, comfy_models_dir: str) -> Tuple[str, str, str]:
        paths = get_model_paths(model_name, comfy_models_dir)
        return paths.get("vae", ""), paths.get("te1", ""), paths.get("te2", "")
    
    def get_training_defaults(self, model_name: str, vram_size: str, comfy_models_dir: str) -> Dict[str, Any]:
        base = get_model_training_base(model_name).copy()
        
        if not vram_size:
            vram_size = "16"

        vram_conf = get_model_vram_settings(model_name, vram_size)

        for key in ["block_swap", "fp8_llm", "fp8_scaled"]:
            if key in vram_conf:
                base[key] = vram_conf[key]

        paths = get_model_paths(model_name, comfy_models_dir)
        base["dit_path"] = paths.get("dit", "")
        
        return base
    
    def get_model_script(self, model_name: str) -> str:
        model_config = get_model_config(model_name)
        return model_config.get("script", "")
    
    def get_model_task(self, model_name: str) -> Optional[str]:
        model_config = get_model_config(model_name)
        return model_config.get("task")

