"""
Configuration Manager for Musubi Tuner GUI.
Now loads model configurations from gui_config.json for easier maintenance.
"""
import os
from typing import Dict, Any, Tuple, Optional

# Import from arg_loader for JSON-based config
from musubi_tuner.gui.arg_loader import (
    load_config,
    get_model_config,
    get_model_resolution,
    get_model_vram_settings,
    get_model_training_base,
    get_model_paths,
)


class ConfigManager:
    """
    Manages model configurations for the GUI.
    Configurations are now loaded from gui_config.json.
    """
    
    def __init__(self):
        """Initialize the ConfigManager by loading config from JSON."""
        self._config = load_config()
        self.models = self._config.get("models", {})
    
    def get_resolution(self, model_name: str) -> Tuple[int, int]:
        """Get the default resolution for a model."""
        return get_model_resolution(model_name)
    
    def get_batch_size(self, model_name: str, vram_size: str) -> int:
        """Get the recommended batch size for a model at a given VRAM size."""
        if not vram_size:
            vram_size = "16"
        
        vram_conf = get_model_vram_settings(model_name, vram_size)
        return vram_conf.get("batch_size", 1)
    
    def get_preprocessing_paths(
        self, model_name: str, comfy_models_dir: str
    ) -> Tuple[str, str, str]:
        """Get VAE, TE1, and TE2 paths for preprocessing."""
        paths = get_model_paths(model_name, comfy_models_dir)
        return paths.get("vae", ""), paths.get("te1", ""), paths.get("te2", "")
    
    def get_training_defaults(
        self, model_name: str, vram_size: str, comfy_models_dir: str
    ) -> Dict[str, Any]:
        """
        Get default training parameters for a model.
        Merges base training params with VRAM-specific overrides.
        """
        # Get base training config
        base = get_model_training_base(model_name).copy()
        
        if not vram_size:
            vram_size = "16"
        
        # Merge VRAM settings
        vram_conf = get_model_vram_settings(model_name, vram_size)
        
        # Only take relevant keys for training params
        for key in ["block_swap", "fp8_llm", "fp8_scaled"]:
            if key in vram_conf:
                base[key] = vram_conf[key]
        
        # Get DiT path
        paths = get_model_paths(model_name, comfy_models_dir)
        base["dit_path"] = paths.get("dit", "")
        
        return base
    
    def get_model_script(self, model_name: str) -> str:
        """Get the training script name for a model."""
        model_config = get_model_config(model_name)
        return model_config.get("script", "")
    
    def get_model_task(self, model_name: str) -> Optional[str]:
        """Get the task identifier for a model (e.g., 't2v-14B' for Wan)."""
        model_config = get_model_config(model_name)
        return model_config.get("task")

