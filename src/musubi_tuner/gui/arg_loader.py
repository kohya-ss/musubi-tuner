import json
import os
from typing import Dict, List, Any, Optional

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "gui_config.json")
_config_cache: Optional[Dict[str, Any]] = None

def load_config() -> Dict[str, Any]:
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _config_cache = json.load(f)
        return _config_cache
    except FileNotFoundError:
        print(f"Warning: Config file not found at {_CONFIG_PATH}. Using empty config.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse config file: {e}. Using empty config.")
        return {}

def get_label_key(name: str) -> str:
    return f"lbl_{name}"

def get_cli_flag(name: str) -> str:
    return f"--{name}"

def get_model_list() -> List[str]:
    config = load_config()
    return list(config.get("models", {}).keys())

def get_model_config(model_name: str) -> Dict[str, Any]:
    config = load_config()
    return config.get("models", {}).get(model_name, {})

def get_model_resolution(model_name: str) -> tuple:
    model_config = get_model_config(model_name)
    resolution = model_config.get("resolution", [1024, 1024])
    return tuple(resolution)

def get_model_vram_settings(model_name: str, vram_size: str) -> Dict[str, Any]:
    model_config = get_model_config(model_name)
    vram_settings = model_config.get("vram_settings", {})
    if vram_size in vram_settings:
        return vram_settings[vram_size]
    return vram_settings.get("24", {"batch_size": 1})

def get_model_training_base(model_name: str) -> Dict[str, Any]:
    model_config = get_model_config(model_name)
    return model_config.get("training_base", {})

def get_model_paths(model_name: str, comfy_models_dir: str) -> Dict[str, str]:
    model_config = get_model_config(model_name)
    def join_path(subpath):
        if subpath and comfy_models_dir:
            return os.path.join(comfy_models_dir, *subpath)
        return ""
    return {
        "vae": join_path(model_config.get("vae_subpath")),
        "te1": join_path(model_config.get("te1_subpath")),
        "te2": join_path(model_config.get("te2_subpath")),
        "dit": join_path(model_config.get("dit_subpath")),
    }

def get_essential_training_args() -> List[Dict[str, Any]]:
    return load_config().get("training_args", {}).get("essential", [])

def get_advanced_training_args() -> Dict[str, List[Dict[str, Any]]]:
    return load_config().get("training_args", {}).get("advanced", {})

def get_essential_performance_args() -> List[Dict[str, Any]]:
    return load_config().get("performance_args", {}).get("essential", [])

def get_advanced_performance_args() -> List[Dict[str, Any]]:
    return load_config().get("performance_args", {}).get("advanced", [])

def get_essential_dataset_options() -> List[Dict[str, Any]]:
    return load_config().get("dataset_options", {}).get("essential", [])

def get_advanced_dataset_options() -> List[Dict[str, Any]]:
    return load_config().get("dataset_options", {}).get("advanced", [])

def get_model_specific_args_map() -> Dict[str, Any]:
    return load_config().get("model_specific_args", {})

def get_all_args_map() -> Dict[str, Dict[str, Any]]:
    mapping = {}
    config = load_config()

    def process_list(arg_list):
        for arg in arg_list:
            mapping[arg["name"]] = arg

    process_list(config.get("training_args", {}).get("essential", []))
    for group in config.get("training_args", {}).get("advanced", {}).values():
        process_list(group)

    process_list(config.get("performance_args", {}).get("essential", []))
    process_list(config.get("performance_args", {}).get("advanced", []))

    for group in config.get("model_specific_args", {}).values():
        process_list(group.get("args", []))
        
    return mapping

def reload_config():
    global _config_cache
    _config_cache = None
    return load_config()