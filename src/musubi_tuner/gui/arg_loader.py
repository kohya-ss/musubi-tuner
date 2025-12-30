"""
Argument Loader Module
Loads and manages the JSON-based GUI configuration for Musubi Tuner.
"""
import json
import os
from typing import Dict, List, Any, Optional

# Path to the config file
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "gui_config.json")
_config_cache: Optional[Dict[str, Any]] = None


def load_config() -> Dict[str, Any]:
    """Load the GUI configuration from JSON file."""
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


def get_ui_mode_default() -> str:
    """Get the default UI mode (simple or advanced)."""
    config = load_config()
    return config.get("ui_settings", {}).get("default_mode", "simple")


def get_model_list() -> List[str]:
    """Get list of all available model names."""
    config = load_config()
    return list(config.get("models", {}).keys())


def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get configuration for a specific model."""
    config = load_config()
    return config.get("models", {}).get(model_name, {})


def get_model_resolution(model_name: str) -> tuple:
    """Get the default resolution for a model."""
    model_config = get_model_config(model_name)
    resolution = model_config.get("resolution", [1024, 1024])
    return tuple(resolution)


def get_model_vram_settings(model_name: str, vram_size: str) -> Dict[str, Any]:
    """Get VRAM-specific settings for a model."""
    model_config = get_model_config(model_name)
    vram_settings = model_config.get("vram_settings", {})
    
    # Try exact match first
    if vram_size in vram_settings:
        return vram_settings[vram_size]
    
    # Fallback to 24GB settings
    return vram_settings.get("24", {"batch_size": 1})


def get_model_training_base(model_name: str) -> Dict[str, Any]:
    """Get base training parameters for a model."""
    model_config = get_model_config(model_name)
    return model_config.get("training_base", {})


def get_model_paths(model_name: str, comfy_models_dir: str) -> Dict[str, str]:
    """Get all model paths for a given model and ComfyUI directory."""
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
    """Get list of essential training arguments."""
    config = load_config()
    return config.get("training_args", {}).get("essential", [])


def get_advanced_training_args() -> Dict[str, List[Dict[str, Any]]]:
    """Get dictionary of advanced training argument groups."""
    config = load_config()
    return config.get("training_args", {}).get("advanced", {})


def get_all_training_arg_names() -> List[str]:
    """Get all training argument names (essential + advanced)."""
    essential = get_essential_training_args()
    advanced_groups = get_advanced_training_args()
    
    names = [arg["name"] for arg in essential]
    for group in advanced_groups.values():
        names.extend([arg["name"] for arg in group])
    
    return names


def get_essential_performance_args() -> List[Dict[str, Any]]:
    """Get list of essential performance arguments."""
    config = load_config()
    return config.get("performance_args", {}).get("essential", [])


def get_advanced_performance_args() -> List[Dict[str, Any]]:
    """Get list of advanced performance arguments."""
    config = load_config()
    return config.get("performance_args", {}).get("advanced", [])


def get_essential_dataset_options() -> List[Dict[str, Any]]:
    """Get list of essential dataset options."""
    config = load_config()
    return config.get("dataset_options", {}).get("essential", [])


def get_advanced_dataset_options() -> List[Dict[str, Any]]:
    """Get list of advanced dataset options."""
    config = load_config()
    return config.get("dataset_options", {}).get("advanced", [])


def get_model_specific_args(model_name: str) -> Dict[str, Any]:
    """Get model-specific argument definitions for a model."""
    config = load_config()
    model_specific = config.get("model_specific_args", {})
    
    # Check for exact match
    if model_name in model_specific:
        return model_specific[model_name]
    
    # Check for partial match (e.g., "Wan 2.1 (T2V-14B)" matches "Wan")
    for key in model_specific:
        if key in model_name:
            return model_specific[key]
    
    return {}


def get_model_specific_args_for_arch(model_name: str) -> List[Dict[str, Any]]:
    """Get all model-specific args matching a model architecture."""
    results = []
    config = load_config()
    model_specific = config.get("model_specific_args", {})
    
    for key, value in model_specific.items():
        if key in model_name:
            results.append(value)
    
    return results


def should_skip_arg_in_simple_mode(arg_def: Dict[str, Any]) -> bool:
    """Check if an argument should be skipped in simple mode."""
    # Essential args are never skipped
    return not arg_def.get("essential", False)


def build_cli_args_for_training(
    values: Dict[str, Any],
    is_simple_mode: bool = True
) -> List[str]:
    """
    Build CLI arguments from form values.
    In simple mode, only essential args are included.
    """
    args = []
    essential = get_essential_training_args()
    advanced_groups = get_advanced_training_args()
    
    # Process essential args (always included)
    for arg_def in essential:
        name = arg_def["name"]
        cli_arg = arg_def.get("cli_arg")
        if not cli_arg or name not in values:
            continue
        
        value = values[name]
        if value is None or value == "":
            continue
        
        if arg_def["type"] == "checkbox":
            if value:
                args.append(cli_arg)
        else:
            args.extend([cli_arg, str(value)])
    
    # Process advanced args only in advanced mode
    if not is_simple_mode:
        for group_name, group_args in advanced_groups.items():
            for arg_def in group_args:
                name = arg_def["name"]
                cli_arg = arg_def.get("cli_arg")
                if not cli_arg or name not in values:
                    continue
                
                value = values[name]
                default = arg_def.get("default")
                skip_if_default = arg_def.get("skip_if_default", False)
                
                # Skip if value matches default and skip_if_default is True
                if skip_if_default and value == default:
                    continue
                
                if value is None or value == "":
                    continue
                
                if arg_def["type"] == "checkbox":
                    if value:
                        args.append(cli_arg)
                else:
                    args.extend([cli_arg, str(value)])
    
    return args


def get_arg_default(arg_name: str) -> Any:
    """Get the default value for a training argument by name."""
    # Search in essential args
    for arg_def in get_essential_training_args():
        if arg_def["name"] == arg_name:
            return arg_def.get("default")
    
    # Search in advanced args
    for group in get_advanced_training_args().values():
        for arg_def in group:
            if arg_def["name"] == arg_name:
                return arg_def.get("default")
    
    # Search in performance args
    for arg_def in get_essential_performance_args():
        if arg_def["name"] == arg_name:
            return arg_def.get("default")
    
    for arg_def in get_advanced_performance_args():
        if arg_def["name"] == arg_name:
            return arg_def.get("default")
    
    return None


def reload_config():
    """Force reload the configuration from disk."""
    global _config_cache
    _config_cache = None
    return load_config()
