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
    config = load_config()
    return config.get("training_args", {}).get("essential", [])


def get_advanced_training_args() -> Dict[str, List[Dict[str, Any]]]:
    config = load_config()
    return config.get("training_args", {}).get("advanced", {})

def get_all_training_arg_names() -> List[str]:
    essential = get_essential_training_args()
    advanced_groups = get_advanced_training_args()
    
    names = [arg["name"] for arg in essential]
    for group in advanced_groups.values():
        names.extend([arg["name"] for arg in group])
    
    return names

def get_essential_performance_args() -> List[Dict[str, Any]]:
    config = load_config()
    return config.get("performance_args", {}).get("essential", [])

def get_advanced_performance_args() -> List[Dict[str, Any]]:
    config = load_config()
    return config.get("performance_args", {}).get("advanced", [])

def get_essential_dataset_options() -> List[Dict[str, Any]]:
    config = load_config()
    return config.get("dataset_options", {}).get("essential", [])

def get_advanced_dataset_options() -> List[Dict[str, Any]]:
    config = load_config()
    return config.get("dataset_options", {}).get("advanced", [])

def get_model_specific_args(model_name: str) -> Dict[str, Any]:
    config = load_config()
    model_specific = config.get("model_specific_args", {})

    if model_name in model_specific:
        return model_specific[model_name]

    for key in model_specific:
        if key in model_name:
            return model_specific[key]
    
    return {}

def get_model_specific_args_for_arch(model_name: str) -> List[Dict[str, Any]]:
    results = []
    config = load_config()
    model_specific = config.get("model_specific_args", {})
    
    for key, value in model_specific.items():
        if key in model_name:
            results.append(value)
    
    return results

def should_skip_arg_in_simple_mode(arg_def: Dict[str, Any]) -> bool:
    return not arg_def.get("essential", False)

def build_cli_args_for_training(
    values: Dict[str, Any],
    is_simple_mode: bool = True
) -> List[str]:
    args = []
    essential = get_essential_training_args()
    advanced_groups = get_advanced_training_args()

    for arg_def in essential:
        name = arg_def["name"]
        cli_arg = get_cli_flag(name)
        
        if name not in values:
            continue
        
        value = values[name]
        if value is None or value == "":
            continue
        
        if arg_def["type"] == "checkbox":
            if value:
                args.append(cli_arg)
        else:
            args.extend([cli_arg, str(value)])

    if not is_simple_mode:
        for group_name, group_args in advanced_groups.items():
            for arg_def in group_args:
                name = arg_def["name"]
                cli_arg = get_cli_flag(name)
                
                if name not in values:
                    continue
                
                value = values[name]
                default = arg_def.get("default")
                skip_if_default = arg_def.get("skip_if_default", False)

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
    for arg_def in get_essential_training_args():
        if arg_def["name"] == arg_name:
            return arg_def.get("default")

    for group in get_advanced_training_args().values():
        for arg_def in group:
            if arg_def["name"] == arg_name:
                return arg_def.get("default")

    for arg_def in get_essential_performance_args():
        if arg_def["name"] == arg_name:
            return arg_def.get("default")
    
    for arg_def in get_advanced_performance_args():
        if arg_def["name"] == arg_name:
            return arg_def.get("default")
    
    return None

def reload_config():
    global _config_cache
    _config_cache = None
    return load_config()