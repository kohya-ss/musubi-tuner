# API Adapter for LyCORIS library integration with musubi-tuner
# This file bridges between musubi-tuner's expected interface and LyCORIS's actual API
# LyCORIS: Lora beYond Conventional methods, Other Rank adaptation Implementations

from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

# Import logging for debug messages
from blissful_tuner.blissful_logger import BlissfulLogger
logger = BlissfulLogger(__name__, "green")


def _convert_value(value):
    """Convert string values to their appropriate Python types.

    This is needed because network_args from TOML config come as strings.
    """
    if not isinstance(value, str):
        return value

    # Handle boolean strings
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False

    # Try to convert to int
    try:
        return int(value)
    except ValueError:
        pass

    # Try to convert to float
    try:
        return float(value)
    except ValueError:
        pass

    # Return as-is if no conversion worked
    return value


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: Optional[List[nn.Module]],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    extra_unet_targets: Optional[List[str]] = None,
    **kwargs,
):
    """
    Create a LyCORIS network for training.

    This function acts as an adapter between musubi-tuner's interface and LyCORIS's
    LycorisNetworkKohya class.

    Args:
        multiplier: LoRA weight multiplier
        network_dim: The dimension (rank) for the network
        network_alpha: The alpha parameter for scaling
        vae: VAE model (not used by LyCORIS but kept for compatibility)
        text_encoders: List of text encoder models (can be None for some architectures)
        unet: The UNet/DiT model to apply LyCORIS to
        neuron_dropout: Dropout rate for neurons
        **kwargs: Additional arguments including 'algo' for algorithm selection
                 and other LyCORIS-specific parameters

    Returns:
        LycorisNetworkKohya instance configured with the specified parameters
    """
    # Try to import LyCORIS - provide helpful error if not installed
    # LyCORIS 3.x renamed LycorisNetworkKohya to LycorisNetwork and added create_network helper
    try:
        from lycoris.kohya import create_network as lycoris_create_network
        # Some builds keep LycorisNetworkKohya, others only expose LycorisNetwork
        try:
            from lycoris.kohya import LycorisNetworkKohya  # type: ignore
        except ImportError:
            from lycoris.kohya import LycorisNetwork as LycorisNetworkKohya  # type: ignore
    except ImportError as e:
        raise ImportError(
            "LyCORIS is not installed. Please install it with 'pip install lycoris-lora' "
            "to use LyCORIS network modules.\n"
            f"Original error: {e}"
        )

    # Set defaults if not provided
    if network_dim is None:
        network_dim = 4  # Default rank
        logger.info(f"No network_dim specified, using default: {network_dim}")

    if network_alpha is None:
        network_alpha = 1.0
        logger.info(f"No network_alpha specified, using default: {network_alpha}")

    # Extract algorithm from kwargs, default to 'lora' if not specified
    algo = kwargs.get('algo', 'lora')

    # Log the configuration
    logger.info(f"Creating LyCORIS network with algorithm: {algo}")
    logger.info(f"Network config - dim: {network_dim}, alpha: {network_alpha}, multiplier: {multiplier}")

    # Handle conv-specific dimensions if provided (convert from string if needed)
    conv_dim = _convert_value(kwargs.get('conv_dim', None))
    conv_alpha = _convert_value(kwargs.get('conv_alpha', None))
    if conv_dim is not None:
        logger.info(f"Conv layers - dim: {conv_dim}, alpha: {conv_alpha}")

    # Log LoKR-specific parameters if using LoKR
    if algo == 'lokr':
        lokr_params = {
            'factor': kwargs.get('factor', 'default'),
            'decompose_both': kwargs.get('decompose_both', 'default'),
            'use_tucker': kwargs.get('use_tucker', 'default'),
        }
        logger.info(f"LoKR parameters: {lokr_params}")

    # Auto-detect extra target modules (e.g. WanAttentionBlock) when not explicitly provided
    if extra_unet_targets is None and unet is not None:
        module_names = {m.__class__.__name__ for m in unet.modules()}
        if "WanAttentionBlock" in module_names:
            extra_unet_targets = ["WanAttentionBlock"]

    # Optionally extend the preset with extra UNet target modules (e.g. WanAttentionBlock)
    preset_name = kwargs.get("preset", None)
    if extra_unet_targets:
        try:
            from lycoris.config import PRESET
            # Mutate preset in-place so kohya.create_network uses the widened list
            target_preset_name = preset_name or "full"
            if target_preset_name in PRESET:
                targets = PRESET[target_preset_name].get("unet_target_module", [])
                PRESET[target_preset_name]["unet_target_module"] = list(
                    dict.fromkeys(list(targets) + list(extra_unet_targets))
                )
            else:
                # custom preset path
                from lycoris.utils.preset import read_preset

                preset_dict = read_preset(target_preset_name)
                if preset_dict:
                    preset_dict["unet_target_module"] = list(
                        dict.fromkeys(preset_dict.get("unet_target_module", []) + list(extra_unet_targets))
                    )
                    LycorisNetworkKohya.apply_preset(preset_dict)
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Failed to extend LyCORIS preset with extra_unet_targets: {e}")

    # Create the LyCORIS network using lycoris.kohya.create_network
    # This handles all the algo-to-module mapping internally
    # Build kwargs dict for create_network
    lycoris_kwargs = {
        'algo': algo,  # Algorithm selection: lora, locon, loha, lokr, etc.
    }

    # Only add optional parameters if they're not None
    if neuron_dropout is not None:
        lycoris_kwargs['dropout'] = neuron_dropout
    if conv_dim is not None:
        lycoris_kwargs['conv_dim'] = conv_dim
    if conv_alpha is not None:
        lycoris_kwargs['conv_alpha'] = conv_alpha

    # Add any remaining kwargs not already handled, converting string values to proper types
    for k, v in kwargs.items():
        if k not in ['algo', 'conv_dim', 'conv_alpha', 'neuron_dropout', 'extra_unet_targets']:
            lycoris_kwargs[k] = _convert_value(v)

    # lycoris.kohya.create_network expects text_encoder as list, not None
    network = lycoris_create_network(
        multiplier=multiplier,
        network_dim=network_dim,
        network_alpha=network_alpha,
        vae=vae,
        text_encoder=text_encoders if text_encoders is not None else [],
        unet=unet,
        **lycoris_kwargs
    )

    # Log success
    logger.info(f"Successfully created LyCORIS network using {algo} algorithm")

    return network


def create_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    vae: Optional[nn.Module] = None,
    file: Optional[str] = None,
    for_inference: bool = False,
    preset: str = "full",
    extra_unet_targets: Optional[List[str]] = None,
    **kwargs,
):
    """Reconstruct a LyCORIS network from an in-memory state dict.

    LyCORIS 3.x stopped re-exporting ``create_network_from_weights`` at the top
    level, so musubi-tuner uses this adapter to keep a stable interface. When
    the upstream helper exists (LyCORIS 3.4.0), we delegate to it so that
    module discovery matches the library's own logic. Otherwise we fall back to
    a minimal reimplementation that honours the current API (notably
    ``network_module`` expects a **string** key, not a class).

    Args mirror the original adapter with two additions:
    - ``preset``: LyCORIS preset name to apply before scanning modules.
    - ``extra_unet_targets``: list of extra module class names to ensure are
      considered (e.g. ``["WanAttentionBlock"]`` for Wan DiT).
    """

    # Auto-detect extra targets if not provided and the UNet includes WanAttentionBlock
    if extra_unet_targets is None and unet is not None:
        module_names = {m.__class__.__name__ for m in unet.modules()}
        if "WanAttentionBlock" in module_names:
            extra_unet_targets = ["WanAttentionBlock"]

    try:
        from lycoris.kohya import create_network_from_weights as lyco_create
        try:
            from lycoris.kohya import LycorisNetworkKohya  # type: ignore
        except ImportError:
            from lycoris.kohya import LycorisNetwork as LycorisNetworkKohya  # type: ignore
        lyco_helper_available = True
    except Exception:
        lyco_helper_available = False

    if lyco_helper_available:
        # Optionally extend preset with extra UNet targets (helps Wan DiT)
        if extra_unet_targets:
            try:
                from lycoris.config import PRESET
                if preset in PRESET:
                    targets = PRESET[preset].get("unet_target_module", [])
                    PRESET[preset]["unet_target_module"] = list(
                        dict.fromkeys(list(targets) + list(extra_unet_targets))
                    )
                else:
                    from lycoris.utils.preset import read_preset

                    preset_dict = read_preset(preset)
                    if preset_dict is not None:
                        preset_dict["unet_target_module"] = list(
                            dict.fromkeys(preset_dict.get("unet_target_module", []) + list(extra_unet_targets))
                        )
                        LycorisNetworkKohya.apply_preset(preset_dict)
            except Exception as e:
                logger.warning(f"Failed to extend preset with extra_unet_targets: {e}")

        network, weights = lyco_create(
            multiplier,
            file,
            vae,
            text_encoders,
            unet,
            weights_sd=weights_sd,
            for_inference=for_inference,
            **kwargs,
        )
        return network, weights

    # --------------------------
    # Fallback path (mirrors pre-3.x adapter, updated for 3.x API behaviour)
    # --------------------------
    try:
        try:
            from lycoris.kohya import LycorisNetworkKohya as LycoNet  # type: ignore
        except ImportError:
            from lycoris.kohya import LycorisNetwork as LycoNet  # type: ignore
    except Exception as e:
        raise ImportError(
            "LyCORIS is not installed or incomplete. Please reinstall with 'pip install lycoris-lora'.\n"
            f"Original error: {e}"
        )

    network_module_map = {
        "lora": "locon",
        "locon": "locon",
        "loha": "loha",
        "lokr": "lokr",
        "dylora": "dylora",
        "glora": "glora",
    }

    modules_dim: Dict[str, int] = {}
    modules_alpha: Dict[str, Union[int, float]] = {}
    algo: Optional[str] = None

    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lower = key.lower()
        if algo is None:
            if "lokr" in lower:
                algo = "lokr"
            elif "loha" in lower or "hada" in lower:
                algo = "loha"
            elif "locon" in lower or ("conv" in lower and "lora" in lower):
                algo = "locon"

        if "alpha" in key:
            modules_alpha[key.split(".")[0]] = value
        elif any(x in key for x in ("lora_down", "lokr_w1", "hada_w1")):
            modules_dim[key.split(".")[0]] = value.shape[0]

    if algo is None:
        algo = kwargs.get("algo", "lora")
        logger.info(f"Algorithm not detected from weights, defaulting to '{algo}'")
    else:
        logger.info(f"Detected algorithm from weights: {algo}")

    if not modules_dim:
        logger.warning("Could not read per-module ranks; falling back to dim=4 alpha=1.0")
        network_dim = 4
        network_alpha = 1.0
    else:
        dims = list(modules_dim.values())
        network_dim = max(set(dims), key=dims.count)
        network_alpha = list(modules_alpha.values())[0] if modules_alpha else 1.0

    if extra_unet_targets:
        try:
            preset_dict = {"unet_target_module": list(extra_unet_targets)}
            LycoNet.apply_preset(preset_dict)
        except Exception:
            logger.debug("Could not apply extra_unet_targets on fallback path")

    network_module_key = network_module_map.get(algo.lower(), "locon")

    network = LycoNet(
        text_encoder=text_encoders if text_encoders is not None else [],
        unet=unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        conv_lora_dim=network_dim,
        alpha=network_alpha,
        conv_alpha=network_alpha,
        network_module=network_module_key,
    )

    info = network.load_state_dict(weights_sd, strict=False)
    if info.missing_keys or info.unexpected_keys:
        logger.warning(
            f"Weight loading info - Missing: {len(info.missing_keys)}, Unexpected: {len(info.unexpected_keys)}"
        )
    else:
        logger.info("Successfully loaded all weights")

    return network, weights_sd


# Optional: Export the kohya module directly for advanced users
try:
    from lycoris import kohya
    # Make the kohya module available for direct access if needed
    __all__ = ['create_network', 'create_network_from_weights', 'kohya']
except ImportError:
    # If LyCORIS is not installed, just export our functions
    __all__ = ['create_network', 'create_network_from_weights']
