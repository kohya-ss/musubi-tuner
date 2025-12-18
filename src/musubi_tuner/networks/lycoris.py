# API Adapter for LyCORIS library integration with musubi-tuner
# This file bridges between musubi-tuner's expected interface and LyCORIS's actual API
# LyCORIS: Lora beYond Conventional methods, Other Rank adaptation Implementations

import os
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn

# Import logging for debug messages
from blissful_tuner.blissful_logger import BlissfulLogger
logger = BlissfulLogger(__name__, "green")


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: Optional[List[nn.Module]],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
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
    try:
        from lycoris.kohya import LycorisNetworkKohya
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

    # Handle conv-specific dimensions if provided
    conv_dim = kwargs.get('conv_dim', None)
    conv_alpha = kwargs.get('conv_alpha', None)
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

    # Create the LyCORIS network
    # LycorisNetworkKohya expects similar parameters to regular LoRA
    # Build kwargs dict, only including non-None parameters
    lycoris_kwargs = {
        'text_encoder': text_encoders,  # Can be None for UNet-only training
        'unet': unet,
        'multiplier': multiplier,
        'lora_dim': network_dim,
        'alpha': network_alpha,
        'algo': algo,  # Algorithm selection: lora, locon, loha, lokr, etc.
    }

    # Only add optional parameters if they're not None
    if neuron_dropout is not None:
        lycoris_kwargs['dropout'] = neuron_dropout
    if conv_dim is not None:
        lycoris_kwargs['conv_lora_dim'] = conv_dim
    if conv_alpha is not None:
        lycoris_kwargs['conv_alpha'] = conv_alpha

    # Add any remaining kwargs not already handled
    for k, v in kwargs.items():
        if k not in ['algo', 'conv_dim', 'conv_alpha', 'neuron_dropout']:
            lycoris_kwargs[k] = v

    network = LycorisNetworkKohya(**lycoris_kwargs)

    # Log success
    logger.info(f"Successfully created LyCORIS network using {algo} algorithm")

    return network


def create_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
):
    """
    Create a LyCORIS network from saved weights for inference.

    This function reconstructs a LyCORIS network from a saved state dict,
    useful for loading pre-trained LyCORIS models.

    Args:
        multiplier: LoRA weight multiplier for inference
        weights_sd: State dict containing the saved weights
        text_encoders: List of text encoder models (optional)
        unet: The UNet/DiT model to apply LyCORIS to
        for_inference: Whether this network is for inference only
        **kwargs: Additional arguments

    Returns:
        LycorisNetworkKohya instance loaded with the provided weights
    """
    try:
        from lycoris.kohya import LycorisNetworkKohya
    except ImportError as e:
        raise ImportError(
            "LyCORIS is not installed. Please install it with 'pip install lycoris-lora' "
            "to use LyCORIS network modules.\n"
            f"Original error: {e}"
        )

    # Extract network configuration from the weights
    # LyCORIS saves metadata about dimensions in the state dict
    modules_dim = {}
    modules_alpha = {}
    algo = None

    for key, value in weights_sd.items():
        if "." not in key:
            continue

        # Extract module name and parameter type
        parts = key.split(".")
        module_name = parts[0]

        # Detect algorithm from weight keys if possible
        if algo is None:
            if "lokr" in key.lower():
                algo = "lokr"
            elif "loha" in key.lower() or "hada" in key:
                algo = "loha"
            elif "locon" in key.lower() or ("conv" in key and "lora" in key):
                algo = "locon"
            # Default will be set to 'lora' if not detected

        if "alpha" in key:
            modules_alpha[module_name] = value
        elif "lora_down" in key or "lokr_w1" in key or "hada_w1" in key:
            # Determine dimension from the weight shape
            dim = value.shape[0]
            modules_dim[module_name] = dim

    # Set default algorithm if not detected
    if algo is None:
        algo = kwargs.get('algo', 'lora')
        logger.info(f"Algorithm not detected from weights, using: {algo}")
    else:
        logger.info(f"Detected algorithm from weights: {algo}")

    # Get a representative dimension and alpha if individual module configs not found
    if not modules_dim:
        logger.warning("Could not extract module dimensions from weights, using defaults")
        network_dim = 4
        network_alpha = 1.0
    else:
        # Use the most common dimension as the default
        dims = list(modules_dim.values())
        network_dim = max(set(dims), key=dims.count)
        # Use the first alpha found or default to 1.0
        network_alpha = list(modules_alpha.values())[0] if modules_alpha else 1.0

    logger.info(f"Loading LyCORIS network from weights - algo: {algo}, dim: {network_dim}, alpha: {network_alpha}")

    # Create network with detected/specified configuration
    # Build kwargs dict, only including non-None parameters
    lycoris_kwargs = {
        'text_encoder': text_encoders,
        'unet': unet,
        'multiplier': multiplier,
        'lora_dim': network_dim,
        'alpha': network_alpha,
        'algo': algo,
    }

    # Only add optional parameters if they're not None/empty
    if modules_dim:
        lycoris_kwargs['modules_dim'] = modules_dim
    if modules_alpha:
        lycoris_kwargs['modules_alpha'] = modules_alpha

    # Add any remaining kwargs not already handled
    for k, v in kwargs.items():
        if k not in ['algo', 'modules_dim', 'modules_alpha']:
            lycoris_kwargs[k] = v

    network = LycorisNetworkKohya(**lycoris_kwargs)

    # Load the weights into the network
    info = network.load_state_dict(weights_sd, strict=False)
    if info.missing_keys or info.unexpected_keys:
        logger.warning(f"Weight loading info - Missing: {len(info.missing_keys)}, Unexpected: {len(info.unexpected_keys)}")
        if info.missing_keys:
            logger.debug(f"Missing keys: {info.missing_keys[:5]}...")  # Show first 5
        if info.unexpected_keys:
            logger.debug(f"Unexpected keys: {info.unexpected_keys[:5]}...")  # Show first 5
    else:
        logger.info("Successfully loaded all weights")

    return network


# Optional: Export the kohya module directly for advanced users
try:
    from lycoris import kohya
    # Make the kohya module available for direct access if needed
    __all__ = ['create_network', 'create_network_from_weights', 'kohya']
except ImportError:
    # If LyCORIS is not installed, just export our functions
    __all__ = ['create_network', 'create_network_from_weights']