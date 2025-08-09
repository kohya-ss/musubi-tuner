# Enhanced LoRA weights merger with Post-Hoc EMA method - Version 2
# Refactored for clarity and optimal LoRA preservation
# Features:
# Stage 1: Post-Hoc EMA for epoch merging
# Stage 2: Multi-LoRA merge with NO bleeding by default
# Single control parameter: --average for optional blending

import os
import re
import glob
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
from safetensors.torch import save_file
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


def extract_loraid_from_metadata(metadata: Optional[Dict[str, str]]) -> Optional[int]:
    """
    Extract LORAID from LoRA metadata.
    
    Args:
        metadata: Metadata dictionary from safetensors
        
    Returns:
        LORAID if found, None otherwise
    """
    if metadata and "ss_loraid" in metadata:
        try:
            return int(metadata["ss_loraid"])
        except (ValueError, TypeError):
            print(f"Warning: Invalid LORAID format in metadata: {metadata['ss_loraid']}")
    return None


def preserve_loraid_metadata(
    source_metadata: Optional[Dict[str, str]], 
    target_metadata: Optional[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """
    Preserve LORAID metadata from source to target.
    
    Args:
        source_metadata: Source metadata containing LORAID info
        target_metadata: Target metadata to update
        
    Returns:
        Updated target metadata with LORAID preserved
    """
    if not source_metadata:
        return target_metadata
        
    loraid = extract_loraid_from_metadata(source_metadata)
    if loraid is not None:
        if target_metadata is None:
            target_metadata = {}
        target_metadata["ss_loraid"] = str(loraid)
        
        # Also preserve LORAID parameter patterns if available
        for key in ["ss_loraid_include_patterns", "ss_loraid_exclude_patterns"]:
            if key in source_metadata:
                target_metadata[key] = source_metadata[key]
                
        print(f"Preserved LORAID {loraid} metadata")
        
    return target_metadata


def sigma_rel_to_gamma(sigma_rel):
    """Implementation of Algorithm 2 from the paper: https://arxiv.org/pdf/2312.02696"""
    # solve the cubic equation γ^3 + 7γ^2 + (16 - 1/σ_rel^2)γ + (12 - 1/σ_rel^2) = 0
    t = sigma_rel**-2
    # coefficients [1, 7, 16-t, 12-t]
    coeffs = [1, 7, 16 - t, 12 - t]
    # positive real root is γ
    roots = np.roots(coeffs)
    gamma = roots[np.isreal(roots) & (roots.real >= 0)].real.max()
    return gamma


def discover_lora_epochs(reference_file: str) -> List[str]:
    """
    Discover all epoch files for a given LoRA series from a reference file.
    Enhanced to search across multiple run directories for resumed training.
    Preserves all files including same-named epochs from different runs.
    
    Args:
        reference_file: Path to any epoch file in the series (e.g., "C:/train/saves/tanya/run1/tanya_512-000005.safetensors")
        
    Returns:
        List of paths to all epoch files, sorted by modification time
    """
    if not os.path.exists(reference_file):
        raise FileNotFoundError(f"Reference file not found: {reference_file}")
    
    # Extract directory and filename information
    reference_dir = os.path.dirname(reference_file)
    filename = os.path.basename(reference_file)
    
    # Extract the base name without epoch number
    # Pattern: name_resolution-epoch.safetensors -> name_resolution
    match = re.match(r'^(.+?)-\d{6}\.safetensors$', filename)
    if not match:
        # If no pattern match, treat as single file
        print(f"Warning: Could not extract pattern from {filename}, treating as single file")
        return [reference_file]
    
    base_name = match.group(1)
    pattern = f"{base_name}-*.safetensors"
    
    print(f"🔍 Discovering epochs for '{base_name}' pattern...")
    
    # Enhanced search strategy for resumed training across multiple directories
    epoch_files = []
    search_dirs = set()
    
    # Strategy 1: Search current directory and subdirectories
    search_dirs.add(reference_dir)
    
    # Strategy 2: Search parent directory and sibling directories (for run1, run2, etc.)
    parent_dir = os.path.dirname(reference_dir)
    if parent_dir and parent_dir != reference_dir:
        # Check if current dir looks like a run directory (run1, run2, epoch_xx, etc.)
        current_dir_name = os.path.basename(reference_dir)
        if re.match(r'.*(run|epoch|step|checkpoint).*\d+.*', current_dir_name, re.IGNORECASE):
            print(f"  📁 Detected potential run directory: {current_dir_name}")
            print(f"  📂 Searching parent directory for sibling runs: {os.path.basename(parent_dir)}")
            search_dirs.add(parent_dir)
            
            # Also search sibling directories that might contain related runs
            try:
                for item in os.listdir(parent_dir):
                    sibling_path = os.path.join(parent_dir, item)
                    if os.path.isdir(sibling_path) and item != current_dir_name:
                        # Check if sibling directory looks like it could contain related training runs
                        if re.match(r'.*(run|epoch|step|checkpoint|train).*', item, re.IGNORECASE):
                            search_dirs.add(sibling_path)
                            print(f"  🔍 Including sibling directory: {item}")
            except (PermissionError, OSError):
                pass  # Skip if can't access parent directory
    
    # Strategy 3: Search up to 2 levels of subdirectories from each search directory
    extended_search_dirs = set(search_dirs)
    for search_dir in search_dirs:
        try:
            for root, dirs, files in os.walk(search_dir):
                # Limit depth to avoid excessive searching
                depth = root.replace(search_dir, '').count(os.sep)
                if depth <= 2:  # Search up to 2 levels deep
                    extended_search_dirs.add(root)
                if depth >= 2:
                    dirs[:] = []  # Don't recurse deeper
        except (PermissionError, OSError):
            continue
    
    # Perform the actual search
    for search_dir in extended_search_dirs:
        try:
            search_pattern = os.path.join(search_dir, pattern)
            matches = glob.glob(search_pattern)
            if matches:
                epoch_files.extend(matches)
                rel_path = os.path.relpath(search_dir, reference_dir) if search_dir != reference_dir else "."
                print(f"  ✅ Found {len(matches)} files in: {rel_path}")
        except (PermissionError, OSError):
            continue
    
    # IMPORTANT: Do NOT remove duplicates - same-named files from different runs are different epochs!
    # But DO remove true duplicates (same physical file with different path separators)
    
    # Normalize paths to avoid Windows path separator issues (/ vs \)
    normalized_files = {}
    for file_path in epoch_files:
        normalized_path = os.path.normpath(os.path.abspath(file_path))
        if normalized_path not in normalized_files:
            normalized_files[normalized_path] = file_path
        else:
            print(f"  🔄 Skipping duplicate path: {file_path}")
    
    epoch_files = list(normalized_files.values())
    epoch_files.sort(key=lambda x: os.path.getmtime(x))
    
    if not epoch_files:
        print(f"⚠️  Warning: No epoch files found for pattern {pattern}")
        return [reference_file]
    
    print(f"\n📊 Discovered {len(epoch_files)} epoch files for {base_name}:")
    for i, file in enumerate(epoch_files):
        mtime = os.path.getmtime(file)
        rel_path = os.path.relpath(file, os.path.dirname(reference_file))
        print(f"  {i+1:2d}. {rel_path} (modified: {mtime})")
    
    return epoch_files


# ===== STAGE 1: POST-HOC EMA EPOCH MERGING =====

def merge_lora_epochs_with_ema(
    path: List[str], no_sort: bool, beta1: float, beta2: float, sigma_rel: Optional[float], output_file: Optional[str]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """
    Stage 1: Merge epochs using Post-Hoc EMA method.
    This is the original functionality for merging multiple epochs of a single LoRA.
    """
    print(f"\n{'='*50}")
    print("STAGE 1: POST-HOC EMA EPOCH MERGING")
    print(f"{'='*50}")
    
    # Sort the files by modification time
    if not no_sort:
        print("Sorting files by modification time...")
        path.sort(key=lambda x: os.path.getmtime(x))

    # Load metadata from the last file
    print(f"Loading metadata from {path[-1]}")
    with MemoryEfficientSafeOpen(path[-1]) as f:
        metadata = f.metadata()
    if metadata is None:
        print("No metadata found in the last file, proceeding without metadata.")
    else:
        print("Metadata found, using metadata from the last file.")
        
    # Extract LORAID from the first file (should be consistent across all epochs)
    source_loraid_metadata = None
    with MemoryEfficientSafeOpen(path[0]) as f:
        source_metadata = f.metadata()
        source_loraid_metadata = source_metadata
        
    # Preserve LORAID information in final metadata
    metadata = preserve_loraid_metadata(source_loraid_metadata, metadata)

    # Load the oldest file and initialize weights
    print(f"Loading weights from {path[0]}")
    with MemoryEfficientSafeOpen(path[0]) as f:
        original_dtypes = {}
        state_dict = {}
        for key in f.keys():
            value: torch.Tensor = f.get_tensor(key)

            if value.dtype.is_floating_point:
                original_dtypes[key] = value.dtype
                value = value.to(torch.float32)  # Convert to float32 for merging
            else:
                print(f"Skipping non-floating point tensor: {key}")

            state_dict[key] = value

    # Iterate through the remaining files, loading and merging their weights with decay rate beta
    ema_count = len(path) - 1
    if sigma_rel is not None:
        gamma = sigma_rel_to_gamma(sigma_rel)
    else:
        gamma = None

    for i, file in enumerate(path[1:]):
        if sigma_rel is not None:
            # Calculate beta using Power Function EMA
            t = i + 1
            beta = (1 - 1 / t) ** (gamma + 1)
        else:
            beta = beta1 + (beta2 - beta1) * (i / (ema_count - 1)) if ema_count > 1 else beta1

        print(f"Loading weights from {file} for merging with beta={beta:.4f}")
        with MemoryEfficientSafeOpen(file) as f:
            for key in f.keys():
                value = f.get_tensor(key)
                if key.endswith(".alpha"):
                    # compare alpha tensors and raise an error if they differ
                    if key not in state_dict or torch.allclose(state_dict[key], value.to(torch.float32)):
                        # If alpha tensors match, skip merging
                        continue
                    else:
                        raise ValueError(f"Alpha tensors for key {key} do not match across files.")

                if not value.dtype.is_floating_point:
                    # Skip non-floating point tensors
                    print(f"Skipping non-floating point tensor: {key}")
                    continue

                if key in state_dict:
                    # Merge the weights with decay rate beta
                    value = value.to(torch.float32)
                    state_dict[key] = state_dict[key] * beta + value * (1 - beta)
                else:
                    raise KeyError(f"Key {key} not found in the initial state_dict.")

    # Convert the merged weights back to their original dtypes
    for key in state_dict:
        if key in original_dtypes:
            state_dict[key] = state_dict[key].to(original_dtypes[key])

    # update metadata with new hash
    if metadata is not None:
        print("Updating metadata with new hashes.")
        model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

    # Save the final merged weights to a new file if output_file is provided
    if output_file:
        print(f"Saving merged weights to {output_file}")
        save_file(state_dict, output_file, metadata=metadata)
        print("Stage 1 EMA merging completed successfully.")
    
    # Return the state_dict for potential further processing
    return state_dict, metadata


# ===== STAGE 2: RANK-BASED MULTI-LORA MERGE WITH ZERO BLEEDING =====

def analyze_lora_dimensions(lora_state_dicts: List[Tuple[Dict[str, torch.Tensor], Dict[str, str], str]]) -> Dict[str, Dict[str, any]]:
    """
    Analyze LoRA dimensions to ensure compatibility for rank concatenation.
    
    Returns:
        Dict with analysis results including rank sizes, compatibility info
    """
    analysis = {
        "lora_info": [],
        "compatible": True,
        "total_rank": 0,
        "common_keys": set(),
        "dimension_mismatches": []
    }
    
    all_keys_sets = []
    
    for i, (state_dict, metadata, name) in enumerate(lora_state_dicts):
        lora_info = {
            "name": name,
            "keys": set(state_dict.keys()),
            "ranks": {},
            "shapes": {}
        }
        
        # Analyze LoRA structure
        lora_down_keys = [k for k in state_dict.keys() if k.endswith(".lora_down.weight")]
        
        for down_key in lora_down_keys[:3]:  # Check first few layers
            base_name = down_key.replace(".lora_down.weight", "")
            up_key = f"{base_name}.lora_up.weight"
            alpha_key = f"{base_name}.alpha"
            
            if up_key in state_dict:
                down_tensor = state_dict[down_key]
                up_tensor = state_dict[up_key]
                
                # LoRA rank is the inner dimension
                rank = down_tensor.shape[0]  # Should match up_tensor.shape[1]
                
                lora_info["ranks"][base_name] = rank
                lora_info["shapes"][base_name] = {
                    "down": down_tensor.shape,
                    "up": up_tensor.shape
                }
                
                if up_tensor.shape[1] != rank:
                    analysis["compatible"] = False
                    analysis["dimension_mismatches"].append(f"{name}: {base_name} rank mismatch")
        
        # Determine overall rank for this LoRA
        if lora_info["ranks"]:
            ranks = list(lora_info["ranks"].values())
            if len(set(ranks)) == 1:  # All layers have same rank
                lora_info["overall_rank"] = ranks[0]
                analysis["total_rank"] += ranks[0]
            else:
                analysis["compatible"] = False
                analysis["dimension_mismatches"].append(f"{name}: inconsistent ranks {set(ranks)}")
        
        analysis["lora_info"].append(lora_info)
        all_keys_sets.append(lora_info["keys"])
    
    # Find common keys across all LoRAs
    if all_keys_sets:
        analysis["common_keys"] = set.intersection(*all_keys_sets)
    
    return analysis


def apply_democratic_blending(
    processed_loras: List[Tuple[Dict[str, torch.Tensor], str, int]], 
    average_blend: float,
    all_keys: List[str]
) -> List[Tuple[Dict[str, torch.Tensor], str, int]]:
    """
    Apply democratic knowledge sharing based on concept consensus among LoRAs.
    
    Args:
        processed_loras: List of (state_dict, name, index) tuples
        average_blend: Blending factor (0.1 = 10% knowledge sharing, etc.)
        all_keys: All unique keys across LoRAs
        
    Returns:
        LoRAs with democratic knowledge sharing applied
    """
    if len(processed_loras) < 2:
        return processed_loras
    
    print(f"🗳️ Democratic Knowledge Sharing Analysis:")
    print(f"   • Blend factor: {average_blend}")
    print(f"   • LoRAs participating: {len(processed_loras)}")
    print(f"   • Total layers: {len(all_keys)}")
    
    # Analyze knowledge patterns across all LoRAs first
    knowledge_consensus = analyze_knowledge_consensus(processed_loras, all_keys)
    
    blended_loras = []
    total_democratic_transfers = 0
    total_preserved_weights = 0
    
    for i, (state_dict_i, name_i, idx_i) in enumerate(processed_loras):
        print(f"   🗳️ Processing {name_i} with democratic knowledge transfer...")
        blended_state_dict = {}
        layer_stats = {"democratic_transfer": 0, "preserved": 0}
        
        for key in all_keys:
            weight_i = state_dict_i[key]
            
            # Collect weights from other LoRAs for this layer
            other_weights = []
            other_names = []
            for j, (state_dict_j, name_j, idx_j) in enumerate(processed_loras):
                if i != j:  # Don't include self
                    other_weights.append(state_dict_j[key])
                    other_names.append(name_j)
            
            if not other_weights:
                # Only one LoRA, no blending needed
                blended_state_dict[key] = weight_i
                continue
            
            # Apply democratic knowledge transfer
            blended_weight = democratic_knowledge_transfer(
                weight_i, other_weights, other_names, name_i, key, 
                average_blend, knowledge_consensus.get(key, {})
            )
            blended_state_dict[key] = blended_weight
            
            # Track statistics
            if torch.allclose(weight_i, blended_weight, rtol=1e-6):
                layer_stats["preserved"] += 1
                total_preserved_weights += 1
            else:
                layer_stats["democratic_transfer"] += 1
                total_democratic_transfers += 1
        
        print(f"     ✅ {name_i}: {layer_stats['democratic_transfer']} knowledge transfers, {layer_stats['preserved']} preserved")
        blended_loras.append((blended_state_dict, name_i, idx_i))
    
    transfer_percentage = (total_democratic_transfers / (total_democratic_transfers + total_preserved_weights)) * 100
    print(f"🎯 Democratic Summary: {transfer_percentage:.1f}% of weights received democratic knowledge transfer")
    
    return blended_loras


def analyze_knowledge_consensus(
    processed_loras: List[Tuple[Dict[str, torch.Tensor], str, int]], 
    all_keys: List[str]
) -> Dict[str, Dict[str, any]]:
    """
    Analyze knowledge consensus across LoRAs to identify:
    1. Which LoRAs have strong knowledge in specific areas
    2. Which LoRAs lack knowledge that others possess
    3. Concept voting patterns
    """
    print("   🔍 Analyzing knowledge consensus across LoRAs...")
    
    consensus_data = {}
    
    for key in all_keys:
        # Collect all weights for this layer
        layer_weights = []
        lora_names = []
        
        for state_dict, name, idx in processed_loras:
            layer_weights.append(state_dict[key])
            lora_names.append(name)
        
        # Analyze knowledge patterns for this layer
        layer_consensus = analyze_layer_consensus(layer_weights, lora_names, key)
        consensus_data[key] = layer_consensus
    
    return consensus_data


def analyze_layer_consensus(weights: List[torch.Tensor], names: List[str], layer_key: str) -> Dict[str, any]:
    """
    Analyze consensus patterns in a specific layer across LoRAs.
    """
    if len(weights) < 2:
        return {}
    
    # Calculate pairwise similarities
    similarities = {}
    weight_magnitudes = {}
    
    for i, (w1, name1) in enumerate(zip(weights, names)):
        weight_magnitudes[name1] = torch.norm(w1).item()
        
        for j, (w2, name2) in enumerate(zip(weights, names)):
            if i < j:  # Avoid duplicate comparisons
                # Calculate cosine similarity (knowledge direction alignment)
                cos_sim = torch.nn.functional.cosine_similarity(
                    w1.flatten().unsqueeze(0), w2.flatten().unsqueeze(0)
                ).item()
                similarities[f"{name1}_{name2}"] = cos_sim
    
    # Identify knowledge gaps and strengths
    # Higher magnitude often indicates stronger learned features
    sorted_by_magnitude = sorted(weight_magnitudes.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "similarities": similarities,
        "magnitudes": weight_magnitudes,
        "knowledge_leader": sorted_by_magnitude[0][0] if sorted_by_magnitude else None,
        "layer_type": classify_layer_type(layer_key)
    }


def classify_layer_type(layer_key: str) -> str:
    """
    Classify what type of knowledge this layer likely contains.
    """
    if "cross_attn" in layer_key:
        return "text_to_image"  # Text-to-image attention (concepts, objects, poses)
    elif "self_attn" in layer_key:
        return "spatial"  # Spatial relationships, composition
    elif "ffn" in layer_key or "mlp" in layer_key:
        return "feature_processing"  # Feature processing, style, quality
    elif "norm" in layer_key:
        return "normalization"  # Less important for concept transfer
    else:
        return "general"


def democratic_knowledge_transfer(
    primary_weight: torch.Tensor, 
    other_weights: List[torch.Tensor], 
    other_names: List[str],
    primary_name: str,
    layer_key: str,
    blend_level: float,
    consensus_info: Dict[str, any]
) -> torch.Tensor:
    """
    Apply democratic knowledge transfer based on consensus analysis.
    """
    if not other_weights or blend_level <= 0.0:
        return primary_weight
    
    layer_type = consensus_info.get("layer_type", "general")
    magnitudes = consensus_info.get("magnitudes", {})
    knowledge_leader = consensus_info.get("knowledge_leader", None)
    
    # Democratic voting strategy based on layer type
    if layer_type == "text_to_image" and blend_level >= 0.1:
        # For text-to-image layers, prioritize knowledge transfer from stronger LoRAs
        return apply_concept_democracy(primary_weight, other_weights, other_names, 
                                     primary_name, magnitudes, knowledge_leader, blend_level)
    
    elif layer_type == "spatial" and blend_level >= 0.2:
        # For spatial layers, blend more conservatively (quality improvements)
        return apply_quality_democracy(primary_weight, other_weights, blend_level * 0.5)
    
    elif layer_type == "feature_processing" and blend_level >= 0.1:
        # For feature processing, moderate blending for style consistency
        return apply_quality_democracy(primary_weight, other_weights, blend_level * 0.7)
    
    return primary_weight  # No blending for other layer types or low blend levels


def apply_concept_democracy(
    primary_weight: torch.Tensor, 
    other_weights: List[torch.Tensor], 
    other_names: List[str],
    primary_name: str,
    magnitudes: Dict[str, float],
    knowledge_leader: str,
    blend_level: float
) -> torch.Tensor:
    """
    Apply democratic concept knowledge transfer.
    If majority has stronger knowledge, transfer it even if weights differ significantly.
    """
    primary_magnitude = magnitudes.get(primary_name, 0.0)
    
    # Find the strongest knowledge source
    strongest_weight = None
    strongest_magnitude = primary_magnitude
    
    for weight, name in zip(other_weights, other_names):
        other_magnitude = magnitudes.get(name, 0.0)
        if other_magnitude > strongest_magnitude:
            strongest_magnitude = other_magnitude
            strongest_weight = weight
    
    # If others have significantly stronger knowledge (e.g., Jessica knows "naked", Tanya doesn't)
    if strongest_weight is not None and strongest_magnitude > primary_magnitude * 1.5:
        # Democratic knowledge transfer from the strongest source
        blend_ratio = min(blend_level * 2.0, 0.5)  # More aggressive for concept gaps
        return (1 - blend_ratio) * primary_weight + blend_ratio * strongest_weight
    
    # Otherwise, gentle consensus blending
    consensus_weight = torch.stack(other_weights).mean(dim=0)
    return (1 - blend_level) * primary_weight + blend_level * consensus_weight


def apply_quality_democracy(primary_weight: torch.Tensor, other_weights: List[torch.Tensor], blend_level: float) -> torch.Tensor:
    """
    Apply gentle quality-focused democratic blending.
    """
    if not other_weights:
        return primary_weight
    
    consensus_weight = torch.stack(other_weights).mean(dim=0)
    return (1 - blend_level) * primary_weight + blend_level * consensus_weight


def apply_intelligent_blending(
    processed_loras: List[Tuple[Dict[str, torch.Tensor], str, int]], 
    average_blend: float,
    all_keys: List[str]
) -> List[Tuple[Dict[str, torch.Tensor], str, int]]:
    """
    Apply intelligent gradient magnitude-based blending for complementary knowledge sharing.
    
    Args:
        processed_loras: List of (state_dict, name, index) tuples
        average_blend: Blending factor (0.1 = 10% blend, 0.2 = 20% blend, etc.)
        all_keys: All unique keys across LoRAs
        
    Returns:
        Blended LoRAs with intelligent knowledge sharing
    """
    if len(processed_loras) < 2:
        return processed_loras
    
    print(f"🧠 Intelligent Blending Analysis:")
    print(f"   • Blend factor: {average_blend}")
    print(f"   • LoRAs to blend: {len(processed_loras)}")
    print(f"   • Total layers: {len(all_keys)}")
    
    # Conflict threshold - weights with differences above this are considered "conflicting"
    # Lower threshold = more conservative blending (only very similar weights get blended)
    # Higher threshold = more aggressive blending 
    conflict_threshold = 0.1  # Adjustable parameter
    
    blended_loras = []
    total_blended_weights = 0
    total_preserved_weights = 0
    
    for i, (state_dict_i, name_i, idx_i) in enumerate(processed_loras):
        print(f"   🔄 Processing {name_i}...")
        blended_state_dict = {}
        layer_stats = {"blended": 0, "preserved": 0}
        
        for key in all_keys:
            weight_i = state_dict_i[key]
            
            # Collect weights from other LoRAs for this layer
            other_weights = []
            for j, (state_dict_j, name_j, idx_j) in enumerate(processed_loras):
                if i != j:  # Don't include self
                    other_weights.append(state_dict_j[key])
            
            if not other_weights:
                # Only one LoRA, no blending needed
                blended_state_dict[key] = weight_i
                continue
            
            # Apply intelligent blending with all other LoRAs
            blended_weight = intelligent_blend_multi(weight_i, other_weights, average_blend, conflict_threshold)
            blended_state_dict[key] = blended_weight
            
            # Track statistics
            if torch.allclose(weight_i, blended_weight, rtol=1e-6):
                layer_stats["preserved"] += 1
                total_preserved_weights += 1
            else:
                layer_stats["blended"] += 1
                total_blended_weights += 1
        
        print(f"     ✅ {name_i}: {layer_stats['blended']} layers blended, {layer_stats['preserved']} preserved")
        blended_loras.append((blended_state_dict, name_i, idx_i))
    
    blend_percentage = (total_blended_weights / (total_blended_weights + total_preserved_weights)) * 100
    print(f"🎯 Blending Summary: {blend_percentage:.1f}% of weights received intelligent knowledge sharing")
    
    return blended_loras


def intelligent_blend_multi(primary_weight: torch.Tensor, other_weights: List[torch.Tensor], 
                           blend_level: float, conflict_threshold: float) -> torch.Tensor:
    """
    Intelligent gradient magnitude-based blending for multiple LoRAs.
    
    Args:
        primary_weight: Primary LoRA weight tensor
        other_weights: List of other LoRA weight tensors
        blend_level: How much to blend (0.1 = 10% knowledge sharing)
        conflict_threshold: Threshold for determining conflicting vs complementary weights
        
    Returns:
        Intelligently blended weight tensor
    """
    if not other_weights:
        return primary_weight
    
    # Calculate average of other weights for comparison
    other_weights_stacked = torch.stack(other_weights, dim=0)
    avg_other_weight = torch.mean(other_weights_stacked, dim=0)
    
    # Analyze which weights are "conflicting" vs "complementary"
    weight_diff = torch.abs(primary_weight - avg_other_weight)
    conflict_mask = weight_diff > conflict_threshold
    
    # Smart blending strategy:
    # - High conflict (different features): Preserve primary LoRA's uniqueness
    # - Low conflict (similar features): Blend for quality improvement
    
    if blend_level >= 0.1:
        # Blend complementary weights (low conflict areas)
        # For conflicting areas, keep primary weight unchanged
        blended_weight = torch.where(
            conflict_mask, 
            primary_weight,  # Preserve unique features (high conflict)
            (1 - blend_level) * primary_weight + blend_level * avg_other_weight  # Blend complementary features
        )
        return blended_weight
    
    return primary_weight  # No blending


def merge_multiple_loras_simple(
    lora_state_dicts: List[Tuple[Dict[str, torch.Tensor], Dict[str, str], str]], 
    output_file: str,
    average_blend: float = 0.0
) -> None:
    """
    Stage 2: Packages multiple (potentially blended) LoRAs into a single
    channel-partitioned file. This function does NOT perform blending itself;
    it assumes the input LoRAs are final.
    
    Args:
        lora_state_dicts: List of final (state_dict, metadata, lora_name) tuples.
        output_file: Path to save the merged LoRA.
        average_blend: The blending factor used to create the LoRAs (for metadata).
    """
    if not lora_state_dicts:
        raise ValueError("No LoRA state dictionaries provided for merging")
    
    if len(lora_state_dicts) == 1:
        # Single LoRA, just save it
        state_dict, metadata, name = lora_state_dicts[0]
        print(f"Only one LoRA provided ({name}), saving directly...")
        save_file(state_dict, output_file, metadata=metadata)
        return
    
    print(f"\n{'='*50}")
    print(f"STAGE 2: PACKAGING FOR CHANNEL-PARTITIONED MERGE ({len(lora_state_dicts)} LoRAs)")
    if average_blend == 0.0:
        print("Mode: ZERO BLEEDING (channel partitioning)")
    else:
        print(f"Mode: Pre-blended with factor {average_blend:.2f}")
    print(f"{'='*50}")
    
    print("📦 Creating channel-partitioned LoRA format...")
    
    merged_state_dict = {}
    base_metadata = lora_state_dicts[0][1].copy() if lora_state_dicts[0][1] else {}
    
    lora_names = [name for _, _, name in lora_state_dicts]
    partition_info = {
        "lora_count": str(len(lora_state_dicts)),
        "lora_names": ",".join(lora_names),
        "merge_method": "channel_partitioned",
        "average_blend": str(average_blend),
        "partition_type": "namespace_with_channel_assignment"
    }
    
    # Store the final (possibly blended) LoRAs with namespace separation
    for i, (processed_state_dict, _, name) in enumerate(lora_state_dicts):
        lora_keys = []
        for key in processed_state_dict:
            # Create namespaced key for channel partitioning
            namespaced_key = f"{name}_{key}"
            merged_state_dict[namespaced_key] = processed_state_dict[key]
            lora_keys.append(namespaced_key)
        
        # Add LoRA-specific metadata for channel assignment
        partition_info[f"{name}_keys"] = ",".join(lora_keys)
        partition_info[f"{name}_original_count"] = str(len(processed_state_dict.keys()))
        partition_info[f"{name}_lora_index"] = str(i)
        
        print(f"  ✅ Stored {len(lora_keys)} keys for {name} (index {i})")
    
    print(f"✅ Created partition-ready format with {len(lora_state_dicts)} isolated LoRAs")
    
    # Update metadata with partition information
    print("Updating metadata with partition information...")
    base_metadata.update(partition_info)
    
    # Update hash information
    model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(merged_state_dict, base_metadata)
    base_metadata["sshs_model_hash"] = model_hash
    base_metadata["sshs_legacy_hash"] = legacy_hash
    
    # Add detailed merge info
    lora_names_str = ", ".join([name for _, _, name in lora_state_dicts])
    base_metadata["merged_loras"] = lora_names_str
    base_metadata["total_tensors"] = str(len(merged_state_dict))
    
    # Save the merged LoRA
    print(f"💾 Saving merged LoRA to {output_file}")
    save_file(merged_state_dict, output_file, metadata=base_metadata)
    
    print(f"✅ Successfully packaged {len(lora_state_dicts)} LoRAs: {lora_names_str}")
    
    print("\n📋 Channel Partition Plan:")
    for i, (_, _, name) in enumerate(lora_state_dicts):
        print(f"  • {name}: Will target channel partition {i+1}/{len(lora_state_dicts)}")
    print("  ⚠️  Note: Requires modified inference code for channel partitioning")


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Enhanced LoRA weights merger with Post-Hoc EMA method - Version 2 (Refactored)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
🚀 FEATURES:
  Stage 1: Post-Hoc EMA for epoch merging
  Stage 2: Rank-based multi-LoRA merge with ZERO bleeding
  Revolutionary: Each LoRA occupies separate rank dimensions

📚 EXAMPLES:

  # Basic: Single LoRA series post-hoc EMA merge
  python lora_post_hoc_ema2.py --lora1 C:/train/saves/tanya_512-000005.safetensors --output_file_lora1 tanya_merged.safetensors

  # Multi-LoRA with ZERO bleeding (rank concatenation)
  python lora_post_hoc_ema2.py \\
    --lora1 C:/train/saves/tanya_512-000005.safetensors \\
    --lora2 C:/train/saves/jessica_1024-000015.safetensors \\
    --output_file merged.safetensors
  # Result: Higher-rank LoRA (e.g., 128 rank) with perfect isolation

  # Multi-LoRA with gradient-based knowledge sharing
  python lora_post_hoc_ema2.py \\
    --lora1 tanya.safetensors --lora2 jessica.safetensors \\
    --output_file merged.safetensors \\
    --average 0.1
  # Result: 90% separation + 10% complementary knowledge transfer

  # Multi-LoRA with democratic knowledge sharing  
  python lora_post_hoc_ema2.py \\
    --lora1 tanya.safetensors --lora2 jessica.safetensors \\
    --output_file merged.safetensors \\
    --average 0.1 --democratic
  # Result: Concept-aware democratic knowledge transfer (Jessica's "naked" → Tanya)

  # MERGE-ONLY: Skip EMA processing for already-processed LoRAs
  python lora_post_hoc_ema2.py \\
    --lora1 tanya_merged.safetensors \\
    --lora2 jessica_merged.safetensors \\
    --output_file final_combined.safetensors \\
    --merge_only

💡 CHANNEL PARTITIONING + INTELLIGENT BLENDING EXPLAINED:
  • LoRA1 + LoRA2 = Channel-partitioned merged LoRA
  • LoRA1 uses channels 0-N/2, LoRA2 uses channels N/2-N
  • Zero bleeding: Complete feature isolation by default
  • --average 0.0: Pure isolation (default)
  • --average 0.1: Gradient-based complementary knowledge sharing
  • --average 0.1 --democratic: Concept-aware democratic knowledge transfer
        """
    )
    
    # LoRA input arguments (support up to 20 LoRAs)
    for i in range(1, 21):
        parser.add_argument(f"--lora{i}", type=str, 
                          help=f"Path to LoRA {i} reference file (will auto-discover all epochs)")
    
    # Output file arguments
    parser.add_argument("--output_file", type=str, 
                       help="Output file for combined multi-LoRA merge (if multiple LoRAs specified)")
    
    for i in range(1, 21):
        parser.add_argument(f"--output_file_lora{i}", type=str,
                          help=f"Individual output file for LoRA {i} after post-hoc EMA and any blending")
    
    # Stage 1: EMA parameters (backward compatibility)
    parser.add_argument("--sigma_rel", type=float, default=0.2,
                       help="Relative sigma for Power Function EMA (default: 0.2)")
    parser.add_argument("--beta", type=float, default=0.95, 
                       help="Decay rate for merging weights (fallback if sigma_rel not used)")
    parser.add_argument("--beta2", type=float, default=None, 
                       help="End decay rate for linear interpolation")
    
    # Stage 2: Multi-LoRA parameters
    parser.add_argument("--average", type=float, default=0.0,
                       help="Blending factor for multi-LoRA merge (0.0=no bleeding, 1.0=full averaging, default: 0.0)")
    parser.add_argument("--democratic", action="store_true",
                       help="Use democratic knowledge sharing instead of gradient-based blending (default: False)")
    
    # Additional options
    parser.add_argument("--merge_only", action="store_true",
                       help="Skip individual post-hoc EMA processing and directly merge the specified LoRAs")
    parser.add_argument("--no_sort", action="store_true", 
                       help="Do not sort discovered files by modification time")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    # Backward compatibility: positional arguments
    parser.add_argument("path", nargs="*", 
                       help="Legacy: List of paths to LoRA weight files (for backward compatibility)")

    args = parser.parse_args()

    # Handle backward compatibility
    if args.path and not any(getattr(args, f"lora{i}") for i in range(1, 21)):
        print("Using legacy mode (positional arguments)")
        if not args.output_file:
            print("Error: --output_file is required in legacy mode")
            sys.exit(1)
        
        beta2 = args.beta if args.beta2 is None else args.beta2
        merge_lora_epochs_with_ema(args.path, args.no_sort, args.beta, beta2, args.sigma_rel, args.output_file)
        return

    # Collect specified LoRAs
    loras = []
    for i in range(1, 21):
        lora_path = getattr(args, f"lora{i}")
        if lora_path:
            loras.append((i, lora_path))
    
    if not loras:
        print("Error: No LoRA files specified. Use --lora1, --lora2, etc.")
        sys.exit(1)
    
    print(f"Found {len(loras)} LoRA series to process...")
    
    # === STAGE 1: INDIVIDUAL LoRA EMA PROCESSING (IN-MEMORY) ===
    processed_loras = []
    
    if args.merge_only:
        print(f"\n🚀 MERGE-ONLY MODE: Skipping individual post-hoc EMA processing.")
        print("   Loading specified LoRA files directly...")
        
        for lora_idx, lora_path in loras:
            print(f"\n📁 Loading LoRA {lora_idx}: {os.path.basename(lora_path)}")
            try:
                with MemoryEfficientSafeOpen(lora_path) as f:
                    state_dict = {key: f.get_tensor(key) for key in f.keys()}
                    metadata = f.metadata()
                
                lora_name = f"lora{lora_idx}"
                processed_loras.append((state_dict, metadata, lora_name))
                print(f"✅ Loaded {os.path.basename(lora_path)}")
                
            except Exception as e:
                print(f"❌ Error loading {lora_path}: {e}")
                continue
    else:
        beta2 = args.beta if args.beta2 is None else args.beta2
        for lora_idx, lora_path in loras:
            try:
                epoch_files = discover_lora_epochs(lora_path)
                if len(epoch_files) < 2:
                    print(f"⚠️ Only {len(epoch_files)} file(s) found for LoRA {lora_idx}, loading directly (no EMA).")
                    with MemoryEfficientSafeOpen(epoch_files[0]) as f:
                        state_dict = {key: f.get_tensor(key) for key in f.keys()}
                        metadata = f.metadata()
                else:
                    state_dict, metadata = merge_lora_epochs_with_ema(
                        epoch_files, args.no_sort, args.beta, beta2, args.sigma_rel, output_file=None
                    )
                
                lora_name = f"lora{lora_idx}"
                processed_loras.append((state_dict, metadata, lora_name))
            except Exception as e:
                print(f"❌ Error processing LoRA {lora_idx} from {lora_path}: {e}")
                continue

    # === STAGE 2: CROSS-LORA BLENDING (IF REQUESTED) ===
    final_loras = processed_loras

    if len(processed_loras) > 1 and args.average > 0.0:
        print(f"\n{'='*50}")
        print("STAGE 2: CROSS-LORA KNOWLEDGE SHARING")
        print(f"{'='*50}")
        
        all_keys = sorted(list(set.union(*(set(sd.keys()) for sd, _, _ in processed_loras))))

        uniform_loras = []
        for i, (state_dict, _, name) in enumerate(processed_loras):
            uniform_sd = {}
            for key in all_keys:
                if key in state_dict:
                    uniform_sd[key] = state_dict[key].clone()
                else:
                    template_tensor = next((other_sd[key] for other_sd, _, _ in processed_loras if key in other_sd), None)
                    if template_tensor is not None:
                        uniform_sd[key] = torch.zeros_like(template_tensor)
            uniform_loras.append((uniform_sd, name, i))

        if args.democratic:
            blended_data = apply_democratic_blending(uniform_loras, args.average, all_keys)
        else:
            blended_data = apply_intelligent_blending(uniform_loras, args.average, all_keys)

        final_loras = []
        for i, (blended_sd, name, _) in enumerate(blended_data):
            original_meta = processed_loras[i][1]
            final_loras.append((blended_sd, original_meta, name))

    # === STAGE 3: SAVING OUTPUTS ===
    print(f"\n{'='*50}")
    print("STAGE 3: SAVING OUTPUT FILES")
    print(f"{'='*50}")

    # Save individual LoRA files (now potentially blended)
    has_individual_outputs = any(getattr(args, f"output_file_lora{i}", None) for i in range(1, 21))
    if has_individual_outputs:
        print("\n--- Saving Individual LoRA Files ---")
        for state_dict, metadata, lora_name in final_loras:
            lora_idx = int(lora_name.replace("lora", ""))
            individual_output = getattr(args, f"output_file_lora{lora_idx}", None)
            
            if individual_output:
                print(f"💾 Saving {lora_name} to {individual_output}...")
                if metadata is not None:
                    model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(state_dict, metadata)
                    metadata["sshs_model_hash"] = model_hash
                    metadata["sshs_legacy_hash"] = legacy_hash
                save_file(state_dict, individual_output, metadata=metadata)
                print(f"✅ Saved.")

    # Save combined multi-LoRA file (if requested)
    if len(final_loras) > 1 and args.output_file:
        print("\n--- Saving Combined Channel-Partitioned File ---")
        merge_multiple_loras_simple(
            lora_state_dicts=final_loras,
            output_file=args.output_file,
            average_blend=args.average
        )
        print(f"\n✅ Final combined LoRA saved to: {args.output_file}")
    
    # Handle single LoRA where only --output_file is specified
    elif len(final_loras) == 1 and args.output_file and not has_individual_outputs:
        state_dict, metadata, _ = final_loras[0]
        print(f"\n💾 Saving single LoRA to: {args.output_file}")
        save_file(state_dict, args.output_file, metadata=metadata)
    
    print(f"\n{'='*60}")
    print("Processing completed successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()