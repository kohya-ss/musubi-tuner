from __future__ import annotations

from typing import Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_longcat_text_encoder(batch_tensor_data: Dict[str, List[torch.Tensor]], *, key: str = "t5") -> None:
    """
    Pad variable-length LongCat text encoder embeddings and masks so they can be stacked.
    """
    if key not in batch_tensor_data:
        return

    embeds_entry = batch_tensor_data.pop(key)
    masks_entry = batch_tensor_data.pop(f"{key}_mask", None)

    embeds_list = embeds_entry if isinstance(embeds_entry, list) else [embeds_entry]
    if masks_entry is None:
        masks_list = None
    elif isinstance(masks_entry, list):
        masks_list = masks_entry
    else:
        masks_list = [masks_entry]

    processed_embeds: List[torch.Tensor] = []
    processed_masks: List[torch.Tensor] = []

    for idx, embed in enumerate(embeds_list):
        if embed.dim() == 3 and embed.shape[0] == 1:
            embed = embed.squeeze(0)
        elif embed.dim() == 1:
            embed = embed.unsqueeze(0)

        if embed.dim() != 2:
            raise ValueError(f"Unexpected LongCat text encoder shape: {embed.shape}")

        if masks_list is not None and idx < len(masks_list):
            mask = masks_list[idx]
            if mask.dim() > 1:
                mask = mask.view(-1)
            mask = mask.to(torch.bool)
        else:
            mask = torch.ones(embed.shape[0], dtype=torch.bool)

        valid_length = int(mask.sum().item())
        processed_embeds.append(embed[:valid_length])
        processed_masks.append(mask[:valid_length])

    padded_embeds = pad_sequence(processed_embeds, batch_first=True, padding_value=0)
    padded_masks = pad_sequence(processed_masks, batch_first=True, padding_value=False)

    batch_tensor_data[key] = padded_embeds
    batch_tensor_data[f"{key}_mask"] = padded_masks
