def should_use_unconditional_dit_for_lora_sampling(args) -> bool:
    return bool(getattr(args, "unconditional_dit", None) and getattr(args, "use_unconditional_dit_for_lora_sampling", False))
