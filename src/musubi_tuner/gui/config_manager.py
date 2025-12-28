import os


class ConfigManager:
    def __init__(self):
        self.models = {
            "Z-Image-Turbo": {
                "resolution": (1024, 1024),
                "vae_subpath": ["vae", "ae.safetensors"],
                "te1_subpath": ["text_encoders", "qwen_3_4b.safetensors"],
                "te2_subpath": None,
                "dit_subpath": ["diffusion_models", "z_image_de_turbo_v1_bf16.safetensors"],
                "training_base": {
                    "learning_rate": 1e-3,
                    "default_num_steps": 500,
                    "save_every_n_epochs": 1,
                    "discrete_flow_shift": 2.0,
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": True,
                    "fp8_scaled": True,
                    "fp8_llm": True,
                },
                "vram_settings": {
                    "12": {"batch_size": 1, "block_swap": 0, "fp8_scaled": True, "fp8_llm": True},
                    "16": {"batch_size": 2, "block_swap": 0, "fp8_scaled": True, "fp8_llm": False},
                    "24": {"batch_size": 1, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                    "32": {"batch_size": 2, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                    ">32": {"batch_size": 8, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                },
            },
            "Qwen-Image": {
                "resolution": (1328, 1328),
                "vae_subpath": ["vae", "qwen_image_vae.safetensors"],
                "te1_subpath": ["text_encoders", "qwen_2.5_vl_7b.safetensors"],
                "te2_subpath": None,
                "dit_subpath": ["diffusion_models", "qwen_image_bf16.safetensors"],
                "training_base": {
                    "learning_rate": 1e-3,
                    "default_num_steps": 1000,
                    "save_every_n_epochs": 1,
                    "discrete_flow_shift": 2.2,
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": True,
                    "fp8_scaled": True,
                    "fp8_llm": True,
                },
                "vram_settings": {
                    "12": {"batch_size": 1, "block_swap": 46, "fp8_scaled": True, "fp8_llm": True},
                    "16": {"batch_size": 1, "block_swap": 34, "fp8_scaled": True, "fp8_llm": True},
                    "24": {"batch_size": 1, "block_swap": 10, "fp8_scaled": True, "fp8_llm": False},
                    "32": {"batch_size": 2, "block_swap": 0, "fp8_scaled": True, "fp8_llm": False},
                    ">32": {"batch_size": 2, "block_swap": 0, "fp8_scaled": False, "fp8_llm": False},
                },
            },
            "Wan 2.1 (T2V-14B)": {
                "resolution": (1280, 720),
                "vae_subpath": ["vae", "wan_vae.safetensors"],
                "te1_subpath": ["text_encoders", "google_umt5-xxl"],
                "te2_subpath": ["text_encoders", "clip-vit-large-patch14"],
                "dit_subpath": ["diffusion_models", "wan2.1_t2v_14b_bf16.safetensors"],
                "training_base": {
                    "learning_rate": 2e-4,
                    "discrete_flow_shift": 1.0,
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": True,
                },
                "vram_settings": {
                    "24": {"batch_size": 1, "block_swap": 20, "fp8_scaled": True},
                    "32": {"batch_size": 2, "block_swap": 10, "fp8_scaled": True},
                    ">32": {"batch_size": 4, "block_swap": 0},
                },
            },
            "Wan 2.1 (I2V-14B)": {
                "resolution": (1280, 720),
                "vae_subpath": ["vae", "wan_vae.safetensors"],
                "te1_subpath": ["text_encoders", "google_umt5-xxl"],
                "te2_subpath": ["text_encoders", "clip-vit-large-patch14"],
                "dit_subpath": ["diffusion_models", "wan2.1_i2v_14b_bf16.safetensors"],
                "training_base": {
                    "learning_rate": 2e-4,
                    "discrete_flow_shift": 1.0,
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": True,
                },
                "vram_settings": {
                    "24": {"batch_size": 1, "block_swap": 20, "fp8_scaled": True},
                    "32": {"batch_size": 2, "block_swap": 10, "fp8_scaled": True},
                    ">32": {"batch_size": 4, "block_swap": 0},
                },
            },
            "Wan 2.2 (T2V-5B)": {
                "resolution": (1024, 1024),
                "vae_subpath": ["vae", "wan_2.2_vae.safetensors"],
                "te1_subpath": ["text_encoders", "google_umt5-xxl"],
                "te2_subpath": ["text_encoders", "clip-vit-large-patch14"],
                "dit_subpath": ["diffusion_models", "wan2.2_t2v_5b_bf16.safetensors"],
                "training_base": {
                    "learning_rate": 2e-4,
                    "discrete_flow_shift": 1.0,
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": True,
                },
                "vram_settings": {
                    "16": {"batch_size": 1, "block_swap": 20, "fp8_scaled": True},
                    "24": {"batch_size": 1, "block_swap": 0, "fp8_scaled": True},
                    ">32": {"batch_size": 4, "block_swap": 0},
                },
            },
            "Wan 2.2 (T2V-14B)": {
                "resolution": (1280, 720),
                "vae_subpath": ["vae", "wan_2.2_vae.safetensors"],
                "te1_subpath": ["text_encoders", "google_umt5-xxl"],
                "te2_subpath": ["text_encoders", "clip-vit-large-patch14"],
                "dit_subpath": ["diffusion_models", "wan2.2_t2v_14b_bf16.safetensors"],
                "training_base": {
                    "learning_rate": 2e-4,
                    "discrete_flow_shift": 1.0,
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": True,
                },
                "vram_settings": {
                    "24": {"batch_size": 1, "block_swap": 20, "fp8_scaled": True},
                    "32": {"batch_size": 2, "block_swap": 10, "fp8_scaled": True},
                    ">32": {"batch_size": 4, "block_swap": 0},
                },
            },
            "Wan 2.2 (I2V-14B)": {
                "resolution": (1280, 720),
                "vae_subpath": ["vae", "wan_2.2_vae.safetensors"],
                "te1_subpath": ["text_encoders", "google_umt5-xxl"],
                "te2_subpath": ["text_encoders", "clip-vit-large-patch14"],
                "dit_subpath": ["diffusion_models", "wan2.2_i2v_14b_bf16.safetensors"],
                "training_base": {
                    "learning_rate": 2e-4,
                    "discrete_flow_shift": 1.0,
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": True,
                },
                "vram_settings": {
                    "24": {"batch_size": 1, "block_swap": 20, "fp8_scaled": True},
                    "32": {"batch_size": 2, "block_swap": 10, "fp8_scaled": True},
                    ">32": {"batch_size": 4, "block_swap": 0},
                },
            },
            "HunyuanVideo": {
                "resolution": (960, 544),
                "vae_subpath": ["vae", "hunyuan_video_vae_bf16.safetensors"],
                "te1_subpath": ["text_encoders", "llm"],
                "te2_subpath": ["text_encoders", "clip"],
                "dit_subpath": ["diffusion_models", "hunyuan_video_7b_bf16.safetensors"],
                "training_base": {
                    "learning_rate": 1e-4,
                    "discrete_flow_shift": 7.0,
                    "mixed_precision": "bf16",
                    "gradient_checkpointing": True,
                },
                "vram_settings": {
                    "24": {"batch_size": 1, "block_swap": 36, "fp8_scaled": True, "fp8_llm": True},
                    "32": {"batch_size": 1, "block_swap": 18, "fp8_scaled": True},
                    ">32": {"batch_size": 4, "block_swap": 0},
                },
            },
            "Hunyuan 1.5": {
                "resolution": (1280, 720),
                "vae_subpath": ["vae", "hunyuan_video1.5_vae.safetensors"],
                "te1_subpath": ["text_encoders", "qwen2.5_vl_7b"],
                "te2_subpath": ["text_encoders", "byt5"],
                "dit_subpath": ["diffusion_models", "hunyuan_video1.5_t2v_7b.safetensors"],
                "training_base": {
                    "learning_rate": 2e-4,
                    "discrete_flow_shift": 7.0,
                    "mixed_precision": "bf16",
                },
                "vram_settings": {
                    "24": {"batch_size": 1, "block_swap": 51, "fp8_scaled": True},
                    ">32": {"batch_size": 2, "block_swap": 0},
                },
            },
            "FramePack": {
                "resolution": (1280, 720),
                "vae_subpath": ["vae", "fpack_vae.safetensors"],
                "te1_subpath": ["text_encoders", "llama"],
                "te2_subpath": ["text_encoders", "clip"],
                "dit_subpath": ["diffusion_models", "fpack_dit.safetensors"],
                "training_base": {
                    "learning_rate": 1e-4,
                    "mixed_precision": "bf16",
                },
                "vram_settings": {
                    "24": {"batch_size": 1, "block_swap": 38},
                    ">32": {"batch_size": 4, "block_swap": 0},
                },
            },
            "Flux.1 Kontext": {
                "resolution": (1024, 1024),
                "vae_subpath": ["vae", "ae.safetensors"],
                "te1_subpath": ["text_encoders", "t5-v1_1-xxl"],
                "te2_subpath": ["text_encoders", "clip-vit-large-patch14"],
                "dit_subpath": ["diffusion_models", "flux1-dev-kontext.safetensors"],
                "training_base": {
                    "learning_rate": 4e-4,
                    "discrete_flow_shift": 3.0,
                    "mixed_precision": "bf16",
                },
                "vram_settings": {
                    "24": {"batch_size": 1, "block_swap": 10, "fp8_scaled": True},
                    ">32": {"batch_size": 2, "block_swap": 0},
                },
            },
            "Kandinsky 5": {
                "resolution": (1024, 1024),
                "vae_subpath": ["vae", "k5_vae.safetensors"],
                "te1_subpath": ["text_encoders", "k5_te1.safetensors"],
                "te2_subpath": ["text_encoders", "k5_te2.safetensors"],
                "dit_subpath": ["diffusion_models", "k5_dit.safetensors"],
                "training_base": {
                    "learning_rate": 1e-4,
                    "discrete_flow_shift": 1.0,
                    "mixed_precision": "bf16",
                },
                "vram_settings": {
                    "24": {"batch_size": 1, "block_swap": 20, "fp8_scaled": True},
                    ">32": {"batch_size": 4, "block_swap": 0},
                },
            },
        }

    def get_resolution(self, model_name):
        return self.models.get(model_name, {}).get("resolution", (1024, 1024))

    def get_batch_size(self, model_name, vram_size):
        if not vram_size:
            vram_size = "16"

        vram_conf = self.models.get(model_name, {}).get("vram_settings", {}).get(vram_size, {})
        if not vram_conf:
            # Try to find a fallback if exact size not matched (e.g. if 16 is selected but only 24+ defined)
            vram_conf = self.models.get(model_name, {}).get("vram_settings", {}).get("24", {})
        
        return vram_conf.get("batch_size", 1)

    def get_preprocessing_paths(self, model_name, comfy_models_dir):
        conf = self.models.get(model_name, {})
        vae = conf.get("vae_subpath")
        te1 = conf.get("te1_subpath")
        te2 = conf.get("te2_subpath")

        def join_path(subpath):
            if subpath and comfy_models_dir:
                return os.path.join(comfy_models_dir, *subpath)
            return ""

        return join_path(vae), join_path(te1), join_path(te2)

    def get_training_defaults(self, model_name, vram_size, comfy_models_dir):
        conf = self.models.get(model_name, {})
        base = conf.get("training_base", {}).copy()

        if not vram_size:
            vram_size = "16"

        # Merge VRAM settings
        vram_conf = conf.get("vram_settings", {}).get(vram_size, {})
        if not vram_conf:
             vram_conf = conf.get("vram_settings", {}).get("24", {})

        # Only take relevant keys for training params (block_swap), ignore batch_size as it's for dataset
        for key in ["block_swap", "fp8_llm", "fp8_scaled"]:
            if key in vram_conf:
                base[key] = vram_conf[key]

        # DiT path
        dit_sub = conf.get("dit_subpath")
        if dit_sub and comfy_models_dir:
            base["dit_path"] = os.path.join(comfy_models_dir, *dit_sub)
        else:
            base["dit_path"] = ""

        return base
