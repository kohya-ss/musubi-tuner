# This file includes code derived from:
# https://github.com/kandinskylab/kandinsky-5
# Copyright (c) 2025 Kandinsky Lab
# Licensed under the MIT License

import json
import logging
import struct
from typing import Optional, Union

import transformers
import torch
from torch.distributed.device_mesh import DeviceMesh
import torchvision
from peft import PeftConfig, LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
from safetensors.torch import load_file
from torchvision.transforms import ToPILImage

from .generation_utils import generate_sample

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True


def read_safetensors_json(file_path):
    """Reads the metadata (JSON header) from a safetensors file."""
    with open(file_path, "rb") as f:
        # Step 1: Read the first 8 bytes to get the size of the header (N)
        header_size_bytes = f.read(8)
        header_size = struct.unpack("Q", header_size_bytes)[0]  # 'Q' is for unsigned 64-bit integer

        # Step 2: Read the next N bytes which contain the JSON header
        header_bytes = f.read(header_size)
        header_str = header_bytes.decode("utf-8")

        # Step 3: Parse the JSON header
        header = json.loads(header_str)
        return header


class Kandinsky5T2VPipeline:
    RESOLUTIONS = {
        512: [(512, 512), (512, 768), (768, 512)],
        1024: [(1024, 1024), (640, 1408), (1408, 640), (768, 1280), (1280, 768), (896, 1152), (1152, 896)],
    }

    def __init__(
        self,
        device_map: Union[str, torch.device, dict],  # {"dit": cuda:0, "vae": cuda:1, "text_embedder": cuda:1 }
        dit,
        text_embedder,
        vae,
        local_dit_rank: int = 0,
        world_size: int = 1,
        conf=None,
        offload: bool = False,
        device_mesh: DeviceMesh = None,
    ):
        self.dit = dit
        self.text_embedder = text_embedder
        self.vae = vae

        self.resolution = conf.metrics.resolution

        self.device_map = device_map
        self.local_dit_rank = local_dit_rank
        self.world_size = world_size
        self.conf = conf
        self.num_steps = conf.model.num_steps
        self.guidance_weight = conf.model.guidance_weight

        self.offload = offload

        self._hf_peft_config_loaded = False
        self.peft_config = {}
        self.peft_triggers = {}
        self.peft_trigger = ""
        self.device_mesh = device_mesh

    def expand_prompt(self, prompt):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are a prompt beautifier that transforms short user video descriptions into rich, detailed English prompts specifically optimized for video generation models.
        Here are some example descriptions from the dataset that the model was trained:
        1. "In a dimly lit room with a cluttered background, papers are pinned to the wall and various objects rest on a desk. Three men stand present: one wearing a red sweater, another in a black sweater, and the third in a gray shirt. The man in the gray shirt speaks and makes hand gestures, while the other two men look forward. The camera remains stationary, focusing on the three men throughout the sequence. A gritty and realistic visual style prevails, marked by a greenish tint that contributes to a moody atmosphere. Low lighting casts shadows, enhancing the tense mood of the scene."
        2. "In an office setting, a man sits at a desk wearing a gray sweater and seated in a black office chair. A wooden cabinet with framed pictures stands beside him, alongside a small plant and a lit desk lamp. Engaged in a conversation, he makes various hand gestures to emphasize his points. His hands move in different positions, indicating different ideas or points. The camera remains stationary, focusing on the man throughout. Warm lighting creates a cozy atmosphere. The man appears to be explaining something. The overall visual style is professional and polished, suitable for a business or educational context."
        3. "A person works on a wooden object resembling a sunburst pattern, holding it in their left hand while using their right hand to insert a thin wire into the gaps between the wooden pieces. The background features a natural outdoor setting with greenery and a tree trunk visible. The camera stays focused on the hands and the wooden object throughout, capturing the detailed process of assembling the wooden structure. The person carefully threads the wire through the gaps, ensuring the wooden pieces are securely fastened together. The scene unfolds with a naturalistic and instructional style, emphasizing the craftsmanship and the methodical steps taken to complete the task."
        IImportantly! These are just examples from a large training dataset of 200 million videos.
        Rewrite Prompt: "{prompt}" to get high-quality video generation. Answer only with expanded prompt.""",
                    },
                ],
            }
        ]
        text = self.text_embedder.embedder.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.text_embedder.embedder.processor(
            text=[text],
            images=None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.text_embedder.embedder.model.device)
        generated_ids = self.text_embedder.embedder.model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.text_embedder.embedder.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def __call__(
        self,
        text: str,
        time_length: int = 5,  # time in seconds 0 if you want generate image
        width: int = 768,
        height: int = 512,
        seed: int = None,
        num_steps: int = None,
        guidance_weight: float = None,
        scheduler_scale: float = 10.0,
        negative_caption: str = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards",
        expand_prompts: bool = True,
        save_path: str = None,
        progress: bool = True,
    ):
        num_steps = self.num_steps if num_steps is None else num_steps
        guidance_weight = self.guidance_weight if guidance_weight is None else guidance_weight
        # SEED
        if seed is None:
            if self.local_dit_rank == 0:
                seed = torch.randint(2**32 - 1, (1,)).to(self.local_dit_rank)
            else:
                seed = torch.empty((1,), dtype=torch.int64).to(self.local_dit_rank)

            if self.world_size > 1:
                torch.distributed.broadcast(seed, 0)

            seed = seed.item()

        if (height, width) not in self.RESOLUTIONS[self.resolution]:
            raise ValueError(f"Wrong height, width pair. Available (height, width) are: {self.RESOLUTIONS[self.resolution]}")

        # PREPARATION
        num_frames = 1 if time_length == 0 else time_length * 24 // 4 + 1

        caption = text
        if expand_prompts:
            transformers.set_seed(seed)
            if self.local_dit_rank == 0:
                if self.offload:
                    self.text_embedder = self.text_embedder.to(self.device_map["text_embedder"])
                caption = self.peft_trigger + self.expand_prompt(caption)
            if self.world_size > 1:
                caption = [caption]
                torch.distributed.broadcast_object_list(caption, 0)
                caption = caption[0]

        shape = (1, num_frames, height // 8, width // 8, 16)

        # GENERATION
        images = generate_sample(
            shape,
            caption,
            self.dit,
            self.vae,
            self.conf,
            text_embedder=self.text_embedder,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            negative_caption=negative_caption,
            seed=seed,
            device=self.device_map["dit"],
            vae_device=self.device_map["vae"],
            text_embedder_device=self.device_map["text_embedder"],
            progress=progress,
            offload=self.offload,
            tp_mesh=self.device_mesh,
        )
        torch.cuda.empty_cache()

        if self.offload:
            self.text_embedder = self.text_embedder.to(device=self.device_map["text_embedder"])

        # RESULTS
        if self.local_dit_rank == 0:
            if time_length == 0:
                return_images = []
                for image in images.squeeze(2).cpu():
                    return_images.append(ToPILImage()(image))
                if save_path is not None:
                    if isinstance(save_path, str):
                        save_path = [save_path]
                    if len(save_path) == len(return_images):
                        for path, image in zip(save_path, return_images):
                            image.save(path)
                return return_images
            else:
                if save_path is not None:
                    if isinstance(save_path, str):
                        save_path = [save_path]
                    if len(save_path) == len(images):
                        for path, video in zip(save_path, images):
                            torchvision.io.write_video(
                                path,
                                video.float().permute(1, 2, 3, 0).cpu().numpy(),
                                fps=24,
                                options={"crf": "5"},
                            )
                return images

    def load_adapter(
        self,
        adapter_config: Union[PeftConfig, str],
        adapter_path: Optional[str] = None,
        adapter_name: Optional[str] = None,
        trigger: Optional[str] = None,
    ) -> None:
        if adapter_name is None:
            adapter_name = "default"
        if self._hf_peft_config_loaded and adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if not isinstance(adapter_config, PeftConfig):
            try:
                with open(adapter_config, "r") as f:
                    adapter_config = json.load(f)
                adapter_config = LoraConfig(**adapter_config)
            except:
                raise TypeError(f"adapter_config should be an instance of PeftConfig or a path to a json file.")
        self.peft_config[adapter_name] = adapter_config

        inject_adapter_in_model(adapter_config, self.dit, adapter_name)

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        adapter_state_dict = load_file(adapter_path)
        adapter_metadata = read_safetensors_json(adapter_path)
        if trigger is not None:
            self.peft_trigger = trigger
        else:
            if "__metadata__" in adapter_metadata and "trigger" in adapter_metadata["__metadata__"]:
                self.peft_trigger = adapter_metadata["__metadata__"]["trigger"]
            else:
                self.peft_trigger = ""
        self.peft_triggers[adapter_name] = self.peft_trigger

        processed_adapter_state_dict = {}
        for key, value in adapter_state_dict.items():
            new_key = key
            for prefix in ["base_model.model.", "transformer."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    break

            new_key = new_key.replace(".default", "")
            processed_adapter_state_dict[new_key] = value

        incompatible_keys = set_peft_model_state_dict(self.dit, processed_adapter_state_dict, adapter_name)

        if incompatible_keys is not None:
            err_msg = ""
            origin_name = "state_dict"
            # Check for unexpected keys.
            if hasattr(incompatible_keys, "unexpected_keys") and len(incompatible_keys.unexpected_keys) > 0:
                err_msg = (
                    f"Loading adapter weights from {origin_name} led to unexpected keys not found in the model: "
                    f"{', '.join(incompatible_keys.unexpected_keys)}. "
                )

            # Check for missing keys.
            missing_keys = getattr(incompatible_keys, "missing_keys", None)
            if missing_keys:
                # Filter missing keys specific to the current adapter, as missing base model keys are expected.
                lora_missing_keys = [k for k in missing_keys if "lora_" in k and adapter_name in k]
                if lora_missing_keys:
                    err_msg += (
                        f"Loading adapter weights from {origin_name} led to missing keys in the model: "
                        f"{', '.join(lora_missing_keys)}"
                    )

            if err_msg:
                logging.warning(err_msg)

        self.set_adapter(adapter_name)

    def set_adapter(self, adapter_name: Union[list[str], str]) -> None:
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        elif isinstance(adapter_name, list):
            missing = set(adapter_name) - set(self.peft_config)
            if len(missing) > 0:
                raise ValueError(
                    f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                    f" current loaded adapters are: {list(self.peft_config.keys())}"
                )
        elif adapter_name not in self.peft_config:
            raise ValueError(
                f"Adapter with name {adapter_name} not found. Please pass the correct adapter name among {list(self.peft_config.keys())}"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        _adapters_has_been_set = False

        for _, module in self.dit.named_modules():
            if isinstance(module, BaseTunerLayer):
                # The recent version of PEFT need to call `enable_adapters` instead
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=True)
                else:
                    module.disable_adapters = False
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # For backward compatibility with previous PEFT versions
                if hasattr(module, "set_adapter"):
                    module.set_adapter(adapter_name)
                else:
                    module.active_adapter = adapter_name
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )
        self.peft_trigger = self.peft_triggers[adapter_name]

    def disable_adapters(self) -> None:
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        for _, module in self.dit.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                # The recent version of PEFT need to call `enable_adapters` instead
                if hasattr(module, "enable_adapters"):
                    module.enable_adapters(enabled=False)
                else:
                    module.disable_adapters = True
        self.peft_trigger = ""
