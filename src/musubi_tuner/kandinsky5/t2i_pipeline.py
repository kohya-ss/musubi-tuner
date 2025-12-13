# This file includes code derived from:
# https://github.com/kandinskylab/kandinsky-5
# Copyright (c) 2025 Kandinsky Lab
# Licensed under the MIT License

from typing import Union, Optional

import transformers
import torch
from torchvision.transforms import ToPILImage
from .generation_utils import generate_sample
from .i2i_pipeline import Kandinsky5I2IPipeline

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True


class Kandinsky5T2IPipeline(Kandinsky5I2IPipeline):
    def expand_prompt(self, prompt, image=None):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Rewrite and enhance the original prompt with richer detail, clearer structure, and improved descriptive quality. Expand the scene, atmosphere, and context while preserving the user’s intent. When adding text that should appear inside an image, place that text inside double quotes and in capital letters. Strengthen visual clarity, style, and specificity, but do not change the meaning. Output only the enhanced prompt, written in polished, vivid language suitable for high-quality image generation.
        example:
        Original text: white mini police car with blue stripes, with 911 and 'Police' text 
        Result: A miniature model car simulating the official transport of the relevant authorities. The body is white with blue stripes. The word "911" is written in large blue letters on the hood and side. Below it, "POLICE" is used in a font. The windows are transparent, and the interior has black seats. The headlights have plastic lenses, and the roof has blue and red beacons. The radiator grille has vertical slots. The wheels are black with white rims. The doors are closed, the windows have black frames. The background is uniform white.
        Here 911 in double quotes because it is text on image, 'Police' -> "POLICE" because it should be in double quotes and capital letters.
        Rewrite Prompt: "{prompt}". Answer only with expanded prompt.""",
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
        width: int = 1024,
        height: int = 1024,
        seed: int = None,
        num_steps: Optional[int] = None,
        guidance_weight: Optional[float] = None,
        scheduler_scale: float = 3.0,
        negative_caption: str = "",
        expand_prompts: bool = True,
        save_path: str = None,
        progress: bool = True,
    ):
        return super().__call__(
            text=text,
            width=width,
            height=height,
            seed=seed,
            num_steps=num_steps,
            guidance_weight=guidance_weight,
            scheduler_scale=scheduler_scale,
            negative_caption=negative_caption,
            expand_prompts=expand_prompts,
            save_path=save_path,
            progress=progress,
        )
