Optimal  Training Configuration for WAN2.2 LoRA using Blissful Tuner:

1. Your Role and Objective:

You are an expert AI model trainer with deep specialization in text-to-image diffusion models, the Hugging Face ecosystem (Diffusers, Transformers), and advanced fine-tuning techniques like LoRA and Fine tune training.

Your primary objective is to analyze my specific hardware, dataset, and goals to provide an optimal, ready-to-use configuration for training a Wan-AI/Wan2.2-T2V-A14B LoRA (https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B) of myself using the Blissfull Tuner (https://github.com/Sarania/blissful-tuner/), a fork of Musubi Tuner by kohya_ss (https://github.com/kohya-ss/musubi-tuner). Please expect to also answer specific questions regarding my training strategy and the capabilities of the provided framework.

2. My Goal & Setup:
Primary Goal: To train a persona Wan-AI/Wan2.2-T2V-A14B LoRA/LyCORIS that we can use to create either single frame image generations or even text to video generations of my likeness


Training Technique: LoRA 

Training Framework: Blissful Tuner

Dataset:

Content: 269 high-quality still images of myself (File location: /root/DATASETS/DLAY/wan)
Captioning: Each image has a corresponding .txt file describing the scene. The captions use the unique identifier "DLAY" to refer to me, alongside the class "man".
Masks: Each image has  a black and white mask: /root/DATASETS/DLAY/mask_weighted for masked loss training (if it even does)
Resolutions: The primary dataset is 1024x1024. 

3. Core Questions and Tasks:

Based on my setup and the full codebase provided below, please perform the following:

Appropriate Accelerate configuration given my current server hardware and software stack. 

Would you recommend any modifications? 

datasets: The correct folder_path and resolution configuration.

train: batch_size, gradient_accumulation, steps, optimizer, lr, dtype, and any VRAM-optimization flags like gradient_checkpointing or cache_text_embeddings.

network: The precise configuration for a finetune training run, including type, linear, linear_alpha, and any other relevant LyCORIS parameters from the codebase.

model: The optimal settings given my hardware, including quantization (quantize, qtype) and MOE parameters (train_high_noise, train_low_noise, switch_boundary_every).

Please include recommendaitons for any advanced and/or experimental training options for my training configuration.

Blissful Tuner fork of Musubi Tuner:

Does blissful tuner offer any additional features or options to diffusion model training compared to what musubi tuner on its own offers? Or does Blissful Tuner it only focus on inference and generating images? Are there blissful tuner features and options that you recommend I integrate into my Wan2.2-T2V-A14B training command?

Trigger Word Strategy:
My captions include "dlay man" to identify me. Is this sufficient for a Dreambooth-style training, or should I also be using the trigger_word parameter in the config? What is the best practice within this framework to ensure "dlay man" robustly triggers my likeness?

4. Knowledge Gap Analysis:

After reviewing the codebase and my questions, please identify any ambiguities or knowledge gaps that prevent you from providing a recommendation with full confidence. For example: Are there experimental features in blissful-tuner whose behavior is not fully documented? Are there any additional and/or advanced features that Blissful Tuner has available for training that you would recommend I employ in my training attempt? Are there any pitfalls that might affect stability or performance? I will do my best to provide clarity on these points if you highlight them. 

5. Instructions for Your Response:

Please assist me with setting up Blissful tuner for Wan-AI/Wan2.2-T2V-A14B LoRA training. 
Clearly explain your reasoning for key parameter choices (like batch_size, lr, and block swapping), directly referencing my GH200.
Answer each of the numbered questions (2, 3, and 4) and the knowledge gap analysis (5) in separate, clearly marked sections.
When discussing the code, reference specific file paths and function/class names from the provided codebase to support your analysis.
Please do not make any changes to codebase or install/uninstall/modify my environment/python modules during this review just yet. Please begin your analysis.