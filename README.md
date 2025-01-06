# Musubi Tuner

[English](./README.md) | [日本語](./README.ja.md)

This repository provides scripts for training LoRA (Low-Rank Adaptation) models with HunyuanVideo.

__This repository is under development.__

### Recent Updates

- 06, Jan, 2025
    - Added `--split_attn` option to `hv_train_network.py` and `hv_generate_video.py` to process attention in chunks. Inference with SageAttention is expected to be about 10% faster. There is almost no impact during training. If `--split_attn` is not specified, it will be processed in the conventional way. Cannot be specified when `attn_mode` is `flash`.

- 05, Jan, 2025
    - Added `images` to the save format in `hv_generate_video.py`. You can generate images from latents saved with `--latent_path`. You can also specify multiple latents with `--latent_path` for batch processing (increases VRAM usage).

- 04, Jan, 2025
    - Added support for loading Text Encoder weights from .safetensors files. See [Model Download](#model-download) for instructions.
    - Changed the format of latents saved by `hv_generate_video.py` to .safetensors. Metadata such as prompts will be saved in the .safetensors file. Use `--no_metadata` to disable saving metadata.

- 03, Jan, 2025: The noise initialization method during inference has changed. When the same seed is specified, the common frames will be the same even if the number of generated frames is different. Please note that the inference results will be different from before even with the same seed.

(For example, when 25 frames are specified, the time length of the latent is 7, and when 45 frames are specified, the time length of the latent is 12, but the first 7 frames of both will have the same noise value when the same seed is specified.)

### Hardware Requirements

- VRAM: 12GB or more recommended for image training, 24GB or more recommended for video training
    - Depends on resolution, etc. For 12GB, use a resolution of 960x544 or lower and use memory-saving options such as `--blocks_to_swap`, `--fp8_llm`, etc.
- Main Memory: 64GB or more recommended, 32GB + swap may work

### Features

- Memory-efficient implementation
- Windows compatible (Linux compatibility not yet verified)
- Multi-GPU support not implemented

## Installation

Create a virtual environment and install PyTorch and torchvision matching your CUDA version. Verified to work with version 2.5.1.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

Optionally, you can use FlashAttention and SageAttention (for inference only; see [SageAttention Installation](#sageattention-installation) for installation instructions).

Additionally, install `ascii-magic` (used for dataset verification), `matplotlib` (used for timestep visualization), and `tensorboard` (used for logging training progress) as needed:

```bash
pip install ascii-magic matplotlib tensorboard
```

## Model Download

There are two ways to download the model.

### Use the Official HunyuanVideo Model

Download the model following the [official README](https://github.com/Tencent/HunyuanVideo/blob/main/ckpts/README.md) and place it in your chosen directory with the following structure:

```
  ckpts
    ├──hunyuan-video-t2v-720p
    │  ├──transformers
    │  ├──vae
    ├──text_encoder
    ├──text_encoder_2
    ├──...
```

### Using ComfyUI Models for Text Encoder

This method is easier.

For DiT and VAE, use the HunyuanVideo models.

From https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/transformers, download [mp_rank_00_model_states.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt) and place it in your chosen directory.

(Note: The fp8 model on the same page is unverified.)

From https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/vae, download [pytorch_model.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/vae/pytorch_model.pt) and place it in your chosen directory.

For the Text Encoder, use the models provided by ComfyUI. Refer to [ComfyUI's page](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/), from https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/tree/main/split_files/text_encoders, download `llava_llama3_fp16.safetensors` (Text Encoder 1, LLM) and `clip_l.safetensors` (Text Encoder 2, CLIP)  and place them in your chosen directory.

(Note: The fp8 LLM model on the same page is unverified.)

## Usage

### Dataset Configuration

Please refer to [dataset configuration guide](./dataset/dataset_config.md).

### Latent Pre-caching

Latent pre-caching is required. Create the cache using the following command:

```bash
python cache_latents.py --dataset_config path/to/toml --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```

For additional options, use `python cache_latents.py --help`.

If you're running low on VRAM, reduce `--vae_spatial_tile_sample_min_size` to around 128 and lower the `--batch_size`.

Use `--debug_mode image` to display dataset images and captions in a new window, or `--debug_mode console` to display them in the console (requires `ascii-magic`).

### Text Encoder Output Pre-caching

Text Encoder output pre-caching is required. Create the cache using the following command:

```bash
python cache_text_encoder_outputs.py --dataset_config path/to/toml  --text_encoder1 path/to/ckpts/text_encoder --text_encoder2 path/to/ckpts/text_encoder_2 --batch_size 16
```

For additional options, use `python cache_text_encoder_outputs.py --help`.

Adjust `--batch_size` according to your available VRAM.

For systems with limited VRAM (less than ~16GB), use `--fp8_llm` to run the LLM in fp8 mode.

### Training

Start training using the following command (input as a single line):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 hv_train_network.py 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt 
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base 
    --optimizer_type adamw8bit --learning_rate 1e-3 --gradient_checkpointing 
    --max_data_loader_n_workers 2 --persistent_data_loader_workers 
    --network_module networks.lora --network_dim 32 
    --timestep_sampling sigmoid --discrete_flow_shift 1.0 
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42
    --output_dir path/to/output_dir --output_name name-of-lora
```

For additional options, use `python hv_train_network.py --help` (note that many options are unverified).

Specifying `--fp8_base` runs DiT in fp8 mode. Without this flag, mixed precision data type will be used. fp8 can significantly reduce memory consumption but may impact output quality. If `--fp8_base` is not specified, 24GB or more VRAM is recommended. Use `--blocks_to_swap` as needed.

If you're running low on VRAM, use `--blocks_to_swap` to offload some blocks to CPU. Maximum value is 36.

(The idea of block swap is based on the implementation by 2kpr. Thanks again to 2kpr.)

Use `--sdpa` for PyTorch's scaled dot product attention, or `--flash_attn` for FlashAttention (untested). `--sage_attn` uses SageAttention, but SageAttention is not yet supported for training and may not work correctly.

Sample video generation is not yet implemented.

The format of LoRA trained is the same as `sd-scripts`.

`--show_timesteps` can be set to `image` (requires `matplotlib`) or `console` to display timestep distribution and loss weighting during training.

Appropriate learning rates, training steps, timestep distribution, loss weighting, etc. are not yet known. Feedback is welcome.

### Inference

Generate videos using the following command:

```bash
python hv_generate_video.py --fp8 --video_size 544 960 --video_length 5 --infer_steps 30 
    --prompt "A cat walks on the grass, realistic style."  --save_path path/to/save/dir --output_type both 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt --attn_mode sdpa --split_attn
    --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt 
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 
    --text_encoder1 path/to/ckpts/text_encoder 
    --text_encoder2 path/to/ckpts/text_encoder_2 
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

For additional options, use `python hv_generate_video.py --help`.

Specifying `--fp8` runs DiT in fp8 mode. fp8 can significantly reduce memory consumption but may impact output quality.

If you're running low on VRAM, use `--blocks_to_swap` to offload some blocks to CPU. Maximum value is 38.

For `--attn_mode`, specify either `flash`, `torch`, `sageattn`, or `sdpa` (same as `torch`). These correspond to FlashAttention, scaled dot product attention, and SageAttention respectively. Default is `torch`. SageAttention is effective for VRAM reduction.

Specifing `--split_attn` will process attention in chunks. Inference with SageAttention is expected to be about 10% faster. Cannot be specified when `attn_mode` is `flash`.

For `--output_type`, specify either `both`, `latent`, `video` or `images`. `both` outputs both latents and video. Recommended to use `both` in case of Out of Memory errors during VAE processing. You can specify saved latents with `--latent_path` and use `--output_type video` (or `images`) to only perform VAE decoding.

`--seed` is optional. A random seed will be used if not specified.

`--video_length` should be specified as "a multiple of 4 plus 1".

### Convert LoRA to another format

You can convert LoRA to a format compatible with ComfyUI (presumed to be Diffusion-pipe) using the following command:

```bash
python convert_lora.py --input path/to/musubi_lora.safetensors --output path/to/another_format.safetensors --target other
```

Specify the input and output file paths with `--input` and `--output`, respectively.

Specify `other` for `--target`. Use `default` to convert from another format to the format of this repository.

## Miscellaneous

### SageAttention Installation

sdbsd has provided a Windows-compatible SageAttention implementation and pre-built wheels here:  https://github.com/sdbds/SageAttention-for-windows. After installing triton, if your Python, PyTorch, and CUDA versions match, you can download and install the pre-built wheel from the [Releases](https://github.com/sdbds/SageAttention-for-windows/releases) page. Thanks to sdbsd for this contribution.

For reference, the build and installation instructions are as follows. You may need to update Microsoft Visual C++ Redistributable to the latest version.

1. Download and install triton 3.1.0 wheel matching your Python version from [here](https://github.com/woct0rdho/triton-windows/releases/tag/v3.1.0-windows.post5).

2. Install Microsoft Visual Studio 2022 or Build Tools for Visual Studio 2022, configured for C++ builds.

3. Clone the SageAttention repository in your preferred directory:
    ```shell
    git clone https://github.com/thu-ml/SageAttention.git
    ```

    You can skip step 4 by using the sdbsd repository mentioned above by `git clone https://github.com/sdbds/SageAttention-for-windows.git`.

4. Open `math.cuh` in the `SageAttention/csrc` folder and change `ushort` to `unsigned short` on lines 71 and 146, then save.

5. Open `x64 Native Tools Command Prompt for VS 2022` from the Start menu under Visual Studio 2022.

6. Activate your venv, navigate to the SageAttention folder, and run the following command. If you get a DISTUTILS not configured error, set `set DISTUTILS_USE_SDK=1` and try again:
    ```shell
    python setup.py install
    ```

This completes the SageAttention installation.


## Disclaimer

This repository is unofficial and not affiliated with the official HunyuanVideo repository. 

This repository is experimental and under active development. While we welcome community usage and feedback, please note:

- This is not intended for production use
- Features and APIs may change without notice
- Some functionalities are still experimental and may not work as expected
- Video training features are still under development

If you encounter any issues or bugs, please create an Issue in this repository with:
- A detailed description of the problem
- Steps to reproduce
- Your environment details (OS, GPU, VRAM, Python version, etc.)
- Any relevant error messages or logs

## Contributing

We welcome contributions! However, please note:

- Due to limited maintainer resources, PR reviews and merges may take some time
- Before starting work on major changes, please open an Issue for discussion
- For PRs:
  - Keep changes focused and reasonably sized
  - Include clear descriptions
  - Follow the existing code style
  - Ensure documentation is updated

## License

Code under the `hunyuan_model` directory is modified from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and follows their license.

Other code is under the Apache License 2.0. Some code is copied and modified from Diffusers.
