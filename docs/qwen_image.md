# Qwen-Image

## Overview / 概要

This document describes the usage of the Qwen-Image architecture within the Musubi Tuner framework. Qwen-Image is a text-to-image generation model.

This feature is experimental.

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内でのQwen-Imageアーキテクチャの使用法について説明しています。Qwen-Imageはテキストから画像を生成するモデルです。

この機能は実験的なものです。
</details>

## Download the model / モデルのダウンロード

You need to download the DiT, VAE, and Text Encoder (Qwen2.5-VL) models.

- **DiT, Text Encoder (Qwen2.5-VL)**: For DiT and Text Encoder, download `split_files/diffusion_models/qwen_image_bf16.safetensors` and `split_files/text_encoders/qwen_2.5_vl_7b.safetensors` from https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI, respectively. **The fp8_scaled version cannot be used.**

- **VAE**: For VAE, download `vae/diffusion_pytorch_model.safetensors` from https://huggingface.co/Qwen/Qwen-Image. **ComfyUI's VAE weights cannot be used.**

<details>
<summary>日本語</summary>

DiT, VAE, Text Encoder (Qwen2.5-VL) のモデルをダウンロードする必要があります。

- **DiT, Text Encoder (Qwen2.5-VL)**: DiTおよびText Encoderは、`split_files/diffusion_models/qwen_image_bf16.safetensors` と `split_files/text_encoders/qwen_2.5_vl_7b.safetensors` を https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI からそれぞれダウンロードしてください。**fp8_scaledバージョンは使用できません。**

- **VAE**: VAEのために、`vae/diffusion_pytorch_model.safetensors` を https://huggingface.co/Qwen/Qwen-Image からダウンロードしてください。**ComfyUIのVAEウェイトは使用できません。**

</details>

## Pre-caching / 事前キャッシング

### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching uses a dedicated script for Qwen-Image.

```bash
python src/musubi_tuner/qwen_image_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/vae_model
```

- Uses `qwen_image_cache_latents.py`.
- The `--vae` argument is required.

<details>
<summary>日本語</summary>

latentの事前キャッシングはQwen-Image専用のスクリプトを使用します。

- `qwen_image_cache_latents.py`を使用します。
- `--vae`引数を指定してください。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text encoder output pre-caching also uses a dedicated script.

```bash
python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder path/to/text_encoder \
    --batch_size 16
```

- Uses `qwen_image_cache_text_encoder_outputs.py`.
- Requires the `--text_encoder` (Qwen2.5-VL) argument.
- Use the `--fp8_vl` option to run the Text Encoder in fp8 mode for VRAM savings for <16GB GPUs.

<details>
<summary>日本語</summary>

テキストエンコーダー出力の事前キャッシングも専用のスクリプトを使用します。

- `qwen_image_cache_text_encoder_outputs.py`を使用します。
- `--text_encoder` (Qwen2.5-VL) 引数が必要です。
- VRAMを節約するために、fp8 でテキストエンコーダを実行する`--fp8_vl`オプションが使用可能です。VRAMが16GB未満のGPU向けです。

</details>

## Training / 学習

Training uses a dedicated script `qwen_image_train_network.py`.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/qwen_image_train_network.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift \
    --weighting_scheme none --discrete_flow_shift 3.0 \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```

- Uses `qwen_image_train_network.py`.
- **Requires** specifying `--dit`, `--vae`, and `--text_encoder`.
- The LoRA network for Qwen-Image (`networks.lora_qwen_image`) is automatically selected.
- `--mixed_precision bf16` is recommended for Qwen-Image training.
- Memory saving options like `--fp8_base` and `--fp8_scaled` (for DiT), and `--fp8_vl` (for Text Encoder) are available. 
- `--gradient_checkpointing` is available for memory savings.

`--fp8_vl` is recommended for GPUs with less than 16GB of VRAM.

`--sdpa` uses PyTorch's scaled dot product attention. Other options like `--xformers` and `--flash_attn` are available. `flash3` cannot be used currently.

If you specify `--split_attn`, the attention computation will be split, slightly reducing memory usage. Please specify `--split_attn` if you are using anything other than `--sdpa`.

`--timestep_sampling` allows you to choose the sampling method for the timesteps. `shift` with `--discrete_flow_shift` is the default. `qwen_shift` is also available. `qwen_shift` is a same method during inference. It uses the dynamic shift value based on the resolution of each image (typically around 2.2 for 1328x1328 images).

`--discrete_flow_shift` is set quite low for Qwen-Image during inference (as described), so a lower value than other models may be preferable.

The appropriate settings for each parameter are unknown. Feedback is welcome.

### VRAM Usage Estimates with Memory Saving Options

For 1024x1024 training with the batch size of 1, `--mixed_precision bf16` and `--gradient_checkpointing` is enabled and `--xformers` is used.

|options|VRAM Usage|
|-------|----------|
|no   |42GB|
|`--fp8_base --fp8_scaled`|30GB|
|+ `--blocks_to_swap 16`|24GB|
|+ `--blocks_to_swap 45`|12GB|

64GB main RAM system is recommended with `--blocks_to_swap`.

If `--blocks_to_swap` is more than 45, the main RAM usage will increase significantly.

<details>
<summary>日本語</summary>

Qwen-Imageの学習は専用のスクリプト`qwen_image_train_network.py`を使用します。

- `qwen_image_train_network.py`を使用します。
- `--dit`、`--vae`、`--text_encoder`を指定する必要があります。
- Qwen-Image用のLoRAネットワーク（`networks.lora_qwen_image`）は自動的に選択されます。
- Qwen-Imageの学習には`--mixed_precision bf16`を推奨します。
- `--fp8_base`や`--fp8_scaled`（DiT用）、`--fp8_vl`（テキストエンコーダー用）などのメモリ節約オプションが利用可能です。
- メモリ節約のために`--gradient_checkpointing`が利用可能です。

GPUのVRAMが16GB未満の場合は、`--fp8_vl`を推奨します。

`--sdpa`はPyTorchのscaled dot product attentionを用います。他に `--xformers`、`--flash_attn` があります。`--flash3`は現在使用できません。

`--split_attn` を指定すると、attentionの計算が分割され、メモリ使用量がわずかに削減されます。`--sdpa` 以外を使用する場合は、`--split_attn` を指定してください。

`--timestep_sampling` では、タイムステップのサンプリング方法を選択できます。`shift` と `--discrete_flow_shift` の組み合わせがデフォルトです。`qwen_shift` も利用可能です。`qwen_shift` は推論時と同じ方法で、各画像の解像度に基づいた動的シフト値を使用します（通常、1328x1328画像の場合は約2.2です）。

`--discrete_flow_shift`は、Qwen-Imageでは前述のように推論時にかなり低めなため、他のモデルよりも低めが良いかもしれません。

それぞれのパラメータの適切な設定は不明です。フィードバックをお待ちしています。

### メモリ節約オプションを使用した場合のVRAM使用量の目安

1024x1024の学習でバッチサイズ1の場合、`--mixed_precision bf16`と`--gradient_checkpointing`を指定し、`--xformers`を使用した場合のVRAM使用量の目安は以下の通りです。

|オプション|VRAM使用量|
|-------|----------|
|no   |42GB|
|`--fp8_base --fp8_scaled`|30GB|
|+ `--blocks_to_swap 16`|24GB|
|+ `--blocks_to_swap 45`|12GB|

`--blocks_to_swap`を使用する場合は、64GBのメインRAMを推奨します。

`--blocks_to_swap`が45を超えると、メインRAMの使用量が大幅に増加します。

</details>

## Inference / 推論

Inference uses a dedicated script `qwen_image_generate_image.py`.

```bash
python src/musubi_tuner/qwen_image_generate_image.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --prompt "A cat" \
    --negative_prompt " " \
    --image_size 1024 1024 --infer_steps 25 \
    --guidance_scale 4.0 \
    --attn_mode sdpa \
    --save_path path/to/save/dir \
    --output_type images \
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

- Uses `qwen_image_generate_image.py`.
- **Requires** specifying `--dit`, `--vae`, and `--text_encoder`.
- `--image_size` is the size of the generated image, height and width are specified in that order.
- `--prompt`: Prompt for generation.
- `--guidance_scale` controls the classifier-free guidance scale.
- Memory saving options like `--fp8_scaled` (for DiT) are available.
- `--text_encoder_cpu` enables CPU inference for the text encoder. Recommended for systems with limited GPU resources (less than 16GB VRAM).
- LoRA loading options (`--lora_weight`, `--lora_multiplier`) are available.

You can specify the discrete flow shift using `--flow_shift`. If omitted, the default value (dynamic shifting based on the image size) will be used.

`xformers`, `flash` and `sageattn` are also available as attention modes. However `sageattn` is not confirmed to work yet.

<details>
<summary>日本語</summary>

Qwen-Imageの推論は専用のスクリプト`qwen_image_generate_image.py`を使用します。

- `qwen_image_generate_image.py`を使用します。
- `--dit`、`--vae`、`--text_encoder`を指定する必要があります。
- `--image_size`は生成する画像のサイズで、高さと幅をその順番で指定します。
- `--prompt`: 生成用のプロンプトです。
- `--guidance_scale`は、classifier-freeガイダンスのスケールを制御します。
- DiTのメモリ使用量を削減するために、`--fp8_scaled`オプションを指定可能です。
- `--text_encoder_cpu`を指定するとテキストエンコーダーをCPUで推論します。GPUのVRAMが16GB未満のシステムでは、CPU推論を推奨します。
- LoRAの読み込みオプション（`--lora_weight`、`--lora_multiplier`）が利用可能です。

`--flow_shift`を指定することで、離散フローシフトを設定できます。省略すると、デフォルト値（画像サイズに基づく動的シフト）が使用されます。

`xformers`、`flash`、`sageattn`もattentionモードとして利用可能です。ただし、`sageattn`はまだ動作確認が取れていません。

</details>
