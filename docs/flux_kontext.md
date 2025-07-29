# FLUX.1 Kontext

## Overview / 概要

This document describes the usage of the [FLUX.1 Kontext](https://github.com/black-forest-labs/flux) \[dev\] architecture within the Musubi Tuner framework. FLUX.1 Kontext is an image generation model that can take a reference image as input.

This feature is experimental.

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内での[FLUX.1 Kontext](https://github.com/black-forest-labs/flux) \[dev\] アーキテクチャの使用法について説明しています。FLUX.1 Kontextは、参照画像をコンテキストとして入力できる画像生成モデルです。

この機能は実験的なものです。
</details>

## Download the model / モデルのダウンロード

You need to download the DiT, AE, Text Encoder 1 (T5-XXL), and Text Encoder 2 (CLIP-L) models.

- **DiT, AE**: Download from the [black-forest-labs/FLUX.1-kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) repository. Use `flux1-kontext-dev.safetensors` and `ae.safetensors`. The weights in the subfolder are in Diffusers format and cannot be used.
- **Text Encoder 1 (T5-XXL), Text Encoder 2 (CLIP-L)**: Download from the [ComfyUI FLUX Text Encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) repository. Please use `t5xxl_fp16.safetensors` for T5-XXL. Thanks to ComfyUI for providing these models.

<details>
<summary>日本語</summary>

DiT, AE, Text Encoder 1 (T5-XXL), Text Encoder 2 (CLIP-L) のモデルをダウンロードする必要があります。

- **DiT, AE**: [black-forest-labs/FLUX.1-kontext](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) リポジトリからダウンロードしてください。`flux1-kontext-dev.safetensors` および `ae.safetensors` を使用してください。サブフォルダ内の重みはDiffusers形式なので使用できません。
- **Text Encoder 1 (T5-XXL), Text Encoder 2 (CLIP-L)**: [ComfyUIのFLUX Text Encoders](https://huggingface.co/comfyanonymous/flux_text_encoders) リポジトリからダウンロードしてください。T5-XXLには`t5xxl_fp16.safetensors`を使用してください。これらのモデルをご提供いただいたComfyUIに感謝します。
</details>

## Pre-caching / 事前キャッシング

### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching uses a dedicated script for FLUX.1 Kontext.

```bash
python src/musubi_tuner/flux_kontext_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/ae_model
```

- Note that the `--vae` argument is required, not `--ae`.
- Uses `flux_kontext_cache_latents.py`.
- The dataset must be an image dataset.
- The `control_images` in the dataset config is used as the reference image. See [Dataset Config](../src/musubi_tuner/dataset/dataset_config.md#flux1-kontext-dev) for details.

<details>
<summary>日本語</summary>

latentの事前キャッシングはFLUX.1 Kontext専用のスクリプトを使用します。

- `flux_kontext_cache_latents.py`を使用します。
- `--ae`ではなく、`--vae`引数を指定してください。
- データセットは画像データセットである必要があります。
- データセット設定の`control_images`が参照画像として使用されます。詳細は[データセット設定](../src/musubi_tuner/dataset/dataset_config.md#flux1-kontext-dev)を参照してください。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text encoder output pre-caching also uses a dedicated script.

```bash
python src/musubi_tuner/flux_kontext_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder1 path/to/text_encoder1 \
    --text_encoder2 path/to/text_encoder2 \
    --batch_size 16
```

- Uses `flux_kontext_cache_text_encoder_outputs.py`.
- Requires both `--text_encoder1` (T5) and `--text_encoder2` (CLIP) arguments.
- Use `--fp8_t5` option to run the T5 Text Encoder in fp8 mode for VRAM savings.
- The larger the batch size, the more VRAM is required. Adjust `--batch_size` according to your VRAM capacity.

<details>
<summary>日本語</summary>

テキストエンコーダー出力の事前キャッシングも専用のスクリプトを使用します。

- `flux_kontext_cache_text_encoder_outputs.py`を使用します。
- T5とCLIPの両方の引数が必要です。
- T5テキストエンコーダーをfp8モードで実行するための`--fp8_t5`オプションを使用します。
- バッチサイズが大きいほど、より多くのVRAMが必要です。VRAM容量に応じて`--batch_size`を調整してください。

</details>

## Training / 学習

Training uses a dedicated script `flux_kontext_train_network.py`.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/flux_kontext_train_network.py \
    --dit path/to/dit_model \
    --vae path/to/ae_model \
    --text_encoder1 path/to/text_encoder1 \
    --text_encoder2 path/to/text_encoder2 \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling flux_shift --weighting_scheme none \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_flux --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```

- Uses `flux_kontext_train_network.py`.
- **Requires** specifying `--vae` (not `--ae`), `--text_encoder1`, and `--text_encoder2`.
- **Requires** specifying `--network_module networks.lora_flux`.
- `--mixed_precision bf16` is recommended for FLUX.1 Kontext training.
- `--timestep_sampling flux_shift` is recommended for FLUX.1 Kontext.
- Memory saving options like `--fp8` (for DiT) and `--fp8_t5` (for Text Encoder 1) are available. `--fp8_scaled` is recommended when using `--fp8` for DiT.
- `--gradient_checkpointing` is available for memory savings.

<details>
<summary>日本語</summary>

FLUX.1 Kontextの学習は専用のスクリプト`flux_kontext_train_network.py`を使用します。

- `flux_kontext_train_network.py`を使用します。
- `--ae`、`--text_encoder1`、`--text_encoder2`を指定する必要があります。
- `--network_module networks.lora_flux`を指定する必要があります。
- FLUX.1 Kontextの学習には`--mixed_precision bf16`を推奨します。
- FLUX.1 Kontextには`--timestep_sampling flux_shift`を推奨します。
- `--fp8`（DiT用）や`--fp8_t5`（テキストエンコーダー1用）などのメモリ節約オプションが利用可能です。`--fp8_scaled`を使用することをお勧めします。
- メモリ節約のために`--gradient_checkpointing`が利用可能です。

</details>

## Inference / 推論

Inference uses a dedicated script `flux_kontext_generate_image.py`.

```bash
python src/musubi_tuner/flux_kontext_generate_image.py \
    --dit path/to/dit_model \
    --vae path/to/ae_model \
    --text_encoder1 path/to/text_encoder1 \
    --text_encoder2 path/to/text_encoder2 \
    --control_image_path path/to/control_image.jpg \
    --prompt "A cat" \
    --image_size 1024 1024 --infer_steps 25 \
    --attn_mode sdpa --fp8_scaled \
    --save_path path/to/save/dir --output_type images \
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

- Uses `flux_kontext_generate_image.py`.
- **Requires** specifying `--vae`, `--text_encoder1`, and `--text_encoder2`.
- **Requires** specifying `--control_image_path` for the reference image.
- `--no_resize_control`: By default, the control image is resized to the recommended resolution for FLUX.1 Kontext. If you specify this option, this resizing is skipped, and the image is used as-is.
    
    This feature is not officially supported by FLUX.1 Kontext, but it is available for experimental use.

- `--image_size` is the size of the generated image, height and width are specified in that order.
- `--prompt`: Prompt for generation.
- `--fp8_scaled` option is available for DiT to reduce memory usage. Quality may be slightly lower. `--fp8_t5` option is available to reduce memory usage of Text Encoder 1. `--fp8` alone is also an option for DiT but `--fp8_scaled` potentially offers better quality.
- LoRA loading options (`--lora_weight`, `--lora_multiplier`, `--include_patterns`, `--exclude_patterns`) are available. `--lycoris` is also supported.
- `--embedded_cfg_scale` (default 2.5) controls the distilled guidance scale.
- `--save_merged_model` option is available to save the DiT model after merging LoRA weights. Inference is skipped if this is specified.

<details>
<summary>日本語</summary>

FLUX.1 Kontextの推論は専用のスクリプト`flux_kontext_generate_image.py`を使用します。

- `flux_kontext_generate_image.py`を使用します。
- `--vae`、`--text_encoder1`、`--text_encoder2`を指定する必要があります。
- `--control_image_path`を指定する必要があります（参照画像）。
- `--no_resize_control`: デフォルトでは、参照画像はFLUX.1 Kontextの推奨解像度にリサイズされます。このオプションを指定すると、このリサイズはスキップされ、画像はそのままのサイズで使用されます。

    この機能はFLUX.1 Kontextでは公式にサポートされていませんが、実験的に使用可能です。

- `--image_size`は生成する画像のサイズで、高さと幅をその順番で指定します。
- `--prompt`: 生成用のプロンプトです。
- DiTのメモリ使用量を削減するために、`--fp8_scaled`オプションを指定可能です。品質はやや低下する可能性があります。またText Encoder 1のメモリ使用量を削減するために、`--fp8_t5`オプションを指定可能です。DiT用に`--fp8`単独のオプションも用意されていますが、`--fp8_scaled`の方が品質が良い可能性があります。
- LoRAの読み込みオプション（`--lora_weight`、`--lora_multiplier`、`--include_patterns`、`--exclude_patterns`）が利用可能です。LyCORISもサポートされています。
- `--embedded_cfg_scale`（デフォルト2.5）は、蒸留されたガイダンススケールを制御します。
- `--save_merged_model`オプションは、LoRAの重みをマージした後にDiTモデルを保存するためのオプションです。これを指定すると推論はスキップされます。

</details>