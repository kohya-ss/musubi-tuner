# FLUX.2

## Overview / 概要

This document describes the usage of the [FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) \[dev\] architecture within the Musubi Tuner framework. FLUX.2-dev is an image generation model and edit model that can take a reference image as input.

This feature is experimental.

Latent pre-caching, training, and inference options can be found in the `--help` output. Many options are shared with HunyuanVideo, so refer to the [HunyuanVideo documentation](./hunyuan_video.md) as needed.

<details>
<summary>日本語</summary>

</details>

## Download the model / モデルのダウンロード

You need to download the DiT, AE, Text Encoder models.

### FLUX.2 [dev]

- **DiT, AE**: Download from the [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) repository. Use `flux2-dev.safetensors` and `ae.safetensors`. The weights in the subfolder are in Diffusers format and cannot be used.
- **Text Encoder (Mistral 3)**: Download all the split files from the [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) repository and specify the first file (e.g., `00001-of-00010.safetensors`) in the arguments.

<details>
<summary>日本語</summary>

DiT, AE, Text Encoder のモデルをダウンロードする必要があります。

- **DiT, AE**: [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) リポジトリからダウンロードしてください。`flux2-dev.safetensors` および `ae.safetensors` を使用してください。サブフォルダ内の重みはDiffusers形式なので使用できません。
- **Text Encoder (Mistral 3)**: Download all the split files from the [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev) repository and specify the first file (e.g., `00001-of-00010.safetensors`) in the arguments.
</details>

### FLUX.2 [klein] 4B / base 4B

- **DiT 4B**: Download from the [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) repository. Use `flux2-klein-4b.safetensors`.
- **DiT base 4B**: Download from the [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) repository. Use `flux2-klein-base-4b.safetensors`.
- **AE**: Download from the [black-forest-labs/FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) repository. Use `ae.safetensors`. `vae/diffusion_pytorch_model.safetensors` in the subfolder is in Diffusers format and cannot be used.
- **Qwen3 4B Text Encoder**: Download all the split files from the [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) repository and specify the first file (e.g., `00001-of-00002.safetensors`) in the arguments.

If you already have the weights for Qwen3 4B used in Z-Image, you can use them as is. Refer to the [Z-Image documentation](./zimage.md#download-the-model--モデルのダウンロード) for details.

<details>
<summary>日本語</summary>

- **DiT 4B**: [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) リポジトリからダウンロードしてください。`flux2-klein-4b.safetensors` を使用してください。
- **DiT base 4B**: [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) リポジトリからダウンロードしてください。`flux2-klein-base-4b.safetensors` を使用してください。
- **AE**: [black-forest-labs/FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) リポジトリからダウンロードしてください。`ae.safetensors` を使用してください。サブフォルダ内の `vae/diffusion_pytorch_model.safetensors` はDiffusers形式なので使用できません。
- **Qwen3 4B Text Encoder**: [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) リポジトリから分割されたすべてのファイルをダウンロードし、最初のファイル（例：`00001-of-00002.safetensors`）を引数で指定してください。

Qwen3 4Bの重みは、すでにZ-Imageで用いているものがあればそのまま使用可能です。[Z-Imageのドキュメント](./zimage.md#download-the-model--モデルのダウンロード)を参照してください。

</details>

### FLUX.2 [klein] 9B / base 9B

- **DiT 9B**: Download from the [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) repository. Use `flux2-klein-9b.safetensors`.
- **DiT base 9B**: Download from the [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) repository. Use `flux2-klein-base-9b.safetensors`.
- **AE**: Download from the [black-forest-labs/FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) repository. Use `ae.safetensors`. `vae/diffusion_pytorch_model.safetensors` in the subfolder is in Diffusers format and cannot be used.
- **Qwen3 8B Text Encoder**: Download all the split files from the [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) repository and specify the first file (e.g., `00001-of-00004.safetensors`) in the arguments.

<details>
<summary>日本語</summary>

- **DiT 9B**: [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) リポジトリからダウンロードしてください。`flux2-klein-9b.safetensors` を使用してください。
- **DiT base 9B**: [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) リポジトリからダウンロードしてください。`flux2-klein-base-9b.safetensors` を使用してください。
- **AE**: [black-forest-labs/FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev) リポジトリからダウンロードしてください。`ae.safetensors` を使用してください。サブフォルダ内の `vae/diffusion_pytorch_model.safetensors` はDiffusers形式なので使用できません。
- **Qwen3 8B Text Encoder**: [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) リポジトリから分割されたすべてのファイルをダウンロードし、最初のファイル（例：`00001-of-00004.safetensors`）を引数で指定してください。
</details>


## Specifying Model Version / モデルバージョンの指定

When specifying the model version in various scripts, use the following options:

|type|version|sampling guidance scale|num sampling steps|
|----|--------|----|----|
|flux.2-dev|`--model_version dev`|4.0|50|
|flux.2-klein-4b|`--model_version klein-4b`|1.0|4|
|flux.2-klein-base-4b|`--model_version klein-base-4b`|4.0|50|
|flux.2-klein-9b|`--model_version klein-9b`|1.0|4|
|flux.2-klein-base-9b|`--model_version klein-base-9b`|4.0|50|

For model training, it is recommended to use klein base 4B or 9B. The dev and klein 4B/9B are distilled models primarily intended for inference.

<details>
<summary>日本語</summary>

それぞれのスクリプトでモデルバージョンを指定する際には、英語版の文章を参考にして`--model_version`オプションを使用してください。

モデルの学習を行う場合は、klein base 4Bまたは9Bを使用することをお勧めします。dev、およびklein 4B/9Bは蒸留モデルであり、主に推論用です。

</details>

## Pre-caching / 事前キャッシング

### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching uses a dedicated script for FLUX.2.

```bash
python src/musubi_tuner/flux_2_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/ae_model \
    --model_version dev
```

- Note that the `--vae` argument is required, not `--ae`.
- Uses `flux_2_cache_latents.py`.
- The dataset must be an image dataset.
- Use the `--model_version` option for Flux.2 Klein training (if omitted, defaults to `dev`).
- The `control_images` in the dataset config is used as the reference image. See [Dataset Config](./dataset_config.md#flux1-kontext-dev) for details.
- `--vae_dtype` option is available to specify the VAE weight data type. Default is `float32`, `bfloat16` can also be specified. Specifying `bfloat16` reduces VRAM usage.

<details>
<summary>日本語</summary>

latentの事前キャッシングはFLUX.2専用のスクリプトを使用します。

- `flux_2_cache_latents.py`を使用します。
- `--ae`ではなく、`--vae`引数を指定してください。
- データセットは画像データセットである必要があります。
- データセット設定の`control_images`が参照画像として使用されます。詳細は[データセット設定](./dataset_config.md#flux1-kontext-dev)を参照してください。
- `--vae_dtype`オプションは、VAEの重みデータ型を指定するためのオプションです。デフォルトは`float32`で、`bfloat16`も指定可能です。`bfloat16`を指定するとVRAM使用量が削減されます。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text encoder output pre-caching also uses a dedicated script.

```bash
python src/musubi_tuner/flux_2_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder path/to/text_encoder \
    --batch_size 16 \
    --model_version dev
```

- Uses `flux_2_cache_text_encoder_outputs.py`.
- Requires `--text_encoder` argument
- Use the `--model_version` option for Flux.2 Klein training (if omitted, defaults to `dev`).
- Use `--fp8_text_encoder` option to run the Text Encoder in fp8 mode for VRAM savings.
- The larger the batch size, the more VRAM is required. Adjust `--batch_size` according to your VRAM capacity.

<details>
<summary>日本語</summary>

テキストエンコーダー出力の事前キャッシングも専用のスクリプトを使用します。

- `flux_2_cache_text_encoder_outputs.py`を使用します。
- テキストエンコーダーをfp8モードで実行するための`--fp8_text_encoder`オプションを使用します。
- バッチサイズが大きいほど、より多くのVRAMが必要です。VRAM容量に応じて`--batch_size`を調整してください。

</details>

## Training / 学習

Training uses a dedicated script `flux_2_train_network.py`.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/flux_2_train_network.py \
    --model_version dev \
    --dit path/to/dit_model \
    --vae path/to/ae_model \
    --text_encoder path/to/text_encoder \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling flux2_shift --weighting_scheme none \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_flux_2 --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```

- Uses `flux_2_train_network.py`.
- **Requires** specifying `--vae` (not `--ae`), `--text_encoder`
- **Requires** specifying `--network_module networks.lora_flux_2`.
- `--mixed_precision bf16` is recommended for FLUX.2 training.
- `--timestep_sampling flux2_shift` is recommended for FLUX.2.
- Use the `--model_version` option for Flux.2 Klein training (if omitted, defaults to `dev`).
- Memory saving options like `--fp8_base --fp8_scaled` (for DiT, specify both) and `--fp8_text_encoder` (for Text Encoder) are available. `--fp8_scaled` is recommended when using `--fp8_base` for DiT.
-  `--gradient_checkpointing` and `--gradient_checkpointing_cpu_offload` are available for memory savings. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.
- `--vae_dtype` option is available to specify the VAE weight data type. Default is `float32`, `bfloat16` can also be specified.
- Instead of `--sdpa`, `--xformers` and `--flash_attn` can also be used. Make sure the related libraries are installed.

`--fp8_text_encoder` option is not available for dev (Mistral 3).

Some blocks can be offloaded to CPU for memory savings using the `--blocks_to_swap` option. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.

In FLUX.2, since DoubleStreamBlock uses more memory than SingleStreamBlock and the number of each block varies by model, the actual number of offloaded blocks is automatically adjusted (double block + single block * 2 = number of swap blocks).

The maximum values of `blocks_to_swap` per model when combined with the `--fp8_base --fp8_scaled` options are as follows:

|Model Type|Maximum blocks_to_swap|
|----|----|
|flux.2-dev|29|
|flux.2-klein-4b|13|
|flux.2-klein-9b|16|

<details>
<summary>日本語</summary>

FLUX.2の学習は専用のスクリプト`flux_2_train_network.py`を使用します。

- `flux_2_train_network.py`を使用します。
- `--ae`、`--text_encoder` を指定する必要があります。
- `--network_module networks.lora_flux_2`を指定する必要があります。
- FLUX.2の学習には`--mixed_precision bf16`を推奨します。
- FLUX.2には`--timestep_sampling flux2_shift`を推奨します。
- `--fp8_base --fp8_scaled`（DiT用、両方指定してください）や`--fp8_text_encoder`（テキストエンコーダー用）などのメモリ節約オプションが利用可能です。`--fp8_base`をDiTに使用する場合は、`--fp8_scaled`を推奨します。
- メモリ節約のために`--gradient_checkpointing`が利用可能です。
- `--vae_dtype`オプションは、VAEの重みデータ型を指定するためのオプションです。デフォルトは`float32`で、`bfloat16`も指定可能です。
- `--sdpa`の代わりに`--xformers`および`--flash_attn`を使用することも可能です。関連するライブラリがインストールされていることを確認してください。

`--fp8_text_encoder`オプションはdev（Mistral 3）では使用できません。

一部のブロックをメモリ節約のためにCPUにオフロードする`--blocks_to_swap`オプションも利用可能です。詳細は[HunyuanVideoのドキュメント](./hunyuan_video.md#memory-optimization)を参照してください。

FLUX.2ではDoubleStreamBlockのメモリ使用量がSingleStreamBlockよりも大きいのと、それぞれのブロック数がモデルごとに異なるため、実際にオフロードされるブロック数は自動調整されます（double block + single block * 2 = swap block数）。

`--fp8_base --fp8_scaled`オプションと組み合わせたときの、モデルごとの`blocks_to_swap`の最大値は以下の通りです。

|モデル種類|blocks_to_swapの最大値|
|----|----|
|flux.2-dev|29|
|flux.2-klein-4b|13|
|flux.2-klein-9b|16|

</details>

## Inference / 推論

Inference uses a dedicated script `flux_2_generate_image.py`.

```bash
python src/musubi_tuner/flux_2_generate_image.py \
    --model_version dev \
    --dit path/to/dit_model \
    --vae path/to/ae_model \
    --text_encoder path/to/text_encoder \
    --control_image_path path/to/control_image.jpg \
    --prompt "A cat" \
    --image_size 1024 1024 --infer_steps 50 \
    --fp8_scaled \
    --save_path path/to/save/dir --output_type images \
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

- Uses `flux_2_generate_image.py`.
- **Requires** specifying `--vae`, `--text_encoder`
- **Requires** specifying `--control_image_path` for the reference image.
- Use the `--model_version` option for Flux.2 Klein inference (if omitted, defaults to `dev`).
- `--no_resize_control`: By default, the control image is resized to the recommended resolution for FLUX.2. If you specify this option, this resizing is skipped, and the image is used as-is.
    
    This feature is not officially supported by FLUX.2, but it is available for experimental use.

- `--image_size` is the size of the generated image, height and width are specified in that order.
- `--prompt`: Prompt for generation.
- `--fp8_scaled` option is available for DiT to reduce memory usage. Quality may be slightly lower. `--fp8_text_encoder` option is available to reduce memory usage of Text Encoder. `--fp8` alone is also an option for DiT but `--fp8_scaled` potentially offers better quality.
- LoRA loading options (`--lora_weight`, `--lora_multiplier`, `--include_patterns`, `--exclude_patterns`) are available. `--lycoris` is also supported.
- `--embedded_cfg_scale` (default 2.5) controls the distilled guidance scale.
- `--save_merged_model` option is available to save the DiT model after merging LoRA weights. Inference is skipped if this is specified.

<details>
<summary>日本語</summary>

FLUX.2の推論は専用のスクリプト`flux_2_generate_image.py`を使用します。

- `flux_2_generate_image.py`を使用します。
- `--vae`、`--text_encoder` を指定する必要があります。
- `--control_image_path`を指定する必要があります（参照画像）。
- `--no_resize_control`: デフォルトでは、参照画像はFLUX.2の推奨解像度にリサイズされます。このオプションを指定すると、このリサイズはスキップされ、画像はそのままのサイズで使用されます。

    この機能はFLUX.2では公式にサポートされていませんが、実験的に使用可能です。

- `--image_size`は生成する画像のサイズで、高さと幅をその順番で指定します。
- `--prompt`: 生成用のプロンプトです。
- DiTのメモリ使用量を削減するために、`--fp8_scaled`オプションを指定可能です。品質はやや低下する可能性があります。またText Encoder 1のメモリ使用量を削減するために、`--fp8_text_encoder`オプションを指定可能です。DiT用に`--fp8`単独のオプションも用意されていますが、`--fp8_scaled`の方が品質が良い可能性があります。
- LoRAの読み込みオプション（`--lora_weight`、`--lora_multiplier`、`--include_patterns`、`--exclude_patterns`）が利用可能です。LyCORISもサポートされています。
- `--embedded_cfg_scale`（デフォルト2.5）は、蒸留されたガイダンススケールを制御します。
- `--save_merged_model`オプションは、LoRAの重みをマージした後にDiTモデルを保存するためのオプションです。これを指定すると推論はスキップされます。

</details>