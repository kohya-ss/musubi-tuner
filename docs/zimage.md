# Z-Image

## Overview / 概要

This document describes the usage of Z-Image architecture within the Musubi Tuner framework. Z-Image is a model architecture that supports text-to-image generation.

Pre-caching, training, and inference options can be found via `--help`. Many options are shared with HunyuanVideo, so refer to the [HunyuanVideo documentation](./hunyuan_video.md) as needed.

This feature is experimental.

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内でのZ-Imageアーキテクチャの使用法について説明しています。Z-Imageはテキストから画像を生成することができるモデルアーキテクチャです。

事前キャッシング、学習、推論のオプションは`--help`で確認してください。HunyuanVideoと共通のオプションが多くありますので、必要に応じて[HunyuanVideoのドキュメント](./hunyuan_video.md)も参照してください。

この機能は実験的なものです。

</details>

## Download the model / モデルのダウンロード

You need to download the DiT, VAE, and Text Encoder (Qwen3) models. You can use either of the following:

- **Official Repository**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/)
    - For DiT and Text Encoder, download all the split files and specify the first file (e.g., `00001-of-00004.safetensors`) in the arguments.
    - You do not need to download files other than `*.safetensors`.
- **ComfyUI Weights**: [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo)

You need to prepare the following models:

- **DiT**: The transformer model.
- **VAE**: The autoencoder model.
- **Text Encoder**: Qwen3 model.

<details>
<summary>日本語</summary>

DiT, VAE, Text Encoder (Qwen3) のモデルをダウンロードする必要があります。以下のいずれかを使用してください：

- **公式リポジトリ**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/)
    - DiT、Text Encoderは、分割された複数のファイルをすべてダウンロードし、引数には `00001-of-00004.safetensors` のような最初のファイルを指定してください。
    - `*.safetensors` ファイル以外はダウンロードする必要はありません。
- **ComfyUI用重み**: [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo)

以下のモデルを準備してください：

- **DiT**: Transformerモデル。
- **VAE**: Autoencoderモデル。
- **Text Encoder**: Qwen3モデル。

</details>

## Pre-caching / 事前キャッシング

### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching uses a dedicated script for Z-Image.

```bash
python src/musubi_tuner/zimage_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/vae_model
```

- Uses `zimage_cache_latents.py`.
- The dataset should be an image dataset.
- Z-Image does not support control images, so only target image latents are cached.

<details>
<summary>日本語</summary>

latentの事前キャッシングはZ-Image専用のスクリプトを使用します。

- `zimage_cache_latents.py`を使用します。
- データセットは画像データセットである必要があります。
- Z-Imageはコントロール画像をサポートしていないため、ターゲット画像のlatentのみがキャッシュされます。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text encoder output pre-caching also uses a dedicated script.

```bash
python src/musubi_tuner/zimage_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder path/to/text_encoder \
    --batch_size 16
```

- Uses `zimage_cache_text_encoder_outputs.py`.
- Requires `--text_encoder` (Qwen3).
- Use `--fp8_llm` option to run the Text Encoder in fp8 mode for VRAM savings.
- Larger batch sizes require more VRAM. Adjust `--batch_size` according to your VRAM capacity.

<details>
<summary>日本語</summary>

テキストエンコーダー出力の事前キャッシングも専用のスクリプトを使用します。

- `zimage_cache_text_encoder_outputs.py`を使用します。
- `--text_encoder`（Qwen3）が必要です。
- テキストエンコーダーをfp8モードで実行するための`--fp8_llm`オプションを使用することでVRAMを節約できます。
- バッチサイズが大きいほど、より多くのVRAMが必要です。VRAM容量に応じて`--batch_size`を調整してください。

</details>

## Training / 学習

Training uses a dedicated script `zimage_train_network.py`.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/zimage_train_network.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.0 \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_zimage --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```

- Uses `zimage_train_network.py`.
- **Requires** specifying `--vae` and `--text_encoder`.
- **Requires** specifying `--network_module networks.lora_zimage`.
- It is not yet clear whether `--mixed_precision bf16` or `fp16` is better for Z-Image training.
- The timestep sampling settings for Z-Image training are unclear, but it may be good to base them on `--timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.0` and adjust as needed.
- Memory saving options like `--fp8_base` and `--fp8_scaled` (for DiT) and `--fp8_llm` (for Text Encoder) are available.
- `--gradient_checkpointing` is available for memory savings. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.

<details>
<summary>日本語</summary>

Z-Imageの学習は専用のスクリプト`zimage_train_network.py`を使用します。

コマンド例は英語版を参照してください。

- `zimage_train_network.py`を使用します。
- `--vae`、`--text_encoder`を指定する必要があります。
- `--network_module networks.lora_zimage`を指定する必要があります。
- Z-Imageの学習に`--mixed_precision bf16`と`fp16`のどちらが良いかはまだ不明です。
- Z-Imageのタイムステップサンプリング設定は不明ですが、`--timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.0`をベースに調整すると良いかもしれません。
- `--fp8_base`、`--fp8_scaled`（DiT用）や`--fp8_llm`（テキストエンコーダー用）などのメモリ節約オプションが利用可能です。
- メモリ節約のために`--gradient_checkpointing`が利用可能です。詳細は[HunyuanVideoドキュメント](./hunyuan_video.md#memory-optimization)を参照してください。

</details>

### Converting LoRA weights to ComfyUI format / LoRA重みをComfyUI形式に変換する

A script is provided to convert Z-Image LoRA weights to ComfyUI format.

```bash
python src/musubi_tuner/networks/convert_z_image_lora_to_comfy.py \
    path/to/zimage_lora.safetensors \
    path/to/output_comfy_lora.safetensors
```

- The script is `convert_z_image_lora_to_comfy.py`.
- The first argument is the input Z-Image LoRA weights file.
- The second argument is the output ComfyUI-format LoRA weights file.

<details>
<summary>日本語</summary>

Z-ImageのLoRA重みをComfyUI形式に変換するスクリプトが提供されています。

- スクリプトは`convert_z_image_lora_to_comfy.py`です。
- 最初の引数は入力のZ-Image LoRA重みファイルです。
- 2番目の引数は出力のComfyUI形式のLoRA重みファイルです。

</details>

### Memory Optimization

- `--fp8_base` and `--fp8_scaled` options are available to reduce memory usage of DiT (specify both together). Quality may degrade slightly.
- `--fp8_llm` option is available to reduce memory usage of Text Encoder (Qwen3).
- `--gradient_checkpointing` and `--gradient_checkpointing_cpu_offload` are available for memory savings. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.
- `--blocks_to_swap` option is available to offload some blocks to CPU. The maximum number of blocks that can be offloaded is 28.

<details>
<summary>日本語</summary>

- DiTのメモリ使用量を削減するために、`--fp8_base`と`--fp8_scaled`オプションを指定可能です（同時に指定してください）。品質はやや低下する可能性があります。
- Text Encoder (Qwen3)のメモリ使用量を削減するために、`--fp8_llm`オプションを指定可能です。
- メモリ節約のために`--gradient_checkpointing`と`--gradient_checkpointing_cpu_offload`が利用可能です。詳細は[HunyuanVideoドキュメント](./hunyuan_video.md#memory-optimization)を参照してください。
- `--blocks_to_swap`オプションで、一部のブロックをCPUにオフロードできます。オフロード可能な最大ブロック数は28です。

</details>

### Attention

- `--sdpa` for PyTorch's scaled dot product attention (does not require additional dependencies).
- `--flash_attn` for [FlashAttention](https://github.com/Dao-AILab/flash-attention).
- `--xformers` for xformers (requires `--split_attn`).
- `--sage_attn` for SageAttention (not yet supported for training).
- `--split_attn` processes attention in chunks, reducing VRAM usage slightly.

<details>
<summary>日本語</summary>

- `--sdpa`でPyTorchのscaled dot product attentionを使用（追加の依存ライブラリを必要としません）。
- `--flash_attn`で[FlashAttention](https://github.com/Dao-AILab/flash-attention)を使用。
- `--xformers`でxformersの利用も可能（`--split_attn`が必要）。
- `--sage_attn`でSageAttentionを使用（現時点では学習に未対応）。
- `--split_attn`を指定すると、attentionを分割して処理し、VRAM使用量をわずかに減らします。

</details>

## Inference / 推論

Inference uses a dedicated script `zimage_generate_image.py`.

```bash
python src/musubi_tuner/zimage_generate_image.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --prompt "A cat" \
    --image_size 1024 1024 --infer_steps 25 \
    --flow_shift 3.0 --guidance_scale 0.0 \
    --attn_mode torch \
    --save_path path/to/save/dir \
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

- Uses `zimage_generate_image.py`.
- `--flow_shift` defaults to 3.0.
- `--guidance_scale` defaults to 0.0 (no classifier-free guidance).
- `--fp8` and `--fp8_scaled` options are available for DiT.
- `--fp8_llm` option is available for Text Encoder.

<details>
<summary>日本語</summary>

推論は専用のスクリプト`zimage_generate_image.py`を使用します。

コマンド例は英語版を参照してください。

- `zimage_generate_image.py`を使用します。
- `--flow_shift`のデフォルトは3.0です。
- `--guidance_scale`のデフォルトは0.0（Classifier-Free Guidanceなし）です。
- `--fp8`および`--fp8_scaled`オプションがDiTで利用可能です。
- `--fp8_llm`オプションがテキストエンコーダーで利用可能です。
- `--blocks_to_swap`オプションで、一部のブロックをCPUにオフロードできます。オフロード可能な最大ブロック数は28です。
- LoRAの読み込みオプション（`--lora_weight`、`--lora_multiplier`、`--include_patterns`、`--exclude_patterns`）が利用可能です。LyCORISもサポートされています。
- `--save_merged_model`オプションは、LoRAの重みをマージした後にDiTモデルを保存するためのオプションです。これを指定すると推論はスキップされます。

</details>
