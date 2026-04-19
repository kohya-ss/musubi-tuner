# ERNIE-Image

## Overview / 概要

This document describes the usage of the [ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) architecture within the Musubi Tuner framework. ERNIE-Image is a text-to-image generation model from Baidu.

Pre-caching, training, and inference options can be found via `--help`. Many options are shared with HunyuanVideo, so refer to the [HunyuanVideo documentation](./hunyuan_video.md) as needed.

This feature is experimental.

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内での[ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image)アーキテクチャの使用法について説明しています。ERNIE-ImageはBaiduによるテキストから画像を生成するモデルです。

事前キャッシング、学習、推論のオプションは`--help`で確認してください。HunyuanVideoと共通のオプションが多くありますので、必要に応じて[HunyuanVideoのドキュメント](./hunyuan_video.md)も参照してください。

この機能は実験的なものです。

</details>

## Download the model / モデルのダウンロード

You need to download the DiT, VAE (AE), and Text Encoder (Mistral 3) models. You can use either the official weights or the ComfyUI repackaged weights.

- **Official Repository**: [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image)
    - **DiT**: `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` and `00002-of-00002.safetensors`. Download all the split files and specify the first file in the arguments.
    - **Text Encoder (Mistral 3)**: `text_encoder/model.safetensors`.
    - **VAE (AE)**: `vae/diffusion_pytorch_model.safetensors`.
- **ComfyUI Repackaged**: [Comfy-Org/ERNIE-Image](https://huggingface.co/Comfy-Org/ERNIE-Image)
    - **DiT**: `diffusion_models/ernie-image.safetensors`.
    - **Text Encoder (Mistral 3)**: `text_encoders/ministral-3-3b.safetensors`.
    - **VAE (AE)**: `vae/flux2-vae.safetensors`.

The VAE (AE) is compatible with FLUX.2. If you already have the FLUX.2 AE weights (`ae.safetensors`), you can reuse them as is without downloading again. See the [FLUX.2 documentation](./flux_2.md) for details.

<details>
<summary>日本語</summary>

DiT、VAE (AE)、Text Encoder (Mistral 3) のモデルをダウンロードする必要があります。公式の重み、またはComfyUI用に再パッケージされた重みのいずれかを使用できます。

- **公式リポジトリ**: [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image)
    - **DiT**: `transformer/diffusion_pytorch_model-00001-of-00002.safetensors` および `00002-of-00002.safetensors`。分割されたすべてのファイルをダウンロードし、引数には最初のファイルを指定してください。
    - **Text Encoder (Mistral 3)**: `text_encoder/model.safetensors`。
    - **VAE (AE)**: `vae/diffusion_pytorch_model.safetensors`。
- **ComfyUI用重み**: [Comfy-Org/ERNIE-Image](https://huggingface.co/Comfy-Org/ERNIE-Image)
    - **DiT**: `diffusion_models/ernie-image.safetensors`。
    - **Text Encoder (Mistral 3)**: `text_encoders/ministral-3-3b.safetensors`。
    - **VAE (AE)**: `vae/flux2-vae.safetensors`。

VAE (AE) はFLUX.2のものと互換性があります。すでにFLUX.2のAEの重み（`ae.safetensors`）をダウンロード済みの場合は、再ダウンロードの必要はなくそのまま使用できます。詳細は[FLUX.2のドキュメント](./flux_2.md)を参照してください。

</details>

## Pre-caching / 事前キャッシング

### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching uses a dedicated script for ERNIE-Image.

```bash
python src/musubi_tuner/ernie_image_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/vae_model
```

- Uses `ernie_image_cache_latents.py`.
- The dataset should be an image dataset.
- ERNIE-Image does not support control images, so only target image latents are cached.

<details>
<summary>日本語</summary>

latentの事前キャッシングはERNIE-Image専用のスクリプトを使用します。

- `ernie_image_cache_latents.py`を使用します。
- データセットは画像データセットである必要があります。
- ERNIE-Imageはコントロール画像をサポートしていないため、ターゲット画像のlatentのみがキャッシュされます。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text encoder output pre-caching also uses a dedicated script.

```bash
python src/musubi_tuner/ernie_image_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder path/to/text_encoder \
    --batch_size 16
```

- Uses `ernie_image_cache_text_encoder_outputs.py`.
- Requires `--text_encoder` (Mistral 3).
- Use `--fp8_llm` option to run the Text Encoder in fp8 mode for VRAM savings.
- Larger batch sizes require more VRAM. Adjust `--batch_size` according to your VRAM capacity.

<details>
<summary>日本語</summary>

テキストエンコーダー出力の事前キャッシングも専用のスクリプトを使用します。

- `ernie_image_cache_text_encoder_outputs.py`を使用します。
- `--text_encoder`（Mistral 3）が必要です。
- テキストエンコーダーをfp8モードで実行するための`--fp8_llm`オプションを使用することでVRAMを節約できます。
- バッチサイズが大きいほど、より多くのVRAMが必要です。VRAM容量に応じて`--batch_size`を調整してください。

</details>

## Training / 学習

Training uses a dedicated script `ernie_image_train_network.py`.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/ernie_image_train_network.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 4.0 \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_ernie_image --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```

- Uses `ernie_image_train_network.py`.
- **Requires** specifying `--vae` and `--text_encoder`.
- **Requires** specifying `--network_module networks.lora_ernie_image`.
- The optimal timestep sampling settings for ERNIE-Image are still unclear because no official training framework has been released. The official scheduler config (`scheduler/scheduler_config.json`) specifies `shift: 4.0` for inference, so starting training from `--timestep_sampling shift --weighting_scheme none --discrete_flow_shift 4.0` is recommended; adjust as needed.
- Memory saving options like `--fp8_base` and `--fp8_scaled` (for DiT) and `--fp8_llm` (for Text Encoder) are available.
- `--gradient_checkpointing` and `--gradient_checkpointing_cpu_offload` are available for memory savings. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.

<details>
<summary>日本語</summary>

ERNIE-Imageの学習は専用のスクリプト`ernie_image_train_network.py`を使用します。

コマンド例は英語版を参照してください。

- `ernie_image_train_network.py`を使用します。
- `--vae`、`--text_encoder`を指定する必要があります。
- `--network_module networks.lora_ernie_image`を指定する必要があります。
- 公式の学習フレームワークが未リリースのため、ERNIE-Imageのタイムステップサンプリング設定は不明です。公式のscheduler設定（`scheduler/scheduler_config.json`）では推論時に`shift: 4.0`が指定されているため、`--timestep_sampling shift --weighting_scheme none --discrete_flow_shift 4.0`をベースに調整することを推奨します。
- `--fp8_base`、`--fp8_scaled`（DiT用）や`--fp8_llm`（テキストエンコーダー用）などのメモリ節約オプションが利用可能です。
- メモリ節約のために`--gradient_checkpointing`および`--gradient_checkpointing_cpu_offload`が利用可能です。詳細は[HunyuanVideoドキュメント](./hunyuan_video.md#memory-optimization)を参照してください。

</details>

### Memory Optimization

- `--fp8_base` and `--fp8_scaled` options are available to reduce memory usage of DiT (specify both together). Quality may degrade slightly.
- `--fp8_llm` option is available to reduce memory usage of Text Encoder (Mistral 3).
- `--gradient_checkpointing` and `--gradient_checkpointing_cpu_offload` are available for memory savings. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.
- `--blocks_to_swap` option is available to offload some blocks to CPU.

<details>
<summary>日本語</summary>

- DiTのメモリ使用量を削減するために、`--fp8_base`と`--fp8_scaled`オプションを指定可能です（同時に指定してください）。品質はやや低下する可能性があります。
- Text Encoder (Mistral 3)のメモリ使用量を削減するために、`--fp8_llm`オプションを指定可能です。
- メモリ節約のために`--gradient_checkpointing`と`--gradient_checkpointing_cpu_offload`が利用可能です。詳細は[HunyuanVideoドキュメント](./hunyuan_video.md#memory-optimization)を参照してください。
- `--blocks_to_swap`オプションで、一部のブロックをCPUにオフロードできます。

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

Inference uses a dedicated script `ernie_image_generate_image.py`.

```bash
python src/musubi_tuner/ernie_image_generate_image.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --prompt "A cat" \
    --negative_prompt "bad quality" \
    --image_size 1024 1024 --infer_steps 50 \
    --guidance_scale 4.0 \
    --attn_mode torch \
    --save_path path/to/save/dir \
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

- Uses `ernie_image_generate_image.py`.
- `--guidance_scale` defaults to 4.0 (classifier-free guidance). Specify a value of 1.0 or less to disable CFG.
- `--infer_steps` defaults to 50.
- `--flow_shift` defaults to 4.0 (matches the official `scheduler_config.json`). Set to 1.0 to disable shift.
- `--fp8` and `--fp8_scaled` options are available for DiT.
- `--fp8_text_encoder` option is available for Text Encoder.
- `--blocks_to_swap` option is available to offload some blocks to CPU.
- LoRA loading options (`--lora_weight`, `--lora_multiplier`, `--include_patterns`, `--exclude_patterns`) are available. LyCORIS is also supported.
- `--save_merged_model` option saves the DiT model after LoRA weights are merged. When specified, inference is skipped.

<details>
<summary>日本語</summary>

推論は専用のスクリプト`ernie_image_generate_image.py`を使用します。

コマンド例は英語版を参照してください。

- `ernie_image_generate_image.py`を使用します。
- `--guidance_scale`のデフォルトは4.0（Classifier-Free Guidanceあり）です。1.0以下の値を指定するとCFGを無効化できます。
- `--infer_steps`のデフォルトは50です。
- `--flow_shift`のデフォルトは4.0です（公式の`scheduler_config.json`に合わせています）。1.0を指定するとshiftを無効化できます。
- `--fp8`および`--fp8_scaled`オプションがDiTで利用可能です。
- `--fp8_text_encoder`オプションがテキストエンコーダーで利用可能です。
- `--blocks_to_swap`オプションで、一部のブロックをCPUにオフロードできます。
- LoRAの読み込みオプション（`--lora_weight`、`--lora_multiplier`、`--include_patterns`、`--exclude_patterns`）が利用可能です。LyCORISもサポートされています。
- `--save_merged_model`オプションは、LoRAの重みをマージした後にDiTモデルを保存するためのオプションです。これを指定すると推論はスキップされます。

</details>
