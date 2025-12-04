# HunyuanVideo 1.5

## Overview / 概要

This document describes the usage of HunyuanVideo 1.5 architecture within the Musubi Tuner framework. HunyuanVideo 1.5 is a video generation model that supports both text-to-video (T2V) and image-to-video (I2V) generation.

Pre-caching, training, and inference options can be found via `--help`. Many options are shared with HunyuanVideo, so refer to the [HunyuanVideo documentation](./hunyuan_video.md) as needed.

This feature is experimental.

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内でのHunyuanVideo 1.5アーキテクチャの使用法について説明しています。HunyuanVideo 1.5はテキストから動画を生成（T2V）、および画像から動画を生成（I2V）することができるモデルです。

事前キャッシング、学習、推論のオプションは`--help`で確認してください。HunyuanVideoと共通のオプションが多くありますので、必要に応じて[HunyuanVideoのドキュメント](./hunyuan_video.md)も参照してください。

この機能は実験的なものです。

</details>

## Download the model / モデルのダウンロード

You need to download the DiT, VAE, Text Encoder (Qwen2.5-VL), and BYT5 models.

- **DiT**: Download from [HuggingFace's HunyuanVideo 1.5 site](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main). Use `transformer/720p_i2v/diffusion_pytorch_model.safetensors` for I2V DiT and `transformer/720p_t2v/diffusion_pytorch_model.safetensors` for T2V DiT.
Alternatively, you can use `split_files/diffusion_models/hunyuanvideo1.5_720p_i2v_fp16.safetensors` and `split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors` from [ComfyUI's HunyuanVideo 1.5 weights](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main), but do not use these for bf16 training as the weights are converted to fp16.

- **Text Encoder (Qwen2.5-VL)**: Download from [ComfyUI's HunyuanVideo 1.5 weights](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main). Use `split_files/text_encoders/qwen_2.5_vl_7b.safetensors`.

- **BYT5**: Download from [ComfyUI's HunyuanVideo 1.5 weights](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main). Use `split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors`.

For I2V training or inference, you also need:

- **Image Encoder (SigLIP)**: Download from [ComfyUI's HunyuanVideo 1.5 weights](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main). Use `split_files/clip_vision/sigclip_vision_patch14_384.safetensors`.

<details>
<summary>日本語</summary>

DiT, VAE, Text Encoder (Qwen2.5-VL), BYT5 のモデルをダウンロードする必要があります。

- **DiT**: [HuggingFaceのHunyuanVideo 1.5のサイト](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main) からダウンロードしてください。
I2VのDiTには`transformer/720p_i2v/diffusion_pytorch_model.safetensors`を、T2VのDiTには`transformer/720p_t2v/diffusion_pytorch_model.safetensors`を使用してください。
[ComfyUIのHunyuanVideo 1.5用の重み](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main)の`split_files/diffusion_models/hunyuanvideo1.5_720p_i2v_fp16.safetensors`および`split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors`も使用可能ですが、重みがfp16に変換されているため、bf16学習の時には使用しないでください。

- **VAE**: [HuggingFaceのHunyuanVideo 1.5のサイト](https://huggingface.co/tencent/HunyuanVideo-1.5/tree/main) から `vae/diffusion_pytorch_model.safetensors` をダウンロードしてください。
または、[ComfyUIのHunyuanVideo 1.5用の重み](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main)の`split_files/vae/hunyuanvideo15_vae_fp16.safetensors`も使用可能です。

- **Text Encoder (Qwen2.5-VL)**: [ComfyUIのHunyuanVideo 1.5用の重み](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main)から`split_files/text_encoders/qwen_2.5_vl_7b.safetensors`をダウンロードしてください。

- **BYT5**: [ComfyUIのHunyuanVideo 1.5用の重み](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main)から`split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors`をダウンロードしてください。

I2V学習または推論を行う場合は、さらに以下が必要です：

- **Image Encoder (SigLIP)**: [ComfyUIのHunyuanVideo 1.5用の重み](https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/tree/main)から `split_files/clip_vision/sigclip_vision_patch14_384.safetensors` をダウンロードしてください。

</details>

## Pre-caching / 事前キャッシング

### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching uses a dedicated script for HunyuanVideo 1.5.

```bash
python src/musubi_tuner/hv_1_5_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/vae_model
```

- Uses `hv_1_5_cache_latents.py`.
- The dataset can be either an image dataset or a video dataset.
- `--vae_sample_size` option sets the VAE sample size for tiling. Default is 128. Set to 256 if VRAM is sufficient for better quality. Set to 0 to disable tiling (highest quality but consumes a lot of VRAM).
- `--vae_enable_patch_conv` option enables patch-based convolution in VAE for memory optimization (less effective than `--vae_sample_size`). No quality degradation.
- For I2V training, specify `--i2v` and `--image_encoder path/to/image_encoder` to cache image features and conditional latents.

<details>
<summary>日本語</summary>

latentの事前キャッシングはHunyuanVideo 1.5専用のスクリプトを使用します。

- `hv_1_5_cache_latents.py`を使用します。
- データセットは画像データセットまたは動画データセットのいずれかです。
- `--vae_sample_size`オプションでVAEのタイリング用サンプルサイズを設定します。デフォルトは128です。VRAMが十分な場合は256に設定すると品質が向上します。0に設定するとタイリングを無効にします（最良の品質ですが非常に多くのVRAMを消費します）。
- `--vae_enable_patch_conv`オプションでVAEのパッチベース畳み込みを有効にし、メモリを最適化します（メモリ削減効果は`--vae_sample_size`よりも落ちます）。品質の劣化はありません。
- I2V学習の場合は、`--i2v`と`--image_encoder path/to/image_encoder`を指定して、画像の特徴と条件付きlatentをキャッシュします。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text encoder output pre-caching also uses a dedicated script.

```bash
python src/musubi_tuner/hv_1_5_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder path/to/text_encoder \
    --byt5 path/to/byt5 \
    --batch_size 16
```

- Uses `hv_1_5_cache_text_encoder_outputs.py`.
- Requires both `--text_encoder` (Qwen2.5-VL) and `--byt5` arguments.
- Use `--fp8_vl` option to run the Qwen2.5-VL Text Encoder in fp8 mode for VRAM savings.
- The larger the batch size, the more VRAM is required. Adjust `--batch_size` according to your VRAM capacity.

<details>
<summary>日本語</summary>

テキストエンコーダー出力の事前キャッシングも専用のスクリプトを使用します。

- `hv_1_5_cache_text_encoder_outputs.py`を使用します。
- `--text_encoder`（Qwen2.5-VL）と`--byt5`の両方の引数が必要です。
- Qwen2.5-VLテキストエンコーダーをfp8モードで実行するための`--fp8_vl`オプションを使用します。
- バッチサイズが大きいほど、より多くのVRAMが必要です。VRAM容量に応じて`--batch_size`を調整してください。

</details>

## Training / 学習

Training uses a dedicated script `hv_1_5_train_network.py`.

### Text-to-Video (T2V) Training

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hv_1_5_train_network.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --byt5 path/to/byt5 \
    --dataset_config path/to/toml \
    --task t2v \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.0 
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_hv_1_5 --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```

### Image-to-Video (I2V) Training

For I2V training, specify `--task i2v` and provide the `--image_encoder` path:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hv_1_5_train_network.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --byt5 path/to/byt5 \
    --image_encoder path/to/image_encoder \
    --dataset_config path/to/toml \
    --task i2v \
    --sdpa --mixed_precision bf16 \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.0 \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_hv_1_5 --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```

- Uses `hv_1_5_train_network.py`.
- **Requires** specifying `--vae`, `--text_encoder`, and `--byt5`.
- **Requires** specifying `--network_module networks.lora_hv_1_5`.
- **Requires** specifying `--task` as either `t2v` or `i2v`.
- For I2V training, `--image_encoder` is required.
- It is not yet clear whether `--mixed_precision bf16` or `fp16` is better for HunyuanVideo 1.5 training.
- The timestep sampling settings for HunyuanVideo 1.5 training are unclear, but it may be good to base them on `--timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.0` and adjust as needed.
- The recommended optimizer is `--optimizer_type Muon`, but it is only available in PyTorch 2.9 and later. If your PyTorch version is older, use `--optimizer_type adamw8bit` or similar.
- Memory saving options like `--fp8_scaled` (for DiT) and `--fp8_vl` (for Text Encoder) are available.
- `--gradient_checkpointing` is available for memory savings. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.

<details>
<summary>日本語</summary>

HunyuanVideo 1.5の学習は専用のスクリプト`hv_1_5_train_network.py`を使用します。

**Text-to-Video (T2V) 学習**

コマンド例は英語版を参照してください。

**Image-to-Video (I2V) 学習**

I2V学習を行う場合、`--task i2v`を指定し、`--image_encoder`パスを提供します：

コマンド例は英語版を参照してください。

- `hv_1_5_train_network.py`を使用します。
- `--vae`、`--text_encoder`、`--byt5`を指定する必要があります。
- `--network_module networks.lora_hv_1_5`を指定する必要があります。
- `--task`に`t2v`または`i2v`を指定する必要があります。
- I2V学習の場合は、`--image_encoder`が必要です。
- HunyuanVideo 1.5の学習に`--mixed_precision bf16`と`fp16`のどちらが良いかはまだ不明です。
- HunyuanVideo 1.5のタイムステップサンプリング設定は不明ですが、`--timestep_sampling shift --weighting_scheme none --discrete_flow_shift 2.0`をベースに調整すると良いかもしれません。
- オプティマイザには`--optimizer_type Muon`を推奨しますが、PyTorch 2.9以降でのみ利用可能です。PyTorchのバージョンが古い場合は`--optimizer_type adamw8bit`などを使用してください。
- `--fp8_scaled`（DiT用）や`--fp8_vl`（テキストエンコーダー用）などのメモリ節約オプションが利用可能です。
- メモリ節約のために`--gradient_checkpointing`が利用可能です。詳細は[HunyuanVideoドキュメント](./hunyuan_video.md#memory-optimization)を参照してください。

</details>

### Memory Optimization

- `--fp8_scaled` option is available for DiT to reduce memory usage. Quality may be slightly lower.
- `--fp8_vl` option is available to reduce memory usage of Text Encoder (Qwen2.5-VL).
- `--vae_sample_size` (default 128) controls VAE tiling size. Set to 256 if VRAM is sufficient for better quality. Set to 0 to disable tiling.
- `--vae_enable_patch_conv` enables patch-based convolution in VAE for memory optimization.
- `--gradient_checkpointing` and `--gradient_checkpointing_cpu_offload` are available for memory savings. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.
- `--blocks_to_swap` option is available to offload some blocks to CPU. The maximum number of blocks that can be offloaded is 51.

<details>
<summary>日本語</summary>

- DiTのメモリ使用量を削減するために、`--fp8_scaled`オプションを指定可能です。品質はやや低下する可能性があります。
- Text Encoder (Qwen2.5-VL)のメモリ使用量を削減するために、`--fp8_vl`オプションを指定可能です。
- `--vae_sample_size`（デフォルト128）でVAEのタイリングサイズを制御します。VRAMが十分な場合は256に設定すると品質が向上します。0に設定するとタイリングを無効にします。
- `--vae_enable_patch_conv`でVAEのパッチベース畳み込みを有効にし、メモリを最適化します。
- メモリ節約のために`--gradient_checkpointing`と`--gradient_checkpointing_cpu_offload`が利用可能です。詳細は[HunyuanVideoドキュメント](./hunyuan_video.md#memory-optimization)を参照してください。
- `--blocks_to_swap`オプションで、一部のブロックをCPUにオフロードできます。オフロード可能な最大ブロック数は51です。

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

### Other Options

For sample video generation during training, PyTorch Dynamo optimization, and other advanced configurations, refer to the [HunyuanVideo documentation](./hunyuan_video.md).

<details>
<summary>日本語</summary>

学習中のサンプル動画生成、PyTorch Dynamoによる最適化、その他の高度な設定については、[HunyuanVideoドキュメント](./hunyuan_video.md)を参照してください。

</details>

### Coverting LoRA weights to ComfyUI format / LoRA重みをComfyUI形式に変換する

A script is provided to convert HunyuanVideo 1.5 LoRA weights to ComfyUI format.

```bash
python src/musubi_tuner/networks/convert_hunyuan_video_1_5_lora_to_comfy.py \
    path/to/hv_1_5_lora.safetensors \
    path/to/output_comfy_lora.safetensors
```

- The script is `convert_hunyuan_video_1_5_lora_to_comfy.py`.
- The first argument is the input HunyuanVideo 1.5 LoRA weights file.
- The second argument is the output ComfyUI-format LoRA weights file.
- `--reverse` option is available to convert from ComfyUI format to HunyuanVideo 1.5 format. Only works for LoRA weights converted by this script.

<details>
<summary>日本語</summary>

HunyuanVideo 1.5のLoRA重みをComfyUI形式に変換するスクリプトが提供されています。

- スクリプトは`convert_hunyuan_video_1_5_lora_to_comfy.py`です。
- 最初の引数は入力のHunyuanVideo 1.5 LoRA重みファイルです。
- 2番目の引数は出力のComfyUI形式のLoRA重みファイルです。
- `--reverse`オプションで、ComfyUI形式からHunyuanVideo 1.5形式への変換も可能です。このオプションは、このスクリプトで変換されたLoRA重みに対してのみ機能します。  

</details>

## Inference / 推論

Inference uses a dedicated script `hv_1_5_generate_video.py`.

The recommended number of frames is 121 and the recommended number of inference steps is 50 in the official script, but the samples below use smaller values.

### Text-to-Video (T2V) Inference

```bash
python src/musubi_tuner/hv_1_5_generate_video.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --byt5 path/to/byt5 \
    --prompt "A cat" \
    --video_size 720 1280 --video_length 21 --infer_steps 25 \
    --attn_mode sdpa --fp8_scaled \
    --save_path path/to/save/dir --output_type video \
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

### Image-to-Video (I2V) Inference

For I2V inference, specify the `--image_path` and `--image_encoder`:

```bash
python src/musubi_tuner/hv_1_5_generate_video.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --byt5 path/to/byt5 \
    --image_encoder path/to/image_encoder \
    --image_path path/to/image.jpg \
    --prompt "A cat walking" \
    --video_size 720 1280 --video_length 21 --infer_steps 25 \
    --attn_mode sdpa --fp8_scaled \
    --save_path path/to/save/dir --output_type video \
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

- Uses `hv_1_5_generate_video.py`.
- **Requires** specifying `--vae`, `--text_encoder`, and `--byt5`.
- For I2V inference, `--image_path` and `--image_encoder` are required.
- `--video_size` is the size of the generated video, height and width are specified in that order.
- `--video_length` should be specified as "a multiple of 4 plus 1".
- `--prompt`: Prompt for generation.
- `--fp8_scaled` option is available for DiT to reduce memory usage. Quality may be slightly lower.
- `--vae_sample_size` (default 128) controls VAE tiling size. Set to 256 if VRAM is sufficient for better quality. Set to 0 to disable tiling.
- `--vae_enable_patch_conv` enables patch-based convolution in VAE for memory optimization.
- `--blocks_to_swap` option is available to offload some blocks to CPU. The maximum number of blocks that can be offloaded is 51.
- LoRA loading options (`--lora_weight`, `--lora_multiplier`, `--include_patterns`, `--exclude_patterns`) are available. `--lycoris` is also supported.
- `--guidance_scale` (default 6.0) controls the classifier-free guidance scale.
- `--flow_shift` (default 7.0) controls the discrete flow shift.
- `--save_merged_model` option is available to save the DiT model after merging LoRA weights. Inference is skipped if this is specified.

For 121 frames at 720p (1280x720) size, VRAM usage is around 20GB even with `--blocks_to_swap 51`.

<details>
<summary>日本語</summary>

HunyuanVideo 1.5の推論は専用のスクリプト`hv_1_5_generate_video.py`を使用します。

公式スクリプトの推奨フレーム数は121、推論ステップ数は50ですが、サンプルでは少なめにしています。

**Text-to-Video (T2V) 推論**

コマンド例は英語版を参照してください。

**Image-to-Video (I2V) 推論**

I2V推論を行う場合、`--image_path`と`--image_encoder`を指定します：

コマンド例は英語版を参照してください。

- `hv_1_5_generate_video.py`を使用します。
- `--vae`、`--text_encoder`、`--byt5`を指定する必要があります。
- I2V推論の場合は、`--image_path`と`--image_encoder`が必要です。
- `--video_size`は生成する動画のサイズで、高さと幅をその順番で指定します。
- `--video_length`は「4の倍数+1」を指定してください。
- `--prompt`: 生成用のプロンプトです。
- DiTのメモリ使用量を削減するために、`--fp8_scaled`オプションを指定可能です。品質はやや低下する可能性があります。
- `--blocks_to_swap`オプションで、一部のブロックをCPUにオフロードできます。オフロード可能な最大ブロック数は51です。
- `--vae_sample_size`（デフォルト128）でVAEのタイリングサイズを制御します。VRAMが十分な場合は256に設定すると品質が向上します。0に設定するとタイリングを無効にします。
- `--vae_enable_patch_conv`でVAEのパッチベース畳み込みを有効にし、メモリを最適化します。
- LoRAの読み込みオプション（`--lora_weight`、`--lora_multiplier`、`--include_patterns`、`--exclude_patterns`）が利用可能です。LyCORISもサポートされています。
- `--guidance_scale`（デフォルト6.0）は、classifier-free guidanceスケールを制御します。
- `--flow_shift`（デフォルト7.0）は、discrete flow shiftを制御します。
- `--save_merged_model`オプションは、LoRAの重みをマージした後にDiTモデルを保存するためのオプションです。これを指定すると推論はスキップされます。

720p (1280x720) サイズで121フレームの場合、`--blocks_to_swap 51`を指定してもVRAM使用量は約20GB程度になります。

</details>
