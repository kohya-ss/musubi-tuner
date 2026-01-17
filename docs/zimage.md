# Z-Image

## Overview / 概要

This document describes the usage of Z-Image architecture within the Musubi Tuner framework. Z-Image is a model architecture that supports text-to-image generation.

Pre-caching, training, and inference options can be found via `--help`. Many options are shared with HunyuanVideo, so refer to the [HunyuanVideo documentation](./hunyuan_video.md) as needed.

This feature is experimental.

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内でのZ-Imageアーキテクチャの使用法について説明しています。Z-Imageはテキストから画像を生成することができるモデルアーキテクチャです。Z-Imageは現在蒸留モデルであるTurbo版しかリリースされていないため、学習は不安定です。モデルのダウンロードの項も参照してください。

事前キャッシング、学習、推論のオプションは`--help`で確認してください。HunyuanVideoと共通のオプションが多くありますので、必要に応じて[HunyuanVideoのドキュメント](./hunyuan_video.md)も参照してください。

この機能は実験的なものです。

</details>

## Download the model / モデルのダウンロード

You need to download the DiT, VAE, and Text Encoder (Qwen3) models. 

Since the base model has not been released, it is recommended to use AI Toolkit/ostris's De-Turbo model. Download `z_image_de_turbo_v1_bf16.safetensors` from [ostris/Z-Image-De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) and use it as the DiT model.

The Turbo version DiT, VAE, and Text Encoder can be obtained from Tongyi-MAI's official repository or ComfyUI weights. You can use either of the following:

- **Official Repository**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/)
    - For DiT and Text Encoder, download all the split files and specify the first file (e.g., `00001-of-00004.safetensors`) in the arguments.
    - You do not need to download files other than `*.safetensors`.
- **ComfyUI Weights**: [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo)

You need to prepare the following models:

- **DiT**: The transformer model.
- **VAE**: The autoencoder model.
- **Text Encoder**: Qwen3 model.

As another option, you can also use ostris's [ostris/zimage_turbo_training_adapter](https://huggingface.co/ostris/zimage_turbo_training_adapter) to train by combining the Turbo version with an adapter. In this case, download `zimage_turbo_training_adapter_v2.safetensors`, etc., and specify this LoRA weight in the `--base_weights` option during training.

We would like to express our deep gratitude to ostris for providing the De-Turbo model and Training Adapter.

<details>
<summary>日本語</summary>

DiT, VAE, Text Encoder (Qwen3) のモデルをダウンロードする必要があります。

Baseモデルがリリースされていないため、AI Toolkit/ostris氏のDe-Turboモデルを使用することをお勧めします。[ostris/Z-Image-De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo) から `z_image_de_turbo_v1_bf16.safetensors` をダウンロードし、DiTモデルとして使用してください。

Turbo版のDiT、VAEとText EncoderはTongyi-MAIの公式リポジトリまたはComfyUI用重みから取得できます。以下のいずれかを使用してください：

- **公式リポジトリ**: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/)
    - DiT、Text Encoderは、分割された複数のファイルをすべてダウンロードし、引数には `00001-of-00004.safetensors` のような最初のファイルを指定してください。
    - `*.safetensors` ファイル以外はダウンロードする必要はありません。
- **ComfyUI用重み**: [Comfy-Org/z_image_turbo](https://huggingface.co/Comfy-Org/z_image_turbo)

以下のモデルを準備してください：

- **DiT**: Transformerモデル。
- **VAE**: Autoencoderモデル。
- **Text Encoder**: Qwen3モデル。

別のオプションとして、ostris氏の [ostris/zimage_turbo_training_adapter](https://huggingface.co/ostris/zimage_turbo_training_adapter) を使用して、Turbo版とAdapterを組み合わせて学習することもできます。この場合は、`zimage_turbo_training_adapter_v2.safetensors` 等をダウンロードし、学習時に `--base_weights` オプションにこのLoRA重みを指定してください。

De-TurboモデルおよびTraining Adapterを提供してくださった ostris 氏に深く感謝します。

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

### Converting LoRA weights to Diffusers format for ComfyUI / LoRA重みをComfyUIで使用可能なDiffusers形式に変換する

A script is provided to convert Z-Image LoRA weights to Diffusers format for ComfyUI.

```bash
python src/musubi_tuner/networks/convert_lora.py \
    --input path/to/zimage_lora.safetensors \
    --output path/to/output_diffusers_lora.safetensors \
    --target other
```

- The script is `convert_lora.py`.
- `--input` argument is the input Z-Image LoRA weights file.
- `--output` argument is the output Diffusers format LoRA weights file.
- `--target other` means Diffusers format can be used in ComfyUI.

`networks\convert_z_image_lora_to_comfy.py` can also be used for this purpose, but the converted weights may not work correctly with nunchaku.

<details>
<summary>日本語</summary>

Z-ImageのLoRA重みをComfyUIで使用できるDiffusers形式に変換するスクリプトが提供されています。

- スクリプトは`convert_lora.py`です。
- `--input`引数は入力のZ-Image LoRA重みファイルです。
- `--output`引数は出力のDiffusers形式のLoRA重みファイルです。
- `--target other`はComfyUIで使用できるDiffusers形式を意味します。

`networks\convert_z_image_lora_to_comfy.py`もこの目的で使用できますが、変換された重みがnunchakuで正しく動作しない可能性があります。

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

### Sample images during training with De-Turbo model or Training Adapter / De-TurboモデルまたはTraining Adapterで学習中にサンプル画像を生成する

When training with the De-Turbo model or Training Adapter, add negative prompt and CFG scale to the sampling options to generate sample images with CFG. It is also recommended to increase the number of steps. `--l` specifies the CFG scale.

```text
A beautiful landscape painting of mountains during sunset.  --n bad quality --w 1280 --h 720 --fs 3 --s 20 --d 1234 --l 5
```

<details>
<summary>日本語</summary>
 
 De-TurboモデルまたはTraining Adapterで学習する場合、サンプリングオプションにネガティブプロンプトとCFGスケールを追加して、CFGありでサンプル画像を生成してください。またステップ数も増やすことをお勧めします。`--l`でCFGスケールを指定します。
 
 ```text
A beautiful landscape painting of mountains during sunset.  --n bad quality --w 1280 --h 720 --fs 3 --s 20 --d 1234 --l 5
```

 </details>

## Finetuning

Finetuning uses a dedicated script `zimage_train.py`. This script performs full finetuning of the model, not LoRA. Sample usage is as follows:

```bash
accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/zimage_train.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model \
    --text_encoder path/to/text_encoder \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 --gradient_checkpointing \
    --optimizer_type adafactor --learning_rate 1e-6 --fused_backward_pass \
    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" \
    --max_grad_norm 0 --lr_scheduler constant_with_warmup --lr_warmup_steps 10 \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-model
```

- Uses `zimage_train.py`.
- Finetuning requires a large amount of VRAM. The use of memory saving options is strongly recommended.
- `--full_bf16`: Loads the model weights in bfloat16 format to significantly reduce VRAM usage. 
- `--optimizer_type adafactor`: Using Adafactor is recommended for finetuning.
- `--fused_backward_pass`: Reduces VRAM usage during the backward pass when using Adafactor.
- `--mem_eff_save`: Reduces main memory (RAM) usage when saving checkpoints.
- `--blocks_to_swap`: Swaps model blocks between VRAM and main memory to reduce VRAM usage. This is effective when VRAM is limited.
- `--disable_numpy_memmap`: Disables numpy memory mapping for model loading, loading with standard file read. Increases RAM usage but may speed up model loading in some cases.

`--full_bf16` reduces VRAM usage by about 30GB but may impact model accuracy as the weights are kept in bfloat16. Note that the optimizer state is still kept in float32. In addition, it is recommended to use this with an optimizer that supports stochastic rounding. In this repository, Adafactor optimizer with `--fused_backward_pass` option supports stochastic rounding.

When using `--mem_eff_save`, please note that traditional saving methods are still used when saving the optimizer state in `--save_state`, requiring about 20GB of main memory.

### Recommended Settings

We are still exploring the optimal settings. The configurations above are just examples, so please adjust them as needed. We welcome your feedback.

If you have ample VRAM, you can use any optimizer of your choice. `--full_bf16` is not recommended.

For limited VRAM environments (e.g., 48GB or less), you may need to use `--full_bf16`, the Adafactor optimizer, and `--fused_backward_pass`. Settings above are the recommended options for that case. Please adjust `--lr_warmup_steps` to a value between approximately 10 and 100.

`--fused_backward_pass` is not currently compatible with gradient accumulation, and max grad norm may not function as expected, so it is recommended to specify `--max_grad_norm 0`.

If your VRAM is even more constrained, you can enable block swapping by specifying a value for `--blocks_to_swap`.

Experience with other models suggests that the learning rate may need to be reduced significantly; something in the range of 1e-6 to 1e-5 might be a good place to start.

<details>
<summary>日本語</summary>

Finetuningは専用のスクリプト`zimage_train.py`を使用します。このスクリプトはLoRAではなく、モデル全体のfinetuningを行います。

- `zimage_train.py`を使用します。
- Finetuningは大量のVRAMを必要とします。メモリ節約オプションの使用を強く推奨します。
- `--full_bf16`: モデルの重みをbfloat16形式で読み込み、VRAM使用量を大幅に削減します。
- `--optimizer_type adafactor`: FinetuningではAdafactorの使用が推奨されます。
- `--fused_backward_pass`: Adafactor使用時に、backward pass中のVRAM使用量を削減します。
- `--mem_eff_save`: チェックポイント保存時のメインメモリ（RAM）使用量を削減します。
- `--blocks_to_swap`: モデルのブロックをVRAMとメインメモリ間でスワップし、VRAM使用量を削減します。VRAMが少ない場合に有効です。
- `--disable_numpy_memmap`: モデル読み込み時のnumpyメモリマッピングを無効化し、標準のファイル読み込みで読み込みを行います。RAM使用量は増加しますが、場合によってはモデルの読み込みが高速化されます。

`--full_bf16`はVRAM使用量を約30GB削減しますが、重みがbfloat16で保持されるため、モデルの精度に影響を与える可能性があります。オプティマイザの状態はfloat32で保持されます。また、効率的な学習のために、stochastic roundingをサポートするオプティマイザとの併用が推奨されます。このリポジトリでは、`adafactor`オプティマイザに`--fused_backward_pass`オプションの組み合わせでstochastic roundingをサポートしています。

`--mem_eff_save`を使用する場合でも、`--save_state`においてはオプティマイザの状態を保存する際に従来の保存方法が依然として使用されるため、約20GBのメインメモリが必要であることに注意してください。

### 推奨設定

最適な設定はまだ調査中です。上記の構成はあくまで一例ですので、必要に応じて調整してください。フィードバックをお待ちしております。

十分なVRAMがある場合は、お好みのオプティマイザを使用できます。`--full_bf16`は推奨されません。

VRAMが限られている環境（例：48GB以下）の場合は、`--full_bf16`、Adafactorオプティマイザ、および`--fused_backward_pass`を使用する必要があるかもしれません。上記の設定はその場合の推奨オプションです。`--lr_warmup_steps`は約10から100の間の値に調整してください。

現時点では`--fused_backward_pass`はgradient accumulationに対応していません。またmax grad normも想定通りに動作しない可能性があるため、`--max_grad_norm 0`を指定することを推奨します。

さらにVRAMが制約されている場合は、`--blocks_to_swap`に値を指定してブロックスワッピングを有効にできます。

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
