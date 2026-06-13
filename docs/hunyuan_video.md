> 📝 Click on the language section to expand / 言語をクリックして展開

# HunyuanVideo

## Overview / 概要

This document describes the usage of the [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) architecture within the Musubi Tuner framework. HunyuanVideo is a video generation model that supports text-to-video generation.

This feature is experimental.

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内での[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)アーキテクチャの使用法について説明しています。HunyuanVideoはテキストから動画を生成するモデルです。

この機能は実験的なものです。
</details>

## Download the model / モデルのダウンロード

There are two ways to download the model.

### Use the Official HunyuanVideo Model / 公式HunyuanVideoモデルを使う

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

### Using ComfyUI Models for Text Encoder / Text EncoderにComfyUI提供のモデルを使う

This method is easier.

For DiT and VAE, use the HunyuanVideo models.

From https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/transformers, download [mp_rank_00_model_states.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt) and place it in your chosen directory.

(Note: The fp8 model on the same page is unverified.)

If you are training with `--fp8_base`, you can use `mp_rank_00_model_states_fp8.safetensors` from [here](https://huggingface.co/kohya-ss/HunyuanVideo-fp8_e4m3fn-unofficial) instead of `mp_rank_00_model_states.pt`. (This file is unofficial and simply converts the weights to float8_e4m3fn.)

From https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/vae, download [pytorch_model.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/vae/pytorch_model.pt) and place it in your chosen directory.

For the Text Encoder, use the models provided by ComfyUI. Refer to [ComfyUI's page](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/), from https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/tree/main/split_files/text_encoders, download `llava_llama3_fp16.safetensors` (Text Encoder 1, LLM) and `clip_l.safetensors` (Text Encoder 2, CLIP) and place them in your chosen directory.

(Note: The fp8 LLM model on the same page is unverified.)

<details>
<summary>日本語</summary>

以下のいずれかの方法で、モデルをダウンロードしてください。

### HunyuanVideoの公式モデルを使う 

[公式のREADME](https://github.com/Tencent/HunyuanVideo/blob/main/ckpts/README.md)を参考にダウンロードし、任意のディレクトリに以下のように配置します。

```
  ckpts
    ├──hunyuan-video-t2v-720p
    │  ├──transformers
    │  ├──vae
    ├──text_encoder
    ├──text_encoder_2
    ├──...
```

### Text EncoderにComfyUI提供のモデルを使う

こちらの方法の方がより簡単です。DiTとVAEのモデルはHumyuanVideoのものを使用します。

https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/transformers から、[mp_rank_00_model_states.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt) をダウンロードし、任意のディレクトリに配置します。

（同じページにfp8のモデルもありますが、未検証です。）

`--fp8_base`を指定して学習する場合は、`mp_rank_00_model_states.pt`の代わりに、[こちら](https://huggingface.co/kohya-ss/HunyuanVideo-fp8_e4m3fn-unofficial)の`mp_rank_00_model_states_fp8.safetensors`を使用可能です。（このファイルは非公式のもので、重みを単純にfloat8_e4m3fnに変換したものです。）

また、https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/vae から、[pytorch_model.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/vae/pytorch_model.pt) をダウンロードし、任意のディレクトリに配置します。

Text EncoderにはComfyUI提供のモデルを使用させていただきます。[ComyUIのページ](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/)を参考に、https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/tree/main/split_files/text_encoders から、llava_llama3_fp16.safetensors （Text Encoder 1、LLM）と、clip_l.safetensors （Text Encoder 2、CLIP）をダウンロードし、任意のディレクトリに配置します。

（同じページにfp8のLLMモデルもありますが、動作未検証です。）

</details>

## Pre-caching / 事前キャッシング

### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching is required. Create the cache using the following command:

If you have installed using pip:

```bash
python src/musubi_tuner/cache_latents.py --dataset_config path/to/toml --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```

If you have installed with `uv`, you can use `uv run --extra cu124` to run the script. If CUDA 12.8 or 13.0 is supported, `uv run --extra cu128` or `uv run --extra cu130` is also available. Other scripts can be run in the same way. (Note that the installation with `uv` is experimental. Feedback is welcome. If you encounter any issues, please use the pip-based installation.)

```bash
uv run --extra cu124 src/musubi_tuner/cache_latents.py --dataset_config path/to/toml --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```

For additional options, use `python src/musubi_tuner/cache_latents.py --help`.

If you're running low on VRAM, reduce `--vae_spatial_tile_sample_min_size` to around 128 and lower the `--batch_size` (`--vae_spatial_tile_sample_min_size` may not exist in architectures other than HunyuanVideo, see the documentation for each architecture).

If you are using an AMD GPU and/or are experiencing slow latent caching, consider trying `--disable_cudnn_backend`. For some details, see [this pull request](https://github.com/kohya-ss/musubi-tuner/pull/592).

Use `--debug_mode image` to display dataset images and captions in a new window, or `--debug_mode console` to display them in the console (requires `ascii-magic`). 

With `--debug_mode video`, images or videos will be saved in the cache directory (please delete them after checking). The bitrate of the saved video is set to 1Mbps for preview purposes. The images decoded from the original video (not degraded) are used for the cache (for training).

When `--debug_mode` is specified, the actual caching process is not performed.

By default, cache files not included in the dataset are automatically deleted. You can still keep cache files as before by specifying `--keep_cache`.

<details>
<summary>日本語</summary>

latentの事前キャッシュは必須です。以下のコマンドを使用して、事前キャッシュを作成してください。（pipによるインストールの場合）

```bash
python src/musubi_tuner/cache_latents.py --dataset_config path/to/toml --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```

uvでインストールした場合は、`uv run --extra cu124 python src/musubi_tuner/cache_latents.py ...`のように、`uv run --extra cu124`を先頭につけてください。CUDA 12.8や13.0に対応している場合は、`uv run --extra cu128`や`uv run --extra cu130`も利用可能です。以下のコマンドも同様です。

その他のオプションは`python src/musubi_tuner/cache_latents.py --help`で確認できます。

VRAMが足りない場合は、`--vae_spatial_tile_sample_min_size`を128程度に減らし、`--batch_size`を小さくしてください。

`--debug_mode image` を指定するとデータセットの画像とキャプションが新規ウィンドウに表示されます。`--debug_mode console`でコンソールに表示されます（`ascii-magic`が必要）。

`--debug_mode video`で、キャッシュディレクトリに画像または動画が保存されます（確認後、削除してください）。動画のビットレートは確認用に低くしてあります。実際には元動画の画像が学習に使用されます。

`--debug_mode`指定時は、実際のキャッシュ処理は行われません。

デフォルトではデータセットに含まれないキャッシュファイルは自動的に削除されます。`--keep_cache`を指定すると、キャッシュファイルを残すことができます。

</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text Encoder output pre-caching is required. Create the cache using the following command:

```bash
python src/musubi_tuner/cache_text_encoder_outputs.py --dataset_config path/to/toml  --text_encoder1 path/to/ckpts/text_encoder --text_encoder2 path/to/ckpts/text_encoder_2 --batch_size 16
```

or for uv:

```bash
uv run --extra cu124 src/musubi_tuner/cache_text_encoder_outputs.py --dataset_config path/to/toml  --text_encoder1 path/to/ckpts/text_encoder --text_encoder2 path/to/ckpts/text_encoder_2 --batch_size 16
```

For additional options, use `python src/musubi_tuner/cache_text_encoder_outputs.py --help`.

Adjust `--batch_size` according to your available VRAM.

For systems with limited VRAM (less than ~16GB), use `--fp8_llm` to run the LLM in fp8 mode.

By default, cache files not included in the dataset are automatically deleted. You can still keep cache files as before by specifying `--keep_cache`.

<details>
<summary>日本語</summary>

Text Encoder出力の事前キャッシュは必須です。以下のコマンドを使用して、事前キャッシュを作成してください。

```bash
python src/musubi_tuner/cache_text_encoder_outputs.py --dataset_config path/to/toml  --text_encoder1 path/to/ckpts/text_encoder --text_encoder2 path/to/ckpts/text_encoder_2 --batch_size 16
```

その他のオプションは`python src/musubi_tuner/cache_text_encoder_outputs.py --help`で確認できます。

`--batch_size`はVRAMに合わせて調整してください。

VRAMが足りない場合（16GB程度未満の場合）は、`--fp8_llm`を指定して、fp8でLLMを実行してください。

デフォルトではデータセットに含まれないキャッシュファイルは自動的に削除されます。`--keep_cache`を指定すると、キャッシュファイルを残すことができます。

</details>

## Training / 学習

Start training using the following command (input as a single line):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hv_train_network.py 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt 
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base 
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing 
    --max_data_loader_n_workers 2 --persistent_data_loader_workers 
    --network_module networks.lora --network_dim 32 
    --timestep_sampling shift --discrete_flow_shift 7.0 
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42
    --output_dir path/to/output_dir --output_name name-of-lora
```

or for uv:

```bash
uv run --extra cu124 accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hv_train_network.py 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt 
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base 
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing 
    --max_data_loader_n_workers 2 --persistent_data_loader_workers 
    --network_module networks.lora --network_dim 32 
    --timestep_sampling shift --discrete_flow_shift 7.0 
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42
    --output_dir path/to/output_dir --output_name name-of-lora
```

If the details of the image are not learned well, try lowering the discrete flow shift to around 3.0.

The training settings are still experimental. Appropriate learning rates, training steps, timestep distribution, loss weighting, etc. are not yet known. Feedback is welcome.

For additional options, use `python src/musubi_tuner/hv_train_network.py --help` (note that many options are unverified).

### Save Precision

`--save_precision` specifies the precision used when saving network weights. The available values are `float`/`fp32`, `fp16`, and `bf16`. If omitted, network weights are saved in fp32, matching the precision used to train the LoRA weights.

Saving in fp32 preserves more information for post-processing such as post-hoc EMA, merging, extraction, and weight analysis. When training with `--mixed_precision bf16`(`fp16`), the saved LoRA file may be about twice as large as before. To keep the previous behavior, specify `--save_precision bf16`(`fp16`).

### Memory Optimization

`--gradient_checkpointing` enables gradient checkpointing to reduce VRAM usage. Gradient checkpointing is a memory-saving technique that trades off computation time for memory usage by recomputing certain intermediate results during the backward pass instead of storing them all in memory. This is particularly useful for training large models such as HunyuanVideo, where VRAM can be a limiting factor. However, it may slow down training. If you have sufficient VRAM, you can disable it.

Specifying `--fp8_base` runs DiT in fp8 mode. Without this flag, mixed precision data type will be used. fp8 can significantly reduce memory consumption but may impact output quality. If `--fp8_base` is not specified, 24GB or more VRAM is recommended. Use `--blocks_to_swap` as needed.

If you're running low on VRAM, use `--blocks_to_swap` to offload some blocks to CPU. Maximum value is 36.

(The idea of block swap is based on the implementation by 2kpr. Thanks again to 2kpr.)

`--use_pinned_memory_for_block_swap` can be used to enable pinned memory for block swapping. This may improve performance when swapping blocks between CPU and GPU. However, it may increase shared VRAM usage on Windows systems. Use this option based on your system configuration (e.g., available system RAM and VRAM). In some environments, not specifying this option may result in faster performance.

`--gradient_checkpointing_cpu_offload` can be used to offload activations to CPU when using gradient checkpointing. This can further reduce VRAM usage, but may slow down training. This option is especially useful when the latent resolution (or video length) is high and VRAM is limited. This option must be used together with `--gradient_checkpointing`. See [PR #537](https://github.com/kohya-ss/musubi-tuner/pull/537) for more details.

### Attention

Use `--sdpa` for PyTorch's scaled dot product attention. Use `--flash_attn` for [FlashAttention](https://github.com/Dao-AILab/flash-attention). Use `--xformers` for xformers, but specify `--split_attn` when using xformers. `--sage_attn` for SageAttention, but SageAttention is not yet supported for training, so it raises a ValueError.

`--split_attn` processes attention in chunks. Speed may be slightly reduced, but VRAM usage is slightly reduced.

### Timestep Sampling
You can also specify the range of timesteps 
with `--min_timestep` and `--max_timestep`. See [advanced configuration](../advanced_config.md#specify-time-step-range-for-training--学習時のタイムステップ範囲の指定) for details.

`--show_timesteps` can be set to `image` (requires `matplotlib`) or `console` to display timestep distribution and loss weighting during training. (When using `flux_shift` and `qwen_shift`, the distribution will be for images with a resolution of 1024x1024.)

### Other Options

The format of LoRA trained is the same as `sd-scripts`.

You can record logs during training. Refer to [Save and view logs in TensorBoard format](../advanced_config.md#save-and-view-logs-in-tensorboard-format--tensorboard形式のログの保存と参照).

For PyTorch Dynamo optimization, refer to [this document](../advanced_config.md#pytorch-dynamo-optimization-for-model-training--モデルの学習におけるpytorch-dynamoの最適化).

For sample image generation during training, refer to [this document](../sampling_during_training.md). For advanced configuration, refer to [this document](../advanced_config.md).

<details>
<summary>日本語</summary>

以下のコマンドを使用して、学習を開始します（実際には一行で入力してください）。

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hv_train_network.py 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt 
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base 
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing 
    --max_data_loader_n_workers 2 --persistent_data_loader_workers 
    --network_module networks.lora --network_dim 32 
    --timestep_sampling shift --discrete_flow_shift 7.0 
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42
    --output_dir path/to/output_dir --output_name name-of-lora
```

ディテールが甘くなる場合は、discrete flow shiftを3.0程度に下げてみてください。

ただ、適切な学習率、学習ステップ数、timestepsの分布、loss weightingなどのパラメータは、以前として不明な点が数多くあります。情報提供をお待ちしています。

その他のオプションは`python src/musubi_tuner/hv_train_network.py --help`で確認できます（ただし多くのオプションは動作未確認です）。

**保存精度**

`--save_precision`でネットワーク重みを保存する際の精度を指定できます。指定可能な値は`float`/`fp32`、`fp16`、`bf16`です。省略時は、LoRA重みの学習精度に合わせてfp32で保存されます。

fp32で保存すると、post-hoc EMA、マージ、抽出、重み解析などの後処理でより多くの情報を保持できます。`--mixed_precision bf16`(`fp16`)で学習している場合、保存されるLoRAファイルのサイズが従来のおよそ2倍になることがあります。従来と同じ挙動にしたい場合は`--save_precision bf16`(`fp16`)を指定してください。

**メモリ最適化**

`--gradient_checkpointing`でgradient checkpointingを有効にします。VRAM使用量を削減できます。gradient checkpointingは、バックワードパス中に一部の中間結果をすべてメモリに保存するのではなく、再計算することで、計算時間とメモリ使用量をトレードオフするメモリ節約技術です。HunyuanVideoのような大規模モデルの学習ではVRAMが制約となることが多いため、特に有用です。ただし学習が遅くなる可能性があります。十分なVRAMがある場合は無効にしても構いません。

`--fp8_base`を指定すると、DiTがfp8で学習されます。未指定時はmixed precisionのデータ型が使用されます。fp8は大きく消費メモリを削減できますが、品質は低下する可能性があります。`--fp8_base`を指定しない場合はVRAM 24GB以上を推奨します。また必要に応じて`--blocks_to_swap`を使用してください。

VRAMが足りない場合は、`--blocks_to_swap`を指定して、一部のブロックをCPUにオフロードしてください。最大36が指定できます。

（block swapのアイデアは2kpr氏の実装に基づくものです。2kpr氏にあらためて感謝します。）

`--use_pinned_memory_for_block_swap`を指定すると、block swapにピン留めメモリを使用します。CPUとGPU間でブロックをスワップする際のパフォーマンスが向上する可能性があります。ただし、Windows環境では共有VRAM使用量が増加する可能性があります。システム構成（利用可能なシステムRAMやVRAMなど）に応じて、このオプションを使用してください。環境によっては指定しないほうが高速になる場合もあります。

`--gradient_checkpointing_cpu_offload`を指定すると、gradient checkpointing使用時にアクティベーションをCPUにオフロードします。これによりVRAM使用量をさらに削減できますが、学習が遅くなる可能性があります。latent解像度（または動画長）が高く、VRAMが限られている場合に特に有用です。このオプションは`--gradient_checkpointing`と併用する必要があります。詳細は[PR #537](https://github.com/Dao-AILab/flash-attention/pull/537)を参照してください。

**Attention**

`--sdpa`でPyTorchのscaled dot product attentionを使用します。`--flash_attn`で[FlashAttention]:(https://github.com/Dao-AILab/flash-attention)を使用します。`--xformers`でxformersの利用も可能ですが、xformersを使う場合は`--split_attn`を指定してください。`--sage_attn`でSageAttentionを使用しますが、SageAttentionは現時点では学習に未対応のため、エラーが発生します。

`--split_attn`を指定すると、attentionを分割して処理します。速度が多少低下しますが、VRAM使用量はわずかに減ります。

**タイムステップサンプリング**

`--min_timestep`と`--max_timestep`を指定すると、学習時のタイムステップの範囲を指定できます。詳細は[高度な設定](../advanced_config.md#specify-time-step-range-for-training--学習時のタイムステップ範囲の指定)を参照してください。

`--show_timesteps`に`image`（`matplotlib`が必要）または`console`を指定すると、学習時のtimestepsの分布とtimestepsごとのloss weightingが確認できます。（`flux_shift`と`qwen_shift`を使用する場合は画像の解像度が1024x1024の場合の分布になります。）

**その他のオプション**

学習されるLoRAの形式は、`sd-scripts`と同じです。

学習時のログの記録が可能です。[TensorBoard形式のログの保存と参照](../advanced_config.md#save-and-view-logs-in-tensorboard-format--tensorboard形式のログの保存と参照)を参照してください。

PyTorch Dynamoによる最適化を行う場合は、[こちら](../advanced_config.md#pytorch-dynamo-optimization-for-model-training--モデルの学習におけるpytorch-dynamoの最適化)を参照してください。

学習中のサンプル画像生成については、[こちらのドキュメント](../sampling_during_training.md)を参照してください。その他の高度な設定については[こちらのドキュメント](../advanced_config.md)を参照してください。

</details>

### Merging LoRA Weights / LoRAの重みのマージ

```bash
python src/musubi_tuner/merge_lora.py \
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --lora_weight path/to/lora.safetensors \
    --save_merged_model path/to/merged_model.safetensors \
    --device cpu \
    --lora_multiplier 1.0
```

or for uv:

```bash
uv run --extra cu124 src/musubi_tuner/merge_lora.py \
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --lora_weight path/to/lora.safetensors \
    --save_merged_model path/to/merged_model.safetensors \
    --device cpu \
    --lora_multiplier 1.0
```

Specify the device to perform the calculation (`cpu` or `cuda`, etc.) with `--device`. Calculation will be faster if `cuda` is specified.

Specify the LoRA weights to merge with `--lora_weight` and the multiplier for the LoRA weights with `--lora_multiplier`. Multiple values can be specified, and the number of values must match.

<details>
<summary>日本語</summary>

```bash
python src/musubi_tuner/merge_lora.py \
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --lora_weight path/to/lora.safetensors \
    --save_merged_model path/to/merged_model.safetensors \
    --device cpu \
    --lora_multiplier 1.0
```

`--device`には計算を行うデバイス（`cpu`または`cuda`等）を指定してください。`cuda`を指定すると計算が高速化されます。

`--lora_weight`にはマージするLoRAの重みを、`--lora_multiplier`にはLoRAの重みの係数を、それぞれ指定してください。複数個が指定可能で、両者の数は一致させてください。

</details>

## Inference / 推論

Generate videos using the following command:

```bash
python src/musubi_tuner/hv_generate_video.py --fp8 --video_size 544 960 --video_length 5 --infer_steps 30 
    --prompt "A cat walks on the grass, realistic style."  --save_path path/to/save/dir --output_type both 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt --attn_mode sdpa --split_attn
    --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt 
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 
    --text_encoder1 path/to/ckpts/text_encoder 
    --text_encoder2 path/to/ckpts/text_encoder_2 
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

or for uv:

```bash
uv run --extra cu124 src/musubi_tuner/hv_generate_video.py --fp8 --video_size 544 960 --video_length 5 --infer_steps 30 
    --prompt "A cat walks on the grass, realistic style."  --save_path path/to/save/dir --output_type both 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt --attn_mode sdpa --split_attn
    --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt 
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 
    --text_encoder1 path/to/ckpts/text_encoder 
    --text_encoder2 path/to/ckpts/text_encoder_2 
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

For additional options, use `python src/musubi_tuner/hv_generate_video.py --help`.

Specifying `--fp8` runs DiT in fp8 mode. fp8 can significantly reduce memory consumption but may impact output quality.

`--fp8_fast` option is also available for faster inference on RTX 40x0 GPUs. This option requires `--fp8` option. 

If you're running low on VRAM, use `--blocks_to_swap` to offload some blocks to CPU. Maximum value is 38.

For `--attn_mode`, specify either `flash`, `torch`, `sageattn`, `xformers`, or `sdpa` (same as `torch`). These correspond to FlashAttention, scaled dot product attention, SageAttention, and xformers, respectively. Default is `torch`. SageAttention is effective for VRAM reduction.

Specifing `--split_attn` will process attention in chunks. Inference with SageAttention is expected to be about 10% faster.

For `--output_type`, specify either `both`, `latent`, `video` or `images`. `both` outputs both latents and video. Recommended to use `both` in case of Out of Memory errors during VAE processing. You can specify saved latents with `--latent_path` and use `--output_type video` (or `images`) to only perform VAE decoding.

`--seed` is optional. A random seed will be used if not specified.

`--video_length` should be specified as "a multiple of 4 plus 1".

`--flow_shift` can be specified to shift the timestep (discrete flow shift). The default value when omitted is 7.0, which is the recommended value for 50 inference steps. In the HunyuanVideo paper, 7.0 is recommended for 50 steps, and 17.0 is recommended for less than 20 steps (e.g. 10).

By specifying `--video_path`, video2video inference is possible. Specify a video file or a directory containing multiple image files (the image files are sorted by file name and used as frames). An error will occur if the video is shorter than `--video_length`. You can specify the strength with `--strength`. It can be specified from 0 to 1.0, and the larger the value, the greater the change from the original video.

Note that video2video inference is experimental.

`--compile` option enables PyTorch's compile feature (experimental). Requires triton. On Windows, also requires Visual C++ build tools installed and PyTorch>=2.6.0 (Visual C++ build tools is also required). See [the torch.compile documentation](torch_compile.md) for details.

The `--compile` option takes a long time to run the first time, but speeds up on subsequent runs.

You can save the DiT model after LoRA merge with the `--save_merged_model` option. Specify `--save_merged_model path/to/merged_model.safetensors`. Note that inference will not be performed when this option is specified.

<details>
<summary>日本語</summary>

以下のコマンドを使用して動画を生成します。

```bash
python src/musubi_tuner/hv_generate_video.py --fp8 --video_size 544 960 --video_length 5 --infer_steps 30 
    --prompt "A cat walks on the grass, realistic style."  --save_path path/to/save/dir --output_type both 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt --attn_mode sdpa --split_attn
    --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt 
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 
    --text_encoder1 path/to/ckpts/text_encoder 
    --text_encoder2 path/to/ckpts/text_encoder_2 
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

その他のオプションは`python src/musubi_tuner/hv_generate_video.py --help`で確認できます。

`--fp8`を指定すると、DiTがfp8で推論されます。fp8は大きく消費メモリを削減できますが、品質は低下する可能性があります。

RTX 40x0シリーズのGPUを使用している場合は、`--fp8_fast`オプションを指定することで、高速推論が可能です。このオプションを指定する場合は、`--fp8`も指定してください。

VRAMが足りない場合は、`--blocks_to_swap`を指定して、一部のブロックをCPUにオフロードしてください。最大38が指定できます。

`--attn_mode`には`flash`、`torch`、`sageattn`、`xformers`または`sdpa`（`torch`指定時と同じ）のいずれかを指定してください。それぞれFlashAttention、scaled dot product attention、SageAttention、xformersに対応します。デフォルトは`torch`です。SageAttentionはVRAMの削減に有効です。

`--split_attn`を指定すると、attentionを分割して処理します。SageAttention利用時で10%程度の高速化が見込まれます。

`--output_type`には`both`、`latent`、`video`、`images`のいずれかを指定してください。`both`はlatentと動画の両方を出力します。VAEでOut of Memoryエラーが発生する場合に備えて、`both`を指定することをお勧めします。`--latent_path`に保存されたlatentを指定し、`--output_type video` （または`images`）としてスクリプトを実行すると、VAEのdecodeのみを行えます。

`--seed`は省略可能です。指定しない場合はランダムなシードが使用されます。

`--video_length`は「4の倍数+1」を指定してください。

`--flow_shift`にタイムステップのシフト値（discrete flow shift）を指定可能です。省略時のデフォルト値は7.0で、これは推論ステップ数が50の時の推奨値です。HunyuanVideoの論文では、ステップ数50の場合は7.0、ステップ数20未満（10など）で17.0が推奨されています。

`--video_path`に読み込む動画を指定すると、video2videoの推論が可能です。動画ファイルを指定するか、複数の画像ファイルが入ったディレクトリを指定してください（画像ファイルはファイル名でソートされ、各フレームとして用いられます）。`--video_length`よりも短い動画を指定するとエラーになります。`--strength`で強度を指定できます。0~1.0で指定でき、大きいほど元の動画からの変化が大きくなります。

なおvideo2video推論の処理は実験的なものです。

`--compile`オプションでPyTorchのコンパイル機能を有効にします（実験的機能）。tritonのインストールが必要です。また、WindowsではVisual C++ build toolsが必要で、かつPyTorch>=2.6.0でのみ動作します。詳細は[torch.compileのドキュメント](torch_compile.md)を参照してください。

`--compile`は初回実行時にかなりの時間がかかりますが、2回目以降は高速化されます。

`--save_merged_model`オプションで、LoRAマージ後のDiTモデルを保存できます。`--save_merged_model path/to/merged_model.safetensors`のように指定してください。なおこのオプションを指定すると推論は行われません。

</details>

### Inference with SkyReels V1 / SkyReels V1での推論

SkyReels V1 T2V and I2V models are supported (inference only). 

The model can be downloaded from [here](https://huggingface.co/Kijai/SkyReels-V1-Hunyuan_comfy). Many thanks to Kijai for providing the model. `skyreels_hunyuan_i2v_bf16.safetensors` is the I2V model, and `skyreels_hunyuan_t2v_bf16.safetensors` is the T2V model. The models other than bf16 are not tested (`fp8_e4m3fn` may work).

For T2V inference, add the following options to the inference command:

```bash
--guidance_scale 6.0 --embedded_cfg_scale 1.0 --negative_prompt "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion" --split_uncond
```

SkyReels V1 seems to require a classfier free guidance (negative prompt).`--guidance_scale` is a guidance scale for the negative prompt. The recommended value is 6.0 from the official repository. The default is 1.0, it means no classifier free guidance.

`--embedded_cfg_scale` is a scale of the embedded guidance. The recommended value is 1.0 from the official repository (it may mean no embedded guidance).

`--negative_prompt` is a negative prompt for the classifier free guidance. The above sample is from the official repository. If you don't specify this, and specify `--guidance_scale` other than 1.0, an empty string will be used as the negative prompt.

`--split_uncond` is a flag to split the model call into unconditional and conditional parts. This reduces VRAM usage but may slow down inference. If `--split_attn` is specified, `--split_uncond` is automatically set.

You can also perform image2video inference with SkyReels V1 I2V model. Specify the image file path with `--image_path`. The image will be resized to the given `--video_size`.

```bash
--image_path path/to/image.jpg
``` 

<details>
<summary>日本語</summary>

SkyReels V1のT2VとI2Vモデルがサポートされています（推論のみ）。

モデルは[こちら](https://huggingface.co/Kijai/SkyReels-V1-Hunyuan_comfy)からダウンロードできます。モデルを提供してくださったKijai氏に感謝します。`skyreels_hunyuan_i2v_bf16.safetensors`がI2Vモデル、`skyreels_hunyuan_t2v_bf16.safetensors`がT2Vモデルです。`bf16`以外の形式は未検証です（`fp8_e4m3fn`は動作するかもしれません）。

T2V推論を行う場合、以下のオプションを推論コマンドに追加してください：

```bash
--guidance_scale 6.0 --embedded_cfg_scale 1.0 --negative_prompt "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion" --split_uncond
```

SkyReels V1はclassifier free guidance（ネガティブプロンプト）を必要とするようです。`--guidance_scale`はネガティブプロンプトのガイダンススケールです。公式リポジトリの推奨値は6.0です。デフォルトは1.0で、この場合はclassifier free guidanceは使用されません（ネガティブプロンプトは無視されます）。

`--embedded_cfg_scale`は埋め込みガイダンスのスケールです。公式リポジトリの推奨値は1.0です（埋め込みガイダンスなしを意味すると思われます）。

`--negative_prompt`はいわゆるネガティブプロンプトです。上記のサンプルは公式リポジトリのものです。`--guidance_scale`を指定し、`--negative_prompt`を指定しなかった場合は、空文字列が使用されます。

`--split_uncond`を指定すると、モデル呼び出しをuncondとcond（ネガティブプロンプトとプロンプト）に分割します。VRAM使用量が減りますが、推論速度は低下する可能性があります。`--split_attn`が指定されている場合、`--split_uncond`は自動的に有効になります。

</details>

### Convert LoRA to another format / LoRAの形式の変換

You can convert LoRA to a format (presumed to be Diffusion-pipe) compatible with another inference environment (Diffusers, ComfyUI etc.) using the following command:

```bash
python src/musubi_tuner/convert_lora.py --input path/to/musubi_lora.safetensors --output path/to/another_format.safetensors --target other
```

or for uv:

```bash
uv run --extra cu124 src/musubi_tuner/convert_lora.py --input path/to/musubi_lora.safetensors --output path/to/another_format.safetensors --target other
```

Specify the input and output file paths with `--input` and `--output`, respectively.

Specify `other` for `--target`. Use `default` to convert from another format to the format of this repository.

<details>
<summary>日本語</summary>

他の推論環境（DiffusersやComfyUI）で使用可能な形式（Diffusion-pipe または Diffusers と思われる）への変換は以下のコマンドで行えます。

```bash
python src/musubi_tuner/convert_lora.py --input path/to/musubi_lora.safetensors --output path/to/another_format.safetensors --target other
```

`--input`と`--output`はそれぞれ入力と出力のファイルパスを指定してください。

`--target`には`other`を指定してください。`default`を指定すると、他の形式から当リポジトリの形式に変換できます。

</details>
