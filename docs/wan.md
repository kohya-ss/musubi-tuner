> 📝 Click on the language section to expand / 言語をクリックして展開

# Wan 2.1/2.2

## Overview / 概要

This is an unofficial training and inference script for [Wan2.1](https://github.com/Wan-Video/Wan2.1) and [Wan2.2](https://github.com/Wan-Video/Wan2.2). The features are as follows.

- fp8 support and memory reduction by block swap: Inference of a 720x1280x81frames videos with 24GB VRAM, training with 720x1280 images with 24GB VRAM
- Inference without installing Flash attention (using PyTorch's scaled dot product attention)
- Supports xformers (training and inference) and Sage attention (inference only)
- Support for Wan2.2 model architecture, only for 14B models

This feature is experimental.

<details>
<summary>日本語</summary>

[Wan2.1](https://github.com/Wan-Video/Wan2.1) および [Wan2.2](https://github.com/Wan-Video/Wan2.2) の非公式の学習および推論スクリプトです。

以下の特徴があります。

- fp8対応およびblock swapによる省メモリ化：720x1280x81framesの動画を24GB VRAMで推論可能、720x1280の画像での学習が24GB VRAMで可能
- Flash attentionのインストールなしでの実行（PyTorchのscaled dot product attentionを使用）
- xformers（学習と推論）およびSage attention（推論のみ）対応
- Wan2.2モデルアーキテクチャのサポート（14Bモデルのみ）

この機能は実験的なものです。

</details>

## Download the model / モデルのダウンロード

### Wan2.1

Download the T5 `models_t5_umt5-xxl-enc-bf16.pth` and CLIP `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` from the following page: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main

Download the VAE from the above page `Wan2.1_VAE.pth` or download `split_files/vae/wan_2.1_vae.safetensors` from the following page: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae

Download the DiT weights from the following page: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models

Wan2.1 Fun Control model weights can be downloaded from [here](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-Control). Navigate to each weight page and download. The Fun Control model seems to support not only T2V but also I2V tasks.

Please select the appropriate weights according to T2V, I2V, resolution, model size, etc. 

`fp16` and `bf16` models can be used, and `fp8_e4m3fn` models can be used if `--fp8` (or `--fp8_base`) is specified without specifying `--fp8_scaled`. **Please note that `fp8_scaled` models are not supported even with `--fp8_scaled`.**

(Thanks to Comfy-Org for providing the repackaged weights.)

### Wan2.2

T5 is same as Wan2.1. CLIP is not required for Wan2.2.

VAE is also same as Wan2.1. Please use `Wan2.1_VAE.pth` from the above page. `Wan2.2_VAE.pth` is for 5B model, not compatible with 14B model.

Download the DiT weights from the following page: https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models

The Wan2.2 model consists of two DiT models, one for high noise and one for low noise. Please download both.

`fp16` models can be used. **Please note that `fp8_scaled` models are not supported even with `--fp8_scaled`.**

### Model support matrix / モデルサポートマトリックス

* columns: training dtype (行：学習時のデータ型)
* rows: model dtype (列：モデルのデータ型)

| model \ training |bf16|fp16|--fp8_base|--fp8base & --fp8_scaled|
|---|---|---|---|---|
|bf16|✓|--|✓|✓|
|fp16|--|✓|✓|✓|
|fp8_e4m3fn|--|--|✓|--|
|fp8_scaled|--|--|--|--|

<details>
<summary>日本語</summary>

### Wan2.1

T5 `models_t5_umt5-xxl-enc-bf16.pth` およびCLIP `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` を、次のページからダウンロードしてください：https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main

VAEは上のページから `Wan2.1_VAE.pth` をダウンロードするか、次のページから `split_files/vae/wan_2.1_vae.safetensors` をダウンロードしてください：https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae

DiTの重みを次のページからダウンロードしてください：https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models

Wan2.1 Fun Controlモデルの重みは、[こちら](https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-Control)から、それぞれの重みのページに遷移し、ダウンロードしてください。Fun ControlモデルはT2VだけでなくI2Vタスクにも対応しているようです。

T2VやI2V、解像度、モデルサイズなどにより適切な重みを選択してください。

`fp16` および `bf16` モデルを使用できます。また、`--fp8` （または`--fp8_base`）を指定し`--fp8_scaled`を指定をしないときには `fp8_e4m3fn` モデルを使用できます。**`fp8_scaled` モデルはいずれの場合もサポートされていませんのでご注意ください。**

（repackaged版の重みを提供してくださっているComfy-Orgに感謝いたします。）

### Wan2.2

T5はWan2.1と同じです。Wan2.2ではCLIPは不要です。

VAEは上のページから `Wan2.1_VAE.pth` をダウンロードしてください。`Wan2.2_VAE.pth` は5Bモデル用で、14Bモデルには対応していません。

DiTの重みを次のページからダウンロードしてください：https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/tree/main/split_files/diffusion_models

Wan2.2モデルは高ノイズ用と低ノイズ用の2つのDiTモデルで構成されています。両方をダウンロードしてください。

`fp16` モデルを使用できます。**`fp8_scaled` モデルはサポートされませんのでご注意ください。**

</details>

## Pre-caching / 事前キャッシュ

Pre-caching is almost the same as in HunyuanVideo, but some options may differ. See [HunyuanVideo documentation](./hunyuan_video.md#pre-caching--事前キャッシング) and `--help` for details. 

### Latent Pre-caching

Create the cache using the following command:

```bash
python src/musubi_tuner/wan_cache_latents.py --dataset_config path/to/toml --vae path/to/wan_vae.safetensors
```

**If you train I2V models, add `--i2v` option to the above command.** For Wan2.1, add `--clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` to specify the CLIP model. If not specified, the training will raise an error. For Wan2.2, CLIP model is not required.

If you're running low on VRAM, specify `--vae_cache_cpu` to use the CPU for the VAE internal cache, which will reduce VRAM usage somewhat.

The control video settings are required for training the Fun-Control model. Please refer to [Dataset Settings](./dataset_config.md#sample-for-video-dataset-with-control-images) for details.

<details>
<summary>日本語</summary>

事前キャッシングはHunyuanVideoとほぼ同じです。オプションが異なる場合がありますので、詳細は[HunyuanVideoのドキュメント](./hunyuan_video.md#pre-caching--事前キャッシング)および`--help`を参照してください。

latentの事前キャッシングは上のコマンド例を使用してキャッシュを作成してください。

**I2Vモデルを学習する場合は、`--i2v` オプションを上のコマンドに追加してください。**Wan2.1の場合は、`--clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` を追加してCLIPモデルを指定してください。指定しないと学習時にエラーが発生します。Wan2.2ではCLIPモデルは不要です。

VRAMが不足している場合は、`--vae_cache_cpu` を指定するとVAEの内部キャッシュにCPUを使うことで、使用VRAMを多少削減できます。

Fun-Controlモデルを学習する場合は、制御用動画の設定が必要です。[データセット設定](./dataset_config.md#sample-for-video-dataset-with-control-images)を参照してください。

</details>

### Text Encoder Output Pre-caching

Text encoder output pre-caching is also almost the same as in HunyuanVideo. Create the cache using the following command:

```bash
python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config path/to/toml  --t5 path/to/models_t5_umt5-xxl-enc-bf16.pth --batch_size 16 
```

Adjust `--batch_size` according to your available VRAM.

For systems with limited VRAM (less than ~16GB), use `--fp8_t5` to run the T5 in fp8 mode.

<details>
<summary>日本語</summary>

テキストエンコーダ出力の事前キャッシングもHunyuanVideoとほぼ同じです。上のコマンド例を使用してキャッシュを作成してください。

使用可能なVRAMに合わせて `--batch_size` を調整してください。

VRAMが限られているシステム（約16GB未満）の場合は、T5をfp8モードで実行するために `--fp8_t5` を使用してください。

</details>

## Training / 学習

### Training

Start training using the following command (input as a single line):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/wan_train_network.py \
    --task t2v-1.3B \
    --dit path/to/wan2.1_xxx_bf16.safetensors \
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_wan --network_dim 32 \
    --timestep_sampling shift --discrete_flow_shift 3.0 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```
The above is an example. The appropriate values for `timestep_sampling` and `discrete_flow_shift` need to be determined by experimentation.

For additional options, use `python src/musubi_tuner/wan_train_network.py --help` (note that many options are unverified).

`--task` is one of `t2v-1.3B`, `t2v-14B`, `i2v-14B`, `t2i-14B` (for Wan2.1 official models), `t2v-1.3B-FC`, `t2v-14B-FC`, and `i2v-14B-FC` (for Wan2.1 Fun Control model), `t2v-A14B`, `i2v-A14B` (for Wan2.2 14B models). Specify the DiT weights for the task with `--dit`.

You can limit the range of timesteps for training with `--min_timestep` and `--max_timestep`. The values are specified in the range of 0 to 1000 (not 0.0 to 1.0). See [here](./advanced_config.md#specify-time-step-range-for-training--学習時のタイムステップ範囲の指定) for details. 

For Wan2.2 models, if you want to train with either the high-noise model or the low-noise model, specify the model with `--dit` as in Wan2.1. In this case, it is recommended to specify the range of timesteps described in the table below, and `--preserve_distribution_shape` to maintain the distribution shape.

If you want to train LoRA for both models simultaneously, you need to specify the low-noise model with `--dit` and the high-noise model with `--dit_high_noise`. The two models are switched at the timestep specified by `--timestep_boundary`. The default value is 0.9 for I2V and 0.875 for T2V. `--timestep_boundary` can be specified in the range of 0.0 to 1.0, or in the range of 0 to 1000.

When training Wan2.2 high and low models, you can use `--offload_inactive_dit` to offload the inactive DiT model to the CPU, which can save VRAM (only works when `--blocks_to_swap` is not specified). Please note that in Windows environment, this offloading uses shared VRAM. Even with fp8/fp8_scaled, about 42GB of shared VRAM is required for the two models combined, which means that about 96GB or more of main RAM is required. If you have less main RAM, using `--blocks_to_swap` will use less main RAM.

`--gradient_checkpointing` and `--gradient_checkpointing_cpu_offload` are available for memory savings. See [HunyuanVideo documentation](./hunyuan_video.md#memory-optimization) for details.

For Wan2.2 models, `--discrete_flow_shift` may need to be adjusted based on I2V and T2V. According to the official implementation, the shift values in inference are 12.0 for T2V and 5.0 for I2V. The shift values during training do not necessarily have to match those during inference, but they may serve as a useful reference.

`--force_v2_1_time_embedding` uses the same shape of time embedding as Wan2.1. This can reduce VRAM usage during inference and training (the larger the resolution and number of frames, the greater the reduction). Although this is different from the official implementation of Wan2.2, it seems that there is no effect on inference or training within the range that has been confirmed.

Don't forget to specify `--network_module networks.lora_wan`.

Other options are mostly the same as `hv_train_network.py`. See [HunyuanVideo documentation](./hunyuan_video.md#training--学習) and `--help` for details.

The trained LoRA weights are seemed to be compatible with ComfyUI (may depend on the nodes used).

#### Recommended Min/Max Timestep Settings for Wan2.2

| Model | Min Timestep | Max Timestep |
|-------|--------------|--------------|
| I2V low noise  | 0            | 900         |
| I2V high noise | 900          | 1000         |
| T2V low noise  | 0            | 875         |
| T2V high noise | 875          | 1000         |

<details>
<summary>日本語</summary>

サンプルは英語版を参照してください。

`timestep_sampling`や`discrete_flow_shift`は一例です。どのような値が適切かは実験が必要です。

その他のオプションについては `python src/musubi_tuner/wan_train_network.py --help` を使用してください（多くのオプションは未検証です）。

`--task` には `t2v-1.3B`, `t2v-14B`, `i2v-14B`, `t2i-14B` （これらはWan2.1公式モデル）、`t2v-1.3B-FC`, `t2v-14B-FC`, `i2v-14B-FC`（Wan2.1-Fun Controlモデル）、`t2v-A14B`, `i2v-A14B`（Wan2.2 14Bモデル）を指定します。`--dit`に、taskに応じたDiTの重みを指定してください。

`--min_timestep`と`--max_timestep`で学習するタイムステップの範囲を限定できます。値は0から1000の範囲で指定します。詳細は[こちら](./advanced_config.md#specify-time-step-range-for-training--学習時のタイムステップ範囲の指定)を参照してください。

Wan2.2モデルの場合、高ノイズ用モデルまたは低ノイズ用モデルのどちらかで学習する場合は、Wan2.1の場合と同様に、`--dit`にそのモデルを指定してください。またこの場合、英語版サンプル内の表に示すようにタイムステップの範囲を指定し、`--preserve_distribution_shape` を指定して分布形状を維持することをお勧めします。

両方のモデルへのLoRAを学習する場合は、`--dit`に低ノイズ用モデルを、`--dit_high_noise`に高ノイズ用モデルを指定します。2つのモデルは`--timestep_boundary`で指定されたタイムステップで切り替わります。デフォルトはI2Vの場合は0.9、T2Vの場合は0.875です。`--timestep_boundary`は0.0から1.0の範囲の値、または0から1000の範囲の値で指定できます。

またWan2.2モデルで両方のモデルを学習するとき、`--offload_inactive_dit`を使用すると、使用していないDiTモデルをCPUにオフロードすることができ、VRAMを節約できます（`--blocks_to_swap`未指定時のみ有効）。なお、Windows環境の場合、このオフロードには共有VRAMが使用されます。fp8/fp8_scaledの場合でも2つのモデル合計で約42GBの共有VRAMが必要となり、つまりメインRAMが96GB程度以上必要になりますのでご注意ください。メインRAMが少ない場合、`--blocks_to_swap`を使用する方がメインRAMの使用量は少なくなります。

Wan2.2の場合、I2VとT2Vで`--discrete_flow_shift`を調整する必要があるかもしれません。公式実装によると、推論時のシフト値はT2Vで12.0、I2Vで5.0です。学習時のシフト値は推論時度必ずしも合わせる必要はありませんが、参考になるかもしれません。

`--force_v2_1_time_embedding` を指定すると、Wan2.1と同じ形状の時間埋め込みを使用します。これにより推論中、学習中のVRAM使用量を削減できます（解像度やフレーム数が大きいほど削減量も大きくなります）。Wan2.2の公式実装とは異なりますが、確認した範囲では推論、学習共に影響はないようです。

`--network_module` に `networks.lora_wan` を指定することを忘れないでください。

その他のオプションは、ほぼ`hv_train_network.py`と同様です。[HunyuanVideoのドキュメント](./hunyuan_video.md#training--学習)および`--help`を参照してください。

学習後のLoRAの重みはそのままComfyUIで使用できるようです（用いるノードにもよります）。

</details>

### Command line options for training with sampling / サンプル画像生成に関連する学習時のコマンドラインオプション

Example of command line options for training with sampling / 記述例:  

```bash
--vae path/to/wan_vae.safetensors \
--t5 path/to/models_t5_umt5-xxl-enc-bf16.pth \
--sample_prompts /path/to/prompt_file.txt \
--sample_every_n_epochs 1 --sample_every_n_steps 1000 --sample_at_first
```
Each option is the same as when generating images or as HunyuanVideo. Please refer to [here](/docs/sampling_during_training.md) for details.

If you train I2V models for Wan2.1, add `--clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` to specify the CLIP model. For Wan2.2, CLIP model is not required.

You can specify the initial image, the negative prompt and the control video (for Wan2.1-Fun-Control) in the prompt file. Please refer to [here](/docs/sampling_during_training.md#prompt-file--プロンプトファイル).

<details>
<summary>日本語</summary>

各オプションは推論時、およびHunyuanVideoの場合と同様です。[こちら](/docs/sampling_during_training.md)を参照してください。

Wan2.1のI2Vモデルを学習する場合は、`--clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` を追加してCLIPモデルを指定してください。Wan2.2ではCLIPモデルは不要です。

プロンプトファイルで、初期画像やネガティブプロンプト、制御動画（Wan2.1-Fun-Control用）等を指定できます。[こちら](/docs/sampling_during_training.md#prompt-file--プロンプトファイル)を参照してください。

</details>


## Inference / 推論

### Inference Options Comparison / 推論オプション比較

#### Speed Comparison (Faster → Slower) / 速度比較（速い→遅い）
*Note: Results may vary depending on GPU type*

fp8_fast > bf16/fp16 (no block swap) > fp8 > fp8_scaled > bf16/fp16 (block swap)

#### Quality Comparison (Higher → Lower) / 品質比較（高→低）

bf16/fp16 > fp8_scaled > fp8 >> fp8_fast

### T2V Inference / T2V推論

The following is an example of T2V inference (input as a single line):

```bash
python src/musubi_tuner/wan_generate_video.py --fp8 --task t2v-1.3B --video_size  832 480 --video_length 81 --infer_steps 20 \
--prompt "prompt for the video" --save_path path/to/save.mp4 --output_type both \
--dit path/to/wan2.1_t2v_1.3B_bf16_etc.safetensors --vae path/to/wan_2.1_vae.safetensors \
--t5 path/to/models_t5_umt5-xxl-enc-bf16.pth \
--attn_mode torch
```

`--task` is one of `t2v-1.3B`, `t2v-14B`, `i2v-14B`, `t2i-14B` (these are Wan2.1 official models), `t2v-1.3B-FC`, `t2v-14B-FC` and `i2v-14B-FC` (for Wan2.1-Fun Control model), `t2v-A14B`, `i2v-A14B` (for Wan2.2 14B models).

For Wan2.2 models, you can specify the low-noise model with `--dit` and the high-noise model with `--dit_high_noise`. The two models are switched at the timestep specified by `--timestep_boundary`. The default is described above. If you omit the high-noise model, the low-noise model will be used for all timesteps.

When inferring Wan2 .2 high and low models, you can use `--offload_inactive_dit` to offload the inactive DiT model to the CPU, or `--lazy_loading` to enable lazy loading for DiT models, which can save VRAM. `--offload_inactive_dit` only works when `--blocks_to_swap` is not specified, so use `--lazy_loading` instead. Without these options, both models will remain on the GPU, which may use more VRAM.

`--attn_mode` is `torch`, `sdpa` (same as `torch`), `xformers`, `sageattn`,`flash2`, `flash` (same as `flash2`) or `flash3`. `torch` is the default. Other options require the corresponding library to be installed. `flash3` (Flash attention 3) is not tested.

Specifying `--fp8` runs DiT in fp8 mode. fp8 can significantly reduce memory consumption but may impact output quality.

`--fp8_scaled` can be specified in addition to `--fp8` to run the model in fp8 weights optimization. This increases memory consumption and speed slightly but improves output quality. See [here](advanced_config.md#fp8-weight-optimization-for-models--モデルの重みのfp8への最適化) for details.

`--fp8_fast` option is also available for faster inference on RTX 40x0 GPUs. This option requires `--fp8_scaled` option. **This option seems to degrade the output quality.**

`--fp8_t5` can be used to specify the T5 model in fp8 format. This option reduces memory usage for the T5 model.  

`--negative_prompt` can be used to specify a negative prompt. If omitted, the default negative prompt is used.

`--flow_shift` can be used to specify the flow shift (default 3.0 for I2V with 480p, 5.0 for others).

`--guidance_scale` can be used to specify the guidance scale for classifier free guidance (default 5.0). For Wan2.2, `--guidance_scale_high_noise` also can be specified to set a different scale for the high-noise model.

`--blocks_to_swap` is the number of blocks to swap during inference. The default value is None (no block swap). The maximum value is 39 for 14B model and 29 for 1.3B model.

`--force_v2_1_time_embedding` uses the same shape of time embedding as Wan2.1 for Wan2.2. See the training section for details.

`--vae_cache_cpu` enables VAE cache in main memory. This reduces VRAM usage slightly but processing is slower.

`--compile` enables torch.compile. See [here](/README.md#inference) for details.

`--trim_tail_frames` can be used to trim the tail frames when saving. The default is 0.

`--cfg_skip_mode` specifies the mode for skipping CFG in different steps. The default is `none` (all steps).`--cfg_apply_ratio` specifies the ratio of steps where CFG is applied. See below for details.

`--include_patterns` and `--exclude_patterns` can be used to specify which LoRA modules to apply or exclude during training. If not specified, all modules are applied by default. These options accept regular expressions. 

`--include_patterns` specifies the modules to be applied, and `--exclude_patterns` specifies the modules to be excluded. The regular expression is matched against the LoRA key name, and include takes precedence.

The key name to be searched is in sd-scripts format (`lora_unet_<module_name with dot replaced by _>`). For example, `lora_unet_blocks_9_cross_attn_k`.

For example, if you specify `--exclude_patterns "blocks_[23]\d_"` , it will exclude modules containing `blocks_20` to `blocks_39`. If you specify `--include_patterns "cross_attn" --exclude_patterns "blocks_(0|1|2|3|4)_"`, it will apply LoRA to modules containing `cross_attn` and not containing `blocks_0` to `blocks_4`.

If you specify multiple LoRA weights, please specify them with multiple arguments. For example: `--include_patterns "cross_attn" ".*" --exclude_patterns "dummy_do_not_exclude" "blocks_(0|1|2|3|4)"`. `".*"` is a regex that matches everything. `dummy_do_not_exclude` is a dummy regex that does not match anything.

`--cpu_noise` generates initial noise on the CPU. This may result in the same results as ComfyUI with the same seed (depending on other settings).

If you are using the Fun Control model, specify the control video with `--control_path`. You can specify a video file or a folder containing multiple image files. The number of frames in the video file (or the number of images) should be at least the number specified in `--video_length` (plus 1 frame if you specify `--end_image_path`).

Please try to match the aspect ratio of the control video with the aspect ratio specified in `--video_size` (there may be some deviation from the initial image of I2V due to the use of bucketing processing).

Other options are same as `hv_generate_video.py` (some options are not supported, please check the help).

<details>
<summary>日本語</summary>

`--task` には `t2v-1.3B`, `t2v-14B`, `i2v-14B`, `t2i-14B` （これらはWan2.1公式モデル）、`t2v-1.3B-FC`, `t2v-14B-FC`, `i2v-14B-FC`（Wan2.1-Fun Controlモデル）、`t2v-A14B`, `i2v-A14B`（Wan2.2 14Bモデル）を指定します。

Wan2.2モデルの場合、`--dit`に低ノイズ用モデルを、`--dit_high_noise`に高ノイズ用モデルを指定します。2つのモデルは`--timestep_boundary`で指定されたタイムステップで切り替わります。高ノイズ用モデルを省略した場合は、低ノイズ用モデルが全てのタイムステップで使用されます。

またWan2.2モデルで両方のモデルを用いて推論するとき、`--offload_inactive_dit`を使用すると、使用していないDiTモデルをCPUにオフロードすることができます。また`--lazy_loading`を使用すると、DiTモデルの遅延読み込みを有効します。これらのオプションによりVRAMを節約できます。`--offload_inactive_dit`は`--blocks_to_swap`が指定されていない場合にのみ利用できます。`--block_to_swap`を使うときには`--lazy_loading`を使用してください。これらのオプションを指定しないと両方のモデルがGPUに置かれますので、VRAMを多く使用します。

`--attn_mode` には `torch`, `sdpa`（`torch`と同じ）、`xformers`, `sageattn`, `flash2`, `flash`（`flash2`と同じ）, `flash3` のいずれかを指定します。デフォルトは `torch` です。その他のオプションを使用する場合は、対応するライブラリをインストールする必要があります。`flash3`（Flash attention 3）は未テストです。

`--fp8` を指定するとDiTモデルをfp8形式で実行します。fp8はメモリ消費を大幅に削減できますが、出力品質に影響を与える可能性があります。
    
`--fp8_scaled` を `--fp8` と併用すると、fp8への重み量子化を行います。メモリ消費と速度はわずかに悪化しますが、出力品質が向上します。詳しくは[こちら](advanced_config.md#fp8-weight-optimization-for-models--モデルの重みのfp8への最適化)を参照してください。

`--fp8_fast` オプションはRTX 40x0 GPUでの高速推論に使用されるオプションです。このオプションは `--fp8_scaled` オプションが必要です。**出力品質が劣化するようです。**

`--fp8_t5` を指定するとT5モデルをfp8形式で実行します。T5モデル呼び出し時のメモリ使用量を削減します。

`--negative_prompt` でネガティブプロンプトを指定できます。省略した場合はデフォルトのネガティブプロンプトが使用されます。

`--flow_shift` でflow shiftを指定できます（480pのI2Vの場合はデフォルト3.0、それ以外は5.0）。

`--guidance_scale` でclassifier free guianceのガイダンススケールを指定できます（デフォルト5.0）。Wan2.2の場合は、`--guidance_scale_high_noise` で高ノイズ用モデルのガイダンススケールを別に指定できます。

`--blocks_to_swap` は推論時のblock swapの数です。デフォルト値はNone（block swapなし）です。最大値は14Bモデルの場合39、1.3Bモデルの場合29です。

`--force_v2_1_time_embedding` はWan2.2の場合に有効で、Wan2.1と同じ形状の時間埋め込みを使用します。詳細は学習セクションを参照してください。

`--vae_cache_cpu` を有効にすると、VAEのキャッシュをメインメモリに保持します。VRAM使用量が多少減りますが、処理は遅くなります。

`--compile`でtorch.compileを有効にします。詳細については[こちら](/README.md#inference)を参照してください。

`--trim_tail_frames` で保存時に末尾のフレームをトリミングできます。デフォルトは0です。

`--cfg_skip_mode` は異なるステップでCFGをスキップするモードを指定します。デフォルトは `none`（全ステップ）。`--cfg_apply_ratio` はCFGが適用されるステップの割合を指定します。詳細は後述します。

LoRAのどのモジュールを適用するかを、`--include_patterns`と`--exclude_patterns`で指定できます（未指定時・デフォルトは全モジュール適用されます
）。これらのオプションには、正規表現を指定します。`--include_patterns`は適用するモジュール、`--exclude_patterns`は適用しないモジュールを指定します。正規表現がLoRAのキー名に含まれるかどうかで判断され、includeが優先されます。

検索対象となるキー名は sd-scripts 形式（`lora_unet_<モジュール名のドットを_に置換したもの>`）です。例：`lora_unet_blocks_9_cross_attn_k`

たとえば `--exclude_patterns "blocks_[23]\d_"`のみを指定すると、`blocks_20`から`blocks_39`を含むモジュールが除外されます。`--include_patterns "cross_attn" --exclude_patterns "blocks_(0|1|2|3|4)_"`のようにincludeとexcludeを指定すると、`cross_attn`を含むモジュールで、かつ`blocks_0`から`blocks_4`を含まないモジュールにLoRAが適用されます。

複数のLoRAの重みを指定する場合は、複数個の引数で指定してください。例：`--include_patterns "cross_attn" ".*" --exclude_patterns "dummy_do_not_exclude" "blocks_(0|1|2|3|4)"` `".*"`は全てにマッチする正規表現です。`dummy_do_not_exclude`は何にもマッチしないダミーの正規表現です。

`--cpu_noise`を指定すると初期ノイズをCPUで生成します。これにより同一seed時の結果がComfyUIと同じになる可能性があります（他の設定にもよります）。

Fun Controlモデルを使用する場合は、`--control_path`で制御用の映像を指定します。動画ファイル、または複数枚の画像ファイルを含んだフォルダを指定できます。動画ファイルのフレーム数（または画像の枚数）は、`--video_length`で指定したフレーム数以上にしてください（後述の`--end_image_path`を指定した場合は、さらに+1フレーム）。

制御用の映像のアスペクト比は、`--video_size`で指定したアスペクト比とできるかぎり合わせてください（bucketingの処理を流用しているためI2Vの初期画像とズレる場合があります）。

その他のオプションは `hv_generate_video.py` と同じです（一部のオプションはサポートされていないため、ヘルプを確認してください）。

</details>

#### CFG Skip Mode / CFGスキップモード

 These options allow you to balance generation speed against prompt accuracy. More skipped steps results in faster generation with potential quality degradation.

Setting `--cfg_apply_ratio` to 0.5 speeds up the denoising loop by up to 25%.

`--cfg_skip_mode` specified one of the following modes:

- `early`: Skips CFG in early steps for faster generation, applying guidance mainly in later refinement steps
- `late`: Skips CFG in later steps, applying guidance during initial structure formation
- `middle`: Skips CFG in middle steps, applying guidance in both early and later steps
- `early_late`: Skips CFG in both early and late steps, applying only in middle steps
- `alternate`: Applies CFG in alternate steps based on the specified ratio
- `none`: Applies CFG at all steps (default)

`--cfg_apply_ratio` specifies a value from 0.0 to 1.0 controlling the proportion of steps where CFG is applied. For example, setting 0.5 means CFG will be applied in only 50% of the steps.

If num_steps is 10, the following table shows the steps where CFG is applied based on the `--cfg_skip_mode` option (A means CFG is applied, S means it is skipped, `--cfg_apply_ratio` is 0.6):

| skip mode | CFG apply pattern |
|---|---|
| early | SSSSAAAAAA |
| late | AAAAAASSSS |
| middle | AAASSSSAAA |
| early_late | SSAAAAAASS |
| alternate | SASASAASAS |

The appropriate settings are unknown, but you may want to try `late` or `early_late` mode with a ratio of around 0.3 to 0.5.
<details>
<summary>日本語</summary>
これらのオプションは、生成速度とプロンプトの精度のバランスを取ることができます。スキップされるステップが多いほど、生成速度が速くなりますが、品質が低下する可能性があります。

ratioに0.5を指定することで、デノイジングのループが最大25%程度、高速化されます。

`--cfg_skip_mode` は次のモードのいずれかを指定します：

- `early`：初期のステップでCFGをスキップして、主に終盤の精細化のステップで適用します
- `late`：終盤のステップでCFGをスキップし、初期の構造が決まる段階で適用します
- `middle`：中間のステップでCFGをスキップし、初期と終盤のステップの両方で適用します
- `early_late`：初期と終盤のステップの両方でCFGをスキップし、中間のステップのみ適用します
- `alternate`：指定された割合に基づいてCFGを適用します

`--cfg_apply_ratio` は、CFGが適用されるステップの割合を0.0から1.0の値で指定します。たとえば、0.5に設定すると、CFGはステップの50%のみで適用されます。

具体的なパターンは上のテーブルを参照してください。

適切な設定は不明ですが、モードは`late`または`early_late`、ratioは0.3~0.5程度から試してみると良いかもしれません。
</details>

#### Skip Layer Guidance

Skip Layer Guidance is a feature that uses the output of a model with some blocks skipped as the unconditional output of classifier free guidance. It was originally proposed in [SD 3.5](https://github.com/comfyanonymous/ComfyUI/pull/5404) and first applied in Wan2GP in [this PR](https://github.com/deepbeepmeep/Wan2GP/pull/61). It may improve the quality of generated videos.

The implementation of SD 3.5 is [here](https://github.com/Stability-AI/sd3.5/blob/main/sd3_impls.py), and the implementation of Wan2GP (the PR mentioned above) has some different specifications. This inference script allows you to choose between the two methods.

*The SD3.5 method applies slg output in addition to cond and uncond (slows down the speed). The Wan2GP method uses only cond and slg output.*

The following arguments are available:

- `--slg_mode`: Specifies the SLG mode. `original` for SD 3.5 method, `uncond` for Wan2GP method. Default is None (no SLG).
- `--slg_layers`: Specifies the indices of the blocks (layers) to skip in SLG, separated by commas. Example: `--slg_layers 4,5,6`. Default is empty (no skip). If this option is not specified, `--slg_mode` is ignored.
- `--slg_scale`: Specifies the scale of SLG when `original`. Default is 3.0.
- `--slg_start`: Specifies the start step of SLG application in inference steps from 0.0 to 1.0. Default is 0.0 (applied from the beginning).
- `--slg_end`: Specifies the end step of SLG application in inference steps from 0.0 to 1.0. Default is 0.3 (applied up to 30% from the beginning).

Appropriate settings are unknown, but you may want to try `original` mode with a scale of around 3.0 and a start ratio of 0.0 and an end ratio of 0.5, with layers 4, 5, and 6 skipped.

<details>
<summary>日本語</summary>
Skip Layer Guidanceは、一部のblockをスキップしたモデル出力をclassifier free guidanceのunconditional出力に使用する機能です。元々は[SD 3.5](https://github.com/comfyanonymous/ComfyUI/pull/5404)で提案されたもので、Wan2.1には[Wan2GPのこちらのPR](https://github.com/deepbeepmeep/Wan2GP/pull/61)で初めて適用されました。生成動画の品質が向上する可能性があります。

SD 3.5の実装は[こちら](https://github.com/Stability-AI/sd3.5/blob/main/sd3_impls.py)で、Wan2GPの実装（前述のPR）は一部仕様が異なります。この推論スクリプトでは両者の方式を選択できるようになっています。

※SD3.5方式はcondとuncondに加えてslg outputを適用します（速度が低下します）。Wan2GP方式はcondとslg outputのみを使用します。

以下の引数があります。

- `--slg_mode`：SLGのモードを指定します。`original`でSD 3.5の方式、`uncond`でWan2GPの方式です。デフォルトはNoneで、SLGを使用しません。
- `--slg_layers`：SLGでスキップするblock (layer)のインデクスをカンマ区切りで指定します。例：`--slg_layers 4,5,6`。デフォルトは空（スキップしない）です。このオプションを指定しないと`--slg_mode`は無視されます。
- `--slg_scale`：`original`のときのSLGのスケールを指定します。デフォルトは3.0です。
- `--slg_start`：推論ステップのSLG適用開始ステップを0.0から1.0の割合で指定します。デフォルトは0.0です（最初から適用）。
- `--slg_end`：推論ステップのSLG適用終了ステップを0.0から1.0の割合で指定します。デフォルトは0.3です（最初から30%まで適用）。

適切な設定は不明ですが、`original`モードでスケールを3.0程度、開始割合を0.0、終了割合を0.5程度に設定し、4, 5, 6のlayerをスキップする設定から始めると良いかもしれません。
</details>

### I2V Inference / I2V推論

The following is an example of I2V inference (input as a single line):

```bash
python src/musubi_tuner/wan_generate_video.py --fp8 --task i2v-14B --video_size 832 480 --video_length 81 --infer_steps 20 \
--prompt "prompt for the video" --save_path path/to/save.mp4 --output_type both \
--dit path/to/wan2.1_i2v_480p_14B_bf16_etc.safetensors --vae path/to/wan_2.1_vae.safetensors \
--t5 path/to/models_t5_umt5-xxl-enc-bf16.pth --clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
--attn_mode torch --image_path path/to/image.jpg
```

For Wan2.1, add `--clip` to specify the CLIP model. For Wan2.2, CLIP model is not required. `--image_path` is the path to the image to be used as the initial frame.

`--end_image_path` can be used to specify the end image. This option is experimental. When this option is specified, the saved video will be slightly longer than the specified number of frames and will have noise, so it is recommended to specify `--trim_tail_frames 3` to trim the tail frames.

You can also use the Fun Control model for I2V inference. Specify the control video with `--control_path`. 

Other options are same as T2V inference.

<details>
<summary>日本語</summary>
Wan2.1の場合は`--clip` を追加してCLIPモデルを指定します。Wan2.2ではCLIPモデルは不要です。`--image_path` は初期フレームとして使用する画像のパスです。

`--end_image_path` で終了画像を指定できます。このオプションは実験的なものです。このオプションを指定すると、保存される動画が指定フレーム数よりもやや多くなり、かつノイズが乗るため、`--trim_tail_frames 3` などを指定して末尾のフレームをトリミングすることをお勧めします。

I2V推論でもFun Controlモデルが使用できます。`--control_path` で制御用の映像を指定します。

その他のオプションはT2V推論と同じです。
</details>

### New Batch and Interactive Modes / 新しいバッチモードとインタラクティブモード

In addition to single video generation, Wan 2.1/2.2 now supports batch generation from file and interactive prompt input:

#### Batch Mode from File / ファイルからのバッチモード

Generate multiple videos from prompts stored in a text file:

```bash
python src/musubi_tuner/wan_generate_video.py --from_file prompts.txt --task t2v-14B \
--dit path/to/model.safetensors --vae path/to/vae.safetensors \
--t5 path/to/t5_model.pth --save_path output_directory
```

The prompts file format:
- One prompt per line
- Empty lines and lines starting with # are ignored (comments)
- Each line can include prompt-specific parameters using command-line style format:

```
A beautiful sunset over mountains --w 832 --h 480 --f 81 --d 42 --s 20
A busy city street at night --w 480 --h 832 --g 7.5 --n low quality, blurry
```

Supported inline parameters (if ommitted, default values from the command line are used):
- `--w`: Width
- `--h`: Height
- `--f`: Frame count
- `--d`: Seed
- `--s`: Inference steps
- `--g` or `--l`: Guidance scale
- `--fs`: Flow shift
- `--i`: Image path (for I2V)
- `--cn`: Control path (for Fun Control)
- `--n`: Negative prompt

In batch mode, models are loaded once and reused for all prompts, significantly improving overall generation time compared to multiple single runs.

#### Interactive Mode / インタラクティブモード

Interactive command-line interface for entering prompts:

```bash
python src/musubi_tuner/wan_generate_video.py --interactive --task t2v-14B \
--dit path/to/model.safetensors --vae path/to/vae.safetensors \
--t5 path/to/t5_model.pth --save_path output_directory
```

In interactive mode:
- Enter prompts directly at the command line
- Use the same inline parameter format as batch mode
- Use Ctrl+D (or Ctrl+Z on Windows) to exit
- Models remain loaded between generations for efficiency

<details>
<summary>日本語</summary>
単一動画の生成に加えて、Wan 2.1/2.2は現在、ファイルからのバッチ生成とインタラクティブなプロンプト入力をサポートしています。

#### ファイルからのバッチモード

テキストファイルに保存されたプロンプトから複数の動画を生成します：

```bash
python src/musubi_tuner/wan_generate_video.py --from_file prompts.txt --task t2v-14B \
--dit path/to/model.safetensors --vae path/to/vae.safetensors \
--t5 path/to/t5_model.pth --save_path output_directory
```

プロンプトファイルの形式：
- 1行に1つのプロンプト
- 空行や#で始まる行は無視されます（コメント）
- 各行にはコマンドライン形式でプロンプト固有のパラメータを含めることができます：

サポートされているインラインパラメータ（省略した場合、コマンドラインのデフォルト値が使用されます）
- `--w`: 幅
- `--h`: 高さ
- `--f`: フレーム数
- `--d`: シード
- `--s`: 推論ステップ
- `--g` または `--l`: ガイダンススケール
- `--fs`: フローシフト
- `--i`: 画像パス（I2V用）
- `--cn`: コントロールパス（Fun Control用）
- `--n`: ネガティブプロンプト

バッチモードでは、モデルは一度だけロードされ、すべてのプロンプトで再利用されるため、複数回の単一実行と比較して全体的な生成時間が大幅に改善されます。

#### インタラクティブモード

プロンプトを入力するためのインタラクティブなコマンドラインインターフェース：

```bash
python src/musubi_tuner/wan_generate_video.py --interactive --task t2v-14B \
--dit path/to/model.safetensors --vae path/to/vae.safetensors \
--t5 path/to/t5_model.pth --save_path output_directory
```

インタラクティブモードでは：
- コマンドラインで直接プロンプトを入力
- バッチモードと同じインラインパラメータ形式を使用
- 終了するには Ctrl+D (Windowsでは Ctrl+Z) を使用
- 効率のため、モデルは生成間で読み込まれたままになります

</details>
