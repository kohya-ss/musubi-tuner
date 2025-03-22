> 📝 Click on the language section to expand / 言語をクリックして展開

# Wan 2.1

## Overview / 概要

This is an unofficial training and inference script for [Wan2.1](https://github.com/Wan-Video/Wan2.1). The features are as follows.

- fp8 support and memory reduction by block swap: Inference of a 720x1280x81frames videos with 24GB VRAM, training with 720x1280 images with 24GB VRAM
- Inference without installing Flash attention (using PyTorch's scaled dot product attention)
- Supports xformers and Sage attention

This feature is experimental.

<details>
<summary>日本語</summary>
[Wan2.1](https://github.com/Wan-Video/Wan2.1) の非公式の学習および推論スクリプトです。

以下の特徴があります。

- fp8対応およびblock swapによる省メモリ化：720x1280x81framesの動画を24GB VRAMで推論可能、720x1280の画像での学習が24GB VRAMで可能
- Flash attentionのインストールなしでの実行（PyTorchのscaled dot product attentionを使用）
- xformersおよびSage attention対応

この機能は実験的なものです。
</details>

## Download the model / モデルのダウンロード

Download the T5 `models_t5_umt5-xxl-enc-bf16.pth` and CLIP `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` from the following page: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main

Download the VAE from the above page `Wan2.1_VAE.pth` or download `split_files/vae/wan_2.1_vae.safetensors` from the following page: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae

Download the DiT weights from the following page: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models

Please select the appropriate weights according to T2V, I2V, resolution, model size, etc. 

`fp16` and `bf16` models can be used, and `fp8_e4m3fn` models can be used if `--fp8` (or `--fp8_base`) is specified without specifying `--fp8_scaled`. **Please note that `fp8_scaled` models are not supported even with `--fp8_scaled`.**

(Thanks to Comfy-Org for providing the repackaged weights.)

### Model support matrix / モデルサポートマトリックス

* columns: training dtype (行：学習時のデータ型)
* rows: model dtype (列：モデルのデータ型)

| model \ training |bf16|fp16|--fp8_base|--fp8base & --fp8_scaled|
|--|--|--|--|--|
|bf16|✓|--|✓|✓|
|fp16|--|✓|✓|✓|
|fp8_e4m3fn|--|--|✓|--|
|fp8_scaled|--|--|--|--|

<details>
<summary>日本語</summary>
T5 `models_t5_umt5-xxl-enc-bf16.pth` およびCLIP `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` を、次のページからダウンロードしてください：https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P/tree/main

VAEは上のページから `Wan2.1_VAE.pth` をダウンロードするか、次のページから `split_files/vae/wan_2.1_vae.safetensors` をダウンロードしてください：https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/vae

DiTの重みを次のページからダウンロードしてください：https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/tree/main/split_files/diffusion_models

T2VやI2V、解像度、モデルサイズなどにより適切な重みを選択してください。

`fp16` および `bf16` モデルを使用できます。また、`--fp8` （または`--fp8_base`）を指定し`--fp8_scaled`を指定をしないときには `fp8_e4m3fn` モデルを使用できます。**`fp8_scaled` モデルはいずれの場合もサポートされていませんのでご注意ください。**

（repackaged版の重みを提供してくださっているComfy-Orgに感謝いたします。）
</details>

## Pre-caching / 事前キャッシュ

### Latent Pre-caching

Latent pre-caching is almost the same as in HunyuanVideo. Create the cache using the following command:

```bash
python wan_cache_latents.py --dataset_config path/to/toml --vae path/to/wan_2.1_vae.safetensors
```

If you train I2V models, add `--clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` to specify the CLIP model. If not specified, the training will raise an error.

If you're running low on VRAM, specify `--vae_cache_cpu` to use the CPU for the VAE internal cache, which will reduce VRAM usage somewhat.

<details>
<summary>日本語</summary>
latentの事前キャッシングはHunyuanVideoとほぼ同じです。上のコマンド例を使用してキャッシュを作成してください。

I2Vモデルを学習する場合は、`--clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` を追加してCLIPモデルを指定してください。指定しないと学習時にエラーが発生します。

VRAMが不足している場合は、`--vae_cache_cpu` を指定するとVAEの内部キャッシュにCPUを使うことで、使用VRAMを多少削減できます。
</details>

### Text Encoder Output Pre-caching

Text encoder output pre-caching is also almost the same as in HunyuanVideo. Create the cache using the following command:

```bash
python wan_cache_text_encoder_outputs.py --dataset_config path/to/toml  --t5 path/to/models_t5_umt5-xxl-enc-bf16.pth --batch_size 16 
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
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py 
    --task t2v-1.3B 
    --dit path/to/wan2.1_xxx_bf16.safetensors 
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base 
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing 
    --max_data_loader_n_workers 2 --persistent_data_loader_workers 
    --network_module networks.lora_wan --network_dim 32 
    --timestep_sampling shift --discrete_flow_shift 3.0 
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42
    --output_dir path/to/output_dir --output_name name-of-lora
```
The above is an example. The appropriate values for `timestep_sampling` and `discrete_flow_shift` need to be determined by experimentation.

For additional options, use `python wan_train_network.py --help` (note that many options are unverified).

`--task` is one of `t2v-1.3B`, `t2v-14B`, `i2v-14B` and `t2i-14B`. Specify the DiT weights for the task with `--dit`.

Don't forget to specify `--network_module networks.lora_wan`.

Other options are mostly the same as `hv_train_network.py`.

Use `convert_lora.py` for converting the LoRA weights after training, as in HunyuanVideo.

<details>
<summary>日本語</summary>
`timestep_sampling`や`discrete_flow_shift`は一例です。どのような値が適切かは実験が必要です。

その他のオプションについては `python wan_train_network.py --help` を使用してください（多くのオプションは未検証です）。

`--task` には `t2v-1.3B`, `t2v-14B`, `i2v-14B`, `t2i-14B` のいずれかを指定します。`--dit`に、taskに応じたDiTの重みを指定してください。

 `--network_module` に `networks.lora_wan` を指定することを忘れないでください。

その他のオプションは、ほぼ`hv_train_network.py`と同様です。

学習後のLoRAの重みの変換は、HunyuanVideoと同様に`convert_lora.py`を使用してください。
</details>

### Command line options for training with sampling / サンプル画像生成に関連する学習時のコマンドラインオプション

Example of command line options for training with sampling / 記述例:  

```bash
--vae path/to/wan_2.1_vae.safetensors 
--t5 path/to/models_t5_umt5-xxl-enc-bf16.pth 
--sample_prompts /path/to/prompt_file.txt 
--sample_every_n_epochs 1 --sample_every_n_steps 1000 -- sample_at_first
```
Each option is the same as when generating images or as HunyuanVideo. Please refer to [here](/docs/sampling_during_training.md) for details.

If you train I2V models, add `--clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` to specify the CLIP model. 

You can specify the initial image and negative prompts in the prompt file. Please refer to [here](/docs/sampling_during_training.md#prompt-file--プロンプトファイル).

<details>
<summary>日本語</summary>
各オプションは推論時、およびHunyuanVideoの場合と同様です。[こちら](/docs/sampling_during_training.md)を参照してください。

I2Vモデルを学習する場合は、`--clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` を追加してCLIPモデルを指定してください。

プロンプトファイルで、初期画像やネガティブプロンプト等を指定できます。[こちら](/docs/sampling_during_training.md#prompt-file--プロンプトファイル)を参照してください。
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
python wan_generate_video.py --fp8 --task t2v-1.3B --video_size  832 480 --video_length 81 --infer_steps 20 
--prompt "prompt for the video" --save_path path/to/save.mp4 --output_type both 
--dit path/to/wan2.1_t2v_1.3B_bf16_etc.safetensors --vae path/to/wan_2.1_vae.safetensors 
--t5 path/to/models_t5_umt5-xxl-enc-bf16.pth 
--attn_mode torch
```

`--task` is one of `t2v-1.3B`, `t2v-14B`, `i2v-14B` and `t2i-14B`.

`--attn_mode` is `torch`, `sdpa` (same as `torch`), `xformers`, `sageattn`,`flash2`, `flash` (same as `flash2`) or `flash3`. `torch` is the default. Other options require the corresponding library to be installed. `flash3` (Flash attention 3) is not tested.

Specifying `--fp8` runs DiT in fp8 mode. fp8 can significantly reduce memory consumption but may impact output quality.

`--fp8_scaled` can be specified in addition to `--fp8` to run the model in fp8 weights optimization. This increases memory consumption and speed slightly but improves output quality. See [here](advanced_config.md#fp8-weight-optimization-for-models--モデルの重みのfp8への最適化) for details.

`--fp8_fast` option is also available for faster inference on RTX 40x0 GPUs. This option requires `--fp8_scaled` option. **This option seems to degrade the output quality.**

`--fp8_t5` can be used to specify the T5 model in fp8 format. This option reduces memory usage for the T5 model.  

`--negative_prompt` can be used to specify a negative prompt. If omitted, the default negative prompt is used.

`--flow_shift` can be used to specify the flow shift (default 3.0 for I2V with 480p, 5.0 for others).

`--guidance_scale` can be used to specify the guidance scale for classifier free guidance (default 5.0).

`--blocks_to_swap` is the number of blocks to swap during inference. The default value is None (no block swap). The maximum value is 39 for 14B model and 29 for 1.3B model.

`--vae_cache_cpu` enables VAE cache in main memory. This reduces VRAM usage slightly but processing is slower.

`--compile` enables torch.compile. See [here](/README.md#inference) for details.

`--trim_tail_frames` can be used to trim the tail frames when saving. The default is 0.

`--cfg_skip_mode` specifies the mode for skipping CFG in different steps. The default is `none` (all steps).`--cfg_apply_ratio` specifies the ratio of steps where CFG is applied. See below for details.

Other options are same as `hv_generate_video.py` (some options are not supported, please check the help).

<details>
<summary>日本語</summary>
`--task` には `t2v-1.3B`, `t2v-14B`, `i2v-14B`, `t2i-14B` のいずれかを指定します。

`--attn_mode` には `torch`, `sdpa`（`torch`と同じ）、`xformers`, `sageattn`, `flash2`, `flash`（`flash2`と同じ）, `flash3` のいずれかを指定します。デフォルトは `torch` です。その他のオプションを使用する場合は、対応するライブラリをインストールする必要があります。`flash3`（Flash attention 3）は未テストです。

`--fp8` を指定するとDiTモデルをfp8形式で実行します。fp8はメモリ消費を大幅に削減できますが、出力品質に影響を与える可能性があります。
    
`--fp8_scaled` を `--fp8` と併用すると、fp8への重み量子化を行います。メモリ消費と速度はわずかに悪化しますが、出力品質が向上します。詳しくは[こちら](advanced_config.md#fp8-weight-optimization-for-models--モデルの重みのfp8への最適化)を参照してください。

`--fp8_fast` オプションはRTX 40x0 GPUでの高速推論に使用されるオプションです。このオプションは `--fp8_scaled` オプションが必要です。**出力品質が劣化するようです。**

`--fp8_t5` を指定するとT5モデルをfp8形式で実行します。T5モデル呼び出し時のメモリ使用量を削減します。

`--negative_prompt` でネガティブプロンプトを指定できます。省略した場合はデフォルトのネガティブプロンプトが使用されます。

`--flow_shift` でflow shiftを指定できます（480pのI2Vの場合はデフォルト3.0、それ以外は5.0）。

`--guidance_scale` でclassifier free guianceのガイダンススケールを指定できます（デフォルト5.0）。

`--blocks_to_swap` は推論時のblock swapの数です。デフォルト値はNone（block swapなし）です。最大値は14Bモデルの場合39、1.3Bモデルの場合29です。

`--vae_cache_cpu` を有効にすると、VAEのキャッシュをメインメモリに保持します。VRAM使用量が多少減りますが、処理は遅くなります。

`--compile`でtorch.compileを有効にします。詳細については[こちら](/README.md#inference)を参照してください。

`--trim_tail_frames` で保存時に末尾のフレームをトリミングできます。デフォルトは0です。

`--cfg_skip_mode` は異なるステップでCFGをスキップするモードを指定します。デフォルトは `none`（全ステップ）。`--cfg_apply_ratio` はCFGが適用されるステップの割合を指定します。詳細は後述します。

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

### I2V Inference / I2V推論

The following is an example of I2V inference (input as a single line):

```bash
python wan_generate_video.py --fp8 --task i2v-14B --video_size 832 480 --video_length 81 --infer_steps 20 
--prompt "prompt for the video" --save_path path/to/save.mp4 --output_type both 
--dit path/to/wan2.1_i2v_480p_14B_bf16_etc.safetensors --vae path/to/wan_2.1_vae.safetensors 
--t5 path/to/models_t5_umt5-xxl-enc-bf16.pth --clip path/to/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth 
--attn_mode torch --image_path path/to/image.jpg
```

Add `--clip` to specify the CLIP model. `--image_path` is the path to the image to be used as the initial frame.

`--end_image_path` can be used to specify the end image. This option is experimental. When this option is specified, the saved video will be slightly longer than the specified number of frames and will have noise, so it is recommended to specify `--trim_tail_frames 3` to trim the tail frames.

Other options are same as T2V inference.

<details>
<summary>日本語</summary>
`--clip` を追加してCLIPモデルを指定します。`--image_path` は初期フレームとして使用する画像のパスです。

`--end_image_path` で終了画像を指定できます。このオプションは実験的なものです。このオプションを指定すると、保存される動画が指定フレーム数よりもやや多くなり、かつノイズが乗るため、`--trim_tail_frames 3` などを指定して末尾のフレームをトリミングすることをお勧めします。


その他のオプションはT2V推論と同じです。
</details>
