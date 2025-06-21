# FramePack

## Overview / 概要

This document describes the usage of the [FramePack](https://github.com/lllyasviel/FramePack) architecture within the Musubi Tuner framework. FramePack is a novel video generation architecture developed by lllyasviel.

Key differences from HunyuanVideo:
- FramePack only supports Image-to-Video (I2V) generation. Text-to-Video (T2V) is not supported.
- It utilizes a different DiT model architecture and requires an additional Image Encoder. VAE is same as HunyuanVideo. Text Encoders seem to be the same as HunyuanVideo but we employ the original FramePack method to utilize them.
- Caching and training scripts are specific to FramePack (`fpack_*.py`).
- Due to its progressive generation nature, VRAM usage can be significantly lower, especially for longer videos, compared to other architectures.

The official documentation does not provide detailed explanations on how to train the model, but it is based on the FramePack implementation and paper.

This feature is experimental.

For one-frame inference and training, see [here](./framepack_1f.md).

<details>
<summary>日本語</summary>

このドキュメントは、Musubi Tunerフレームワーク内での[FramePack](https://github.com/lllyasviel/FramePack) アーキテクチャの使用法について説明しています。FramePackは、lllyasviel氏にによって開発された新しいビデオ生成アーキテクチャです。

HunyuanVideoとの主な違いは次のとおりです。
- FramePackは、画像からビデオ（I2V）生成のみをサポートしています。テキストからビデオ（T2V）はサポートされていません。
- 異なるDiTモデルアーキテクチャを使用し、追加の画像エンコーダーが必要です。VAEはHunyuanVideoと同じです。テキストエンコーダーはHunyuanVideoと同じと思われますが、FramePack公式と同じ方法で推論を行っています。
- キャッシングと学習スクリプトはFramePack専用（`fpack_*.py`）です。
- セクションずつ生成するため、他のアーキテクチャと比較して、特に長いビデオの場合、VRAM使用量が大幅に少なくなる可能性があります。

学習方法について公式からは詳細な説明はありませんが、FramePackの実装と論文を参考にしています。

この機能は実験的なものです。

1フレーム推論、学習については[こちら](./framepack_1f.md)を参照してください。
</details>

## Download the model / モデルのダウンロード

You need to download the DiT, VAE, Text Encoder 1 (LLaMA), Text Encoder 2 (CLIP), and Image Encoder (SigLIP) models specifically for FramePack. Several download options are available for each component.

***Note:** The weights are publicly available on the following page: [maybleMyers/framepack_h1111](https://huggingface.co/maybleMyers/framepack_h1111) (except for FramePack-F1). Thank you maybleMyers!

### DiT Model

Choose one of the following methods:

1.  **From lllyasviel's Hugging Face repo:** Download the three `.safetensors` files (starting with `diffusion_pytorch_model-00001-of-00003.safetensors`) from [lllyasviel/FramePackI2V_HY](https://huggingface.co/lllyasviel/FramePackI2V_HY). Specify the path to the first file (`...-00001-of-00003.safetensors`) as the `--dit` argument. For FramePack-F1, download from [lllyasviel/FramePack_F1_I2V_HY_20250503](https://huggingface.co/lllyasviel/FramePack_F1_I2V_HY_20250503).

2.  **From local FramePack installation:** If you have cloned and run the official FramePack repository, the model might be downloaded locally. Specify the path to the snapshot directory, e.g., `path/to/FramePack/hf_download/hub/models--lllyasviel--FramePackI2V_HY/snapshots/<hex-uuid-folder>`. FramePack-F1 is also available in the same way.

3.  **From Kijai's Hugging Face repo:** Download the single file `FramePackI2V_HY_bf16.safetensors` from [Kijai/HunyuanVideo_comfy](https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/FramePackI2V_HY_bf16.safetensors). Specify the path to this file as the `--dit` argument. No FramePack-F1 model is available here currently.

### VAE Model

Choose one of the following methods:

1.  **Use official HunyuanVideo VAE:** Follow the instructions in the main [README.md](../README.md#model-download).
2.  **From hunyuanvideo-community Hugging Face repo:** Download `vae/diffusion_pytorch_model.safetensors` from [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo).
3.  **From local FramePack installation:** If you have cloned and run the official FramePack repository, the VAE might be downloaded locally within the HunyuanVideo community model snapshot. Specify the path to the snapshot directory, e.g., `path/to/FramePack/hf_download/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/<hex-uuid-folder>`.

### Text Encoder 1 (LLaMA) Model

Choose one of the following methods:

1.  **From Comfy-Org Hugging Face repo:** Download `split_files/text_encoders/llava_llama3_fp16.safetensors` from [Comfy-Org/HunyuanVideo_repackaged](https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged).
2.  **From hunyuanvideo-community Hugging Face repo:** Download the four `.safetensors` files (starting with `text_encoder/model-00001-of-00004.safetensors`) from [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo). Specify the path to the first file (`...-00001-of-00004.safetensors`) as the `--text_encoder1` argument.
3.  **From local FramePack installation:** (Same as VAE) Specify the path to the HunyuanVideo community model snapshot directory, e.g., `path/to/FramePack/hf_download/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/<hex-uuid-folder>`.

### Text Encoder 2 (CLIP) Model

Choose one of the following methods:

1.  **From Comfy-Org Hugging Face repo:** Download `split_files/text_encoders/clip_l.safetensors` from [Comfy-Org/HunyuanVideo_repackaged](https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged).
2.  **From hunyuanvideo-community Hugging Face repo:** Download `text_encoder_2/model.safetensors` from [hunyuanvideo-community/HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo).
3.  **From local FramePack installation:** (Same as VAE) Specify the path to the HunyuanVideo community model snapshot directory, e.g., `path/to/FramePack/hf_download/hub/models--hunyuanvideo-community--HunyuanVideo/snapshots/<hex-uuid-folder>`.

### Image Encoder (SigLIP) Model

Choose one of the following methods:

1.  **From Comfy-Org Hugging Face repo:** Download `sigclip_vision_patch14_384.safetensors` from [Comfy-Org/sigclip_vision_384](https://huggingface.co/Comfy-Org/sigclip_vision_384).
2.  **From lllyasviel's Hugging Face repo:** Download `image_encoder/model.safetensors` from [lllyasviel/flux_redux_bfl](https://huggingface.co/lllyasviel/flux_redux_bfl).
3.  **From local FramePack installation:** If you have cloned and run the official FramePack repository, the model might be downloaded locally. Specify the path to the snapshot directory, e.g., `path/to/FramePack/hf_download/hub/models--lllyasviel--flux_redux_bfl/snapshots/<hex-uuid-folder>`.

<details>
<summary>日本語</summary>

※以下のページに重みが一括で公開されています（FramePack-F1を除く）。maybleMyers 氏に感謝いたします。: https://huggingface.co/maybleMyers/framepack_h1111

DiT、VAE、テキストエンコーダー1（LLaMA）、テキストエンコーダー2（CLIP）、および画像エンコーダー（SigLIP）モデルは複数の方法でダウンロードできます。英語の説明を参考にして、ダウンロードしてください。

FramePack公式のリポジトリをクローンして実行した場合、モデルはローカルにダウンロードされている可能性があります。スナップショットディレクトリへのパスを指定してください。例：`path/to/FramePack/hf_download/hub/models--lllyasviel--flux_redux_bfl/snapshots/<hex-uuid-folder>`

HunyuanVideoの推論をComfyUIですでに行っている場合、いくつかのモデルはすでにダウンロードされている可能性があります。
</details>

## Pre-caching / 事前キャッシング

The default resolution for FramePack is 640x640. See [the source code](../src/musubi_tuner/frame_pack/bucket_tools.py) for the default resolution of each bucket. 

The dataset for training must be a video dataset. Image datasets are not supported. You can train on videos of any length. Specify `frame_extraction` as `full` and set `max_frames` to a sufficiently large value. However, if the video is too long, you may run out of VRAM during VAE encoding.

### Latent Pre-caching / latentの事前キャッシング

Latent pre-caching uses a dedicated script for FramePack. You **must** provide the Image Encoder model.

```bash
python src/musubi_tuner/fpack_cache_latents.py \
    --dataset_config path/to/toml \
    --vae path/to/vae_model.safetensors \
    --image_encoder path/to/image_encoder_model.safetensors \
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 
```

Key differences from HunyuanVideo caching:
-  Uses `fpack_cache_latents.py`.
-  Requires the `--image_encoder` argument pointing to the downloaded SigLIP model.
-  The script generates multiple cache files per video, each corresponding to a different section, with the section index appended to the filename (e.g., `..._frame_pos-0000-count_...` becomes `..._frame_pos-0000-0000-count_...`, `..._frame_pos-0000-0001-count_...`, etc.).
-   Image embeddings are calculated using the Image Encoder and stored in the cache files alongside the latents.

For VRAM savings during VAE decoding, consider using `--vae_chunk_size` and `--vae_spatial_tile_sample_min_size`. If VRAM is overflowing and using shared memory, it is recommended to set `--vae_chunk_size` to 16 or 8, and `--vae_spatial_tile_sample_min_size` to 64 or 32.

Specifying `--f1` is required for FramePack-F1 training. For one-frame training, specify `--one_frame`. If you change the presence of these options, please overwrite the existing cache without specifying `--skip_existing`.

`--one_frame_no_2x` and `--one_frame_no_4x` options are available for one-frame training, described in the next section. 

**FramePack-F1 support:**
You can apply the FramePack-F1 sampling method by specifying `--f1` during caching. The training script also requires specifying `--f1` to change the options during sample generation.

By default, the sampling method used is Inverted anti-drifting (the same as during inference with the original FramePack model, using the latent and index in reverse order), described in the paper. You can switch to FramePack-F1 sampling (Vanilla sampling, using the temporally ordered latent and index) by specifying `--f1`.

<details>
<summary>日本語</summary>

FramePackのデフォルト解像度は640x640です。各バケットのデフォルト解像度については、[ソースコード](../src/musubi_tuner/frame_pack/bucket_tools.py)を参照してください。

画像データセットでの学習は行えません。また動画の長さによらず学習可能です。 `frame_extraction` に `full` を指定して、`max_frames` に十分に大きな値を指定してください。ただし、あまりにも長いとVAEのencodeでVRAMが不足する可能性があります。

latentの事前キャッシングはFramePack専用のスクリプトを使用します。画像エンコーダーモデルを指定する必要があります。

HunyuanVideoのキャッシングとの主な違いは次のとおりです。
-  `fpack_cache_latents.py`を使用します。
-  ダウンロードしたSigLIPモデルを指す`--image_encoder`引数が必要です。
-  スクリプトは、各ビデオに対して複数のキャッシュファイルを生成します。各ファイルは異なるセクションに対応し、セクションインデックスがファイル名に追加されます（例：`..._frame_pos-0000-count_...`は`..._frame_pos-0000-0000-count_...`、`..._frame_pos-0000-0001-count_...`などになります）。
-  画像埋め込みは画像エンコーダーを使用して計算され、latentとともにキャッシュファイルに保存されます。

VAEのdecode時のVRAM節約のために、`--vae_chunk_size`と`--vae_spatial_tile_sample_min_size`を使用することを検討してください。VRAMがあふれて共有メモリを使用している場合には、`--vae_chunk_size`を16、8などに、`--vae_spatial_tile_sample_min_size`を64、32などに変更することをお勧めします。

FramePack-F1の学習を行う場合は`--f1`を指定してください。これらのオプションの有無を変更する場合には、`--skip_existing`を指定せずに既存のキャッシュを上書きしてください。

**FramePack-F1のサポート：**
キャッシュ時のオプションに`--f1`を指定することで、FramePack-F1のサンプリング方法を適用できます。学習スクリプトについても`--f1`を指定してサンプル生成時のオプションを変更する必要があります。

デフォルトでは、論文のサンプリング方法 Inverted anti-drifting （無印のFramePackの推論時と同じ、逆順の latent と index を使用）を使用します。`--f1`を指定すると FramePack-F1 の Vanilla sampling （時間順の latent と index を使用）に変更できます。
</details>

### Text Encoder Output Pre-caching / テキストエンコーダー出力の事前キャッシング

Text encoder output pre-caching also uses a dedicated script.

```bash
python src/musubi_tuner/fpack_cache_text_encoder_outputs.py \
    --dataset_config path/to/toml \
    --text_encoder1 path/to/text_encoder1 \
    --text_encoder2 path/to/text_encoder2 \
    --batch_size 16
```

Key differences from HunyuanVideo caching:
-   Uses `fpack_cache_text_encoder_outputs.py`.
-   Requires both `--text_encoder1` (LLaMA) and `--text_encoder2` (CLIP) arguments.
-   Uses `--fp8_llm` option to run the LLaMA Text Encoder 1 in fp8 mode for VRAM savings (similar to `--fp8_t5` in Wan2.1).
-   Saves LLaMA embeddings, attention mask, and CLIP pooler output to the cache file.

<details>
<summary>日本語</summary>

テキストエンコーダー出力の事前キャッシングも専用のスクリプトを使用します。

HunyuanVideoのキャッシングとの主な違いは次のとおりです。
-  `fpack_cache_text_encoder_outputs.py`を使用します。
- LLaMAとCLIPの両方の引数が必要です。
-  LLaMAテキストエンコーダー1をfp8モードで実行するための`--fp8_llm`オプションを使用します（Wan2.1の`--fp8_t5`に似ています）。
-  LLaMAの埋め込み、アテンションマスク、CLIPのプーラー出力をキャッシュファイルに保存します。

</details>


## Training / 学習

### Training

Training uses a dedicated script `fpack_train_network.py`. Remember FramePack only supports I2V training.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/fpack_train_network.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model.safetensors \
    --text_encoder1 path/to/text_encoder1 \
    --text_encoder2 path/to/text_encoder2 \
    --image_encoder path/to/image_encoder_model.safetensors \
    --dataset_config path/to/toml \
    --sdpa --mixed_precision bf16 \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 3.0 \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_framepack --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name name-of-lora
```

If you use the command prompt (Windows, not PowerShell), you may need to write them in a single line, or use `^` instead of `\` at the end of each line to continue the command.

The maximum value for `--blocks_to_swap` is 36. The default resolution for FramePack is 640x640, which requires around 17GB of VRAM. If you run out of VRAM, consider lowering the dataset resolution.

Key differences from HunyuanVideo training:
-   Uses `fpack_train_network.py`.
- `--f1` option is available for FramePack-F1 model training. You need to specify the FramePack-F1 model as `--dit`. This option only changes the sample generation during training. The training process itself is the same as the original FramePack model.
-   **Requires** specifying `--vae`, `--text_encoder1`, `--text_encoder2`, and `--image_encoder`.
-   **Requires** specifying `--network_module networks.lora_framepack`.
-  Optional `--latent_window_size` argument (default 9, should match caching).
-   Memory saving options like `--fp8` (for DiT) and `--fp8_llm` (for Text Encoder 1) are available. `--fp8_scaled` is recommended when using `--fp8` for DiT.
-   `--vae_chunk_size` and `--vae_spatial_tile_sample_min_size` options are available for the VAE to prevent out-of-memory during sampling (similar to caching).
-  `--gradient_checkpointing` is available for memory savings.
- If you encounter an error when the batch size is greater than 1 (especially when specifying `--sdpa` or `--xformers`, it will always result in an error), please specify `--split_attn`.
<!-- -   Use `convert_lora.py` for converting the LoRA weights after training, similar to HunyuanVideo. -->

Training settings (learning rate, optimizers, etc.) are experimental. Feedback is welcome.

<details>
<summary>日本語</summary>

FramePackの学習は専用のスクリプト`fpack_train_network.py`を使用します。FramePackはI2V学習のみをサポートしています。

コマンド記述例は英語版を参考にしてください。WindowsでPowerShellではなくコマンドプロンプトを使用している場合、コマンドを1行で記述するか、各行の末尾に`\`の代わりに`^`を付けてコマンドを続ける必要があります。

`--blocks_to_swap`の最大値は36です。FramePackのデフォルト解像度（640x640）では、17GB程度のVRAMが必要です。VRAM容量が不足する場合は、データセットの解像度を下げてください。

HunyuanVideoの学習との主な違いは次のとおりです。
-  `fpack_train_network.py`を使用します。
- FramePack-F1モデルの学習時には`--f1`を指定してください。この場合、`--dit`にFramePack-F1モデルを指定する必要があります。このオプションは学習時のサンプル生成時のみに影響し、学習プロセス自体は元のFramePackモデルと同じです。
-  `--vae`、`--text_encoder1`、`--text_encoder2`、`--image_encoder`を指定する必要があります。
-  `--network_module networks.lora_framepack`を指定する必要があります。
-  必要に応じて`--latent_window_size`引数（デフォルト9）を指定できます（キャッシング時と一致させる必要があります）。
-  `--fp8`（DiT用）や`--fp8_llm`（テキストエンコーダー1用）などのメモリ節約オプションが利用可能です。`--fp8_scaled`を使用することをお勧めします。
-  サンプル生成時にメモリ不足を防ぐため、VAE用の`--vae_chunk_size`、`--vae_spatial_tile_sample_min_size`オプションが利用可能です（キャッシング時と同様）。
-  メモリ節約のために`--gradient_checkpointing`が利用可能です。
- バッチサイズが1より大きい場合にエラーが出た時には（特に`--sdpa`や`--xformers`を指定すると必ずエラーになります。）、`--split_attn`を指定してください。

</details>

## Inference

Inference uses a dedicated script `fpack_generate_video.py`.

```bash
python src/musubi_tuner/fpack_generate_video.py \
    --dit path/to/dit_model \
    --vae path/to/vae_model.safetensors \
    --text_encoder1 path/to/text_encoder1 \
    --text_encoder2 path/to/text_encoder2 \
    --image_encoder path/to/image_encoder_model.safetensors \
    --image_path path/to/start_image.jpg \
    --prompt "A cat walks on the grass, realistic style." \
    --video_size 512 768 --video_seconds 5 --fps 30 --infer_steps 25 \
    --attn_mode sdpa --fp8_scaled \
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 \
    --save_path path/to/save/dir --output_type both \
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```
<!-- --embedded_cfg_scale 10.0 --guidance_scale 1.0 \ -->

Key differences from HunyuanVideo inference:
-   Uses `fpack_generate_video.py`.
- `--f1` option is available for FramePack-F1 model inference (forward generation). You need to specify the FramePack-F1 model as `--dit`.
-   **Requires** specifying `--vae`, `--text_encoder1`, `--text_encoder2`, and `--image_encoder`.
-   **Requires** specifying `--image_path` for the starting frame.
-   **Requires** specifying `--video_seconds` or `--video_sections`. `--video_seconds` specifies the length of the video in seconds, while `--video_sections` specifies the number of sections. If `--video_sections` is specified, `--video_seconds` is ignored.
- `--video_size` is the size of the generated video, height and width are specified in that order.
-   `--prompt`: Prompt for generation.
-  Optional `--latent_window_size` argument (default 9, should match caching and training).
-  `--fp8_scaled` option is available for DiT to reduce memory usage. Quality may be slightly lower. `--fp8_llm` option is available to reduce memory usage of Text Encoder 1. `--fp8` alone is also an option for DiT but `--fp8_scaled` potentially offers better quality.
-   LoRA loading options (`--lora_weight`, `--lora_multiplier`, `--include_patterns`, `--exclude_patterns`) are available. `--lycoris` is also supported.
-   `--embedded_cfg_scale` (default 10.0) controls the distilled guidance scale.
-   `--guidance_scale` (default 1.0) controls the standard classifier-free guidance scale. **Changing this from 1.0 is generally not recommended for the base FramePack model.**
-   `--guidance_rescale` (default 0.0) is available but typically not needed.
-   `--bulk_decode` option can decode all frames at once, potentially faster but uses more VRAM during decoding. `--vae_chunk_size` and `--vae_spatial_tile_sample_min_size` options are recommended to prevent out-of-memory errors.
-   `--sample_solver` (default `unipc`) is available but only `unipc` is implemented.
-   `--save_merged_model` option is available to save the DiT model after merging LoRA weights. Inference is skipped if this is specified.
- `--latent_paddings` option overrides the default padding for each section. Specify it as a comma-separated list of integers, e.g., `--latent_paddings 0,0,0,0`. This option is ignored if `--f1` is specified.
- `--custom_system_prompt` option overrides the default system prompt for the LLaMA Text Encoder 1. Specify it as a string. See [here](../src/musubi_tunerhunyuan_model/text_encoder.py#L152) for the default system prompt.
- `--rope_scaling_timestep_threshold` option is the RoPE scaling timestep threshold, default is None (disabled). If set, RoPE scaling is applied only when the timestep exceeds the threshold. Start with around 800 and adjust as needed. This option is intended for one-frame inference and may not be suitable for other cases.
- `--rope_scaling_factor` option is the RoPE scaling factor, default is 0.5, assuming a resolution of 2x. For 1.5x resolution, around 0.7 is recommended.

Other options like `--video_size`, `--fps`, `--infer_steps`, `--save_path`, `--output_type`, `--seed`, `--attn_mode`, `--blocks_to_swap`, `--vae_chunk_size`, `--vae_spatial_tile_sample_min_size` function similarly to HunyuanVideo/Wan2.1 where applicable.

`--output_type` supports `latent_images` in addition to the options available in HunyuanVideo/Wan2.1. This option saves the latent and image files in the specified directory. 

The LoRA weights that can be specified in `--lora_weight` are not limited to the FramePack weights trained in this repository. You can also specify the HunyuanVideo LoRA weights from this repository and the HunyuanVideo LoRA weights from diffusion-pipe (automatic detection).

The maximum value for `--blocks_to_swap` is 38.

<details>
<summary>日本語</summary>

FramePackの推論は専用のスクリプト`fpack_generate_video.py`を使用します。コマンド記述例は英語版を参考にしてください。

HunyuanVideoの推論との主な違いは次のとおりです。
-  `fpack_generate_video.py`を使用します。
- `--f1`を指定すると、FramePack-F1モデルの推論を行います（順方向で生成）。`--dit`にFramePack-F1モデルを指定する必要があります。
-  `--vae`、`--text_encoder1`、`--text_encoder2`、`--image_encoder`を指定する必要があります。
-  `--image_path`を指定する必要があります（開始フレーム）。
-  `--video_seconds` または `--video_sections` を指定する必要があります。`--video_seconds`は秒単位でのビデオの長さを指定し、`--video_sections`はセクション数を指定します。`--video_sections`を指定した場合、`--video_seconds`は無視されます。
-  `--video_size`は生成するビデオのサイズで、高さと幅をその順番で指定します。
-   `--prompt`: 生成用のプロンプトです。
-  必要に応じて`--latent_window_size`引数（デフォルト9）を指定できます（キャッシング時、学習時と一致させる必要があります）。
- DiTのメモリ使用量を削減するために、`--fp8_scaled`オプションを指定可能です。品質はやや低下する可能性があります。またText Encoder 1のメモリ使用量を削減するために、`--fp8_llm`オプションを指定可能です。DiT用に`--fp8`単独のオプションも用意されていますが、`--fp8_scaled`の方が品質が良い可能性があります。
-  LoRAの読み込みオプション（`--lora_weight`、`--lora_multiplier`、`--include_patterns`、`--exclude_patterns`）が利用可能です。LyCORISもサポートされています。
-  `--embedded_cfg_scale`（デフォルト10.0）は、蒸留されたガイダンススケールを制御します。通常は変更しないでください。
-  `--guidance_scale`（デフォルト1.0）は、標準の分類器フリーガイダンススケールを制御します。**FramePackモデルのベースモデルでは、通常1.0から変更しないことをお勧めします。**
-  `--guidance_rescale`（デフォルト0.0）も利用可能ですが、通常は必要ありません。
-  `--bulk_decode`オプションは、すべてのフレームを一度にデコードできるオプションです。高速ですが、デコード中にVRAMを多く使用します。VRAM不足エラーを防ぐために、`--vae_chunk_size`と`--vae_spatial_tile_sample_min_size`オプションを指定することをお勧めします。
-  `--sample_solver`（デフォルト`unipc`）は利用可能ですが、`unipc`のみが実装されています。
-  `--save_merged_model`オプションは、LoRAの重みをマージした後にDiTモデルを保存するためのオプションです。これを指定すると推論はスキップされます。
- `--latent_paddings`オプションは、各セクションのデフォルトのパディングを上書きします。カンマ区切りの整数リストとして指定します。例：`--latent_paddings 0,0,0,0`。`--f1`を指定した場合は無視されます。
- `--custom_system_prompt`オプションは、LLaMA Text Encoder 1のデフォルトのシステムプロンプトを上書きします。文字列として指定します。デフォルトのシステムプロンプトは[こちら](../src/musubi_tuner/hunyuan_model/text_encoder.py#L152)を参照してください。
- `--rope_scaling_timestep_threshold`オプションはRoPEスケーリングのタイムステップ閾値で、デフォルトはNone（無効）です。設定すると、タイムステップが閾値以上の場合にのみRoPEスケーリングが適用されます。800程度から初めて調整してください。1フレーム推論時での使用を想定しており、それ以外の場合は想定していません。
- `--rope_scaling_factor`オプションはRoPEスケーリング係数で、デフォルトは0.5で、解像度が2倍の場合を想定しています。1.5倍なら0.7程度が良いでしょう。

`--video_size`、`--fps`、`--infer_steps`、`--save_path`、`--output_type`、`--seed`、`--attn_mode`、`--blocks_to_swap`、`--vae_chunk_size`、`--vae_spatial_tile_sample_min_size`などの他のオプションは、HunyuanVideo/Wan2.1と同様に機能します。

`--lora_weight`に指定できるLoRAの重みは、当リポジトリで学習したFramePackの重み以外に、当リポジトリのHunyuanVideoのLoRA、diffusion-pipeのHunyuanVideoのLoRAが指定可能です（自動判定）。

`--blocks_to_swap`の最大値は38です。
</details>

## Batch and Interactive Modes / バッチモードとインタラクティブモード

In addition to single video generation, FramePack now supports batch generation from file and interactive prompt input:

### Batch Mode from File / ファイルからのバッチモード

Generate multiple videos from prompts stored in a text file:

```bash
python src/musubi_tuner/fpack_generate_video.py --from_file prompts.txt 
--dit path/to/dit_model --vae path/to/vae_model.safetensors 
--text_encoder1 path/to/text_encoder1 --text_encoder2 path/to/text_encoder2 
--image_encoder path/to/image_encoder_model.safetensors --save_path output_directory
```

The prompts file format:
- One prompt per line
- Empty lines and lines starting with # are ignored (comments)
- Each line can include prompt-specific parameters using command-line style format:

```
A beautiful sunset over mountains --w 832 --h 480 --f 5 --d 42 --s 20 --i path/to/start_image.jpg
A busy city street at night --w 480 --h 832 --i path/to/another_start.jpg
```

Supported inline parameters (if omitted, default values from the command line are used):
- `--w`: Width
- `--h`: Height
- `--f`: Video seconds
- `--d`: Seed
- `--s`: Inference steps
- `--g` or `--l`: Guidance scale
- `--i`: Image path (for start image)
- `--im`: Image mask path
- `--n`: Negative prompt
- `--vs`: Video sections
- `--ei`: End image path
- `--ci`: Control image path (explained in one-frame inference documentation)
- `--cim`: Control image mask path (explained in one-frame inference documentation)
- `--of`: One frame inference mode options (same as `--one_frame_inference` in the command line), options for one-frame inference

In batch mode, models are loaded once and reused for all prompts, significantly improving overall generation time compared to multiple single runs.

### Interactive Mode / インタラクティブモード

Interactive command-line interface for entering prompts:

```bash
python src/musubi_tuner/fpack_generate_video.py --interactive
--dit path/to/dit_model --vae path/to/vae_model.safetensors 
--text_encoder1 path/to/text_encoder1 --text_encoder2 path/to/text_encoder2 
--image_encoder path/to/image_encoder_model.safetensors --save_path output_directory
```

In interactive mode:
- Enter prompts directly at the command line
- Use the same inline parameter format as batch mode
- Use Ctrl+D (or Ctrl+Z on Windows) to exit
- Models remain loaded between generations for efficiency

<details>
<summary>日本語</summary>

単一動画の生成に加えて、FramePackは現在、ファイルからのバッチ生成とインタラクティブなプロンプト入力をサポートしています。

#### ファイルからのバッチモード

テキストファイルに保存されたプロンプトから複数の動画を生成します：

```bash
python src/musubi_tuner/fpack_generate_video.py --from_file prompts.txt 
--dit path/to/dit_model --vae path/to/vae_model.safetensors 
--text_encoder1 path/to/text_encoder1 --text_encoder2 path/to/text_encoder2 
--image_encoder path/to/image_encoder_model.safetensors --save_path output_directory
```

プロンプトファイルの形式（サンプルは英語ドキュメントを参照）：
- 1行に1つのプロンプト
- 空行や#で始まる行は無視されます（コメント）
- 各行にはコマンドライン形式でプロンプト固有のパラメータを含めることができます：

サポートされているインラインパラメータ（省略した場合、コマンドラインのデフォルト値が使用されます）
- `--w`: 幅
- `--h`: 高さ
- `--f`: 動画の秒数
- `--d`: シード
- `--s`: 推論ステップ
- `--g` または `--l`: ガイダンススケール
- `--i`: 画像パス（開始画像用）
- `--im`: 画像マスクパス
- `--n`: ネガティブプロンプト
- `--vs`: 動画セクション数
- `--ei`: 終了画像パス
- `--ci`: 制御画像パス（1フレーム推論のドキュメントで解説）
- `--cim`: 制御画像マスクパス（1フレーム推論のドキュメントで解説）
- `--of`: 1フレーム推論モードオプション（コマンドラインの`--one_frame_inference`と同様、1フレーム推論のオプション）

バッチモードでは、モデルは一度だけロードされ、すべてのプロンプトで再利用されるため、複数回の単一実行と比較して全体的な生成時間が大幅に改善されます。

#### インタラクティブモード

プロンプトを入力するためのインタラクティブなコマンドラインインターフェース：

```bash
python src/musubi_tuner/fpack_generate_video.py --interactive
--dit path/to/dit_model --vae path/to/vae_model.safetensors 
--text_encoder1 path/to/text_encoder1 --text_encoder2 path/to/text_encoder2 
--image_encoder path/to/image_encoder_model.safetensors --save_path output_directory
```

インタラクティブモードでは：
- コマンドラインで直接プロンプトを入力
- バッチモードと同じインラインパラメータ形式を使用
- 終了するには Ctrl+D (Windowsでは Ctrl+Z) を使用
- 効率のため、モデルは生成間で読み込まれたままになります
</details>

## Advanced Video Control Features (Experimental) / 高度なビデオ制御機能（実験的）

This section describes experimental features added to the `fpack_generate_video.py` script to provide finer control over the generated video content, particularly useful for longer videos or sequences requiring specific transitions or states. These features leverage the Inverted Anti-drifting sampling method inherent to FramePack.

### **1. End Image Guidance (`--end_image_path`)**

*   **Functionality:** Guides the generation process to make the final frame(s) of the video resemble a specified target image.
*   **Usage:** `--end_image_path <path_to_image_file>`
*   **Mechanism:** The provided image is encoded using the VAE. This latent representation is used as a target or starting point during the generation of the final video section (which is the first step in Inverted Anti-drifting).
*   **Use Cases:** Defining a clear ending for the video, such as a character striking a specific pose or a product appearing in a close-up.

This option is ignored if `--f1` is specified. The end image is not used in the FramePack-F1 model.

### **2. Section Start Image Guidance (`--image_path` Extended Format)**

*   **Functionality:** Guides specific sections within the video to start with a visual state close to a provided image.
    * You can force the start image by setting `--latent_paddings` to `0,0,0,0` (specify the number of sections as a comma-separated list). If `latent_paddings` is set to 1 or more, the specified image will be used as a reference image (default behavior).
*   **Usage:** `--image_path "SECTION_SPEC:path/to/image.jpg;;;SECTION_SPEC:path/to/another.jpg;;;..."`
    *   `SECTION_SPEC`: Defines the target section(s). Rules:
        *   `0`: The first section of the video (generated last in Inverted Anti-drifting).
        *   `-1`: The last section of the video (generated first).
        *   `N` (non-negative integer): The N-th section (0-indexed).
        *   `-N` (negative integer): The N-th section from the end.
        *   `S-E` (range, e.g., `0-2`): Applies the same image guidance to sections S through E (inclusive).
    *   Use `;;;` as a separator between definitions.
    *   If no image is specified for a section, generation proceeds based on the prompt and preceding (future time) section context.
*   **Mechanism:** When generating a specific section, if a corresponding start image is provided, its VAE latent representation is strongly referenced as the "initial state" for that section. This guides the beginning of the section towards the specified image while attempting to maintain temporal consistency with the subsequent (already generated) section.
*   **Use Cases:** Defining clear starting points for scene changes, specifying character poses or attire at the beginning of certain sections.

### **3. Section-Specific Prompts (`--prompt` Extended Format)**

*   **Functionality:** Allows providing different text prompts for different sections of the video, enabling more granular control over the narrative or action flow.
*   **Usage:** `--prompt "SECTION_SPEC:Prompt text for section(s);;;SECTION_SPEC:Another prompt;;;..."`
    *   `SECTION_SPEC`: Uses the same rules as `--image_path`.
    *   Use `;;;` as a separator.
    *   If a prompt for a specific section is not provided, the prompt associated with index `0` (or the closest specified applicable prompt) is typically used. Check behavior if defaults are critical.
*   **Mechanism:** During the generation of each section, the corresponding section-specific prompt is used as the primary textual guidance for the model.
*   **Prompt Content Recommendation** when using `--latent_paddings 0,0,0,0` without `--f1` (original FramePack model):
    *   Recall that FramePack uses Inverted Anti-drifting and references future context.
    *   It is recommended to describe "**the main content or state change that should occur in the current section, *and* the subsequent events or states leading towards the end of the video**" in the prompt for each section.
    *   Including the content of subsequent sections in the current section's prompt helps the model maintain context and overall coherence.
    *   Example: For section 1, the prompt might describe what happens in section 1 *and* briefly summarize section 2 (and beyond).
    *   However, based on observations (e.g., the `latent_paddings` comment), the model's ability to perfectly utilize very long-term context might be limited. Experimentation is key. Describing just the "goal for the current section" might also work. Start by trying the "section and onwards" approach.
* Use the default prompt when `latent_paddings` is >= 1 or `--latent_paddings` is not specified, or when using `--f1` (FramePack-F1 model). 
*   **Use Cases:** Describing evolving storylines, gradual changes in character actions or emotions, step-by-step processes over time.

### **Combined Usage Example** (with `--f1` not specified)

Generating a 3-section video of "A dog runs towards a thrown ball, catches it, and runs back":

```bash
python src/musubi_tuner/fpack_generate_video.py \
 --prompt "0:A dog runs towards a thrown ball, catches it, and runs back;;;1:The dog catches the ball and then runs back towards the viewer;;;2:The dog runs back towards the viewer holding the ball" \
 --image_path "0:./img_start_running.png;;;1:./img_catching.png;;;2:./img_running_back.png" \
 --end_image_path ./img_returned.png \
 --save_path ./output \
 # ... other arguments
```

*   **Generation Order:** Section 2 -> Section 1 -> Section 0
*   **Generating Section 2:**
    *   Prompt: "The dog runs back towards the viewer holding the ball"
    *   Start Image: `./img_running_back.png`
    *   End Image: `./img_returned.png` (Initial target)
*   **Generating Section 1:**
    *   Prompt: "The dog catches the ball and then runs back towards the viewer"
    *   Start Image: `./img_catching.png`
    *   Future Context: Generated Section 2 latent
*   **Generating Section 0:**
    *   Prompt: "A dog runs towards a thrown ball, catches it, and runs back"
    *   Start Image: `./img_start_running.png`
    *   Future Context: Generated Section 1 & 2 latents

### **Important Considerations**

*   **Inverted Generation:** Always remember that generation proceeds from the end of the video towards the beginning. Section `-1` (the last section, `2` in the example) is generated first.
*   **Continuity vs. Guidance:** While start image guidance is powerful, drastically different images between sections might lead to unnatural transitions. Balance guidance strength with the need for smooth flow.
*   **Prompt Optimization:** The prompt content recommendation is a starting point. Fine-tune prompts based on observed model behavior and desired output quality.

<details>
<summary>日本語</summary>

### **高度な動画制御機能（実験的）**

このセクションでは、`fpack_generate_video.py` スクリプトに追加された実験的な機能について説明します。これらの機能は、生成される動画の内容をより詳細に制御するためのもので、特に長い動画や特定の遷移・状態が必要なシーケンスに役立ちます。これらの機能は、FramePack固有のInverted Anti-driftingサンプリング方式を活用しています。

#### **1. 終端画像ガイダンス (`--end_image_path`)**

*   **機能:** 動画の最後のフレーム（群）を指定したターゲット画像に近づけるように生成を誘導します。
*   **書式:** `--end_image_path <画像ファイルパス>`
*   **動作:** 指定された画像はVAEでエンコードされ、その潜在表現が動画の最終セクション（Inverted Anti-driftingでは最初に生成される）の生成時の目標または開始点として使用されます。
*   **用途:** キャラクターが特定のポーズで終わる、特定の商品がクローズアップで終わるなど、動画の結末を明確に定義する場合。

このオプションは、`--f1`を指定した場合は無視されます。FramePack-F1モデルでは終端画像は使用されません。

#### **2. セクション開始画像ガイダンス (`--image_path` 拡張書式)**

*   **機能:** 動画内の特定のセクションが、指定された画像に近い視覚状態から始まるように誘導します。
    * `--latent_paddings`を`0,0,0,0`（カンマ区切りでセクション数だけ指定）に設定することで、セクションの開始画像を強制できます。`latent_paddings`が1以上の場合、指定された画像は参照画像として使用されます。
*   **書式:** `--image_path "セクション指定子:画像パス;;;セクション指定子:別の画像パス;;;..."`
    *   `セクション指定子`: 対象セクションを定義します。ルール：
        *   `0`: 動画の最初のセクション（Inverted Anti-driftingでは最後に生成）。
        *   `-1`: 動画の最後のセクション（最初に生成）。
        *   `N`（非負整数）: N番目のセクション（0始まり）。
        *   `-N`（負整数）: 最後からN番目のセクション。
        *   `S-E`（範囲, 例:`0-2`）: セクションSからE（両端含む）に同じ画像を適用。
    *   区切り文字は `;;;` です。
    *   セクションに画像が指定されていない場合、プロンプトと後続（未来時刻）セクションのコンテキストに基づいて生成されます。
*   **動作:** 特定セクションの生成時、対応する開始画像が指定されていれば、そのVAE潜在表現がそのセクションの「初期状態」として強く参照されます。これにより、後続（生成済み）セクションとの時間的連続性を維持しようとしつつ、セクションの始まりを指定画像に近づけます。
*   **用途:** シーン変更の起点を明確にする、特定のセクション開始時のキャラクターのポーズや服装を指定するなど。

#### **3. セクション別プロンプト (`--prompt` 拡張書式)**

*   **機能:** 動画のセクションごとに異なるテキストプロンプトを与え、物語やアクションの流れをより細かく指示できます。
*   **書式:** `--prompt "セクション指定子:プロンプトテキスト;;;セクション指定子:別のプロンプト;;;..."`
    *   `セクション指定子`: `--image_path` と同じルールです。
    *   区切り文字は `;;;` です。
    *   特定セクションのプロンプトがない場合、通常はインデックス`0`に関連付けられたプロンプト（または最も近い適用可能な指定プロンプト）が使用されます。デフォルトの挙動が重要な場合は確認してください。
*   **動作:** 各セクションの生成時、対応するセクション別プロンプトがモデルへの主要なテキスト指示として使用されます。
*  `latent_paddings`に`0`を指定した場合（非F1モデル）の **プロンプト内容の推奨:**
    *   FramePackはInverted Anti-driftingを採用し、未来のコンテキストを参照することを思い出してください。
    *   各セクションのプロンプトには、「**現在のセクションで起こるべき主要な内容や状態変化、*および*それに続く動画の終端までの内容**」を記述することを推奨します。
    *   現在のセクションのプロンプトに後続セクションの内容を含めることで、モデルが全体的な文脈を把握し、一貫性を保つのに役立ちます。
    *   例：セクション1のプロンプトには、セクション1の内容 *と* セクション2の簡単な要約を記述します。
    *   ただし、モデルの長期コンテキスト完全利用能力には限界がある可能性も示唆されています（例：`latent_paddings`コメント）。実験が鍵となります。「現在のセクションの目標」のみを記述するだけでも機能する場合があります。まずは「セクションと以降」アプローチを試すことをお勧めします。
* 使用するプロンプトは、`latent_paddings`が`1`以上または指定されていない場合、または`--f1`（FramePack-F1モデル）を使用している場合は、通常のプロンプト内容を記述してください。
*   **用途:** 時間経過に伴うストーリーの変化、キャラクターの行動や感情の段階的な変化、段階的なプロセスなどを記述する場合。

#### **組み合わせ使用例** （`--f1`未指定時）

「投げられたボールに向かって犬が走り、それを捕まえ、走って戻ってくる」3セクション動画の生成：
（コマンド記述例は英語版を参考にしてください）

*   **生成順序:** セクション2 → セクション1 → セクション0
*   **セクション2生成時:**
    *   プロンプト: "犬がボールを咥えてこちらに向かって走ってくる"
    *   開始画像: `./img_running_back.png`
    *   終端画像: `./img_returned.png` （初期目標）
*   **セクション1生成時:**
    *   プロンプト: "犬がボールを捕まえ、その後こちらに向かって走ってくる"
    *   開始画像: `./img_catching.png`
    *   未来コンテキスト: 生成済みセクション2の潜在表現
*   **セクション0生成時:**
    *   プロンプト: "犬が投げられたボールに向かって走り、それを捕まえ、走って戻ってくる"
    *   開始画像: `./img_start_running.png`
    *   未来コンテキスト: 生成済みセクション1 & 2の潜在表現

#### **重要な考慮事項**

*   **逆順生成:** 生成は動画の終わりから始まりに向かって進むことを常に意識してください。セクション`-1`（最後のセクション、上の例では `2`）が最初に生成されます。
*   **連続性とガイダンスのバランス:** 開始画像ガイダンスは強力ですが、セクション間で画像が大きく異なると、遷移が不自然になる可能性があります。ガイダンスの強さとスムーズな流れの必要性のバランスを取ってください。
*   **プロンプトの最適化:** 推奨されるプロンプト内容はあくまでも参考です。モデルの観察された挙動と望ましい出力品質に基づいてプロンプトを微調整してください。

</details>
