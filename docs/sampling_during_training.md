> 📝 Click on the language section to expand / 言語をクリックして展開

# Sampling during training / 学習中のサンプル画像生成

By preparing a prompt file, you can generate sample images during training.

Please be aware that it consumes a considerable amount of VRAM, so be careful when generating sample images for videos with a large number of frames. Also, since it takes time to generate, adjust the frequency of sample image generation as needed.

<details>
<summary>日本語</summary>

プロンプトファイルを用意することで、学習中にサンプル画像を生成することができます。

VRAMをそれなりに消費しますので、特にフレーム数が多い動画を生成する場合は注意してください。また生成には時間がかかりますので、サンプル画像生成の頻度は適宜調整してください。
</details>

## How to use / 使い方

### Command line options for training with sampling / サンプル画像生成に関連する学習時のコマンドラインオプション

Example of command line options for training with sampling / 記述例:  

```bash
--vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt 
--vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128
--text_encoder1 path/to/ckpts/text_encoder 
--text_encoder2 path/to/ckpts/text_encoder_2 
--sample_prompts /path/to/prompt_file.txt 
--sample_every_n_epochs 1 --sample_every_n_steps 1000 --sample_at_first
```

`--vae`, `--vae_chunk_size`, `--vae_spatial_tile_sample_min_size`, `--text_encoder1`, `--text_encoder2` are the same as when generating images, so please refer to [here](/README.md#inference) for details. `--fp8_llm` can also be specified.

`--sample_prompts` specifies the path to the prompt file used for sample image generation. Details are described below.

`--sample_every_n_epochs` specifies how often to generate sample images in epochs, and `--sample_every_n_steps` specifies how often to generate sample images in steps.

`--sample_at_first` is specified when generating sample images at the beginning of training.

Sample images and videos are saved in the `sample` directory in the directory specified by `--output_dir`. They are saved as `.png` for still images and `.mp4` for videos.

<details>
<summary>日本語</summary>

`--vae`、`--vae_chunk_size`、`--vae_spatial_tile_sample_min_size`、`--text_encoder1`、`--text_encoder2`は、画像生成時と同様ですので、詳細は[こちら](/README.ja.md#推論)を参照してください。`--fp8_llm`も指定可能です。

`--sample_prompts`は、サンプル画像生成に使用するプロンプトファイルのパスを指定します。詳細は後述します。

`--sample_every_n_epochs`は、何エポックごとにサンプル画像を生成するかを、`--sample_every_n_steps`は、何ステップごとにサンプル画像を生成するかを指定します。

`--sample_at_first`は、学習開始時にサンプル画像を生成する場合に指定します。

サンプル画像、動画は、`--output_dir`で指定したディレクトリ内の、`sample`ディレクトリに保存されます。静止画の場合は`.png`、動画の場合は`.mp4`で保存されます。
</details>

### Prompt file / プロンプトファイル

The prompt file is a text file that contains the prompts for generating sample images. The example is as follows. / プロンプトファイルは、サンプル画像生成のためのプロンプトを記述したテキストファイルです。例は以下の通りです。

```
# prompt 1: for generating a cat video
A cat walks on the grass, realistic style. --w 640 --h 480 --f 25 --d 1 --s 20

# prompt 2: for generating a dog image
A dog runs on the beach, realistic style. --w 960 --h 544 --f 1 --d 2 --s 20
```

A line starting with `#` is a comment.

* `--w` specifies the width of the generated image or video. The default is 256.
* `--h` specifies the height. The default is 256.
* `--f` specifies the number of frames. The default is 1, which generates a still image.
* `--d` specifies the seed. The default is random.
* `--s` specifies the number of steps in generation. The default is 20.
* `--g` specifies the embedded guidance scale (not CFG scale). The default is 6.0 for HunyuanVideo, 10.0 for FramePack, 2.5 for FLUX.1 Kontext which is the default value during inference of each architecture. Specify 1.0 for SkyReels V1 models. Ignore this option for Wan2.1 models.
* `--fs` specifies the discrete flow shift. The default is 14.5, which corresponds to the number of steps 20. In the HunyuanVideo paper, 7.0 is recommended for 50 steps, and 17.0 is recommended for less than 20 steps (e.g. 10). Ignore this option for FramePack models (it uses 10.0). Set 0 to use 'flux_shift' for FLUX.1 Kontext models.

If you train I2V models, you must add the following option.

* `--i path/to/image.png`: the image path for image2video inference. PNG, JPG and other formats are supported.

If you train Wan2.1-Fun-Control models, you must add the following option.

* `--cn path/to/control_video_or_dir_of_images`: the path to the video or directory containing multiple images for control.

If you train the model with classifier free guidance (such as Wan2.1), you can use the additional options below.

*`--n negative prompt...`: the negative prompt for the classifier free guidance. The default prompt for each model is used if omitted.
*`--l 6.0`: the classifier free guidance scale. Should be set to 6.0 for SkyReels V1 models. 5.0 is the default value for Wan2.1 (if omitted).

If you train the model with control images (such as FramePack one frame inference or FLUX.1 Kontext), you can use the additional options below.

* `--ci path/to/control_image.jpg`: the control image path for inference. If you specify this option, the control image is used for inference. PNG, JPG and other formats are supported.

### Sample image generation during Qwen-Image-Layered training / Qwen-Image-Layeredの学習中のサンプルイメージ生成

`--f` option is treated as the number of output layers.

The prompt can be omitted when generating sample images during Qwen-Image-Layered training. In this case, the prompt is generated based on the control image by Qwen2.5-VL.

※ Since Qwen-Image-Layered models generate "original image + multiple layer images", the number of images generated is the number specified by the `--f` option + 1. The second and subsequent images are separated layer images.

<details>
<summary>日本語</summary>

`#` で始まる行はコメントです。

* `--w` 生成画像、動画の幅を指定します。省略時は256です。
* `--h` 高さを指定します。省略時は256です。
* `--f` フレーム数を指定します。省略時は1で、静止画を生成します。
* `--d` シードを指定します。省略時はランダムです。
* `--s` 生成におけるステップ数を指定します。省略時は20です。
* `--g` embedded guidance scaleを指定します（CFG scaleではありません）。省略時はHunyuanVideoは6.0、FramePackは10.0で、各アーキテクチャの推論時のデフォルト値です。SkyReels V1モデルの場合は1.0を指定してください。FLUX.1 Kontextの場合は2.5を指定してください。Wan2.1モデルの場合はこのオプションは無視されます。
* `--fs` discrete flow shiftを指定します。省略時は14.5で、ステップ数20の場合に対応した値です。HunyuanVideoの論文では、ステップ数50の場合は7.0、ステップ数20未満（10など）で17.0が推奨されています。FramePackモデルはこのオプションは無視され、10.0が使用されます。FLUX.1 Kontextモデルでは、0を指定すると `flux_shift` が使用されます。

I2Vモデルを学習する場合、以下のオプションを追加してください。

* `--i path/to/image.png`: image2video推論用の画像パス。PNG、JPGなどの形式がサポートされています。

Wan2.1-Fun-Controlモデルを学習する場合、以下のオプションを追加してください。

* `--cn path/to/control_video_or_dir_of_images`: control用の動画または複数枚の画像を含むディレクトリのパス。

classifier free guidance（ネガティブプロンプト）を必要とするモデル（Wan2.1など）を学習する場合、以下の追加オプションを使用できます。

*`--n negative prompt...`: classifier free guidance用のネガティブプロンプト。省略時はモデルごとのデフォルトプロンプトが使用されます。
*`--l 6.0`: classifier free guidance scale。SkyReels V1モデルの場合は6.0に設定してください。Wan2.1の場合はデフォルト値が5.0です（省略時）。

制御画像を使用するモデル（FramePackの1フレーム推論やFLUX.1 Kontextなど）を学習する場合、以下の追加オプションを使用できます。

* `--ci path/to/control_image.jpg`: 推論用の制御画像パス。このオプションを指定すると、制御画像が推論に使用されます。PNG、JPGなどの形式がサポートされています。

**Qwen-Image-Layeredの学習中のサンプルイメージ生成**

`--f`オプションが出力レイヤー数として扱われます。

Qwen-Image-Layeredの学習中にサンプル画像を生成する際、プロンプトは省略可能です。この場合、プロンプトはQwen2.5-VLによってコントロール画像に基づいて生成されます。

※ Qwen-Image-Layeredモデルは「元画像＋複数のレイヤー画像」を生成するため、`--f`オプションで指定した数＋1枚の画像が生成されます。2枚目以降が分離されたレイヤー画像です。

</details>
