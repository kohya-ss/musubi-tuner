> 📝 Click on the language section to expand / 言語をクリックして展開

# Tools

This document provides documentation for utility tools available in this project. 

## Image Captioning with Qwen2.5-VL (`src/musubi_tuner/caption_images_by_qwen_vl.py`)

This script automatically generates captions for a directory of images using a fine-tuned Qwen2.5-VL model. It's designed to help prepare datasets for training by creating captions from the images themselves.

The Qwen2.5-VL model used in Qwen-Image is not confirmed to be the same as the original Qwen2.5-VL-Instruct model, but it appears to work for caption generation based on the tests conducted.

<details>
<summary>日本語</summary>

このスクリプトは、Qwen2.5-VLモデルを使用して、指定されたディレクトリ内の画像に対するキャプションを自動生成します。画像自体からキャプションを作成することで、学習用データセットの準備を支援することを目的としています。

Qwen-Imageで使用されているQwen2.5-VLモデルは、元のQwen2.5-VL-Instructモデルと同じかどうか不明ですが、試した範囲ではキャプション生成も動作するようです。

</details>

### Arguments

-   `--image_dir` (required): Path to the directory containing the images to be captioned.
-   `--model_path` (required): Path to the Qwen2.5-VL model. See [here](./qwen_image.md#download-the-model--モデルのダウンロード) for instructions.
-   `--output_file` (optional): Path to the output JSONL file. This is required if `--output_format` is `jsonl`.
-   `--max_new_tokens` (optional, default: 1024): The maximum number of new tokens to generate for each caption.
-   `--prompt` (optional, default: see script): A custom prompt to use for caption generation. You can use `\n` for newlines.
-   `--max_size` (optional, default: 1280): The maximum size of the image. Images are resized to fit within a `max_size` x `max_size` area while maintaining aspect ratio.
-   `--fp8_vl` (optional, flag): If specified, the Qwen2.5-VL model is loaded in fp8 precision for lower memory usage.
-   `--output_format` (optional, default: `jsonl`): The output format. Can be `jsonl` to save all captions in a single JSONL file, or `text` to save a separate `.txt` file for each image.

`--max_size` can be reduced to decrease the image size passed to the VLM. This can reduce the memory usage of the VLM, but may also decrease the quality of the generated captions.

The default prompt is defined in the [source file](./src/musubi_tuner/caption_images_by_qwen_vl.py). It is based on the [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324).

<details>
<summary>日本語</summary>

-   `--image_dir` (必須): キャプションを生成する画像が含まれるディレクトリへのパス。
-   `--model_path` (必須): Qwen2.5-VLモデルへのパス。詳細は[こちら](./qwen_image.md#download-the-model--モデルのダウンロード)を参照してください。
-   `--output_file` (任意): 出力先のJSONLファイルへのパス。`--output_format`が`jsonl`の場合に必須です。
-   `--max_new_tokens` (任意, デフォルト: 1024): 各キャプションで生成する新しいトークンの最大数。
-   `--prompt` (任意, デフォルト: スクリプト内参照): キャプション生成に使用するカスタムプロンプト。`\n`で改行を指定できます。
-   `--max_size` (任意, デフォルト: 1280): 画像の最大サイズ。アスペクト比を維持したまま、画像の合計ピクセル数が`max_size` x `max_size`の領域に収まるようにリサイズされます。
-   `--fp8_vl` (任意, フラグ): 指定された場合、Qwen2.5-VLモデルがfp8精度で読み込まれ、メモリ使用量が削減されます。
-   `--output_format` (任意, デフォルト: `jsonl`): 出力形式。`jsonl`を指定するとすべてのキャプションが単一のJSONLファイルに保存され、`text`を指定すると画像ごとに個別の`.txt`ファイルが保存されます。

`--max_size` を小さくするとVLMに渡される画像サイズが小さくなります。これにより、VLMのメモリ使用量が削減されますが、生成されるキャプションの品質が低下する可能性があります。

プロンプトのデフォルトは、[ソースファイル](./src/musubi_tuner/caption_images_by_qwen_vl.py)内で定義されています。[Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)を参考にしたものです。

</details>

### Usage Examples

**1. Basic Usage (JSONL Output)**

```bash
python src/musubi_tuner/caption_images_by_qwen_vl.py \
  --image_dir /path/to/images \
  --model_path /path/to/qwen_model.safetensors \
  --output_file /path/to/captions.jsonl
```

**2. Text File Output**

This will create a `.txt` file with the same name as each image in the `/path/to/images` directory.

```bash
python src/musubi_tuner/caption_images_by_qwen_vl.py \
  --image_dir /path/to/images \
  --model_path /path/to/qwen_model.safetensors \
  --output_format text
```

**3. Advanced Usage (fp8, Custom Prompt, and Max Size)**

```bash
python src/musubi_tuner/caption_images_by_qwen_vl.py \
  --image_dir /path/to/images \
  --model_path /path/to/qwen_model.safetensors \
  --output_file /path/to/captions.jsonl \
  --fp8_vl \
  --max_size 1024 \
  --prompt "A detailed and descriptive caption for this image is:\n"
```
