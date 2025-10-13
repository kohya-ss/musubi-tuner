# FramePack One Frame (Single Frame) Inference and Training / FramePack 1フレーム推論と学習

## Overview / 概要

This document explains advanced inference and training methods using the FramePack model, particularly focusing on **"1-frame inference"** and its extensions. These features aim to leverage FramePack's flexibility to enable diverse image generation and editing tasks beyond simple video generation.

### The Concept and Development of 1-Frame Inference

While FramePack is originally a model for generating sequential video frames (or frame sections), it was discovered that by focusing on its internal structure, particularly how it handles temporal information with RoPE (Rotary Position Embedding), interesting control over single-frame generation is possible.

1.  **Basic 1-Frame Inference**:
    *   It takes an initial image and a prompt as input, limiting the number of generated frames to just one.
    *   In this process, by intentionally setting a large RoPE timestamp (`target_index`) for the single frame to be generated, a single static image can be obtained that reflects temporal and semantic changes from the initial image according to the prompt.
    *   This utilizes FramePack's characteristic of being highly sensitive to RoPE timestamps, as it supports bidirectional contexts like "Inverted anti-drifting." This allows for operations similar to natural language-based image editing, albeit in a limited capacity, without requiring additional training.

2.  **Kisekaeichi Method (Feature Merging via Post-Reference)**:
    *   This method, an extension of basic 1-frame inference, was **proposed by furusu**. In addition to the initial image, it also uses a reference image corresponding to a "next section-start image" (treated as `clean_latent_post`) as input.
    *   The RoPE timestamp (`target_index`) for the image to be generated is set to an intermediate value between the timestamps of the initial image and the section-end image.
    *   More importantly, masking (e.g., zeroing out specific regions) is applied to the latent representation of each reference image. For example, by setting masks to extract a character's face and body shape from the initial image and clothing textures from the reference image, an image can be generated that fuses the desired features of both, similar to a character "dress-up" or outfit swapping. This method can also be fundamentally achieved without additional training.

3.  **1f-mc (one frame multi-control) Method (Proximal Frame Blending)**:
    *   This method was **proposed by mattyamonaca**. It takes two reference images as input: an initial image (e.g., at `t=0`) and a subsequent image (e.g., at `t=1`, the first frame of a section), and generates a single image blending their features.
    *   Unlike Kisekaeichi, latent masking is typically not performed.
    *   To fully leverage this method, additional training using LoRA (Low-Rank Adaptation) is recommended. Through training, the model can better learn the relationship and blending method between the two input images to achieve specific editing effects.

### Integration into a Generalized Control Framework

The concepts utilized in the methods above—specifying reference images, manipulating timestamps, and applying latent masks—have been generalized to create a more flexible control framework.
Users can arbitrarily specify the following elements for both inference and LoRA training:

*   **Control Images**: Any set of input images intended to influence the model.
*   **Clean Latent Index (Indices)**: Timestamps corresponding to each control image. These are treated as `clean latent index` internally by FramePack and can be set to any position on the time axis. This is specified as `control_index`.
*   **Latent Masks**: Masks applied to the latent representation of each control image, allowing selective control over which features from the control images are utilized. This is specified as `control_image_mask_path` or the alpha channel of the control image.
*   **Target Index**: The timestamp for the single frame to be generated.

This generalized control framework, along with corresponding extensions to the inference and LoRA training tools, has enabled advanced applications such as:

*   Development of LoRAs that stabilize 1-frame inference effects (e.g., a camera orbiting effect) that were previously unstable with prompts alone.
*   Development of Kisekaeichi LoRAs that learn to perform desired feature merging under specific conditions (e.g., ignoring character information from a clothing reference image), thereby automating the masking process through learning.

These features maximize FramePack's potential and open up new creative possibilities in static image generation and editing. Subsequent sections will detail the specific options for utilizing these functionalities.

<details>
<summary>日本語</summary>

このドキュメントでは、FramePackモデルを用いた高度な推論および学習手法、特に「1フレーム推論」とその拡張機能について解説します。これらの機能は、FramePackの柔軟性を活かし、動画生成に留まらない多様な画像生成・編集タスクを実現することを目的としています。

### 1フレーム推論の発想と発展

FramePackは本来、連続する動画フレーム（またはフレームセクション）を生成するモデルですが、その内部構造、特に時間情報を扱うRoPE (Rotary Position Embedding) の扱いに着目することで、単一フレームの生成においても興味深い制御が可能になることが発見されました。

1.  **基本的な1フレーム推論**:
    *   開始画像とプロンプトを入力とし、生成するフレーム数を1フレームに限定します。
    *   この際、生成する1フレームに割り当てるRoPEのタイムスタンプ（`target_index`）を意図的に大きな値に設定することで、開始画像からプロンプトに従って時間的・意味的に変化した単一の静止画を得ることができます。
    *   これは、FramePackがInverted anti-driftingなどの双方向コンテキストに対応するため、RoPEのタイムスタンプに対して敏感に反応する特性を利用したものです。これにより、学習なしで限定的ながら自然言語による画像編集に近い操作が可能です。

2.  **kisekaeichi方式 (ポスト参照による特徴マージ)**:
    *   基本的な1フレーム推論を発展させたこの方式は、**furusu氏により提案されました**。開始画像に加え、「次のセクションの開始画像」に相当する参照画像（`clean_latent_post`として扱われる）も入力として利用します。
    *   生成する画像のRoPEタイムスタンプ（`target_index`）を、開始画像のタイムスタンプとセクション終端画像のタイムスタンプの中間的な値に設定します。
    *   さらに重要な点として、各参照画像のlatent表現に対してマスク処理（特定領域を0で埋めるなど）を施します。例えば、開始画像からはキャラクターの顔や体型を、参照画像からは服装のテクスチャを抽出するようにマスクを設定することで、キャラクターの「着せ替え」のような、両者の望ましい特徴を融合させた画像を生成できます。この手法も基本的には学習不要で実現可能です。

3.  **1f-mc (one frame multi-control) 方式 (近接フレームブレンド)**:
    *   この方式は、**mattyamonaca氏により提案されました**。開始画像（例: `t=0`）と、その直後の画像（例: `t=1`、セクションの最初のフレーム）の2つを参照画像として入力し、それらの特徴をブレンドした単一画像を生成します。
    *   kisekaeichiとは異なり、latentマスクは通常行いません。
    *   この方式の真価を発揮するには、LoRA (Low-Rank Adaptation) による追加学習が推奨されます。学習により、モデルは2つの入力画像間の関係性やブレンド方法をより適切に学習し、特定の編集効果を実現できます。

### 汎用的な制御フレームワークへの統合

上記の各手法で利用されていた「参照画像の指定」「タイムスタンプの操作」「latentマスクの適用」といった概念を一般化し、より柔軟な制御を可能にするための拡張が行われました。
ユーザーは以下の要素を任意に指定して、推論およびLoRA学習を行うことができます。

*   **制御画像 (Control Images)**: モデルに影響を与えるための任意の入力画像群。
*   **Clean Latent Index (Indices)**: 各制御画像に対応するタイムスタンプ。FramePack内部の`clean latent index`として扱われ、時間軸上の任意の位置を指定可能です。`control_index`として指定します。
*   **Latentマスク (Latent Masks)**: 各制御画像のlatentに適用するマスク。これにより、制御画像から利用する特徴を選択的に制御します。`control_image_mask_path`または制御画像のアルファチャンネルとして指定します。
*   **Target Index**: 生成したい単一フレームのタイムスタンプ。

この汎用的な制御フレームワークと、それに対応した推論ツールおよびLoRA学習ツールの拡張により、以下のような高度な応用が可能になりました。

*   プロンプトだけでは不安定だった1フレーム推論の効果（例: カメラ旋回）を安定化させるLoRAの開発。
*   マスク処理を手動で行う代わりに、特定の条件下（例: 服の参照画像からキャラクター情報を無視する）で望ましい特徴マージを行うように学習させたkisekaeichi LoRAの開発。

これらの機能は、FramePackのポテンシャルを最大限に引き出し、静止画生成・編集における新たな創造の可能性を拓くものです。以降のセクションでは、これらの機能を実際に利用するための具体的なオプションについて説明します。

</details>

## One Frame (Single Frame) Training / 1フレーム学習

**This feature is experimental.** It trains in the same way as one frame inference.

The dataset must be an image dataset. If you use caption files, you need to specify `control_directory` and place the **start images** in that directory. The `image_directory` should contain the images after the change. The filenames of both directories must match. Caption files should be placed in the `image_directory`.

If you use JSONL files, specify them as `{"image_path": "/path/to/target_image1.jpg", "control_path": "/path/to/source_image1.jpg", "caption": "The object changes to red."}`. The `image_path` should point to the images after the change, and `control_path` should point to the starting images. 

For the dataset configuration, see [here](./dataset_config.md#sample-for-image-dataset-with-control-images) and [here](./dataset_config.md#framepack-one-frame-training). There are also examples for kisekaeichi and 1f-mc settings.

For single frame training, specify `--one_frame` in `fpack_cache_latents.py` to create the cache. You can also use `--one_frame_no_2x` and `--one_frame_no_4x` options, which have the same meaning as `no_2x` and `no_4x` during inference. It is recommended to set these options to match the inference settings.

If you change whether to use one frame training or these options, please overwrite the existing cache without specifying `--skip_existing`.

Specify `--one_frame` in `fpack_train_network.py` to change the inference method during sample generation. 

The optimal training settings are currently unknown. Feedback is welcome.

### Example of prompt file description for sample generation

The command line options `--one_frame_inference` corresponds to `--of`, and `--control_image_path` corresponds to `--ci`.

Note that `--ci` can be specified multiple times, but `--control_image_path` is specified as `--control_image_path img1.png img2.png`, while `--ci` is specified as `--ci img1.png --ci img2.png`.

Normal single frame training:
```
The girl wears a school uniform. --i path/to/start.png --ci path/to/start.png --of no_2x,no_4x,target_index=1,control_index=0 --d 1111 --f 1 --s 10 --fs 7 --d 1234 --w 384 --h 576
```

Kisekaeichi training:
```
The girl wears a school uniform. --i path/to/start_with_alpha.png --ci path/to/ref_with_alpha.png --ci path/to/start_with_alpha.png --of no_post,no_2x,no_4x,target_index=5,control_index=0;10 --d 1111 --f 1 --s 10 --fs 7 --d 1234 --w 384 --h 576
```

<details>
<summary>日本語</summary>

**この機能は実験的なものです。** 1フレーム推論と同様の方法で学習を行います。

データセットは画像データセットである必要があります。キャプションファイルを用いる場合は、`control_directory`を追加で指定し、そのディレクトリに**開始画像**を格納してください。`image_directory`には変化後の画像を格納します。両者のファイル名は一致させる必要があります。キャプションファイルは`image_directory`に格納してください。

JSONLファイルを用いる場合は、`{"image_path": "/path/to/target_image1.jpg", "control_path": "/path/to/source_image1.jpg", "caption": "The object changes to red"}`のように指定してください。`image_path`は変化後の画像、`control_path`は開始画像を指定します。

データセットの設定については、[こちら](./dataset_config.md#sample-for-image-dataset-with-control-images)と[こちら](./dataset_config.md#framepack-one-frame-training)も参照してください。kisekaeichiと1f-mcの設定例もそちらにあります。

1フレーム学習時は、`fpack_cache_latents.py`に`--one_frame`を指定してキャッシュを作成してください。また`--one_frame_no_2x`と`--one_frame_no_4x`オプションも利用可能です。推論時の`no_2x`、`no_4x`と同じ意味を持ちますので、推論時と同じ設定にすることをお勧めします。

1フレーム学習か否かを変更する場合、またこれらのオプションを変更する場合は、`--skip_existing`を指定せずに既存のキャッシュを上書きしてください。

また、`fpack_train_network.py`に`--one_frame`を指定してサンプル画像生成時の推論方法を変更してください。

最適な学習設定は今のところ不明です。フィードバックを歓迎します。

**サンプル生成のプロンプトファイル記述例**

コマンドラインオプション`--one_frame_inference`に相当する `--of`と、`--control_image_path`に相当する`--ci`が用意されています。

※ `--ci`は複数指定可能ですが、`--control_image_path`は`--control_image_path img1.png img2.png`のようにスペースで区切るのに対して、`--ci`は`--ci img1.png --ci img2.png`のように指定するので注意してください。

通常の1フレーム学習:
```
The girl wears a school uniform. --i path/to/start.png --ci path/to/start.png --of no_2x,no_4x,target_index=1,control_index=0 --d 1111 --f 1 --s 10 --fs 7 --d 1234 --w 384 --h 576
```

kisekaeichi方式:
```
The girl wears a school uniform. --i path/to/start_with_alpha.png --ci path/to/ref_with_alpha.png --ci path/to/start_with_alpha.png --of no_post,no_2x,no_4x,target_index=5,control_index=0;10 --d 1111 --f 1 --s 10 --fs 7 --d 1234 --w 384 --h 576
```

</details>

## One (single) Frame Inference / 1フレーム推論

**This feature is highly experimental** and not officially supported. It is intended for users who want to explore the potential of FramePack for one frame inference, which is not a standard feature of the model.

This script also allows for one frame inference, which is not an official feature of FramePack but rather a custom implementation.

Theoretically, it generates an image after a specified time from the starting image, following the prompt. This means that, although limited, it allows for natural language-based image editing.

To perform one frame inference, specify some option in the `--one_frame_inference` option. Here is an example:

```bash
--video_sections 1 --output_type latent_images --one_frame_inference default --image_path start_image.png --control_image_path start_image.png
```

The `--image_path` is used to obtain the SIGCLIP features for one frame inference. Normally, you should specify the starting image. The `--control_image_path` is newly used to specify the control image, but for normal one frame inference, you should also specify the starting image.

The `--one_frame_inference` option is recommended to be set to `default` or `no_2x,no_4x`. If you specify `--output_type` as `latent_images`, both the latent and image will be saved.

You can specify the following strings in the `--one_frame_inference` option, separated by commas:

-   `no_2x`: Generates without passing clean latents 2x with zero vectors to the model. Slightly improves generation speed. The impact on generation results is unknown.
-   `no_4x`: Generates without passing clean latents 4x with zero vectors to the model. Slightly improves generation speed. The impact on generation results is unknown.
-   `no_post`: Generates without passing clean latents post with zero vectors to the model. Improves generation speed by about 20%, but may result in unstable generation. 
-   `target_index=<integer>`: Specifies the index of the image to be generated. The default is the last frame (i.e., `latent_window_size`).

For example, you can use `--one_frame_inference default` to pass clean latents 2x, clean latents 4x, and post to the model. `--one_frame_inference no_2x,no_4x` if you want to skip passing clean latents 2x and 4x to the model. `--one_frame_inference target_index=9` can be used to specify the target index for the generated image.

The `--one_frame_inference` option also supports advanced inference, which is described in the next section. This option allows for more detailed control using additional parameters like `target_index` and `control_index` within this option.

Normally, specify `--video_sections 1` to indicate only one section (one image).

Increasing `target_index` from the default of 9 may result in larger changes. It has been confirmed that generation can be performed without breaking up to around 40.

The `--one_frame_auto_resize` option has been added to automatically adjust the image size based on the control image size when `--one_frame_inference` is specified. If this option is enabled, the image size will be adjusted to the nearest bucket size with the specified width\*height, based on the control image size while maintaining the aspect ratio. This can be useful when the multiple generation of images with different sizes is required.

The `--end_image_path` is ignored for one frame inference.

<details>
<summary>日本語</summary>

**この機能は非常に実験的であり**、公式にはサポートされていません。FramePackを使用して1フレーム推論の可能性を試したいユーザーに向けたものです。

このスクリプトでは、単一画像の推論を行うこともできます。FramePack公式の機能ではなく、独自の実装です。

理論的には、開始画像から、プロンプトに従い、指定時間経過後の画像を生成します。つまり制限付きですが自然言語による画像編集を行うことができます。

単一画像推論を行うには`--one_frame_inference`オプションに、何らかのオプションを指定してください。記述例は以下の通りです。

```bash
--video_sections 1 --output_type latent_images --one_frame_inference default --image_path start_image.png --control_image_path start_image.png
```

`--image_path`は、1フレーム推論ではSIGCLIPの特徴量を取得するために用いられます。通常は開始画像を指定してください。`--control_image_path`は新しく追加された引数で、制御用画像を指定するために用いられますが、通常は開始画像を指定してください。

`--one_frame_inference`のオプションは、`default`または `no_2x,no_4x`を推奨します。`--output_type`に`latent_images`を指定するとlatentと画像の両方が保存されます。

`--one_frame_inference`のオプションには、カンマ区切りで以下のオプションを任意個数指定できます。

- `no_2x`: ゼロベクトルの clean latents 2xをモデルに渡さずに生成します。わずかに生成速度が向上します。生成結果への影響は不明です。
- `no_4x`: ゼロベクトルの clean latents 4xをモデルに渡さずに生成します。わずかに生成速度が向上します。生成結果への影響は不明です。
-  `no_post`: ゼロベクトルの clean latents の post を渡さずに生成します。生成速度が20%程度向上しますが、生成結果が不安定になる場合があります。
- `target_index=<整数>`: 生成する画像のindexを指定します。デフォルトは最後のフレームです（=latent_window_size）。

たとえば、`--one_frame_inference default`を使用すると、clean latents 2x、clean latents 4x、postをモデルに渡します。`--one_frame_inference no_2x,no_4x`を使用すると、clean latents 2xと4xをモデルに渡すのをスキップします。`--one_frame_inference target_index=9`を使用して、生成する画像のターゲットインデックスを指定できます。

後述の高度な推論では、このオプション内で `target_index`、`control_index` といった追加のパラメータを指定して、より詳細な制御が可能です。

clean latents 2x、clean latents 4x、postをモデルに渡す場合でも値はゼロベクトルですが、値を渡すか否かで結果は変わります。特に`no_post`を指定すると、`latent_window_size`を大きくしたときに生成結果が不安定になる場合があります。

通常は`--video_sections 1` として1セクションのみ（画像1枚）を指定してください。

`target_index` をデフォルトの9から大きくすると、変化量が大きくなる可能性があります。40程度までは破綻なく生成されることを確認しています。

`--one_frame_auto_resize`オプションが追加されました。`--one_frame_inference`を指定した場合に、制御用画像のサイズに基づいて自動的に画像サイズを調整します。このオプションを有効にすると、画像サイズは、アスペクト比を維持しつつ制御用画像のサイズを基準に、指定された幅\*高さの最も近いバケットサイズに調整されます。異なるサイズの画像を複数生成する必要がある場合に便利です。

`--end_image_path`は無視されます。

</details>

## kisekaeichi method (Post Reference Options) and 1f-mc (Multi-Control) / kisekaeichi方式（ポスト参照オプション）と1f-mc（マルチコントロール）

The `kisekaeichi` method was proposed by furusu. The `1f-mc` method was proposed by mattyamonaca in pull request [#304](https://github.com/kohya-ss/musubi-tuner/pull/304). 

In this repository, these methods have been integrated and can be specified with the `--one_frame_inference` option. This allows for specifying any number of control images as clean latents, along with indices. This means you can specify multiple starting images and multiple clean latent posts. Additionally, masks can be applied to each image.

It is expected to work only with FramePack (non-F1 model) and not with F1 models.

The following options have been added to `--one_frame_inference`. These can be used in conjunction with existing flags like `target_index`, `no_post`, `no_2x`, and `no_4x`.

-   `control_index=<integer_or_semicolon_separated_integers>`: Specifies the index(es) of the clean latent for the control image(s). You must specify the same number of indices as the number of control images specified with `--control_image_path`.

Additionally, the following command-line options have been added. These arguments are only valid when `--one_frame_inference` is specified.

-   `--control_image_path <path1> [<path2> ...]` : Specifies the path(s) to control (reference) image(s) for one frame inference. Provide one or more paths separated by spaces. Images with an alpha channel can be specified. If an alpha channel is present, it is used as a mask for the clean latent.
-   `--control_image_mask_path <path1> [<path2> ...]` : Specifies the path(s) to grayscale mask(s) to be applied to the control image(s). Provide one or more paths separated by spaces. Each mask is applied to the corresponding control image. The 255 areas are referenced, while the 0 areas are ignored.

**Example of specifying kisekaeichi:**

The kisekaeichi method works without training, but using a dedicated LoRA may yield better results.

```bash
--video_sections 1 --output_type latent_images --image_path start_image.png --control_image_path start_image.png clean_latent_post_image.png \
--one_frame_inference target_index=1,control_index=0;10,no_post,no_2x,no_4x --control_image_mask_path ctrl_mask1.png ctrl_mask2.png
```

In this example, `start_image.png` (for `clean_latent_pre`) and `clean_latent_post_image.png` (for `clean_latent_post`) are the reference images. The `target_index` specifies the index of the generated image. The `control_index` specifies the clean latent index for each control image, so it will be `0;10`. The masks for the control images are specified with `--control_image_mask_path`.

The optimal values for `target_index` and `control_index` are unknown. The `target_index` should be specified as 1 or higher. The `control_index` should be set to an appropriate value relative to `latent_window_size`. Specifying 1 for `target_index` results in less change from the starting image, but may introduce noise. Specifying 9 or 13 may reduce noise but result in larger changes from the original image.

The `control_index` should be larger than `target_index`. Typically, it is set to `10`, but larger values (e.g., around `13-16`) may also work.

Sample images and command lines for reproduction are as follows:

```bash
python fpack_generate_video.py --video_size 832 480 --video_sections 1 --infer_steps 25 \
    --prompt "The girl in a school blazer in a classroom." --save_path path/to/output --output_type latent_images \
    --dit path/to/dit --vae path/to/vae --text_encoder1 path/to/text_encoder1 --text_encoder2 path/to/text_encoder2 \
    --image_encoder path/to/image_encoder --attn_mode sdpa --vae_spatial_tile_sample_min_size 128 --vae_chunk_size 32 \
    --image_path path/to/kisekaeichi_start.png --control_image_path path/to/kisekaeichi_start.png path/to/kisekaeichi_ref.png 
    --one_frame_inference target_index=1,control_index=0;10,no_2x,no_4x,no_post 
    --control_image_mask_path path/to/kisekaeichi_start_mask.png path/to/kisekaeichi_ref_mask.png --seed 1234
```

Specify `--fp8_scaled` and `--blocks_to_swap` options according to your VRAM capacity.

- [kisekaeichi_start.png](./kisekaeichi_start.png)
- [kisekaeichi_ref.png](./kisekaeichi_ref.png)
- [kisekaeichi_start_mask.png](./kisekaeichi_start_mask.png)
- [kisekaeichi_ref_mask.png](./kisekaeichi_ref_mask.png)

Generation result: [kisekaeichi_result.png](./kisekaeichi_result.png)


**Example of 1f-mc (Multi-Control):**

```bash 
--video_sections 1 --output_type latent_images --image_path start_image.png --control_image_path start_image.png 2nd_image.png \
--one_frame_inference target_index=9,control_index=0;1,no_2x,no_4x 
```

In this example, `start_image.png` is the starting image, and `2nd_image.png` is the reference image. The `target_index=9` specifies the index of the generated image, while `control_index=0;1` specifies the clean latent indices for each control image.

1f-mc is intended to be used in combination with a trained LoRA, so adjust `target_index` and `control_index` according to the LoRA's description.

<details>
<summary>日本語</summary>

`kisekaeichi`方式はfurusu氏により提案されました。また`1f-mc`方式はmattyamonaca氏によりPR [#304](https://github.com/kohya-ss/musubi-tuner/pull/304) で提案されました。

当リポジトリではこれらの方式を統合し、`--one_frame_inference`オプションで指定できるようにしました。これにより、任意の枚数の制御用画像を clean latentとして指定し、さらにインデックスを指定できます。つまり開始画像の複数枚指定やclean latent postの複数枚指定などが可能です。また、それぞれの画像にマスクを適用することもできます。

なお、FramePack無印のみ動作し、F1モデルでは動作しないと思われます。

`--one_frame_inference`に以下のオプションが追加されています。`target_index`、`no_post`、`no_2x`や`no_4x`など既存のフラグと併用できます。

- `control_index=<整数またはセミコロン区切りの整数>`: 制御用画像のclean latentのインデックスを指定します。`--control_image_path`で指定した制御用画像の数と同じ数のインデックスを指定してください。

またコマンドラインオプションに以下が追加されています。これらの引数は`--one_frame_inference`を指定した場合のみ有効です。

- `--control_image_path <パス1> [<パス2> ...]` : 1フレーム推論用の制御用（参照）画像のパスを1つ以上、スペース区切りで指定します。アルファチャンネルを持つ画像が指定可能です。アルファチャンネルがある場合は、clean latentへのマスクとして利用されます。
- `--control_image_mask_path <パス1> [<パス2> ...]` : 制御用画像に適用するグレースケールマスクのパスを1つ以上、スペース区切りで指定します。各マスクは対応する制御用画像に適用されます。255の部分が参照される部分、0の部分が無視される部分です。

**kisekaeichiの指定例**:

kisekaeichi方式は学習なしでも動作しますが、専用のLoRAを使用することで、より良い結果が得られる可能性があります。

```bash
--video_sections 1 --output_type latent_images --image_path start_image.png --control_image_path start_image.png clean_latent_post_image.png \
--one_frame_inference target_index=1,control_index=0;10,no_post,no_2x,no_4x --control_image_mask_path ctrl_mask1.png ctrl_mask2.png
```

`start_image.png`（clean_latent_preに相当）と`clean_latent_post_image.png`は参照画像（clean_latent_postに相当）です。`target_index`は生成する画像のインデックスを指定します。`control_index`はそれぞれの制御用画像のclean latent indexを指定しますので、`0;10` になります。また`--control_image_mask_path`に制御用画像に適用するマスクを指定します。

`target_index`、`control_index`の最適値は不明です。`target_index`は1以上を指定してください。`control_index`は`latent_window_size`に対して適切な値を指定してください。`target_index`に1を指定すると開始画像からの変化が少なくなりますが、ノイズが乗ったりすることが多いようです。9や13などを指定するとノイズは改善されるかもしれませんが、元の画像からの変化が大きくなります。

`control_index`は`target_index`より大きい値を指定してください。通常は`10`ですが、これ以上大きな値、たとえば`13~16程度でも動作するようです。

サンプル画像と再現のためのコマンドラインは以下のようになります。

```bash
python fpack_generate_video.py --video_size 832 480 --video_sections 1 --infer_steps 25 \
    --prompt "The girl in a school blazer in a classroom." --save_path path/to/output --output_type latent_images \
    --dit path/to/dit --vae path/to/vae --text_encoder1 path/to/text_encoder1 --text_encoder2 path/to/text_encoder2 \
    --image_encoder path/to/image_encoder --attn_mode sdpa --vae_spatial_tile_sample_min_size 128 --vae_chunk_size 32 \
    --image_path path/to/kisekaeichi_start.png --control_image_path path/to/kisekaeichi_start.png path/to/kisekaeichi_ref.png 
    --one_frame_inference target_index=1,control_index=0;10,no_2x,no_4x,no_post 
    --control_image_mask_path path/to/kisekaeichi_start_mask.png path/to/kisekaeichi_ref_mask.png --seed 1234
```

VRAM容量に応じて、`--fp8_scaled`や`--blocks_to_swap`等のオプションを調整してください。

- [kisekaeichi_start.png](./kisekaeichi_start.png)
- [kisekaeichi_ref.png](./kisekaeichi_ref.png)
- [kisekaeichi_start_mask.png](./kisekaeichi_start_mask.png)
- [kisekaeichi_ref_mask.png](./kisekaeichi_ref_mask.png)

生成結果:
- [kisekaeichi_result.png](./kisekaeichi_result.png)

**1f-mcの指定例**:

```bash 
--video_sections 1 --output_type latent_images --image_path start_image.png --control_image_path start_image.png 2nd_image.png \
--one_frame_inference target_index=9,control_index=0;1,no_2x,no_4x 
```

この例では、`start_image.png`が開始画像で、`2nd_image.png`が参照画像です。`target_index=9`は生成する画像のインデックスを指定し、`control_index=0;1`はそれぞれの制御用画像のclean latent indexを指定しています。

1f-mcは学習したLoRAと組み合わせることを想定していますので、そのLoRAの説明に従って、`target_index`や`control_index`を調整してください。

</details>