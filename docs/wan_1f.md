# Wan2.1 One Frame (Single Frame) Inference and Training / Wan2.1 1フレーム推論と学習

## Overview / 概要

This document describes the application of "One Frame Inference" found in the FramePack model to Wan2.1.

1. **Basic One Frame Inference**:
    *   Input the starting image and prompt, limiting the number of frames to generate to 1 frame. Use the Wan2.1 I2V model.
    *   Intentionally set a large value for the RoPE timestamp assigned to the generated single frame. This aims to obtain a single static image that has changed temporally and semantically according to the prompt from the starting image.
    *   However, unlike FramePack, using Wan2.1's model as is for inference results in images that are almost identical to the starting image, with noise mixed in. This seems to be due to the characteristics of Wan2.1.
    *   By additionally training a LoRA, it is possible to reflect changes according to the prompt in the generated image while also reducing noise.

2. **Intermediate Frame One Frame Inference**:
    *   Similar to the kisekaeichi method, use the FLF2V (First and Last Frame to Video) method to generate intermediate frames. Use the FLF2V model.
    *   Set the RoPE timestamp of the generated image to an intermediate value between the timestamps of the starting image and the ending image.
    *   (This is a theoretical proposal, implemented but not yet tested.)

<details>
<summary>日本語</summary>

このドキュメントでは、FramePackモデルで見いだされた「1フレーム推論」の、Wan2.1への適用について説明します。

1.  **基本的な1フレーム推論**:
    *   開始画像とプロンプトを入力とし、生成するフレーム数を1フレームに限定します。Wan2.1の I2V モデルを使用します。
    *   この際、生成する1フレームに割り当てるRoPEのタイムスタンプを意図的に大きな値に設定します。これは開始画像からプロンプトに従って時間的・意味的に変化した単一の静止画を得ることを目的としています。
    *   しかしながらFramePackと異なり、Wan2.1のモデルをそのまま利用した推論では、このように設定しても生成される画像は開始画像とほぼ同じものになり、またノイズも混ざります。これはWan2.1の特性によるもの思われます。
    *   追加でLoRAを学習することで、プロンプトに従った変化を生成画像に反映させることが可能で、かつノイズも抑えられることがわかりました。

2.  **中間フレームの1フレーム推論**:
    *   kisekaeichi方式と似た、FLF2V (First and Last Frame to Video) 方式を利用し、中間のフレームを生成します。FLF2Vモデルを使用します。
    *   生成する画像のRoPEタイムスタンプを、開始画像のタイムスタンプと終端画像のタイムスタンプの中間的な値に設定します。

</details>

## One (single) Frame Inference / 1フレーム推論

**This feature is highly experimental** and is not officially supported. It is an independent implementation, not an official feature of Wan2.1.

To perform one-frame inference, specify the `--one_frame_inference` option with `target_index` and `control_index`. In Wan2.1, it is necessary to combine this with LoRA, so please set it up similarly to LoRA training settings. The model used should also be the same.

An example description is as follows:

```bash
--output_type latent_images --image_path start_image.png --control_image_path start_image.png \
--one_frame_inference control_index=0,target_index=1
```

To perform one-frame inference for intermediate frames, specify multiple indices for `control_index` separated by semicolons. The description is as follows:

```bash
--output_type latent_images --image_path start_image.png --control_image_path start_image.png end_image.png \
--one_frame_inference control_index=0;2,target_index=1
```

When specifying `--output_type` as `latent_images`, both latent and image will be saved.

The `--image_path` is used to obtain CLIP features for one-frame inference. Usually, the starting image should be specified. The `--end_image_path` is used to obtain CLIP features for the ending image. Usually, the ending image should be specified.

The `--control_image_path` is a newly added argument to specify the control image. Usually, the starting image (and both starting and ending images for intermediate frame inference) should be specified.

The options for `--one_frame_inference` are specified as comma-separated values. Here, the index represents the RoPE timestamp.

- `target_index=<integer>`: Specifies the index of the generated image.
- `control_index=<integer or semicolon-separated integers>`: Specifies the index of the control image. Please specify the same number of indices as the number of control images specified in `--control_image_path`.

The optimal values for `target_index` and `control_index` are unknown. Please specify `target_index` as 1 or greater. For one-frame inference, specify `control_index=0`. For intermediate frame one-frame inference, specify `control_index=0;2`, where 0 and a value greater than `target_index` are specified.

<details>
<summary>日本語</summary>

**この機能は非常に実験的であり**、公式にはサポートされていません。Wan2.1公式の機能ではなく、独自の実装です。

1フレーム推論を行うには`--one_frame_inference`オプションに `target_index` と `control_index` を指定してください。Wan2.1ではLoRAとの組み合わせが必要になりますので、LoRAの学習設定と同様の設定を行ってください。使用するモデルについても同様です。

記述例は以下の通りです。

```bash
--output_type latent_images --image_path start_image.png --control_image_path start_image.png \
--one_frame_inference control_index=0,target_index=1 
```

中間フレームの1フレーム推論を行うには、`control_index`にセミコロン区切りで複数のインデックスを指定します。以下のように記述します。

```bash
--output_type latent_images --image_path start_image.png --end_image_path end_image.png \
--control_image_path start_image.png end_image.png --one_frame_inference control_index=0;2,target_index=1
```

`--output_type`に`latent_images`を指定するとlatentと画像の両方が保存されます。

`--image_path`は、1フレーム推論ではCLIPの特徴量を取得するために用いられます。通常は開始画像を指定してください。`--end_image_path`は、終了画像のCLIP特徴量を取得するために用いられます。通常は終了画像を指定してください。

`--control_image_path`は新しく追加された引数で、制御用画像を指定するために用いられます。通常は開始画像（中間フレーム推論の場合は開始画像と終了画像の両方）を指定してください。

`--one_frame_inference`のオプションには、カンマ区切りで以下のオプションを指定します。ここでindexはRoPEのタイムスタンプを表します。

- `target_index=<整数>`: 生成する画像のindexを指定します。
- `control_index=<整数またはセミコロン区切りの整数>`: 制御用画像のindexを指定します。`--control_image_path`で指定した制御用画像の数と同じ数のインデックスを指定してください。

`target_index`、`control_index`の最適値は不明です。`target_index`は1以上を指定してください。`control_index`は、1フレーム推論では`control_index=0`を指定します。中間フレームの1フレーム推論では、`control_index=0;2`のように、0と`target_index`より大きい値を指定します。

</details>

## One Frame (Single Frame) Training / 1フレーム学習

**This feature is experimental.** It performs training in a manner similar to one-frame inference.

This currently reuses the dataset settings of the FramePack model. Please refer to the [FramePack documentation](./framepack_1f.md#one-frame-single-frame-training--1フレーム学習) and the [FramePack dataset settings](../src/musubi_tuner/dataset/dataset_config.md#framepack-one-frame-training).

`fp_1f_clean_indices` corresponds to the `control_index` described below.

However, `fp_1f_no_post` is ignored in Wan2.1, and alpha masks are not yet supported.

When performing one-frame training, please create the cache by specifying `--one_frame` in `wan_cache_latents.py`. Also, specify `--one_frame` in `wan_train_network.py` to change the inference method for sample image generation.

In one-frame training, the I2V 14B model is used. Specify `--task i2v-14B` and the corresponding weights. For intermediate frame one-frame training, the FLF2V model is used. Specify `--task flf2v-14B` and the corresponding weights.

In simple experiments for intermediate frame one-frame training, using `control_index=0;2`, `target_index=1` (in dataset settings, `fp_1f_clean_indices = [0, 2]`, `fp_1f_target_index = 1`), yielded better results than `0;10` and `5`.

The optimal training settings are currently unknown. Feedback is welcome.

### Example of prompt file description for sample generation

The description is almost the same as for FramePack. The command line option `--one_frame_inference` corresponds to `--of`, and `--control_image_path` corresponds to `--ci`. `--ei` is used to specify the ending image.

Note that while `--ci` can be specified multiple times, it should be specified as `--ci img1.png --ci img2.png`, unlike `--control_image_path` which is specified as `--control_image_path img1.png img2.png`.

For normal one-frame training:
```
The girl wears a school uniform. --i path/to/start.png --ci path/to/start.png --of target_index=1,control_index=0 --d 1111 --f 1 --s 10 --fs 7 --d 1234 --w 384 --h 576
```

For intermediate frame one-frame training
```
The girl wears a school uniform. --i path/to/start.png --ei path/to/end.png --ci path/to/start.png --ci path/to/end.png --of target_index=1,control_index=0;2 --d 1111 --f 1 --s 10 --fs 7 --d 1234 --w 384 --h 576
```

<details>
<summary>日本語</summary>

**この機能は実験的なものです。** 1フレーム推論と同様の方法で学習を行います。

現在は、FramePackモデルのデータセット設定を流用しています。[FramePackのドキュメント](./framepack_1f.md#one-frame-single-frame-training--1フレーム学習)および
[FramePackのデータセット設定](../src/musubi_tuner/dataset/dataset_config.md#framepack-one-frame-training)を参照してください。

`fp_1f_clean_indices` が後述の `control_index` に相当します。

ただし、`fp_1f_no_post`はWan2.1では無視されます。またアルファ値によるマスクも未対応です。

1フレーム学習時は、`wan_cache_latents.py`に`--one_frame`を指定してキャッシュを作成してください。また、`wan_train_network.py`に`--one_frame`を指定してサンプル画像生成時の推論方法を変更してください。

1フレーム学習ではI2Vの14Bモデルを使用します。`--task i2v-14B`を指定し、該当する重みを指定してください。中間フレームの1フレーム学習では、FLF2Vモデルを使用します。`--task flf2v-14B`を指定し、該当する重みを指定してください。

中間フレーム学習の簡単な実験では、`control_index=0;2`、`target_index=1`が（データセット設定では `fp_1f_clean_indices = [0, 2]`、`fp_1f_target_index = 1`）、`0;10`および`5`よりも良い結果を得られました。

最適な学習設定は今のところ不明です。フィードバックを歓迎します。

**サンプル生成のプロンプトファイル記述例**

FramePackとほぼ同様です。コマンドラインオプション`--one_frame_inference`に相当する `--of`と、`--control_image_path`に相当する`--ci`が用意されています。`--ei`は終端画像を指定します。

※ `--control_image_path`は`--control_image_path img1.png img2.png`のようにスペースで区切るのに対して、`--ci`は`--ci img1.png --ci img2.png`のように指定するので注意してください。

通常の1フレーム学習:
```
The girl wears a school uniform. --i path/to/start.png --ci path/to/start.png --of target_index=1,control_index=0 --d 1111 --f 1 --s 10 --fs 7 --d 1234 --w 384 --h 576
```

中間フレームの1フレーム学習（開始画像と終端画像の両方を指定）:
```
The girl wears a school uniform. --i path/to/start.png --ei path/to/end.png --ci path/to/start.png --ci path/to/end.png --of target_index=1,control_index=0;2 --d 1111 --f 1 --s 10 --fs 7 --d 1234 --w 384 --h 576
```

</details>

