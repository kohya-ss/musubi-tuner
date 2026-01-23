> 📝 Click on the language section to expand / 言語をクリックして展開

## Dataset Configuration

Please create a TOML file for dataset configuration.

Image and video datasets are supported. The configuration file can include multiple datasets, either image or video datasets, with caption text files or metadata JSONL files.

The cache directory must be different for each dataset.

Each video is extracted frame by frame without additional processing and used for training. It is recommended to use videos with a frame rate of 24fps for HunyuanVideo, 16fps for Wan2.1 and 30fps for FramePack. You can check the videos that will be trained using `--debug_mode video` when caching latent (see [here](/README.md#latent-caching)).
<details>
<summary>日本語</summary>

データセットの設定を行うためのTOMLファイルを作成してください。

画像データセットと動画データセットがサポートされています。設定ファイルには、画像または動画データセットを複数含めることができます。キャプションテキストファイルまたはメタデータJSONLファイルを使用できます。

キャッシュディレクトリは、各データセットごとに異なるディレクトリである必要があります。

動画は追加のプロセスなしでフレームごとに抽出され、学習に用いられます。そのため、HunyuanVideoは24fps、Wan2.1は16fps、FramePackは30fpsのフレームレートの動画を使用することをお勧めします。latentキャッシュ時の`--debug_mode video`を使用すると、学習される動画を確認できます（[こちら](/README.ja.md#latentの事前キャッシュ)を参照）。
</details>

### Sample for Image Dataset with Caption Text Files

```toml
# resolution, caption_extension, batch_size, num_repeats, enable_bucket, bucket_no_upscale should be set in either general or datasets
# otherwise, the default values will be used for each item

# general configurations
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/path/to/image_dir"
cache_directory = "/path/to/cache_directory"
num_repeats = 1 # optional, default is 1. Number of times to repeat the dataset. Useful to balance the multiple datasets with different sizes.
# multiple_target = true # optional, default is false. Set to true for Qwen-Image-Layered training.

# other datasets can be added here. each dataset can have different configurations
```

`image_directory` is the directory containing images. The captions are stored in text files with the same filename as the image, but with the extension specified by `caption_extension` (for example, `image1.jpg` and `image1.txt`).

`cache_directory` is optional, default is None to use the same directory as the image directory. However, we recommend to set the cache directory to avoid accidental sharing of the cache files between different datasets.

`num_repeats` is also available. It is optional, default is 1 (no repeat). It repeats the images (or videos) that many times to expand the dataset. For example, if `num_repeats = 2` and there are 20 images in the dataset, each image will be duplicated twice (with the same caption) to have a total of 40 images. It is useful to balance the multiple datasets with different sizes.

For Qwen-Image-Layered training, set `multiple_target = true`. Also, in the `image_directory`, for each "image to be trained + segmentation (layer) results" combination, store the following (if `caption_extension` is `.txt`):

|Item|Example|Note|
|---|---|---|
|Caption file|`image1.txt`| |
|Image to be trained (image to be layered)|`image1.png`| |
|Segmentation (layer) result images|`image1_1.png`, `image1_2.png`, ...|Alpha channel required|

The next combination would be stored as `/path/to/layer_images/image2.txt` for caption, and `/path/to/layer_images/image2.png`, `/path/to/layer_images/image2_0.png`, `/path/to/layer_images/image2_1.png` for images.

<details>
<summary>日本語</summary>

`image_directory`は画像を含むディレクトリのパスです。キャプションは、画像と同じファイル名で、`caption_extension`で指定した拡張子のテキストファイルに格納してください（例：`image1.jpg`と`image1.txt`）。

`cache_directory` はオプションです。デフォルトは画像ディレクトリと同じディレクトリに設定されます。ただし、異なるデータセット間でキャッシュファイルが共有されるのを防ぐために、明示的に別のキャッシュディレクトリを設定することをお勧めします。

`num_repeats` はオプションで、デフォルトは 1 です（繰り返しなし）。画像（や動画）を、その回数だけ単純に繰り返してデータセットを拡張します。たとえば`num_repeats = 2`としたとき、画像20枚のデータセットなら、各画像が2枚ずつ（同一のキャプションで）計40枚存在した場合と同じになります。異なるデータ数のデータセット間でバランスを取るために使用可能です。

resolution, caption_extension, batch_size, num_repeats, enable_bucket, bucket_no_upscale は general または datasets のどちらかに設定してください。省略時は各項目のデフォルト値が使用されます。

`[[datasets]]`以下を追加することで、他のデータセットを追加できます。各データセットには異なる設定を持てます。

Qwen-Image-Layeredの学習の場合、`multiple_target = true`を設定してください。また、`image_directory`内に、それぞれの「学習する画像＋分割結果」組み合わせごとに、以下を格納してください（`caption_extension`が`.txt`の場合）。

|項目|例|備考|
|---|---|---|
|キャプションファイル|`image1.txt`| |
|学習する画像（分割対象の画像）|`image1.png`| |
|分割結果のレイヤー画像群|`image1_1.png`, `image1_2.png`, ...|アルファチャンネル必須|

次の組み合わせは、`/path/to/layer_images/image2.txt`に対して、`/path/to/layer_images/image2.png`, `/path/to/layer_images/image2_0.png`, `/path/to/layer_images/image2_1.png`のように格納します。

</details>

### Sample for Image Dataset with Metadata JSONL File

```toml
# resolution, batch_size, num_repeats, enable_bucket, bucket_no_upscale should be set in either general or datasets
# caption_extension is not required for metadata jsonl file
# cache_directory is required for each dataset with metadata jsonl file

# general configurations
[general]
resolution = [960, 544]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_jsonl_file = "/path/to/metadata.jsonl"
cache_directory = "/path/to/cache_directory" # required for metadata jsonl file
num_repeats = 1 # optional, default is 1. Same as above.
# multiple_target = true # optional, default is false. Set to true for Qwen-Image-Layered training. 

# other datasets can be added here. each dataset can have different configurations
```

JSONL file format for metadata:

```json
{"image_path": "/path/to/image1.jpg", "caption": "A caption for image1"}
{"image_path": "/path/to/image2.jpg", "caption": "A caption for image2"}
```

For Qwen-Image-Layered training, set `multiple_target = true`. Also, in the metadata JSONL file, for each "image to be trained + segmentation (layer) results" combination, specify the image paths with numbered attributes like `image_path_0`, `image_path_1`, etc.

```json
{"image_path_0": "/path/to/image1_base.png", "image_path_1": "/path/to/image1_layer1.png", "image_path_2": "/path/to/image1_layer2.png", "caption": "A caption for image1"}
{"image_path_0": "/path/to/image2_base.png", "image_path_1": "/path/to/image2_layer1.png", "image_path_2": "/path/to/image2_layer2.png", "caption": "A caption for image2"}
```

<details>
<summary>日本語</summary>

resolution, batch_size, num_repeats, enable_bucket, bucket_no_upscale は general または datasets のどちらかに設定してください。省略時は各項目のデフォルト値が使用されます。

metadata jsonl ファイルを使用する場合、caption_extension は必要ありません。また、cache_directory は必須です。

キャプションによるデータセットと同様に、複数のデータセットを追加できます。各データセットには異なる設定を持てます。

Qwen-Image-Layeredの学習の場合、`multiple_target = true`を設定してください。また、metadata jsonl ファイル内で、各画像に対して複数のターゲット画像を指定する場合は、`image_path_0`, `image_path_1`のように数字を付与してください。

</details>


### Sample for Video Dataset with Caption Text Files

```toml
# Common parameters (resolution, caption_extension, batch_size, num_repeats, enable_bucket, bucket_no_upscale) 
# can be set in either general or datasets sections
# Video-specific parameters (target_frames, frame_extraction, frame_stride, frame_sample, max_frames, source_fps)
# must be set in each datasets section

# general configurations
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "/path/to/video_dir"
cache_directory = "/path/to/cache_directory" # recommended to set cache directory
target_frames = [1, 25, 45]
frame_extraction = "head"
source_fps = 30.0 # optional, source fps for videos in the directory, decimal number

[[datasets]]
video_directory = "/path/to/video_dir2"
cache_directory = "/path/to/cache_directory2" # recommended to set cache directory
frame_extraction = "full"
max_frames = 45

# other datasets can be added here. each dataset can have different configurations
```

`video_directory` is the directory containing videos. The captions are stored in text files with the same filename as the video, but with the extension specified by `caption_extension` (for example, `video1.mp4` and `video1.txt`).

__In HunyuanVideo and Wan2.1, the number of `target_frames` must be "N\*4+1" (N=0,1,2,...).__ Otherwise, it will be truncated to the nearest "N*4+1".

In FramePack, it is recommended to set `frame_extraction` to `full` and `max_frames` to a sufficiently large value, as it can handle longer videos. However, if the video is too long, an Out of Memory error may occur during VAE encoding. The videos in FramePack are trimmed to "N * latent_window_size * 4 + 1" frames (for example, 37, 73, 109... if `latent_window_size` is 9).

If the `source_fps` is specified, the videos in the directory are considered to be at this frame rate, and some frames will be skipped to match the model's frame rate (24 for HunyuanVideo and 16 for Wan2.1). __The value must be a decimal number, for example, `30.0` instead of `30`.__ The skipping is done automatically and does not consider the content of the images. Please check if the converted data is correct using `--debug_mode video`.

If `source_fps` is not specified (default), all frames of the video will be used regardless of the video's frame rate.

<details>
<summary>日本語</summary>

共通パラメータ（resolution, caption_extension, batch_size, num_repeats, enable_bucket, bucket_no_upscale）は、generalまたはdatasetsのいずれかに設定できます。
動画固有のパラメータ（target_frames, frame_extraction, frame_stride, frame_sample, max_frames, source_fps）は、各datasetsセクションに設定する必要があります。

`video_directory`は動画を含むディレクトリのパスです。キャプションは、動画と同じファイル名で、`caption_extension`で指定した拡張子のテキストファイルに格納してください（例：`video1.mp4`と`video1.txt`）。

__HunyuanVideoおよびWan2.1では、target_framesの数値は「N\*4+1」である必要があります。__ これ以外の値の場合は、最も近いN\*4+1の値に切り捨てられます。

FramePackでも同様ですが、FramePackでは動画が長くても学習可能なため、 `frame_extraction`に`full` を指定し、`max_frames`を十分に大きな値に設定することをお勧めします。ただし、あまりにも長すぎるとVAEのencodeでOut of Memoryエラーが発生する可能性があります。FramePackの動画は、「N * latent_window_size * 4 + 1」フレームにトリミングされます（latent_window_sizeが9の場合、37、73、109……）。

`source_fps`を指定した場合、ディレクトリ内の動画をこのフレームレートとみなして、モデルのフレームレートにあうようにいくつかのフレームをスキップします（HunyuanVideoは24、Wan2.1は16）。__小数点を含む数値で指定してください。__ 例：`30`ではなく`30.0`。スキップは機械的に行われ、画像の内容は考慮しません。変換後のデータが正しいか、`--debug_mode video`で確認してください。

`source_fps`を指定しない場合、動画のフレームは（動画自体のフレームレートに関係なく）すべて使用されます。

他の注意事項は画像データセットと同様です。
</details>

### Sample for Video Dataset with Metadata JSONL File

```toml
# Common parameters (resolution, caption_extension, batch_size, num_repeats, enable_bucket, bucket_no_upscale) 
# can be set in either general or datasets sections
# Video-specific parameters (target_frames, frame_extraction, frame_stride, frame_sample, max_frames, source_fps)
# must be set in each datasets section

# caption_extension is not required for metadata jsonl file
# cache_directory is required for each dataset with metadata jsonl file

# general configurations
[general]
resolution = [960, 544]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_jsonl_file = "/path/to/metadata.jsonl"
target_frames = [1, 25, 45]
frame_extraction = "head"
cache_directory = "/path/to/cache_directory_head"
source_fps = 30.0 # optional, source fps for videos in the jsonl file
# same metadata jsonl file can be used for multiple datasets
[[datasets]]
video_jsonl_file = "/path/to/metadata.jsonl"
target_frames = [1]
frame_stride = 10
cache_directory = "/path/to/cache_directory_stride"

# other datasets can be added here. each dataset can have different configurations
```

JSONL file format for metadata:

```json
{"video_path": "/path/to/video1.mp4", "caption": "A caption for video1"}
{"video_path": "/path/to/video2.mp4", "caption": "A caption for video2"}
```

`video_path` can be a directory containing multiple images.

<details>
<summary>日本語</summary>
metadata jsonl ファイルを使用する場合、caption_extension は必要ありません。また、cache_directory は必須です。

`video_path`は、複数の画像を含むディレクトリのパスでも構いません。

他の注意事項は今までのデータセットと同様です。
</details>

### frame_extraction Options

- `head`: Extract the first N frames from the video.
- `chunk`: Extract frames by splitting the video into chunks of N frames.
- `slide`: Extract frames from the video with a stride of `frame_stride`.
- `uniform`: Extract `frame_sample` samples uniformly from the video.
- `full`: Extract all frames from the video.

In the case of `full`, the entire video is used, but it is trimmed to "N*4+1" frames. It is also trimmed to the `max_frames` if it exceeds that value. To avoid Out of Memory errors, please set `max_frames`.

The frame extraction methods other than `full` are recommended when the video contains repeated actions. `full` is recommended when each video represents a single complete motion.

For example, consider a video with 40 frames. The following diagrams illustrate each extraction:

<details>
<summary>日本語</summary>

- `head`: 動画から最初のNフレームを抽出します。
- `chunk`: 動画をNフレームずつに分割してフレームを抽出します。
- `slide`: `frame_stride`に指定したフレームごとに動画からNフレームを抽出します。
- `uniform`: 動画から一定間隔で、`frame_sample`個のNフレームを抽出します。
- `full`: 動画から全てのフレームを抽出します。

`full`の場合、各動画の全体を用いますが、「N*4+1」のフレーム数にトリミングされます。また`max_frames`を超える場合もその値にトリミングされます。Out of Memoryエラーを避けるために、`max_frames`を設定してください。

`full`以外の抽出方法は、動画が特定の動作を繰り返している場合にお勧めします。`full`はそれぞれの動画がひとつの完結したモーションの場合にお勧めします。

例えば、40フレームの動画を例とした抽出について、以下の図で説明します。
</details>

```
Original Video, 40 frames: x = frame, o = no frame
oooooooooooooooooooooooooooooooooooooooo

head, target_frames = [1, 13, 25] -> extract head frames:
xooooooooooooooooooooooooooooooooooooooo
xxxxxxxxxxxxxooooooooooooooooooooooooooo
xxxxxxxxxxxxxxxxxxxxxxxxxooooooooooooooo

chunk, target_frames = [13, 25] -> extract frames by splitting into chunks, into 13 and 25 frames:
xxxxxxxxxxxxxooooooooooooooooooooooooooo
oooooooooooooxxxxxxxxxxxxxoooooooooooooo
ooooooooooooooooooooooooooxxxxxxxxxxxxxo
xxxxxxxxxxxxxxxxxxxxxxxxxooooooooooooooo

NOTE: Please do not include 1 in target_frames if you are using the frame_extraction "chunk". It will make the all frames to be extracted.
注: frame_extraction "chunk" を使用する場合、target_frames に 1 を含めないでください。全てのフレームが抽出されてしまいます。

slide, target_frames = [1, 13, 25], frame_stride = 10 -> extract N frames with a stride of 10:
xooooooooooooooooooooooooooooooooooooooo
ooooooooooxooooooooooooooooooooooooooooo
ooooooooooooooooooooxooooooooooooooooooo
ooooooooooooooooooooooooooooooxooooooooo
xxxxxxxxxxxxxooooooooooooooooooooooooooo
ooooooooooxxxxxxxxxxxxxooooooooooooooooo
ooooooooooooooooooooxxxxxxxxxxxxxooooooo
xxxxxxxxxxxxxxxxxxxxxxxxxooooooooooooooo
ooooooooooxxxxxxxxxxxxxxxxxxxxxxxxxooooo

uniform, target_frames =[1, 13, 25], frame_sample = 4 -> extract `frame_sample` samples uniformly, N frames each:
xooooooooooooooooooooooooooooooooooooooo
oooooooooooooxoooooooooooooooooooooooooo
oooooooooooooooooooooooooxoooooooooooooo
ooooooooooooooooooooooooooooooooooooooox
xxxxxxxxxxxxxooooooooooooooooooooooooooo
oooooooooxxxxxxxxxxxxxoooooooooooooooooo
ooooooooooooooooooxxxxxxxxxxxxxooooooooo
oooooooooooooooooooooooooooxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxooooooooooooooo
oooooxxxxxxxxxxxxxxxxxxxxxxxxxoooooooooo
ooooooooooxxxxxxxxxxxxxxxxxxxxxxxxxooooo
oooooooooooooooxxxxxxxxxxxxxxxxxxxxxxxxx

Three Original Videos, 20, 25, 35 frames: x = frame, o = no frame

full, max_frames = 31 -> extract all frames (trimmed to the maximum length):
video1: xxxxxxxxxxxxxxxxx (trimmed to 17 frames)
video2: xxxxxxxxxxxxxxxxxxxxxxxxx (25 frames)
video3: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (trimmed to 31 frames)
```

### Sample for Image Dataset with Control Images

The dataset with control images. This is used for the one frame training for FramePack, or for FLUX.1 Kontext, FLUX.2 and Qwen-Image-Edit training.

The dataset configuration with caption text files is similar to the image dataset, but with an additional `control_directory` parameter.

The control images are used from the `control_directory` with the same filename (or different extension) as the image, for example, `image_dir/image1.jpg` and `control_dir/image1.png`. The images in `image_directory` should be the target images (the images to be generated during inference, the changed images). The `control_directory` should contain the starting images for inference. The captions should be stored in `image_directory`.

If multiple control images are specified, the filenames of the control images should be numbered (excluding the extension). For example, specify `image_dir/image1.jpg` and `control_dir/image1_0.png`, `control_dir/image1_1.png`. You can also specify the numbers with four digits, such as `image1_0000.png`, `image1_0001.png`.

The metadata JSONL file format is the same as the image dataset, but with an additional `control_path` parameter.

```json
{"image_path": "/path/to/image1.jpg", "control_path": "/path/to/control1.png", "caption": "A caption for image1"}
{"image_path": "/path/to/image2.jpg", "control_path": "/path/to/control2.png", "caption": "A caption for image2"}

If multiple control images are specified, the attribute names should be `control_path_0`, `control_path_1`, etc.

```json
{"image_path": "/path/to/image1.jpg", "control_path_0": "/path/to/control1_0.png", "control_path_1": "/path/to/control1_1.png", "caption": "A caption for image1"}
{"image_path": "/path/to/image2.jpg", "control_path_0": "/path/to/control2_0.png", "control_path_1": "/path/to/control2_1.png", "caption": "A caption for image2"}
```

The control images can also have an alpha channel. In this case, the alpha channel of the image is used as a mask for the latent. This is only for the one frame training of FramePack.

<details>
<summary>日本語</summary>

制御画像を持つデータセットです。現時点ではFramePackの単一フレーム学習、FLUX.1 Kontext、FLUX.2、Qwen-Image-Editの学習に使用します。

キャプションファイルを用いる場合は`control_directory`を追加で指定してください。制御画像は、画像と同じファイル名（または拡張子のみが異なるファイル名）の、`control_directory`にある画像が使用されます（例：`image_dir/image1.jpg`と`control_dir/image1.png`）。`image_directory`の画像は学習対象の画像（推論時に生成する画像、変化後の画像）としてください。`control_directory`には推論時の開始画像を格納してください。キャプションは`image_directory`へ格納してください。

複数枚の制御画像が指定可能です。この場合、制御画像のファイル名（拡張子を除く）へ数字を付与してください。例えば、`image_dir/image1.jpg`と`control_dir/image1_0.png`, `control_dir/image1_1.png`のように指定します。`image1_0000.png`, `image1_0001.png`のように数字を4桁で指定することもできます。

メタデータJSONLファイルを使用する場合は、`control_path`を追加してください。複数枚の制御画像を指定する場合は、`control_path_0`, `control_path_1`のように数字を付与してください。

FramePackの単一フレーム学習では、制御画像はアルファチャンネルを持つこともできます。この場合、画像のアルファチャンネルはlatentへのマスクとして使用されます。

</details>

### Resizing Control Images for Image Dataset / 画像データセットでの制御画像のリサイズ

By default, the control images are resized to the same size as the target images. You can change the resizing method with the following options:

- `no_resize_control`: Do not resize the control images. They will be cropped to match the rounding unit of each architecture (for example, 16 pixels).
- `control_resolution`: Resize the control images to the specified resolution. For example, specify `control_resolution = [1024, 1024]`. Aspect Ratio Bucketing will be applied.

```toml
[[datasets]]
# Image directory or metadata jsonl file as above
image_directory = "/path/to/image_dir"
control_directory = "/path/to/control_dir"
control_resolution = [1024, 1024]
no_resize_control = false
```

If both are specified, `control_resolution` is treated as the maximum resolution. That is, if the total number of pixels of the control image exceeds that of `control_resolution`, it will be resized to `control_resolution`.

The recommended resizing method for control images may vary depending on the architecture. Please refer to the section for each architecture.

The previous options `flux_kontext_no_resize_control` and `qwen_image_edit_no_resize_control` are still available, but it is recommended to use `no_resize_control`.

The `qwen_image_edit_control_resolution` is also available, but it is recommended to use `control_resolution`.


**The technical details of `no_resize_control`:**

When this option is specified, the control image is trimmed to a multiple of 16 pixels (depending on the architecture) and converted to latent and passed to the model.

Each element in the batch must have the same resolution, which is adjusted by advanced Aspect Ratio Bucketing (buckets are divided by the resolution of the target image and also the resolution of the control image).

<details>
<summary>日本語</summary>

デフォルトでは、制御画像はターゲット画像と同じサイズにリサイズされます。以下のオプションで、リサイズ方式を変更できます。

- `no_resize_control`: 制御画像をリサイズしません。アーキテクチャごとの丸め単位（16ピクセルなど）に合わせてトリミングされます。
- `control_resolution`: 制御画像を指定した解像度にリサイズします。例えば、`control_resolution = [1024, 1024]`と指定します。Aspect Ratio Bucketingが適用されます。

両方が同時に指定されると、`control_resolution`は最大解像度として扱われます。つまり、制御画像の総ピクセル数が`control_resolution`の総ピクセル数を超える場合、`control_resolution`にリサイズされます。

アーキテクチャにより、推奨の制御画像のリサイズ方法は異なります。各アーキテクチャの節を参照してください。

以前のオプション`flux_kontext_no_resize_control`と`qwen_image_edit_no_resize_control`は使用可能ですが、`no_resize_control`を使用することを推奨します。

`qwen_image_edit_control_resolution`も使用可能ですが、`control_resolution`を使用することを推奨します。

**`no_resize_control`の技術的な詳細:**

このオプションが指定された場合、制御画像は16ピクセルの倍数（アーキテクチャに依存）にトリミングされ、latentに変換されてモデルに渡されます。

バッチ内の各要素は同じ解像度である必要がありますが、ターゲット画像の解像度と制御画像の解像度の両方でバケットが分割される高度なAspect Ratio Bucketingによって調整されます。

</details>

### Sample for Video Dataset with Control Images

The dataset with control videos is used for training ControlNet models. 

The dataset configuration with caption text files is similar to the video dataset, but with an additional `control_directory` parameter. 

The control video for a video is used from the `control_directory` with the same filename (or different extension) as the video, for example, `video_dir/video1.mp4` and `control_dir/video1.mp4` or `control_dir/video1.mov`. The control video can also be a directory without an extension, for example, `video_dir/video1.mp4` and `control_dir/video1`.

```toml
[[datasets]]
video_directory = "/path/to/video_dir"
control_directory = "/path/to/control_dir" # required for dataset with control videos
cache_directory = "/path/to/cache_directory" # recommended to set cache directory
target_frames = [1, 25, 45]
frame_extraction = "head"
```

The dataset configuration with metadata JSONL file is  same as the video dataset, but metadata JSONL file must include the control video paths. The control video path can be a directory containing multiple images.

```json
{"video_path": "/path/to/video1.mp4", "control_path": "/path/to/control1.mp4", "caption": "A caption for video1"}
{"video_path": "/path/to/video2.mp4", "control_path": "/path/to/control2.mp4", "caption": "A caption for video2"}
```

<details>
<summary>日本語</summary>

制御動画を持つデータセットです。ControlNetモデルの学習に使用します。

キャプションを用いる場合のデータセット設定は動画データセットと似ていますが、`control_directory`パラメータが追加されています。上にある例を参照してください。ある動画に対する制御用動画として、動画と同じファイル名（または拡張子のみが異なるファイル名）の、`control_directory`にある動画が使用されます（例：`video_dir/video1.mp4`と`control_dir/video1.mp4`または`control_dir/video1.mov`）。また、拡張子なしのディレクトリ内の、複数枚の画像を制御用動画として使用することもできます（例：`video_dir/video1.mp4`と`control_dir/video1`）。

データセット設定でメタデータJSONLファイルを使用する場合は、動画と制御用動画のパスを含める必要があります。制御用動画のパスは、複数枚の画像を含むディレクトリのパスでも構いません。

</details>

## Architecture-specific Settings / アーキテクチャ固有の設定

The dataset configuration is shared across all architectures. However, some architectures may require additional settings or have specific requirements for the dataset.

### FramePack

For FramePack, you can set the latent window size for training. It is recommended to set it to 9 for FramePack training. The default value is 9, so you can usually omit this setting.

```toml
[[datasets]]
fp_latent_window_size = 9
```

<details>
<summary>日本語</summary>

学習時のlatent window sizeを指定できます。FramePackの学習においては、9を指定することを推奨します。省略時は9が使用されますので、通常は省略して構いません。

</details>

### FramePack One Frame Training

For the default one frame training of FramePack, you need to set the following parameters in the dataset configuration:

```toml
[[datasets]]
fp_1f_clean_indices = [0]
fp_1f_target_index = 9
fp_1f_no_post = false
```

**Advanced Settings:**

**Note that these parameters are still experimental, and the optimal values are not yet known.** The parameters may also change in the future.

`fp_1f_clean_indices` sets the `clean_indices` value passed to the FramePack model. You can specify multiple indices. `fp_1f_target_index` sets the index of the frame to be trained (generated). `fp_1f_no_post` sets whether to add a zero value as `clean_latent_post`, default is `false` (add zero value).

The number of control images should match the number of indices specified in `fp_1f_clean_indices`.

The default values mean that the first image (control image) is at index `0`, and the target image (the changed image) is at index `9`.

For training with 1f-mc, set `fp_1f_clean_indices` to `[0, 1]` and `fp_1f_target_index` to `9` (or another value). This allows you to use multiple control images to train a single generated image. The control images will be two in this case.

```toml
[[datasets]]
fp_1f_clean_indices = [0, 1]
fp_1f_target_index = 9
fp_1f_no_post = false
```

For training with kisekaeichi, set `fp_1f_clean_indices` to `[0, 10]` and `fp_1f_target_index` to `1` (or another value). This allows you to use the starting image (the image just before the generation section) and the image following the generation section (equivalent to `clean_latent_post`) to train the first image of the generated video. The control images will be two in this case. `fp_1f_no_post` should be set to `true`.

```toml
[[datasets]]
fp_1f_clean_indices = [0, 10]
fp_1f_target_index = 1
fp_1f_no_post = true
```

With `fp_1f_clean_indices` and `fp_1f_target_index`, you can specify any number of control images and any index of the target image for training.

If you set `fp_1f_no_post` to `false`, the `clean_latent_post_index` will be `1 + fp1_latent_window_size`.

You can also set the `no_2x` and `no_4x` options for cache scripts to disable the clean latents 2x and 4x.

The 2x indices are `1 + fp1_latent_window_size + 1` for two indices (usually `11, 12`), and the 4x indices are `1 + fp1_latent_window_size + 1 + 2` for sixteen indices (usually `13, 14, ..., 28`), regardless of `fp_1f_no_post` and `no_2x`, `no_4x` settings.

<details>
<summary>日本語</summary>

※ **以下のパラメータは研究中で最適値はまだ不明です。** またパラメータ自体も変更される可能性があります。

デフォルトの1フレーム学習を行う場合、`fp_1f_clean_indices`に`[0]`を、`fp_1f_target_index`に`9`（または5から15程度の値）を、`no_post`に`false`を設定してください。（記述例は英語版ドキュメントを参照、以降同じ。）

**より高度な設定：**

`fp_1f_clean_indices`は、FramePackモデルに渡される `clean_indices` の値を設定します。複数指定が可能です。`fp_1f_target_index`は、学習（生成）対象のフレームのインデックスを設定します。`fp_1f_no_post`は、`clean_latent_post` をゼロ値で追加するかどうかを設定します（デフォルトは`false`で、ゼロ値で追加します）。

制御画像の枚数は`fp_1f_clean_indices`に指定したインデックスの数とあわせてください。

デフォルトの1フレーム学習では、開始画像（制御画像）1枚をインデックス`0`、生成対象の画像（変化後の画像）をインデックス`9`に設定しています。

1f-mcの学習を行う場合は、`fp_1f_clean_indices`に `[0, 1]`を、`fp_1f_target_index`に`9`を設定してください。これにより動画の先頭の2枚の制御画像を使用して、後続の1枚の生成画像を学習します。制御画像は2枚になります。

kisekaeichiの学習を行う場合は、`fp_1f_clean_indices`に `[0, 10]`を、`fp_1f_target_index`に`1`（または他の値）を設定してください。これは、開始画像（生成セクションの直前の画像）（`clean_latent_pre`に相当）と、生成セクションに続く1枚の画像（`clean_latent_post`に相当）を使用して、生成動画の先頭の画像（`target_index=1`）を学習します。制御画像は2枚になります。`f1_1f_no_post`は`true`に設定してください。

`fp_1f_clean_indices`と`fp_1f_target_index`を応用することで、任意の枚数の制御画像を、任意のインデックスを指定して学習することが可能です。

`fp_1f_no_post`を`false`に設定すると、`clean_latent_post_index`は `1 + fp1_latent_window_size` になります。

推論時の `no_2x`、`no_4x`に対応する設定は、キャッシュスクリプトの引数で行えます。なお、2xのindexは `1 + fp1_latent_window_size + 1` からの2個（通常は`11, 12`）、4xのindexは `1 + fp1_latent_window_size + 1 + 2` からの16個になります（通常は`13, 14, ..., 28`）です。これらの値は`fp_1f_no_post`や`no_2x`, `no_4x`の設定に関わらず、常に同じです。

</details>

### FLUX.1 Kontext [dev]

The FLUX.1 Kontext dataset configuration uses an image dataset with control images. However, only one control image can be used.

`fp_1f_*` settings are not used in FLUX.1 Kontext. Masks are also not used.

If you set `no_resize_control`, it disables resizing of the control image. 

Since FLUX.1 Kontext assumes a fixed [resolution of control images](https://github.com/black-forest-labs/flux/blob/1371b2bc70ac80e1078446308dd5b9a2ebc68c87/src/flux/util.py#L584), it may be better to prepare the control images in advance to match these resolutions and use `no_resize_control`.

<details>
<summary>日本語</summary>

FLUX.1 Kontextのデータセット設定は、制御画像を持つ画像データセットを使用します。ただし、制御画像は1枚しか使用できません。

`fp_1f_*`の設定はFLUX.1 Kontextでは使用しません。またマスクも使用されません。

また、`no_resize_control`を設定すると、制御画像のリサイズを無効にします。

FLUX.1 Kontextは[制御画像の固定解像度](https://github.com/black-forest-labs/flux/blob/1371b2bc70ac80e1078446308dd5b9a2ebc68c87/src/flux/util.py#L584)を想定しているため、これらの解像度にあわせて制御画像を事前に用意し、`no_resize_control`を使用する方が良い場合があります。

</details>

### Qwen-Image-Edit and Qwen-Image-Edit-2509/2511

The Qwen-Image-Edit dataset configuration uses an image dataset with control images. However, only one control image can be used for the standard model (not `2509` or `2511`).

By default, the control image is resized to the same resolution (and aspect ratio) as the image.

If you set `no_resize_control`, it disables resizing of the control image. For example, if the image is 960x544 and the control image is 512x512, the control image will remain 512x512.

Also, you can specify the resolution of the control image separately from the training image resolution by using `control_resolution`. If you want to resize the control images the same as the official code, specify [1024,1024]. **We strongly recommend specifying this value.**

`no_resize_control` can be specified together with `control_resolution`.

If `no_resize_control` or `control_resolution` is specified, each control image can have a different resolution. The control image is resized according to the specified settings.

```toml
[[datasets]]
no_resize_control = false # optional, default is false. Disable resizing of control image
control_resolution = [1024, 1024] # optional, default is None. Specify the resolution of the control image.
```

`fp_1f_*` settings are not used in Qwen-Image-Edit.

<details>
<summary>日本語</summary>

Qwen-Image-Editのデータセット設定は、制御画像を持つ画像データセットを使用します。複数枚の制御画像も使用可能ですが、無印（`2509`または`2511`でない）モデルでは1枚のみ使用可能です。

デフォルトでは、制御画像は画像と同じ解像度（およびアスペクト比）にリサイズされます。

`no_resize_control`を設定すると、制御画像のリサイズを無効にします。たとえば、画像が960x544で制御画像が512x512の場合、制御画像は512x512のままになります。

また、`control_resolution`を使用することで、制御画像の解像度を学習画像の解像度と異なる値に指定できます。公式のコードと同じように制御画像をリサイズしたい場合は、[1024, 1024]を指定してください。**この値の指定を強く推奨します。**

`no_resize_control`と `control_resolution`は同時に指定できます。

`no_resize_control`または`control_resolution`が指定された場合、各制御画像は異なる解像度を持つことができます。制御画像は指定された設定に従ってリサイズされます。

```toml
[[datasets]]
no_resize_control = false # オプション、デフォルトはfalse。制御画像のリサイズを無効にします
control_resolution = [1024, 1024] # オプション、デフォルトはNone。制御画像の解像度を指定します
```

`fp_1f_*`の設定はQwen-Image-Editでは使用しません。

</details>

### FLUX.2

The FLUX.2 dataset configuration uses an image dataset with control images (it can also be trained without control images). Multiple control images can be used.

`fp_1f_*` settings are not used in FLUX.2.

If you set `no_resize_control`, it disables resizing of the control images. If you want to follow the official FLUX.2 inference settings, please specify this option.

You can specify the resolution of the control images separately from the training image resolution by using `control_resolution`. If you want to follow the official FLUX.2 inference settings, specify [2024, 2024] (note that it is not 2048) when there is one control image, and [1024, 1024] when there are multiple control images, together with the `no_resize_control` option.

<details>
<summary>日本語</summary>

FLUX.2のデータセット設定は、制御画像を持つ画像データセットを使用します（制御画像なしでも学習できます）。複数枚の制御画像が使用可能です。

`fp_1f_*`の設定はFLUX.2では使用しません。

`no_resize_control`を設定すると、制御画像のリサイズを無効にします。FLUX.2公式の推論時設定に準拠する場合は、このオプションを指定してください。

`control_resolution`を使用して、制御画像の解像度を学習画像の解像度と異なる値に指定できます。FLUX.2公式の推論時設定に準拠する場合は、`no_resize_control`オプションと同時に、制御画像が1枚の場合は`[2024, 2024]`（2048ではないので注意）、制御画像が複数の場合は`[1024, 1024]`を指定してください。

</details>

## Specifications

```toml
# general configurations
[general]
resolution = [960, 544] # optional, [W, H], default is [960, 544]. This is the default resolution for all datasets
caption_extension = ".txt" # optional, default is None. This is the default caption extension for all datasets
batch_size = 1 # optional, default is 1. This is the default batch size for all datasets
num_repeats = 1 # optional, default is 1. Number of times to repeat the dataset. Useful to balance the multiple datasets with different sizes.
enable_bucket = true # optional, default is false. Enable bucketing for datasets
bucket_no_upscale = false # optional, default is false. Disable upscaling for bucketing. Ignored if enable_bucket is false

### Image Dataset

# sample image dataset with caption text files
[[datasets]]
image_directory = "/path/to/image_dir"
caption_extension = ".txt" # required for caption text files, if general caption extension is not set
resolution = [960, 544] # required if general resolution is not set
batch_size = 4 # optional, overwrite the default batch size
num_repeats = 1 # optional, overwrite the default num_repeats
enable_bucket = false # optional, overwrite the default bucketing setting
bucket_no_upscale = true # optional, overwrite the default bucketing setting
cache_directory = "/path/to/cache_directory" # optional, default is None to use the same directory as the image directory. NOTE: caching is always enabled
control_directory = "/path/to/control_dir" # optional, required for dataset with control images

# sample image dataset with metadata **jsonl** file
[[datasets]]
image_jsonl_file = "/path/to/metadata.jsonl" # includes pairs of image files and captions
resolution = [960, 544] # required if general resolution is not set
cache_directory = "/path/to/cache_directory" # required for metadata jsonl file
# caption_extension is not required for metadata jsonl file
# batch_size, num_repeats, enable_bucket, bucket_no_upscale are also available for metadata jsonl file

### Video Dataset

# sample video dataset with caption text files
[[datasets]]
video_directory = "/path/to/video_dir"
caption_extension = ".txt" # required for caption text files, if general caption extension is not set
resolution = [960, 544] # required if general resolution is not set

control_directory = "/path/to/control_dir" # optional, required for dataset with control images

# following configurations must be set in each [[datasets]] section for video datasets

target_frames = [1, 25, 79] # required for video dataset. list of video lengths to extract frames. each element must be N*4+1 (N=0,1,2,...)

# NOTE: Please do not include 1 in target_frames if you are using the frame_extraction "chunk". It will make the all frames to be extracted.

frame_extraction = "head" # optional, "head" or "chunk", "full", "slide", "uniform". Default is "head"
frame_stride = 1 # optional, default is 1, available for "slide" frame extraction
frame_sample = 4 # optional, default is 1 (same as "head"), available for "uniform" frame extraction
max_frames = 129 # optional, default is 129. Maximum number of frames to extract, available for "full" frame extraction
# batch_size, num_repeats, enable_bucket, bucket_no_upscale, cache_directory are also available for video dataset

# sample video dataset with metadata jsonl file
[[datasets]]
video_jsonl_file = "/path/to/metadata.jsonl" # includes pairs of video files and captions

target_frames = [1, 79]

cache_directory = "/path/to/cache_directory" # required for metadata jsonl file
# frame_extraction, frame_stride, frame_sample, max_frames are also available for metadata jsonl file
```

<!-- 
# sample image dataset with lance
[[datasets]]
image_lance_dataset = "/path/to/lance_dataset"
resolution = [960, 544] # required if general resolution is not set
# batch_size, enable_bucket, bucket_no_upscale, cache_directory are also available for lance dataset
-->

The metadata with .json file will be supported in the near future.
