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

# other datasets can be added here. each dataset can have different configurations
```

`cache_directory` is optional, default is None to use the same directory as the image directory. However, we recommend to set the cache directory to avoid accidental sharing of the cache files between different datasets.

`num_repeats` is also available. It is optional, default is 1 (no repeat). It repeats the images (or videos) that many times to expand the dataset. For example, if `num_repeats = 2` and there are 20 images in the dataset, each image will be duplicated twice (with the same caption) to have a total of 40 images. It is useful to balance the multiple datasets with different sizes.

<details>
<summary>日本語</summary>

`cache_directory` はオプションです。デフォルトは画像ディレクトリと同じディレクトリに設定されます。ただし、異なるデータセット間でキャッシュファイルが共有されるのを防ぐために、明示的に別のキャッシュディレクトリを設定することをお勧めします。

`num_repeats` はオプションで、デフォルトは 1 です（繰り返しなし）。画像（や動画）を、その回数だけ単純に繰り返してデータセットを拡張します。たとえば`num_repeats = 2`としたとき、画像20枚のデータセットなら、各画像が2枚ずつ（同一のキャプションで）計40枚存在した場合と同じになります。異なるデータ数のデータセット間でバランスを取るために使用可能です。

resolution, caption_extension, batch_size, num_repeats, enable_bucket, bucket_no_upscale は general または datasets のどちらかに設定してください。省略時は各項目のデフォルト値が使用されます。

`[[datasets]]`以下を追加することで、他のデータセットを追加できます。各データセットには異なる設定を持てます。
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

# other datasets can be added here. each dataset can have different configurations
```

JSONL file format for metadata:

```json
{"image_path": "/path/to/image1.jpg", "caption": "A caption for image1"}
{"image_path": "/path/to/image2.jpg", "caption": "A caption for image2"}
```

<details>
<summary>日本語</summary>

resolution, batch_size, num_repeats, enable_bucket, bucket_no_upscale は general または datasets のどちらかに設定してください。省略時は各項目のデフォルト値が使用されます。

metadata jsonl ファイルを使用する場合、caption_extension は必要ありません。また、cache_directory は必須です。

キャプションによるデータセットと同様に、複数のデータセットを追加できます。各データセットには異なる設定を持てます。
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

__In HunyuanVideo and Wan2.1, the number of `target_frames` must be "N\*4+1" (N=0,1,2,...).__ Otherwise, it will be truncated to the nearest "N*4+1".

In FramePack, it is recommended to set `frame_extraction` to `full` and `max_frames` to a sufficiently large value, as it can handle longer videos. However, if the video is too long, an Out of Memory error may occur during VAE encoding. The videos in FramePack are trimmed to "N * latent_window_size * 4 + 1" frames (for example, 37, 73, 109... if `latent_window_size` is 9).

If the `source_fps` is specified, the videos in the directory are considered to be at this frame rate, and some frames will be skipped to match the model's frame rate (24 for HunyuanVideo and 16 for Wan2.1). __The value must be a decimal number, for example, `30.0` instead of `30`.__ The skipping is done automatically and does not consider the content of the images. Please check if the converted data is correct using `--debug_mode video`.

If `source_fps` is not specified (default), all frames of the video will be used regardless of the video's frame rate.

<details>
<summary>日本語</summary>

共通パラメータ（resolution, caption_extension, batch_size, num_repeats, enable_bucket, bucket_no_upscale）は、generalまたはdatasetsのいずれかに設定できます。
動画固有のパラメータ（target_frames, frame_extraction, frame_stride, frame_sample, max_frames, source_fps）は、各datasetsセクションに設定する必要があります。

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

The dataset with control images is used for training the single frame training for FramePack.

The dataset configuration with caption text files is similar to the image dataset, but with an additional `control_directory` parameter.

The control images are used from the `control_directory` with the same filename (or different extension) as the image, for example, `image_dir/image1.jpg` and `control_dir/image1.png`. The images in `image_directory` should be the target images (the images to be generated during inference, the changed images). The `control_directory` should contain the starting images for inference. The captions should be stored in `image_directory`.

The metadata JSONL file format is the same as the image dataset, but with an additional `control_path` parameter.

```json
{"image_path": "/path/to/image1.jpg", "control_path": "/path/to/control1.png", "caption": "A caption for image1"}
{"image_path": "/path/to/image2.jpg", "control_path": "/path/to/control2.png", "caption": "A caption for image2"}
```

<details>
<summary>日本語</summary>
制御画像を持つデータセットです。FramePackの単一フレーム学習に使用します。

キャプションファイルを用いる場合は`control_directory`を追加で指定してください。制御用画像は、画像と同じファイル名（または拡張子のみが異なるファイル名）の、`control_directory`にある画像が使用されます（例：`image_dir/image1.jpg`と`control_dir/image1.png`）。`image_directory`の画像は学習対象の画像（推論時に生成する画像、変化後の画像）としてください。`control_directory`には推論時の開始画像を格納してください。キャプションは`image_directory`へ格納してください。

メタデータJSONLファイルを使用する場合は、`control_path`を追加してください。
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

frame_extraction = "head" # optional, "head" or "chunk", "slide", "uniform". Default is "head"
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



<!--

```toml
# general configurations
[general]
resolution = [960, 544] # optional, [W, H], default is None. This is the default resolution for all datasets
caption_extension = ".txt" # optional, default is None. This is the default caption extension for all datasets
batch_size = 1 # optional, default is 1. This is the default batch size for all datasets
enable_bucket = true # optional, default is false. Enable bucketing for datasets
bucket_no_upscale = false # optional, default is false. Disable upscaling for bucketing. Ignored if enable_bucket is false

# sample image dataset with caption text files
[[datasets]]
image_directory = "/path/to/image_dir"
caption_extension = ".txt" # required for caption text files, if general caption extension is not set
resolution = [960, 544] # required if general resolution is not set
batch_size = 4 # optional, overwrite the default batch size
enable_bucket = false # optional, overwrite the default bucketing setting
bucket_no_upscale = true # optional, overwrite the default bucketing setting
cache_directory = "/path/to/cache_directory" # optional, default is None to use the same directory as the image directory. NOTE: caching is always enabled

# sample image dataset with metadata **jsonl** file
[[datasets]]
image_jsonl_file = "/path/to/metadata.jsonl" # includes pairs of image files and captions
resolution = [960, 544] # required if general resolution is not set
cache_directory = "/path/to/cache_directory" # required for metadata jsonl file
# caption_extension is not required for metadata jsonl file
# batch_size, enable_bucket, bucket_no_upscale are also available for metadata jsonl file

# sample video dataset with caption text files
[[datasets]]
video_directory = "/path/to/video_dir"
caption_extension = ".txt" # required for caption text files, if general caption extension is not set
resolution = [960, 544] # required if general resolution is not set
target_frames = [1, 25, 79] # required for video dataset. list of video lengths to extract frames. each element must be N*4+1 (N=0,1,2,...)
frame_extraction = "head" # optional, "head" or "chunk", "slide", "uniform". Default is "head"
frame_stride = 1 # optional, default is 1, available for "slide" frame extraction
frame_sample = 4 # optional, default is 1 (same as "head"), available for "uniform" frame extraction
# batch_size, enable_bucket, bucket_no_upscale, cache_directory are also available for video dataset

# sample video dataset with metadata jsonl file
[[datasets]]
video_jsonl_file = "/path/to/metadata.jsonl" # includes pairs of video files and captions
target_frames = [1, 79]
cache_directory = "/path/to/cache_directory" # required for metadata jsonl file
# frame_extraction, frame_stride, frame_sample are also available for metadata jsonl file
```

# sample image dataset with lance
[[datasets]]
image_lance_dataset = "/path/to/lance_dataset"
resolution = [960, 544] # required if general resolution is not set
# batch_size, enable_bucket, bucket_no_upscale, cache_directory are also available for lance dataset

The metadata with .json file will be supported in the near future.




-->