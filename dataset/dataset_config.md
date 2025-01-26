> 📝 Click on the language section to expand / 言語をクリックして展開

## Dataset Configuration

<details>
<summary>English</summary>

Please create a TOML file for dataset configuration.

Image and video datasets are supported. The configuration file can include multiple datasets, either image or video datasets, with caption text files or metadata JSONL files.

The cache directory must be different for each dataset.
</details>

<details>
<summary>日本語</summary>

データセットの設定を行うためのTOMLファイルを作成してください。

画像データセットと動画データセットがサポートされています。設定ファイルには、画像または動画データセットを複数含めることができます。キャプションテキストファイルまたはメタデータJSONLファイルを使用できます。

キャッシュディレクトリは、各データセットごとに異なるディレクトリである必要があります。
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

<details>
<summary>English</summary>

`cache_directory` is optional, default is None to use the same directory as the image directory. However, we recommend to set the cache directory to avoid accidental sharing of the cache files between different datasets.

`num_repeats` is also available. It is optional, default is 1 (no repeat). It repeats the images (or videos) that many times to expand the dataset. For example, if `num_repeats = 2` and there are 20 images in the dataset, each image will be duplicated twice (with the same caption) to have a total of 40 images. It is useful to balance the multiple datasets with different sizes.

</details>

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
# resolution, caption_extension, target_frames, frame_extraction, frame_stride, frame_sample, 
# batch_size, num_repeats, enable_bucket, bucket_no_upscale should be set in either general or datasets
# num_repeats is also available for video dataset, example is not shown here

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

# other datasets can be added here. each dataset can have different configurations
```

<details>
<summary>日本語</summary>

resolution, caption_extension, target_frames, frame_extraction, frame_stride, frame_sample, batch_size, num_repeats, enable_bucket, bucket_no_upscale は general または datasets のどちらかに設定してください。

他の注意事項は画像データセットと同様です。
</details>

### Sample for Video Dataset with Metadata JSONL File

```toml
# resolution, target_frames, frame_extraction, frame_stride, frame_sample, 
# batch_size, num_repeats, enable_bucket, bucket_no_upscale should be set in either general or datasets
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

<details>
<summary>日本語</summary>

resolution, target_frames, frame_extraction, frame_stride, frame_sample, batch_size, num_repeats, enable_bucket, bucket_no_upscale は general または datasets のどちらかに設定してください。

metadata jsonl ファイルを使用する場合、caption_extension は必要ありません。また、cache_directory は必須です。

他の注意事項は今までのデータセットと同様です。
</details>

### frame_extraction Options

<details>
<summary>English</summary>

- `head`: Extract the first N frames from the video.
- `chunk`: Extract frames by splitting the video into chunks of N frames.
- `slide`: Extract frames from the video with a stride of `frame_stride`.
- `uniform`: Extract `frame_sample` samples uniformly from the video.

For example, consider a video with 40 frames. The following diagrams illustrate each extraction:
</details>

<details>
<summary>日本語</summary>

- `head`: 動画から最初のNフレームを抽出します。
- `chunk`: 動画をNフレームずつに分割してフレームを抽出します。
- `slide`: `frame_stride`に指定したフレームごとに動画からNフレームを抽出します。
- `uniform`: 動画から一定間隔で、`frame_sample`個のNフレームを抽出します。

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
```

## Specifications

```toml
# general configurations
[general]
resolution = [960, 544] # optional, [W, H], default is None. This is the default resolution for all datasets
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

target_frames = [1, 25, 79] # required for video dataset. list of video lengths to extract frames. each element must be N*4+1 (N=0,1,2,...)

# NOTE: Please do not include 1 in target_frames if you are using the frame_extraction "chunk". It will make the all frames to be extracted.

frame_extraction = "head" # optional, "head" or "chunk", "slide", "uniform". Default is "head"
frame_stride = 1 # optional, default is 1, available for "slide" frame extraction
frame_sample = 4 # optional, default is 1 (same as "head"), available for "uniform" frame extraction
# batch_size, num_repeats, enable_bucket, bucket_no_upscale, cache_directory are also available for video dataset

# sample video dataset with metadata jsonl file
[[datasets]]
video_jsonl_file = "/path/to/metadata.jsonl" # includes pairs of video files and captions

target_frames = [1, 79]

cache_directory = "/path/to/cache_directory" # required for metadata jsonl file
# frame_extraction, frame_stride, frame_sample are also available for metadata jsonl file
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

────────────────────────────────────────────────────────────────────────
UPDATED DOCUMENTATION: INCLUDING A VALIDATION DATASET
────────────────────────────────────────────────────────────────────────

1. Overview of “val_datasets”  
   Just like you can have multiple datasets for training under the "[[datasets]]" key, you can also specify validation datasets under "[[val_datasets]]". The syntax and options for each validation dataset are exactly the same as for training. The script will:
   • Load and cache your validation datasets the same way it does for training.  
   • Periodically compute a validation loss across these datasets (for example, once per epoch).  
   • Log the validation loss so you can monitor for over- or under-fitting.  

2. Example TOML Configuration with Validation Datasets

Below is a minimal example that shows both training and validation datasets. Notice that “[[datasets]]” is used for training datasets, while “[[val_datasets]]” is reserved for any validation sets you want to include.

--------------------------------------------------------------------------------
[general]
caption_extension = ".txt"
batch_size        = 1
enable_bucket     = true
bucket_no_upscale = false

# PRIMARY TRAIN DATASETS
[[datasets]]
resolution       = [640, 480]
video_directory  = "path/to/training/video"
cache_directory  = "path/to/cache/training/video"
frame_extraction = "head"
target_frames    = [48]

[[datasets]]
resolution      = [640, 480]
image_directory = "path/to/training/image"
cache_directory = "path/to/cache/training/image"

# ... you can add more [[datasets]] blocks if you have more training subsets ...

# VALIDATION DATASETS
[[val_datasets]]
resolution       = [640, 480]
video_directory  = "path/to/validation/video"
cache_directory  = "path/to/cache/validation/video"
frame_extraction = "head"
target_frames    = [48]

[[val_datasets]]
resolution      = [640, 480]
image_directory = "path/to/validation/image"
cache_directory = "path/to/cache/validation/image"

# ... you can add more [[val_datasets]] blocks if you have more validation subsets ...
--------------------------------------------------------------------------------

Notes on usage:  
• The script will treat the “[[datasets]]” entries as training data and “[[val_datasets]]” as validation data. Both sets can be a mix of images or videos.  
• Each dataset or validation dataset must have a unique “cache_directory” to avoid overwriting latents or text-encoder caches.  
• All of the same parameters (resolution, caption_extension, num_repeats, batch_size, etc.) work in exactly the same way under “val_datasets” as they do under “datasets.”  

3. Running the Script with Validation  
Once you have listed your training and validation datasets, you can run the training script as normal. For example:

--------------------------------------------------------------------------------
accelerate launch hv_train_network.py \
  --dit path/to/DiT/model \
  --dataset_config path/to/config_with_val.toml \
  --max_train_epochs 20 \
  ... other arguments ...
--------------------------------------------------------------------------------

During training, the script will:  
• Load and batch the training datasets.  
• Perform training epochs, computing training loss.  
• After each epoch (or at a specified interval), it will compute the validation loss using all “val_datasets.”  
• Log the validation performance (for example, “val_loss=...”) to TensorBoard and/or WandB if logging is enabled.

────────────────────────────────────────────────────────────────────────
ADDITIONAL TIPS
────────────────────────────────────────────────────────────────────────
• If you do not wish to do any validation, simply omit the “[[val_datasets]]” sections.  
• You can add multiple validation blocks, each pointing to different image or video folders or JSONL metadata.  
• The script merges all validation datasets into a single DataLoader when it computes validation loss.

With these changes to your config file, you can systematically evaluate your model’s performance after each epoch (or some other schedule) by leveraging the “val_datasets” blocks.