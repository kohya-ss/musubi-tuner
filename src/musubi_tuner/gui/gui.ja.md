[English](./gui.md) | [日本語](./gui.ja.md)

# Musubi Tuner GUI - ユーザーガイド

このガイドでは、Musubi Tuner GUIのセットアップと使用方法について説明します。Z-Image-TurboやQwen-Imageなどの画像生成モデルでLoRAを学習できます。

## 目次

1. [必要な環境](#必要な環境)
2. [必要なソフトウェアのインストール](#必要なソフトウェアのインストール)
3. [Musubi Tunerのインストール](#musubi-tunerのインストール)
4. [GUIの起動](#guiの起動)
5. [作業手順ガイド](#作業手順ガイド)
6. [各項目の説明](#各項目の説明)
7. [トラブルシューティング](#トラブルシューティング)

---

## 必要な環境

始める前に、以下を確認してください：

- NVIDIA GPU搭載のWindows PC（VRAM 12GB以上推奨、VRAM+メインRAMで64GB以上推奨）
- インターネット接続
- ComfyUIがインストール済みで、必要なモデル（VAE、Text Encoder、DiT）がダウンロード済みであること

---

## 必要なソフトウェアのインストール

### ステップ1：Pythonのインストール

Musubi TunerにはPython 3.10、3.11、または3.12が必要です。システムにインストールされていない場合は、以下の手順でインストールしてください。

1. Pythonの公式サイトにアクセス：https://www.python.org/downloads/
2. Python 3.10、3.11、または3.12の、Windows用 64ビットインストーラーをダウンロード（互換性の観点から **Python 3.12** を推奨）
3. インストーラーを実行
4. **重要**：「Install Now」をクリックする前に、**「Add Python to PATH」にチェック**を入れてください
5. インストールを完了

**インストールの確認**：コマンドプロンプトを開き（`Win + R`を押して`cmd`と入力し、Enterを押す）、以下を実行：
```
python --version
```
`Python 3.12.x`のように表示されれば成功です。

### ステップ2：Gitのインストール

GitはMusubi Tunerのソースコードをダウンロードするために必要です。システムにインストールされていない場合は、以下の手順でインストールしてください。

1. Gitのサイトにアクセス：https://git-scm.com/downloads/win
2. Windowsインストーラーをダウンロード
3. デフォルト設定のままインストーラーを実行（「Next」をクリックし続ける）
4. インストールを完了

**インストールの確認**：コマンドプロンプトで以下を実行：
```
git --version
```
`git version 2.x.x`のように表示されれば成功です。

### ステップ3：uvのインストール

uvは依存関係の管理を簡単にする最新のPythonパッケージマネージャーです。

1. 管理者のコマンドプロンプトを開く
2. 以下のコマンドを実行：
```
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```
3. 変更を反映させるため、**コマンドプロンプトを閉じて再度開く**（通常のコマンドプロンプト）

**インストールの確認**：新しいコマンドプロンプトで以下を実行：
```
uv --version
```
`uv 0.x.x`のように表示されれば成功です。

---

## Musubi Tunerのインストール

### ステップ1：ソースコードのダウンロード

1. コマンドプロンプトを開く
2. Musubi Tunerをインストールしたいフォルダに移動。例：
   ```
   cd C:\Users\YourName\Documents
   ```
3. リポジトリをクローン：
   ```
   git clone https://github.com/kohya-ss/musubi-tuner.git
   ```
4. フォルダに移動：
   ```
   cd musubi-tuner
   ```

### ステップ2：初回セットアップ（自動）

GUIを初めて実行すると、uvが自動的にPyTorchを含むすべての必要な依存関係をダウンロード・インストールします。これには数分かかる場合があります。

---

## GUIの起動

コマンドプロンプトを開き、musubi-tunerフォルダに移動して、CUDAバージョンに応じて以下のいずれかのコマンドを実行します：

### CUDA 12.4の場合（安定したバージョン）

```
uv run --extra cu124 --extra gui python src/musubi_tuner/gui/gui.py
```

### CUDA 12.8の場合（新しいGPU向け）

```
uv run --extra cu128 --extra gui python src/musubi_tuner/gui/gui.py
```

**注意**：どのCUDAバージョンを使うか分からない場合は、まず`cu124`を試してください。

起動後、GUIは以下のようなURLを表示します：
```
Running on local URL:  http://127.0.0.1:7860
```

このURLをWebブラウザで開いてGUIにアクセスします。

**ヒント**：より簡単にGUIを起動するために、バッチファイル（`.bat`）を作成できます：

1. `musubi-tuner`フォルダに`launch_gui.bat`というファイルを作成
2. 以下の内容を追加：
   ```batch
   @echo off
   cd /d "%~dp0"
   uv run --extra cu124 --extra gui python src/musubi_tuner/gui/gui.py
   pause
   ```
3. バッチファイルをダブルクリックしてGUIを起動

---

## 作業手順ガイド

GUIは上から下へ、作業を完了すべき順序で配置されています。

### ステップの概要

1. **プロジェクト設定** - プロジェクトフォルダの作成または読み込み
2. **モデル選択** - モデルアーキテクチャの選択とモデルパスの設定
3. **データセット設定** - 学習解像度とバッチサイズの設定
4. **前処理** - latentとtext encoderの事前キャッシュ
5. **学習** - 学習パラメータの設定と学習の開始
6. **後処理** - ComfyUI用にLoRAを変換（必要な場合）

---

### ステップ1：プロジェクト設定

1. **Project Directory（プロジェクトディレクトリ）**：プロジェクトフォルダのフルパスを入力（例：`C:\MyProjects\my_lora_project`）
2. **「Initialize / Load Project」** をクリック

これにより：
- プロジェクトフォルダが存在しない場合は作成されます
- 学習画像用の`training`サブフォルダが作成されます
- 以前使用したプロジェクトの場合、設定が読み込まれます

**初期化後**、`training`フォルダに学習データを配置してください：
- 画像ファイル（`.jpg`、`.png`など）
- キャプションファイル（画像と同じファイル名で、拡張子は`.txt`）

例：
```
my_lora_project/
  training/
    image001.jpg
    image001.txt
    image002.png
    image002.txt
```

---

### ステップ2：モデル選択

1. **Model Architecture（モデルアーキテクチャ）**：学習したいモデルを選択
   - `Z-Image-Turbo` - 高速な学習、BaseモデルがリリースされていないためLoRA学習がやや不安定
   - `Qwen-Image` - より高品質、VRAMをより多く使用

2. **VRAM Size（VRAMサイズ）**：GPUのVRAMサイズを選択
   - バッチサイズやブロックスワップなどの推奨設定に影響します

3. **ComfyUI Models Directory（ComfyUIモデルディレクトリ）**：ComfyUIの`models`フォルダのパスを入力
   - 例：`C:\ComfyUI\models`
   - このフォルダには`vae`、`text_encoders`、`diffusion_models`サブフォルダが含まれている必要があります
   - 必要なモデルは[こちら](#使用するモデル一覧)を参考にしてください

4. **「Validate ComfyUI Models Directory」** をクリックしてフォルダ構造を確認

---

### ステップ3：データセット設定

1. **「Set Recommended Resolution & Batch Size」** をクリックして、選択したモデルとVRAMに応じた推奨値を自動入力
2. 必要に応じて調整：
   - **Resolution（解像度）Width/Height**：学習画像の解像度
   - **Batch Size（バッチサイズ）**：一度に処理する画像数（大きいほど高速だがVRAMを多く使用）
3. **「Generate Dataset Config」** をクリックして設定ファイルを作成

生成された設定はボタン下のプレビューエリアに表示されます。

---

### ステップ4：前処理

学習前に、latentとtext encoderの出力をキャッシュする必要があります。これにより、画像とキャプションがモデルが使用できる形式に変換されます。

1. **「Set Default Paths」** をクリックして、ComfyUIディレクトリに基づいてモデルパスを自動入力
2. パスが正しいことを確認：
   - **VAE Path**：VAEモデルへのパス
   - **Text Encoder 1 Path**：テキストエンコーダーモデルへのパス
   - **Text Encoder 2 Path**：（一部のモデルのみ、空の場合もあります）

3. **「Cache Latents」** をクリックして完了を待つ
   - 画像をlatent空間にエンコードします
   - ログ出力で進捗を確認できます

4. **「Cache Text Encoder Outputs」** をクリックして完了を待つ
   - キャプションをembeddingにエンコードします
   - テキストエンコーダーの読み込みがあるため、初回は時間がかかる場合があります

---

### ステップ5：学習

1. **「Set Recommended Parameters」** をクリックして、モデルとVRAMに応じた学習設定を自動入力

2. **必須設定の確認**：
   - **Base Model / DiT Path**：ベースとなるdiffusionモデル（DiT）へのパス（推奨ボタンをクリックすると自動入力）
   - **Output Name**：LoRAファイルの名前（例：`my_character_lora`）

3. **基本パラメータ**（デフォルト値を使用可能）：
   - **LoRA Dim**：LoRAのランク/次元（4-32、大きいほど容量が増えるがファイルサイズも増加）
   - **Learning Rate**：学習速度（デフォルト：1e-3 (0.001)、学習が不安定な場合は1e-4程度まで減少させることを推奨）
   - **Epochs**：全画像での学習回数。デフォルトは画像数に基づいて調整されます。過学習になる場合は減少させてください。
   - **Save Every N Epochs**：チェックポイントを保存する頻度

4. **詳細パラメータ**（「Advanced Parameters」アコーディオンを展開）：
   - **Discrete Flow Shift**：デノイジングのどのステップを重視するか（モデル固有のデフォルト値を使用することを推奨）
   - **Block Swap**：モデルレイヤーをCPUにオフロード（VRAMが限られている場合に使用）
   - **Mixed Precision**：精度モード（bf16推奨）
   - **Gradient Checkpointing**：VRAM使用量を削減
   - **FP8オプション**：さらなるメモリ最適化

5. **サンプル画像生成**（オプション）：
   - **「Generate sample images during training」** を有効にして進捗を確認
   - 学習内容を表すサンプルプロンプトを入力
   - サンプル画像のサイズと生成頻度を設定

6. **「Start Training」** をクリックして開始
   - 学習進捗を表示する新しいコマンドウィンドウが開きます
   - 学習の進捗は新しいウィンドウに表示されます
   - GUIには学習が開始されたことを確認するメッセージが表示されます

---

### ステップ6：後処理（オプション）

Z-ImageのLoRAは、ComfyUIで使用するために変換が必要です。以下の手順に従ってください。

1. **「Set Default Paths」** をクリックして、出力名に基づいてパスを自動入力
2. パスを確認：
   - **Input LoRA Path**：学習済みLoRAへのパス
   - **Output ComfyUI LoRA Path**：変換後のLoRAの保存先
3. **「Convert to ComfyUI Format」** をクリック

---

## 使用するモデル一覧

### Z-Image-Turbo

text-encodersとvaeモデルファイルは、[こちら](https://huggingface.co/Comfy-Org/z_image_turbo) の `split_files` 以下の適切なディレクトリからダウンロードしてください。

| 種類 | モデルファイル |
|----------------------|--------------|
| diffusion-models        | safetensors ostris氏の[De-Turbo](https://huggingface.co/ostris/Z-Image-De-Turbo)から z_image_de_turbo_v1_bf16.safetensors を使用|
| text-encoders       | qwen_3_4b.safetensors |
| VAE                 | ae.safetensors |

### Qwen-Image

[こちら](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)の`split_files`以下の適切なディレクトリから必要なモデルファイルをダウンロードしてください。

| 種類                | モデルファイル               |
|----------------------|-----------------------------|
| diffusion-models        | qwen_image_bf16.safetensors |
| text-encoders       | qwen_2.5_vl_7b.safetensors |
| VAE | qwen_image_vae.safetensors |

---

## 各項目の説明

### プロジェクト設定

| 項目 | 説明 |
|------|------|
| Project Directory | 学習プロジェクトのルートフォルダ。すべてのデータと出力がここに保存されます。 |

### モデル設定

| 項目 | 説明 |
|------|------|
| Model Architecture | LoRAを学習するベースモデル。Z-Image-Turboは高速、Qwen-Imageはより高品質。 |
| VRAM Size | GPUのビデオメモリ。推奨バッチサイズやメモリ最適化設定に影響します。 |
| ComfyUI Models Directory | 必要なモデルファイルが含まれるComfyUIの`models`フォルダへのパス。 |

### データセット設定

| 項目 | 説明 |
|------|------|
| Resolution (W/H) | 学習解像度。画像はこのサイズにリサイズ/クロップされます。 |
| Batch Size | 同時に処理する画像数。大きいほど高速ですが、VRAMをより多く使用します。 |

### 前処理

| 項目 | 説明 |
|------|------|
| VAE Path | 画像をlatent空間にエンコードするVAEモデルファイルへのパス。 |
| Text Encoder 1 Path | メインのテキストエンコーダーモデルへのパス。 |
| Text Encoder 2 Path | セカンダリのテキストエンコーダーへのパス（モデルによって必要な場合）。 |

### 学習パラメータ

| 項目 | 説明 |
|------|------|
| Base Model / DiT Path | ベースとなるdiffusionモデル（DiT）へのパス。 |
| Output Name | 保存されるLoRAファイルのベース名（拡張子なし）。 |
| LoRA Dim | LoRAのランク/次元。大きいほど詳細をキャプチャできますが、ファイルサイズも増加。一般的な値：4、8、16、32。 |
| Learning Rate | 学習速度。大きいほど速く学習しますが、オーバーシュートする可能性があります。デフォルト：1e-3（0.001）。 |
| Epochs | 全学習画像を何回学習するか。 |
| Save Every N Epochs | チェックポイントを保存する頻度。サンプル画像の生成頻度も制御します。 |
| Discrete Flow Shift | 学習のダイナミクスに影響するflow matchingパラメータ。モデル固有のデフォルト値が推奨されます。 |
| Block Swap | CPUにオフロードするtransformerブロック数。VRAMが限られている場合に使用。大きいほどVRAM使用量が減りますが、学習が遅くなります。 |
| Mixed Precision | 浮動小数点精度。最新のGPUではbf16を推奨。 |
| Gradient Checkpointing | 一部の値を再計算することでVRAM使用量を削減。やや遅くなりますが、メモリ使用量が減少。 |
| FP8 Scaled | ベースモデルにFP8精度を使用。品質への影響を最小限に抑えてメモリを削減。 |
| FP8 LLM | テキストエンコーダー（LLM）にFP8精度を使用。さらにメモリ使用量を削減。 |
| Additional Arguments | 上級ユーザー向けの追加コマンドライン引数。 |

### サンプル画像生成

| 項目 | 説明 |
|------|------|
| Generate sample images | 学習中にサンプル画像を生成するかどうか。 |
| Sample Prompt | サンプル画像を生成するために使用するテキストプロンプト。 |
| Negative Prompt | サンプル画像で避けたい内容。 |
| Sample Width/Height | サンプル画像の解像度。 |
| Sample Every N Epochs | サンプルを生成する頻度。 |

### 後処理

| 項目 | 説明 |
|------|------|
| Input LoRA Path | 学習済みLoRAファイル（Musubi Tuner形式）へのパス。 |
| Output ComfyUI LoRA Path | 変換後のLoRA（ComfyUI形式）の保存先。 |

---

## トラブルシューティング

### 「Python is not recognized」と表示される
- インストール時に「Add Python to PATH」にチェックを入れたか確認
- このオプションを有効にしてPythonを再インストールしてみる
- または、手動でPythonをシステムのPATHに追加

### 「uv is not recognized」と表示される
- uvをインストールした後、コマンドプロンプトを閉じて再度開く
- インストールコマンドを再度実行してみる

### CUDAエラーまたはメモリ不足
- GUIでより小さいVRAMサイズを選択して、より控えめな設定を取得
- Block Swapを有効にして一部の計算をCPUにオフロード
- バッチサイズを1に減らす
- FP8オプションを有効にしてさらにメモリを節約

### 学習スクリプトがすぐエラーで終了する
- エラーメッセージを確認
- すべてのパスが正しいか確認
- 前処理（Cache LatentsとCache Text Encoder）が正常に完了したか確認

### 学習が遅い
- Block Swapが有効な場合、学習は遅くなります（VRAMが限られている場合は想定内）
- VRAMが不足し、共有VRAMが使用されると、パフォーマンスが大幅に低下します。fp8オプションの使用、Block Swapを大きくする、バッチサイズを減らすなど、メモリ使用量を減らす方法を試してください。
- GPU（CPUではなく）を使用していることを確認
- GPUドライバーが最新であることを確認

### 「Model not found」エラー
- ComfyUIモデルディレクトリが正しいか確認
- 必要なモデルがダウンロードされているか確認
- モデルのファイル名がGUIが期待するものと一致しているか確認（正確なファイル名はconfig_manager.pyを参照）

### GUIが起動しない
- 正しいディレクトリ（musubi-tunerフォルダ）にいることを確認
- 正しいuvコマンドを使用していることを確認（CUDAバージョンに注意）

---

## プロジェクトフォルダの構造

GUIを使用した後、プロジェクトフォルダは以下のようになります：

```
my_lora_project/
  training/           # 学習画像とキャプション
    image001.jpg
    image001.txt
    ...
  cache/              # 前処理済みデータ（自動作成）
    latent_cache/
    text_encoder_cache/
  models/             # 学習済みLoRAファイル（自動作成）
    my_lora.safetensors
    my_lora_comfy.safetensors
    sample/           # 学習中に生成されたサンプル画像
  logs/               # TensorBoardログ（自動作成）
  dataset_config.toml # データセット設定（自動作成）
  musubi_project.toml # GUIプロジェクト設定（自動作成）
  sample_prompt.txt   # サンプルプロンプトファイル（有効時に自動作成）
```

---

## 次のステップ

LoRAを学習した後：

1. ComfyUIで使用する必要がある場合（Z-ImageのLoRAは変換が必要）は、後処理セクションで変換
2. 変換したLoRAをComfyUIの`models/loras`フォルダにコピー
3. ComfyUIでLoRA loaderノードを使用して読み込み

より高度な学習オプションやコマンドラインでの使用方法については、`docs`フォルダ内のMusubi Tunerのメインドキュメントを参照してください。
