# Musubi Tuner

[English](./README.md) | [日本語](./README.ja.md)

## 目次

<details>
<summary>クリックすると展開します</summary>

- [はじめに](#はじめに)
    - [スポンサー](#スポンサー)
    - [スポンサー募集のお知らせ](#スポンサー募集のお知らせ)
    - [最近の更新](#最近の更新)
    - [リリースについて](#リリースについて)
    - [AIコーディングエージェントを使用する開発者の方へ](#AIコーディングエージェントを使用する開発者の方へ)
- [概要](#概要)
    - [ハードウェア要件](#ハードウェア要件)
    - [特徴](#特徴)
- [インストール](#インストール)
    - [pipによるインストール](#pipによるインストール)
    - [uvによるインストール](#uvによるインストール)
    - [Linux/MacOS](#linuxmacos)
    - [Windows](#windows)
- [モデルのダウンロード](#モデルのダウンロード)
- [使い方](#使い方)
    - [データセット設定](#データセット設定)
    - [事前キャッシュと学習](#事前キャッシュと学習)
    - [Accelerateの設定](#Accelerateの設定)
    - [学習と推論](#学習と推論)
- [その他](#その他)
    - [SageAttentionのインストール方法](#SageAttentionのインストール方法)
    - [PyTorchのバージョンについて](#PyTorchのバージョンについて)
- [免責事項](#免責事項)
- [コントリビューションについて](#コントリビューションについて)
- [ライセンス](#ライセンス)
</details>

## はじめに

このリポジトリは、HunyuanVideo、Wan2.1/2.2、FramePack、FLUX.1 Kontext、Qwen-ImageのLoRA学習用のコマンドラインツールです。このリポジトリは非公式であり、公式のHunyuanVideo、Wan2.1/2.2、FramePack、FLUX.1 Kontext、Qwen-Imageのリポジトリとは関係ありません。

アーキテクチャ固有のドキュメントについては、以下を参照してください：
- [HunyuanVideo](./docs/hunyuan_video.md)
- [Wan2.1/2.2](./docs/wan.md)
- [FramePack](./docs/framepack.md)
- [FLUX.1 Kontext](./docs/flux_kontext.md)
- [Qwen-Image](./docs/qwen_image.md)

*リポジトリは開発中です。*

### スポンサー

このプロジェクトを支援してくださる企業・団体の皆様に深く感謝いたします。

<a href="https://aihub.co.jp/">
  <img src="./images/logo_aihub.png" alt="AiHUB株式会社" title="AiHUB株式会社" height="100px">
</a>

### スポンサー募集のお知らせ

このプロジェクトがお役に立ったなら、ご支援いただけると嬉しく思います。 [GitHub Sponsors](https://github.com/sponsors/kohya-ss/)で受け付けています。

### 最近の更新

GitHub Discussionsを有効にしました。コミュニティのQ&A、知識共有、技術情報の交換などにご利用ください。バグ報告や機能リクエストにはIssuesを、質問や経験の共有にはDiscussionsをご利用ください。[Discussionはこちら](https://github.com/kohya-ss/musubi-tuner/discussions)

- 2025/09/20
    - `qwen_image_generate_image.py` で`--from_file`での生成が動作しない不具合が修正されました。[PR #553](https://github.com/kohya-ss/musubi-tuner/pull/553) および [PR #557](https://github.com/kohya-ss/musubi-tuner/pull/557) nmfisher 氏に感謝します。
        - また同スクリプトに`--append_original_name`オプションが追加されました。編集時に元の画像のベース名を出力ファイル名に追加します。

- 2025/09/14
    - Qwen-ImageのLoRA学習で`--fp8_base`を指定し`--fp8_scaled`指定しない時にFlashAttentionまたはxformersで学習するとエラーになる不具合を修正しました。[PR #559](https://github.com/kohya-ss/musubi-tuner/pull/559)
        - なお、メモリが足りない場合以外は`--fp8_scaled`を指定することをお勧めします。

- 2025/09/13
    - `wan_generate_video.py` のFLF2V推論でマスクが誤っていた不具合が修正されました。[PR #548](https://github.com/kohya-ss/musubi-tuner/pull/548) LittleNyima 氏に感謝します。
    - `.safetensors`ファイルの読み込みを高速化しました。[PR #556](https://github.com/kohya-ss/musubi-tuner/pull/556)
        - モデルの読み込みが最大で1.5倍ほど高速化されます。

- 2025/09/08
    - ruffによるコード解析を導入しました。またコントリビューションのガイドライン（[英語版](./CONTRIBUTING.md)と[日本語版](./CONTRIBUTING.ja.md)）を追加しました。
        - [Issue #524](https://github.com/kohya-ss/musubi-tuner/issues/524) および [PR #538](https://github.com/kohya-ss/musubi-tuner/pull/538)　arledesma氏に深く感謝します。
    - ActivationのCPU offloadingを追加しました。[PR #537](https://github.com/kohya-ss/musubi-tuner/pull/537)
        - block swapと組み合わせて使用することも可能です。
        - 特に長い動画や大きなバッチサイズで学習する際に、VRAMの使用量を削減できます。block swapと組み合わせることでこれらの学習が可能になる場合があります。
        - 詳細はPRおよび[HunyuanVideoのドキュメント](./docs/hunyuan_video.md#memory-optimization)を参照してください。

- 2025/09/06
    - 新しいLRスケジューラRexを追加しました。[PR #513](https://github.com/kohya-ss/musubi-tuner/pull/513) xzuyn氏に感謝します。
        - powerを1未満に設定した Polynomial Scheduler に似ていますが、Rexは学習率の減少がより緩やかです。
        - 詳細は[高度な設定のドキュメント](./docs/advanced_config.md#rex)を参照してください。

- 2025/09/02 (update)
    - Qwen-Imageのfine tuningに対応しました。[PR #492](https://github.com/kohya-ss/musubi-tuner/pull/492)
        - LoRA学習ではなくモデル全体を学習します。詳細は[Qwen-Imageのドキュメントのfinetuningの節](./docs/qwen_image.md#finetuning)を参照してください。

- 2025/09/02
    - ruffによるコード解析を導入しました。[PR #483](https://github.com/kohya-ss/musubi-tuner/pull/483) および[PR #488](https://github.com/kohya-ss/musubi-tuner/pull/488) arledesma 氏に感謝します。
        - ruffはPythonのコード解析、整形ツールです。
    - コードの貢献をいただく際は、`ruff check`を実行してコードスタイルを確認していただけると助かります。`ruff --fix`で自動修正も可能です。
        - なおコードの整形はblackで行うか、ruffのblack互換フォーマットを使い、`line-length`を`132`に設定してください。
        - ガイドライン等をのちほど整備する予定です。

### リリースについて

Musubi Tunerの解説記事執筆や、関連ツールの開発に取り組んでくださる方々に感謝いたします。このプロジェクトは開発中のため、互換性のない変更や機能追加が起きる可能性があります。想定外の互換性問題を避けるため、参照用として[リリース](https://github.com/kohya-ss/musubi-tuner/releases)をお使いください。

最新のリリースとバージョン履歴は[リリースページ](https://github.com/kohya-ss/musubi-tuner/releases)で確認できます。

### AIコーディングエージェントを使用する開発者の方へ

このリポジトリでは、ClaudeやGeminiのようなAIエージェントが、プロジェクトの概要や構造を理解しやすくするためのエージェント向け文書（プロンプト）を用意しています。

これらを使用するためには、プロジェクトのルートディレクトリに各エージェント向けの設定ファイルを作成し、明示的に読み込む必要があります。

**セットアップ手順:**

1.  プロジェクトのルートに `CLAUDE.md` や `GEMINI.md` ファイルを作成します。
2.  `CLAUDE.md` に以下の行を追加して、リポジトリが推奨するプロンプトをインポートします（現在、両者はほぼ同じ内容です）：

    ```markdown
    @./.ai/claude.prompt.md
    ```

    Geminiの場合はこちらです：

    ```markdown
    @./.ai/gemini.prompt.md
    ```

3.  インポートした行の後に、必要な指示を適宜追加してください（例：`Always respond in Japanese.`）。

このアプローチにより、共有されたプロジェクトのコンテキストを活用しつつ、エージェントに与える指示を各ユーザーが自由に制御できます。`CLAUDE.md` と `GEMINI.md` はすでに `.gitignore` に記載されているため、リポジトリにコミットされることはありません。

## 概要

### ハードウェア要件

- VRAM: 静止画での学習は12GB以上推奨、動画での学習は24GB以上推奨。
    - *解像度等の学習設定により異なります。*12GBでは解像度 960x544 以下とし、`--blocks_to_swap`、`--fp8_llm`等の省メモリオプションを使用してください。
- メインメモリ: 64GB以上を推奨、32GB+スワップで動作するかもしれませんが、未検証です。

### 特徴

- 省メモリに特化
- Windows対応（Linuxでの動作報告もあります）
- マルチGPUには対応していません

## インストール

### pipによるインストール

Python 3.10以上を使用してください（3.10で動作確認済み）。

適当な仮想環境を作成し、ご利用のCUDAバージョンに合わせたPyTorchとtorchvisionをインストールしてください。

PyTorchはバージョン2.5.1以上を使用してください（[補足](#PyTorchのバージョンについて)）。

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

以下のコマンドを使用して、必要な依存関係をインストールします。

```bash
pip install -e .
```

オプションとして、FlashAttention、SageAttention（**推論にのみ使用できます**、インストール方法は[こちら](#SageAttentionのインストール方法)を参照）を使用できます。

また、`ascii-magic`（データセットの確認に使用）、`matplotlib`（timestepsの可視化に使用）、`tensorboard`（学習ログの記録に使用）、`prompt-toolkit`を必要に応じてインストールしてください。

`prompt-toolkit`をインストールするとWan2.1およびFramePackのinteractive modeでの編集に、自動的に使用されます。特にLinux環境でプロンプトの編集が容易になります。

```bash
pip install ascii-magic matplotlib tensorboard prompt-toolkit
```

### uvによるインストール

uvを使用してインストールすることもできますが、uvによるインストールは試験的なものです。フィードバックを歓迎します。

#### Linux/MacOS

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

表示される指示に従い、pathを設定してください。

#### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

表示される指示に従い、PATHを設定するか、この時点でシステムを再起動してください。

## モデルのダウンロード

モデルのダウンロード手順はアーキテクチャによって異なります。各アーキテクチャの詳細については、以下のドキュメントを参照してください：

- [HunyuanVideoのモデルダウンロード](./docs/hunyuan_video.md#download-the-model--モデルのダウンロード)
- [Wan2.1/2.2のモデルダウンロード](./docs/wan.md#download-the-model--モデルのダウンロード)
- [FramePackのモデルダウンロード](./docs/framepack.md#download-the-model--モデルのダウンロード)
- [FLUX.1 Kontextのモデルダウンロード](./docs/flux_kontext.md#download-the-model--モデルのダウンロード)
- [Qwen-Imageのモデルダウンロード](./docs/qwen_image.md#download-the-model--モデルのダウンロード)

## 使い方

### データセット設定

[こちら](./src/musubi_tuner/dataset/dataset_config.md)を参照してください。

### 事前キャッシュと学習

各アーキテクチャは固有の事前キャッシュと学習手順が必要です。詳細については、以下のドキュメントを参照してください：

- [HunyuanVideoの使用方法](./docs/hunyuan_video.md)
- [Wan2.1/2.2の使用方法](./docs/wan.md)
- [FramePackの使用方法](./docs/framepack.md)
- [FLUX.1 Kontextの使用方法](./docs/flux_kontext.md)
- [Qwen-Imageの使用方法](./docs/qwen_image.md)

### Accelerateの設定

`accelerate config`を実行して、Accelerateの設定を行います。それぞれの質問に、環境に応じた適切な値を選択してください（値を直接入力するか、矢印キーとエンターで選択、大文字がデフォルトなので、デフォルト値でよい場合は何も入力せずエンター）。GPU 1台での学習の場合、以下のように答えてください。

```txt
- In which compute environment are you running?: This machine
- Which type of machine are you using?: No distributed training
- Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)?[yes/NO]: NO
- Do you wish to optimize your script with torch dynamo?[yes/NO]: NO
- Do you want to use DeepSpeed? [yes/NO]: NO
- What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]: all
- Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: NO
- Do you wish to use mixed precision?: bf16
```

※場合によって ``ValueError: fp16 mixed precision requires a GPU`` というエラーが出ることがあるようです。この場合、6番目の質問（
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:``）に「0」と答えてください。（id `0`、つまり1台目のGPUが使われます。）

### 学習と推論

学習と推論の手順はアーキテクチャによって大きく異なります。詳細な手順については、対応するドキュメントを参照してください：

- [HunyuanVideoの学習と推論](./docs/hunyuan_video.md)
- [Wan2.1/2.2の学習と推論](./docs/wan.md)
- [FramePackの学習と推論](./docs/framepack.md)
- [FLUX.1 Kontextの学習と推論](./docs/flux_kontext.md)
- [Qwen-Imageの学習と推論](./docs/qwen_image.md)

高度な設定オプションや追加機能については、以下を参照してください：
- [高度な設定](./docs/advanced_config.md)
- [学習中のサンプル生成](./docs/sampling_during_training.md)
- [ツールとユーティリティ](./docs/tools.md)

## その他

### SageAttentionのインストール方法

sdbds氏によるWindows対応のSageAttentionのwheelが https://github.com/sdbds/SageAttention-for-windows で公開されています。triton をインストールし、Python、PyTorch、CUDAのバージョンが一致する場合は、[Releases](https://github.com/sdbds/SageAttention-for-windows/releases)からビルド済みwheelをダウンロードしてインストールすることが可能です。sdbds氏に感謝します。

参考までに、以下は、SageAttentionをビルドしインストールするための簡単な手順です。Microsoft Visual C++ 再頒布可能パッケージを最新にする必要があるかもしれません。

1. Pythonのバージョンに応じたtriton 3.1.0のwhellを[こちら](https://github.com/woct0rdho/triton-windows/releases/tag/v3.1.0-windows.post5)からダウンロードしてインストールします。

2. Microsoft Visual Studio 2022かBuild Tools for Visual Studio 2022を、C++のビルドができるよう設定し、インストールします。（上のRedditの投稿を参照してください）。

3. 任意のフォルダにSageAttentionのリポジトリをクローンします。
    ```shell
    git clone https://github.com/thu-ml/SageAttention.git
    ```

4. スタートメニューから Visual Studio 2022 内の `x64 Native Tools Command Prompt for VS 2022` を選択してコマンドプロンプトを開きます。

5. venvを有効にし、SageAttentionのフォルダに移動して以下のコマンドを実行します。DISTUTILSが設定されていない、のようなエラーが出た場合は `set DISTUTILS_USE_SDK=1`としてから再度実行してください。
    ```shell
    python setup.py install
    ```

以上でSageAttentionのインストールが完了です。

### PyTorchのバージョンについて

`--attn_mode`に`torch`を指定する場合、2.5.1以降のPyTorchを使用してください（それより前のバージョンでは生成される動画が真っ黒になるようです）。

古いバージョンを使う場合、xformersやSageAttentionを使用してください。

## 免責事項

このリポジトリは非公式であり、サポートされているアーキテクチャの公式リポジトリとは関係ありません。また、このリポジトリは開発中で、実験的なものです。テストおよびフィードバックを歓迎しますが、以下の点にご注意ください：

- 実際の稼働環境での動作を意図したものではありません
- 機能やAPIは予告なく変更されることがあります
- いくつもの機能が未検証です
- 動画学習機能はまだ開発中です

問題やバグについては、以下の情報とともにIssueを作成してください：

- 問題の詳細な説明
- 再現手順
- 環境の詳細（OS、GPU、VRAM、Pythonバージョンなど）
- 関連するエラーメッセージやログ

## コントリビューションについて

コントリビューションを歓迎します。 [CONTRIBUTING.md](./CONTRIBUTING.md)および[CONTRIBUTING.ja.md](./CONTRIBUTING.ja.md)をご覧ください。

## ライセンス

`hunyuan_model`ディレクトリ以下のコードは、[HunyuanVideo](https://github.com/Tencent/HunyuanVideo)のコードを一部改変して使用しているため、そちらのライセンスに従います。

`wan`ディレクトリ以下のコードは、[Wan2.1](https://github.com/Wan-Video/Wan2.1)のコードを一部改変して使用しています。ライセンスはApache License 2.0です。

`frame_pack`ディレクトリ以下のコードは、[frame_pack](https://github.com/lllyasviel/FramePack)のコードを一部改変して使用しています。ライセンスはApache License 2.0です。

他のコードはApache License 2.0に従います。一部Diffusersのコードをコピー、改変して使用しています。
