# DLSS-NR 310.8.0 训练适配

## English Summary

This is an experimental, standalone supervised-training path for DLSS-NR 310.8.0,
not a diffusion trainer or an NVIDIA DLL replacement. It provides canonical
weight conversion, default FP32 full/LoRA training, still/closed-loop inference, adapter
merge, and checked optimizer-state resume. The two dataset examples enable the
project's shared resolution buckets with a 1024 x 1024 area budget and `bucket_no_upscale` enabled.
Set `enable_bucket = false` for the original strict fixed-resolution behavior.
Each input and its target, controls, motion and masks must share the original pixel grid.

Users must provide their own weights, paired RGB data, fixed condition settings or encoded control lanes and,
for non-reset temporal frames, motion and validity masks. No weights, DLLs or game
assets are included. See [source attribution](../src/musubi_tuner/dlssnr/NOTICE.md).

Normal full/LoRA training requires pretrained canonical weights, complete
conversion provenance and a successful source round-trip, but no forward-validation
report. Optional evidence is checked only when explicitly supplied; training
eligibility does not certify native/DLL compatibility. `--development_smoke`
remains a developer-only option for incomplete provenance or random initialization.
The surrogate uses half/E4M3 forward
publications with surrogate gradients and FP32 master weights; it is not ordinary
FP32 SiLU/softmax and is not bit-identical to the DLL. Non-reentrant gradient
checkpointing and single-node DDP retain the default numerical profile.
CUDA FP16/BF16, LoRA-only FP8 frozen storage and alternative attention require
explicit `--numerics_profile train_experimental`. SDPA supports windows and
global attention; optional FlashAttention is global-only, xFormers is capability
checked, and SageAttention is inference-only. Missing or incompatible optional
extensions fail explicitly, without fallback to another named backend.
Outputs precede the DLL's Natural/Cinematic color grading.

Training writes all artifacts under `output_dir/output_name/`. A new run refuses
an existing run directory; use `--resume` for a matching checkpoint. Sample IDs
must be portable filenames, unique without regard to case. The commands and
manifest contract below apply to both full and LoRA training.

Use `--dataset_config` for a **dataset-only TOML** with `[general]` and one or more
`[[datasets]]` entries. Model, optimizer, LoRA, loss, evaluation and output settings
are CLI arguments, using the project's standard names such as `--optimizer_type`,
`--optimizer_args`, `--learning_rate`, `--network_dim` and `--network_alpha`.
The old all-in-one `--config_file` training interface is no longer accepted.
Dataset-relative paths resolve against the TOML directory; CLI paths resolve
against the working directory. The generated `run_config.json` is an effective
configuration snapshot for provenance/resume, not another user configuration file.

这条路径默认使用 FP32 主权重和矩阵累加，用来做配对监督。中间激活、注意力和归约包含明确的 half/E4M3 fake quant 与替代梯度，不能视作普通 FP32 SiLU/softmax 网络。实验模式可以改变矩阵计算精度和注意力核心，但不会自动取得原生一致性资格。输出仍在 proxy 颜色空间里，没有 DLL 后面的 Natural / Cinematic 调色。

## 配置分工

和项目其他架构一样，数据集使用 `--dataset_config`，训练超参数使用命令行。单帧全量和 LoRA 共用 [单帧数据集示例](../configs/dlssnr_dataset_single.toml)，时序训练使用 [时序数据集示例](../configs/dlssnr_dataset_temporal.toml)。

```toml
[general]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "../data/target"
control_directory = "../data/input"
num_repeats = 1
nr_controls_mode = "fixed"
nr_style = 0
nr_tone = 1.0
nr_structure = 1.0
nr_skin = -1.0
nr_auto_mask = true
# validation_manifest = "../data/validation_single.jsonl"
# sequence_manifest = "../data/validation_sequences.jsonl"
```

`resolution`、`batch_size`、`enable_bucket`、`bucket_no_upscale` 和 `num_repeats` 可在各个 `[[datasets]]` 中覆盖 `[general]`。多个数据集分别保持其批量大小、重复次数、尺寸和条件设置。目录模式复用项目图像编辑的数据源匹配规则，但每张目标图必须且只能匹配一张输入控制图；缺配、多配、像素网格不一致会报错，不会静默选第一张或独立缩放。

未填写 `resolution` 时默认使用 `[1024, 1024]`；显式填写的旧尺寸不变。开启分桶时这是面积预算，不会将每张图强制拉伸成正方形。

五通道条件在加载时根据 `nr_style`、`nr_tone`、`nr_structure`、`nr_skin`、`nr_auto_mask` 生成，不落地额外 JSONL/NPY。tone/structure 范围为 `[0,1]`；skin 可为 `[0,1]` 或 `-1`（跟随 structure）。auto mask 是模型条件开关，不会生成训练监督蒙版，也不引入分割模型。style 只影响网络条件，输出仍不包含 DLL 后处理调色。

旧数据仍可用 `train_manifest` 替代两个目录字段；默认 `nr_controls_mode = "files"`，读取原有 `controls_path`。选择 `fixed` 时 TOML 条件显式覆盖该数据集及其验证清单内的 `controls_path`，不读取被覆盖的 NPY。时序数据继续用清单明确声明帧序、motion 和有效性蒙版，不会把普通目录图片自动伪装成时序片段。`caption_extension` 和 `cache_directory` 可随公共数据集配置保留，但 NR 不读取 caption 或扩散缓存。

TOML 中的 `[model]`、`[training]`、`[optimizer]`、`[lora]`、`[loss]`、`[output]` 等旧训练配置会明确报错，不再作为另一套超参数来源。自动生成的 `run_config.json` 保留完整展开值，仅供追溯和恢复校验。

## 分桶与配对变换

分桶调用项目公共 `BucketSelector`，同桶组批调用 `BucketBatchManager` 的公共样本选择接口，不读取 diffusion 的 latent/text-encoder 缓存。NR 使用独立架构标识 `nr`，分桶步长为 16，桶的每一轴至少 48 像素；模型内部 padded field 的几何和前向没有改变。

- `enable_bucket = true` 时，`resolution` 的宽高乘积表示桶面积预算，选桶依据原始输入的宽高比，不会把所有图强制变成正方形。比如预算为 `128 x 128` 时，`320 x 192` 的输入进入 `160 x 96` 桶，竖图进入对应的竖桶。
- `bucket_no_upscale = true` 沿用项目语义：面积不超过预算的图像以原尺寸向下对齐到 16 的倍数，不放大。若向下对齐后的某一轴不足 48，会明确拒绝，不会偷偷放大或填充。超过预算的图像仍按宽高比选候选桶并做覆盖裁剪；极端宽高比下可能放大短轴，这个开关并非对任意输入都保证不放大。
- 不写 `enable_bucket` 时仍默认关闭，保留原来的严格固定尺寸模式：所有配对张量必须已符合 `resolution`，尺寸不符就报错。示例 TOML 显式开启分桶和 no-upscale。
- 所有配对张量必须先在原始像素网格上对齐。同一 clip 的所有帧必须具有相同的原始输入尺寸，并共享同一个桶、等比覆盖缩放和中心裁剪位置；不做逐帧随机裁剪，也不独立缩放 target 或 controls 来掩盖错误配对。
- RGB、encoded controls 和连续 loss mask 使用同一 FP32 bilinear/antialias 缩放；二值 history/temporal mask 使用对齐像素中心的 nearest-exact，保持二值。motion 使用 bilinear，并将 x/y 位移分别乘以实际取整后的缩放宽高比。同位置裁剪当前帧和前一帧，裁剪偏移相互抵消；越出裁剪区域的历史仍由重投影的 inside mask 排除。
- 一个 microbatch 内只有同尺寸、同帧数的样本。不同桶可出现在同一次梯度累积中，损失继续按有效像素总数归一化。每个桶不足 batch size 的尾批保留，不丢弃、不重复填充；`consumed_samples` 记录实际消费数。

索引阶段只读取源图头部或 NPY 的 mmap shape 来选桶，不将像素常驻内存。训练前仍会逐项解码校验配对、数值和变换后剩余的监督区域。训练和验证使用同一变换规则，`run_config.json` 的 `bucket_plan` 记录实际桶、每桶样本数、每轮批数及确定性批次顺序指纹。NR 保留固定的桶/清单顺序和确定性中心裁剪，不新增随机 shuffle 或随机增强。

分桶不等于原生输出等价：图像增强通常不满足“先增强再缩放”等于“先缩放再增强”。要优先保留原生配对的像素尺度，使用 no-upscale，并准备面积预算内、16 对齐的源尺寸；缩放后的配对只能作为训练数据变换，不能作为 DLL 前向一致性的证据。

## 当前范围

根目录脚本保持 thin entry point。模型、数值、数据、训练损失和评估位于 `src/musubi_tuner/dlssnr/`，LoRA 位于 `networks/lora_dlssnr.py`，生命周期位于 `training/dlssnr_trainer.py`。全量和 LoRA 共用 `NRTrainModule`、Accelerate 包装及同一更新循环，不调用 diffusion trainer。

- 默认单进程 FP32，自动选择 CUDA，缺少 CUDA 时使用 CPU；`--device` 可设为 `auto`、`cpu` 或 `cuda`。显式请求不可用的 CUDA 会报错。混合精度只支持 CUDA，DDP 支持单机多进程。
- 单帧及一个有限时序段，支持全量/LoRA 梯度累积、定期保存与恢复。累积按有效元素总数归一化，计步单位是 optimizer update。
- 数据配置和命令行参数均严格校验。数据清单路径以 TOML 所在目录为基准，命令行路径以当前工作目录为基准；最终配置和来源指纹写入 `run_config.json`。
- 数据集中的 `validation_manifest` 和 `sequence_manifest` 会实际执行，使用独立学生 history。默认比较初始基座，写入 `evaluation/stepNNNNNN.json`；评估不会改变训练 RNG 或 dropout 状态。
- manifest 按需解码，不将整个训练集常驻内存；推理和评估按帧回放。

原生参考前向、真实片段上的兼容性和质量验收尚未完成。布局字节 round-trip、浮点训练测试和原生兼容是不同结论。

2026-09-24 已完成第一轮 Steam 截图与本地 310.8.0 显卡适配版 DLL 的对照，结论是**当前 FP32 前向不一致**。4 张截图、19 组图片/参数组合覆盖 style、intensity、tone、structure、skin 和 auto mask；默认参数的输出与 DLL 之间 MAE 为 0.0294 至 0.0420。相同 packed 权重不代表前向计算正确，也不能据此宣称训练产物兼容原生 DLL。详见 [截图前向实验记录](dlssnr_forward_experiment_2026-09-24.md)。

同日后续已排除主要默认参数因素，并修复 cubic 激活及关键数值边界。默认参数下四图 MAE 降至 0.00262 至 0.00520，19 组中 9 组达到原有三项显示指标门槛（含一组零强度直通），**尚未完成全部前向验收**。全量测试为 172 passed，包含真实权重 CUDA 全量/LoRA 更新及合并对照。详见 [可训练前向对齐记录](dlssnr_forward_alignment_2026-09-24.md)。数值实现版本已更新，旧训练状态不能跨版本 exact resume。

## 训练输入与验证证据

正常全量和 LoRA 训练必须提供 `--model_dir`、canonical 权重、匹配的 schema/profile、完整来源文件和通过 source round-trip 的转换报告；仍会检查实际数据和张量。**forward validation report 不再是训练前置条件**，不需要为此开启开发模式。缺少模型或来源文件时仍会停止，不会隐式随机初始化或绕过来源检查。

`--development_smoke` 仅保留给明确的开发 smoke 测试：允许来源文件不完整，或省略 `--model_dir` 使用随机权重。它不等于普通训练开关，也不能绕过显式提交的无效验收报告。

需要绑定验收证据时，显式传入 `--forward_validation_report`；报告必须对应 `--model_dir`，路径不存在、身份不匹配或检查未通过都会报错，开发模式也不例外。普通训练不会自动读取目录内的 `forward_validation_report.json`，避免未请求的旧报告改变训练或恢复行为。工具调用可使用 `inspect_canonical(..., require_forward_validation=True)` 强制要求证据；未提供 `validation_report` 路径时，该调用才读取 canonical 目录内的默认报告。

报告契约仍为 `dlssnr_forward_validation_v1`，包含 `profile`、`numerics_profile`、当前 `model_sha256`、`implementation_sha256`、固定的 `reference_identity`、`float_validated` 和 `checks`。检查项为 `raw_head`、`neural_preclamp`、`rendered_proxy`；全部必须来自实际参考验证。实现指纹由 `json_sha256(implementation_identity())` 计算。普通训练 evaluation 不能代替这份报告，也不要手工填入通过标记。

运行元数据以 `source_forward_validated` 单独记录是否检查过有效的基座报告。无报告时该字段为 `false`，并保留 `experimental_surrogate=true`，但不会把 `development_smoke` 自动设为 `true`。开发 smoke 产物也始终保留实验标记。无论基座是否有报告，新训练产物的 `float_validated`、`temporal_validated`、`native_export_validated` 均为 `false`；基座证据不能证明更新后权重的兼容性。

## 显存与运行模式

显存不足时先加 `--gradient_checkpointing`。它在梯度开启的训练段重算 FFN/attention，覆盖首尾块、编码器、ViT 和解码器；burn-in 与评估不重算。全量和 LoRA 均支持，冻结输入不会截断 LoRA 梯度，dropout 的随机状态会在重算时恢复。此开关默认关闭，不要求切换数值 profile。

需要其他选项时显式选择实验模式：

```text
--numerics_profile train_experimental --mixed_precision bf16 --gradient_checkpointing
```

FP16/BF16 只用于选定的投影和矩阵乘法，训练参数、LoRA delta、明确的 half/E4 publications、归约、loss 和 history 保留 FP32 边界。FP16 使用 GradScaler，先 unscale 再检查/裁剪梯度；溢出时缩小 scale，重放同一有效批次和 RNG，不提前推进 update 或样本计数。`--max_overflow_retries` 默认 16；非有限前向或重试耗尽会停止，不写成功检查点。BF16 不使用 scaler。Accelerate 的 mixed-precision 环境设置须与显式 CLI 参数一致。

FP8 初期只压缩 LoRA 的冻结投影基座：

```text
--numerics_profile train_experimental --gradient_checkpointing --fp8_base --fp8_scaled
```

`--fp8_scaled` 使用项目公共的 block-64 量化，输入宽度不整除 64 时按输出通道缩放；不写该参数则使用 E4M3FN 直接存储，超出 `[-448,448]` 的权重明确拒绝。输入适配器、RGB/logit heads、priors 和标量保持原表示。计算时还原 FP32 基座再加 LoRA delta，**不代表执行 FP8 GEMM**。检查点有助于避免反向一直保留临时还原矩阵。产物绑定原始基座、实际量化基座及 scale 身份，不能把量化适配器直接加到未经相同量化的原始基座上。

注意力选项互斥，默认仍是 NR 自定义算子：

| 参数 | 范围与限制 |
| --- | --- |
| `--sdpa` | 实验窗口和全局注意力，保留窗口 prior；PyTorch 自行选择其内部 kernel，不等于保证使用 Flash kernel。 |
| `--xformers` | 实际执行所需 dtype、bias 和反向能力探测；不支持的组合报错，可显式限制 `--attention_scope global`。 |
| `--flash_attn` | 需 FP16/BF16 和显式 `--attention_scope global`；窗口仍使用 NR 算子，不能丢弃 learned prior。 |
| `--sage_attn` | 初期仅支持 FP16/BF16 的全局推理；训练会拒绝，未伪造一个替代 backward。 |
| `--attention_backend native` | 显式选择 NR 算子，也可在推理时覆盖已保存的实验后端。 |

实验注意力对预处理后的 Q/K/V 使用 `scale=1.0` 的标准 softmax，不额外乘 `1/sqrt(32)`。窗口外零 token 继续参与带 prior 的归一化；全局层的人工对齐 padding 不作为真实 key。这与 NR 的 half/E4 指数及归约不同，需要单独做输出质量验收。可选扩展按需加载，未安装或 ABI 不匹配时不会悄悄换后端。

### 多卡

例如，单机双进程训练可使用以下启动方式；其他训练参数与单卡相同：

```bash
torchrun --standalone --nproc_per_node=2 dlssnr_train.py \
  --dataset_config configs/dlssnr_dataset_single.toml --model_dir models/canonical_dlssnr \
  --gradient_checkpointing --gradient_accumulation_steps 2 \
  --output_dir output/dlssnr_ddp --output_name full_ddp --save_state
```

也支持正确配置的 `accelerate launch`。Linux CUDA 使用 NCCL，Windows/CPU 沿用 Gloo 初始化方式。每卡仍保存完整模型；DDP 不是模型分片，不会让单张卡自动装下更大的模型。

分配位置为 `(update * accumulation + micro) * world_size + rank`，尾批不填充，按确定性批次计划继续下一轮。损失分母在整个跨卡累积批次上求有效像素总数，不平均各卡自己的 mean loss。主进程负责共享权重、日志和评估，每卡 RNG/scaler 都纳入状态。恢复必须保持相同 world size、数据计划和运行策略；不支持跨 world size 自动重分片、FSDP、ZeRO、DeepSpeed 或多机。

### 本机实测

2026-10-01，RTX 4090 / PyTorch `2.13.0+cu130`，同一份真实 canonical 权重，单帧有效图 `512 x 512`（内部 field 为 `576 x 512`）、batch 1、seed 42、AdamW；LoRA 为 ViT rank 16。每种模式独立进程，先更新 1 次，再测量 2 次。表中是 PyTorch **峰值 allocated** 显存，不含其他进程占用；时间样本很少，仅供本机对照。

| 模式 | 峰值 GiB | 更新耗时中位数，秒 |
| --- | ---: | ---: |
| 全量 FP32 | 15.92 | 1.96 |
| 全量 FP32 + checkpoint | 4.52 | 2.90 |
| 全量 BF16 + checkpoint | 4.46 | 3.17 |
| 全量 FP16 + checkpoint | 4.46 | 4.65 |
| 全量 BF16 + checkpoint + SDPA | 3.95 | 3.12 |
| LoRA FP32 | 8.17 | 1.39 |
| LoRA FP32 + checkpoint | 2.96 | 1.94 |
| LoRA BF16 + checkpoint | 2.91 | 2.14 |
| LoRA BF16 + checkpoint + scaled FP8 | 2.52 | 2.35 |

全量 checkpoint 的显存下降约 72%，LoRA 约 64%；两组 FP32 开关对照的计时步骤 loss 相同。FP16 有 1 次计时内溢出重试，耗时包含该重试。这里最有效的省显存选项是 checkpoint，不是 AMP；混合精度的额外收益较小，也没有在本机提速。实验模式的 loss 数值有所不同，这些合成输入测试不能证明学习质量或原生一致性。

可用相同工具复测，逐种模式另开进程并选择不同输出文件：

```bash
python tools/benchmark_dlssnr_runtime.py --model_dir models/canonical_dlssnr \
  --output output/benchmark-checkpoint.json --width 512 --height 512 \
  --gradient_checkpointing --warmup 1 --steps 2
```

本机已执行真实权重 CUDA 的 AMP、SDPA 更新，以及 FP8 LoRA 的更新与合并检查；双进程 CPU/Gloo 已覆盖不等有效像素、尾批和恢复。本机只有一张 GPU，未验证真实双 GPU；隔离测试环境没有 FlashAttention、SageAttention 或 xFormers 扩展，其实际 CUDA kernels 尚未验证。安装扩展不等于通过能力与质量验收。

## 命令

把自行准备的 OpenDLSS-NR `models/nr` 目录转成 canonical 权重：

```bash
python dlssnr_convert_model.py --source_dir ../OpenDLSS-NR/models/nr --output_dir models/canonical_dlssnr --verify_roundtrip
```

单帧全量训练。以下多行命令使用 Bash 续行；PowerShell 可合并为一行，或将行末反斜线换成反引号：

```bash
python dlssnr_train.py \
  --dataset_config configs/dlssnr_dataset_single.toml \
  --model_dir models/canonical_dlssnr \
  --gradient_checkpointing \
  --optimizer_type AdamW --learning_rate 1e-5 \
  --optimizer_args weight_decay=0.0 \
  --output_dir output/dlssnr_full_single --output_name dlssnr_310_8_0_full \
  --max_train_steps 1000 --save_every_n_steps 100 --save_state
```

时序全量训练：

```bash
python dlssnr_train.py \
  --dataset_config configs/dlssnr_dataset_temporal.toml \
  --model_dir models/canonical_dlssnr \
  --training_mode temporal --sequence_length 4 --burn_in 2 --tbptt_length 2 \
  --gradient_checkpointing \
  --loss_temporal 0.10 --optimizer_type AdamW --learning_rate 1e-5 \
  --output_dir output/dlssnr_full_temporal --output_name dlssnr_310_8_0_temporal \
  --max_train_steps 1000 --save_every_n_steps 100 --save_state
```

ViT LoRA。rank、alpha 和优化器都直接从参数选择，使用同一份单帧数据集 TOML：

```bash
python dlssnr_train_network.py \
  --dataset_config configs/dlssnr_dataset_single.toml \
  --model_dir models/canonical_dlssnr \
  --network_dim 16 --network_alpha 16 --network_dropout 0.0 \
  --gradient_checkpointing \
  --optimizer_type AdamW --learning_rate 1e-4 \
  --optimizer_args weight_decay=0.0 betas=0.9,0.999 \
  --output_dir output/dlssnr_lora --output_name dlssnr_310_8_0_lora_vit \
  --max_train_steps 1000 --save_every_n_steps 100 --save_state
```

常用参数：

| 参数 | 用途 |
| --- | --- |
| `--optimizer_type` / `--optimizer_args` | 复用项目公共优化器构造逻辑，例如 `AdamW`、`SGD`、`AdamW8bit` 或完整类路径。额外依赖须自行安装。 |
| `--learning_rate` | 全量默认 `1e-5`，LoRA 默认 `1e-4`；AdamW 默认 weight decay 为 `0.0`。 |
| `--network_dim` / `--network_alpha` | ViT LoRA rank 默认 `16`；未给 alpha 时跟随 rank。 |
| `--network_dropout` / `--network_args` | dropout 默认 `0`。多尺度 LoRA 用 `--network_args profile=multiscale`，可通过 `rank_by_width` / `alpha_by_width` 字典细化，不能同时给单一 dim/alpha。 |
| `--max_train_steps` / `--gradient_accumulation_steps` / `--seed` | 更新次数、梯度累积和随机种子。 |
| `--max_grad_norm` | 在累积后的更新边界裁剪梯度，默认 `0`，即不裁剪。 |
| `--gradient_checkpointing` | 重算激活以降低显存，默认关闭，保持默认数值 profile。 |
| `--numerics_profile train_experimental` / `--mixed_precision` | 显式实验模式；CUDA precision 可选 `no`、`fp16`、`bf16`，主权重仍为 FP32。 |
| `--fp8_base` / `--fp8_scaled` | 仅 LoRA 的冻结基座存储量化；scaled 需配合 base，均需实验模式。 |
| `--sdpa` / `--xformers` / `--flash_attn` / `--attention_scope` | 显式实验注意力与作用范围；限制见上表。 |
| `--loss_pre` / `--loss_out` / `--loss_edge` / `--loss_temporal` | 损失权重，默认分别为 `1`、`1`、`0.05`、`0`。 |
| `--prior_lr_multiplier` / `--scale_lr_multiplier` / `--temporal_blend_lr_multiplier` | 全量训练参数组倍率，默认 `0.1`；LoRA 入口不接受这些参数。 |
| `--sample_every_n_steps` / `--min_sequence_frames` | 验证间隔和时序评估的最小帧数。非零验证间隔要求数据 TOML 配置验证清单。 |
| `--no-compare_baseline` | 关闭初始基座对照，默认开启。 |
| `--save_state` / `--save_every_n_steps` / `--resume` | 保存训练状态、周期和恢复点；未给 `--save_state` 时仅保存权重。 |
| `--forward_validation_report` | 可选基座验收证据；仅显式指定时检查，不是训练前置条件。 |
| `--development_smoke` | 仅限开发测试，显式允许不完整来源或随机初始化；普通训练不需要。 |

例如，将优化器选项换成 `--optimizer_type SGD --optimizer_args momentum=0.9 weight_decay=0.0`，会实际构造 SGD，而不是只改变配置标签。Adafactor 必须显式给 `--optimizer_args relative_step=False warmup_init=False`。学习率调度复用项目公共工厂，支持 constant、constant_with_warmup、linear、cosine、cosine_with_restarts、cosine_with_min_lr、polynomial、inverse_sqrt、warmup_stable_decay。相关 warmup/decay、cycles、power、timescale、min_lr_ratio 参数沿用公共 CLI。每次成功的全局 optimizer update 推进一次调度器；梯度累积、DDP rank 和溢出重试不额外推进。调度器状态随训练状态保存并参与恢复；改变调度配置不能 exact resume。schedule-free 的权重切换和需要 closure 的优化器仍不支持。

从 optimizer 更新边界继续时，在**原来的完整训练命令**后追加恢复参数，保留原有有效参数和数据配置。例如：

```text
--resume output/dlssnr_full_single/dlssnr_310_8_0_full/state-step000100
```

LoRA 的恢复目录对应 `output/dlssnr_lora/dlssnr_310_8_0_lora_vit/state-stepNNNNNN`。只加载权重、不带 optimizer 状态属于重新开跑，不是 resume。

静止图和闭环片段。推理清单可以不写 target。片段里非 reset 帧必须有 motion 和 history mask：

```bash
python dlssnr_generate_image.py --model_dir models/canonical_dlssnr --sample_manifest data/inference_single.jsonl --bucket_width 512 --bucket_height 512 --output_dir output/preview
python dlssnr_generate_video.py --model_dir models/canonical_dlssnr --sequence_manifest data/inference_sequence.jsonl --bucket_width 512 --bucket_height 512 --output_dir output/video
python dlssnr_merge_lora.py --base_model_dir models/canonical_dlssnr --adapter output/dlssnr_lora/dlssnr_310_8_0_lora_vit/final/adapter.safetensors --output_dir output/dlssnr_merged
```

推理不指定 runtime 参数时继承产物中的策略；没有策略的旧模型使用 FP32/native 默认值。显式覆盖会记录在输出目录的 `inference_metadata.json`，片段则记录在片段子目录。切回 baseline 须同时选择兼容参数，例如 `--numerics_profile train_surrogate --mixed_precision no --attention_backend native`。FP8 开关可通过 `--no-fp8_base` / `--no-fp8_scaled` 关闭。Sage 全局推理需显式 `--numerics_profile train_experimental --mixed_precision bf16 --sage_attn --attention_scope global`，且本机扩展必须可用。

## 现在明确不做的事

不支持全量 FP8 优化器训练、SageAttention 训练、CPU activation offload、block swapping、FSDP/ZeRO/DeepSpeed、多机或 TF32。未知 CLI 参数仍报错；实验模式仅面向 `float_runtime`，不能选择 `native_roundtrip`。

历史重投影用的是双线性，不是原版五点 Catmull-Rom。历史按 FP32 保存，不是向零截断到 float16。

像素缓存和多 worker 预取尚未实现，数据 TOML 不接受缓存占位字段。FP32 入口会暂时关闭 TF32，结束后恢复原来的全局设置。

## 数据约定

每条 JSONL 记录必须包含唯一且可用于文件名的 `sample_id`、`sequence_id`、`schema = dlssnr_pairs_v1`、`source_encoding = srgb_proxy`、`controls_encoding = dlssnr_lanes_10_14_v1`。训练及配对评估还需 `target_encoding = srgb_proxy` 和 target。首帧必须 reset，`frame_index` 非负且严格递增；训练与验证不能共享 sequence ID。

`sample_id` 在所有平台上按不区分大小写的规则检查重复，例如 `Frame` 和 `frame` 不能出现在同一清单中。名称不能包含路径分隔符、Windows 非法字符、控制字符、末尾句点或空格，也不能是 `NUL`、`CON` 等 Windows 保留名称。清单索引阶段即报错，不会写出部分推理结果后才发现冲突；原始 ID 不会被自动改写。

- RGB 支持 8-bit 图像，或 `[3,H,W]` 的数值 NPY，proxy 值域为 `[0,1]`。不将高位深图片静默压成 8-bit。
- controls 为有限值 `[5,H,W]` NPY。motion 的布局由记录级 `motion_layout = chw|hwc` 明确指定，单位是 current-to-previous 像素位移。
- mask 支持单通道 PNG 或 `[1,H,W]` NPY。history/temporal mask 必须为二值；可选 `loss_mask_path` 支持 `[0,1]` 权重。
- temporal loss 非零时，非 reset 帧必须提供 temporal mask，且 `tbptt_length >= 2`。损失只计算可反传段内部的相邻帧，不跨越 burn-in 边界。
- `--loss_temporal 0` 允许不提供 temporal mask，但非 reset 帧仍必须有 motion 和 history mask。

## 保存与恢复

每次训练使用 `output_dir/output_name/` 作为独立运行目录，`final/`、周期权重、训练状态、日志和评估报告均位于该目录下。`--output_name` 必须满足上述可跨平台使用的名称规则，不能包含路径。新训练拒绝使用已存在的同名运行目录；继续训练须显式传入 `--resume`，重新训练须选择其他名称或输出目录。

旧产物不会自动搬迁。展开配置仍使用 schema version 2，训练状态现为 `dlssnr_train_state_v3`，包含 scaler、各 rank 状态与 runtime 身份。旧版 v1/v2 状态及旧实现指纹不能跨版本 exact resume。全量旧权重可作为新运行的基座；旧 LoRA 可先合并成 canonical 基座再开新运行，但这不恢复原优化器或原 adapter 参数化。不要手改检查点身份标记。

全量 `final/` 保存 FP32 权重、canonical 配置、来源 manifest、原始 opaque records、数值与预处理信息及训练元数据。runtime 同时写入必需的 model config 和 safetensors header，二者不一致或缺失会被拒绝，不依赖可选训练 sidecar 才能重现策略。LoRA v2 保存 adapter、rank/alpha/targets、完整基座身份和必需的 runtime/量化信息，仍可读取符合旧契约的 v1 FP32 adapter。FP8 adapter 合并会重建并核验量化基座，再输出普通 FP32 的基座加 delta，标记已 materialize，默认加载不会二次量化。合并保留 canonical 来源与 opaque 数据。

LoRA 前向使用权重侧的 `W + alpha/r * B@A`，与合并权重走相同的投影计算。真实权重测试发现，拆成两条 GEMM 再相加的 FP32 误差会在 ViT 中明显放大，不能依靠小线性层测试证明全网合并一致。dropout 作为 rank 空间的修正项加入，评估时关闭。这个选择会增加临时矩阵和反向计算开销，优先保证一致性；产物记录 `canonical_weight_plus_delta_v1`。

`--save_state` 在定期及最终更新边界生成 `state-stepNNNNNN/`。状态包含 optimizer、实际全局消费样本数、各 rank 的 CPU/当前 CUDA 设备/Python/NumPy RNG、FP16 scaler、展开配置与数据/组批计划/基座/实现指纹。全部状态完成后才发布校验清单。恢复位置由成功 update、梯度累积和 world size 定位到全局批次，支持尾批和跨轮恢复。不传此参数时仍输出 `stepNNNNNN/` 权重，但这些目录不能 exact resume。

仅修改数据 TOML 注释或 CLI 参数排列顺序不会阻止恢复；修改学习率、优化器、rank、分桶、precision、attention、FP8 策略、world size、引用的数据、基座或训练实现会拒绝 exact resume。显式报告的路径和内容也属于恢复身份，未显式请求的目录内报告不参与该身份。CUDA 反向部分算子的非确定性仍需按数值容差验收，不承诺所有硬件间逐位一致。

旧 adapter 若缺少完整基座身份、rank/alpha 或 forward mode 元数据，也不会被静默接受。保留旧文件，重新生成受校验的产物。

## 历史验证与回归

2026-09-23，本机全库测试 159 项通过，无跳过，Ruff 检查通过。测试包含原始 310.8.0 权重在 RTX 4090 上的 FP32 全量/LoRA 单步更新、全部 ViT LoRA targets 的梯度和合并后的 raw-head 对照。CUDA smoke 使用 48×48 合成配对输入，不代表正式分辨率的性能、真实数据训练质量或 native parity。

开发环境未安装项目包时，可在仓库根目录设置 `PYTHONPATH` 后测试；正常运行仍应使用项目依赖环境。真实权重 CUDA 测试需本机源权重，默认查找仓库相邻的 `../OpenDLSS-NR/models/nr`，也可通过 `DLSSNR_SOURCE_DIR` 显式指定目录。缺少权重时，这些外部权重测试会跳过；不代表已通过真实权重验收。

```powershell
$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:DLSSNR_RUN_CUDA_SMOKE = "1"
python -m pytest -q --tb=short
```
