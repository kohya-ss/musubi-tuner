# DLSS-NR 310.8.0 → Musubi Tuner 训练适配规格

**版本：** 0.2（整合修订版）  
**日期：** 2026-09-22  
**文档类型：** 开发 RFC / 接口契约 / 验收规格  
**状态：** 设计规格，已有部分实现；实际能力和限制见 [实现说明](dlssnr.md)。本文不是已经通过模型、训练或部署测试的报告。  
**目标仓库：** `sdbds/musubi-tuner`  
**模型 profile：** `dlss_nr_310_8_0`  
**替代文档：** `dlssnr_musubi_tuner_adaptation_spec.md` v0.1。本版为完整替代稿，不需要与旧版拼接使用。

> **2026-10-01 接口修订：** 训练入口现使用 `--dataset_config` 加命令行训练参数；TOML 只保留 `[general]` / `[[datasets]]` 的数据集设置。本文中的旧版完整训练 TOML 示例及 `--config_file` 用法不再作为当前接口契约，现行参数、示例和迁移约束以 [实现说明](dlssnr.md) 为准。模型、优化器、LoRA rank/alpha、损失、保存和恢复设置均从 CLI 传入。

> **核心决策：保留 Musubi 作为宿主，新增 NR 专用像素域监督训练入口；先建立全量微调基线，再加入时序训练和分级 LoRA。NR 模型、数值实现、数据与时序算法独立于扩散训练器。**
>
> 权重布局以用户提供的 DLSS-NR 310.8.0 本地二进制审计为已知输入，不再称为“待猜测的候选网络”。但布局正确不等于计算正确；数值前向、梯度、时序与部署兼容性仍必须分别验收。

## 目录

1. [约定、证据与范围](#1-约定证据与范围)
2. [训练器选型与架构决策](#2-训练器选型与架构决策)
3. [310.8.0 权重清单与字节账](#3-31080-权重清单与字节账)
4. [固定网络结构](#4-固定网络结构)
5. [磁盘布局与 canonical 转换](#5-磁盘布局与-canonical-转换)
6. [数值 profile 与兼容性等级](#6-数值-profile-与兼容性等级)
7. [特征、几何与帧处理接口](#7-特征几何与帧处理接口)
8. [数据集与缓存](#8-数据集与缓存)
9. [输出合成与监督损失](#9-输出合成与监督损失)
10. [时序训练与推理状态机](#10-时序训练与推理状态机)
11. [全量训练与参数管理](#11-全量训练与参数管理)
12. [LoRA 适配](#12-lora-适配)
13. [Musubi 宿主集成](#13-musubi-宿主集成)
14. [精度、显存与性能策略](#14-精度显存与性能策略)
15. [保存、恢复与导出](#15-保存恢复与导出)
16. [评估与验收标准](#16-评估与验收标准)
17. [文件与模块变更](#17-文件与模块变更)
18. [配置与目标命令](#18-配置与目标命令)
19. [失败处理与拒绝项](#19-失败处理与拒绝项)
20. [分阶段交付](#20-分阶段交付)
21. [相对 v0.1 的修订](#21-相对-v01-的修订)
22. [附录 A：block 1 字节布局](#22-附录-ablock-1-字节布局)
23. [附录 B：原始数值指纹](#23-附录-b原始数值指纹)
24. [来源与复核入口](#24-来源与复核入口)

## 1. 约定、证据与范围

### 1.1 规范语言

**MUST** 表示交付硬要求；**SHOULD** 表示建议要求；**MAY** 表示可选能力。除明确标注为既有源码的部分外，本文中的脚本、类、配置字段及命令都是**拟新增接口**，当前仓库不能直接运行这些示例。

事实、测量与设计分开记录：

| 标记 | 含义 | 使用边界 |
|---|---|---|
| U1 | 用户提供的本地 bin / manifest 审计 | 本 profile 的权重结构与数值基线；本次整理没有重新读取用户的二进制 |
| S1–S5 | 前次对两个训练器的源码核对 | 仅对下列固定提交作出接口判断，不声称是未来版本的稳定 API |
| S6–S9 | OpenDLSS-NR 项目的一手实现说明 | 补充运行时图、几何、数值与历史处理；不是 NVIDIA 官方训练配方 |
| 设计要求 | 本文提出的训练方法、接口、超参数及门槛 | 必须实现和测试；不能当成已测结果或 NVIDIA 原始设置 |

用户提供的 NVIDIA 报告 PDF 未读到原文；用户本地访问返回 403。本文**不依赖未读到的 PDF 正文、图表或页码**，也不据此断言官方预训练、蒸馏或 diffusion 参数化。

### 1.2 固定版本基线

| 项目 | 基线 |
|---|---|
| 模型 | DLSS-NR 310.8.0；U1 描述的完整 11-stage 网络 |
| Musubi | `4e7c7149249e7715e9168920feb4c420423abba7` |
| sd-scripts 对比基线 | `4e624302e0088e39933b31cbc71f24212e900f5f` |
| 模型语义 schema | `dlssnr_canonical_v1`（拟定义） |
| 数据 schema | `dlssnr_pairs_v1`（拟定义） |

以上提交为前次源码核对快照，不表示本次重新查询了远端最新状态。实施时 MUST 记录实际 checkout commit。OpenDLSS 参考实现也 MUST 固定 commit 或内容哈希，不能只存 `main`。

原始 11 个 stage、manifest、参考实现与 HF 文件的身份分别追踪。**相同名称或相同逻辑模型，不代表它们的容器字节与 SHA-256 相同。**

### 1.3 目标与非目标

目标是基于现有权重，对配准的“输入渲染帧 → 期望输出帧”进行微调，支持单帧与显式历史的因果时序模式。

主线交付包括固定 profile 的严格转换、可验证前向、配对像素数据、全量训练、分级 LoRA、短序列训练、连续视频评估、精确训练状态恢复与 canonical 导出。原始格式的**未训练权重 round-trip** 属于转换验收；**训练后重新量化部署**属于另外的兼容性阶段。

不包含从零预训练、完整教师蒸馏、文本生成、DLSS-SR、Frame Generation、Ray Reconstruction、官方 DLL 回写、官方游戏运行时兼容承诺、4K 实时性能承诺。单 GPU 是首个支持范围；DDP、混合精度、量化感知训练与其他性能后端各有独立门槛。

## 2. 训练器选型与架构决策

### 2.1 选择 Musubi，但不沿用其默认扩散数据流

Musubi 的公共基类已有 `process_batch`、`compute_loss`、`prepare_sampling`、模型/优化器/保存生命周期 hook，并可返回分项 loss 指标。它也已有视频 datasource、帧位置和片段来源等数据设施。[S1] [S2]

`sd-scripts` 同样已有可覆盖的 `process_batch`、`get_noise_scheduler` 和模型加载接口，并具有验证循环。其 optimizer、dataset、checkpoint 等工具也已模块化；不能将它描述为“不支持非标准损失”或“没有可复用工具”。不过默认启动流程仍围绕 tokenizer、VAE、text encoder 与 latent strategy；任意 `dataset_class` 路径也不会自动得到 validation dataset。[S4] [S5]

| 对比项 | Musubi | sd-scripts | 对本项目的决策 |
|---|---|---|---|
| 自定义监督 step | 有 batch/loss hook 与分项日志 | 有 batch/scheduler hook | 两者均可承载，不以此排除任何一方 |
| NR 时序基础 | 可借用视频索引、媒体和来源追踪 | 可接入自定义 dataset | Musubi 作为长期宿主更方便，但历史/warp/TBPTT 两边都需新增 |
| 无 VAE/文本的完整入口 | 默认循环仍有扩散假设 | 启动 strategy 仍有文本/latent 假设 | 两边都不能仅替换最后一次模型调用 |
| 验证基础 | 可借用采样与日志生命周期 | 已有 validation loop，但默认遍历 timestep | 新建 NR evaluator，不把 diffusion 验证冒充 NR 回放 |
| 原生 fragment / 数值兼容 | 无现成 DLSS-NR 支持 | 无现成 DLSS-NR 支持 | 由独立 NR 核心实现解决 |

该选择是代码组织与接入成本判断，不是训练速度、峰值显存或质量的实测排名。

### 2.2 采用的组件边界

```text
NR 专用 CLI / 配置
        |
Musubi 服务适配层 + NR supervised runner
        |
NRTrainingStep / NREvaluator
        |
NRFramePipeline
   ├── NRPreprocessor / GeometryResolver / NoiseGenerator
   ├── NRModel（canonical 参数 + 指定数值 profile）
   └── NRCompositor / 显式 NRHistoryState
        |
CheckpointAdapter / NativePacker
```

NR 核心不导入 VAE、文本 tokenizer、diffusion scheduler 或 Musubi 默认加噪算法。全量与 LoRA 共用 pipeline、training step、数据契约、loss 与 evaluator。

### 2.3 不采用的实现方式

不复制 Wan/FLUX 的 diffusion step；不把 RGB 假装成 latent；不制造假的 timestep；不构造空 VAE 维持生命周期；不为了调用通用加载器要求用户传一个实际被忽略的 `--sdpa`。

首版采用**独立 NR supervised runner**，不调用 `NetworkTrainer.train()`。旧版提出的两个通用 latent/noise hook 不再是主线依赖，也不为 NR 进行无关的公共循环重构。底层日志、优化器、LR scheduler、Accelerate 和 checkpoint 工具仍复用 Musubi。

## 3. 310.8.0 权重清单与字节账

本节全部结构数值来自 U1；求和与计数是对 U1 的算术核对。

### 3.1 Stage 清单

| stage | 载荷字节 | 内容 |
|---|---:|---|
| enc32 | 106,432 | block 0–4，32 通道 |
| enc64 | 255,216 | block 5–8 |
| enc128 | 1,215,856 | block 9–14 |
| enc256 | 5,644,912 | block 15–22 |
| enc512 | 16,269,840 | block 23–30；每块 4 条记录，block 30 另有 layer4 |
| vit | 100,697,232 | block 31–38 |
| dec512 | 16,270,848 | block 39–47；39 是纯 transition |
| dec256 | 5,645,408 | block 48–55 |
| dec128 | 1,216,096 | block 56–61 |
| dec64 | 255,328 | block 62–65 |
| dec32 | 106,610 | block 66–70，含独立的 2 字节 blend scale |
| **合计** | **147,683,778** | **140.842 MiB，11 个 stage** |

stage 的实际文件名取自原 manifest；表中名称是 stage ID，不能猜测文件扩展名。各 stage 的载荷区间连续、无空洞。153 条记录由 152 条 `blockN.layerM.layer` 和一条 `block70.layer0.blend_scale` 组成；后者位于 block 70 主体前。

**147,683,778 是上述 stage 载荷总量。** 对含额外 header 的 safetensors 或其他容器，必须分别检查容器大小与提取后的 payload，不能直接要求两种文件总长相等。

### 3.2 参数与非参数字节

| 项目 | 元素数 | 原始 dtype | 有效参数字节 |
|---|---:|---|---:|
| E4M3 矩阵 | 143,831,040 | E4M3 | 143,831,040 |
| f16 scale / prior 等 | 1,922,720 | little-endian f16 | 3,845,440 |
| 逐 head 温度 | 714 | little-endian f32 | 2,856 |
| 输入与输出 f16 GEMM | 640 | little-endian f16 | 1,280 |
| **合计，不含独立 blend scale** | **145,755,114** | 混合 | **147,680,616** |

其中 attention prior 为 `1,875,968 = 458 × 64 × 64` 个 f16；两个 f16 GEMM 分别有 512 和 128 个有效元素。

```text
147,683,778 - 147,680,616 = 3,162 字节

3,162 = 2,376  全零 padding
      +   768  head 中 384 个未使用 half 槽
      +    16  8 个未读 ViT layer3 标量
      +     2  独立 blend_scale
```

独立 `blend_scale` 参与合成。将其计入模型标量槽位后，总逻辑数为 **145,755,115**。这不等于每种训练模式的可更新参数数：固定零通道、单帧冻结分支与用户冻结策略必须另行扣除并打印。

八个 ViT `layer3` 标量和 padding 不是优化器参数。head 空槽不进入逻辑矩阵。全部来源字节都必须得到明确分类，不允许使用“未识别尾部”掩盖错误。

## 4. 固定网络结构

### 4.1 拓扑与 block 编号

共有 71 个 block 编号，但只有 **70 个含注意力的 mixer**：62 个 window mixer、8 个 global ViT mixer；block 39 没有注意力。head 维恒为 32，window 为 8×8。[U1]

| block | 分辨率级 | C | heads | 结构与特殊项 |
|---|---|---:|---:|---|
| 0 | full | 32 | 1 | 输入 f16 16→32；全分辨率 mixer，输出保留给最后 skip |
| 1–4 | L0 | 32 | 1 | dense FFN；block 4 附带 32→64 transition |
| 5–8 | L1 | 64 | 2 | 2 路 bottleneck；block 8 附带 64→128 |
| 9–14 | L2 | 128 | 4 | 4 路；block 14 附带 128→256 |
| 15–22 | L3 | 256 | 8 | 8 路；block 22 附带 256→512 |
| 23–30 | L4 | 512 | 16 | 8 条 64 宽分支；block 30 layer4 为 512→1024 |
| 31–38 | L5 | 1024 | 32 | global attention，无 prior |
| 39 | L5→L4 | 1024→512 | — | 仅 projection、nearest upsample 与 512 个 skip scale |
| 40–47 | L4 | 512 | 16 | 与 23–30 相同的 mixer 变体 |
| 48–55 | L3 | 256 | 8 | block 48 内含 512→256 decoder transition |
| 56–61 | L2 | 128 | 4 | block 56 内含 256→128 |
| 62–65 | L1 | 64 | 2 | block 62 内含 128→64 |
| 66–69 | L0 | 32 | 1 | block 66 内含 64→32 |
| 70 | full | 32 | 1 | 两组 full-resolution merge scale、mixer、f16 32→4 head |

输入经过 block 0 后池化进入 L0；其他 encoder transition 按运行图做 2×2 平均池化与升维。decoder 在低分辨率做投影、nearest 2× 上采样，并与相应 encoder skip 合成。block 69→70 使用两组 `[32]` scale，分别作用于上采样后的 block 69 和 block 0。池化和 nearest 没有可训练权重。[U1] [S6]

不把 71 个编号生成为 71 个同构 `TransformerBlock`。`blocks.39` 必须是 transition 模块，不能获得 FFN、QKV、prior 或虚构的 LoRA targets。

### 4.2 Mixer 与 FFN 变体

每个 mixer 按 **FFN → attention** 顺序执行，两条残差各有逐通道 f16 scale。没有 Linear bias、LayerNorm 仿射或 router。cosine Q/K normalization 必须保留；“没有 LayerNorm 仿射”不等于“没有归一化”。[U1] [S6]

| C | FFN 的逻辑结构 |
|---:|---|
| 32 | 32→128，SiLU，再 128→32；没有第三层 |
| 64/128/256 | E=C/32 路；每路读完整 C，做 C→128，SiLU，再 128→32；拼接成 C，最后 C→C。最终 contraction 接残差 |
| 512 | 8 路；每路 512→64→256→64；**SiLU 仅在 256 宽中间输出之后**；拼成 512，再 512→512 并接残差 |
| 1024 | 1024→4096，SiLU，再 4096→1024 |

512 变体的第一层也可表示为一个 512→512 projection 后切成 8 路，但权重映射、支路顺序和 publication 必须保持等价。64/128/256 变体不是先把输入切成小组；每一路都读取完整的 C。所有分支对每个 token 执行，不存在 top-k、router loss 或 load-balancing loss。

### 4.3 注意力与相位

QKV 的逻辑输出维采用 head-major：head h 的 96 列依次是 Q32、K32、V32，不能 reshape 成全局 `[Q_all,K_all,V_all]`。window prior 的逻辑形状为 `[heads,64,64]`；ViT 没有 prior。[U1]

窗口相位按 `(0,0)、(-4,-4)、(-4,0)、(0,-4)` 循环。同一分辨率的 decoder 继续 encoder 的计数；不是每个 stage 都重置，也不是普通双相位 Swin。block 0 为 full 级 phase 0，block 70 为 full 级 phase 1。[S6]

window 与 ViT 的 Q scaling、softmax 归约、padding 处理存在差别，见第 6 节。不得无依据加入 RoPE、RMSNorm、SwiGLU、learned timestep embedding 或任何新 conditioning 分支。

## 5. 磁盘布局与 canonical 转换

### 5.1 固定 schema 的转换器

主路径接受 `manifest.json + 11 个 stage 文件`，按固定 profile 验证与拆解。可增加装有同一 packed records 的 safetensors 容器适配，但只有 record 对应关系与载荷字节验证通过才允许接入。

本阶段是**确定性转换与回归测试**，不是重新进行开放式架构搜索。禁止未知布局猜测、随机补层、宽松 shape reshape 或 `strict=False` 兜底。

Inspector MUST 在 CPU 上完成文件边界、长度、源哈希、record inventory、全零区域与数值指纹检查。manifest 中的文件路径必须解析到授权的 source 目录内，拒绝目录穿越、越界区间和超出预期资源上限的声明。直接解析 safetensors header/安全数组；不运行远程代码，不以 pickle 作为权重格式兜底。

### 5.2 Record 拆分规则

32/64/128/256 的普通 window record 顺序：[U1]

```text
FFN E4M3 matrices
16-byte zero padding
FFN skip scale: f16[C]
16-byte zero padding
QKV: E4M3[C,3C], head-major
prior: heads × 8192 bytes
temperature: f32[heads], padded to 16-byte alignment
attention output projection: E4M3[C,C]
attention skip scale: f16[C]
16-byte zero padding，或 encoder 的 C→2C transition
```

block 4 的 transition 后仍有 16 个零字节；更宽的三个对应 encoder 末块没有这段额外尾零。

特殊记录必须有独立 descriptor：block 0 在 W2 与 FFN scale 之间插入 f16 输入 adapter；48/56/62/66 插入上采样投影与第二组 skip scale；block 70 用两组 f16[32] 替代 QKV 前的一段 16 字节 padding，并在 attention skip 后隔 16 个零字节存 head。不能只用普通 record 长度公式处理这些变体。

512 mixer 的四段布局：

| record | 字节 | 内容 |
|---|---:|---|
| layer0 | 524,288 | 8 路的 512→64、64→256、256→64 |
| layer1 | 263,168 | 512→512 与 f16 skip[512] |
| layer2 | 917,568 | QKV 512×1536、16 个 prior、16 个 f32 温度 |
| layer3 | 263,168 | projection 512×512 与 f16 skip[512] |

ViT 的五段布局：

| record | 字节 | 内容 |
|---|---:|---|
| layer0 | 4,194,320 | 1024×4096 E4M3，尾部 16 个零字节 |
| layer1 | 4,196,352 | 4096×1024 E4M3，f16 skip[1024] |
| layer2 | 3,145,856 | **32 个 f32 温度在前**，再 QKV 1024×3072 |
| layer3 | 2 | 原样保留，不参与前向 |
| layer4 | 1,050,624 | projection 1024×1024，f16 skip[1024] |

block 30 layer4 为 512×1024 E4M3，再加 16 个尾部零字节，记录长度 524,304。block 39 为 1024×512 E4M3 加 f16[512] skip scale，记录长度 525,312。所有偏移计算必须最终与 manifest 的每条长度相等。

### 5.3 E4M3 fragment 与通道置换

下面的 `/` 在索引公式中统一表示整数除法；逻辑矩阵为 `[K,N]`：[U1]

```text
kTile      = k // 32
nTile      = n // 128
nHalf      = (n % 128) // 64
nGroup     = (n % 64) // 16
lane       = (n % 8) * 4 + (k % 16) // 4
byteInLane = ((n % 16) // 8) * 8 + ((k % 32) // 16) * 4 + k % 4

offset = kTile * (N * 32)
       + nTile * 4096 + nHalf * 2048 + nGroup * 512
       + lane * 16 + byteInLane
```

每个 32 通道组内还有 bit1..bit3 的旋转。转换器 MUST 按 OpenDLSS 加载器的映射方向还原**自然输入通道序**，并用完整 32 项置换表及其逆表测试，不能只依据“旋转”二字猜方向。[S7]

应分别测试 fragment addressing、通道置换与矩阵转置，避免一次往返中两个相反错误互相抵消。使用 one-hot 输入与可区分的合成矩阵，核对 `y = x @ W[K,N]`。

E4M3 byte 是浮点编码，不是 0–255 的数值。先解码/按 dtype 解释，再转 FP32；不能把 U8 `.to(float32)` 当成去量化。没有证据时不添加额外的 per-tensor dequant scale。

### 5.4 f16 GEMM、prior 与温度

两个 f16 GEMM 使用 m16n8k16、16×16 tile。逻辑索引可按以下 half 槽位公式恢复：[S7]

```text
tile     = (k // 16) * ceil(N / 16) + n // 16
lane     = (n % 8) * 4 + (k % 8) // 2
fragment = (2 if k % 16 >= 8 else 0) + k % 2
half_slot = tile * 256 + lane * 8 + ((n % 16) // 8) * 4 + fragment
```

输入 16→32 占满 1024 字节；输出 32→4 也占 1024 字节，但 N 补至 16，只有 128 个 half 是逻辑参数，另外 384 个 half 保留为零。

prior 是 f16，但不是普通行主序矩阵。必须先解 16×16 fragment，再将 query 和 key 的 4×4 tiled token 序还原为自然 8×8 token 序：

```text
x = t % 8; y = t // 8
physical_token = (y // 4) * 32 + (x // 4) * 16 + (y % 4) * 4 + x % 4
```

canonical prior 统一为 `[head, query_natural, key_natural]`。运行时为 native reference 选择其他物理排列时，由 backend 显式重排，不把运行时布局混入 canonical 参数名。f32 温度保持 FP32，不当成 FP8 权重缩放系数。

### 5.5 Canonical 表示

```text
canonical_dlssnr/
├── model.safetensors
├── model_config.json
├── preprocessing.json
├── numerics.json
├── source_manifest.json
├── conversion_report.json
└── opaque_records.safetensors
```

canonical 的默认参数存储为 FP32，以无损容纳源 E4M3、f16 及 f32 数值。不能因为 E4M3 矩阵可精确扩展到 BF16，就把原有 f16 prior/scale 与 f32 温度也无损性地称作 BF16；它们的精度不同。

普通 Linear 权重存为 `[out_features,in_features]`，由 `[K,N]` 明确转置。典型参数路径为拟定义的 `blocks.1.ffn.fc1.weight`、`blocks.23.ffn.branches.0.in_proj.weight`、`blocks.31.attn.qkv.weight` 和 `blocks.39.proj.weight`。每个 canonical 参数都必须有原始 record、子区间、原 dtype 和排列映射。

head 在可训练结构中拆为 RGB `[3,32]` 与 logit `[1,32]` 两个无 bias Linear，导出时按原顺序重新拼为 `[4,32]`。这允许单帧模式真正冻结 logit 分支，同时不改变逻辑输出顺序。`native_reference` 将两组参数重新拼合后按原始 32→4 GEMM 执行；参数所有权的拆分不能成为改变原生 GEMM/舍入的理由。

`model_config.json` MUST 包含完整 block descriptor、FFN variant、transition 关联、相位、QKV 排列、head 定义、schema 与 profile。8 个未读标量使用 U8 原始字节保存，不注册为 `nn.Parameter`。

### 5.6 转换验收

未训练 round-trip MUST 从 canonical 与 opaque 数据重建 stage：每个 byte 与源相同，包含 E4M3 的负零、padding、head 空槽、未读标量与独立 blend scale。canonical 转换不能只保存“数值近似相等”而丢掉来源的符号零。

通过自洽 round-trip 仍不足以证明计算排列正确；同时需要独立 one-hot 布局测试及匹配 reference 的逐层 fixture。所有 source hash 和结果写入报告。

## 6. 数值 profile 与兼容性等级

### 6.1 两条明确的执行路径

| profile | 用途 | 约束 |
|---|---|---|
| `native_reference` | 前向验证、原生运行时对照与导出验收 | 复现已固定参考实现的舍入、publication、近似算子及运行语义；不要求可反传 |
| `train_surrogate` | 自动求导与训练 | 使用 canonical 参数与显式可微近似；所有偏差必须记录并与 reference 比较 |

OpenDLSS 数值文档将 E4M3/half 边界、低精度 accumulator、残差注入位置、近似 SiLU 和 softmax 归约列为网络行为的一部分。因此，“解包后换成 `nn.Linear + SiLU + SDPA`”不是自动成立的原生等价实现。[S8]

`numerics.json` MUST 为每个 profile 记录算子实现版本、计算 dtype、舍入点、Q/K 归一化规则、softmax 变体、history publication 与所有 surrogate backward。只有名称相同、内容不同的配置必须有不同内容哈希。

### 6.2 Native reference 的最低语义要求

必须保留或以 fixture 证明等效的要点包括：[S6] [S8]

- 激活 publication 经过 half 再到 E4M3，保留规定的舍入、饱和和符号零语义；不能把 f32→E4M3 当成自然等价。
- FFN/attention 残差作为相应 accumulator 的初值，而非任意放到 GEMM 结束后相加；保留 K 归约与 split-K 次序。
- 32 通道与更宽 block 对 FFN 残差中间值的 half/E4M3 使用不对称，不能统一量化后继续。
- 近似 SiLU 的半精度运算与普通 `SiLU` 不直接等价；window/global 的指数近似、Q scaling、softmax 归约及归一化位置分别处理。
- window 超出 field 的零 token 按参考规则参与注意力分母；ViT 的 token padding 修正遵循其自己的规则。
- 特定零 Q/K 行导致的已定义非有限中间值处理与真正训练发散分开；不能在训练图任意位置用 `nan_to_num` 掩盖错误。

精确常数、查表与低精度运算由固定的 reference 实现/fixtures 导入，不从 PDF、网络名称或通用 transformer 习惯推断。数值实现的每次优化都必须重新跑对应 parity。

### 6.3 Surrogate 的范围与测试

首个 `train_surrogate` SHOULD 先建立 FP32 浮点基线，然后逐项评估替换带来的误差。可以使用标准可微算子，但它们必须以显式 approximation flags 记录，而不是静默取代 native 路径。

需要近似量化前向时，可以增加 fake quant/STE 或自定义 backward；必须注明前向与反向各自的函数，测试梯度有限性、微型优化及复算一致性。舍入/bitcast 不具备可直接沿用的普通导数；“存在 gradient tensor”不是梯度近似有效的证据。

冻结 LoRA 基座不能对整个模型使用 `no_grad`；可微路径仍需穿过冻结矩阵到达 adapter。

### 6.4 三个独立的验收标签

`layout_verified`：源布局、数值和映射通过验证。

`float_validated`：浮点训练后端相对 reference 的前向与时序误差在预先声明范围内；**不表示 bit-exact**。

`native_export_validated`：训练后的模型重新量化/打包，经指定 native runtime 加载和回放，达到预先声明门槛。

这些标签必须分别保存。不能因为其中一个成立自动设置其余两个。仅近似实验通过时可以标记 `experimental_surrogate`，但不能发布成已兼容的 NR 权重。

### 6.5 初始工程门槛

在固定验证集、输入、控制、seed、geometry 和 history 规则下，初始 surrogate 对照门槛沿用以下建议值：每个用例的最终 proxy RGB 上 `MAE ≤ 1/255`、`PSNR ≥ 45 dB`、`P99 absolute error ≤ 4/255`，同时报告 head 三通道、logit 与 blend weight 的误差。

这些是**建议的项目验收值，不是已测结果或官方标准**。黑白饱和输出可能掩盖 head 偏差，因此必须同时保存 preclamp/head 诊断，不得仅比较裁剪后的图。门槛修改必须先修改有版本的验收配置，不能失败后静默放宽。

### 6.6 训练目标部署路径

`float_runtime`：部署到本适配器明确支持的浮点 runtime。允许已验证的 surrogate，但必须报告相对原生基线的变化。

`native_roundtrip`：计划重新量化到 E4M3 与原生 fragment。必须从早期引入量化回放/导出评估；浮点更新可能被量化抹去，不能只在训练完成后检查一次。v1 canonical 训练阶段可记录该意图，但在量化阶段通过前不得宣称部署完成。

`--fp8_base` 等通用训练器选项不能实现这个区别：源文件 FP8 存储、计算时 FP8、以及量化感知训练是三件不同的事情。

## 7. 特征、几何与帧处理接口

### 7.1 16 通道输入

原生特征为 f32 `[pixel,16]`；canonical pipeline 使用 `[B,16,Hpad,Wpad]`，只改变逻辑存储视图，不改变通道含义。[U1] [S6][S9]

| lane | 含义 |
|---|---|
| 0–2 | 由显式 seed 与 padded 像素坐标生成的三路 Gaussian 特征 |
| 3 | 常数 1，不是可猜测的 diffusion timestep |
| 4–6 | 当前 display proxy 的中心化 RGB |
| 7–9 | 重投影的上一帧**模型合成输出**的中心化 RGB |
| 10 | style id 的 profile 编码；参考表达为 style id / 128 |
| 11 | local tone 编码 |
| 12–14 | structure / skin / auto-mask 条件三元组 |
| 15 | 常数 0 |

中心化按参考规则：

```text
centered = f16((f16(proxy) - 0.5) * 0.125)
```

不是常见的 `[-1,1]` 归一化。输入 adapter 的恒零 lane 15 对应逻辑权重行全零；在 PyTorch `[out,in]` 格式下，它是 `weight[:,15]`。

### 7.2 颜色域与控制条件

v1 数据仅接受明确的 `srgb_proxy`：source 与 target 是同一 proxy 域的 `[0,1]` 值。普通截图经过何种 tone map 不可猜测。`linear_hdr`、显示端 ACES、HDR tone upgrade 等需要独立导入/profile 验证，不属于默认训练链。

控制条件的主路径是**已编码的 lanes 10–14**，缓存形状为 `[5,H,W]`。同一组空间恒定条件可由一个显式 5 元向量广播，但必须有 `controls_encoding` 与来源记录。未知默认值不得命名成 `checkpoint_default` 后自动补零。

v1 不从 caption 生成条件，不根据 UI 滑杆范围猜映射。UI→feature 编码器只有在提供固定规则和参考用例后才开放。控制数据缺失必须报错；不能把缺失数据与合法的零值混淆。

### 7.3 GeometryResolver

MUST 区分 valid rectangle、padded field、六级尺寸、各 block 的窗口相位、输入镜像区与 attention 零 token 区。[S6]

层尺寸递推按参考的 `align_up(ceil(previous/2),4)`，但 **field 本身不能只按 64 的倍数估算**。使用固定 reference 的完整 geometry resolver，并为其保留代码版本与输入输出 fixture。

最少包含下列宽×高用例：[S6]

| valid W×H | field W×H | 说明 |
|---|---|---|
| 512×512 | 576×512 | L0–L5：288×256、144×128、72×64、36×32、20×16、12×8 |
| 768×768 | 832×768 | 验证额外 field 列 |
| 644×768 | 768×768 | 非整齐宽度 |
| 1920×1080 | 1920×1152 | 非方形视频 |
| 3840×2160 | 3840×2176 | 高分辨率推理 |

输入 field 外围是镜像图像采样，但噪声使用各自的 padded 坐标。window 出界 token 则按零向量与 reference 分母规则处理，两者不能混用。

首版兼容范围要求 valid 两个维度均至少 33，且通过 resolver 验证；这是参考实现的小尺寸兼容边界，不是从权重推导出的理论网络下限。未支持尺寸明确拒绝，不静默缩放。

训练 crop 是独立的渲染视图，会改变全局上下文与 field，不能声称 crop 推理等价于整帧对应区域。噪声坐标与 crop ID 的处理在第 10 节固定。

### 7.4 Pipeline 契约

```python
# 目标接口说明，不是已经实现的代码。
output = pipeline.forward_frame(
    source_proxy=source,        # [B,3,H,W]
    history=history_state,      # None 或显式 NRHistoryState
    motion=motion_px,          # [B,2,H,W], current -> previous
    history_valid=valid,        # [B,1,H,W]
    reset=reset,               # [B]
    controls=control_features, # [B,5,H,W], 已编码 lanes 10–14
    frame_seed=frame_seed,     # 每个样本的确定性 seed
)
```

返回对象 MUST 包含 `raw_head`、`neural_preclamp`、`neural_proxy`、`rendered_proxy`、`blend_weight`、`next_history` 与 geometry/valid region。输出给 loss/导出前必须裁回 valid rectangle；不能把反射 padding 纳入监督。

`NRModel.forward` 不包含跨帧可变全局状态。history 显式属于 sequence；pipeline 不把上一 batch 的结果藏在 module 成员里。训练、采样、评估和推理共用同一 pipeline 实现。

## 8. 数据集与缓存

### 8.1 配对数据要求

source/target 必须几何配准，target 可以是同场景目标渲染、人工处理结果或有来源记录的教师输出。不配对照片、caption 或只有 source 的集合不能自动变成监督训练集。

禁止自动令 `target = source` 后声称训练增强能力。源与目标的编码、尺寸、相机视角/配准关系和来源记录必须可审计。教师生成数据记录教师版本及条件；它不等于学生在线历史。

### 8.2 JSONL manifest

一行表示一个独立、有序片段。以下为目标格式；示例中的各文件需由实际数据提供：

```json
{"schema":"dlssnr_pairs_v1","sample_id":"shot001_clip000","sequence_id":"shot001","source_encoding":"srgb_proxy","target_encoding":"srgb_proxy","controls_encoding":"dlssnr_lanes_10_14_v1","frames":[{"frame_index":0,"input_path":"input/0000.png","target_path":"target/0000.png","controls_path":"controls/0000.npy","reset":true},{"frame_index":1,"input_path":"input/0001.png","target_path":"target/0001.png","controls_path":"controls/0001.npy","motion_path":"motion/0001.npy","history_valid_path":"masks/history_0001.png","temporal_valid_path":"masks/temporal_0001.png","reset":false}]}
```

相对路径以 JSONL 文件所在目录为基准。配置文件中的路径以该 TOML 所在目录为基准。检查 sample ID 唯一、sequence 分组、文件存在、frame index 严格递增、尺寸不变和 reset 合法。

上面仅展示两帧的字段关系；第 18 节时序训练配置要求四帧样本，数据校验必须拒绝以两帧示例直接运行四帧配置。单帧模式每行只有一帧并 reset。

flow 数组输入明确为 `[H,W,2]` 或通过 manifest 声明的布局；controls 为 `[5,H,W]`；禁止仅凭某一维等于 2/5 来猜转置。数组必须无 pickle、有限值，且通过单位/方向校验。

### 8.3 统一 batch

NR batch 采用 `[B,T,C,H,W]`：

| 字段 | dtype / shape | 语义 |
|---|---|---|
| `nr_source` | float `[B,T,3,H,W]` | 当前 proxy |
| `nr_target` | float `[B,T,3,H,W]` | 配对目标 |
| `nr_motion` | f32 `[B,T,2,H,W]` | current-to-previous 像素位移 |
| `nr_controls` | f32 `[B,T,5,H,W]` | 已编码控制 lanes |
| `nr_history_valid` | bool `[B,T,1,H,W]` | 历史可采样性 |
| `nr_temporal_valid` | bool `[B,T,1,H,W]` | 可靠时序监督区域 |
| `nr_loss_mask` | float `[B,T,1,H,W]` | 有效监督，默认 valid region 内为 1 |
| `nr_reset` | bool `[B,T]` | 首帧/切镜头 reset |
| `nr_time_valid` | bool `[B,T]` | 真实帧标记；v1 默认丢弃短尾片段 |
| metadata | IDs、frame index、crop、编码、指纹 | 复现与来源 |

单帧/显式 reset 帧允许不提供 motion，collator 可填零供统一张量接口；**history 无效由 reset/valid 决定，不由零 flow 决定**。时序模式非 reset 帧缺 flow 必须报错。

`history_valid` 与 `temporal_valid` 不同。前者为真不代表没有显露/遮挡错误；时序损失启用时必须提供可靠 temporal mask。没有时，只能显式将该损失权重设为 0，不可静默伪造全有效区域。关闭 temporal loss 不等于取消学生历史回放；最终图像损失仍可监督混合分支。

### 8.4 增强、采样与划分

同一 clip 的 source、target、flow、controls 和 mask 使用完全相同的 crop/resize/flip；不能逐帧独立随机增强。flow 随 resize 按横纵比例缩放；水平翻转反转 dx，垂直翻转反转 dy。

裁剪/变换后重新计算历史采样是否仍在 valid 矩形内；插值即使有 border clamp，也不能把原先出界的位置重新标为有效。缩放 mask 使用明确的保守规则，不把二值有效性任意线性插值后当作布尔值。

v1 使用固定 crop/bucket 并提前缓存；新增 crop 必须有新的样本/缓存指纹。source 与 target 禁止独立曝光或颜色增强。训练/验证按 scene 或 sequence 隔离，不随机拆相邻帧；manifest 验证器检查 sequence ID 交集。

每个 dataset item 是一个 clip，不是预组装 batch；`NRBucketBatchSampler` 与 collator 负责同形状组批。不得再套 Musubi 某些预 batch dataset 的额外 `batch_size=1`，避免出现双重 batch 维度。

### 8.5 像素缓存

缓存只保存 source、target、flow、controls、mask、固定变换、几何和来源；**禁止缓存会随着当前学生权重更新而过期的历史预测**。固定教师输出可作为 target 缓存，但必须带 teacher hash。

默认保留输入精度：浮点数据为 FP32，原始无损 U8 图像可保留 U8 后按明确规则解码。若另提供 FP16 缓存模式，必须声明其有损性并单独生成 key；不能用于声称源数据逐位不变的 parity fixture。flow 保持 FP32。

cache key 至少含 schema、各源文件指纹、编码、controls 版本、geometry/preprocess 版本、crop/resize/flip、frame index、flow 坐标契约和 mask 版本。模型权重本身不应导致纯像素缓存失效，但预处理/profile 变化必须失效。

## 9. 输出合成与监督损失

### 9.1 合成公式

当前 proxy 为 x，raw head 的 RGB 为 r，logit 为 a，重投影历史为 h：[S6] [S9]

```text
n_raw = x + r / 4
n     = clamp(n_raw, 0, 1)
w     = clamp(sigmoid(a) * blend_scale, 0, 1)
w     = 0，若 reset 或无有效历史
output = (1 - w) * n + w * h
```

`blend_scale` 从 checkpoint 读取，U1 初始值为约 **0.739746**，确切 half 值为 **0.73974609375**。不要将初始值写死成不可保存的常量；它是有单独来源的合成参数。

history 输入无效时回退到当前 proxy；不能读取任意上一帧像素或黑色历史。下一帧的状态来自**合成后的输出**，不是 residual、上一帧 source 或默认 GT。

reference 的 history 存储使用向零截断到 half；一般 `.half()` 的舍入不直接等价。训练可使用有文档的 surrogate/STE，但前向偏差与循环累积必须测试。[S9]

### 9.2 训练目标不是默认 diffusion target

本项目执行现有单步渲染器的配对监督微调，不向 target 加通用 timestep 噪声、不回归 epsilon/velocity、不使用 SD3 loss weighting/flow shift。该决策不意味着单步模型不能由 diffusion 蒸馏而来，只是不在没有原始配方时伪造那套训练过程。

Gaussian 输入 lanes 0–2 仍属于模型输入，不能因为不使用 diffusion loss 就删除。

### 9.3 起始损失

设 `rho(e) = sqrt(e² + 1e-6) - 1e-3`。mask reduction 使用实际有效元素数，不是 padded 图像面积。

```text
L_pre  = masked_mean(rho(n_raw - target))
L_out  = masked_mean(rho(output - target))
L_edge = masked_mean(rho(grad(output) - grad(target)))

L_temp = masked_mean(rho(
             [output_t - Warp(output_(t-1), motion_t)]
           - [target_t - Warp(target_(t-1), motion_t)]
         ))

L = 1.0 * L_pre + 1.0 * L_out + 0.05 * L_edge + 0.10 * L_temp
```

这是建议的调试起点，不是官方 recipe 或已验证最优值。`L_pre` 是为裁剪前分支提供梯度的**辅助约束**，会额外推动未混合结果接近 target；它不是由最终合成损失严格推导出的必要项，正式调优需要做权重消融。

`L_edge` v1 用明确的横纵一阶差分，mask 取相邻有效像素交集并按差分元素数归一化。`L_temp` 只用于有效相邻帧，比较与目标一致的时间变化，而非强迫所有亮度/材质变化消失。GT warp 使用相同坐标与插值规则，但不对 GT 强制施加模型 history 的有损存储。

RGB reduction 分母为 `3 × mask 权重和`；对时间与 batch 合并时仍按有效元素总数计。burn-in 帧不计单帧监督，reset 后首帧不计跨边界 temporal loss。

### 9.4 无效区域与诊断

完全无监督样本在预检查阶段拒绝；训练中有效数为零需要显式处理。DDP 的 mask 分母应按实际全局有效数归一化；某一 rank 不能自行跳过 backward。具体同步策略见第 13 节。

记录 `loss/pre`、`loss/out`、`loss/edge`、`loss/temporal`、有效像素/帧比例、blend 分位数、preclamp 饱和率、各参数组 gradient norm、非有限值和 update norm。未测指标不填零冒充结果。

感知损失默认关闭；增加时必须固定模型、权重 hash、颜色归一化和 reduction。训练期的深度/法线等辅助监督不能无依据加入原生 16 通道输入。

## 10. 时序训练与推理状态机

### 10.1 Motion 契约

内部 backward flow 为 current-to-previous，单位为 valid 图像像素，x 向右、y 向下。前后帧保持同分辨率：

```text
previous_x = current_x + dx
previous_y = current_y + dy

gx = 2 * (previous_x + 0.5) / W - 1
gy = 2 * (previous_y + 0.5) / H - 1
```

最后两式表示 pixel-center、`align_corners=False` 的归一化坐标。UV/NDC/jitter 由离线导入器显式转换，记录单位、符号、y 方向与抖动约定，不能自动猜测。

reference 使用 five-tap Catmull–Rom 重建历史，输入 history lanes 与输出合成都应使用同一采样值。[S9] bilinear 只能作为命名的近似模式；不能训练用 bilinear、部署换 Catmull–Rom 却不回归测试。

### 10.2 v1 Clip 与 TBPTT

默认单帧：`sequence_length=1, burn_in=0, tbptt_length=1`。

默认时序：`sequence_length=4, burn_in=2, tbptt_length=2`。前两帧按时间顺序在 `no_grad` 下用当前学生模型建立 history；边界 detach 后，后两帧构成一个可反传段，段内不逐帧 detach。

**为使首版状态、显存和恢复行为明确，一个 microbatch 只包含一个可反传段，要求 `sequence_length = burn_in + tbptt_length`。** 更长源视频由数据层切成满足该式的 clip；跨 clip 不保留学生 history。多段在线 TBPTT、跨 batch stateful sampler 属于后续扩展，不在未实现时暴露参数。

这保证首版真的具有有限展开，而不是仅 detach history、却仍把所有历史 loss 的图保留到一个无限长序列结束。一次 optimizer update 可累积多个独立 clip 的 microbatch。

### 10.3 Reset 与边界

片段首帧、切镜头、resize、seek、模型切换或 preprocessing profile 变化都 reset。batch 内每个 sequence 独立；reset 必须切断相应样本 history 的梯度和 temporal pair，不影响其他样本。

首帧历史特征回退到当前 proxy，合成历史权重为零。短尾片段默认丢弃；不能重复最后一帧后仍标记成真实运动数据。采样/评估使用独立状态，不得污染训练。

默认 history 由学生自身预测构建。teacher forcing 不属于 v1 默认；若未来开放，必须有明确比例/调度，验证仍使用学生闭环，不能用 GT history 的结果证明闭环稳定。

### 10.4 随机性与重算

噪声包含两层契约：外层产生每帧 seed，内层严格按 reference 的 padded 坐标哈希和 Gaussian 规则生成 lanes。

训练 seed policy v1 使用稳定哈希组合 `global_seed + epoch + sample_id + crop_id + frame_index`，不得使用跨进程不稳定的 Python `hash()`；不依赖 worker 调度顺序。评估不随 epoch 改 seed。具体哈希序列化、截取位数及算法版本写入配置。

crop 视为独立视图，噪声使用该视图 padded 坐标，crop ID 纳入 seed；不宣称与整帧噪声逐像素相同。gradient checkpointing 的重算必须复用相同 features 或可复现 seed，不能再次抽样另一份噪声。

### 10.5 单帧模式的严格限制

没有有效 history 时，合成对 logit 与 blend scale 无有效监督。单帧模式冻结 `head_logit` 和独立 `blend_scale`；head 已拆成两个 Linear，所以不采用对参数切片调用 `requires_grad_(False)` 的无效做法。

冻结最后一行仍不能保证 trunk 改动后时序行为不变。单帧产物必须标记 `temporal_trained=false`，并仍执行闭环视频回归，不能声称已完成时序适配。

`temporal_trained=true` 仅说明执行了学生历史条件下的时序训练；是否通过质量验收由独立的 `temporal_validated` 记录决定。

## 11. 全量训练与参数管理

### 11.1 全量是首个训练基线

先使用固定样本与噪声建立全量优化基线，再评估 LoRA 的容量/冻结策略。这样在 loss 无法下降时，不必同时怀疑 rank、target 选择和参数冻结。

默认 FP32 参数、FP32 AdamW 状态、`mixed_precision=no`、学习率 `1e-5`、constant LR；这些是调试起点，不是正式训练最优设置。通过数值与梯度验收后才开放 BF16/FP16 计算。

“全量”表示所有可参与目标的模块均可更新，但仍遵守恒零通道及单帧无监督分支等约束；实际可更新集合必须显式打印，不能仅报告全部槽位数。

### 11.2 参数组

| 组 | 内容 | 默认 LR multiplier | 默认 weight decay |
|---|---|---:|---:|
| matrices | FFN/QKV/projection/transition | 1.0 | 0.0 |
| priors | window attention prior | 0.1 | 0.0 |
| scales | FFN/attention/transition/full-resolution skip scales、温度 | 0.1 | 0.0 |
| input_rgb_head | 输入 adapter 与 RGB head | 1.0 | 0.0 |
| temporal_head | logit head，仅时序模式更新 | 1.0 | 0.0 |
| temporal_blend | 独立 blend scale，仅时序模式更新 | 0.1 | 0.0 |

这些倍率是可修改的工程起点。参数组必须互斥、无遗漏、无重复；参数组配置与可训练映射写入 checkpoint。增加 decay 时必须分别评估矩阵与非矩阵参数，不能默认把所有 scale/prior 当作普通权重衰减。

### 11.3 不更新的区域与 opaque 字段

`input_adapter.weight[:,15]` 保持源值零。v1 可使用完整参数矩阵配合明确的更新约束：梯度 mask、相应 optimizer moment 清零、step 后恢复源值，并测试 weight decay、resume、adapter merge 后仍保持不变。只做 gradient mask 不足以阻止已有 momentum 或 decoupled weight decay。

head 的 RGB/logit 参数已经拆分；单帧 logit 与 blend scale 直接不进入 optimizer。所有 opaque scalars、padding 和 head fragment 空槽始终不参与求导。

不能要求冻结前后每个元素都出现非零梯度。有效性测试应证明声明参与的分支在覆盖其条件的输入下获得梯度/更新，并解释合法的恒零通道和首步 LoRA 零梯度。

### 11.4 更新正确性断言

启动时核对：

```text
optimizer 中参数集合 == 配置声明的可训练 Parameter 集合
每个 Parameter 只出现一次
被冻结模块不进入 optimizer
参数 master dtype 符合配置
无效更新区域的 mask / moment 策略已注册
```

至少执行一次 step 前后对比：有效参数变化、冻结参数完全不变、梯度有限、保存再加载后同一 backend 前向一致。不能只观察 loss 日志便宣称全量训练成功。

## 12. LoRA 适配

### 12.1 目标与默认策略

拟新增 `musubi_tuner.networks.lora_dlssnr`，只注入 canonical 中语义明确的 Linear。优先 attention QKV/proj 与各 FFN Linear；默认排除输入 adapter、RGB/logit head、prior、温度、skip scale、blend scale 和 transition。

首个 LoRA profile 为 **`vit_only`**：仅 block 31–38 的 QKV、attention projection、FFN expand 与 contract，rank=16、alpha=16、dropout=0。rank/alpha 是 adapter 设置，不改变原模型 head 维或其他结构。

当这 8 个 ViT block 各含上述四个 Linear 时，普通 LoRA 的新增矩阵元素为 **2,097,152**（不计 alpha buffer）；转换器与 injection report 必须独立复算，不因该数字看似正确就忽略 targets。

### 12.2 按宽度配置，而非全网 rank 16

可选第二个 profile `multiscale` 的建议默认值：

| C | rank | alpha |
|---:|---:|---:|
| 32 | 2 | 2 |
| 64 | 4 | 4 |
| 128 | 8 | 8 |
| 256 | 8 | 8 |
| 512 | 16 | 16 |
| 1024 | 16 | 16 |

这是待验证起点，不保证所有 stage 都必须适配。32→32 的 rank 16 LoRA 有 `16×(32+32)=1024` 个元素，与原矩阵相同，没有参数量优势；因此不采用统一 rank 16 的默认值。

任何 rank 均不得大于相应 Linear 的 `min(in,out)`；FFN 每条支路按实际矩阵尺寸检查，不只按所属 C 检查。未知 stage、block 39 attention target 或零命中规则直接失败。

### 12.3 QKV 与 FFN 规则

首版 QKV 作为一个 head-major fused Linear 注入，避免误用默认全 Q/全 K/全 V 的拆分辅助函数。若未来支持 split-QKV，必须使用原有 interleaved 索引显式 gather/scatter，并与 fused 版本独立测试；两种模式禁止同时注入。

512 FFN 的每个 branch Linear、64/128/256 的完整输入 bottleneck 与最后 contraction 都必须正确命名。不要根据字符串 `expert` 引入 router，也不能把已有 branch 合成一层后忽略激活/publication 差异。

### 12.4 参数、保存与合并

base 参数冻结，adapter 参数默认 FP32，建议初始 LR `1e-4`。前向仍可反传到 adapter。一个 adapter 的参数只注册一次，避免同时在 base 和独立 network 中重复注册、重复进入 optimizer 或重复 DDP 包装。

零初始化 B 的普通 LoRA，第一步 A 可以合法地没有非零梯度；测试应在后续 step 验证 A/B 更新，而不是据此误判断图。

注入报告必须列出每个 target、形状、rank、alpha、参数量、排除规则及未命中规则。adapter 文件记录 base identity 与完整 target map。未知 base 或 schema 不能直接加载。

先在 FP32 canonical 上合并 `W + alpha/r × B@A`，再按目标 runtime 导出。先验证同一浮点 backend 下 merged/unmerged 一致性，重新量化误差另行统计；不能把两种误差合并成一个宽松容差。

## 13. Musubi 宿主集成

### 13.1 入口与 runner

拟新增：

```text
dlssnr_train.py            # 全量入口
dlssnr_train_network.py    # LoRA 入口
```

两个入口使用同一 `NRSupervisedTrainer`，仅改变 trainable policy、adapter 创建和保存目标。核心训练 step 不依赖 Accelerate，宿主负责反传、累积、同步、optimizer、scheduler 与日志。

默认 Musubi 基类会冻结 transformer，优化器参数由 network 构建，循环还会创建 flow scheduler/latent noise；这些行为与 NR 全量监督训练不一致，因此主线不调用该 `.train()` 路径。[S1]

### 13.2 复用公共服务，但隔离不相干选项

在拟新增 `training/dlssnr_services.py` 中建立薄适配层，复用当前版本实际提供的能力：

| 服务 | 复用方式 |
|---|---|
| Accelerate / 公共运行设置 | 调用已核对的 accelerator helper，明确映射其所需字段 |
| optimizer 与 LR scheduler | 对现有 `NetworkTrainer.get_optimizer/get_lr_scheduler` 做窄的组合封装，不调用模型/数据/训练循环方法 |
| 日志与 tracker | 复用 tracker 初始化/日志工具，输出 NR 指标 |
| safetensors / 状态管理 | 复用安全序列化与训练状态工具；完整 NR config/来源另行保存 |
| 图像/视频输出 | 复用无模型依赖的输出工具，不调用 diffusion sampler |

需要的 args 由 NR 配置显式映射，不继承整个 diffusion parser 后允许用户传入大量无效开关。若公共 helper 依赖与 NR 无关的运行行为，应在适配层隔离或提交最小的受测提取，不能虚构一个当前不存在的 helper 已经可直接调用。

LR scheduler 以**实际成功 optimizer updates**计步。若复用的 helper 含进程数/累积步数换算，适配层必须明确转换，并以累积等价测试证明没有重复放大步数。`max_train_steps` 也表示 optimizer updates，不表示帧数或 dataloader iterations。

### 13.3 一次清晰的训练对象包装

`NRTrainModule` 包含 pipeline/model（以及注入后的 adapters），其 `forward(batch)` 负责完整 clip 的 burn-in、有限时序展开和 loss 统计。宿主将该对象作为唯一的 model 交给 Accelerator/DDP。

全量与 LoRA 都从它的可训练参数集合构建 optimizer；不把同一组参数分别作为 transformer 与 network 重复包装。不绕过包装对象直接调用裸模型作为训练入口。内部 core 是普通子模块，序列循环留在一次逻辑前向内。

梯度累积只在边界执行有效 update；clip 内部的帧不是独立 optimizer step。更新失败/AMP overflow 时，global update counter 和 LR 不应假装已经前进。

### 13.4 DDP 的独立交付条件

首版单 GPU 先验收；没有通过双 GPU 测试前，多 GPU 启动应给出明确未支持错误，而不是根据宿主能力自动宣称兼容。

DDP 支持阶段必须处理不同 rank 的有效像素数。如果 DDP 平均梯度，对每一损失项可令本地可微 numerator 乘 world size，再除以 all-reduce 得到的全局 denominator，从而得到全局有效元素平均；不能直接平均不等面积的本地均值。

本地无有效监督但其他 rank 有效时，仍需执行兼容同步的零贡献图。全局无效或非有限时，各 rank 统一决定跳过/终止。禁止单 rank 独自跳过 backward 导致通信不匹配。

单帧冻结 temporal head 后不应留下本不需要训练的 unused parameters。分布式是否使用 `find_unused_parameters` 必须由真实控制流测试确定，不通过虚假的 loss 依赖掩盖错误。

### 13.5 Sampling 与 evaluator

NR sample config 指向 source/target/flow/controls manifest，不读取 prompts。采样不加载 VAE/文本模型，不 round 成其他视频 VAE 的 `4n+1` 帧格，也不套 CFG 或多步 scheduler。

采样前后恢复模型 training 状态、RNG 和 train runner 状态；用独立 history。原生/浮点输出由同一 evaluator 收集，明确各自 dtype/profile，不能混称同一后端结果。

## 14. 精度、显存与性能策略

### 14.1 理论参数状态规模

按 U1 的 145,755,114 个逻辑参数计算，额外一个 blend 标量对下表舍入结果无影响：

| 项目 | 理论大小 |
|---|---:|
| 所有参数按 BF16 计 | 0.271 GiB |
| 所有参数按 FP32 计 | 0.543 GiB |
| FP32 参数 + 梯度 + AdamW 两份 FP32 moment | 2.172 GiB |

最后一项是 `P × (4+4+4+4)` 字节，不含激活、workspace、EMA、计算副本、缓存、通信和框架开销，不是总显存承诺。也不表示把所有原始 f16/f32 数值转 BF16 可以保持无损。

未 padding 的 3840×2160 下，一个 BF16 的 32 通道张量约 0.494 GiB，一个 128 通道张量约 1.978 GiB。由此可见高分辨率 activation 值得优先测量，不能只按模型文件大小选显存策略。

### 14.2 优化顺序

先控制 valid crop、batch 与可反传帧数；再使用梯度累积、非重入/经验证的 checkpointing 和窗口计算分块。随后根据实测决定 LoRA、混合精度与 offload。

ViT stage 约占原始载荷的 68.18%，但这不是算力占比。参数量、执行次数与 activation 尺寸分开记录。71 个 block 也不意味着通用 block swap 可直接应用于含 U 形 skip 的执行图。

### 14.3 混合精度与后端限制

FP32 master 参数不通过全模型 `.to(bfloat16)` 丢失。autocast 只影响声明允许的计算区；loss、关键归约与 profile 要求 FP32 的温度计算按契约处理。BF16/FP16 需通过相同的 preclamp/head/时序误差与梯度测试。

v1 不默认支持 FP8 base、block swap、SageAttention、FlashAttention、xFormers 或任意 SDPA 替换。未来优化必须证明所需的 cosine scaling、prior 与自定义 softmax 语义可被保留或偏差已被明确接受。

checkpointing 不在被重算函数内更新外部 history 或 RNG；同一输入/噪声下比较开启/关闭的 loss 与 gradients。长序列不能仅靠减少权重驻留来假装解决所有激活显存。

## 15. 保存、恢复与导出

### 15.1 文件类型分开

| 产物 | 内容 | 能证明什么 |
|---|---|---|
| canonical model | FP32 参数、profile/config、来源 | 本适配器可加载的完整模型 |
| LoRA adapter | A/B、alpha、targets、base identity | 对指定 canonical base 的增量 |
| merged model | base 与 adapter 合并后的 canonical | 同后端浮点推理兼容，需合并测试 |
| training state | optimizer、scheduler、scaler、计步、采样器/RNG | 用于恢复训练，不是部署权重 |
| native repack | 原生 stage/manifest 与量化报告 | 仅在 runtime 回放通过后证明指定部署兼容 |

原始格式 round-trip 与训练后的 native export 是不同测试：前者要求不修改参数时字节完全相同；后者包含量化误差，不应要求等于原始权重。

### 15.2 必备 metadata

每个产物 MUST 记录文档/模型 schema、310.8.0 profile、源文件/stage/manifest hash、canonical layout version、trainer commit、reference identity、preprocess/controls/geometry/numerics 内容哈希、训练模式、可训练映射、optimizer groups、数据编码和 manifest 指纹。

还必须有 seed policy、sequence length、burn-in、TBPTT、`temporal_trained`、`temporal_validated`、部署意图、已完成验收标签和验证报告指针。adapter 额外保存 rank/alpha/targets/base hash；source fingerprint 的统计与训练后统计分开。

用户未提供的原始 hash 由实际导入器计算，不能在示例中写一个假的 SHA。原始数据/权重、社区实现与最终产物的来源材料分别保留；不因某个网页的标签就将所有资产视为相同授权。

### 15.3 Resume 精确性

只在完整 optimizer update 边界写精确恢复点，保存完整模型/adapter、optimizer、scheduler、AMP scaler（适用时）、global updates、epoch、已消费 batch 位置、sampler 状态、各 rank RNG、seed policy 与配置哈希。

状态必须描述**已消费样本**，不是 dataloader 已预取位置。可通过固定 epoch batch plan 与消费游标重建后续 batch；训练前回归测试验证多 worker/prefetch 不改变恢复后的样本顺序。

本设计不跨 batch 保存活动 history，恢复后由下一 clip 的 burn-in 重建。不要序列化 autograd 图。仅恢复权重而不恢复 optimizer/sampler 的行为命名为 `warm_start`，不能称为 exact resume。

配置或 source identity 冲突默认拒绝 resume；允许有意改变配置时必须显式另起运行并记录差异。

### 15.4 Native export 的附加要求

重新量化前先报告权重分布、非有限值、量化变化比例、误差与超出 runtime 支持范围的元素。U1 的初始范围只是 fingerprint；不能将 0.875 或 prior≤0 等测量值自动当成训练约束。

runtime 若存在明确支持范围，例如实现限定的矩阵权重范围，导出器必须使用有来源的 backend contract 验证。失败时拒绝导出或进入明确配置的再训练/量化流程，不做无日志 clipping。

重打包后的模型必须由目标 OpenDLSS/native backend 实际加载，在固定 clip 上评估 RGB、logit、blend、连续历史与重置。普通 canonical 保存成功不意味着官方 DLL 或任何游戏引擎能够使用。

## 16. 评估与验收标准

本节是实现后需要执行的测试，**不是本次整理已完成的模型测试**。

### 16.1 评估套件

固定源/目标、controls、seed 与 reference，保存 source、原始权重输出、微调输出和 target 四方对照。覆盖静止、平移、相机运动、遮挡/显露、切镜头、高频纹理、非整齐分辨率和饱和区域。

至少记录 RGB MAE/PSNR、target-relative temporal error、preclamp/head 误差、logit/blend 分布、饱和比例、非有限值、峰值显存、时间与完整运行配置。速度需注明 GPU、分辨率、精度、warm-up、是否计入 I/O，不得只给单个 FPS 数字。

### 16.2 测试矩阵

| ID | 用例 | 验收要求 |
|---|---|---|
| A01 | stage / record inventory | 11-stage 合计与各长度正确；152 layer + 1 blend，范围无重叠/空洞 |
| A02 | 字节分类 | 参数、padding、head 空槽、opaque 和 blend 账完全闭合 |
| A03 | fragment decode | E4M3 与 f16 的合成 one-hot/独立参考测试通过 |
| A04 | 排列 | 32 项通道置换、head-major QKV、prior 双轴 token 布局独立验证 |
| A05 | 未训练 round-trip | 11-stage 重建逐字节一致，含负零和 opaque |
| A06 | 错 schema / 长度 / source hash | 明确失败，无 `strict=False`、随机补层或静默忽略 |
| B01 | 模块拓扑 | block 39 无 attention；FFN 变体与 transition 位置正确 |
| B02 | 初始前向 | 各 stage/关键 block、raw head、preclamp 和输出均有参考对照 |
| B03 | 数值 profile | native 与 surrogate 分开报告；不把普通 softmax 称为原生等价 |
| B04 | 几何 | 第 7 节全部 fixture 与小尺寸拒绝、窗口相位正确 |
| B05 | 特征 | 16 lanes、常数、条件、中心化、seed 和 padding 与 reference 匹配 |
| B06 | 无效历史 | source fallback、生效的 w=0、无黑色污染 |
| C01 | flow | 整数/亚像素平移、符号反转、UV/NDC 导入，方向与单位正确 |
| C02 | 变换 | crop/resize/flip 后 source/target/flow/mask/control 一致 |
| C03 | 缓存 | 更改 source/controls/profile/crop 等后旧缓存失效 |
| C04 | clip | 不乱序、不跨 batch 泄漏；reset、burn-in 与 TBPTT 边界正确 |
| C05 | 数据划分 | train/validation scene/sequence 无交集 |
| D01 | 全量训练 | optimizer 集合正确，有效分支梯度有限并更新，冻结区与状态不变 |
| D02 | 少样本优化 | 固定 8–16 对样本/固定噪声，1000 次 updates 内主监督损失低于初值的 50%；初值近零时改用参数/梯度验证并单独说明 |
| D03 | 单帧模式 | logit 与 blend 不进入 optimizer，metadata 不声称时序已训练 |
| D04 | 时序模式 | 后续帧 loss 可影响段内较早帧，burn-in 前无梯度；真实 history 下 temporal head 得到有效梯度 |
| D05 | LoRA | target 数量/参数量独立核对，base 不变，adapter 正确更新 |
| D06 | 合并/重载 | 同浮点 backend、同输入下结果在声明容差内一致 |
| D07 | checkpointing/AMP | loss/gradient 回归通过；不改变 seed/history；overflow 计步正确 |
| E01 | Resume | 连续 N 步与 K 步保存恢复至 N 步，同数据/RNG与声明容差；包含多 worker |
| E02 | 长序列 | 至少 64 帧闭环，覆盖静止/运动/遮挡/reset；无 NaN、无无界内存增长 |
| E03 | 原始模型对照 | 相同输入/controls/seed；完整用例而非挑选截图 |
| E04 | Native 导出 | 若声明支持：量化报告、目标 loader、帧和序列回放全部通过 |
| F01 | 无虚假依赖 | 不构造/下载 VAE、文本编码器或 diffusion scheduler |
| F02 | 旧架构回归 | 新 NR 模块不改变旧脚本入口/默认参数/RNG；无侵入式 monkey patch |
| F03 | 双 GPU | 若声明支持：无效 mask、累积、同步、clip/reset、保存恢复与单 GPU 对照通过 |

### 16.3 质量结论与证据

A 类测试仅证明布局，D 类证明优化机制，E 类才涉及闭环行为。loss 降低不等于感知质量改善，PSNR 提升不等于时序稳定。

正式质量验收同时提交未调优 baseline 与候选模型的逐用例指标、视频和失败样例。项目可以调整训练损失或质量阈值，但必须留版本记录，不能通过丢弃失败样本获得“通过”。

## 17. 文件与模块变更

所有路径都是拟新增，根目录脚本只做 thin entry point：

```text
# 根目录：CLI wrappers
 dlssnr_inspect_model.py
 dlssnr_convert_model.py
 dlssnr_validate_model.py
 dlssnr_cache_pixels.py
 dlssnr_train.py
 dlssnr_train_network.py
 dlssnr_generate_image.py
 dlssnr_generate_video.py
 dlssnr_merge_lora.py
 dlssnr_export_native.py

 src/musubi_tuner/
   dlssnr_inspect_model.py
   dlssnr_convert_model.py
   dlssnr_validate_model.py
   dlssnr_cache_pixels.py
   dlssnr_train.py
   dlssnr_train_network.py
   dlssnr_generate_image.py
   dlssnr_generate_video.py
   dlssnr_merge_lora.py
   dlssnr_export_native.py
   dlssnr/
     config.py                  # NR 配置 schema 与严格校验
     checkpoint.py              # manifest/records 读取与 canonical 映射
     packing.py                 # fragment、置换、inverse 与 native repack
     profiles.py                # 310.8.0 block/record descriptor
     geometry.py                # valid/field/levels/window phases
     numerics.py                # profile 选择与可微算子边界
     model.py                   # canonical 网络、FFN 与 transition
     conditioning.py            # lanes 10–14 校验与明确编码
     noise.py                   # 可复现 per-frame seed 与 Gaussian lanes
     preprocess.py              # 输入、padding、中心化
     temporal.py                # warp/history/reset/publication
     pipeline.py                # 共享 forward_frame
     dataset.py                 # 单 clip item、manifest、bucket/collator
     cache.py                   # 像素缓存与失效规则
     losses.py                  # 有效元素归一化的监督目标
     training_step.py            # burn-in + 单 TBPTT 段
     evaluation.py              # 冻结基线与连续闭环评估
   training/
     dlssnr_services.py          # Musubi 公共工具的窄适配
     dlssnr_trainer.py           # 专用 runner，含唯一 NRTrainModule 包装
   networks/
     lora_dlssnr.py

 tests/dlssnr/
   test_inventory.py
   test_packing.py
   test_model_shapes.py
   test_numerics.py
   test_geometry.py
   test_preprocess.py
   test_dataset.py
   test_temporal.py
   test_full_training.py
   test_lora.py
   test_resume.py
   test_evaluation.py
   test_native_export.py
   test_host_regression.py

 configs/dlssnr_full_single.toml
 configs/dlssnr_full_temporal.toml
 configs/dlssnr_lora_vit.toml
 docs/dlssnr.md
 docs/specs/dlssnr_310_8_0_training_spec.md
```

reference kernel 较多时可从 `numerics.py` 拆出子包，不能把 native 索引、trainer 和 dataset 混成一个巨型文件。核心 `dlssnr/*` 不依赖宿主的 diffusion 类。

如需要公共架构标识，在 `dataset/architectures.py` 添加短名 `dlssnr` 与全名 `dlss_neural_rendering`，维护兼容导出；这不意味着 NR dataset 必须进入旧的 latent cache 构建器。

不为此修改现有 Wan/FLUX 训练数学。不必将旧的 `hv_train_network.py` 当作新基类；生命周期可参考独立全量训练实现，但不复制其模型特定 diffusion 行为。[S3]

## 18. 配置与目标命令

**本节全部为实施后的目标接口，不是当前仓库已经支持的命令。** 配置只读 NR schema；未知字段报错，命令行覆盖顺序固定为“schema 默认值 → TOML → 显式 CLI”，最终展开配置写入运行目录。

### 18.1 全量单帧配置

目标文件 `configs/dlssnr_full_single.toml`：

```toml
schema_version = 1

[model]
model_dir = "../models/canonical_dlssnr"
profile = "dlss_nr_310_8_0"
numerics_profile = "train_surrogate"
deployment_target = "float_runtime"

[data]
train_manifest = "../data/train_single.jsonl"
validation_manifest = "../data/validation_single.jsonl"
cache_directory = "../cache/dlssnr_single"
source_encoding = "srgb_proxy"
target_encoding = "srgb_proxy"
controls_encoding = "dlssnr_lanes_10_14_v1"
bucket_size = [512, 512] # width, height
require_cache = true

[training]
mode = "single_frame"
seed = 42
batch_size = 1
sequence_length = 1
burn_in = 0
tbptt_length = 1
gradient_accumulation_steps = 4
max_train_steps = 1000
gradient_checkpointing = false

[optimizer]
type = "AdamW"
learning_rate = 1e-5
weight_decay = 0.0
lr_scheduler = "constant"

[parameter_groups]
prior_lr_multiplier = 0.1
scale_lr_multiplier = 0.1
temporal_blend_lr_multiplier = 0.1

[precision]
mixed_precision = "no"
master_dtype = "float32"

[loss]
pre = 1.0
out = 1.0
edge = 0.05
temporal = 0.0

[evaluation]
sample_every_n_steps = 100
sequence_manifest = "../data/validation_sequences.jsonl"
min_sequence_frames = 64
compare_baseline = true

[output]
output_dir = "../output/dlssnr_full_single"
output_name = "dlssnr_310_8_0_full"
save_every_n_steps = 100
save_state = true
```

这里的 1000 步用于 smoke test，不是质量保证。sequence evaluator 使用完整有序验证片段，不受训练 `sequence_length=1` 截断；其任务是发现单帧微调造成的闭环退化。

### 18.2 全量时序配置

以完整单帧配置为基础生成独立 `configs/dlssnr_full_temporal.toml`，以下为**替换项说明，不是可直接复制追加的重复 TOML 表**：

```toml
[data]
train_manifest = "../data/train_temporal.jsonl"
validation_manifest = "../data/validation_temporal.jsonl"
cache_directory = "../cache/dlssnr_temporal"

[training]
mode = "temporal"
sequence_length = 4
burn_in = 2
tbptt_length = 2

[loss]
temporal = 0.10

[output]
output_dir = "../output/dlssnr_full_temporal"
output_name = "dlssnr_310_8_0_temporal"
```

实现交付时必须提供展开后的完整 TOML，保留未列出的必需字段；不依赖一个未定义的配置继承机制。数据中必须有 controls、非 reset 帧的 motion/history mask，以及启用 temporal loss 所需的 temporal mask。

### 18.3 LoRA 配置

`configs/dlssnr_lora_vit.toml` 从完整单帧配置生成，删除 `[parameter_groups]`，设置 optimizer LR 为 `1e-4`、`output.output_dir = "../output/dlssnr_lora"`、`output.output_name = "dlssnr_310_8_0_lora_vit"`，并增加以下表：

```toml
[lora]
profile = "vit_only"
rank = 16
alpha = 16
dropout = 0.0
qkv_mode = "fused_head_major"
```

全量入口看到 `[lora]` 必须报错；LoRA 入口缺失该表也报错。`multiscale` 模式用明确的 per-width rank/alpha map 替代单一 rank，禁止两个互相矛盾的规则同时生效。LoRA 配置不保留全量参数组的优化行为；base 参数完全冻结。LoRA 入口出现全量专用 `[parameter_groups]` 时必须报错，不能接受后忽略。

### 18.4 转换、校验与缓存

```bash
python dlssnr_inspect_model.py \
  --source_dir models/dlss_nr_310_8_0 \
  --profile dlss_nr_310_8_0 \
  --report_dir audit/dlssnr_310_8_0

python dlssnr_convert_model.py \
  --source_dir models/dlss_nr_310_8_0 \
  --profile dlss_nr_310_8_0 \
  --output_dir models/canonical_dlssnr \
  --verify_roundtrip

python dlssnr_validate_model.py \
  --model_dir models/canonical_dlssnr \
  --reference_manifest fixtures/dlssnr_reference.json \
  --numerics_profile train_surrogate \
  --report_dir audit/forward_validation

python dlssnr_cache_pixels.py \
  --config_file configs/dlssnr_full_single.toml \
  --validate
```

`--verify_roundtrip` 检查未训练权重的字节重建，不代表 forward parity；`--validate` 缓存检查也不等于模型验证。reference manifest 必须指向实际采集/固定的参考，不接受没有文件的声明。

训练入口读取与当前 source/profile 匹配的 conversion 与 baseline forward report；没有报告时只开放明确命名的开发 smoke 模式，其产物标记为 experimental，不宣称 compatibility。

### 18.5 训练

```bash
accelerate launch --num_processes 1 --mixed_precision no dlssnr_train.py \
  --config_file configs/dlssnr_full_single.toml

accelerate launch --num_processes 1 --mixed_precision no dlssnr_train.py \
  --config_file configs/dlssnr_full_temporal.toml

accelerate launch --num_processes 1 --mixed_precision no dlssnr_train_network.py \
  --config_file configs/dlssnr_lora_vit.toml
```

Accelerate launcher 与 TOML 精度发生冲突时必须报错，不能悄悄选择其中一个。训练使用配对数据，不传 prompts、VAE、CFG 或 diffusion timestep 参数。

### 18.6 推理、合并与恢复

```bash
python dlssnr_generate_image.py \
  --model_dir output/dlssnr_full_single/final \
  --sample_manifest data/inference_single.jsonl \
  --numerics_profile train_surrogate \
  --output_dir output/preview

python dlssnr_generate_video.py \
  --model_dir output/dlssnr_full_temporal/final \
  --sequence_manifest data/inference_sequence.jsonl \
  --numerics_profile train_surrogate \
  --output_dir output/video

python dlssnr_merge_lora.py \
  --base_model_dir models/canonical_dlssnr \
  --adapter output/dlssnr_lora/final/adapter.safetensors \
  --output_dir output/dlssnr_merged

accelerate launch --num_processes 1 --mixed_precision no dlssnr_train.py \
  --config_file configs/dlssnr_full_temporal.toml \
  --resume output/dlssnr_full_temporal/state-step000100
```

推理 manifest 允许 target 缺省；训练和配对质量评估不允许。推理仍必须具有 controls，视频非 reset 帧仍需 motion/有效性条件。

训练产物 `final/` 保存完整 canonical 目录；LoRA `final/` 保存 adapter 及其配置/来源。示例 adapter 路径由 LoRA 配置的实际 `output_dir` 决定，实现配置文件必须与文档命令一致。

## 19. 失败处理与拒绝项

| 情况 | 必需行为 |
|---|---|
| stage/record 长度、dtype 或 source hash 不匹配 | 停止；输出具体 record 与预期/实测，不猜 reshape |
| 源 padding 非零、head 空槽非零 | 原始 profile 校验失败；不自动当新增参数 |
| controls 缺失或 encoding 未知 | 停止；不填“默认风格” |
| 时序非 reset 帧无 flow/history mask | 停止；不把零 flow 作为替代 |
| temporal loss 启用但缺可靠 mask | 停止；要求数据补全或显式禁用该 loss |
| 标准 source/target 不配准或编码不明 | 拒绝监督训练，报告数据错误 |
| surrogate 未通过兼容门槛 | 仅实验标签；不声称 native-compatible |
| source 指纹超出 U1 统计 | 在原始模型校验中失败/待确认来源；微调模型使用自身身份，不强行套原始统计 |
| 非有限 loss/gradient/update | 按配置统一终止或明确跳过；记录样本与数值 profile，不静默替零 |
| 不支持的 attention/quant/offload 配置 | 直接报错，不能接受参数后忽略 |
| 全量入口带 LoRA-only 参数 | 配置错误 |
| native export 未通过 loader/回放 | 不设置 `native_export_validated` |

拒绝暴露或显式拒绝 VAE/text encoder/caption/CFG/guidance/timestep sampling/SD3 weighting/flow shift/通用推理步数等无 NR 定义选项。

初始 prior≤0、温度为正、scale 范围及最大权重 0.875 是来源 fingerprint，不足以直接推导训练时的 hard constraint。后续约束若改变参数化或投影规则，必须单独版本化并做训练/部署实验。

## 20. 分阶段交付

| 阶段 | 交付 | 完成门槛 |
|---|---|---|
| P0 固定格式导入 | 310.8.0 descriptor、Inspector、canonical converter、opaque 保存 | 字节账闭合、独立布局测试与未训练 round-trip |
| P1 可验证前向 | model、preprocess、geometry、numerics、reference fixtures | block/head/output 检查与标签明确；不再猜结构 |
| P2 全量单帧 | 配对数据/缓存、专用 runner、全量参数组、保存恢复 | 有效参数更新、少样本优化、单帧与闭环基线对照 |
| P3 时序主线 | warp、学生 history、burn-in、TBPTT、连续 evaluator | temporal 梯度、reset 和至少 64 帧回放通过 |
| P4 LoRA | vit_only；通过后 multiscale、保存与合并 | targets/梯度/冻结/merged parity，训练 step 与 P2/P3 共用 |
| P5 宿主与性能完善 | 文档、旧架构回归、按实测开放 AMP/checkpointing/DDP | 每个支持项各自通过测试，未支持项明确拒绝 |
| P6 可选原生部署 | 量化评估、必要的 QAT/STE、trained native repack | 指定 runtime loader 与序列回放通过，不等于官方 DLL 兼容 |

P0 现在的任务是将 U1 已知结构落实为代码和测试，不再把“可能没有完整网络”作为默认阻塞项。P1 若发现数值偏差，应定位算子/边界，不能用更换训练器代替解决。

P2 可先交付单帧实验工具；只有 P3 和相应质量验证完成后才能声明时序适配。P4 不阻塞全量基线。P6 不属于普通 safetensors 保存的隐含能力。

## 21. 相对 v0.1 的修订

| v0.1 假设/方案 | v0.2 取代内容 |
|---|---|
| 71 block 和 16→4 为待核实候选 | 固定 310.8.0 profile，U1 本地测量作为结构输入 |
| 仅知道约 148 MB，参数数未知 | 固定 stage/record inventory，明确 145,755,114 与独立 blend 标量口径 |
| 统一描述 71 个 mixer | 70 个 attention mixer + block 39 纯 transition |
| 广义权重探索 Gate A | 固定 schema 的严格转换、来源验证与 round-trip |
| 继承默认 trainer 并新增两个 diffusion hook | 专用 supervised runner + Musubi 公共服务薄适配，默认不修改旧循环 |
| 全量与 LoRA 同时作为首个训练落点 | 先全量基线，再时序，再分级 LoRA |
| 全网 rank 16 | vit_only rank 16；可选 multiscale 按宽度配置 |
| `reference` / `train_float` 泛称 | `native_reference` / `train_surrogate` 与独立兼容标签 |
| 不明确的 `checkpoint_default` controls | 必须提供已编码 lanes 或经过验证的具体 encoder |
| 默认 FP16 像素缓存 | 默认保留来源精度；FP16 缓存显式标为另一有损模式 |
| 一般性长片段 TBPTT 描述 | v1 每 microbatch 一个可反传段，片段由数据层切分 |
| 保存/导出边界不够细 | canonical、训练状态、未训练 round-trip、训练后 native export 四者分开验收 |

旧版中的公开 HF 文件 hash、未经本次本地复验的文件身份，不被自动用作 U1 的 stage hash。实际哈希由 converter 记录。未读 PDF 的内容不成为任何结构或训练结论的前提。

## 22. 附录 A：block 1 字节布局

U1 测得 block 1 为 20,672 字节，可作为最小 record fixture：

| 偏移 | 字节 | 内容 |
|---:|---:|---|
| 0 | 4,096 | W1 32×128 E4M3 |
| 4,096 | 4,096 | W2 128×32 E4M3 |
| 8,192 | 16 | 零 padding |
| 8,208 | 64 | ffn skip，f16[32] |
| 8,272 | 16 | 零 padding |
| 8,288 | 3,072 | QKV 32×96 E4M3 |
| 11,360 | 8,192 | prior 1×64×64 f16 |
| 19,552 | 4 + 12 | 1 个 f32 温度与 12 字节零 padding |
| 19,568 | 1,024 | projection 32×32 E4M3 |
| 20,592 | 64 | attention skip，f16[32] |
| 20,656 | 16 | 尾部零 padding |

其他变体不能只依据此表按比例外推。至少各取一个 64/128/256、512、ViT、decoder 首块、block 0/39/70 的独立 fixture。

结构计数还可独立核对：

```text
window blocks by C: 32→10, 64→8, 128→12, 256→16, 512→16
window head count: 10×1 + 8×2 + 12×4 + 16×8 + 16×16 = 458
global head count: 8×32 = 256
all temperatures: 458 + 256 = 714

layer records:
46 个 C≤256 window records
+ 16×4 个 C512 records
+ 8×5 个 ViT records
+ 1 个 block30 transition record
+ 1 个 block39 transition record
= 152
再加独立 blend scale = 153
```

## 23. 附录 B：原始数值指纹

以下全部为 **U1 的原始权重测量**，用于来源验证和解包回归；不是本次重新执行统计的结果，也不是必须施加于训练后的范围约束。

| 项目 | U1 测量 |
|---|---|
| E4M3 元素 | 143,831,040；无 NaN 码 |
| E4M3 最大绝对值 | 0.875；无元素超过所述推理端的 9 |
| +0 / -0 | 3,617,037 / 3,617,790，合计约 5.03% |
| 小权重 | 约一半绝对值小于 0.015625 |
| `abs(w)≥0.125` | 1,852,806，约 1.29% |
| `abs(w)≥0.5` | 1,968 |
| 1024→512 transition | 最大绝对值 0.140625 |
| prior | 1,875,968 个，全部≤0，范围 [-126.5,0]，恰好为零 200 个 |
| 温度 | 714 个 f32，均正，范围约 [0.01453,28.439] |
| FFN skip | 无零值，范围约 [-0.703,1] |
| attention skip | 无零值，范围约 [-0.979,1] |
| 输入 adapter | lane 15 对应 32 个权重全零，其余约 [-0.950,0.869] |
| head | 前三列 RGB residual 绝对值约 0.05 以内，第 4 列最大到 0.486 |
| 独立 blend scale | f16，约 0.739746 |
| padding | 2,376 字节全零 |
| head 空槽 | 384 个 half，全部零 |

8 个 ViT 未读 `layer3` 按 f16 解读约为：

```text
0.109, -0.248, 5.0e-5, -0.0104, 0.186, 0.0411, 0.0825, -558.5
```

这些是便于核对的显示值，实际保存使用原始 2 字节；不把最后一个大负值当损坏并“修复”为零。近似显示值不能用来替代精确字节 fixture。

## 24. 来源与复核入口

<a id="source-u1"></a>

**U1 — 本对话中用户提供的本地审计。** 包含 DLSS-NR 310.8.0 的 11-stage 长度、153 records、逻辑参数数、block/FFN 拓扑、fragment 索引、QKV 排列、prior/scale/温度与实测统计。用户说明 PDF 返回 403，未读正文。本文件按该测量固化 profile；本次没有重新获取或运行用户本地 bin。

**S1 — Musubi 公共 trainer。** [固定提交源码][S1]。重点：`process_batch/compute_loss`、公共服务、模型冻结、network 参数组、flow scheduler 与生命周期 hook。前次读取 blob SHA：`4822c5784a9b180d63128a78ac7ae44b8c63e76c`。

**S2 — Musubi image/video dataset。** [固定提交源码][S2]。重点：frame count、frame position、视频 datasource 与来源追踪；不把这些等同于 NR 状态训练。

**S3 — Musubi 全量训练生命周期参考。** [HiDream-O1 全量入口][S3]。只借鉴独立 optimizer/保存组织，不继承其 diffusion 算法。

**S4 — sd-scripts 公共 trainer。** [固定提交源码][S4]。重点：`process_batch`、`get_noise_scheduler`、启动策略与 `_run_validation_loop`，以及 arbitrary dataset 路径。前次读取 blob SHA：`f228cc301482c9ecebd35a941f5b0ece9d11cd42`。

**S5 — sd-scripts 模块结构说明。** [固定提交开发说明][S5]。用于确认 optimizer/dataset/checkpoint 等模块已拆分；实际接口仍以代码为准。

**S6 — OpenDLSS-NR 图定义。** [network.md][S6]。前次读取 blob SHA：`22db81d3383d55f9a6335d4abaebea6e7799e92e`。

**S7 — OpenDLSS-NR 权重布局。** [weights.md][S7]。前次读取 blob SHA：`d0c4ddf6aaa1aa07f80bc201b49f07ba5d1e4d3f`。

**S8 — OpenDLSS-NR 数值契约。** [numerics.md][S8]。前次读取 blob SHA：`b960ae81c9826822dd945f7a3c7e75458490e731`。

**S9 — OpenDLSS-NR 帧管线与历史。** [frame.md][S9]。前次读取 blob SHA：`90d54ab4a00d434f49a20cd6c75819a7387bdcef`。

**原始模型页面：** [sekkit/open-dlss5-nr][HF]。模型分发页不替代 U1 的本地容器校验，也不自动证明不同文件的逐字节身份。

**原始报告入口（正文未读取）：** [NVIDIA DLSS5 Report][PDF]。本文不引用其页码或未读内容。

[U1]: #source-u1
[S1]: https://github.com/sdbds/musubi-tuner/blob/4e7c7149249e7715e9168920feb4c420423abba7/src/musubi_tuner/training/trainer_base.py
[S2]: https://github.com/sdbds/musubi-tuner/blob/4e7c7149249e7715e9168920feb4c420423abba7/src/musubi_tuner/dataset/image_video_dataset.py
[S3]: https://github.com/sdbds/musubi-tuner/blob/4e7c7149249e7715e9168920feb4c420423abba7/src/musubi_tuner/hidream_o1_train.py
[S4]: https://github.com/sdbds/sd-scripts/blob/4e624302e0088e39933b31cbc71f24212e900f5f/train_network.py
[S5]: https://github.com/sdbds/sd-scripts/blob/4e624302e0088e39933b31cbc71f24212e900f5f/.ai/context/01-overview.md
[S6]: https://github.com/maanHimself/OpenDLSS-NR/blob/main/docs/network.md
[S7]: https://github.com/maanHimself/OpenDLSS-NR/blob/main/docs/weights.md
[S8]: https://github.com/maanHimself/OpenDLSS-NR/blob/main/docs/numerics.md
[S9]: https://github.com/maanHimself/OpenDLSS-NR/blob/main/docs/frame.md
[HF]: https://huggingface.co/sekkit/open-dlss5-nr
[PDF]: https://research.nvidia.com/labs/adlr/files/DLSS5_Report.pdf

---

**交付边界：本次仅整理修订后的规格文档，没有修改远端训练器，没有重新测量本地权重，没有执行模型训练/前向验收，也没有证明训练产物的 NVIDIA 官方运行时兼容性。**
