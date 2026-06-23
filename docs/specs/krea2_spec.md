# Krea 2 接入 musubi-tuner — 技术规格 (Spec)

> 状态：草案 (Draft) — **官方源码已核对（inference/mmdit/encoder/sampling.py），§9 全部关闭，可动手**
> 作者视角：Linus
> 参考模型：`https://huggingface.co/buckets/krea-community/krea-2`（官方 OSS：`oss_raw` 非蒸馏 base + `oss_turbo` 蒸馏，含 `inference/mmdit/encoder/sampling.py` 源码）
> 参考社区权重：`AlperKTS/Krea2_FP8`、ComfyUI 0.25.0+ 原生 Krea2 支持

---

## 0. 需求确认

基于现有信息，我理解你的需求是：

> 在 musubi-tuner 里新增对 **Krea 2** 模型的支持，行为与现有模型（Qwen-Image / FLUX / Z-Image 等）一致：
> 1. **单文件读取** DiT（一个 `.safetensors`），不依赖 diffusers 目录结构；
> 2. 支持 **缓存**（latent 缓存 + text encoder 输出缓存）；
> 3. 支持 **训练**（LoRA，走 `NetworkTrainer` 那一套）；
> 4. 支持 **推理**（`*_generate_image.py`）。
>
> 现阶段只产出 spec，不写实现代码。

如果理解有偏差，先停下来纠正我。

---

## 1. 关键事实：Krea 2 OSS 不是新东西，是三个旧零件的拼装

很多人一听"全新从零训练的基础模型"就开始臆想要写一套全新架构。**这是臆想出来的问题。** 看实际权重清单（来自 `AlperKTS/Krea2_FP8` 的 ComfyUI 运行依赖）：

| 组件 | Krea 2 OSS 使用的权重 | 本仓库现状 |
| --- | --- | --- |
| DiT (扩散主干) | `krea2_turbo_fp8.safetensors`（BF16 约 24.76 GiB），放 `unet/` | **需要从头写新模型类**：Krea 自研架构，非 FLUX 非 Qwen；可复用本仓库的注意力后端 / block-swap / fp8 基础设施 |
| 文本编码器 | `qwen3vl_4b_fp8_scaled.safetensors`（**Qwen3-VL 4B**），放 `text_encoders/` | **已存在**：`hidream_o1/qwen3_vl_transformers.py` + `hidream_o1_utils.load_model` |
| VAE | `qwen_image_vae.safetensors`（**Qwen-Image VAE**），放 `vae/` | **已存在**：`qwen_image/qwen_image_autoencoder_kl.py` + `qwen_image_utils.load_vae` |
| 采样 | flow-matching Euler + 分辨率相关 time-shift（官方 sampling.py，**非 er_sde**）；raw: 28步/CFG4.5；turbo: ~8步/关CFG/固定mu | flow matching 调度器已有，shift 公式需新增 |

**更正（已核实权重）**：早期根据 FP8 作者的口头描述以为是 FLUX 血统，**这是错的**。实际 dump `krea2_turbo_fp8.safetensors` 的 432 个 tensor 后确认：这是 **Krea 自研的单流联合 MMDiT**（28 层、dim 6144、GQA 48/12 头、SwiGLU、QK-RMSNorm、adaLN-single），外加一个专属 `txtfusion` 文本前端。完整结构见 §11 附录。

**结论：3 个零件里，VAE 和文本编码器本仓库已具备；DiT 是一套全新架构，必须老老实实从头写一个模型文件（基础设施可复用，但模块定义与 forward 全新）。外加 4 个脚本壳子。** 这决定了工作量级别和风险点。

---

## 2. Linus 五层分析

### 第一层：数据结构

核心数据流（已按真实权重校正）：

```
图像 (RGB, [-1,1])
  └─ Qwen-Image VAE.encode ─> latent (C=16, F=1, H/8, W/8)
        └─ pack 2x2 patch ─> (B, N, 64)          # first: Linear(64 -> 6144)
              └─ DiT 28 层单流联合块 (dim=6144, GQA 48/12, head_dim=128, SwiGLU 16384, adaLN-single)
                    └─ last: Linear(6144 -> 64) ─> unpack ─> latent ─> VAE.decode ─> 图像

文本 (str)
  └─ Qwen3-VL 4B ─> 多层 hidden states 堆叠 (B, 12层, T, 2560)   # ← 不是单层！
        └─ DiT.txtfusion: projector[1,12] 聚合 12 层 + 2 layerwise + 2 refiner 块 (dim=2560)
              └─ txtmlp: Linear(2560 -> 6144) ─> 文本 token，与图像 token 拼接做联合注意力
```

**关键数据关系：**
- latent 通道 16，pack 后 64，和 Qwen-Image 一模一样 → **latent 缓存格式可直接复用 `save_latent_cache_qwen_image`**。✅
- 文本侧 **不一样**：`txtfusion` 吃的是 Qwen3-VL 的**多层 hidden states 堆叠**（projector `[1,12]` → 聚合 12 层），不是单层 `(T,D)`。→ **文本缓存需要新格式**，不能直接套 `save_text_encoder_output_cache_qwen_image`。⚠️
- `txtfusion` / `txtmlp` 是 **DiT 权重的一部分**（冻结，LoRA 训练时不动）→ 缓存阶段只跑 Qwen3-VL 存多层 hidden；txtfusion 在 DiT forward 内部执行。

谁拥有/修改这些数据：缓存阶段 VAE 跑一次、Qwen3-VL 跑一次落盘；训练阶段只读缓存喂给 DiT；推理阶段实时跑全链路。所有权清晰。

### 第二层：特殊情况识别

需要消除的"特殊情况"陷阱：
- ❌ **不要**为 Krea2 复制一份 Qwen-Image VAE 代码。直接 import 复用。复制 = 两份代码两个 bug。
- ❌ **不要**为 Krea2 复制一份 Qwen3-VL 代码。`hidream_o1` 已经能从单文件 safetensors 加载 Qwen3-VL（`load_single_checkpoint_model`），直接复用。
- ✅ latent 缓存 `(C,F,H,W)` 沿用现有 `save_latent_cache_qwen_image`，没问题。
- ⚠️ **文本缓存必须新建**：Krea2 `txtfusion` 需要 Qwen3-VL 多层 hidden 堆叠 `(num_layers, T, 2560)`，单层格式装不下。这是与 Qwen-Image 的真正区别点。
- ✅ 真正该写的特殊逻辑：DiT 的 forward + 模块定义 + 432-key state_dict 加载，以及配套的多层文本缓存。

### 第三层：复杂度审查

本质一句话：**"把 Qwen-Image 的训练/缓存管线，换上一个 Krea 自研的 DiT 主干，并把文本缓存换成多层 hidden。"**

涉及概念数：DiT 主干（1 个新模块）+ 4 个脚本壳子（cache_latents / cache_text_encoder / train_network / generate_image）+ 架构常量注册。其余全部复用。

能不能再砍一半？能：脚本壳子里 90% 的代码是从 `qwen_image_*` 抄来改名。真正的智力工作只在 `krea2_model.py` 一个文件。

缩进/分层：DiT forward 必须保持 Qwen-Image 那种扁平结构（一个 block 循环），不允许超过 3 层缩进。

### 第四层：破坏性分析（Never break userspace 铁律）

- 新增架构常量、新增脚本、新增 `krea2/` 子包 —— **纯增量，对现有模型零影响**。
- **禁止**修改 `qwen_image_*` / `hidream_o1_*` 的任何公共函数签名。如果发现复用的函数需要小改（例如 Qwen3-VL embed 提取要参数化层号），**新增可选参数并给默认值**，绝不改变现有调用方的行为。
- `architectures.py` 只追加常量，不动既有常量的值（缓存文件名依赖这些短码，改了会让所有人已有缓存失效）。
- `pyproject.toml` 当前**没有** `[project.scripts]`，脚本靠根目录 wrapper 调用。沿用此约定：新增根目录 `krea2_*.py` wrapper，不引入 entry point 机制。

### 第五层：实用性验证

- 这模型真有人用吗？是，OSS 已发布、ComfyUI 0.25+ 原生支持、社区已有 FP8 量化权重。真实需求。
- 复杂度与收益匹配吗？匹配。复用了 VAE + TE 两大块，增量集中在一个 DiT 文件，性价比高。
- **现实风险已缓解**：官方同时开源了 **`oss_raw` 非蒸馏 base**（CFG=4.5/28 步）与 `oss_turbo` 蒸馏版。LoRA 训练应优先用 `oss_raw`（正常 CFG 行为），Turbo 仅作快速推理/预览。原先“只有蒸馏版”的担忧解除。

---

## 3. 核心判断

**✅ 值得做。**

- **数据结构**：latent 与 Qwen-Image 同构（缓存零成本复用）；文本缓存需改为多层 hidden 堆叠。
- **可消除的复杂度**：VAE、Qwen3-VL 加载、latent 缓存、训练循环全部复用；新代码集中在 DiT 模块 + 多层文本缓存。
- **风险点（已大幅收敛）**：DiT 结构、RoPE、select_layers、prompt 模板、采样器全部已从官方源码核实（§11/§12）；非蒸馏 `oss_raw` base 可用。剩余仅工程风险：txtfusion 多层文本缓存格式落地、GQA/QK-norm 注意力在本仓库后端的实现细节、fp8 量化对 txtfusion 小投影的数值影响。

---

## 4. 目标与非目标

### 4.1 目标 (In Scope)
- `krea2` 文生图：单文件 DiT + Qwen3-VL 4B + Qwen-Image VAE。
- latent 缓存、text encoder 输出缓存。
- LoRA 训练（`NetworkTrainer` 体系），含训练中采样预览。
- 推理脚本，支持 fp8（DiT）/ fp8_vl（文本编码器）/ block swap。

### 4.2 非目标 (Out of Scope，本期不做)
- 全量微调（fine-tune）—— 先只做 LoRA，和 Qwen-Image 起步一致。
- Edit / 多参考图 / ControlNet 等条件控制（Krea 2 商用版有 reference image，但 OSS 主干先按纯 t2i 接入）。
- 蒸馏/CFG distill 训练逻辑。
- 视频。

---

## 5. 文件清单（增量，全部新增）

> 命名严格对齐现有 `qwen_image_*` 约定，降低维护者认知负担。

### 5.1 新增子包 `src/musubi_tuner/krea2/`
| 文件 | 职责 | 复用来源 |
| --- | --- | --- |
| `__init__.py` | 包标记 | — |
| `krea2_model.py` | **Krea2 DiT 模型类（28 层单流联合 MMDiT + txtfusion）+ `load_krea2_model()` 单文件加载 + FP8 target/exclude keys** | 从头写（§6.1/§11）；offloader/fp8/注意力后端参考 `qwen_image/qwen_image_model.py`，模块定义不照搬 |
| `krea2_utils.py` | 文本编码器加载与 prompt 编码、VAE 加载、latent pack/unpack、scheduler、shift 计算、模型版本参数 | TE 调 `hidream_o1_utils.load_model`；VAE 调 `qwen_image_utils.load_vae`；pack/unpack 调 `qwen_image_utils` |

### 5.2 新增脚本 `src/musubi_tuner/`
| 文件 | 职责 | 镜像自 |
| --- | --- | --- |
| `krea2_cache_latents.py` | VAE 编码图像 → latent 缓存 | `qwen_image_cache_latents.py` |
| `krea2_cache_text_encoder_outputs.py` | Qwen3-VL 编码 prompt → embed 缓存 | `qwen_image_cache_text_encoder_outputs.py` |
| `krea2_train_network.py` | `Krea2NetworkTrainer(NetworkTrainer)` | `qwen_image_train_network.py` |
| `krea2_generate_image.py` | 推理 | `qwen_image_generate_image.py` |

### 5.3 根目录 wrapper（薄壳，4 个）
`krea2_cache_latents.py` / `krea2_cache_text_encoder_outputs.py` / `krea2_train_network.py` / `krea2_generate_image.py`
内容固定为：
```python
from musubi_tuner.krea2_xxx import main

if __name__ == "__main__":
    main()
```

### 5.4 修改（极小，纯追加）
- `src/musubi_tuner/dataset/architectures.py`：追加
  ```python
  ARCHITECTURE_KREA2 = "k2"
  ARCHITECTURE_KREA2_FULL = "krea2"
  ```
- `src/musubi_tuner/dataset/cache_io.py`：latent 侧复用 `save_latent_cache_qwen_image`；文本侧**必须新增** `save_text_encoder_output_cache_krea2`（存 `(num_layers, T, 2560)` 多层 hidden + 有效长度），不能复用 Qwen-Image 的单层版本。

> 注意：以上是本 spec 唯一会触碰已有文件的两处，且都是**追加**，不改既有行为。

---

## 6. 模块设计细节

### 6.1 DiT 主干 `krea2_model.py`（唯一的真活，结构已核实）

确认的超参（来自 432-tensor dump，见 §11）：

| 参数 | 值 | 来源 key |
| --- | --- | --- |
| `in_channels` | 64 (= 16ch × 2×2 pack) | `first.weight [6144, 64]` |
| `inner_dim` | 6144 | `first.weight [6144, 64]` |
| `num_layers` | 28 | `blocks.0..27` |
| `num_query_heads` | 48 | `attn.wq [6144,6144]` / 128 |
| `num_kv_heads` | 12 (GQA) | `attn.wk/wv [1536,6144]` / 128 |
| `head_dim` | 128 | `attn.qknorm.qnorm.scale [128]` |
| MLP hidden | 16384 (SwiGLU) | `mlp.gate/up [16384,6144]` |
| 文本输入维度 | 2560 (Qwen3-VL 4B) | `txtmlp.weight [6144,2560]` |
| guidance embedding | 无 | 无 `guidance_in`/`vector_in` key |

模块 → key 映射（写模型时按此对齐，`load_state_dict(strict=True)` 必须零缺失）：
- **图像入口** `first`: Linear(64→6144) + bias。
- **时间条件 (adaLN-single)**: `temb`(正弦, tdim=256)→`tmlp`(Seq: Linear 256→6144 / GELU(tanh) / Linear 6144→6144) 得 time_emb；`tproj`(Seq: GELU(tanh) / Linear 6144→36864) 生成全局调制 `tvec`；每块 `DoubleSharedModulation`(`mod.lin [36864]` 学习偏置) 做 `vec+lin` 后 chunk 成 6 组 (prescale/preshift/pregate/postscale/postshift/postgate)。注意激活是 **GELU(tanh)** 不是 SiLU。
- **主干块** `blocks.N`（×28，单流联合）：`prenorm.scale`/`postnorm.scale`(RMSNorm) + 注意力(`wq/wk/wv/wo` + GQA + `qknorm.qnorm/knorm` QK-RMSNorm + 输出 `gate`) + SwiGLU(`mlp.gate/up/down`)。文本 token 与图像 token 拼接后一起过这 28 块。
- **文本前端** `txtfusion`(输入 `(B,L,12,2560)`)：先 reshape `(B*L,12,2560)` 过 `layerwise_blocks`(2 块, 在**层维度 12** 上做 attention, 无 RoPE 无调制) → `projector` Linear(12→1) 压成 `(B,L,2560)` → `refiner_blocks`(2 块, 在 **token 维度** 上 attention) → `txtmlp`(Seq: RMSNorm(2560) / Linear 2560→6144 / GELU(tanh) / Linear 6144→6144)。`txtheads=txtkvheads=20`（无 GQA, head_dim 128）。
- **出口** `last`: RMSNorm `norm.scale` + `modulation.lin [2,6144]`(最终 shift/scale) + `linear`(6144→64) + 额外 `up/down [6144,6144]` 终层精炼。

必须提供的接口（与现有 trainer 约定一致）：
- `class Krea2Transformer2DModel(nn.Module)`，`init_empty_weights()` 构造。
- 属性 `dtype` / `device`；`enable_gradient_checkpointing` / `enable_block_swap(...)` 等 offloader 方法（复用 Qwen-Image 的 offloader 实现，对象换成 `blocks`）。
- `forward(hidden_states, encoder_hidden_states, encoder_hidden_states_mask, timestep, img_shapes, txt_seq_lens, ...) -> noise_pred`（**无 guidance 参数**）。
- 模块级常量 `FP8_OPTIMIZATION_TARGET_KEYS = ["blocks"]`，`EXCLUDE = ["norm", "qknorm", "mod", "tproj", "tmlp", "txtfusion", "last"]`（norm/调制/小投影保精度，参考 FP8 作者只量化 ndim≥2 的 .weight）。
- `load_krea2_model(...)`：用 `load_safetensors_with_lora_and_fp8` 一把梭，支持 fp8 scaled + LoRA 合并，剥离可能的 `model.diffusion_model.` 前缀。

**RoPE（已从官方 `mmdit.py` 核实，见 §12）：** 3 轴位置 `pos=(axis0, H, W)`，head_dim=128 切成 `axes=[32,48,48]`（`headdim-12*(headdim//16)`, `6*(headdim//16)`×2），`theta=1e3`、`ntk=1.0`。文本 token 位置全 0（不旋转），图像 token 用 (行,列) 2D 位置，axis0 恒为 0。拼接顺序 **文本在前、图像在后**（`combined=cat([context,img])`，取输出的 `[txtlen:txtlen+imglen]`），combined 序列长度 pad 到 256 的倍数。所有开放问题已清零。

### 6.2 文本编码器（复用加载器，但取多层 hidden）

`krea2_utils.py` 里：
```python
from musubi_tuner.hidream_o1 import hidream_o1_utils
# load: hidream_o1_utils.load_model(model_path=text_encoder_path, dtype=..., device=...)
#       内部对单文件 .safetensors 走 load_single_checkpoint_model(Qwen3VLForConditionalGeneration)
```
- **加载器直接复用**，模型即 Qwen3-VL 4B，hidden size 2560（已与 `txtmlp` 对齐）。
- prompt 编码：必须 `output_hidden_states=True`，**取 12 层 hidden states 堆叠**（不是单层），交给 DiT 的 `txtfusion.projector` 聚合。`txtfusion` 是 DiT 权重的一部分，不在此处执行。
- **已核实（官方 `encoder.py`）**：`select_layers=(2,5,8,11,14,17,20,23,26,29,32,35)`（每 3 层取一层，共 12 层）；`max_length=512`；system 模板 = `"Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"`（Qwen-Image 式），编码后丢弃前 34 个 token（`prompt_template_encode_start_idx=34`）。输出 `(B, T-34, 12, 2560)` + mask。
- fp8：复用 `hidream_o1` 的 fp8 路径；不足则参考 `qwen_image_utils.prepare_fp8` 的 RMSNorm/DecoderLayer hook（**新增**而非修改既有函数）。

### 6.3 VAE（复用，零新架构）

```python
from musubi_tuner.qwen_image import qwen_image_utils
vae = qwen_image_utils.load_vae(args.vae, input_channels=3, device=..., disable_mmap=True)
```
- `convert_comfyui_state_dict` 已能吃 ComfyUI 版 VAE key，`qwen_image_vae.safetensors` 直接可用。
- latent mean/std、spatial compression 8x、pack 2x2 全部沿用 Qwen-Image 常量。

### 6.4 缓存脚本

- `krea2_cache_latents.py`：把 `qwen_image_cache_latents.py` 的 t2i 分支拿来（去掉 edit/layered/control 分支），VAE 复用，落盘用 `save_latent_cache_qwen_image`。**这条无变化。**
- `krea2_cache_text_encoder_outputs.py`：文本编码器换成 Qwen3-VL（`krea2_utils`），`--text_encoder` 指向 `qwen3vl_4b_fp8_scaled.safetensors`，`--fp8_vl` 保留。**关键差异**：缓存内容是 **12 层 hidden states 堆叠** `(num_layers, T, 2560)` + 有效长度，需在 `cache_io.py` 新增 `save_text_encoder_output_cache_krea2`（多了一个 layer 维度），不能套 Qwen-Image 的单层格式。

### 6.5 训练器 `Krea2NetworkTrainer(NetworkTrainer)`

实现 `NetworkTrainer` 要求的模型相关方法（对照 `QwenImageNetworkTrainer`）：
- `architecture` / `architecture_full_name` → 返回 `ARCHITECTURE_KREA2` / `_FULL`。
- `handle_model_specific_args`：`dit_dtype=bf16`、`_i2v_training=False`、`_control_training=False`。
- `load_vae` → `qwen_image_utils.load_vae`。
- `load_transformer` → `krea2_model.load_krea2_model`。
- `process_sample_prompts` → 用 Qwen3-VL 编码采样 prompt（参考 Qwen-Image 的 t2i 分支，去掉 edit）。
- `do_inference` → flow matching 去噪循环（参考 Qwen-Image t2i 路径：`prepare_latents` → `calculate_shift` → `get_scheduler` → `retrieve_timesteps` → 循环 `model(...)` → CFG → `unpack_latents` → `vae.decode_to_pixels`）。

### 6.6 推理 `krea2_generate_image.py`（采样器已从官方 `sampling.py` 核实，见 §12）
- **两个 checkpoint**：`oss_raw`（非蒸馏 base，默认 steps=28 / CFG guidance=4.5）与 `oss_turbo`（蒸馏，steps≈8 / 关 CFG / 固定 `mu=1.15`）。默认应面向 `oss_raw`，Turbo 作开关。
- **采样器**：flow-matching **Euler**（非 er_sde）。`ts=linspace(1,0,steps+1)`，再做分辨率相关 time-shift：`mu` 在图像 token 数上于 `(x1,y1=0.5)`-`(x2,y2=1.15)` 间线性插值（`x1=(256/16)²`, `x2=(1280/16)²`），`ts=exp(mu)/(exp(mu)+(1/ts-1)^σ)`；蒸馏版传固定 `mu`。
- **CFG 公式**：`v = cond + guidance*(cond-uncond)`（注意是 `cond+g*Δ`，等价标准 CFG scale=g+1）；`guidance<=0` 关闭 CFG 跳过无条件分支。Euler 更新 `img += (t_prev-t_curr)*v`。
- 调度器/shift 与训练 `do_inference` 共用 `krea2_utils` 函数，避免漂移。

---

## 7. CLI 约定（与现有模型一致）

缓存 latent：
```
python krea2_cache_latents.py --dataset_config ds.toml --vae /path/qwen_image_vae.safetensors
```
缓存文本：
```
python krea2_cache_text_encoder_outputs.py --dataset_config ds.toml \
    --text_encoder /path/qwen3vl_4b.safetensors [--fp8_vl]
```
训练：
```
accelerate launch krea2_train_network.py --dataset_config ds.toml \
    --dit /path/krea2.safetensors --vae /path/qwen_image_vae.safetensors \
    --text_encoder /path/qwen3vl_4b.safetensors \
    --network_module networks.lora_krea2 ... [--fp8_scaled] [--fp8_vl] [--blocks_to_swap N]
```
推理：
```
python krea2_generate_image.py --dit ... --vae ... --text_encoder ... \
    --prompt "..." --width 1280 --height 720 --infer_steps 8 --guidance_scale 1.0
```

> LoRA target：主干 `blocks.{0..27}` 下的 `attn.wq/wk/wv/wo` 与 `mlp.gate/up/down`（均为 ndim≥2 的 Linear）。是否一并训练 `txtfusion`/`txtmlp` 作为可选开关。`networks/lora_krea2.py` 按此命名匹配。

---

## 8. 分阶段实施计划（先验证再写）

1. ~~侦察~~ **（已完成）**：已 dump `krea2_turbo_fp8.safetensors` 全部 432 tensor，DiT 结构确认（§11）；文本输入维度 2560（= Qwen3-VL 4B hidden）。剩余待测：txtfusion 取哪 12 层、RoPE 轴、prompt 模板（对照 ComfyUI Krea2 节点）。
2. **VAE 通路**：写 `krea2_cache_latents.py`，验证编码→解码图像可还原（用 Qwen-Image VAE 已知能跑）。
3. **TE 通路**：写 `krea2_cache_text_encoder_outputs.py`，验证 Qwen3-VL 单文件加载 + embed 落盘。
4. **DiT 模型**：按 §11 从头写 `Krea2Transformer2DModel`，对齐 432 个 key，`load_state_dict(strict=True)` 必须无缺失/多余 key。复用注意力后端 / block-swap / fp8 基础设施，不复用 FLUX/Qwen 的模块定义。
5. **推理打通**：`krea2_generate_image.py` 用 OSS 权重复现官方样例（8 步 Turbo），作为正确性基线。
6. **训练打通**：`krea2_train_network.py` 跑通 LoRA，训练中采样可见图像收敛。
7. **fp8 / block swap**：补 fp8_scaled、fp8_vl、blocks_to_swap，验证显存与数值稳定。

每一阶段是独立可验证的，符合"最笨但最清晰"的推进方式。

---

## 9. 开放问题（权重核实后的状态）

| # | 原问题 | 状态 | 结论 |
| --- | --- | --- | --- |
| 1 | DiT 血统（双流/单流） | ✅ 已解决 | **单流联合 MMDiT**，单一 `blocks` 列表，非 FLUX 双流、非 Qwen `transformer_blocks` |
| 2 | DiT 超参 | ✅ 已解决 | 28 层 / dim 6144 / GQA 48Q·12KV × head_dim 128 / in_ch 64 / SwiGLU 16384 / **无 guidance embedding** |
| 3 | Qwen3-VL 接入维度 | ✅ 已解决 | 文本输入 = 2560（Qwen3-VL 4B hidden），`txtmlp` 2560→6144 |
| 4 | 取哪几层 hidden | ✅ 已解决 | `select_layers=(2,5,8,...,35)`，每 3 层取一层共 12 层（官方 `encoder.py`） |
| 5 | 缓存复用 vs 新建 | ✅ 已决 | latent 复用；文本缓存**新建**（存 `(T-34,12,2560)`） |
| 6 | 采样器 | ✅ 已解决 | flow-matching **Euler** + 分辨率相关 time-shift（官方 `sampling.py`），**不是 er_sde** |
| 7 | 是否有非 Turbo base | ✅ 已解决 | **有**：官方 bucket 同发 `oss_raw`(非蒸馏, CFG=4.5/28步) 与 `oss_turbo`(蒸馏)，应优先用 `oss_raw` 训 LoRA |
| 8 | RoPE 轴分配 | ✅ 已解决 | `axes=[32,48,48]`, `theta=1e3`, 3 轴 (axis0=0, H, W)（官方 `mmdit.py`） |

**§9 全部关闭。** 官方源码 `inference.py`/`mmdit.py`/`encoder.py`/`sampling.py` 已逐项核对（见 §12），FP8 dump 的 432 key 与官方 `SingleStreamDiT` 命名完全一致。没有任何阻塞，可以开写。

---

## 10. 一句话总结

Krea 2 = **Krea 自研单流联合 MMDiT（28 层/6144/GQA 48·12/SwiGLU 16384/adaLN-single + txtfusion 多层文本融合） + Qwen-Image VAE + Qwen3-VL 4B 文本编码器**。VAE 和 TE 加载器本仓库已具备；DiT 是全新架构要从头写，文本缓存要改成 `(T,12,2560)` 多层 hidden。**官方四个源码文件已逐行核对（§12），权重 key、RoPE、select_layers、采样器、CFG 全部敲定，且有非蒸馏 `oss_raw` base 可训。没有开放问题，可直接进入 §8 第 2 步开写。**

---

## 11. 附录：`krea2_turbo_fp8.safetensors` 完整张量表（432 tensor）

FP8 元数据：`fp8_format=F8_E4M3`，只量化“以 .weight 结尾且 ndim≥2”的 floating tensor 为 `float8_e4m3fn`，norm/bias/其余保留 F32。

按模块归类（`N` = block 索引）：

```
# 图像入口
first.weight                              F8  [6144, 64]      # Linear(64 -> 6144)
first.bias                                F32 [6144]

# 时间条件 (adaLN-single)  —— Sequential 索引，激活 GELU(tanh)
tmlp.0.weight  F8 [6144, 256]   tmlp.0.bias F32 [6144]       # Linear 256 -> 6144 (idx0)
tmlp.2.weight  F8 [6144, 6144]  tmlp.2.bias F32 [6144]       # Linear 6144 -> 6144 (idx2; idx1=GELU)
tproj.1.weight F8 [36864, 6144] tproj.1.bias F32 [36864]     # idx0=GELU, idx1=Linear 6144 -> 6*6144

# 主干 28 层  blocks.0 .. blocks.27
blocks.N.prenorm.scale                    F32 [6144]          # RMSNorm
blocks.N.postnorm.scale                   F32 [6144]          # RMSNorm
blocks.N.mod.lin                          F32 [36864]         # 每块 6*6144 学习调制偏置 (1-D)
blocks.N.attn.wq.weight                   F8  [6144, 6144]    # 48 头 * 128
blocks.N.attn.wk.weight                   F8  [1536, 6144]    # 12 KV 头 * 128 (GQA)
blocks.N.attn.wv.weight                   F8  [1536, 6144]    # 12 KV 头 * 128 (GQA)
blocks.N.attn.wo.weight                   F8  [6144, 6144]
blocks.N.attn.gate.weight                 F8  [6144, 6144]    # 输出门控
blocks.N.attn.qknorm.qnorm.scale          F32 [128]           # QK-RMSNorm, head_dim=128
blocks.N.attn.qknorm.knorm.scale          F32 [128]
blocks.N.mlp.gate.weight                  F8  [16384, 6144]   # SwiGLU
blocks.N.mlp.up.weight                    F8  [16384, 6144]
blocks.N.mlp.down.weight                  F8  [6144, 16384]

# 文本前端 txtfusion (dim 2560 = Qwen3-VL 4B hidden)
txtfusion.projector.weight                F8  [1, 12]         # 聚合 12 层 hidden
txtfusion.layerwise_blocks.{0,1}.*        # attn 2560 + mlp 6912 + qknorm + prenorm/postnorm
txtfusion.refiner_blocks.{0,1}.*          # 同上结构
txtmlp.0.scale                            F32 [2560]          # idx0 = RMSNorm(2560)
txtmlp.1.weight  F8 [6144, 2560]  txtmlp.1.bias F32 [6144]    # idx1 = Linear 2560 -> 6144
txtmlp.3.weight  F8 [6144, 6144]  txtmlp.3.bias F32 [6144]    # idx3 = Linear 6144 -> 6144 (idx2=GELU)

# 出口 last
last.norm.scale                           F32 [6144]          # RMSNorm
last.modulation.lin                       F32 [2, 6144]       # 最终 shift/scale
last.linear.weight                        F8  [64, 6144]      # 6144 -> 64
last.linear.bias                          F32 [64]
last.up.weight                            F8  [6144, 6144]    # 终层精炼
last.down.weight                          F8  [6144, 6144]
```

顶层前缀统计：`blocks`(364) + `txtfusion`(49) + `last`(6) + `txtmlp`(5) + `tmlp`(4) + `first`(2) + `tproj`(2) = 432。

> 注：该文件是 FP8 作者 repack 版，命名与官方 `SingleStreamDiT` 完全一致。实现时以此 432 key 为唯一真相，`strict=True` 对齐。

---

## 12. 官方参考实现核对（`buckets/krea-community/krea-2`）

官方 OSS 仓提供 `inference.py` / `mmdit.py` / `encoder.py` / `sampling.py`。写 `krea2_model.py` / `krea2_utils.py` 时**照抄数学，换壳对接 musubi 接口**即可。

### 12.1 配置 `single_mmdit_large_wide`
```python
SingleMMDiTConfig(features=6144, tdim=256, txtdim=2560, heads=48, kvheads=12,
                  multiplier=4, layers=28, patch=2, channels=16,
                  txtheads=20, txtkvheads=20, txtlayers=12, theta=1e3, bias=False)
# SwiGLU mlpdim = ceil_to_128( int(2*6144/3) * 4 ) = 16384
```

### 12.2 DiT forward 关键数学（`mmdit.py`）
```python
# RoPE
headdim = 6144 // 48 = 128
axes = [headdim - 12*(headdim//16), 6*(headdim//16), 6*(headdim//16)] = [32, 48, 48]
freqs = PositionalEncoding(axes, theta=1e3, ntk=1.0)(pos)   # pos: (B, L, 3)

# SingleStreamBlock
prescale,preshift,pregate,postscale,postshift,postgate = mod(tvec)   # tvec=tproj(t)
x = x + pregate * attn((1+prescale)*prenorm(x)+preshift, freqs, mask)
x = x + postgate * mlp((1+postscale)*postnorm(x)+postshift)

# Attention (GQA + QK-RMSNorm + sigmoid 输出门)
q,k,v,gate = wq(x),wk(x),wv(x),gate(x)            # q→48头, k/v→12头, headdim128
q,k = qknorm(q), qknorm(k); q,k = ropeapply(q,k,freqs)
out = wo( sdpa(q,k,v,mask,enable_gqa=True) * sigmoid(gate) )

# LastLayer (SimpleModulation: 仅 shift/scale, 输入是 t 不是 tvec)
scale,shift = modulation(t)
x = (1+scale)*norm(x) + shift + up(down(x))       # up/down 是 6144->6144 无激活残差
x = linear(x)                                     # 6144 -> 64

# DiT.forward 总装
img = first(img)                                  # (B,Nimg,64)->(B,Nimg,6144)
t = tmlp(temb(t,256)); tvec = tproj(t)
context = txtmlp(txtfusion(context, mask=_mask(mask[:,:Ltxt])))   # (B,L,12,2560)->(B,L,6144)
combined = cat([context, img], dim=1)             # 文本在前
fulllen -> pad 到 256 倍数 (img/mask/pos 同步 pad)
freqs = posemb(pos)
for block in blocks(28): combined = block(combined, tvec, freqs, _mask(mask))
out = last(combined, t)[:, Ltxt : Ltxt+Nimg]      # 只取图像段
```

### 12.3 文本编码（`encoder.py`）
```python
model_id = "Qwen/Qwen3-VL-4B-Instruct";  max_length = 512
select_layers = (2,5,8,11,14,17,20,23,26,29,32,35)          # 每 3 层一取，共 12
system = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"
prefix = "<|im_start|>system\n"+system+"<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n"
# qwen(output_hidden_states=True); hiddens = stack([hs[i] for i in select_layers], dim=2)
# hiddens = hiddens[:, 34:]  (丢 prefix); mask = mask[:, 34:]   -> (B, T-34, 12, 2560), mask
```

### 12.4 采样（`sampling.py`）
```python
# prepare: 图像 patchify b c (h ph)(w pw) -> b (h w)(c ph pw); 位置 3 轴
imgids[...,0]=0; imgids[...,1]=行; imgids[...,2]=列;  txtpos=zeros(B,Ltxt,3)
pos = cat([txtpos, imgpos]); mask = cat([txtmask, imgmask])     # 文本在前

# 分辨率相关 time-shift (t: 1 -> 0)
ts = linspace(1,0,steps+1)
mu = slope*seq_len + intercept   # 在 (x1,y1=0.5)-(x2,y2=1.15) 间插值; x1=(256/16)², x2=(1280/16)²
ts = exp(mu)/(exp(mu)+(1/ts-1)**sigma)    # 蒸馏版传固定 mu=1.15

# Euler + CFG
for tcurr,tprev in zip(ts[:-1],ts[1:]):
    cond = model(img, context=txt, t=tcurr, pos, mask)
    v = cond + guidance*(cond-uncond) if guidance>0 else cond   # 注意是 cond+gΔ
    img = img + (tprev-tcurr)*v
# ae.compression=8, ae.channels=16; decode -> clamp(-1,1)
```

### 12.5 对接 musubi 的注意点
- 官方 `forward(img, context, t, pos, mask)` 与 musubi trainer 的 `forward(hidden_states, encoder_hidden_states, encoder_hidden_states_mask, timestep, ...)` 需一层适配：`pos`/`combined pad`/`_mask` 在 `krea2_model` 内部构造，对外只暴露标准签名。
- 训练时 timestep 采样应复用同一套 time-shift（`mu` 依图像 token 数），与推理一致；`oss_raw` 用动态 `mu`，Turbo 固定 `mu=1.15`。
- VAE 的 `compression=8`/`channels=16` 与 Qwen-Image VAE 一致；`align = compression*patch = 16`，宽高需 16 的倍数。
- LoRA 默认只打 `blocks.*` 的 `attn.{wq,wk,wv,wo,gate}` 与 `mlp.{gate,up,down}`；`txtfusion`/`txtmlp`/`tproj` 可选。
