---
type: paper
tier: repro
title: "OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models"
arxiv_id: "2604.00688"
source: "Sources/OmniVoice.pdf"
authors: [Han Zhu, Lingxuan Ye, Wei Kang, Zengwei Yao, Liyong Guo, Fangjun Kuang, Zhifeng Han, Weiji Zhuang, Long Lin, Daniel Povey]
year: 2026
venue: "Preprint (under review)"
tags: [TTS, zero-shot, non-autoregressive, discrete-diffusion, masked-generation, multilingual, multi-codebook, LLM-initialization]
concepts: ["[[Non-autoregressiveTTS]]", "[[MaskedGenerativeModeling]]", "[[LLM-basedTTS]]", "[[SpeechTokenizer]]", "[[ResidualVectorQuantization]]", "[[Classifier-FreeGuidance]]", "[[Single-codebookvsMulti-codebook]]", "[[Diffusion-basedTTS]]"]
models: ["[[模型库/SoundStorm|SoundStorm]]", "[[模型库/CosyVoice3|CosyVoice 3]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[Emilia]]", "[[SEED-TTS-Eval]]"]
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-05
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页 + 2 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: OmniVoice 定位于 [[Non-autoregressiveTTS]] 的离散 token 路线 [待确认],具体继承 [[MaskedGenerativeModeling]] [待确认] 中 SoundStorm/MaskGCT 开创的 mask-and-predict 范式,但有两个根本性突破:

1. **架构简化**: MaskGCT 需要 text-to-semantic + semantic-to-acoustic 两阶段级联,OmniVoice 直接 text-to-multi-codebook-acoustic 单阶段,绕过了 semantic token 瓶颈。这与 KB 中 [[SpeechTokenizer]] 记录的主流"先 semantic 后 acoustic"分层路线相悖 -- OmniVoice 论证了单阶段离散 NAR 也可以达到 SOTA。

2. **LLM 初始化进入 NAR**: KB 中 [[LLM-basedTTS]] 记录的 LLM 初始化仅在 AR TTS (CosyVoice 系列等) 中成功,OmniVoice 首次在 NAR 架构中成功复用 AR LLM 权重 (Qwen3-0.6B),且 bidirectional attention 与 causal 预训练不冲突。

**已有认知**: [[ResidualVectorQuantization]] 的多层码本结构 (coarse → fine) 天然支持层级生成,SoundStorm/MaskGCT 都采用 per-layer masking schedule 来对齐 RVQ 层级结构。OmniVoice 的 full-codebook random masking 打破了这一惯例。[[Classifier-FreeGuidance]] 在连续空间扩散模型中广泛使用,OmniVoice 将其扩展到离散 token 的 log-softmax 空间。

**创新判断**: 对比 KB 中 [[Zero-shotSpeechSynthesis]] 记录的当前 SOTA (CosyVoice 3 WER test-zh 0.71%, IndexTTS2 SS 0.865),OmniVoice 在 Seed-TTS test-zh 上达到 WER 0.84% / SIM-o 0.777,与 SOTA 竞争但未全面超越;其独特价值在于 600+ 语言覆盖 + 全开源训练数据。

> 检索命中: [[LLM-basedTTS]]✓, [[Zero-shotSpeechSynthesis]]✓, [[SpeechTokenizer]]✓, [[ResidualVectorQuantization]]✓ | 过滤: [[Non-autoregressiveTTS]](pending-review), [[MaskedGenerativeModeling]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个单阶段离散 NAR TTS 支持 600+ 语言,通过 full-codebook random masking + LLM 初始化实现 SOTA 智能度和说话人相似度
> - **路线**: Text tokens + Multi-codebook acoustic tokens (prompt+masked target) → Bidirectional Transformer (Qwen3-0.6B init) → C 个 codebook-specific prediction heads → 32-step iterative unmasking → Higgs-audio decoder → Waveform
> - **指标**: LibriSpeech-PC WER 1.30% / SIM-o 0.729; Seed-TTS zh WER 0.84% / SIM-o 0.777; FLEURS-102 avg CER 4.00% (82/102 语言 CER≤5%); CMOS +0.44 vs ground truth; RTF 0.032 (16 steps, batch=1) [Table 1, 4, 2, 8]
> - **可借鉴**: (1) Full-codebook random masking: 打破 per-layer masking 惯例,训练效率 C 倍提升且质量更好; (2) 用 AR LLM 权重初始化 bidirectional NAR 模型,免去从头训练; (3) 581k 小时全开源数据 + 语言级重采样公式 (Eq. 2) 的低资源语言扩展策略
> - **局限**: 仅用开源数据训练,部分语言质量受限; 数字/数学模式处理弱; 离散 NAR 尚无类似 flow distillation 的推理加速方法; 32 步推理仍慢于 flow-based 单步方案

## 核心问题

本文要解决两个交叉的难题:

1. **两阶段级联的瓶颈**: 当前 SOTA 离散 NAR TTS (如 MaskGCT) 需要 text→semantic→acoustic 两阶段,存在 error propagation 和 information bottleneck (低比特率 semantic token 丢失声学细节) [§1]。单阶段替代方案 (如 Gallego et al. 2025) 一直在智能度上落后于两阶段系统 [§1]。

2. **语言覆盖的极端不均衡**: 现有多语言 zero-shot TTS 最多覆盖数十种语言 [§2.2],而全球有数百种低资源语言缺乏 TTS 支持。MMS 虽覆盖 1000+ 语言但不支持 zero-shot voice cloning [§2.2]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

OmniVoice 采用单阶段离散 NAR 架构,核心是 discrete masked diffusion objective + bidirectional Transformer [§2.1]:

```
输入:
  Text tokens Y = [instruct tokens | transcript tokens]
  Acoustic token matrix X ∈ R^{T×C} (T 时间步, C=8 codebooks)
     → X_prompt (prefix, unmasked) + X_target (部分 masked)

处理:
  Text → text embedding layer
  Acoustic → C 个 codebook-specific embedding layers → 同时间步求和 → unified embedding
  [Y; X_prompt; X_target] → Bidirectional Transformer (Qwen3-0.6B backbone)

输出:
  C 个 codebook-specific prediction heads → 每个预测对应 codebook 的 token 概率分布
  Loss = -Σ_{(t,c)∈M} log P(x_{t,c} | X, Y; θ)  [Eq. 1]
```

[论文原文] 这种端到端架构"bypassing the complexity and limitations of cascaded pipelines"[§2.1],因为不存在 semantic token 中间表示,所以不会有 information bottleneck 或 error propagation。

[agent 解读] 与 MaskGCT 的根本区别在于: MaskGCT 先用 VQ-VAE 将语音编码为 semantic token 再用 masked generation 生成 acoustic token (两次量化),OmniVoice 直接从 text 到 8-codebook acoustic token (一次量化),减少了信息丢失环节。

### 关键设计选择

#### 1. Full-Codebook Random Masking [§2.1.1]

[论文原文] Per-layer masking (SoundStorm/MaskGCT 采用) 每次只在单个 codebook 层随机 mask 并计算 loss,导致"optimizes only a sparse subset of the token matrix at each iteration, leading to suboptimal training efficiency"[§2.1.1]。

OmniVoice 的方案: 对整个 T×C token 矩阵,**每个 entry 独立** 以 Bernoulli(p_t) 采样 mask,p_t ~ U(0,1) [§2.1.1]:

| 策略 | 每次迭代参与 loss 的 token 比例 | 效果 |
| --- | --- | --- |
| Per-layer (SoundStorm/MaskGCT) | 平均 ~1/C (仅选中的一层) | 收敛慢 |
| Full-codebook random | 平均 ~50% (全矩阵的一半) | C 倍更多 token 参与训练 |

[论文原文] "on average, 50% of the tokens are used for loss computation, C times more than per-layer masking strategy, significantly accelerating convergence and boosting generative quality" [§2.1.1]。

[agent 解读] 为什么 per-layer masking 不是必须的? Per-layer 设计是为了对齐逐层推理 (先生成低层再生成高层),但 OmniVoice 的推理也打破了这个约束 -- 它用 confidence-based 选择 + layer penalty 来鼓励 (但不强制) 低层先 unmask。这说明 RVQ 层间的严格顺序约束可以被放松。

消融验证 [Table 5]:
- SoundStorm-style mask: WER 3.00, SIM-o 0.661
- MaskGCT-style mask: WER 2.04, SIM-o 0.660
- Full-codebook random mask: WER 1.57, SIM-o 0.697
- Full-codebook + single-codebook loss: WER 2.85 (对照,证明 dense loss 而非仅 masking 策略的贡献)

#### 2. LLM 初始化 [§2.1.2]

[论文原文] 单阶段离散 NAR TTS 在智能度上一直不如两阶段或 AR 系统。OmniVoice 用 Qwen3-0.6B 预训练权重初始化 bidirectional Transformer backbone,"OmniVoice is the first NAR TTS model that successfully leverages LLM initialization"[§2.1.2]。

[论文原文] 为什么 causal LLM 的权重能用于 bidirectional 架构? 作者的回答是"we empirically find that their pre-trained knowledge translates well to our bidirectional architecture"[§2.1.2]。

[agent 解读] 这可能因为 Transformer 的 self-attention 权重和 FFN 权重编码的语言知识 (词汇关系、句法结构) 在本质上是双向的 -- causal mask 只是推理时的约束,不影响权重本身的知识表达。CosyVoice 系列在 AR TTS 中也利用了 LLM 初始化,但它们是 causal→causal,而 OmniVoice 是 causal→bidirectional,这是更大胆的跨范式迁移。

消融验证 [Table 6]:
| 初始化方式 | LR | LibriSpeech WER | Seed-en WER | Seed-zh WER |
| --- | --- | --- | --- | --- |
| LLM init | 1e-4 | 1.57 | 1.72 | 0.89 |
| Random init (最优 LR) | 5e-4 | 2.56 | 2.07 | 1.01 |

[Table 6] 即使穷搜学习率,随机初始化的 WER 仍显著高于 LLM 初始化,确认语言先验对智能度的关键贡献。

#### 3. 多语言数据策略 [§2.2]

- **数据来源**: 50 个开源数据集, 581k 小时, 600+ 语言 [§2.2, Fig. 3]
- **语言重采样**: 低资源语言按 r_i = max(1, round((D_max/D_i)^{1-β})) 上采样,β=0.8 [Eq. 2]
- **文本处理**: 直接用 LLM 的 subword tokenizer,不做 G2P 转换 [§2.2]

[论文原文] 这个策略的关键选择是"eliminating cumbersome grapheme-to-phoneme conversion and language-specific text normalization"[§2.2],这对 600+ 语言至关重要,因为大多数低资源语言没有可用的 G2P 工具。

#### 4. 多维可控性 [§2.3]

- **Prompt denoising**: 训练时对 prompt 注入噪声,搭配 `<|denoise|>` 指令 token,使模型从噪声 prompt 生成干净语音 [§2.3.1]
- **Speaker-attribute voice design**: 通过属性 (性别/年龄/口音等) 指令控制音色,无需音频 prompt [§2.3.2]
- **Phonetic override**: 混合 pinyin/phoneme 替换,解决多音字和专业术语问题 [§2.3.3]

### 模块细节

#### Embedding 层 (`OmniVoice.__init__`, `_prepare_embed_inputs`)

- **Input**: `input_ids: [B, C, L]` (C=8 codebooks), `audio_mask: [B, L]` (bool, True=audio position)
- **Output**: `inputs_embeds: [B, L, H]` (H=hidden_size)
- **Structure**: 文本 token 和音频 token 使用不同 embedding:
  - 文本: 直接用 LLM 的 `get_input_embeddings()` 处理 `input_ids[:, 0, :]` (第 0 层)
  - 音频: 用一个统一的 `nn.Embedding(C * V, H)`,每层 token ID 加上 `layer_offset = layer_idx * V` (V=1025),C 个 codebook embedding **逐层求和** 得到 unified embedding
  - 最终通过 `audio_mask` 选择: audio position 用 audio embedding,text position 用 text embedding
- **Key params**: `audio_vocab_size=1025` (1024 real + 1 mask), `audio_mask_id=1024`

[agent 解读] 这个 embedding 设计比 MaskGCT 更简洁 -- MaskGCT 需要分别处理 semantic 和 acoustic 两种 token space,而 OmniVoice 只有一个统一的 audio token space。`C * V` 的 flat embedding 表通过 offset 实现层区分,比 C 个独立 embedding table 更紧凑(共享底层 CUDA kernel)。

#### Prediction Head (`audio_heads`)

- **Input**: `hidden_states: [B, L, H]`
- **Output**: `audio_logits: [B, C, L, V]`
- **Structure**: 单个 `nn.Linear(H, C * V, bias=False)`,输出 reshape 为 `[B, L, C, V]` 再 permute 为 `[B, C, L, V]`
- **Key params**: C=8, V=1025

[agent 解读] 用单个大 Linear 而非 C 个独立 head,在 GPU 上更高效(一次 matmul 而非 C 次)。论文称"C independent, codebook-specific prediction heads"[§2.1],但实现上是一个 fused head。

#### Loss 计算 (`forward`)

- **Codebook-weighted cross-entropy**: 每层 token 的 cross-entropy 分别计算均值,再用 `audio_codebook_weights = [8, 8, 6, 6, 4, 4, 2, 2]` 归一化后加权求和
- 低层 codebook (coarse) 权重更高,因为低层错误对音质影响更大
- `labels == -100` 的位置不参与 loss (标准 PyTorch ignore_index)

```python
per_token_loss = F.cross_entropy(logits.permute(0,3,1,2), labels, reduction="none", ignore_index=-100)
# shape: [B, C, L]
layer_means = (per_token_loss * valid_mask).sum(dim=(0,2)) / valid_mask.sum(dim=(0,2)).clamp(min=1.0)
# shape: [C]
loss = (layer_means * normalized_weights).sum()
```

[agent 解读] 这种加权策略与 RVQ 的层级结构对齐 -- 第 1-2 层 (权重 8) 编码粗粒度频谱包络,第 7-8 层 (权重 2) 编码细粒度残差。论文未显式讨论此权重选择,但消融 [Table 5] 中 "single-codebook loss" 的对照实验间接验证了多层加权 loss 的重要性。

### 训练策略

#### Masking 逻辑 (`OmniVoiceSampleProcessor.__call__`)

```python
# 1. 随机决定是否 drop conditioning (10% 概率)
drop_cond = random.uniform(0,1) < 0.1

# 2. 如果 drop,则 prompt_ratio=0, 丢弃 text/lang/instruct (纯 unconditional)
# 3. 否则,随机 prompt_ratio ~ U(0.0, 0.3),mask_ratio ~ U(0.0, 1.0)

# 4. 音频 token 前 prompt_length 帧不 mask (prompt region)
# 5. 剩余帧每个 entry 独立以 mask_ratio 采样 mask
token_mask = torch.rand(maskable_region.shape) < mask_ratio
audio_inputs[:, prompt_length:][token_mask] = audio_mask_id  # 1024
audio_labels[:, prompt_length:][~token_mask] = -100  # 不计 loss
```

**关键配置参数** (从 `train_config_emilia.json`):
| 参数 | 值 | 含义 |
| --- | --- | --- |
| `drop_cond_ratio` | 0.1 | 10% 概率丢弃所有条件 (CFG 训练) |
| `prompt_ratio_range` | [0.0, 0.3] | prompt 占比 0-30% |
| `mask_ratio_range` | [0.0, 1.0] | mask 比例 0-100% |
| `language_ratio` | 0.8 (多语言) / 0.0 (Emilia) | 是否附加语言 ID |
| `use_pinyin_ratio` | 0.3 (多语言) / 0.0 (Emilia) | Pinyin 文本替换概率 |
| `instruct_ratio` | 1.0 (多语言) / 0.0 (Emilia) | 指令 token 使用概率 |

#### Sequence Packing (`PackingDataCollator`)

- 使用 `flex_attention` 时,多个样本拼接成一个长序列 [1, C, L] (L = `batch_tokens` = 8192)
- `document_ids` 标记每个 token 属于哪个文档,通过 `create_block_mask` 确保跨文档不注意
- 不使用 `flex_attention` 时 fallback 到 `PaddingDataCollator`,标准 [B, C, max_len] padding

[agent 解读] Sequence packing 在 LLM 训练中常用但在 TTS 中较少见。OmniVoice 能用 packing 是因为 bidirectional attention mask 本身就需要 custom mask (不是标准 causal),所以 flex_attention 的 block mask 自然兼容 packing 场景。

#### Loss 设计

- **Formula**: L = Σ_c w_c * (1/|M_c|) Σ_{(t,c)∈M} -log P(x_{t,c} | X, Y; θ)
- **Meaning**: 加权的 per-layer masked cross-entropy,低层权重更高
- **Weight**: [8, 8, 6, 6, 4, 4, 2, 2] → 归一化后 [0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]

#### 训练配置

- **Optimizer**: AdamW, peak LR 1e-4, weight_decay 0.01, max_grad_norm 1.0 [§3.3]
- **Learning rate**: cosine schedule, 3% warmup (ratio-based) [§3.3]
- **Batch size**: 8192 tokens/GPU (sequence packing) [§3.3]
- **Precision**: BF16, TF32 enabled [§3.3]
- **Hardware**: 8x H800 GPUs [§3.3]
- **多语言版**: 2M updates, 9.66 天; Emilia 版: 300k updates, 1.33 天 [§3.3]
- **Attention**: `flex_attention` (default), fallback to `sdpa`

#### 数据处理

- **Audio tokenizer**: Higgs-audio v2, 8 codebooks, 24kHz [§3.2]
- **预处理管线**: 原始音频 → speech restoration model (去噪增强) → rule-based 过滤 → Higgs-audio encode → WebDataset tar + JSONL labels
- **Token 格式**: `audio_tokens: [C, T]` 存为 npy 格式,打包进 WebDataset tar shard
- **数据加载**: WebDataset → SampleDecoder → OmniVoiceSampleProcessor → PackingDataCollator → model.forward()

### 推理流程

#### 完整推理 pipeline (`generate` → `_generate_iterative`)

```
1. _preprocess_all():
   - 文本 tokenize (Qwen3 subword tokenizer)
   - 如有 ref_audio: load → remove_silence → trim → Higgs-audio encode → ref_audio_tokens [C, T]
   - 如有 instruct: validate + normalize (性别/年龄/口音等)
   - 估计目标长度: RuleDurationEstimator (基于字符权重的比例缩放)

2. _prepare_inference_inputs():
   - 构建 style tokens: <|denoise|> + <|lang_start|>XX<|lang_end|> + <|instruct_start|>XX<|instruct_end|>
   - 构建 text tokens: <|text_start|>ref_text + target_text<|text_end|>
   - 构建 target: 全 MASK (1024) 的 [1, C, T_target]
   - 拼接: [style | text | ref_audio | masked_target]
   - audio_mask: 标记哪些位置是 audio (用于 embedding 选择)

3. _generate_iterative() — 32 步迭代解码:
   - Batch 包含 2*B 个样本: 前 B 个是 conditional, 后 B 个是 unconditional (CFG)
   - unconditional 只包含 masked target 部分 (无 text/style/prompt)
   
   For step in range(32):
     a. 前向: model(batch_input_ids, batch_audio_mask, batch_attention_mask) → logits
     b. CFG: log_probs = log_softmax(c_logits + scale * (c_logits - u_logits))
        - guidance_scale = 2.0
     c. Token 选择: argmax (确定性, class_temperature=0.0)
     d. Position 选择:
        - confidence = log_probs.max(dim=-1)  (每个位置的最大 log-prob)
        - confidence -= layer_idx * layer_penalty_factor  (鼓励低层先 unmask)
        - Gumbel sampling: confidence / temperature + Gumbel_noise  (temperature=5.0)
        - 选 top-k 个 masked 位置 unmask (k 由 schedule 决定)
     e. Schedule: rn = τ*(n/N) / (1+(τ-1)*(n/N)), τ=0.1 (前密后疏)
     f. 更新: batch_input_ids 中对应位置从 MASK 替换为预测 token

4. _decode_and_post_process():
   - Higgs-audio decode → waveform
   - remove_silence (mid_sil=500ms)
   - volume normalization (match ref_rms)
   - fade_and_pad
```

#### Duration 估计 (`RuleDurationEstimator`)

- 基于 Unicode 范围的字符权重表: CJK=3.0, Latin=1.0, Arabic=1.5, Hangul=2.5 等
- 估计公式: `T_target = T_ref * W_target / W_ref` [Eq. 4]
- 短文本有 boost: 当估计值 < `low_threshold` (50 frames) 时,用 power-curve 提升

#### 长文本分块 (`_generate_chunked`)

- 当估计时长 > 30s 时触发分块
- 按标点符号切分文本,每块约 15s
- 如有 ref_audio: 所有块共享同一 ref_audio
- 如无 ref_audio: 第 0 块先生成,后续块以第 0 块输出作为 ref
- 块间 cross-fade 拼接

## 实验

### 中英文评估

| 指标 | OmniVoice | OmniVoice-Emilia | MaskGCT | F5-TTS | CosyVoice3 | Qwen3-TTS | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIM-o ↑ | **0.729** | 0.697 | 0.691 | 0.655 | 0.694 | 0.704 | LibriSpeech-PC | [Table 1] |
| WER ↓ | **1.30** | 1.57 | 2.26 | 1.89 | 1.59 | 1.60 | LibriSpeech-PC | [Table 1] |
| UTMOS ↑ | 4.28 | 4.23 | 3.91 | 3.89 | 4.28 | **4.41** | LibriSpeech-PC | [Table 1] |
| SIM-o ↑ | **0.741** | 0.717 | 0.713 | 0.664 | 0.696 | 0.708 | Seed-TTS en | [Table 1] |
| WER ↓ | 1.60 | 1.72 | 2.88 | 1.85 | 2.17 | **1.54** | Seed-TTS en | [Table 1] |
| SIM-o ↑ | 0.777 | 0.765 | 0.773 | 0.750 | **0.778** | 0.766 | Seed-TTS zh | [Table 1] |
| WER ↓ | **0.84** | 0.89 | 2.40 | 1.53 | 1.14 | 1.15 | Seed-TTS zh | [Table 1] |
| CMOS ↑ | **+0.44** | +0.42 | -0.38 | - | - | +0.40 | 主观 | [Table 2] |
| SMOS ↑ | **3.80** | 3.58 | 3.20 | - | - | 3.65 | 主观 | [Table 2] |

OmniVoice-Emilia 在相同 Emilia 训练数据下全面超越 NAR baselines (F5-TTS, ZipVoice, MaskGCT);完整多语言版在说话人相似度和智能度上与 CosyVoice 3 / Qwen3-TTS 等 AR 系统竞争 [§4.1]。

### 多语言评估

| 指标 | OmniVoice | MiniMax-Speech | ElevenLabs v2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| Avg SIM-o ↑ | **0.830** | 0.766 | 0.655 | MiniMax-24 | [Table 3] |
| Avg WER ↓ | **2.850** | 3.774 | 10.950 | MiniMax-24 | [Table 3] |

| 指标 | OmniVoice | Ground truth | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Avg CER ↓ | **4.00%** | 5.11% | FLEURS-102 | [Table 4] |
| Languages CER≤5% | **82** | 75 | FLEURS-102 | [Table 4] |
| Languages CER≤10% | **95** | 92 | FLEURS-102 | [Table 4] |

[论文原文] OmniVoice 在 FLEURS-102 上的 avg CER (4.00%) 甚至低于 ground truth (5.11%),作者谨慎指出"we do not claim OmniVoice can generate speech of better quality than the ground truth for all languages. However, OmniVoice's performance has exceeded the measurement capability of existing ASR models"[§4.2]。

[agent 解读] 这个结果的合理解释是: OmniVoice 生成的语音比 FLEURS 原始录音更"干净"(经过 speech restoration 预处理的训练数据),ASR 模型对干净语音识别更准,导致 CER 反而更低。这不等同于合成质量超越真人。

### 消融实验

1. **Masking 策略** [Table 5]: Full-codebook random > MaskGCT-style > SoundStorm-style (WER: 1.57 vs 2.04 vs 3.00)
2. **LLM 初始化** [Table 6]: LLM init WER 1.57 vs random init 最优 WER 2.52 (差距显著)
3. **Prompt denoising** [Table 7]: 开启后 UTMOS 4.23→4.32, SIM-o 0.697→0.668 (更干净但更标准化)

### 推理速度

| Steps | BS=1 RTF | BS=8 RTF | 出处 |
| --- | --- | --- | --- |
| 16 | 0.0319 | 0.0224 | [Table 8] |
| 32 | 0.0598 | 0.0414 | [Table 8] |

[Table 8] 16 步推理 RTF 0.032,优于 ZipVoice 同设定下的 0.056,且 16 步质量仍可接受 (Appendix B)。

## 复现要点

### 1. 环境与依赖

代码仓库: https://github.com/k2-fsa/OmniVoice (Apache 2.0, PyPI 可装: `pip install omnivoice`)

核心依赖: PyTorch 2.8+, transformers (含 HiggsAudioV2TokenizerModel), accelerate, webdataset, torchaudio。`flex_attention` 需要较新的 PyTorch 版本;如 GPU 不支持可 fallback 到 SDPA (设 `attn_implementation: "sdpa"`)。

**预训练模型**: `k2-fsa/OmniVoice` (HuggingFace), 总参数 0.8B [§3.2, Table 1]。

**Audio tokenizer**: Higgs-audio v2 (`eustlb/higgs-audio-v2-tokenizer`), 8 codebooks, 24kHz。注意 MPS 不支持 (output channels > 65536),需 fallback 到 CPU。

### 2. 推理复现 (最简路径)

```python
from omnivoice import OmniVoice
import torch, soundfile as sf

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)

# Voice cloning
audio = model.generate(text="...", ref_audio="ref.wav", ref_text="...", num_step=32)
sf.write("out.wav", audio[0], 24000)

# Voice design (无 ref_audio)
audio = model.generate(text="...", instruct="female, british accent")
```

**关键推理参数** (默认值来自 `OmniVoiceGenerationConfig`):
| 参数 | 默认值 | 作用 | 调整建议 |
| --- | --- | --- | --- |
| `num_step` | 32 | 迭代解码步数 | 16 步质量仍可接受,RTF 减半 |
| `guidance_scale` | 2.0 | CFG 强度 | 增大→更贴合文本但可能失真 |
| `t_shift` | 0.1 | schedule 前移参数 | 越小→前期 unmask 越密集 |
| `layer_penalty_factor` | 5.0 | 低层优先解码强度 | 0=不偏好层顺序 |
| `position_temperature` | 5.0 | 位置选择随机性 | 越大→越随机 |
| `class_temperature` | 0.0 | token 选择温度 | 0=argmax (确定性) |
| `denoise` | True | prompt 去噪模式 | False=不去噪 |
| `audio_chunk_threshold` | 30.0s | 触发分块生成的阈值 | |
| `audio_chunk_duration` | 15.0s | 每块目标时长 | |

### 3. 训练复现

#### 3a. Emilia 双语训练 (论文 Emilia 变体)

```bash
# 0. 下载 Emilia 数据 + JSONL manifests
# 1. Tokenize: 用 Higgs-audio v2 提取 8-codebook tokens → WebDataset tar shards
bash examples/run_emilia.sh  # stage=0,1,2
```

训练配置 (`train_config_emilia.json`):
- 300k updates, LR=1e-4, cosine schedule, 3% warmup
- 8 GPUs, 8192 tokens/GPU (packing), BF16
- 无 language_ratio / instruct_ratio / pinyin (纯 TTS,无多语言特性)

预计: 1.33 天 (8x H800) [§3.3]。

#### 3b. 微调 (从预训练 checkpoint)

```bash
# 准备 JSONL: {"id": "...", "audio_path": "...", "text": "...", "language_id": "..."}
bash examples/run_finetune.sh
```

微调配置 (`train_config_finetune.json`):
- `init_from_checkpoint: "k2-fsa/OmniVoice"`, 5k updates, LR=1e-5
- 其余同 Emilia 配置

### 4. 数据准备要点

- **JSONL manifest 格式**: `{"id": "sample_001", "audio_path": "/path/to/audio.wav", "text": "transcription", "language_id": "en"}`
- **Tokenization 管线**: `omnivoice.scripts.extract_audio_tokens` -- 接收 JSONL,输出 WebDataset tar shards (.tar + .jsonl 对)
- **WebDataset 格式**: 每个 shard 包含 `.npy` (audio tokens [C, T]) + metadata。manifest 文件 `data.lst` 格式: `tar_path label_jsonl_path num_items num_seconds`
- **数据过滤**: 使用 speech restoration model (Sidon) 增强 + rule-based 过滤无效转录 [§2.2]

### 5. 架构细节与复现陷阱

1. **Embedding offset 机制**: 音频 embedding 表大小是 `C * V = 8 * 1025 = 8200`,第 c 层 token id 加 `c * 1025` 后查表,8 层 embedding 求和。错误实现这一步会导致层间混淆。

2. **flex_attention block mask**: 用 `create_block_mask` + `document_ids` 实现 sequence packing 中的跨文档隔离。如不使用 flex_attention,需切换到 PaddingDataCollator + 4D attention mask (`[B, 1, max_len, max_len]`,bidirectional,仅屏蔽 padding)。

3. **CFG 实现**: 推理时 batch 翻倍为 `2*B`,前 B 个是 conditional (完整输入),后 B 个是 unconditional (只有 masked target)。CFG 在 **log-softmax 空间**操作: `log_softmax(c_log_probs + scale * (c_log_probs - u_log_probs))`。注意外层还有一个 `log_softmax` -- 这是双重 softmax normalize。

4. **Codebook-weighted loss**: 权重 `[8,8,6,6,4,4,2,2]` 归一化后使用。每层 loss 独立计算均值再加权求和,不是简单的 per-token 加权。label=-100 的 position 不参与该层的 loss 和 count。

5. **Prompt denoising 训练**: 需要 `clean_start_token_idx` 字段标记干净语音起始位置。训练时,prompt 区域注入合成噪声 + 混响,模型学习从噪声 prompt 恢复干净语音。此功能通过 `<|denoise|>` special token 触发。

6. **Special tokens**: 需要添加 7 个: `<|denoise|>`, `<|lang_start|>`, `<|lang_end|>`, `<|instruct_start|>`, `<|instruct_end|>`, `<|text_start|>`, `<|text_end|>`。从非 OmniVoice checkpoint 初始化时需 `resize_token_embeddings`。

7. **Duration 估计**: 推理时的目标帧数由 `RuleDurationEstimator` 基于字符权重估计。CJK 字符权重 3.0x Latin,Arabic 1.5x。这个估计器不依赖额外 duration predictor 模型。

8. **长文本处理**: 估计时长 > 30s 时自动按标点分块 (每块 ~15s),块间 cross-fade 拼接。无 ref_audio 时,第 0 块的输出自动作为后续块的 ref。

### 6. 评估复现

```bash
pip install omnivoice[eval]
bash examples/run_eval.sh
```

支持: `librispeech_pc`, `seedtts_en`, `seedtts_zh`, `fleurs`, `minimax`。

评估工具:
- **SIM-o**: WavLM-based ECAPA-TDNN speaker verification model
- **WER/CER**: Hubert ASR (LibriSpeech), Paraformer-zh (中文), Omnilingual ASR (FLEURS), Whisper-large-v3 (其余)
- **UTMOS**: 自动 MOS 评估

### 7. 资源需求估计

| 场景 | GPU | 显存 | 时间 |
| --- | --- | --- | --- |
| 推理 (单句) | 1x H20/A100 | ~4-6 GB (FP16) | RTF 0.032-0.060 |
| 训练 Emilia (300k steps) | 8x H800 | ~40 GB/GPU (BF16) | 1.33 天 |
| 训练多语言 (2M steps) | 8x H800 | ~40 GB/GPU (BF16) | 9.66 天 |
| 微调 (5k steps) | 2x GPU | ~20 GB/GPU | 数小时 |

## 局限性

1. **仅开源数据**: 数据标注质量和声学质量不均匀,有提升空间 [§E]
2. **数字/数学处理**: 未做 text normalization,复杂数字序列可能出错 [§E]
3. **推理加速**: 连续 NAR (如 flow matching) 可用 flow distillation 减少步数,离散 NAR 目前无类似技术 [§E]
4. **指令跟随能力**: 受限于 instruction-tuning 数据的多样性和质量 [§E]
5. **Prompt denoising 的 trade-off**: 开启后 SIM-o 降低 (0.697→0.668),即去噪能力以牺牲部分说话人相似度为代价 [Table 7]

## 点评

**核心贡献**: OmniVoice 在两个方向上推进了 NAR TTS 的边界:

1. **架构简化的成功验证**: 此前 MaskGCT 的成功建立在"两阶段是必要的"假设上,OmniVoice 证明单阶段离散 NAR 可以达到甚至超越两阶段系统 -- 只要训练效率 (full-codebook masking) 和初始化 (LLM) 到位。这简化了系统设计,减少了维护成本。

2. **LLM→NAR 的知识迁移**: 首次验证 causal LLM 权重可以初始化 bidirectional NAR 架构并带来显著智能度提升,这打开了一个新的研究方向 -- NAR TTS 不必从零学语言知识。

**对 600+ 语言覆盖的态度应审慎**: Fig. 4 和 Table 10 显示,低资源语言 (<10 小时) 虽然有不少 CER<5%,但也有多个语言 CER>10% (如 Urdu 28.73%, Cantonese 21.92%, Lao 25.51%)。"600+ 语言"的宣传值与实际质量保证之间存在差距。

**与 ZipVoice 的关系**: 同一团队 (Xiaomi + Daniel Povey) 的前序工作 ZipVoice 是 flow-matching NAR,OmniVoice 转向 discrete diffusion,表明该团队在探索 NAR TTS 的不同建模范式。

**代码质量评价** [agent 解读]: 开源实现非常完整,覆盖推理/训练/微调/评估全流程。代码结构清晰 (model/data/training 分层),支持 PyPI 安装和 HuggingFace 集成。Sequence packing + flex_attention 的训练实现是亮点。Gradio demo 和 Colab notebook 降低了使用门槛。WebDataset 数据管线成熟但对不熟悉 WDS 格式的用户有学习成本。

## 可复用的 idea

1. **Full-codebook random masking**: 对任何使用 multi-codebook token 的 masked generation 任务 (音乐、音效、多模态),打破 per-layer masking 惯例可能都能获得类似的训练加速。核心 insight 是 dense loss > sparse loss,即使推理时仍需近似层级顺序。

2. **AR LLM → NAR 初始化**: 对 bidirectional Transformer 架构,直接用 causal LLM 预训练权重初始化是一个低成本的提升手段。前提是 NAR backbone 与 LLM 架构相同 (OmniVoice 的 backbone 就是 Qwen3 架构);对架构不同的 NAR 模型需要额外适配。

3. **语言级重采样公式**: Eq. 2 的 r_i = max(1, round((D_max/D_i)^{1-β})) 是一个简洁实用的低资源语言上采样方案,β 可调 (0=均匀, 1=自然分布, 0.8=温和平衡)。

4. **Script-dependent duration estimation**: 用 per-character 权重根据文字系统 (CJK vs Latin) 调整目标时长,比统一字符计数更合理,且不需要额外 duration predictor。代码中的 `RuleDurationEstimator` 可直接复用。

5. **Sequence packing + flex_attention 用于 NAR TTS**: 将 LLM 训练中的 sequence packing 技术引入 bidirectional NAR TTS,通过 document_ids + block mask 实现跨样本隔离,提升 GPU 利用率。

6. **Fused prediction head**: 用单个 `nn.Linear(H, C*V)` 替代 C 个独立 head,GPU 上更高效。通过 reshape + permute 实现逻辑上的 per-codebook 输出。
