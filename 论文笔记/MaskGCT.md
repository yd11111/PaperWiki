---
type: paper
tier: deep
title: "MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer"
arxiv_id: "2409.00750"
source: "Sources/MaskGCT.pdf"
authors: [Yuancheng Wang, Haoyue Zhan, Liwei Liu, Ruihong Zeng, Haotian Guo, Jiachen Zheng, Qiang Zhang, Xueyao Zhang, Shunsi Zhang, Zhizheng Wu]
year: 2024
venue: "Preprint"
tags: [TTS, zero-shot, non-autoregressive, masked-generative, speech-codec, semantic-token]
concepts: ["[[SpeechTokenizer]]", "[[ResidualVectorQuantization]]", "[[ConditionalFlowMatching]]"]
models: ["[[SoundStream]]", "[[EnCodec]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[Emilia]]", "[[SEED-TTS-Eval]]"]
kb_context_sources: 0
status: draft
created: 2026-06-01
updated: 2026-06-01
---

## KB 背景

> [!info] KB 背景 (KB 检索未启用 — P1 阶段)

## 速查

> [!summary] 速查
> - **一句话**: 用 masked generative modeling 完全替代自回归,构建无需 text-speech alignment 和 phone-level duration 的非自回归 zero-shot TTS
> - **路线**: Text + prompt → T2S (Masked Generative Transformer 695M, 50 步迭代) → semantic tokens (50Hz) → S2A (Masked Generative 353M, 逐层) → 12 层 acoustic tokens → Vocos decoder → Waveform
> - **指标**: SIM-O 0.728 / WER 2.466% (SeedTTS test-en); SIM-O 0.777 / WER 2.183% (SeedTTS test-zh); SMOS 4.24 / 4.11 (LibriSpeech / test-zh); 训练数据 100K h Emilia
> - **可借鉴**: VQ-VAE 量化 w2v-BERT 2.0 特征做 semantic codec (优于 k-means,已被 IndexTTS2 采用); text+prompt 做 prefix + bidirectional attention 的 in-context learning 范式; flow matching 预测 total duration 而非 phone-level
> - **局限**: 仍需外部 duration predictor; 多语言扩展仅重训 T2S 未联合 tokenizer/S2A; 代码已开源但 100K h Emilia 数据集获取受限

## 核心问题

现有 zero-shot TTS 系统存在两大范式缺陷 [§1]:
1. **自回归 (AR) 系统** (VALL-E, VoiceCraft): 逐 token 生成导致鲁棒性差 (hallucination, 重复)、推理速度慢,且缺乏对生成语音总时长的显式控制
2. **非自回归 (NAR) 系统** (VoiceBox, NaturalSpeech 3): 需要显式的 text-speech alignment 监督和 phone-level duration prediction,导致流程复杂,且生成的语音多样性受限

MaskGCT 的目标: 构建一个**完全非自回归**的 TTS 系统,既不需要 text-speech alignment 监督,也不需要 phone-level duration prediction,同时保持对语音总时长的灵活控制 [§1]。

## 方法: 它怎么 work

### 整体架构

MaskGCT 是一个四组件 two-stage 系统 [Fig 1]:

1. **Speech Semantic Codec**: 将语音波形编码为 semantic token 序列 (VQ-VAE on w2v-BERT 2.0 features)
2. **Text-to-Semantic (T2S) Model**: 给定文本和 prompt semantic tokens,通过 masked generative modeling 预测目标 semantic tokens
3. **Semantic-to-Acoustic (S2A) Model**: 给定 semantic tokens 和 prompt acoustic tokens,通过 masked generative modeling 预测多层 acoustic tokens
4. **Speech Acoustic Codec**: 将 acoustic tokens 解码为语音波形 (RVQ-based codec with Vocos decoder)

### 关键设计选择

#### 1. Speech Semantic Representation Codec [§3.2.1]

**问题**: 以往使用 k-means 离散化 SSL features 会丢失信息,尤其是声调语言 (中文) 的韵律信息 [§3.2.1]。

**方案**: 训练 VQ-VAE 来量化 w2v-BERT 2.0 第 17 层的 hidden states [§3.2.1, Appendix A.4]:
- Encoder/Decoder: 各 12 层 ConvNext blocks, kernel size 7, hidden 384 [Table 9]
- 使用 factorized codes (灵感来自 VQ-GAN 改进): 将 encoder 输出投影到低维空间 (codebook dimension 8) [§3.2.1]
- Codebook size: 8,192 entries, 单层 codebook [Table 9]
- Sample rate: 16K, hop size 320 → 50 Hz token rate [Table 9]
- 参数量: 44M [Table 9]

**损失函数** [§3.2.1]:
$$\mathcal{L}_\text{total} = \frac{1}{Td}(\lambda_\text{rec} \cdot ||S - \hat{S}||_1 + \lambda_\text{codebook} \cdot ||sg(\mathcal{E}(S)) - E||_2 + \lambda_\text{commit} \cdot ||sg(E) - \mathcal{E}(S)||_2)$$

**为什么 VQ-VAE 优于 k-means**: k-means 是 post-hoc 聚类,codebook 与 feature 空间的结构无关;VQ-VAE 是 end-to-end 优化,encoder-decoder 与 codebook 联合学习,可以将信息有效压缩到低维空间同时最小化重建损失,保留更多韵律/声调信息。

#### 2. Text-to-Semantic Model (T2S) [§3.2.2]

**架构**: Llama-style Transformer (16 layers, 1536 dim, 695M params for T2S-Large) [Table 7, §A.1]:
- Gated Linear Units with GELU (SwiGLU) activation [§A.1]
- Rotary Position Encoding (RoPE, theta=10000) [§A.1]
- **Bidirectional** attention (替代 causal) [§3.2.2]
- Adaptive RMSNorm (接受 timestep t 作为条件) [§3.2.2]

**训练**: Mask-and-predict paradigm [§3.1, §3.2.2]:
- 输入: (P, S^p) 作为 prefix condition + masked semantic token sequence S_t
- 其中 P 是 text token 序列, S^p 是 prompt semantic tokens (in-context learning)
- 以概率 gamma(t) 随机 mask target semantic tokens
- 训练目标: 预测被 mask 的 token (negative log-likelihood) [§3.1]
- Mask schedule: gamma(t) = sin(pi*t / 2T) [§3.1]
- Classifier-free guidance: 以 0.15 概率 drop prompt [§C]

**推理**: Iterative parallel decoding [§3.1]:
- 从全 mask 状态开始, 共 S 步 (默认 50 步)
- 每步预测所有 mask 位置, 根据 confidence score remask 低置信度 token
- Gumbel noise 添加到 confidence 中 (增加多样性) [§4.1]
- Top-k 采样 (k=20), 温度从 1.5 退火到 0 [§4.1]
- CFG scale: 2.5, rescale factor: 0.75 [§4.1]

**为什么 masked generative 比 AR 更适合 TTS**:
1. Bidirectional attention 可以利用全局上下文 → 更好的韵律一致性
2. 并行生成 → 推理步数固定 (25-50 步),不随语音长度线性增长 [§4.2.2]
3. 天然支持 length control: 只需指定 mask 数量即可控制总时长 [Table 1]
4. 更鲁棒: 不存在 AR 的 error accumulation 和 hallucination 问题 [§4.2.2, Appendix J]

#### 3. Semantic-to-Acoustic Model (S2A) [§3.2.3]

**基于 SoundStorm 框架** [§3.2.3]:
- 给定 N 层 RVQ acoustic tokens A^{1:N}, 训练时随机选择一层 j
- Mask 第 j 层的 acoustic tokens, conditioned on prompt acoustic tokens A^p, semantic tokens S, 以及 layer 1 到 j-1 的 acoustic tokens
- Layer sampling schedule: p(j) = 1 - 2j/(N(N+1)) (偏向 coarse layers) [§3.2.3]
- 架构: 16 layers, 1024 dim, 353M params [Table 7]
- 推理: 从 coarse 到 fine 逐层生成, 每层使用 iterative parallel decoding
- 默认步数: [40, 16, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] (共 12 层) [§4.1]

#### 4. Duration Predictor [§A.5]

- 使用 flow matching 预测 total duration (不是 phone-level) [§A.5]
- 架构: 12 layers, 12 heads, hidden 768 [§A.5]
- 训练: in-context learning (随机取 prefix phoneme + duration 作为 prompt)
- CFG guidance, drop prompt 概率 0.15 [§A.5]
- 推理: midpoint ODE solver, 4 steps [§A.5]
- Duration aligner: MAS (Monotonic Alignment Search) between phoneme and w2v-BERT 2.0 semantic features 获取 ground truth [§A.5]

#### 5. Speech Acoustic Codec [§3.2.4]

- RVQ-based codec, 12 layers, codebook size 1024, dimension 8 [Table 9]
- Sample rate: 24K, hop size 480 → 50 Hz token rate [Table 9]
- Encoder: 遵循 DAC/RVQGAN [§3.2.4]
- Decoder: 使用 Vocos (iSTFT-based, 无需 upsampling) [§3.2.4]
- 训练: L_rec + L_adv (MPD + multi-band multi-scale STFT discriminator) + L_feat + L_codebook + L_commit [§A.4]
- 参数量: 170M [Table 9]

### 训练策略

- **数据**: Emilia dataset, 100K hours in-the-wild speech (50K 英文 + 50K 中文) [§4.1]
- **硬件**: 8x NVIDIA A100 80GB GPUs [§4.1]
- **优化器**: AdamW, lr=1e-4, 32K warmup steps, inverse square root schedule [§4.1]
- **Text tokenization**: G2P (Grapheme-to-Phoneme) by default [§4.1, §A.6]
- **多语言扩展**: 额外日语 2.5K h、韩语 7.4K h、德语 6.9K h、法语 8.2K h (仅重训 T2S) [§E]

## 实验

| 指标 | 本文 (MaskGCT) | Baseline (best) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| SIM-O | 0.697 | 0.67 (NatSpeech3) | LibriSpeech test-clean | [Table 2] |
| WER | 2.012 | 1.94 (NatSpeech3) | LibriSpeech test-clean | [Table 2] |
| FSD | 0.746 | 0.762 (VoiceBox) | LibriSpeech test-clean | [Table 2] |
| SMOS | 4.33 | 4.26 (NatSpeech3) | LibriSpeech test-clean | [Table 2] |
| CMOS | +0.13 | +0.16 (NatSpeech3) | LibriSpeech test-clean | [Table 2] |
| SIM-O | 0.728 | 0.643 (CosyVoice) | SeedTTS test-en | [Table 2] |
| WER | 2.466 | 3.248 (XTTS-v2) | SeedTTS test-en | [Table 2] |
| SMOS | 4.24 | 3.80 (VoiceBox) | SeedTTS test-en | [Table 2] |
| SIM-O | 0.777 | 0.750 (CosyVoice) | SeedTTS test-zh | [Table 2] |
| WER | 2.183 | 2.876 (XTTS-v2) | SeedTTS test-zh | [Table 2] |
| SMOS | 4.11 | 3.86 (Ground Truth) | SeedTTS test-zh | [Table 2] |

**MaskGCT vs AR+SoundStorm** [Table 3]:
- MaskGCT 在所有三个测试集上全面优于 AR T2S + SoundStorm S2A 的组合
- SIM-O: +0.015 (LibriSpeech), +0.034 (test-en), +0.027 (test-zh)
- CMOS: +0.12 (LibriSpeech), +0.08 (test-en), +0.37 (test-zh)

**Model scaling** [Table 6]:
- T2S-Large (695M) vs T2S-Base (315M): 性能提升显著但非线性

**Inference steps ablation** [§4.4, Fig 4]:
- SIM 在 ~10 步即可饱和; WER 需要 ~25 步达到最优
- 推荐实际使用 25 步 (balance SIM & WER)

**多语言** [Table 11]: 在日/韩/德/法四种语言上 SIM-O 均超 baseline (XTTS-v2, Emilia-AR/NAR)

## 局限性

1. **Speech content editing 鲁棒性不足**: 作者推测是因为训练时 mask 覆盖全序列而非局部 [§H]
2. **需要外部 duration predictor**: 虽然不需要 phone-level duration,但仍需要预测 total duration
3. **多语言训练未联合 tokenizer**: 多语言扩展时只重训了 T2S,tokenizer 和 S2A 沿用中英数据 [§E]
4. **Inference 步数权衡**: 虽然不随语音长度增长,但 T2S 50 步 + S2A 多层迭代仍有一定开销

## 点评

MaskGCT 的核心贡献在于证明了 **masked generative modeling 可以完全替代自回归模型用于 TTS 的语义阶段**,这是此前 SoundStorm 只在声学阶段做到的。关键 insight 是将 text+prompt 作为 prefix condition 利用 in-context learning,配合 bidirectional attention,使得模型无需显式 alignment 即可学会 text-to-speech 的映射。

**Semantic codec** 的设计 (VQ-VAE on w2v-BERT 2.0) 是被 IndexTTS2 等后续工作直接采用的组件。相比 k-means 离散化,VQ-VAE 的端到端优化保留了更多韵律信息,同时单 codebook 的设计使得 T2S 阶段可以直接建模为一个 flat 序列的 mask-and-predict 问题。

与同期 NaturalSpeech 3 的对比: NS3 也使用了 masked generative model + factored codec,但仍需 text-speech alignment 和 duration prediction;MaskGCT 通过 in-context learning 完全去除了这些依赖。

## 可复用的 idea

1. **VQ-VAE semantic codec**: 用单层 VQ-VAE (而非 k-means) 量化 SSL features → 保留更多信息的 semantic token (已被 IndexTTS2 采用)
2. **Masked generative T2S**: text+prompt 作为不被 mask 的 prefix, bidirectional attention + iterative decoding → 不需要 alignment 的非自回归生成
3. **Adaptive RMSNorm with timestep**: 将 masking timestep t 注入到 normalization 层,使模型感知当前 mask 比例
4. **Flow matching duration predictor**: 使用 flow matching 预测 total duration (non-phonemic granularity) + in-context learning
5. **Layer-wise linear sampling for S2A**: p(j) = 1 - 2j/(N(N+1)) 偏向 coarse layers → 前几层获得更多训练

> [!review] 自动审阅 (2026-06-02)
> **结论:** pass-with-fixes
> **评分:** 理解 8 | 溯源 7 | 严谨 8 | 导航 6 | 安全 7
> **Claim 标注率:** 85% (53/62)
> **问题:** 0 high, 3 medium, 3 low
> - [medium/bad-linking] frontmatter: models 缺少 [[MaskGCT]] 自身; concepts 缺少 [[MaskedGenerativeModeling]]、[[w2v-BERT2.0]]、[[VQ-VAE]]
> - [medium/fact-inference-mixing] 方法 > Semantic Codec & T2S: '为什么 VQ-VAE 优于 k-means' 和 '为什么 masked generative 比 AR 更适合 TTS' 未区分论文原文与 agent 解读
> - [medium/missing-lineage] KB 背景: 空占位符,未定位 MaskGCT 在 NAR TTS 谱系中的位置 (SoundStorm → MaskGCT; vs NaturalSpeech 3)
> **反向更新:** ✅

## 代码级分析

> [!info] 代码来源
> - 仓库: https://github.com/open-mmlab/Amphion (models/tts/maskgct/)
> - commit: 26f6883
> - 分析日期: 2026-06-10

### 架构验证

论文 Fig 1 的 two-stage 架构在代码中完全对应:

- **T2S**: `maskgct_t2s.py:MaskGCT_T2S` — 使用 `DiffLlamaPrefix` (llama_nar.py) 作为 backbone,bidirectional attention (非 causal)
- **S2A**: `maskgct_s2a.py:MaskGCT_S2A` — 使用 `DiffLlama` (llama_nar.py) 作为 backbone,同样 bidirectional

**关键发现**: 论文称使用 "Llama-style Transformer",代码确实继承自 HuggingFace 的 `LlamaModel`,但做了两个关键修改:
1. `_prepare_decoder_attention_mask` 被重写为 **non-causal mask** (llama_nar.py:256-293),即标准 Llama 的 causal mask 被替换为 bidirectional mask
2. `LlamaDecoderLayer` 被替换为 `LlamaNARDecoderLayer`,将 `RMSNorm` 替换为 `LlamaAdaptiveRMSNorm`(接受 timestep 作为条件)

### 论文未写的实现细节

1. **Mask 概率下限 0.2** (`maskgct_t2s.py:118-120`): `mask_prob = torch.where(mask_prob < 0.2, 0.2, mask_prob)`。论文公式 gamma(t)=sin(pi*t/2T) 理论上可以很小,但代码强制最小 mask 20%。这防止了过早 "完成" 导致的训练不稳定。

2. **Prompt 长度随机化** (`maskgct_t2s.py:133-135`): 训练时 prompt 长度从 `U[min(T/4, 5), T*0.4]` 均匀采样,S2A 从 `U[min(T/4, 5), T/2]`。CFG drop 概率: T2S 0.2, S2A 0.15。

3. **S2A 层采样的 linear schedule** (`maskgct_s2a.py:166-179`): 论文说 `p(j) = 1 - 2j/(N(N+1))`(偏向 coarse),代码实现为 `weights = [num_quantizer - i for i in range(num_quantizer)]` 然后归一化,即 `p(j) = (N-j) / sum`。这与论文公式等价(都是线性递减)。

4. **S2A 的 mask token 填充** (`maskgct_s2a.py:239-245`): 对于 mask_layer 以上的层,非 prompt 部分全部用 mask_token 填充(而非零)。这意味着 S2A 在推理时也知道"上层还没有生成",与论文描述一致但代码更清晰。

5. **CFG rescale 策略** (`maskgct_t2s.py:310-312`): `rescale_embeds = embeds * pos_emb_std / embeds.std()`,即用条件输出的标准差重新归一化 CFG 输出。rescale_cfg 控制混合比例。论文 Appendix C 简要提及,但这个 std-based rescale 对避免 CFG 过强导致的失真至关重要。

6. **Adaptive RMSNorm 的零初始化** (`llama_nar.py:38-39`): `nn.init.zeros_(self.to_weight.weight)` + `nn.init.ones_(self.to_weight.bias)`,即初始化时 adaptive norm 退化为标准 RMSNorm (权重=1)。这保证了训练初期的稳定性。

### 训练 pipeline 拆解

```
T2S 训练:
  Phone IDs → phone_emb [B, T_phone, 1536]
  Semantic tokens x0 [B, T_sem] → forward_diffusion:
    1. sample t ~ U(0,1), mask_prob = sin(pi*t/2)
    2. 随机选 prompt_len → 前 prompt_len 帧不 mask
    3. 以 mask_prob 概率 Bernoulli 采样 mask
    4. xt = mask * mask_token + (1-mask) * cond_emb(x0)
  xt + phone_emb → DiffLlamaPrefix(bidirectional) → embeds
  to_logit(embeds) → logits [B, T, 8192]
  CE loss on masked positions only

S2A 训练:
  Acoustic tokens x0 [B, T, 12] + semantic cond [B, T]
  → forward_diffusion:
    1. sample mask_layer (linear schedule)
    2. layers < mask_layer: 用真实 token embedding
    3. layer == mask_layer: Bernoulli mask
    4. layers > mask_layer: 非 prompt 全 mask
  xt + layer_emb + cond → DiffLlama(bidirectional) → embeds
  to_logits[mask_layer](embeds) → logits [B, T, 1024]
  CE loss on masked positions of mask_layer only
```

### 推理 pipeline 拆解

```
T2S 推理 (reverse_diffusion, maskgct_t2s.py:226-363):
  1. phone_id → phone_emb
  2. prompt semantic tokens → cond_emb → cur_prompt
  3. seq 全初始化为 0, mask 全 True
  4. for i in range(50 steps):
     a. cur = cum + mask * mask_token + ~mask * cond_emb(seq)
     b. [cur_prompt, cur] → DiffLlamaPrefix → embeds
     c. (optional) CFG: mask_embeds = DiffLlamaPrefix(cur_only) → rescale
     d. top-k(0.98) → gumbel_sample(temp * t_anneal) → sampled_ids
     e. seq[mask] = sampled_ids[mask]
     f. confidence = softmax(logits).gather(sampled_ids)
     g. + gumbel_noise * choice_temp → 选最低 confidence 位置重新 mask
     h. mask_num = sin(pi*t_next/2) * seq_len
  5. 输出: seq [B, T_target] (semantic tokens)

S2A 推理 (reverse_diffusion, maskgct_s2a.py:317-469):
  同 T2S 但逐层生成:
  for mask_layer in 0..11:
    steps = n_timesteps[mask_layer]  # 默认 [40,16,1,1,...,1]
    相同的 mask-predict-remask 循环
    完成后: cum += token_emb(seq)
  输出: xt [B, T, 12] (12层 acoustic tokens)
```

### 关键超参数表

| 参数 | 论文值 | 代码实际值 | 备注 |
|------|--------|-----------|------|
| T2S hidden_size | 1536 | 1536 (cfg) | maskgct_t2s.py |
| T2S num_layers | 16 | 16 | maskgct_t2s.py |
| T2S num_heads | 16 | 16 | maskgct_t2s.py |
| T2S params | 695M | ~695M | 论文 Table 7 |
| S2A hidden_size | 1024 | 1024 | maskgct_s2a.py |
| S2A num_layers | 16 | 16 | maskgct_s2a.py |
| S2A num_quantizer | 12 | 12 | maskgct_s2a.py |
| Semantic codebook | 8192 | cond_codebook_size=8192 | maskgct_t2s.py |
| Acoustic codebook | 1024 | codebook_size=1024 | maskgct_s2a.py |
| T2S CFG drop | 0.2 | cfg_scale=0.2 | maskgct_t2s.py:41 |
| S2A CFG drop | 0.15 | cfg_scale=0.15 | maskgct_s2a.py:43 |
| Mask schedule | sin(pi*t/2) | mask_prob = sin(t*pi/2) | 两处一致 |
| Min mask prob | 未提及 | 0.2 | maskgct_t2s.py:118 |
| S2A layer schedule | linear | weights=[N-i] | maskgct_s2a.py:167 |
| Inference T2S steps | 50 | n_timesteps=40 (默认) | 代码默认 40 非 50 |
| Inference S2A steps | [40,16,1*10] | [10,4,4,4,4,4,4,4] (默认) | 代码默认更少 |
| Phone emb padding_idx | 未提及 | 1023 | maskgct_t2s.py:98 |
| RoPE theta | 10000 | 10000 (Llama default) | LlamaConfig |
| Intermediate size | 未提及 | hidden*4 | llama_nar.py:219 |

### 复现 checklist (基于代码)

- [x] 环境依赖: transformers, einops, torch; 需要 Amphion 完整环境
- [ ] 数据准备: 需要 Emilia 100K h (受限获取) + w2v-BERT 2.0 特征提取
- [x] 预训练模型依赖: 权重通过 HuggingFace `amphion/MaskGCT` 下载; semantic codec、acoustic codec 均有 checkpoint
- [ ] 训练命令: Amphion 框架内训练,需配置 JSON + 多组件串联
- [x] 推理命令: `maskgct_demo.ipynb` 提供完整推理 demo
- [x] 已知坑: (1) T2S 默认步数代码是 40 而非论文的 50; (2) LlamaNARDecoderLayer 有重复定义 (llama_nar.py:56 和 129,完全相同的 __init__ 和 forward); (3) 需要 w2v-BERT 2.0 做 semantic 特征提取,较大模型

### 代码质量与可复现性评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 工程质量 | 3/5 | 代码可读但有冗余 (LlamaNARDecoderLayer 的重复定义),缺少 type hints |
| 文档完善度 | 3/5 | 有 Jupyter demo 和 README,但训练文档不足 |
| 社区活跃度 | 4/5 | Amphion 活跃维护 (7.5K+ stars),持续更新 |
| 复现难度 | 3/5 | 推理可复现,训练需要大规模数据 + 多组件协调 |
