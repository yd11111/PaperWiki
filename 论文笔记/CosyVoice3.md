---
type: paper
tier: deep
title: "CosyVoice 3: Towards In-the-wild Speech Generation via Scaling-up and Post-training"
arxiv_id: "2505.17589"
source: "Sources/CosyVoice3.pdf"
authors: [Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, Keyu An, Guanrou Yang, Yabin Li, Yanni Chen, Zhifu Gao, Qian Chen, Yue Gu, Mengzhe Chen, Yafeng Chen, Shiliang Zhang, Wen Wang, Jieping Ye]
year: 2025
venue: "arXiv preprint"
tags: [TTS, zero-shot, multilingual, scaling, post-training, speech-tokenizer, reinforcement-learning]
concepts: ["[[FiniteScalarQuantization]]", "[[ConditionalFlowMatching]]", "[[DifferentiableRewardOptimization]]", "[[SpeechTokenizer]]", "[[Gumbel-Softmax]]"]
models: ["[[CosyVoice3]]", "[[CosyVoice2]]", "[[MinMo]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]", "[[Cross-lingualVoiceCloning]]", "[[InstructedSpeechGeneration]]"]
datasets: ["[[SEED-TTS-Eval]]", "[[CV3-Eval]]"]
kb_context_sources: 0
status: reviewed
created: 2026-06-01
updated: 2026-06-01
---

## KB 背景

> [!info] KB 背景 (KB 检索未启用 — P1 阶段)
> P3 阶段启用 KB 检索后,此节将自动填充知识库背景。

## 速查

> [!summary] 速查
> - **一句话**: 监督式多任务 speech tokenizer + DiffRO token-level RL + 100 万小时数据 scaling,构建 9 语种 in-the-wild zero-shot TTS
> - **路线**: Text → LLM (1.5B, AR) → discrete speech tokens (25Hz) → CFM (DiT 300M) → Mel → Vocoder → Waveform
> - **指标**: CER 0.71% / WER 1.45% (SEED-TTS-Eval test-zh/en), SS 0.775; 相对 CosyVoice 2 内容一致性提升 44%-51%; 跨语言 WER zh→en 5.09 vs 13.5 (CV3-Eval)
> - **可借鉴**: DiffRO 绕过 CFM/vocoder 在 token 层用 Gumbel-Softmax 做可微 reward 优化; 在预训练 backbone (MinMo) 中间层插入 FSQ 的监督式 tokenizer 设计模式
> - **局限**: 1.5B 在 test-hard 上反而逊于 0.5B (scaling 未充分验证); 模型和 100 万小时数据均未开源; 缺少与 VALL-E 2 等强 baseline 直接对比

## 核心问题

CosyVoice 2 虽然在中英文广播场景下表现良好,但在以下方面存在明显局限:
1. **语言覆盖不足** — 仅支持中英文,无法应对 in-the-wild 多语言场景
2. **领域/风格多样性差** — 数据域单一,文本格式有限
3. **数据与模型规模未充分探索** — 未验证 TTS 领域的 scaling law
4. **缺少有效的 post-training 策略** — 无法在预训练后进一步提升内容一致性

CosyVoice 3 试图通过四个维度解决:新 speech tokenizer、DiffRO post-training、数据规模从 1 万小时扩至 100 万小时、模型参数从 0.5B 扩至 1.5B [§1]。

## 方法: 它怎么 work

### 整体架构

CosyVoice 3 沿用 CosyVoice 2 的 coarse-to-fine 两阶段架构 [§2]:
1. **LM 阶段**: 自回归 LLM 将文本 token 映射为离散 speech token(语义层)
2. **CFM 阶段**: Conditional Flow Matching 模型将 speech token 转换为 Mel spectrogram(声学层),再由 vocoder 合成波形

关键改进集中在 speech tokenizer 设计、post-training 策略、以及 scaling。

### 关键设计选择

#### 1. 基于 MinMo 的监督式多任务 Speech Tokenizer [§2.1]

**为什么有效**: CosyVoice 2 使用 SenseVoice-Large ASR 编码器 + FSQ;CosyVoice 3 改用 MinMo(一个在 140 万小时语音上预训练的多模态 LLM)作为 backbone。MinMo 本身已在对话、多语种 ASR、情感识别等任务上达到 SOTA,因此其 intermediate representations 天然携带丰富的副语言信息(情感、语种、说话人特征)。

具体做法:
- 在 MinMo 的 Voice Encoder_1(12 层 Transformer + RoPE)中间插入 FSQ 模块 [§2.1]
- FSQ 将中间表征投影到 D 维低秩空间,每维量化到 [-K, K],再投影回原始维度 [Eq.1]
- Speech token 通过 (2K+1) 进制索引计算得到 [Eq.2]
- 量化后的表征继续通过 Voice Encoder_2 和 MinMo LLM 进行多任务监督训练(ASR 365K h、LID 85K h、SER 48K h、AED 21K h、SA 11K h)[Table 3]

**为什么多任务监督比自监督更好**: 自监督 tokenizer(如 HuBERT、W2v-BERT 2.0)学到的 token 混杂语义和声学信息;监督式 tokenizer 只保留语义,声学干扰被过滤,使得下游 CFM 可以更好地从 speaker prompt 中提取音色特征,而非从 token 中读取(已被 Table 12 实验验证:在 3000 小时数据上,监督 tokenizer 在 speaker similarity 和 content consistency 上全面优于 W2v-BERT 2.0 和 SoundStream)[Table 12]。

Token rate: 25 Hz(每秒 25 个 speech token)[§2.1]。

#### 2. Differentiable Reward Optimization (DiffRO) [§2.2]

**核心问题**: 传统 RL 用于 TTS 时,需要 CFM/vocoder 把 speech token 转成音频后才能计算 reward,计算量大且生成的音频高度相似导致正负样本难以区分。

**DiffRO 的解决方案**: 直接在 token 层面优化,绕过音频生成。
1. 训练一个 Token2Text model(类似反向 ASR):输入 speech token 序列,输出文本的后验概率
2. Reward = log P_ASR(正确文本 | speech tokens) [Eq.4] — 衡量 token 序列是否清晰可被识别
3. 使用 Gumbel-Softmax 对 LLM 输出的 token logits 采样,使梯度可通过 [Eq.3]
4. 加入 KL 散度约束防止偏离参考模型 [Eq.5-6] — 但关键差异: KL 计算在 **token-level logits** 上(而非 sequence-level posterior),粒度更细

**Multi-task Reward (MTR)**: 除 ASR reward 外,还可加入 SER、AED、MOS 等下游任务的 reward [Eq.7],通过指令控制语音属性。

**为什么有效**: DiffRO 将 RL 的信号直接反馈到 LLM 的 token 选择上,无需穿过 CFM/vocoder 的计算图。相当于让 LLM "知道"每个 token 选择对最终可懂度的影响。实验显示 DiffRO 带来 20%~50% 的相对 WER 改进 [§5.4],在低资源语言和跨语言场景中尤为显著(韩语相对改进 68.7%)[§5.4]。

#### 3. Pronunciation Inpainting [§2.3]

解决多音字问题: 将中文单音字替换为拼音、英文单音词替换为 CMU 音素,混合输入使模型可控发音。RepMono + MixPhn 方案在中英文分别达到 100% 和 100% 纠正率 [Table 13]。

#### 4. Self-training for Text Normalization [§2.4]

用 LLM (Qwen-Max) 做正向/逆向 TN,构造 raw text-audio 配对,使系统可以直接合成含数字/特殊符号的原始文本。

#### 5. Instructed Speech Generation [§2.5]

instruction-following 数据从 1500 小时扩至 5000 小时,风格类型从有限扩至 100+ 种(情感、语速、音色、方言、角色扮演等)[Table 1]。支持自然语言指令和细粒度标记(如 `[laughter]`、`[breath]`、`<strong>XXX</strong>`)。

### 训练策略

训练流程分四阶段 [Fig.2b]:
1. **Large-scale Pretraining**: 用全部 100 万小时数据,从 text-based LLM 初始化 → 零样本 LM + CFM
2. **DiffRO Post-training**: 在筛选数据上用 DiffRO 优化 → 提升内容一致性
3. **Continual Pretraining**: 在情感/指令/多语言数据上持续预训练,Text2Token LM 不变 → 转移能力到 SFT 模型
4. **Speaker Fine-tune**: 在多说话人数据上微调,随机 mask speaker/style prompt 防止灾难性遗忘 [§2.6.2]

模型规模:
- LM: 0.5B → 1.5B 参数 [§4.2]
- CFM: 采用 DiT 架构,从 100M 扩至 300M 参数,去掉了 CosyVoice 2 的 text encoder 和 length regularization module [§4.2]

## 实验

| 指标 | 本文 (CosyVoice 3-1.5B_RL) | Baseline (CosyVoice 2) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CER (%) test-zh | 0.71 | 1.45 | SEED-TTS-Eval | [Table 4] |
| WER (%) test-en | 1.45 | 2.57 | SEED-TTS-Eval | [Table 4] |
| CER (%) test-hard | 5.66 | 6.83 | SEED-TTS-Eval | [Table 4] |
| SS (WavLM) test-zh | 0.775 (0.836) | 0.748 (0.806) | SEED-TTS-Eval | [Table 4] |
| MOS (中文) | 4.48 | 4.25 | 主观评估 | [Fig.4] |
| MOS (英文) | 4.43 | 4.34 | 主观评估 | [Fig.4] |
| MOS (平均) | 4.45 | 4.36 | 主观评估 | [Fig.4] |
| Cross-lingual WER zh→en | 5.09 | 13.5 | CV3-Eval | [Table 7] |
| Style SIM (Expresso) | 68.25 | 60.98 | Expresso | [Table 14] |

**内容一致性**: CosyVoice 3 相对 CosyVoice 2 在 test-zh 上提升 44%,test-en 上提升 51% [§5.1]。DiffRO 在所有条件下带来 12%~35% 的额外相对提升 [§5.1]。

**多语言覆盖**: CosyVoice 3 是唯一支持全部 9 种语言的系统(zh, en, ja, ko, de, es, fr, it, ru),其他开源模型仅覆盖部分 [Table 5]。

**跨语言克隆**: CosyVoice 3-1.5B + DiffRO 在 4 个方向的 WER 均优于所有 baseline;CosyVoice 2 因日文字符转换问题在 ja 方向表现很差(48.1%),CosyVoice 3 显著改善(3.05%) [Table 7]。

**情感克隆**: DiffRO-EMO 变体在 text-related 和 text-unrelated 子集上均达到最高情感准确率(happy 0.98/0.98, sad 0.68/0.50, angry 0.84/0.68) [Table 9]。

**Speech Tokenizer 对比**: 在 170K 小时数据上,CosyVoice 3.0 tokenizer 相比 CosyVoice 2.0 tokenizer 在 test-en CER 上从 2.57 降至 2.46,SS 从 0.736 升至 0.747 [Table 12]。

## 局限性

1. **音色不可控** — 无法通过文本指令编辑音色(timbre),对角色扮演类应用有限 [§7]
2. **歌唱生成能力弱** — 当前 tokenizer 和 LM 未纳入歌唱数据 [§7]
3. **大模型未充分发挥** — 1.5B 模型在 test-hard 上反而略逊于 0.5B(5.66 vs 5.09),因可用于 post-training 的高质量数据不足 [§5.1]
4. **Speaker similarity 的 "hacking" 问题** — DiffRO 优化 WER 时可能略降 speaker similarity,需引入 speaker 相似度 reward 来平衡 [§5.4]
5. **hard samples 上 DiffRO 效果有限** — 稀有词、绕口令、重复词等对 reward model 仍构成挑战 [§5.4]

## 点评

**优势**:
- DiffRO 是一个优雅的工程创新: 绕开 CFM/vocoder 在 token 层直接优化,大幅降低 RL 在 TTS 中的计算成本,且适用于所有基于离散 token 的 TTS 系统
- 监督式多任务 tokenizer 的设计思路清晰: 让 FSQ 插入一个已经"见过"丰富副语言信息的 backbone,比从头训练语义更丰富
- 数据工程扎实: multilingual pipeline 的 6 步处理(VAD → 降噪 → ASR → 标点 → 音量 → 过滤)是工业级实践的很好参考
- CV3-Eval benchmark 填补了多语言 + in-the-wild TTS 评估的空白

**不足**:
- 1.5B 模型表现不稳定(部分指标逊于 0.5B),说明 scaling 尚未找到最优配方
- 缺少与 VALL-E 2、Voicebox 等强 baseline 的直接对比(可能因开源模型不可得)
- DiffRO 的 multi-task reward 部分(MTR)实验覆盖不够充分,仅展示了 EMO 一个变体

## 可复用的 idea

1. **Token-level RL**: 在离散 token 空间用 Gumbel-Softmax + token-level KL 做可微 reward 优化,适用于任何 LM → discrete token → downstream renderer 的 pipeline
2. **监督式 tokenizer 设计模式**: 在大型预训练 speech model 的中间层插入量化模块,通过多任务监督压入目标信息(语义),同时排除不需要的信息(声学细节)
3. **Pronunciation Inpainting**: 用 mixed text+phoneme 输入解决多音字,比纯 G2P 或纯 BPE 更灵活且可控
4. **跨语言能力迁移**: 通过 continual pretraining + 辅助多语种数据,将单语说话人变为多语说话人(polyglot training)
5. **Multilingual data pipeline**: 6 步处理流程(特别是 cross-validation ASR + MFA 标点对齐)可作为标准参考

> [!review] 自动审阅 (2026-06-02)
> **结论:** pass-with-fixes
> **评分:** 理解 9 | 溯源 8 | 严谨 8 | 导航 7 | 安全 8
> **Claim 标注率:** 88% (44/50)
> **问题:** 0 high, 2 medium, 3 low
> - [medium/traceability-gap] 速查卡片 > 指标行: 5 个关键数字 (CER, WER, SS, 相对提升, 跨语言 WER) 缺少 [Table N] 标注
> - [medium/fact-inference-mixing] 方法 > MinMo 监督式 Tokenizer: MinMo SOTA 声明无本文出处,因果链未区分论文原文与 agent 解读
> **反向更新:** ✅

## 代码级分析

> [!info] 代码来源
> - 仓库: https://github.com/FunAudioLLM/CosyVoice
> - commit: 074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc
> - 分析日期: 2026-06-10
> - 备注: CosyVoice3 对应 `CosyVoice3LM` 类 + `CausalMaskedDiffWithDiT` 类,配置文件 `cosyvoice3.yaml`

### 架构验证

CosyVoice3 vs CosyVoice2 的代码级差异:

| 组件 | CosyVoice2 | CosyVoice3 | 代码位置 |
|------|-----------|-----------|---------|
| LLM 类 | `Qwen2LM` | `CosyVoice3LM` (继承 Qwen2LM) | `llm.py:664` |
| CFM 类 | `CausalMaskedDiffWithXvec` | `CausalMaskedDiffWithDiT` | `flow.py:284` |
| Tokenizer | `speech_tokenizer_v2` | `speech_tokenizer_v3` | `cosyvoice.py:204` |
| Text tokenizer | `CosyVoice2Tokenizer` | `CosyVoice3Tokenizer` | `tokenizer.py:274` |
| Special tokens | sos/task_id 从 llm_embedding 取 | sos/task_id/fill 从 speech_embedding 取 | `llm.py:681-684` |

**关键架构变化**:

1. **LLM 初始化方式改变** (`llm.py:664-706`): `CosyVoice3LM` 的 `sos`/`eos_token`/`task_id`/`fill_token` 全部使用 speech_embedding 的后续 index (speech_token_size+0/1/2/3),不再使用独立的 `llm_embedding`。`speech_embedding` 大小为 `speech_token_size + 200`,`llm_decoder` 输出也是 `speech_token_size + 200`。这意味着 CosyVoice3 预留了 200 个特殊 token 位。

2. **CFM 使用 DiT 替代 UNet** (`flow.py:284-414`): `CausalMaskedDiffWithDiT` 没有 `encoder`(CosyVoice2 有一个 Conformer encoder),取而代之的是一个 `pre_lookahead_layer` 将 speech token embedding 直接投影,然后通过 `repeat_interleave(token_mel_ratio)` 上采样到 mel 帧率。DiT 作为 CFM 的 estimator。

3. **指令 token 支持** (`llm.py:308-315,387-393`): CosyVoice3 的 forward 方法额外处理 `instruct_token`,通过 Qwen2 的 embed_tokens 嵌入后插入到 sos 和 text_token 之间。推理时,hardcode 检查 `151646` (`<|endofprompt|>` token) 是否在输入中 (`llm.py:479`)。

4. **Pronunciation inpainting** (`tokenizer.py:288-307`): `CosyVoice3Tokenizer` 注册了大量 CMU 音素 token (如 `[AA0]`, `[AE1]`) 和中文拼音 token (如 `[à]`, `[ái]`) 作为 special tokens,这是论文中 Pronunciation Inpainting 的代码实现。

### 论文未写的实现细节

1. **静音 token 过滤列表** (`cosyvoice/cli/model.py:423`): CosyVoice3 定义了 11 个 silent_tokens: `[1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]`,这些 FSQ token 对应静音和呼吸音,连续超过 5 个时跳过,避免生成过长停顿。

2. **DiT CFM 无 length_regulator** (`flow.py:345-348`): CosyVoice3 的 CFM 不需要 CosyVoice v1 的 length_regulator,直接用 `repeat_interleave(token_mel_ratio=2)` 将 token 序列上采样到 mel 帧级别。这大幅简化了 token-to-mel 的对齐。

3. **Pre-lookahead 层** (`flow.py:313,346`): `pre_lookahead_layer` 是一个独立的网络层(非标准 encoder),用于在流式推理时将当前 token 与未来 `pre_lookahead_len` 个 token 做 local attention,代替 CosyVoice2 中完整 encoder 的 chunk attention。

4. **Vocoder 不同的缓存策略** (`model.py:425-449`): CosyVoice3 的 `token2wav` 不再使用 fade_in_out 和独立的 mel_overlap,而是累积所有 mel 到 `hift_cache_dict[uuid]['mel']`,vocoder 每次对完整 mel 做推理,通过 `speech_offset` 跟踪已输出的位置。这意味着 vocoder 每次推理的计算量会随时间增长,但避免了拼接伪影。

5. **`<|endofprompt|>` hardcode** (`llm.py:479,587-589`): 推理时强制要求输入中包含 token ID 151646,这是 Qwen2 的 `<|endofprompt|>` special token。bistream 模式下,会在这个 token 处分割 prompt/指令部分和正文部分。

6. **CosyVoice3 不支持 JIT** (`cosyvoice.py:191`): `CosyVoice3.__init__` 没有 `load_jit` 参数,因为 DiT 架构不适合 TorchScript 导出。

### 训练 pipeline 拆解

```
原始音频
  → speech_tokenizer_v3.onnx (MinMo FSQ) → speech_token (codebook=6561)
  → mel_spectrogram → speech_feat (80-dim)
  → campplus.onnx → embedding (192-dim)
  → Qwen2 BPE tokenizer + 音素 special tokens → text_token + instruct_token

训练:
  text_token → Qwen2.embed_tokens → text_emb
  instruct_token → Qwen2.embed_tokens → instruct_emb (仅 CosyVoice3)
  speech_token → speech_embedding → speech_emb
  sos/task_id/fill → speech_embedding[special_idx]
  
  50% unistream: [sos, instruct_emb, text_emb, task_id, speech_emb]
  50% bistream: [sos, instruct_emb, text_5, speech_15, ..., task_id, rest_speech]
  
  LLM: Qwen2ForCausalLM → hidden_states[-1] → llm_decoder (6561+200)
  → LabelSmoothingLoss
  
  CFM (DiT):
  speech_token → input_embedding → pre_lookahead_layer → repeat(2) → h
  条件: h + spk_embedding + masked_mel
  50% streaming / 50% non-streaming
  → CausalConditionalCFM.compute_loss
```

### 推理 pipeline 拆解

```
输入文本 (可含拼音/音素标记) + prompt 音频
  → prompt 音频 → speech_tokenizer_v3 → prompt_speech_token
  → prompt 音频 → mel_extractor → prompt_feat
  → prompt 音频 → campplus → flow_embedding
  → 文本 + <|endofprompt|> → Qwen2 BPE + 音素 tokens → text
  
  Stage 1 (LLM, 异步线程):
    CosyVoice3LM: sos/task_id 从 speech_embedding 取
    指令通过 <|endofprompt|> 分隔
    AR decode, top-k=25
    静音 token 过滤 (max 5 连续)
    
  Stage 2 (DiT CFM, 主线程):
    token → input_embedding → pre_lookahead_layer → repeat(2) → h
    固定噪声 + 10 步 Euler + cosine scheduler + CFG(0.7)
    → mel (streaming: 逐 chunk, 累积 vocoder)
    
  Vocoder: cumulative mel → HiFi-GAN → waveform → offset tracking
```

### 关键超参数表

| 参数 | 论文值 | 代码实际值 | 备注 |
|------|--------|-----------|------|
| LLM 参数量 | 0.5B / 1.5B | Qwen2ForCausalLM | 取决于加载的模型 |
| CFM 架构 | DiT 300M | DiT (无参数细节在代码中) | 配置在 YAML |
| Speech token vocab | 6561 | 6561 | 同 CosyVoice2 |
| Special token 预留 | 未明确 | 200 个 | `speech_token_size + 200` |
| Token-mel ratio | 未明确 | 2 | `token_mel_ratio=2` |
| Silent tokens | 未提及 | 11 个 FSQ 码 | 硬编码在 model.py |
| 音素 tokens | CMU + 拼音 | ~250 个 special tokens | `CosyVoice3Tokenizer` |
| `<endofprompt>` | 未明确 | token ID 151646 | Qwen2 special token |

### 三版本共用/独有模块总结

| 模块 | CosyVoice v1 | CosyVoice2 | CosyVoice3 |
|------|-------------|-----------|-----------|
| LLM 类 | `TransformerLM` | `Qwen2LM` | `CosyVoice3LM` |
| Text encoder | Conformer | Qwen2 embed_tokens | Qwen2 embed_tokens |
| Speaker emb (LLM) | 有 (x-vector) | 无 | 无 |
| Bistream | 无 | 有 | 有 (继承) |
| CFM backbone | UNet (`ConditionalDecoder`) | Causal UNet (`CausalConditionalCFM`) | DiT |
| Length regulator | 有 | encoder 内置 | repeat_interleave |
| Tokenizer ONNX | v1 (VQ 4096) | v2 (FSQ 6561) | v3 (MinMo FSQ 6561) |
| Text tokenizer | whisper-based | Qwen2 + specials | Qwen2 + 音素 specials |
| DPO 训练 | 无 | 有 | 有 (继承) |
| 指令 token | 无 | 无 | 有 |
| vLLM 支持 | 无 | 有 | 有 (继承) |
| TRT 支持 | 有 | 有 | 有 (DiT fp16 有警告) |

### 复现 checklist (基于代码)

- [ ] 环境依赖: 同 CosyVoice2 + MinMo 相关依赖
- [ ] 数据准备: speech_tokenizer_v3 (MinMo FSQ), mel, speaker embedding
- [ ] 预训练模型依赖: Qwen2.5 (0.5B/1.5B), speech_tokenizer_v3.onnx, campplus.onnx
- [ ] 训练命令: `cosyvoice/bin/train.py` + cosyvoice3.yaml
- [ ] 推理命令: `CosyVoice3(model_dir).inference_zero_shot(text, prompt_text, prompt_wav)`
- [ ] 已知坑: (1) MinMo tokenizer 训练完全不在仓库中; (2) DiT TRT fp16 有性能问题; (3) 音素 token 的使用需要特殊的 text 预处理

### 代码质量与可复现性评估

- **工程质量**: 4/5 - 继承良好的模块化,但 CosyVoice3LM 中有多处 `if self.__class__.__name__ == 'CosyVoice3LM'` 的类型检查,显示代码是增量开发而非重构
- **文档完善度**: 2/5 - CosyVoice3 的文档最少,音素 token 使用方式和指令格式缺乏说明
- **社区活跃度**: 5/5 - 最新版本,社区关注度最高
- **复现难度**: 4/5 - MinMo tokenizer 完全黑盒,1.5B LLM 训练需要大量计算资源和 100 万小时数据
