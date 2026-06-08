---
type: paper
tier: deep
title: "DrVoice: Parallel Speech-Text Voice Conversation Model via Dual-Resolution Speech Representations"
arxiv_id: "2506.09349"
source: "Sources/DrVoice.pdf"
authors: [Chao-Hong Tan, Qian Chen, Wen Wang, Chong Deng, Qinglin Zhang, Luyao Cheng, Hai Yu, Xin Zhang, Xiang Lv, Tianyu Zhao, Chong Zhang, Yukun Ma, Yafeng Chen, Hui Wang, Jiaqing Liu, Xiangang Li, Jieping Ye]
year: 2025
venue: "arXiv preprint"
tags: [speech-LM, parallel-speech-text, dual-resolution, speech-tokenizer, joint-modeling, grouping, SRH]
concepts: ["[[SemanticvsAcousticTokens]]", "[[SpeechTokenizer]]", "[[SpeechLanguageModel]]", "[[Speech-TextAlignment]]", "[[TokenRateandBitrateTrade-offs]]", "[[ModalityAdaptationforSpeechLLM]]", "[[ConditionalFlowMatching]]"]
models: ["[[MinMo]]", "[[CosyVoice]]", "[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认实体页: [[SemanticvsAcousticTokens]], [[SpeechTokenizer]], [[SpeechLanguageModel]]; 3 个待确认: [[Speech-LLMIntegrationTaxonomy]], [[TokenRateandBitrateTrade-offs]], [[MinMo]])
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: DrVoice 属于 Speech-LLM Integration Taxonomy 中 Audio-token-based Integration 的 parallel speech-text 子类(与 Moshi、Kimi-Audio 同类),区别于 text-driven 方案(Qwen2.5-Omni)和 interleaved 方案(GLM-4-Voice、Baichuan-Omni-1.5)。其核心创新在 token rate 维度: 通过 grouping 将 LLM 输入帧率从主流 12.5Hz 降至 5Hz,直接触碰 [[TokenRateandBitrateTrade-offs]] 中"序列长度 vs 建模质量"的核心 trade-off。

**已有认知**: (1) SpeechTokenizer 页记录了 CosyVoice 系列的 S3 supervised semantic tokenizer 路线——DrVoice 正是采用 S3Tokenizer + CosyVoice detokenizer,继承同团队技术栈。(2) SemanticvsAcousticTokens 页确认 semantic tokens 与文本对齐好但缺声学细节,需要 post-processing(Flow Matching/vocoder)恢复——DrVoice 的 SRH 可视为在 token 生成层面的替代方案,通过自回归 refinement 而非后处理来恢复 ungrouped 细节。(3) SpeechLanguageModel 页记录了 parallel 与 interleaved 两种 joint modeling 范式——DrVoice 选择 parallel 路线并通过 dual-resolution 减轻其计算成本劣势。

**创新判断**: DrVoice 的核心新贡献是 DRSR(grouping + SRH ungrouping)机制。KB 中 TokenRateandBitrateTrade-offs 记录了"低 token rate 对 LM 有巨大优势"但通常牺牲质量的 trade-off——DrVoice 声称通过 SRH 在不牺牲质量的前提下实现了 5Hz 帧率,这在现有 KB 知识中是未见的极低帧率方案。

> 检索命中: [[SemanticvsAcousticTokens]]✓, [[SpeechTokenizer]]✓, [[SpeechLanguageModel]]✓ | 过滤: [[Speech-LLMIntegrationTaxonomy]](pending-review), [[TokenRateandBitrateTrade-offs]](pending-review), [[MinMo]](pending-review) | 未命中但可能相关: [[Full-duplexSpokenDialogue]]

## 速查

> [!summary] 速查
> - **一句话**: 通过 dual-resolution speech representations (DRSR) 将 LLM 输入帧率降至 5Hz,结合 Speech Refined Head (SRH) 自回归恢复 25Hz 输出,在 parallel speech-text 框架下以最低计算成本达到 7B 级 SOTA
> - **路线**: User speech → Whisper-v3 encoder + Adapter(理解侧) / S3Tokenizer 25Hz → Grouping k=5 → 5Hz(生成侧) → Shared LLM Layer → Text Head(文本) + SRH ungrouping(语音) → CosyVoice Flow Matching + HiFi-GAN
> - **指标**: OpenAudioBench 72.04, VoiceBench 80.17, UltraEval-Audio 56.66, BBA 74.0 (均 SOTA among ~7B); UTMOS 4.29; ASR-WER 8.36; ~50% GPU hours 节省
> - **可借鉴**: (1) Grouping+SRH 双分辨率设计可迁移到任何 parallel speech-text 系统; (2) Core-Cocktail 两阶段训练(高 LR 全参 → merge 回 base → 低 LR 精调)是通用的 LLM 知识保留策略; (3) CoM-Mixing 的 7 种交互模式系统 prompt 设计
> - **局限**: 无全双工; ASR-WER 8.36 落后 Qwen2.5-Omni (3.48); 仅英文; 未开源完整代码

## 核心问题

1. **频率失配**: Speech tokens (~25Hz) 与 text tokens (~3Hz) 的时间分辨率差异导致 LLM 序列过长、语义信息被稀释 [§1]
2. **Text-driven 的单向性**: 文本生成完成后再合成语音,缺乏 speech→text 的反馈回路,无法感知语音生成状态 [§1]
3. **Joint model 的文本退化**: Speech token 的加入干扰 LLM 原有文本能力 [§1]
4. **计算效率**: 12.5Hz 输入帧率在 joint model 中带来显著计算成本 [§1]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

DrVoice 是 parallel speech-text 架构 [Fig 1]:

**输入侧(用户端)**:
- Whisper-Large-v3 encoder 提取连续语音表征 → Adapter 降采样对齐 LLM hidden dim [§3.1]
- 文本输入直接进 LLM

**生成侧(助手端)**:
- S3Tokenizer 将语音波形转为 25Hz semantic token 序列 [§3.1]
- **Grouping**: 每 k=5 个 speech tokens 拼接 → Linear 投影 → 5Hz grouped representation,输入 LLM [Eq.3]
- Shared LLM Layer 处理 combined embedding: c_t = E_speech(s_t) + E_text(t_t) [Eq.1]
- **Text Head**: 预测文本 token [Eq.8]
- **SRH (Speech Refined Head)**: 从 LLM hidden state → Linear ungrouping → k 步自回归生成 speech tokens [Eq.4-6]

**合成侧**:
- CosyVoice Flow Matching 模型: speech tokens + speaker embedding → Mel spectrogram [§3.1]
- HiFi-GAN vocoder: Mel → waveform [§3.1]

### 关键设计选择

**1. Dual-Resolution Speech Representations (DRSR)**

核心创新。解决两个问题:

(a) **Grouping (25Hz → 5Hz)**: 将 k=5 个连续 speech token embeddings 拼接后经 Linear 投影为单个 grouped representation [Eq.3]:

g_i = Linear(concat(s_{ik}, s_{ik+1}, ..., s_{(i+1)k-1}))

[论文原文] 这缓解了 speech-text 频率失配(5Hz 接近 text 的 ~3Hz),减少 LLM 序列长度,降低计算成本 [§3.2]。

[agent 解读] 不同于 SLAM-Omni (Chen et al., 2024a) 对 audio logits 做 linear projection 实现多 token 并行预测,DrVoice 专门设计了 ungrouping+SRH 来自回归恢复 individual tokens。这是关键区分: grouped representation 只用于 LLM 输入,输出端仍保持 25Hz 分辨率。

(b) **Ungrouping + SRH**: LLM 最后一层 hidden state 通过 Linear 投影到 group-sized embedding,再 split 为 k 个 sub-embeddings [Eq.4-5]:

h_ug = W_p * h[SLLM], 然后 Split_k(h_ug) = [h_ug^(1), ..., h_ug^(k)]

SRH 以这 k 个 sub-embeddings 为条件,自回归生成 k 个 speech tokens [Eq.6]:
L_SRH = -sum log P(s_i | s_{<i}, H_{<i})

[论文原文] 直接 grouping 用于生成时会丢失细粒度声学细节;SRH 通过自回归 refinement 恢复这些细节 [§3.2]。

**2. Parallel Speech-Text Modeling**

受 Moshi 启发,在 assistant 端做 modality alignment [§3.2]:
- 文本和语音 token 的 embedding 相加作为 LLM 输入(非拼接)
- 短序列用 <|SIL|> padding 对齐长度
- 联合自回归生成: y_t = (s_t, t_t) [Eq.2]

[agent 解读] 相加而非拼接是关键设计——拼接会使序列翻倍,相加保持序列长度不变,但要求两种模态在 embedding space 中兼容。

**3. 非对称输入输出设计**

用户端用 Whisper encoder(连续表征),助手端用 S3Tokenizer(离散 tokens)[§3.2]。

[论文原文] 人机交互天然非对称: 用户输入通常单模态(文本或语音),助手响应需要协调的多模态输出 [§3.2]。

[agent 解读] 这也是工程实用性考量——Whisper 的连续表征在理解任务上更强,而离散 tokens 在生成任务上更适合 LLM 自回归框架。

### 训练策略

**1. 初始化**:
- Speech Encoder: Whisper-Large-v3 权重 [§3.3]
- Shared LLM: Qwen2.5-7B-Instruct [Appendix A]
- SRH: 预训练 TTS 模型权重(Qwen2.5-0.5B 在 T2M 数据上训练)[§3.3, Appendix A]
- S3Tokenizer + Detokenizer: CosyVoice 权重,全程冻结 [§3.3]

**2. CoM-Mixing Training** [§3.3]:
- Chain-of-Modality (CoM): 先生成完整文本响应,再进行 parallel speech-text 生成
- 7 种交互模式(S2M, S2T, T2M, T2T, STC, SAC, SUC)[Table 1],通过 system prompt 控制输出模式
- 混合这 7 种模式的数据训练,使模型具备灵活的模态切换能力

**3. Core-Cocktail Training** [§3.3]:
- **Stage 1**: 高学习率(1e-4 → 1e-5)全参微调,快速移动参数
- **Merge**: 将 Stage 1 模型 M1 与 base LLM M0 插值: M_r = αM1 + (1-α)M0,α=0(即完全恢复 base LLM!)
- **Stage 2**: 低学习率(2e-5 → 2e-6)在 merged model 上精调

[agent 解读] α=0 是令人惊讶的设计——Stage 1 的全部参数变化被丢弃,但 SRH/Adapter 等非 LLM 模块保留了 Stage 1 的训练。这意味着 Core-Cocktail 的实际效果是: 先用高 LR 训练好外围模块(SRH, Adapter),然后在 base LLM 上用低 LR 做最终联合优化。Table 6 验证了这一策略的有效性: Stage 1 后 avg 从 81.77 降到 70.19,Stage 2 恢复到 74.73。

**4. Reinforcement Learning** [Appendix C]:
- DPO: 用 ASR 数据构建偏好数据,增强真实语音理解
- GSPO: 在数学问题上训练,增强推理能力

**5. 数据**:
- SRH 预训练: ~100K 小时 audio-text paired data [§4.1]
- Post-training: ~3B text tokens 合成语音(CosyVoice),选 ~26K 小时 S2S + ~20K 小时 S2T + ~10K 小时 ASR data [§4.1]

## 实验

| 指标 | DrVoice-7B | Kimi-Audio | Qwen2.5-Omni | Baichuan-Omni-1.5 | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAudioBench Overall | **72.04** | 69.08 | 66.34 | 64.54 | OAB | [Table 2] |
| VoiceBench Overall | **80.17** | 71.14 | 72.83 | 63.84 | VB | [Table 2] |
| UltraEval-Audio Overall | **56.66** | 46.89 | 50.46 | 48.67 | UEA | [Table 2] |
| Big Bench Audio Overall | **74.0** | 45.8 | 53.9 | — | BBA | [Table 2] |
| UTMOS | 4.29 | 3.06 | 4.28 | 4.17 | UEA speech | [Table 3] |
| ASR-WER | 8.36 | 21.06 | **3.48** | 13.17 | UEA speech | [Table 3] |
| Frame Rate (In/Out) | **5/5** | 12.5/12.5 | 25/tau | 12.5/12.5+tau | — | [Table 2] |

**消融实验** (DrVoice-Small, 1.5B) [Table 4]:

| 消融项 | S2M(T) | S2T | T2T | 变化 |
| --- | --- | --- | --- | --- |
| Full model | 68.67 | 72.33 | 75.33 | baseline |
| w/o CSE | 61.67 | 62.33 | 74.00 | S2T -13.8% |
| w/o SRH-Pretrain | 38.33 | 56.00 | 73.33 | S2M(T) -44.2% |
| w/o SRH | 21.67 | 56.00 | 73.00 | S2M(T) -68.4% |
| w/o CoM-Mixing | 58.00 | 58.00 | 68.33 | S2M(T) -15.5%, T2T -9.3% |

**Grouping Factor 消融** [Table 7, Fig 2]:

| k | S2T | S2M(T/S) | GPU Hours (7B, CSE) |
| --- | --- | --- | --- |
| 1 | 55.67 | 4.00/2.67 | 3360 |
| 3 | 64.67 | 15.67/5.00 | 1808 |
| 5 | 63.33 | 37.67/28.00 | 1808 |
| 7 | 62.67 | 36.00/16.67 | — |

## 局限性

1. **无全双工**: DrVoice 不支持用户在模型生成时打断,论文在 Appendix D 承认这是主要局限,计划用 TDM 方案解决 [Appendix D]
2. **ASR-WER 落后**: 8.36 vs Qwen2.5-Omni 的 3.48。论文归因于 Qwen2.5-Omni 直接将 text 喂给 Talker,而 DrVoice 只送 hidden states 给 SRH [§4.2]。[agent 解读] 这暴露了 parallel 架构的固有劣势——speech 生成只能获取 LLM hidden state 的隐式信息,而非显式文本 token
3. **仅英文**: 论文未提及多语言能力,所有评测仅覆盖英文
4. **α=0 的信息丢失**: Core-Cocktail 中 α=0 意味着 Stage 1 对 LLM 的所有参数更新被丢弃,最终 avg 74.73 仍低于 text baseline 81.77(差 7 分),知识保留并不完美
5. **数据合成依赖**: 训练数据大量使用 CosyVoice 合成语音(~3B tokens),合成语音与真实语音的 domain gap 可能限制真实场景性能
6. **SRH 延迟**: 每个 grouped position 需要 k=5 次自回归 forward pass,理论上增加推理延迟(论文未报告实际延迟数字)

## 点评

**亮点**:
1. DRSR 是一个优雅的工程方案: grouping 降低 LLM 负担,SRH 恢复细节,两者解耦且各司其职。消融证据强有力——k=5 时 S2M 从 4.00 跳到 37.67,同时 GPU hours 减半 [Table 7, Fig 2]
2. 4 个 benchmark 全面 SOTA 的结果令人印象深刻,特别是在 Big Bench Audio 上大幅领先(74.0 vs 次优 55.8)
3. Core-Cocktail 训练策略虽然 α=0 看起来激进,但消融数据证明其有效性——关键 insight 是 Stage 1 训练的真正价值在于非 LLM 模块(SRH, Adapter)的优化

**疑问**:
1. α=0 意味着 LLM 权重完全恢复为 base model。那 Stage 1 对 LLM 的训练有什么意义?论文声称"rapidly move parameters into a more favorable region" [§3.3],但 α=0 时这些参数被完全替换,唯一保留的是 SRH/Adapter 的权重。[agent 解读] 真实原因可能是: Stage 1 主要用来训练 SRH 和 Adapter,LLM 的训练是副作用
2. grouping factor k=5 时 S2T 63.33 反而低于 k=3 的 64.67 [Table 7],但 S2M 大幅提升。这暗示 grouping 对理解和生成的影响方向可能不一致
3. 与 Moshi 的对比缺失: 作为同类 parallel 架构,论文未直接与 Moshi 对比,而 Moshi 在 Table 2 中也未出现

**在 Speech-LLM 演进中的位置**: DrVoice 代表了 parallel speech-text 路线的效率优化方向。与 Kimi-Audio (12.5Hz, dual-tokenizer) 相比,DrVoice 用 grouping 实现了更激进的帧率降低;与 Qwen2.5-Omni (text-driven Thinker-Talker) 相比,DrVoice 保留了 speech→text 反馈回路。DRSR 的 grouping+SRH 思路具有通用性,理论上可迁移到任何 token-based speech-text 系统。

## 可复用的 idea

1. **Grouping + SRH 双分辨率**: 任何需要处理高帧率 speech tokens 的 LLM 系统都可以用 grouping 降低输入分辨率 + 专用 head 恢复输出分辨率。k=5 是一个实验验证过的好起点
2. **Core-Cocktail 训练**: 高 LR 全参训练外围模块 → merge α=0 恢复 LLM → 低 LR 联合精调。适用于任何需要在预训练 LLM 上添加新模态的场景
3. **CoM-Mixing 7 模式**: 通过 system prompt 控制模型的输出模态,一个模型覆盖 S2M/S2T/T2M/T2T + 3 种 chain-of-modality 模式
4. **SRH 预训练**: 先用 T2M 数据独立训练小模型初始化 SRH,比端到端训练效率高很多(预训练 vs 不预训练: S2M 38.33 vs 21.67)
5. **非对称 encoder 设计**: 用户端连续表征(Whisper)+ 助手端离散 tokens(S3Tokenizer),各取所长
