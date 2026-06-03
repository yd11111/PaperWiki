---
type: paper
tier: deep
title: "NVSpeech: An Integrated and Scalable Pipeline for Human-Like Speech Modeling with Paralinguistic Vocalizations"
arxiv_id: "2508.04195"
source: "https://arxiv.org/abs/2508.04195"
authors: [Huan Liao, Qinke Ni, Yuancheng Wang, Yiheng Lu, Haoyue Zhan, Pengyuan Xie, Qiang Zhang, Zhizheng Wu]
year: 2025
venue: "arXiv"
tags: [paralinguistic, non-verbal-vocalization, ASR, TTS, dataset, human-like-speech, expressive-TTS, Mandarin]
concepts: ["[[Prosody Modeling]]", "[[Emotion Control in TTS]]", "[[Instruction-Guided Speech Synthesis]]", "[[Speech Tokenizer]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]", "[[模型库/Whisper|Whisper]]"]
tasks: []
datasets: ["[[数据集/Emilia|Emilia]]"]
kb_context_sources: 2
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页: [[Prosody Modeling]], [[CosyVoice 2]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **[[Prosody Modeling]]** (confirmed): 韵律建模覆盖 duration, pitch, energy, pause 四个物理维度。NVSpeech 扩展了韵律建模的范畴 — 副语言发声 (笑声、叹气、咳嗽等) 是韵律的自然延伸,传统系统完全忽视。NVSpeech 的 word-level annotation 填补了韵律建模中 paralinguistic cues 的数据空白。
>
> **[[CosyVoice 2]]** (confirmed): 阿里巴巴通义实验室的流式零样本 TTS 模型。NVSpeech 直接以 CosyVoice 和 CosyVoice2 为 TTS backbone,通过扩展其词表来支持副语言标签的显式插入。
>
> **[[Emotion Control in TTS]]** [待确认]: 情感控制 TTS 的核心挑战是情感与其他语音属性深度纠缠。NVSpeech 的方法从不同角度切入 — 不直接建模抽象情感,而是建模具体的副语言行为 (笑、叹气、犹豫),这些行为是情感的外在表现。
>
> **[[Instruction-Guided Speech Synthesis]]** [待确认]: 指令引导合成将 TTS 重构为指令跟随任务。NVSpeech 的方法与指令引导互补 — 通过在文本中显式插入 `[Laughter]`、`[Breathing]` 等标签实现 token-level 控制,是比自然语言指令更精确的副语言控制方式。
>
> 检索命中: [[Prosody Modeling]], [[CosyVoice 2]] | 过滤: [[Emotion Control in TTS]](pending-review), [[Instruction-Guided Speech Synthesis]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个集数据集构建、副语言感知 ASR、可控 TTS 于一体的端到端 pipeline,以 18 类 word-level 副语言标注为核心,实现副语言发声的自动识别和可控合成
> - **路线**: [Step 1] 人工标注 48K 话语 (18 类 PV, word-level) → [Step 2] 训练 paralinguistic-aware ASR (SenseVoice 最优, F1 0.73/0.85) → [Step 3] ASR 自动标注 174K 话语 (573.4h) → [Step 4] CosyVoice/CosyVoice2 词表扩展 + 微调 → 可控 PV 插入的人类般语音合成
> - **指标**: Tagging SenseVoice F1 0.73; ASR in-domain CER 4.61% + F1 0.83; TTS CosyVoice2 (large-scale) CER_w/o_para 3.73% + UTMOS 2.67 + listener win rate 75.4% [Table 3, 4, 5, Fig 5]
> - **可借鉴**: (1) 18 类 word-level PV 分类体系 (生理/话语态度/语篇标记) 可复用于中文语音标注; (2) ASR 内联 PV token 方案 — 将 PV 作为文本 token 而非后处理; (3) 词表扩展 fine-tune 方式给已有 TTS 系统添加 PV 控制能力; (4) 跨域泛化验证 (游戏→开放域)
> - **局限**: 仅中文 (Mandarin); 人工标注 48K 话语成本高; 游戏语音为主可能存在域偏移; UTMOS 分数偏低 (~2.5); 仅测试 CosyVoice 系列

## 核心问题

传统 ASR 和 TTS 系统忽视副语言发声 (paralinguistic vocalizations, PVs) — 笑声、叹气、呼吸、犹豫等非语言声音 [§Introduction]。这些信号在自然口语中普遍存在,编码情感、意图和交互状态。核心问题有三个 gap [Fig 1]:

1. **数据 gap**: 现有语音数据集缺乏 word-level 的 PV 标注 [§Introduction]
2. **识别 gap**: 传统 ASR 忽略 PV,无法同时转写词汇内容和副语言线索 [§Introduction]
3. **生成 gap**: 传统 TTS 无法在任意位置显式插入 PV,合成语音缺乏自发性 [§Introduction]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构: 四步 Pipeline [Fig 2]

NVSpeech 的 pipeline 包含四步 [§Introduction, Fig 2]:

```
Step 1: Word-level Human Annotation (48K utterances, 76h)
    ↓
Step 2: Paralinguistic-aware Tagging & ASR Model Training
    ↓
Step 3: Large-scale Automatic Annotation (174K utterances, 573.4h)
    ↓
Step 4: Expressive TTS Fine-tuning (CosyVoice/CosyVoice2 + vocab expansion)
```

### Step 1: 人工标注数据集 [§The NVSpeech Dataset]

#### 18 类副语言分类体系 [§Dataset Construction]

标签体系分三大类 [§Dataset Construction, Table 4 (Appendix E)]:
1. **Non-verbal vocalizations (NVV)**: Breathing, Crying, Laughter, Cough, Sigh, Shh
2. **Prosodic and attitudinal cues**: Surprise-ah/oh/wa/yo, Confirmation-en, Dissatisfaction-hnn, Question-ah/oh/ei/en/yi
3. **Discourse-like markers**: Uhm

[论文原文] 这些标签源自对普通话自发语音的实证分析,捕捉高频 PV 功能和语音学上可区分的类别,支持话语连贯性 [§Dataset Construction]。

#### 数据来源和标注 [§Dataset Construction]

- **核心数据**: 原神 (Genshin) + 崩坏: 星穹铁道 (StarRail) 中文语音 — 多样化游戏内场景 (问候/战斗/叙事) [§Dataset Construction]
- **NVV 增强**: 500 coughing + 500 crying clips from Nonspeech7k; 166 合成话语 (CosyVoice2 + DeepSeek-R1 生成) 覆盖稀少类别 (Surprise-yo, Question-en, Shh) [§Dataset Construction]
- **标注流程**: 10 名标注员,标注 UI 支持 PV 标签点击插入 [Fig 6 (Appendix F)],5% 交叉标注,Cohen's kappa > 0.85 [§Dataset Construction]
- **最终规模**: 48,430 utterances, 76 hours, 1,578 speakers [Table 1]

### Step 2: 副语言感知模型训练 [§Paralinguistic Speech Recognition Model]

#### 2a. Paralinguistic Tagging [§Paralinguistic Tagging]

将 PV 标注为句子级多标签分类问题 [§Paralinguistic Tagging]:

$$\mathcal{L}_{\text{BCE}} = -\sum_{c=1}^{C} [y_c \log \hat{y}_c + (1-y_c) \log(1-\hat{y}_c)]$$

**Baselines** [§Paralinguistic Tagging]: PANNs (CNN, Wavegram_Logmel_Cnn14), SenseVoice-Small (Transformer, encoder-only), Qwen-Audio (Whisper encoder + LLM decoder)

**结果** [Table 3]: SenseVoice 最优 F1 0.73,PANNs 0.72,Qwen-Audio 0.61。SenseVoice 受益于伪标签数据和 ASR-aware 训练 [论文原文]。

#### 2b. Paralinguistic-Aware ASR [§Paralinguistic Aware Speech Recognition]

将 PV 标签作为特殊 token 内联到转写文本中,如 `"不知道[Breathing]，我从来没有想过"` [§Paralinguistic Aware Speech Recognition]。

**四种模型** [§Baseline Models]:
1. **Paraformer** (非自回归, CIF 机制): 训练扩展词表+CTC loss [Eq.2-3]
2. **SenseVoice-Small** (非自回归 encoder-only): task embedding + CTC loss [Eq.4-5]
3. **Qwen-Audio** (Whisper + LLM): instruction prompt + LM loss [Eq.6]
4. **Whisper**: autoregressive LM loss [Eq.6]

**结果** [Table 4]:
- **In-domain**: SenseVoice CER 4.61% + F1 0.83 最优; Paraformer Para Det Rate 96.1% 最高
- **Open-domain**: SenseVoice CER 3.79% + F1 0.85 仍然最强,确认跨域泛化 [§Results]

[论文原文] SenseVoice 的 encoder-only SAN-M 架构使其同时在词汇准确性和 PV 检测上取得最佳平衡 [§Results]。Qwen-Audio 在所有指标上最差,因为 Whisper-initialized encoder 和 LLM decoder 对语义抽象优化,对细粒度 PV 线索不敏感 [§Results]。

### Step 3: 大规模自动标注 [§Large-scale automatically scaled datasets]

使用 Step 2 中最优的 SenseVoice ASR 模型自动标注大量未标记数据 [§Large-scale automatically scaled datasets]:

- **数据来源**: (1) Genshin/StarRail 未标注部分; (2) Emilia 子集 (He et al., 2025); (3) in-the-wild 录音 (talk shows, interviews, debates, audiobooks) [§Large-scale automatically scaled datasets]
- **增强**: 1,362 NVV clips (Crying, Cough) from Nonspeech7k [§Large-scale automatically scaled datasets]
- **全部中文**, 场景从游戏到自发对话 [§Large-scale automatically scaled datasets]
- **最终规模**: 174,179 utterances, 573.4 hours, >1,964 speakers [Table 1]

### Step 4: 表达性 TTS 微调 [§Paralinguistic-enhanced TTS Experiments]

#### TTS 方法 [§Experimental Setups]

在 CosyVoice 和 CosyVoice2 上微调:
1. 扩展词表,添加 PV 特殊 token (如 `[Laughter]`, `[Breathing]`)
2. 训练数据: 35% 普通语音 + 65% PV-rich 话语 [§Experimental Setups]
3. 目标文本示例: `"[Shh]你走…[Breathing]我留听"` [Fig 2, Step 4]

[论文原文] 通过词表扩展而非外部条件信号,PV 成为 first-class token,支持在任意词位置插入 PV [§Experimental Setups]。

### 关键设计选择

#### 1. Word-level vs Sentence-level 标注 [§Introduction, Table 1]

[论文原文] 现有 PV 数据集 (SMC, SVC, RAMC, VocalSound, Nonspeech7k) 大多只提供 sentence-level 或 segment-level 标注,无法精确对齐 PV 与词汇内容的时序关系 [Table 1]。NVSpeech 的 word-level 标注是首个支持 in-context PV 建模的中文大规模数据集 [§Introduction]。

#### 2. 为什么 18 个类别? [§Dataset Construction]

[论文原文] 标签源自对普通话自发语音的实证分析。高频 PV (Breathing, Laughter) 和功能性可区分的话语标记 (Question-ah vs Surprise-ah 靠语调/语境区分) 被保留 [§Dataset Construction]。

[agent 解读] 18 类的粒度在"足够细以区分语义功能"和"足够粗以保证标注一致性"之间取得平衡。Cohen's kappa > 0.85 证明标注一致性良好。

#### 3. 为什么不直接用 emotion labels? [agent 解读]

NVSpeech 的方法论选择是建模**可观察的物理行为**(笑声、叹气）而非**抽象情感状态**（高兴、悲伤）。这避免了情感标注的主观性问题,且 PV 行为可通过 ASR 自动检测并大规模扩展标注,而情感标注通常需要昂贵的人工判断。

### 训练策略

**CosyVoice** [Appendix A]: 官方微调脚本, 4×A100 80GB, 30 epochs, dynamic batchsize, lr 1e-5, Adam [Appendix A]
**CosyVoice2** [Appendix A]: 4×A100 80GB, 20 epochs, dynamic batchsize, lr 1e-5, Adam, ~35 GB GPU [Appendix A]

## 实验

| 指标 | CosyVoice (pre) | CosyVoice (human) | CosyVoice (auto-eq) | CosyVoice (auto-large) | CosyVoice2 (pre) | CosyVoice2 (auto-large) | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CER ↓ (in) | - | 8.78% | 8.59% | 7.96% | - | 7.83% | In-domain | [Table 5] |
| CER_w/o_para ↓ (in) | 7.42% | 4.21% | 4.07% | 4.05% | 3.13% | 3.73% | In-domain | [Table 5] |
| SIM ↑ (in) | 0.727 | 0.736 | 0.736 | **0.733** | 0.710 | 0.700 | In-domain | [Table 5] |
| UTMOS ↑ (in) | **2.69** | 2.54 | 2.54 | 2.57 | **2.69** | 2.67 | In-domain | [Table 5] |
| CER ↓ (open) | - | 11.09% | 9.97% | 9.99% | - | 8.44% | Open-domain | [Table 5] |
| UTMOS ↑ (open) | 2.49 | 2.35 | 2.35 | **2.39** | 2.25 | 2.26 | Open-domain | [Table 5] |

**Human evaluation** [§Human Evaluation, Table 6, Fig 5]:
- **Listener preference** (vs pre-trained): CosyVoice win rate 78.7%, CosyVoice2 win rate 75.4% [Fig 5]
- **NMOS** (naturalness): CosyVoice 3.9±0.20, CosyVoice2 4.0±0.16 [Table 6]
- **QMOS** (quality): CosyVoice 4.04±0.15, CosyVoice2 3.96±0.14 [Table 6]
- **PV Recall**: CosyVoice 0.604, CosyVoice2 0.619 [Table 6]

**关键发现**:
1. **Human-labeled 微调启用 PV 生成**: 从无 PV 能力到有 PV 能力,CER 略升但获得可控 PV 插入 [Table 5, §Main Results]
2. **Auto-labeled (equal size) 匹配人工标注**: 同等数据量下自动标注数据的效果与人工标注相当 (comparable SIM 和 UTMOS) [Table 5, §Main Results]
3. **大规模自动标注进一步提升**: Large-scale auto-labeled 数据 CER_w/o_para 最低 (4.05% CosyVoice in-domain),最高 12.8% 相对 CER 降低 [Table 5, §Main Results]
4. **不损害基础 TTS 质量**: 微调后 UTMOS 基本持平 (2.69→2.57),SIM 略有波动但整体稳定 [Table 5]
5. **强烈的人类偏好**: 75-79% 听众更偏好 PV-enhanced 语音 [Fig 5],证明 PV 对自然度的重要性

## 局限性

- **仅中文 (Mandarin)**: 虽然补充实验 [Appendix D] 在英语数据集 (DisfluencySpeech, NonverbalTTS) 上验证了跨语种可行性,但主 pipeline 和数据集仅覆盖中文 [§Introduction]
- **游戏语音域偏移**: 人工标注数据以原神/星穹铁道为主,游戏语音的表演风格可能与自然对话有差异 [agent 解读]
- **人工标注成本**: 48K 话语的 word-level 标注需要 10 名标注员,成本较高 [§Dataset Construction]
- **UTMOS 偏低**: 所有模型 UTMOS 在 2.25-2.69 范围,远低于 TTS 领域通常的 3.5+ 水平 [Table 5]。可能与游戏/自发语音的特殊录音条件有关 [agent 解读]
- **PV Recall ~60%**: 约 40% 的 PV 标签未被正确合成 [Table 6],仍有提升空间
- **仅测试 CosyVoice 系列**: 未在其他 TTS 架构上验证 (如 VALL-E, MaskGCT, Seed-TTS)

## 点评

1. **填补重要空白**: NVSpeech 抓住了一个被长期忽视但极其重要的问题 — 副语言发声。传统 TTS 生成的语音"太干净"，缺乏自然口语中的呼吸、笑声、犹豫,这是合成语音"不像人"的重要原因之一 [agent 解读]。

2. **端到端 pipeline 的完整性**: 从标注体系→数据→ASR→自动扩展→TTS,四步 pipeline 形成闭环。ASR 自动标注使 pipeline 可扩展,不依赖持续的人工标注 [agent 解读]。

3. **Word-level 标注的关键意义**: 这是本文最核心的贡献。Sentence-level 标注只能告诉你"这句话有笑声"，word-level 标注精确到"这个词之后有笑声"，后者才能实现 TTS 中的 context-aware PV 插入 [agent 解读]。

4. **与 [[Emotion Control in TTS]] 的互补**: NVSpeech 不直接建模"快乐/悲伤"等抽象情感,而是建模笑声、叹气等具体行为。这两种方法互补: 情感控制提供高层意图,PV 控制提供底层实现 [agent 解读]。

5. **实用但需注意域偏移**: 游戏语音数据的表演性可能使模型在某些场景过度戏剧化。Open-domain 测试集部分缓解了这一担忧,但更多真实场景验证仍需要 [agent 解读]。

6. **与 Emilia 的生态协同**: NVSpeech 使用 Emilia 子集进行自动标注扩展,展示了大规模通用语音数据集 (Emilia) 如何赋能下游特化任务 [agent 解读]。

## 可复用的 idea

1. **18 类 word-level PV 分类体系**: 可直接复用于任何中文语音标注任务,三大类 (生理/态度/话语标记) 的组织方式清晰
2. **ASR 内联 PV token**: 将 PV 作为特殊 token 嵌入 ASR 输出 (如 `"不知道[Breathing]"`),而非后处理检测,使 PV 成为转写的一等公民
3. **词表扩展微调**: 给已有 TTS 系统添加新控制维度的低成本方法 — 只需扩展词表并在标注数据上微调,无需修改架构
4. **人工标注→ASR 自动扩展的 bootstrap 范式**: 小量人工标注训练 ASR → ASR 自动标注大量数据 → TTS 微调,可推广到任何需要特殊标注的场景
5. **In-domain + Open-domain 双测试集设计**: 分别验证受控场景性能和真实场景泛化,是 robustness 评估的最佳实践
