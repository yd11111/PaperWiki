# Context-Aware VoiceAgent TTS 实施计划

> 2026-06 | 从单句合成到上下文感知的完整路径
>
> 当前状态: ASR+TTS 级联 VoiceAgent，TTS 单句合成，不知道上下文
>
> 参考: [[2026-ContextAwareCSS调研|CSS 调研 (18篇)]] + [[Survey-TTSEvaluation2025|TTS 评估汇总 (23篇)]] + [[Survey-SpeechAudioLLMEvaluation2025|LALM 评估汇总 (28篇)]] + 汇报版 v3

---

## 总体路线图

```
当前状态                Step 1                  Step 2                  Step 3
──────────            ──────────              ──────────              ──────────
ASR → LLM → TTS      ASR → LLM → TTS         ASR → LLM → TTS         统一模型
TTS 只看当前句         + LLM 推断风格指令        + 音频上下文条件          直接对话级建模

不改模型               不改 TTS 模型            改 TTS 架构              改全栈
成本最低               依赖 LLM 能力            依赖对话数据              最大投入
立刻可做               2-4 周                   1-2 个季度               中长期
```

**三步之间的依赖关系:**
- Step 1 是 Step 2 的**验证器** — 如果文本上下文都没用，音频上下文也不会有用
- Step 1 的评估体系直接复用到 Step 2（加上跨句一致性维度）
- Step 1 积累的偏好数据是 Step 2 做 RL/DPO 的基础
- Step 3 依赖 Step 2 的对话数据管线和评估框架

---

## Step 1: 文本上下文注入（不改 TTS 模型）

### 核心思路

利用 pipeline 中已有的 LLM，从对话历史推断当前句的合适风格，转化为 TTS 可接受的控制信号。**TTS 模型本身不改，只改喂给它的输入。**

### 方案 1A: LLM 生成风格标签

**做什么**: LLM 在生成回复文本的同时，输出一组风格标签（情感类别、语速、音量等），这些标签作为 TTS 的 instruct 条件。

**具体实现**:

```
用户: "堵死了，烦死了，到底还要多久啊！"
                ↓ ASR
LLM 输入: [对话历史] + [用户当前输入]
                ↓
LLM 输出:
{
  "reply": "我理解您的心情，帮您看看有没有更快的路线。",
  "style": {
    "emotion": "empathetic",      // 安抚
    "speed": "slightly_slow",     // 稍慢
    "energy": "soft",             // 柔和
    "tone": "warm"                // 温暖
  }
}
                ↓
TTS 输入: [instruct] 用温暖安抚的语气，语速稍慢 [/instruct] 我理解您的心情...
```

**参考案例 1: [[ChatGPT-EDSS]]** (2023, Interspeech)
- ChatGPT 从对话历史提取 intention/emotion/style 三元组
- 三元组通过 BERT 编码 → 线性映射 → FastSpeech 2 条件
- 结果: ChatGPT 推断的风格与人工情感标签表现相当（MOS 3.52 vs 3.54，无显著差异）
- **关键启示**: LLM 推断风格的能力已经接近人类标注，不需要精细标签也能work

**参考案例 2: [[ConversationalTTS-RL]]** (2026, Meta)
- LLaMA 3 70B 生成 textual style tokens
- AR prosody model 用 ICL (in-context learning) 方式：用 fine-grained audio prompts 做风格参考
- 关键发现: **AR 和声学阶段不需要同说话人 prompt — 韵律和音色天然解耦**
- 效果: ICL vs Zero-shot CVAD net win rate +79.6%; ICL vs GPT-4o +5.6%

**实施步骤**:
1. 设计 LLM prompt template，让 LLM 在生成回复时附带风格 JSON
2. 建立风格标签到 TTS instruct 指令的映射规则
3. 在现有 TTS 的 instruct 接口上接入
4. A/B 测试：有风格注入 vs 无风格注入

**预期效果**: 解决"语气完全不搭"的最明显问题，但无法保证跨句一致性。

### 方案 1B: LLM 生成 CoT 风格规划

**做什么**: 不只输出标签，而是让 LLM 进行 Chain-of-Thought 推理，逐步分析上下文再决定风格。

**参考案例: [[CapTalk]]** (2026)
- CoT 控制序列: 对每个 turn 规划 emotion/tone/pitch/energy/speed
- 使用说话人内部相对韵律（归一化到说话人中位数），消除说话人差异
- CoT 预测准确率: emotion 78.5%, tone 76.75%, pitch 83.75%, energy 82.5%, speed 91.25%
- 有 CoT vs 无 CoT: 偏好率 65.5% vs 34.5%

**具体实现**:

```
LLM System Prompt:
你是 VoiceAgent 的风格规划模块。根据对话历史，推理当前回复应该怎么说。

分析过程:
1. 用户当前情绪: [从最近几轮推断]
2. 对话走向: [是在解决问题/闲聊/安抚]
3. 合适的回复语气: [应该匹配/对比/缓和]
4. 具体参数:
   - emotion: [从预定义列表选]
   - pitch: [high/mid/low，相对于说话人基线]
   - energy: [soft/mid/strong]
   - speed: [slow/mid/fast]
```

**参考案例: [[OV-InstructTTS]]** (2026)
- LALM 推理链 (`<think>` block) 将开放词汇叙事指令映射到 emotion labels + acoustic descriptions + paralinguistic tags
- 关键发现: **reasoning chain + enriched transcription 一起用效果最好 (71.57)；只有 fine-grained labels 而没有 reasoning 反而会损害效果 (-0.72)**
- 意味着: CoT 推理过程本身是有价值的，不能跳过直接给标签

**实施步骤**:
1. 设计 CoT prompt，覆盖 VoiceAgent 核心场景（导航/闲聊/客服）
2. 用 few-shot 示例校准 LLM 的风格推理
3. 对比 1A（直接标签）和 1B（CoT 推理），量化差异

### 方案 1C: 检索增强（RAG）

**做什么**: 维护一个"对话-风格"历史库，当前对话匹配到相似历史对话后，复用其风格配置。

**参考案例: [[RADKA-CSS]]** (2025)
- 多属性联合检索: semantic similarity (Sentence-BERT) + style similarity (Wav2Vec2.0-IEMOCAP) + speaker identity (x-vector)
- Top-K=25 检索，softmax 加权聚合音频风格特征
- 结果: N-DMOS 3.904 (+0.184 vs ECSS)，去掉 stored dialogue 知识贡献 -0.197
- **关键发现: CSS 是知识密集型任务，检索相似历史对话显著改善风格渲染**

**实施步骤**:
1. 将 VoiceAgent 历史对话（文本+合成音频+用户反馈）存入向量数据库
2. 对每个新对话，检索最相似的 K 个历史对话
3. 提取历史对话的风格配置（哪些参数效果好/被用户认可）
4. 将检索到的风格配置作为 LLM 推理的参考

**适用场景**: VoiceAgent 运行一段时间后，积累了足够的历史数据。初期数据不足时不适用。

### Step 1 评估方案

| 指标 | 怎么测 | 工具 | baseline |
|---|---|---|---|
| **语境适配度** | A/B 偏好: 有上下文 vs 无上下文，听完整对话选更自然的 | 人工标注 | 无上下文版本 |
| **情感准确率** | 分类器判断合成语音的情感是否匹配 LLM 指定的标签 | emotion2vec / SER 模型 | 当前默认合成 |
| **指令遵循度** | rubric 分解: 拆成二值项逐项评估 | LALM judge (参考 [[AnyAudio-Judge]]) | — |
| **通用质量不退步** | WER/CER + SIM + UTMOS 硬约束 | Whisper + WavLM + UTMOS | 当前模型指标 |
| **延迟影响** | LLM 推理增加的延迟 | 端到端计时 | 当前延迟 |

**语境适配度的具体评测方法**:

准备 30-50 段多轮对话（覆盖导航/闲聊/客服场景），每段 5-10 轮。每段对话用两个版本合成:
- 版本 A: 当前系统（无上下文）
- 版本 B: 加入文本上下文

让 5+ 标注员听完整对话，逐句标注"语气是否合适"(1-5分) + 整段偏好选择。

**自动化评估**:

参考 [[MCLP]] (2026, ICML) 的方法 — 用 LALM 的 continuation log-probability 作为风格一致性代理指标:

```
构造输入: [前一句文本, 前一句音频, 当前句文本]
计算: LALM 对当前句音频 token 的 log-probability
delta-MCLP > 0.1 时, win rate > 0.8 (论文验证)
```

这个指标的好处是**不需要人工标注就能快速迭代**。

### Step 1 优化迭代

```
v0: 固定 prompt template → A/B 测试 → 找到最佳 prompt
v1: 按场景分 prompt (导航/闲聊/客服各一套)
v2: 引入 few-shot (成功案例作为示例)
v3: 用偏好数据 fine-tune LLM 的风格推理能力 (如果 v0-v2 效果不够)
```

**数据需求**: 30-50 段多轮对话脚本（可从现有 VoiceAgent 日志提取），5+ 标注员做 A/B 评测。

---

## Step 2: 音频上下文注入（模型架构改动）

### 前提条件
- Step 1 已验证: 文本上下文确实能改善语境适配度
- 已积累 Step 1 的评测数据和偏好标注
- 已确定需要改进的维度（跨句一致性？情感连续性？）

### 路线 2A: Style Embedding 条件注入（轻量改动）

**做什么**: 前一句（或前几句）合成音频的 style embedding 作为当前句 TTS 的额外条件输入。TTS 主体架构不变，只加一个条件注入模块。

**具体实现**:

```
前 N 句合成音频 → Style Encoder → style embedding
                                      ↓ cross-attention / concat
当前文本 + instruct → TTS Encoder → Decoder → 当前句音频
```

**参考案例 1: [[TextAwareContextAwareTTS]]** (2024, Interspeech)
- 两个模块:
  - Text-aware Style Modeling: CLIP 风格的对比学习，对齐 T5 文本编码器和 Chinese-HuBERT 语音风格编码器，连续风格空间 dim=384
  - Context Encoder: 前一句/当前句/下一句 + BERT embeddings → 融合 VQ 量化的 style embedding (codebook 64x32)
- 训练策略: 两阶段 — 先无上下文预训练，再上下文感知 fine-tune
- 数据: 6kH 中文有声书训练文本编码器；100H 多风格训练语音编码器；20H 有声书 (带句子顺序) 做上下文 fine-tune
- 结果: EMOS 3.61 → 3.93 (+0.32, VITS 骨架); EMOS 3.80 → 4.05 (+0.25, LM 骨架)
- **关键技术**: alpha/beta 阈值半监督对比学习，从语音风格相似度构造正/负样本，无需人工标注

**参考案例 2: [[DiffCSS]]** (2025)
- FACodec (NaturalSpeech 3) 提取韵律 embedding → learnable query cross-attention 压缩到定长
- 上下文: Sentence-T5 文本 + prosody embedding 交替排列，N=4 前文 turns，cross-attention 注入
- **关键发现: 去掉声学上下文的影响 (NDB 4→10) 远大于去掉文本上下文 (NDB 4→5) — 音频上下文比文本上下文更重要**
- 训练: ParlerTTS pretrained on LibriTTS-R 585h → finetune on DailyTalk 20h；Diffusion 冻结 TTS 单独训练
- 结果: E-MOS 3.602 / C-MOS 3.574 (vs GRU baseline 3.209/3.177)

**实施步骤**:
1. 选择 style encoder: 可复用现有 speaker encoder 的架构，增加 emotion/prosody 维度
2. 设计上下文 embedding 注入方式: cross-attention 到 TTS decoder (参考 DiffCSS)
3. 两阶段训练:
   - Phase 1: 冻结 TTS 主体，只训练 style encoder + 注入模块（用有声书数据）
   - Phase 2: 全模型 fine-tune（用对话数据，小 lr）
4. 硬约束验证: SIM 和 WER 不退步

**优势**: 改动最小，风险最低，可以快速验证音频上下文是否有效
**劣势**: 前几句的 style embedding 是"压缩"过的，信息损失不可避免

### 路线 2B: Text-Speech Interleaved 全上下文（重量改动）

**做什么**: 把对话历史的文本 token 和音频 token 交替排列到同一个序列中，让模型在自回归过程中直接看到完整上下文。

**参考案例: [[FireRedTTS2]]** (2025, 小红书)

这是目前工业界最完整的对话 TTS 实现，值得详细参考:

**架构**:
```
输入序列: [S1]<text_1><audio_1>[S2]<text_2><audio_2>...[Sn]<text_n>
                                                        ↓ AR 生成
输出: <audio_n>
```
- Dual-Transformer: 大 backbone (Qwen2.5) 预测第 1 层 RVQ，小 decoder (Qwen2.5) 生成剩余 15 层
- 12.5Hz Streaming Tokenizer: Whisper encoder 50Hz → 4x 下采样 → 12.5Hz，每帧编码 80ms
  - 3 分钟对话: 9000 token → 2250 token (4x 压缩)
  - codebook 2048 entries, 16 层 RVQ

**训练策略（三阶段课程学习）**:
```
Stage 1: 1.1M h 单人独白预训练 (建立基础 TTS 能力)
    ↓
Stage 2: 300K h 多人对话后训练 (2-5 speakers, 学习对话模式)
    ↓
Stage 3: SFT (少量高质量数据精调)
```

**关键设计决策**:
1. 情感从**显式标签→隐式推断**: v1 用 13 种显式情感标签 (准确率 97-100%)，v2 从上下文自动推断 (83-93%)，但用户感知更自然
2. Dual-transformer 优于 delay pattern: 每个时间步完整看到所有历史 token，首帧延迟更低
3. 低帧率是对话场景的关键: 12.5Hz 使 context window 能放更多对话历史

**效果**:
| 指标 | FireRedTTS-2 | 对比 |
|---|---|---|
| Podcast CER (zh) | 2.08 | beat MoonCast 3.81 |
| 情感准确率 | 83-93% (6类) | 隐式推断，无需标签 |
| 说话人数 | 最多 4 人 | 3 分钟连续 |
| SIM | 0.753 | 多说话人场景 |

**参考案例: [[DialoSpeech]]** (2025)
- 双轨 LLM (LLaMA-based, 0.5B): causal cross-attention 连接双语音 token 流
- `[spkchange]` 标签 + `<SIL>` token 做隐式 turn-taking
- 关键发现: **monologue-to-dialogue curriculum 是必须的** — 不做 curriculum 直接训对话数据，WER=116%（完全崩溃）
- 数据: 10Kh (3K 专业中文对话 + 5K 播客 + 2K Fisher 英文)
- 结果: CER 2.27% (vs CosyVoice2 2.81%), Spontaneity MOS 3.96 (vs CosyVoice2 3.44)

**参考案例: [[ZipVoice-Dialog]]** (2025, 小米)
- 123M NAR (Zipformer + CFM)，极致效率
- **monologue-to-dialogue curriculum learning**: 不做的话 alignment 完全崩溃 (WER=116%)
- learnable speaker-turn embeddings: 解决说话人区分问题
- 开源 OpenDialog 数据集 (6.8Kh)
- 结果: CMOS +1.17 vs MoonCast, 15x 更快

**实施步骤**:
1. Tokenizer 确认: 当前 tokenizer 的帧率是否支持长上下文？如果 > 25Hz，考虑降帧率
2. 数据准备:
   - 收集/清洗对话数据（有声书、播客、VoiceAgent 历史录音）
   - 构建 text-speech interleaved 训练格式
3. 训练:
   - 不要从头训练！用现有单句模型做初始化
   - Curriculum: 单句 → 2 轮对话 → 5 轮对话 → 10 轮对话，逐步增加上下文长度
   - 参考 DialoSpeech/ZipVoice-Dialog: **curriculum 是必须的，直接训对话数据会崩溃**
4. 推理适配:
   - 缓存前几轮的 KV-cache，避免重复计算
   - 设置最大上下文长度（比如 5 轮），超过后滑窗

### Step 2 评估方案

除了 Step 1 的所有指标外，新增:

| 指标 | 怎么测 | 参考 |
|---|---|---|
| **跨句一致性** | 相邻句 speaker embedding 距离的方差；F0 均值/方差的变化幅度 | — |
| **Style Amnesia 测试** | 设定一个情绪，连续合成 10 句，第 1/5/10 句的情感分类结果是否一致 | — |
| **MCLP** | 构造 [transcript, audio, transcript]，计算 LALM continuation log-probability | [[MCLP]]: delta>0.1, win rate>0.8 |
| **DS-WED 韵律多样性** | 确保 context-aware 不会让韵律变得单调 | [[ProsodyEval]]: r=0.77 with human |
| **多轮对话评估** | 5 维打分: intent/SIM/context/emotion/naturalness | [[UniSRM]] T4: acc 88.89% |
| **整段偏好** | 完整 5-10 轮对话 A/B 偏好 | 人工标注 |

**MCLP 的具体使用方式** (参考论文):

```python
# 伪代码
def compute_mclp(lalm, transcript, audio_tokens, next_transcript):
    """
    构造双 turn 上下文，计算 LALM 对音频 token 的 log-probability。
    固定 transcript 消除内容变量，likelihood 变化仅反映风格差异。
    """
    context = concat(transcript, audio_tokens, next_transcript)
    log_probs = lalm.forward(context)  # 计算每个 audio token 的 log prob
    return mean(log_probs[audio_token_positions])

# 比较两个系统
mclp_A = compute_mclp(lalm, text, audio_system_A, next_text)
mclp_B = compute_mclp(lalm, text, audio_system_B, next_text)
# delta > 0.1 → 显著偏好 (win rate > 0.8)
```

### Step 2 优化手段

**数据侧优化**:
- 有声书: 天然长上下文 + 丰富韵律变化，最容易获取
- 播客: 多人对话 + 自然语气切换，需要 speaker diarization 预处理
- VoiceAgent 历史: 最匹配目标场景，但数据量可能不足
- 合成数据增强: 用 Step 1 的 LLM 风格推断 + 当前 TTS 合成对话，再人工标注偏好

**RL/DPO 优化** (在 Step 2 模型基础上):

参考 [[ConversationalTTS-RL]] 的 RL 方案:
```
reward = α_AES × AES(waveform) - α_CTC × L_CTC
```
- AES: Aesthetic quality score (声学质量)
- CTC: 防止 reward hacking / 文本幻觉的正则化
- KL-regularized policy gradient, 6 MC samples per transcript

参考 [[Multi-RewardGRPO]] 的多维 reward:
- 5 个 reward 函数: R_crt (WER) + R_sim (SIM) + R_mos (MOS) + R_pro (韵律对齐) + R_ent (文本相关性)
- R_pro 贡献 +0.24 MOS, R_ent +0.20 MOS
- **10K 样本即可获得可衡量的 RL 提升**

**防 Reward Hacking** (从同行教训学习):
- [[NoVerifiableRewardforProsody]]: GRPO 优化韵律会导致 pitch variability 显著下降 → 用 iterative DPO + 约 200 human preference pairs/round 替代
- [[Seed-TTS]]: 过度优化 WER 牺牲自然度 → 多维 Pareto reward

### Step 2 数据需求

| 数据类型 | 来源 | 规模目标 | 用途 |
|---|---|---|---|
| **有声书** | 公开 + 采购 | 500h+ | context-aware fine-tune |
| **播客/对话** | 公开 (OpenDialog 6.8Kh) + 内部录制 | 200h+ | 对话模式学习 |
| **VoiceAgent 对话** | 线上日志 | 100h+ | 目标场景适配 |
| **偏好标注** | 人工 A/B | 5K+ 对 | DPO/RM 训练 |
| **风格标注** | 人工多维 | 2K+ 样本 | 属性级 RM |

---

## Step 3: 端到端对话语音建模（中长期方向）

### 方向概述

TTS 不再是 pipeline 中的独立模块，语音生成成为 LLM 的原生能力。

**参考**:
- [[CSM]] (Sesame): Llama backbone + multi-codebook AR，多轮 audio tokens 作为条件，14.7K stars
- Qwen3.5-Omni: Thinker-Talker + ARIA 自适应对齐
- Step-Audio 2.5: 统一 MoE backbone + 三分支特化 (ASR/TTS/Realtime)
- Kimi-Audio: 离散语义 token + 连续 Whisper 特征双输入

### 前提条件（Step 3 启动需要）

1. ✅ Step 2 验证了上下文信息对 TTS 质量有显著提升
2. ✅ 对话数据管线成熟（清洗/标注/格式化）
3. ✅ 评估体系能评估对话级语音质量
4. ✅ 有足够的训练数据（100Kh+ 级别）
5. ✅ 算力支持大模型训练（7B+）

### 关键技术选型

| 选项 | 参考 | 优势 | 劣势 |
|---|---|---|---|
| AR + 全上下文 | FireRedTTS-2, CSM | 最自然的上下文建模 | 长序列推理慢 |
| NAR + curriculum | ZipVoice-Dialog | 极致效率 (123M) | 上下文建模能力弱于 AR |
| 统一大模型 | Step-Audio 2.5, Qwen3.5-Omni | 理解+生成一体化 | 需要最大投入 |

**建议**: 先从 Step 2B 的 text-speech interleaved 路线自然演进，不急于跳到全新架构。

---

## 评估体系（贯穿三步）

### 分层评估框架

```
┌─────────────────────────────────────────────────┐
│  Layer 3: 对话级体验评估                          │
│  用户偏好 A/B + 交互延迟 + NPS                    │
│  ← Step 2/3 新增                                 │
├─────────────────────────────────────────────────┤
│  Layer 2: 上下文评估                              │
│  语境适配度 + 跨句一致性 + Style Amnesia + MCLP    │
│  ← Step 1 建立, Step 2 扩展                      │
├─────────────────────────────────────────────────┤
│  Layer 1: 单句基线 (硬约束)                       │
│  WER/CER + SIM + UTMOS/DNSMOS                    │
│  ← 贯穿所有步骤                                  │
└─────────────────────────────────────────────────┘
```

### 自动化评估 Pipeline

每次模型迭代自动跑:

```bash
# Layer 1: 硬约束 (GATE — 不过则拒绝)
run_wer --model $MODEL --data seed_tts_eval  # CER < 2%
run_sim --model $MODEL --data seed_tts_eval  # SIM > 0.80
run_utmos --model $MODEL --data seed_tts_eval  # UTMOS > 3.8

# Layer 2: 上下文评估 (DIAGNOSTIC)
run_context_eval --model $MODEL --data voiceagent_dialogues
  # 语境适配度 (LALM judge)
  # 跨句 SIM 方差
  # Style Amnesia (10句连续测试)
  # MCLP score

# Layer 3: 对话级 (PERIODIC — 每周人工评测)
# 30段对话 × 5 标注员 × A/B 偏好
```

### 评估工具选型

| 需求 | 推荐方案 | 参考 |
|---|---|---|
| 单句 MOS 预测 | UTMOS / DNSMOS | 行业标配 |
| 语境适配度 judge | LALM + rubric 分解 | [[AnyAudio-Judge]]: 指令→二值 rubric→逐项 yes/no |
| 风格一致性自动指标 | MCLP | [[MCLP]]: delta>0.1, win rate>0.8 |
| 韵律多样性 | DS-WED | [[ProsodyEval]]: r=0.77 |
| 多维对话评估 | UniSRM 5 维 | [[UniSRM]] T4: 88.89% acc |

---

## 数据建设路线图

### Phase 1: 利用现有数据 (Step 1 阶段)

```
VoiceAgent 线上日志 → ASR 转写 → 对话文本 → LLM 风格标注
     ↓
30-50 段代表性对话 → A/B 评测数据
```

### Phase 2: 对话数据采集 (Step 2 准备)

| 来源 | 处理 | 用途 |
|---|---|---|
| 有声书 | VAD + 说话人聚类 + 章节切分 | 长上下文+丰富韵律 |
| 播客 | Speaker diarization + 重叠检测 + ASR | 多人自然对话 |
| OpenDialog (6.8Kh, 开源) | 直接使用 | 基线对话数据 |
| 内部录制 | VoiceAgent 场景模拟录制 | 目标场景数据 |

**数据管线参考** ([[FireRedTTS]]):
```
源分离 → 语音增强 → VAD 分割 → 说话人聚类 → ASR 转写
→ 三维过滤 (DNSMOS>3.3 + 滚降频率>7kHz + ASR 置信度>0.8)
```
v1 从 624Kh 清洗到 248Kh (保留率 40%)

### Phase 3: 偏好数据标注 (RM 训练)

```
合成对话 (有/无上下文两版本) → 标注员 A/B 偏好 → 偏好对数据
                                                    ↓
                                              RM 训练 → DPO
```

目标: 10K 偏好对 (参考 SpeechJudge 99K 最终规模，但初期 10K 足以启动)

---

## 落地节奏

```
Week 1-2          Week 3-4          Month 2           Month 3           Q4
─────────        ─────────        ─────────         ─────────        ─────────
[Step 1A]        [Step 1 评测]     [Step 1 迭代]      [Step 2A 启动]    [Step 2B]
LLM prompt       A/B baseline     场景分 prompt       style encoder    interleaved
风格推断          语境适配度       + few-shot 优化     设计+训练         全上下文
                 首批数据                             有声书数据准备     对话数据

[评测框架 v1]    [评测框架 v2]                        [偏好标注启动]    [RM 训练]
Layer 1 跑通     Layer 2 跑通                         A/B 数据采集      DPO 实验
MCLP 接入        LALM judge
```

### 关键里程碑

| 时间 | 里程碑 | 成功标准 |
|---|---|---|
| Week 2 | Step 1A 上线 | LLM 风格推断 + TTS 合成 pipeline 跑通 |
| Week 4 | Step 1 评测完成 | 有上下文 vs 无上下文 A/B 偏好率 > 60% |
| Month 2 | Step 1 优化完成 | 语境适配度提升可量化，通用质量不退步 |
| Month 3 | Step 2A 原型 | Style embedding 注入 demo，跨句 SIM 方差下降 |
| Q4 | Step 2B 上线 | 5 轮对话连续合成，Style Amnesia 测试通过 |

### 风险和应对

| 风险 | 应对 |
|---|---|
| LLM 风格推断不稳定 | 限制标签空间（5-8 个离散类别），不做开放生成 |
| 增加延迟 | LLM 推断和 TTS 可以流水线化；风格推断可以缓存 (同一对话状态不变则复用) |
| 对话数据不足 | 先用有声书 (容易获取) 验证方法，再补充真实对话 |
| Curriculum 训练崩溃 | 严格遵循 mono→dialogue 课程 (ZipVoice-Dialog: 不做 curriculum WER=116%) |
| 通用质量退步 | SIM/WER 硬约束，每次迭代必跑 Layer 1 |

---

## 总结

**核心策略: 用最低成本的 Step 1 验证方向，用评测量化每一步的进步，用数据驱动持续优化。**

Step 1 (文本上下文) 的投入产出比最高 — 不改模型、2-4 周出结果、立刻可用。如果 A/B 偏好率 > 60%，说明方向正确，值得投入 Step 2。

Step 2 的核心参考是 FireRedTTS-2 (text-speech interleaved) 和 DiffCSS (声学上下文 > 文本上下文)。关键是 **curriculum 训练不能跳过**，否则会崩溃。

评测贯穿始终: MCLP 做自动化快速迭代，A/B 偏好做阶段性校准，Layer 1 硬约束确保不退步。
