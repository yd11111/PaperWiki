# TTS 论文调研 Prompt（检索驱动版 v2）

> 使用说明：在 Claude Code 中直接粘贴 Phase 1 prompt 开始执行。
> 每个 Phase 完成后，用户确认再进入下一个 Phase。

---

## Phase 1: 检索收集（核心步骤）

```
你是一名语音方向论文检索助理。我需要整理 2025-01-01 至今 TTS 方向的重要论文。

⚠️ 核心规则：
1. 所有论文必须来自实际检索，禁止凭记忆输出。
2. 若标题或摘要无法明确判断论文核心任务属于 TTS / speech synthesis，
   不要纳入核心列表，改放入"待核实/边界相关"列表。
3. 优先保证"真实可核查"而非数量。宁可少收不可误收。

### 你的工作方式

1. 用 Playwright 逐批检索以下来源，每完成 2 个批次输出一次中间结果，等待确认后继续
2. 每篇论文必须附上可验证的 arXiv ID 或会议链接
3. arXiv 为主检索源；若某批次结果明显偏少（< 5 篇），额外用 Semantic Scholar 补搜同批关键词
4. 所有搜索限定日期范围 2025-01-01 至今

### 检索计划（按子方向逐批执行）

请依次执行以下批次。每批先搜 arXiv（arxiv.org/search 高级搜索），
结果偏少时补搜 Semantic Scholar（semanticscholar.org）。

| 批次 | 搜索关键词 | 子方向 |
|------|-----------|--------|
| 1 | "text to speech" OR "speech synthesis" | 通用 TTS |
| 2 | "zero-shot TTS" OR "zero-shot speech synthesis" OR "voice cloning" OR "speaker adaptation" OR "few-shot TTS" | 零样本/声音克隆 |
| 3 | ("LLM" OR "large language model" OR "language model" OR "speech language model") AND ("text to speech" OR "speech synthesis" OR "TTS") | LLM-based TTS |
| 4 | ("codec" OR "token" OR "discrete speech representation" OR "semantic token" OR "acoustic token") AND ("speech synthesis" OR "text to speech" OR "TTS") | Codec/Token-based |
| 5 | "controllable TTS" OR "controllable speech synthesis" OR "expressive speech synthesis" OR "emotional TTS" OR "style TTS" OR "prompt-based TTS" OR "promptable TTS" | 可控/情感 TTS |
| 6 | "multilingual TTS" OR "cross-lingual speech synthesis" OR "code-switching TTS" OR "polyglot TTS" | 多语言 TTS |
| 7 | "streaming TTS" OR "real-time speech synthesis" OR "low latency TTS" OR "low-latency speech synthesis" OR "incremental TTS" | 流式/低延迟 |
| 8 | ("diffusion" AND ("speech synthesis" OR "text to speech" OR "TTS")) OR ("flow matching" AND ("speech synthesis" OR "TTS")) OR "non-autoregressive TTS" | Diffusion/Flow/NAR |
| 9 | "dialogue TTS" OR "long-form speech synthesis" OR "conversational TTS" OR "spoken dialogue generation" | 对话/长文本 TTS |

对每一批：
- 打开 arXiv 搜索页面，设置日期范围 2025-01-01 至今
- 浏览搜索结果，记录 title / arXiv ID / date / 子方向
- 翻页直到结果不再相关（通常 3-5 页）
- 如果该批 arXiv 结果 < 5 篇，补搜 Semantic Scholar 同关键词
- 每 2 个批次输出一次中间结果表格

### 补充来源（9 个批次完成后）

在 arXiv 分批检索完成后，额外执行以下补漏：

1. **Semantic Scholar 综合补漏**:
   搜索 "text-to-speech 2025"、"speech synthesis 2025"，
   与已有结果去重后补充遗漏论文

2. **会议论文补漏**（如页面可访问）:
   - Interspeech 2025 accepted papers
   - ICASSP 2025 speech synthesis session
   - ICLR / NeurIPS / ACL 2025 中明确 TTS 相关的论文

3. **知名团队检查**（可选，如前述来源已充分则跳过）:
   检查以下团队 2025 年是否有遗漏的 TTS 工作:
   ByteDance, Microsoft, Google, NVIDIA, Meta, Alibaba DAMO,
   Tencent AI Lab, Zhejiang University, SJTU, THU

### 去重规则

我已有一批论文笔记，请在输出前对标题做排除匹配：
- 完全同名或明显同一工作（如 v1/v2）→ 排除
- 疑似相关但不确定是否同一篇 → 保留并标注"疑似已有"，不要误删

已有笔记列表：
ALLD, AudioLM, BASE TTS, BEATs, CLEAR, CoT-ST, CosyVoice, CosyVoice 2, CosyVoice 3,
DAC, Dynamic-SUPERB, EmergentTTS-Eval, Emilia, EmotionThinker, FELLE, FireRedTTS,
FireRedTTS 2, Fish-Speech, FlexiCodec, FlexiVoice, FlowDec, FunAudioLLM, GLM-TTS,
GSRM, HierSpeech++, HuBERT, IndexTTS2, LatentLM, MELLE, Make-A-Voice,
MambaVoiceCloning, MaskGCT, Mega-TTS, Mega-TTS 2, Moshi, NLLB, NVSpeech,
NaturalSpeech 2, NaturalSpeech 3, NaturalVoices, PersonaPlex, RIO, RepCodec,
SC VALL-E, SNAC, SPEAR-TTS, Seamless, Seed-TTS, Seed-VC, SiTok, SongGen,
SoundStorm, SoundStream, SpeechAlign, SpeechJudge, SpeechWorldModel, StableToken,
Step-Audio, Step-Audio 2.5, Step-Audio-EditX, SwanSphere, SwanVoice, TTSDS2,
TextrolSpeech, Tortoise TTS, UniAudio, VALL-E, VITS, VQ-VAE, VibeVoice,
WavLM, Whisper, XEUS, YourTTS, w2v-BERT, w2v-BERT 2.0, wav2vec 2.0,
RL-for-Audio-LLM, NAC Token Language Analysis, SSL Suprasegmental Analysis,
Swanbench-Speech, USM-VC, STITCH

### 每次输出格式（每 2 批一次）

**核心列表：**

| # | Title | arXiv ID | Date | Subfield | One-line Summary | Open-source | 检索来源 |
|---|-------|----------|------|----------|-----------------|-------------|---------|

**待核实/边界相关：**

| # | Title | arXiv ID | Date | 归入原因 | 为什么不确定 |
|---|-------|----------|------|---------|-------------|

其中"检索来源"填你实际访问的搜索结果 URL 或页面名称，便于我核查。
```

---

## Phase 2: 筛选与分级

```
上一步检索到了 N 篇论文（列表附上）。现在请做筛选和分级。

### 筛选标准

请根据以下维度对每篇论文打分（1-5），然后按总分排序：

| 维度 | 5 分 | 1 分 |
|------|------|------|
| TTS 相关度 | TTS 是核心贡献 | TTS 只是下游应用之一 |
| 团队/会议 | 顶会 oral/spotlight + 知名团队 | 未知团队 + 仅 arXiv preprint |
| 技术新颖性 | 提出新范式/新架构/新训练策略 | 增量改进/工程优化/参数调优 |
| 社区信号 | 已开源 / 有 demo / 来自活跃团队 / HF trending / GitHub stars 高 | 无代码 + 无项目页 + 无社区讨论 |

注意：2025 年新论文尚未积累 citation，"社区信号"不要依赖引用数，
重点看 开源情况 / 项目页 / demo / 团队活跃度 / 社区讨论。

### 分级规则

- **A 级（必读）**: 总分 ≥ 16，或在某个子方向有标志性贡献
- **B 级（推荐）**: 总分 12-15，有值得关注的技术点
- **C 级（可选）**: 总分 8-11，增量工作但有参考价值
- **排除**: 总分 < 8 或与 TTS 弱相关

**子方向保底规则**: 若某子方向全部论文总分均 < 12（即无 A/B 级），
仍保留该方向得分最高的 1-2 篇为 B 级，防止细分方向被完全刷掉。

### 输出

1. A/B/C 三级论文列表，每篇附：
   - 四维评分 (如 5/4/3/4 = 16)
   - 一句话评分理由
2. 按子方向的分布统计表
3. 与我已有 vault 的覆盖差距分析：
   - 哪些子方向我覆盖充分？
   - 哪些子方向有明显缺口？
   - 有没有新兴子方向是我 vault 完全没有的？
```

---

## Phase 3: 入库规划

```
基于 Phase 2 的 A/B 级论文，请生成入库计划。

### 对每篇 A 级论文，输出：

| 字段 | 内容 |
|------|------|
| Title | 论文标题 |
| arxiv_id | arXiv ID |
| 建议 tier | deep / enhanced-card / card |
| 优先级 | P0(本周) / P1(两周内) / P2(月内) |
| 理由 | 为什么值得这个 tier（一句话） |
| 关联已有笔记 | vault 中哪些现有笔记与它关系密切（基于主题/方法/引用关系推测，标注"推测"） |
| 预期新增概念 | 可能需要新建的概念页 |

### 对 B 级论文，只需：

| Title | arxiv_id | 建议 tier | 一句话理由 |

### 执行顺序建议

请综合以下因素给出建议的精读顺序：
1. **依赖关系**: 如果论文 B 引用/基于论文 A 的方法，先读 A
2. **概念聚类**: 同子方向的放一起读，减少上下文切换
3. **难度递进**: 同方向内先读综述/基础工作，再读最新改进
4. **与已有知识的连接**: 优先读与 vault 已有笔记关联紧密的，便于概念页扩展

输出格式：编号列表，附简短理由。
```

---

## 使用方式

### 方式 A: 在 Claude Code 中执行（推荐）
直接粘贴 Phase 1 的 prompt，Claude Code 会用 Playwright 执行实际搜索。
每 2 批结果确认后继续。三个 Phase 可以在同一个 session 完成。

### 方式 B: 给外部 LLM 使用（有幻觉风险）
如果在 ChatGPT/Claude Web 等无工具环境使用，将 Phase 1 改为：
- 去掉 Playwright 相关指令
- 加上："请基于你的训练数据列出论文，所有条目标注置信度（高/中/低），
  低置信度的单独列出供我人工核查。不确定是否存在的论文禁止放入核心列表。"
- 认识到这种方式幻觉风险较高，需要人工逐一核查 arXiv ID
- 建议配合 Connected Papers / Semantic Scholar 人工验证

### 预期产出
- Phase 1: 原始论文列表（50-150 篇，取决于搜索范围）
- Phase 2: 筛选后 A 级 10-20 篇 + B 级 20-40 篇
- Phase 3: 可直接执行的入库计划，与 paperwiki-reader skill 对接
