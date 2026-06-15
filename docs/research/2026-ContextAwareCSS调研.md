# Context-Aware Conversational TTS 调研报告

> 调研日期: 2026-06-09
> 目的: 2026 下半年规划 — 对话场景下上下文驱动的隐式风格推断 TTS

---

## 一、方向定义

**核心问题**: 给定对话上下文（多轮文本+可选音频历史），模型自动推断每句话合适的情感、语气、韵律表达，无需用户逐句给 style instruction。

**与相邻方向的区分**:
- vs Instruct TTS: 不需要 per-utterance 显式指令，context is the instruction
- vs Full-duplex/Omni: 不追求端到端实时，聚焦 context → style inference 的可解释环节
- vs Emotion TTS: 不是固定 N 类分类，而是连续的、上下文决定的表达空间

---

## 二、技术路线全景（4 种 Pattern）

### Pattern A: LLM 作为外部风格推断器
对话上下文 → LLM 推断 emotion/style → TTS 条件渲染

代表:
- [[论文笔记/ConversationalTTS-RL|ConversationalTTS-RL]] (2026): cascaded prompting + online RL
- [[论文笔记/CapTalk|CapTalk]] (2026): CoT planning 逐 turn 规划 emotion/tone/pitch/energy/speed
- [[论文笔记/ChatGPT-EDSS|ChatGPT-EDSS]] (2023): ChatGPT 提取意图/情感词
- Chain-Talker (2025, ACL): 三阶段认知模拟
- JELLY (2025, ICASSP): LLM + multi-LoRA

### Pattern B: 专用上下文编码器
对话历史 → 图/Transformer 编码器 → style vector → TTS decoder

代表:
- [[论文笔记/DiffCSS|DiffCSS]] (2025): Diffusion 多样性采样
- [[论文笔记/RADKA-CSS|RADKA-CSS]] (2025): RAG + 异构图
- ECSS (2023, AAAI 2024): 异构图 + 对比学习
- MFCIG-CSS (2025, EMNLP): 词级多模态交互图
- [[论文笔记/TextAwareContextAwareTTS|TextAwareContextAwareTTS]] (2024): CLIP 对比学习

### Pattern C: 端到端隐式（Speech LLM 自身学会）
多轮音频 token → 大模型 → 自动带风格的语音 token

代表:
- SASLM (2026, EMNLP sub): VIB 自蒸馏 + self-reward
- [[论文笔记/Moshi|Moshi]] (2024): full-duplex 多流
- [[论文笔记/CSM|CSM]] (2025): Sesame 对话模型
- [[论文笔记/VibeVoice|VibeVoice]] (2025): 90min 长对话
- GPT-4o Voice Mode

### Pattern D: RAG 增强
当前对话 + 检索相似历史对话 → 聚合风格参考

代表:
- [[论文笔记/RADKA-CSS|RADKA-CSS]] (2025)
- [[论文笔记/AutoStyle-TTS|AutoStyle-TTS]] (2025)

---

## 三、核心论文清单

### 已在 KB 中（9 篇直接相关）

| 论文 | 年份 | Pattern | 核心做法 |
|------|------|---------|----------|
| [[论文笔记/ConversationalTTS-RL|ConversationalTTS-RL]] | 2026 | A | LLM cascaded prompting + AR prosody + RL |
| [[论文笔记/CapTalk|CapTalk]] | 2026 | A | CoT 规划 turn-level 属性 |
| [[论文笔记/DiffCSS|DiffCSS]] | 2025 | B | Diffusion 多样 prosody 采样 |
| [[论文笔记/RADKA-CSS|RADKA-CSS]] | 2025 | B+D | RAG 检索 + 异构图聚合 |
| [[论文笔记/DialoSpeech|DialoSpeech]] | 2025 | C | 双轨 LLM + chunked CFM |
| [[论文笔记/TextAwareContextAwareTTS|TextAwareContextAwareTTS]] | 2024 | B | CLIP 对比学习 |
| [[论文笔记/ChatGPT-EDSS|ChatGPT-EDSS]] | 2023 | A | ChatGPT 提取风格词 |
| [[论文笔记/JointDialogueSpeech|JointDialogueSpeech]] | 2023 | A | LLM 同时生成回复+韵律 |
| [[论文笔记/EmotionAwareProsodic|EmotionAwareProsodic]] | 2023 | B | 情感→短语断句→TTS |

### 待入库（6 篇）

| 论文 | arXiv | 年份/会议 | Pattern | 核心创新 |
|------|-------|-----------|---------|----------|
| SASLM | 2604.11424 | 2026/EMNLP sub | C | 范式转换: VIB 自蒸馏表达意图, self-reward 对齐 |
| Chain-Talker | 2505.12597 | 2025/ACL Findings | A | 三阶段认知: 情感理解→语义理解→共情渲染 |
| MFCIG-CSS | 2509.06074 | 2025/EMNLP | B | 首次词级跨模态交互图 |
| JELLY | 2501.04904 | 2025/ICASSP | A | LLM + multi-LoRA 联合情感+推理 |
| GPT-Talker | 2407.21491 | 2024/ACM MM | A+C | GPT 预测 semantic+style token; NCSSD 236h |
| ECSS | 2312.11947 | 2023/AAAI 2024 | B | 异构图 + 对比学习; DailyTalk 情感标注 |

---

## 四、数据集 & 评估

| 数据集 | 规模 | 语言 | 特点 |
|--------|------|------|------|
| DailyTalk | ~20h | EN | CSS 标准 benchmark，表演性对话 |
| NCSSD | 236h | ZH+EN | 自然对话(即兴+影视) |
| Open-Source Full-Duplex | 15h | ZH+EN | 双轨录制，重叠/反馈/笑声 |

| 评估框架 | 来源 | 侧重 |
|----------|------|------|
| EchoMind | ICLR 2026 | 共情 SLM: 语音感知→推理→生成 |
| FD-Bench | Interspeech 2025 | Full-duplex 系统 |
| 标准 MOS | 通用 | 自然度+表达力+上下文一致性 |

**关键评估 Gap**: 无标准化 "contextual appropriateness" 指标。

---

## 五、竞争格局

| 团队 | 单位 | 代表作 | 路线 |
|------|------|--------|------|
| Rui Liu + Haizhou Li | 内蒙古/NUS/CUHK-SZ | FCTalker→ECSS→GPT-Talker→RADKA-CSS→MFCIG-CSS→Chain-Talker | 图编码+多模态 |
| Kuang Wang + Haizhou Li | 上海师大+阿里达摩 | SASLM | LLM self-aware |
| 吴志勇组 | 清华 | DiffCSS | 扩散多样性 |
| 李承桓组 | Korea Univ | JELLY | LLM+LoRA |
| Lei Xie | 西北工大 | Controllable CSS, FireRedTTS-2 | 自发性语音 |

---

## 六、研究 Gap & 定位机会

| Gap | 说明 | 潜在切入 |
|-----|------|----------|
| 多轮情感轨迹建模 | 现有逐句推断,无对话级情感弧线 | Emotion trajectory planning |
| 音频上下文利用不足 | 多数只用文本,音频仅 utterance-level | 前几轮实际韵律指导下一轮 |
| 流式实时 | CSS offline, full-duplex 隐式不可控 | Streaming context-aware style |
| 个性化 | 无用户适应 | User-adaptive style policy |
| 数据瓶颈 | DailyTalk 太小太假 | 真实对话大规模数据集 |
| 多方对话 | 全假设两人 | 多人会议/群聊 |
| Diversity vs Control | DiffCSS 多样但不可选 | 可控多样性采样 |
| 评估缺失 | 无标准化 contextual appropriateness | 新评估协议 |

---

## 七、建议定位

**最有区分度**: Pattern C 升级版 — 在 Speech LLM 上做显式的 context-aware style self-inference

差异化要素:
1. 多轮 emotion trajectory（不只当前句）
2. 利用前几轮的实际语音韵律作为 context signal
3. 支持流式推理
4. EchoMind benchmark 作为评估目标

数据侧: 构建真实场景、多情感的对话数据集（非表演性）本身即贡献。
