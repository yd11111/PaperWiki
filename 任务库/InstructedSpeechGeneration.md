---
type: task
title: "Instructed Speech Generation"
aliases: [指令式语音生成, Instruction-following TTS]
tags: [TTS, controllable, instruction-following, emotion, style]
key_approaches: ["Natural language instruction", "Fine-grained markers", "Style prompt"]
key_models: ["[[论文笔记/CosyVoice3|CosyVoice 3]]", "[[CosyVoice2]]", "[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/Seed-TTS|Seed-TTS]]", "[[论文笔记/Spark-TTS|Spark-TTS]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/Survey-ControllableTTS|Xie et al. Survey 2024]]", "[[论文笔记/DeepDubbing|DeepDubbing]]", "[[论文笔记/HD-PPT|HD-PPT]]", "[[论文笔记/Ming-UniAudio|Ming-UniAudio]]", "[[论文笔记/Audiobook-CC|Audiobook-CC]]", "[[论文笔记/VoiceSculptor|VoiceSculptor]]"]
benchmarks: ["[[CV3-Eval]]", "[[论文笔记/InstructTTSEval|InstructTTSEval]]", "[[论文笔记/UltraVoice|UltraVoice]]"]
metrics: [Style Similarity, WER, MOS, Emotion Accuracy]
status: confirmed
lifecycle: active
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---

## 问题定义

通过自然语言指令或结构化标记控制合成语音的情感、语速、音色风格、方言、角色等属性。例如:"用快乐的语气说"、"模仿机器人声音"、"用四川话说"。

与传统 TTS 的区别: 不仅克隆参考语音的特征,还能按指令生成参考中未出现的风格。

## 主流方法

1. **自然语言指令**: 将 style description 作为文本前缀输入 LLM,如 "You are Speaker A. Please talk happily."
2. **细粒度标记**: 在合成文本中插入控制标记(如 `[laughter]`、`[breath]`、`<strong>XXX</strong>`)
3. **Multi-task reward (MTR)**: 通过 DiffRO + SER/AED reward 强化情感表达

## 代表模型

- [[论文笔记/CosyVoice3|CosyVoice 3]] (2025): 100+ 种风格,5000 小时 instruction-following 数据
- CosyVoice 2 (2024): 基础指令能力
- [[论文笔记/Seed-TTS|Seed-TTS]] (2024): 通过 Speaker Fine-tuning + Instruction Fine-tuning 支持情感/expressiveness/speaking rate/style 控制;RL-SER 变体将 SER accuracy 作为 reward,emotion control accuracy 从 ICL 的 0.44 提升至 0.80 (happy) [Table 9]
- [[论文笔记/IndexTTS2|IndexTTS2]] (2025): 通过 T2E 模块实现自然语言情感控制,将 DeepSeek-R1 的情感分布预测能力蒸馏到 Qwen-3-1.7b,支持 7 种情感的 soft 混合控制
- [[论文笔记/SwanTale|SwanTale]] (ByteDance, 2026): 纯 caption(环境+说话人风格+fine-grained content)驱动,在**单条波形**里同时生成多说话人语音+环境音+局部音效,无需参考录音即可"设计声音"。InstructTTSEval ZH-APS 86.1(best)、EN-APS 84.2(tie best),但 RP(角色扮演)两语言都弱 [SwanTale Table 6];另建 SwanBench-Scene(广告/漫剧/通用场景,Mean MOS 4.22 best)评声学质量 [Table 7]。是 instruct 从"仅语音"扩展到"语音+场景音统一生成"的代表。

## 评估

### Benchmarks

- Expresso dataset: 8 种 expressive speaking styles
- CV3-Eval Emotional Voice Cloning subset
- 内部数据集: 50+ 种情感/方言/角色风格
- **InstructTTSEval** (Huang et al., 2025): 首个分层级指令遵循 benchmark,3 层任务 (APS/DSD/RP) x 12 副语言特征 x 2 语言 = 6K 测试用例,用 Gemini-as-Judge 自动评估。闭源最佳 gemini-flash EN-Avg 88.7%,开源最佳 VoxInstruct 50.4% [Table 5]

### Metrics

- Style Similarity (SIM): 风格相似度
- Emotion Accuracy: 情感分类准确率
- WER: 指令生成时的内容一致性
- MOS: 自然度

### 当前 SOTA

| 模型 | 指标 | 值 | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CosyVoice 3-1.5B | Style SIM | 81.06 | Internal Dataset | CosyVoice 3 Table 14 |
| CosyVoice 3-1.5B + DiffRO-EMO | Emotion Acc (happy, text-related) | 0.98 | CV3-Eval | CosyVoice 3 Table 9 |

- [[论文笔记/VoiceSculptor|VoiceSculptor]] (ASLP@NPU, 2026): 开源 voice design + voice clone 解耦架构,通过 CoT 属性推理 + RAG 指令增强实现 NL 驱动的细粒度语音属性控制; InstructTTSEval-Zh 开源 SOTA (AVG 67.6%, 超越 MiMo-Audio 7B 64.5%)
- [[论文笔记/MOSS-VoiceGenerator|MOSS-VoiceGenerator]] (OpenMOSS, 2026): 以影视数据为核心的 NL description voice design 模型; InstructTTSEval EN DSD 82.0% (超过 Qwen3-TTS-VD/MIMO-Audio) [Table 1]; 主观 pairwise preference 全维度优于 MIMO-Audio (63.1%) / MiniMax (61.9%) / Qwen3-TTS-VD (61.9%) [Fig 4]

### Open-Vocabulary 指令 (OV-InstructTTS)

[[论文笔记/OV-InstructTTS|OV-InstructTTS]] (Ren et al., CASIA/Tsinghua, 2026): 将 InstructTTS 指令空间从预定义声学属性的组合推向叙事上下文衍生的开放词汇指令。基于 Step-Audio-2-mini-Base LALM + reasoning chain (先推断情感/声学/副语言属性再生成),在 OV-Speech 数据集 (316K utterances from 83 novels) 上训练。Gemini Score 70.42, MOS 4.28, ICMOS 3.91 [Table 2]。

## 开放问题

- 音色(timbre)尚不可通过文本指令控制,需要额外研究
- 歌唱风格生成尚未纳入
- 指令理解的精确度: 复杂组合指令(同时控制情感+语速+方言)的效果
- 评估难题: InstructTTSEval (2025) 迈出第一步,但 Gemini-as-Judge 存在 self-preference bias (Gemini TTS 得分超过 reference audio),且仅 True/False 二分评估粒度粗
- **Spoken dialogue 场景**: UltraVoice (Tu et al., 2025) 首次将多维度风格控制从 TTS 迁移到端到端 spoken dialogue 模型 (SLAM-Omni/VocalNet),830h 六维度 (emotion/speed/volume/accent/language/composite) 数据集 SFT 后 IFR +14-40pp, MOS +29-42%,且通用对话能力同步提升 (URO-Bench +8-11%)。详见 [[论文笔记/UltraVoice|UltraVoice]]。
