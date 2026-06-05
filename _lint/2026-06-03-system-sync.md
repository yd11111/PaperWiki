# 全面系统同步报告 — 2026-06-03

## 总览

| 指标 | 数值 |
|------|------|
| 论文笔记 | 94 篇 (65 deep + 2 repro + 1 enhanced-card + 12 card + 14 无 tier) |
| 实体页 | 92 个 (62 概念 + 20 模型 + 4 任务 + 6 数据集) |
| MOC | 3 个 (TTS-总览 / 语音编码与量化 / 零样本语音合成) |
| 审阅报告 | 10 个 |
| 可信层 | 22 confirmed 实体 + 1 reviewed 笔记 (CosyVoice 3) |

---

## 1. MOC 覆盖 — FAIL (48/67 缺失)

仅 19/67 deep/repro 笔记被 MOC 收录。以下 48 篇未被任何 MOC 包含:

| 类别 | 未收录笔记 |
|------|-----------|
| 基础 TTS | BASE TTS, Fish-Speech, GLM-TTS, HierSpeech++, MELLE, Make-A-Voice, NaturalSpeech 2, NaturalSpeech 3, SC VALL-E, VALL-E, VITS, YourTTS |
| 语音克隆/转换 | MambaVoiceCloning, Seed-VC, USM-VC, TextrolSpeech |
| 多模态/对话 | FlexiVoice, FunAudioLLM, Moshi, PersonaPlex, Step-Audio, Step-Audio 2.5, Step-Audio-EditX, NVSpeech |
| Codec/Token | FlexiCodec, StableToken, STITCH, NAC Token Language Analysis |
| SSL/表征 | BEATs, Dynamic-SUPERB, HuBERT, SSL Suprasegmental Analysis, WavLM, Whisper, XEUS, w2v-BERT, w2v-BERT 2.0, wav2vec 2.0 |
| 对齐/评估 | SpeechAlign, SpeechJudge, RL-for-Audio-LLM, GSRM, RIO |
| 其他 | FireRedTTS, FireRedTTS 2, Mega-TTS 2, SoundStorm, UniAudio, Emilia, SongGen |

**建议**: 需要创建新 sub-MOC (如 "语音表征与SSL"、"语音大模型与对话"、"TTS 模型演进") 来覆盖这些论文，或扩展现有 MOC。

---

## 2. 审阅覆盖 — FAIL (56/67 缺失)

仅 11/67 deep/repro 笔记有 `[!review]` callout:

**已审阅**: BASE TTS, CosyVoice, CosyVoice 3, DAC, IndexTTS2, MaskGCT, Mega-TTS 2, NaturalSpeech 2, Seed-TTS, SoundStream, VALL-E

**缺失审阅**: 其余 56 篇 deep/repro 笔记均无审阅 callout。

---

## 3. Frontmatter 一致性 — PASS

- 所有 65 deep + 2 repro 笔记的 title/tier/status/tags 字段齐全
- 所有 92 个实体页的 title/status/lifecycle/tags 字段齐全

---

## 4. 死链 — 7 个

| 死链 | 来源文件 | 原因/建议 |
|------|---------|----------|
| `[[W2v-BERT 2.0]]` | MaskGCT.md | 大小写不一致，应为 `[[w2v-BERT2.0]]` |
| `[[数据集/AISHELL-3]]` | CosyVoice.md | 数据集页不存在，需创建或改为纯文本 |
| `[[数据集/Common Voice]]` | CosyVoice.md | 同上 |
| `[[数据集/LibriSpeech]]` | CosyVoice.md, CosyVoice 2.md, NaturalSpeech 3.md | 同上 |
| `[[数据集/LibriTTS]]` | CosyVoice.md | 同上 |
| `[[数据集/MLS]]` | CosyVoice.md | 同上 |
| `[[模型库/DAC]]` | FlexiCodec.md | DAC 在论文笔记/不在模型库/，应改为 `[[论文笔记/DAC\|DAC]]` |

---

## 5. 孤儿页 — 1 个

| 页面 | 说明 |
|------|------|
| 模型库/SenseVoice.md | 无任何入链 |

---

## 6. 概念去重 — PASS (无明显重叠)

62 个概念页 title 无明显语义重复。但所有概念页的 aliases 字段均为空 — 后续可补充常见别名以提升 KB 检索命中率。

---

## 7. 可信层进度

| 层 | 数量 | 占比 |
|----|------|------|
| confirmed 实体 | 22/92 | 23.9% |
| pending-review 实体 | 70/92 | 76.1% |
| reviewed 笔记 | 1/67 | 1.5% |
| draft deep/repro 笔记 | 66/67 | 98.5% |

**Review backlog alert**: 70 个 pending-review 实体页 (阈值 10)，66 个 draft deep/repro 笔记 (阈值 5)。

---

## 8. 概念溯源 — 10 个待补

以下概念页 key_papers ≥ 3 但缺 origin_paper:

| 概念 | key_papers 数 |
|------|-------------|
| Speaker Verification | 30 |
| Anti-spoofing and Deepfake Detection | 28 |
| Speech Tokenizer | 26 |
| Codebook Collapse | 22 |
| Mel Spectrogram | 17 |
| Speaker Adaptation | 15 |
| Conditional Flow Matching | 15 |
| Differentiable Reward Optimization | 13 |
| Multi-scale STFT Discriminator | 13 |
| Gradient Reversal Layer | 11 |

---

## 9. 系统文档 — 需更新

CLAUDE.md 中 vault 状态需更新:
- 论文笔记: 92 → **94** 篇
- 新增 14 篇无 tier 笔记 (ALLD, CoT-ST, EmergentTTS-Eval, EmotionThinker, FELLE, FlowDec, NLLB, NaturalVoices, Seamless, SiTok, SpeechWorldModel, Survey-Controllable TTS, TTSDS2, Tortoise TTS)

---

## 10. 无 tier 笔记 — 14 篇

以下笔记缺少 `tier` 字段，无法被分层体系管理:

ALLD, CoT-ST, EmergentTTS-Eval, EmotionThinker, FELLE, FlowDec, NLLB, NaturalVoices, Seamless, SiTok, SpeechWorldModel, Survey-Controllable TTS, TTSDS2, Tortoise TTS

---

## 优先行动建议

1. **[高] MOC 扩展** — 创建 2-3 个新 sub-MOC 覆盖 SSL/表征、语音大模型/对话、TTS 模型演进
2. **[高] 死链修复** — 7 个死链需修复 (1 个大小写、5 个缺数据集页、1 个路径错误)
3. **[中] 审阅补齐** — 56 篇 deep/repro 缺审阅，可分批处理
4. **[中] 无 tier 笔记** — 14 篇需补 tier 字段
5. **[低] 概念溯源** — 10 个概念页可补 origin_paper
6. **[低] Aliases 补充** — 62 个概念页 aliases 均为空
7. **[信息] CLAUDE.md 更新** — vault 状态数字过时
