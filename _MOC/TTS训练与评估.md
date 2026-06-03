# TTS 训练与评估

> [!note] 人工备注
> (此区域自动刷新时保留)

## 核心概念
- [[Differentiable Reward Optimization]] — Token-level 可微 reward 优化, TTS post-training
- [[TTS Evaluation]] — TTS 评估方法论 (MOS, A/B, 自动指标)
- [[Emotion Control in TTS]] — TTS 情感控制
- [[Audio Understanding]] — 音频理解, 评估相关

## 代表模型

## 相关论文 (仅 deep/repro)

### Post-training
- [[论文笔记/SpeechAlign|SpeechAlign]] — 2024, 首次偏好学习于 codec LM, golden vs synthetic, iterative DPO, WER 6.0/SIM 0.90
- [[论文笔记/RL-for-Audio-LLM|RL-for-Audio-LLM]] — 2025, 通义 RL 框架, GRPO vs DiffRO, Combined+Filter, ASR WER -5.3%
- [[论文笔记/GSRM|GSRM]] — 2025, generative speech reward model, acoustic feature+CoT, PCC 0.465, online RLHF 82% win
- [[论文笔记/RIO|RIO]] — 2025, reverse inference optimization, PPC, WER 3.4/SIM 0.96/bad case 1%
- [[论文笔记/Multi-Reward GRPO|Multi-Reward GRPO]] — 2025, 多奖励 GRPO 于单码本 TTS LLM, 5 reward (WER+SIM+len+ent+prosody), CER 1.10/MOS 4.21
- [[论文笔记/DMOSpeech 2|DMOSpeech 2]] — 2025 (AAAI 2026), component-level GRPO 靶向 duration predictor, SIM+WER reward, 1.5K RL steps, WER 1.752% en, RTF 0.032
- [[论文笔记/FPO|FPO]] — 2025, token-level 选择性 DPO, segmental error 二分类 (temporal/semantic-phonetic), 3-4x 数据效率, CosyVoice bad case 21%->9%

### Evaluation
- [[论文笔记/SpeechJudge|SpeechJudge]] — 2025, TTS naturalness 完整评估套件, 99K pairwise + GRM (77.2% accuracy)
- [[论文笔记/EmergentTTS-Eval|EmergentTTS-Eval]] — 2025, 1645-sample benchmark + LALM-as-judge, Spearman 90.5%
- [[论文笔记/ALLD|ALLD]] — 2025, descriptive speech quality evaluation + token-level DPO distillation, MOS MSE 0.17
- [[论文笔记/TTSDS|TTSDS]] — 2024, 首个分布级多因子 TTS 评估(5因子 x Wasserstein-2), 35 系统 ρ=0.60-0.83
- [[论文笔记/TTSDS2|TTSDS2]] — 2026, 分布式 TTS 评估指标(Wasserstein-2), 唯一 ρ>0.5 客观指标, 14 语言
- [[论文笔记/TTS-PRISM|TTS-PRISM]] — 2026, 12 维分层诊断框架(中文), schema-driven instruction tuning, 7B 单次推理超越 30B+ 通用模型

### Security & Traceability
- [[论文笔记/TraceableSpeech|TraceableSpeech]] — 2024, Interspeech, VALL-E+HiFiCodec 端到端水印联合训练, frame-wise broadcast, PESQ 3.641/MOS 3.959

### Emotion Control
- [[论文笔记/EmoSphere-TTS|EmoSphere-TTS]] — 2024, Interspeech, AVD 伪标签 + 球面坐标解耦风格/强度, nMOS 3.88, ECA 94.02%
- [[论文笔记/Controlling Emotion TTS NL Prompts|Controlling Emotion TTS NL Prompts]] — 2024, Interspeech, 情感文本作 NL prompt + SE block 融合 + curriculum learning, Cramer's V 0.80, MOS 3.37

### Understanding & Data
- [[论文笔记/EmotionThinker|EmotionThinker]] — 2026, RL-based 可解释语音情感推理, GRPO-PTR, SER Avg 68.89%
- [[论文笔记/SpeechWorldModel|SpeechWorldModel]] — 2026, 因果图模块化语音理解(4模块DAG), EM 97.80% 超越 Gemini 2.5 Pro
- [[论文笔记/NaturalVoices|NaturalVoices]] — 2024, 大规模自发情感语音数据集(3846h, 2467 speakers)
- [[论文笔记/TITW|TITW]] — 2024, 首批标准化 noisy-TTS 训练数据集(VoxCeleb1→TITW-Easy 173h/Hard 189h), DNSMOS 过滤 pipeline + KSKT/KSUT 评估协议
- [[论文笔记/StoryTTS|StoryTTS]] — 2024, 61h 中文评书表现力数据集, LLM 五维度文本表现力标注, MOS 4.09

## 相关任务
- [[Zero-shot Speech Synthesis]]

## 相关数据集
- [[SEED-TTS-Eval]] — ByteDance, 零样本 TTS 标准 benchmark
- [[Emilia]] — 101K+ hrs 多语言训练数据

## 演进脉络

```
Post-training 演进:
  SpeechAlign (2024, DPO on codec LM)
  → FPO (2025, token-level selective DPO for TTS)
  → RL-for-Audio-LLM (2025, GRPO vs DiffRO 统一框架)
  → GSRM (2025, generative speech reward model)
  → RIO (2025, reverse inference optimization)
  → Multi-Reward GRPO (2025, 多奖励 GRPO for single-codebook TTS)
  → DMOSpeech 2 (2025, component-level GRPO 靶向 duration predictor)

Evaluation 演进:
  MOS + PESQ/STOI (传统)
  → DNSMOS/NISQA (深度学习 MOS 预测)
  → TTSDS (2024, 首个分布级多因子评估, 5因子, ρ=0.60-0.83)
  → SpeechJudge (2025, pairwise preference + GRM)
  → EmergentTTS-Eval (2025, LALM-as-judge)
  → ALLD (2025, descriptive evaluation + distillation)
  → TTSDS2 (2026, distributional 升级, 14 languages, ρ>0.5)
  → TTS-PRISM (2026, 12-dim diagnostic, schema-driven, Mandarin)

Understanding:
  → EmotionThinker (2026, RL-based reasoning)
  → SpeechWorldModel (2026, causal graph)
  → NaturalVoices (2024, spontaneous emotion data)
```

## 历史参考

（暂无已废弃实体）
