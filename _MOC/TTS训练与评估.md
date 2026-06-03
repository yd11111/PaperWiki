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

### Evaluation
- [[论文笔记/SpeechJudge|SpeechJudge]] — 2025, TTS naturalness 完整评估套件, 99K pairwise + GRM (77.2% accuracy)
- [[论文笔记/EmergentTTS-Eval|EmergentTTS-Eval]] — 2025, 1645-sample benchmark + LALM-as-judge, Spearman 90.5%
- [[论文笔记/ALLD|ALLD]] — 2025, descriptive speech quality evaluation + token-level DPO distillation, MOS MSE 0.17
- [[论文笔记/TTSDS2|TTSDS2]] — 2026, 分布式 TTS 评估指标(Wasserstein-2), 唯一 ρ>0.5 客观指标, 14 语言

### Understanding & Data
- [[论文笔记/EmotionThinker|EmotionThinker]] — 2026, RL-based 可解释语音情感推理, GRPO-PTR, SER Avg 68.89%
- [[论文笔记/SpeechWorldModel|SpeechWorldModel]] — 2026, 因果图模块化语音理解(4模块DAG), EM 97.80% 超越 Gemini 2.5 Pro
- [[论文笔记/NaturalVoices|NaturalVoices]] — 2024, 大规模自发情感语音数据集(3846h, 2467 speakers)

## 相关任务
- [[Zero-shot Speech Synthesis]]

## 相关数据集
- [[SEED-TTS-Eval]] — ByteDance, 零样本 TTS 标准 benchmark
- [[Emilia]] — 101K+ hrs 多语言训练数据

## 演进脉络

```
Post-training 演进:
  SpeechAlign (2024, DPO on codec LM)
  → RL-for-Audio-LLM (2025, GRPO vs DiffRO 统一框架)
  → GSRM (2025, generative speech reward model)
  → RIO (2025, reverse inference optimization)

Evaluation 演进:
  MOS + PESQ/STOI (传统)
  → DNSMOS/NISQA (深度学习 MOS 预测)
  → SpeechJudge (2025, pairwise preference + GRM)
  → EmergentTTS-Eval (2025, LALM-as-judge)
  → ALLD (2025, descriptive evaluation + distillation)
  → TTSDS2 (2026, distributional, 14 languages, ρ>0.5)

Understanding:
  → EmotionThinker (2026, RL-based reasoning)
  → SpeechWorldModel (2026, causal graph)
  → NaturalVoices (2024, spontaneous emotion data)
```

## 历史参考

（暂无已废弃实体）
