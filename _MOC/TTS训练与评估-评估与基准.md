# TTS 训练与评估-评估与基准

<!-- scope: 本页覆盖 TTS/语音评估方法、基准测试、质量指标、安全评估和负责任 TTS。不包含: 后训练优化方法(归后训练优化), 纯 TTS 生成模型(归零样本语音合成)。 -->

> [!note] 人工备注
> (此区域自动刷新时保留)

## 相关论文

- [[论文笔记/TTSDS|TTSDS]] — 2024, 分布级多因子 TTS 评估 (作者 claim 首个)(5因子 x Wasserstein-2), 35 系统 ρ=0.60-0.83
- [[论文笔记/TraceableSpeech|TraceableSpeech]] — 2024, Interspeech, VALL-E+HiFiCodec 端到端水印联合训练, frame-wise broadcast, PESQ 3.641/MOS 3.959
- [[论文笔记/HUME-VOCALBURST|HUME-VOCALBURST]] — 2024, NeurIPS 2023 MLA Workshop, 四个情感/副语言音频数据集 (HUME-PROSODY, HUME-VOCALBURST)
- [[论文笔记/SpeechJudge|SpeechJudge]] — 2025, TTS naturalness 完整评估套件, 99K pairwise + GRM (77.2% accuracy)
- [[论文笔记/EmergentTTS-Eval|EmergentTTS-Eval]] — 2025, 1645-sample benchmark + LALM-as-judge, Spearman 90.5%
- [[论文笔记/ALLD|ALLD]] — 2025, descriptive speech quality evaluation + token-level DPO distillation, MOS MSE 0.17
- [[论文笔记/InstructTTSEval|InstructTTSEval]] — 2025, 指令遵循 TTS benchmark (作者 claim 首个), 3 层任务(APS/DSD/RP) x 12 特征 x 2 语言, Gemini-as-Judge
- [[论文笔记/AudioMOSChallenge2025|AudioMOS Challenge 2025]] — 2025, 语音/音乐/通用音频自动 MOS 预测挑战赛 (作者 claim 首个), 3 赛道(TTM/Audiobox/多采样率), 24 队, SSL+集成为主流方案
- [[论文笔记/Vox-Evaluator|Vox-Evaluator]] — 2025, 统一多级评估器 (error localization + transcription + quality score), 驱动 inference-time 语音纠正 + DPO 偏好对齐
- [[论文笔记/ProsodyEval|ProsodyEval]] — 2025, DS-WED (semantic token 加权编辑距离韵律多样性指标) + 人类标注韵律多样性 benchmark (作者 claim 首个)
- [[论文笔记/QualiSpeech|QualiSpeech]] — 2025, 首个覆盖合成+真实语音的 11 维低级语音质量评估数据集, 含自然语言描述和推理
- [[论文笔记/VCBBench|VCB Bench]] — 2025, 首个基于全真人录音的中文语音对话 LALM 评估基准, 三维度评估
- [[论文笔记/TTSDS2|TTSDS2]] — 2026, 分布式 TTS 评估指标(Wasserstein-2), ρ>0.5 客观指标 (作者 claim 唯一), 14 语言
- [[论文笔记/TTS-PRISM|TTS-PRISM]] — 2026, 12 维分层诊断框架(中文), schema-driven instruction tuning, 7B 单次推理超越 30B+ 通用模型
- [[论文笔记/AnyAudio-Judge|AnyAudio-Judge]] — 2026, dynamic rubric-based evaluation, 将复杂音频指令分解为可验证的二值 rubric items
- [[论文笔记/EntityBindingFailures|Entity Binding Failures]] — 2026, 揭示 Speech LLM 的 S2T/T2T modality gap 集中于 entity tracking 的逻辑推理任务
- [[论文笔记/HowOpenIsOpenTTS|How Open is Open TTS?]] — 2026, 系统性评估四个开源 TTS 在 Romanian 低资源场景下的可用性与合成质量
- [[论文笔记/IterateDifferentiate|Iterate to Differentiate]] — 2026, 递归迭代合成放大模型间性能差距, 恢复客观指标对 SOTA TTS 的区分力
- [[论文笔记/MCLP|MCLP]] — 2026, LALM continuation likelihood 量化风格一致性的可解释指标
- [[论文笔记/MINT-Bench|MINT-Bench]] — 2026, 首个结构化多语言 instruction-following TTS benchmark, 分层 taxonomy + 层级混合评估
- [[论文笔记/SALMONN-Guard|SALMONN-Guard]] — 2026, SACRED-Bench (语音-音频组合黑盒攻击 benchmark) + SALMONN-Guard (联合检查语音+音频+文本)
- [[论文笔记/Survey-ResponsibleTTSEvaluation|Towards Responsible Evaluation for TTS]] — 2026, 首篇 TTS 评估 position paper, 三层 Responsible Evaluation 框架
- [[论文笔记/UniSRM|UniSRM]] — 2026, 基于 Qwen2.5-Omni-7B 的统一语音 reward model, SFT+RCR-GRPO 两阶段, 4 种评估任务
- [[论文笔记/VoxPrivacy|VoxPrivacy]] — 2026, 首个评估 SLM 在多说话人环境中保护上下文敏感信息能力的三层 benchmark
- [[论文笔记/VoxSafeBench|VoxSafeBench]] — 2026, 首个联合评估 SLM 安全/公平/隐私的 benchmark, Two-Tier 设计分离内容风险与语音上下文风险
- [[论文笔记/WildASR|WildASR]] — 2026, 多语言 ASR 诊断 benchmark, 真实人声 + 受控扰动沿三轴因子隔离评估
- [[论文笔记/WhenDoSpeechLLMsBehaveLikeASR-LLMPipelines|Cascade Equivalence Hypothesis]] — 2026, matched-backbone 行为测试 + logit lens/LEACE 机制分析证明 Speech LLM 行为等价性

## 演进脉络

```
Evaluation 演进:
  MOS + PESQ/STOI (传统)
  → DNSMOS/NISQA (深度学习 MOS 预测)
  → TTSDS (2024, 分布级多因子评估 (作者 claim 首个), 5因子, ρ=0.60-0.83)
  → SpeechJudge (2025, pairwise preference + GRM)
  → EmergentTTS-Eval (2025, LALM-as-judge)
  → ALLD (2025, descriptive evaluation + distillation)
  → QualiSpeech (2025, 11 维低级质量评估 + NL 描述)
  → ProsodyEval (2025, semantic-token 韵律多样性指标)
  → VCB Bench (2025, 中文语音对话评估)
  → TTSDS2 (2026, distributional 升级, 14 languages, ρ>0.5)
  → TTS-PRISM (2026, 12-dim diagnostic, schema-driven, Mandarin)
  → IterateDifferentiate (2026, 迭代合成恢复指标区分力)
  → UniSRM (2026, 统一 reward model for assessment)
  → MCLP (2026, continuation likelihood 风格一致性)

Benchmark 演进:
  InstructTTSEval (2025, instruction-following benchmark, Gemini-as-Judge)
  → AudioMOS Challenge 2025 (2025, MOS 预测从 speech 扩展到 music/general audio)
  → AnyAudio-Judge (2026, dynamic rubric-based evaluation)
  → MINT-Bench (2026, multilingual instruction-following benchmark)
  → WildASR (2026, 多语言 ASR 诊断 benchmark)

Safety & Responsible:
  TraceableSpeech (2024, 端到端水印)
  → SALMONN-Guard (2026, 语音-音频组合攻击 + guard)
  → VoxPrivacy (2026, 交互隐私 benchmark)
  → VoxSafeBench (2026, 安全/公平/隐私联合 benchmark)
  → Survey-ResponsibleTTSEvaluation (2026, responsible evaluation position paper)

Modality Gap Analysis:
  EntityBindingFailures (2026, entity binding 诊断)
  → WhenDoSpeechLLMsBehaveLikeASR (2026, cascade equivalence hypothesis)
```
