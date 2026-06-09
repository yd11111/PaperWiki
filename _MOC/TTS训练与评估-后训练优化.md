# TTS 训练与评估-后训练优化

<!-- scope: 本页覆盖 TTS 后训练优化方法 (DPO/GRPO/RL/蒸馏/偏好对齐/reward model)。不包含: TTS 评估指标/benchmark(归评估与基准), 纯模型架构(归零样本语音合成)。 -->

> [!note] 人工备注
> (此区域自动刷新时保留)

## 相关论文

- [[论文笔记/SpeechAlign|SpeechAlign]] — 2024, 偏好学习于 codec LM (作者 claim 首次), golden vs synthetic, iterative DPO, WER 6.0/SIM 0.90
- [[论文笔记/DMOSpeech|DMOSpeech]] — 2024, Columbia+Adobe, DMD2 蒸馏 + 端到端 CTC+SV loss 优化 (作者 claim 首次, non-RL), student 超越 teacher, SIM 0.69, WER 1.94, RTF 0.07
- [[论文笔记/PreferenceAlignment|Preference Alignment Improves LM-Based TTS]] — 2024, 1.15B LM-based TTS 上系统验证 DPO 最佳实践, 1 小时数据显著提升
- [[论文笔记/DiffRO|DiffRO]] — 2025, 通义, DiffRO 原始论文, Token2Reward+Gumbel-Softmax+MTR, WER 1.56→0.78 (zh), 零样本情感控制
- [[论文笔记/RL-for-Audio-LLM|RL-for-Audio-LLM]] — 2025, 通义 RL 框架, GRPO vs DiffRO, Combined+Filter, ASR WER -5.3%
- [[论文笔记/GSRM|GSRM]] — 2025, generative speech reward model, acoustic feature+CoT, PCC 0.465, online RLHF 82% win
- [[论文笔记/RIO|RIO]] — 2025, reverse inference optimization, PPC, WER 3.4/SIM 0.96/bad case 1%
- [[论文笔记/Multi-RewardGRPO|Multi-Reward GRPO]] — 2025, 多奖励 GRPO 于单码本 TTS LLM, 5 reward (WER+SIM+len+ent+prosody), CER 1.10/MOS 4.21
- [[论文笔记/DMOSpeech2|DMOSpeech 2]] — 2025 (AAAI 2026), component-level GRPO 靶向 duration predictor, SIM+WER reward, 1.5K RL steps, WER 1.752% en, RTF 0.032
- [[论文笔记/FPO|FPO]] — 2025, NWPU, token-level 选择性 DPO, segmental error 二分类 (temporal/semantic-phonetic), 3-4x 数据效率, CosyVoice bad case 21%->9%
- [[论文笔记/MPO|MPO]] — 2025, NWPU, multidimensional preference set (各维度独立选极值) + CE loss 正则化防 DPO 退化, CER 3.90/SIM 0.577, ABX 52.3% win vs baseline
- [[论文笔记/Koel-TTS|Koel-TTS]] — 2025, NVIDIA, ASR+SV reward 驱动 DPO/RPO + Pareto 最优多目标偏好配对, CER 0.55% (benchmark: LibriTTS unseen), GT-as-Chosen 失败实验
- [[论文笔记/F5R-TTS|F5R-TTS]] — 2025, Tencent, 将 GRPO 集成到 NAR flow-matching TTS (作者 claim 首次), output probabilization 使 CFM 兼容 RL, WER -29.5% / SIM +4.6%
- [[论文笔记/DLPO|DLPO]] — 2025, OSU, RLHF 微调 diffusion TTS (WaveGrad 2), diffusion loss 作为 reward 正则项, UTMOS 3.65/NISQA 4.02, 67% 人类偏好
- [[论文笔记/LatinX|LatinX]] — 2025, USP, utterance-level DPO + Pareto dominance 多指标偏好标注, 多语言 TTS 对齐, 揭示客观-主观 speaker similarity gap
- [[论文笔记/ARDM-DPO|ARDM-DPO]] — 2025, 将 DPO 扩展到自回归扩散模型 (ARDM) (作者 claim 首次), DiTAR 上 F0 方差翻倍 + CER 降低 25%
- [[论文笔记/NoVerifiableRewardforProsody|No Verifiable Reward for Prosody]] — 2025, GRPO CER/NLL reward 导致韵律坍缩, iterative DPO ~200 人类偏好对/轮即可恢复自然韵律
- [[论文笔记/RLAIF-SPA|RLAIF-SPA]] — 2025, GRPO 优化结构化 AI 反馈 (4 维韵律-情感标签匹配 + WER 惩罚), 无需人工情感标注
- [[论文笔记/TKTO|TKTO]] — 2025, contrastive LLMs 估计 token-level importance + KTO token-level 推广, 消除 speech tokenizer-LM 对齐要求
- [[论文笔记/Align2Speak|Align2Speak]] — 2025, GRPO 在线 RL + ASR/SV/PESQ 多目标 reward, 30 分钟配对数据下 CER 降低 8 倍以上, 低资源语言 TTS
- [[论文笔记/GRPO-TTS|GRPO-TTS]] — 2025, Whisper CER + NLL 复合 reward + GRPO 微调 CosyVoice2/Llasa-1B, 无需额外训练模型
- [[论文笔记/RRPO|RRPO]] — 2025, DiffRO 框架中三层混合正则化 (LS + EAM + Adv) 构建鲁棒 Reward Model
- [[论文笔记/Step-Audio-AQAA|Step-Audio-AQAA]] — 2025, 130B 端到端 LALM, 双码本 tokenizer + tri-codebook interleaved output + masked DPO + 三模型权重合并
- [[论文笔记/Step-Audio-R1|Step-Audio-R1]] — 2025, 首个成功让音频语言模型受益于 extended reasoning, Modality-Grounded Reasoning Distillation (MGRD)
- [[论文笔记/W3AR|W3AR]] — 2025, frozen Whisper cross-attention map 提取 word-level attention purity 和 alignment 作为 fine-grained reward
- [[论文笔记/video-SALMONN-o1|video-SALMONN-o1]] — 2025, 首个开源推理增强音视频 LLM, process DPO (pDPO) + contrastive step selection
- [[论文笔记/video-SALMONN2|video-SALMONN 2]] — 2025, MrDPO (multi-round DPO + LoRA proxy + gDPO loss) 优化视频 captioning
- [[论文笔记/BridgingAudio-TextGap|CORD]] — 2026, on-policy 跨模态蒸馏 (token 级加权 KL + 序列级 GRPO) 弥合 LALM 音频-文本推理差距
- [[论文笔记/ConversationalTTS-RL|ConversationalTTS-RL]] — 2026, cascaded prompting + ICL-based online RL 增强对话 TTS
- [[论文笔记/CosyEdit2|CosyEdit2]] — 2026, CosyVoice2 上两阶段 post-training (SFT → editing-oriented GRPO)
- [[论文笔记/EditContentPreserveAcoustics|EditContentPreserveAcoustics]] — 2026, text-based speech editing 迁移到 semantic token 空间 + self-consistency rewards
- [[论文笔记/MimicLM|MimicLM]] — 2026, "role-swapping" 数据构造 + interleaved text-audio 建模 + DPO 对齐实现零样本语音模仿
- [[论文笔记/Step-Audio-R1.5|Step-Audio-R1.5]] — 2026, 提出 "verifiable reward trap" 概念, RLVR 系统性损害音频模型对话自然度
- [[论文笔记/TARS|TARS]] — 2026, on-policy RL + layer-wise representation alignment + output-level behavior 闭合 Speech LLM 模态推理差距

## 演进脉络

```
Post-training 演进:
  SpeechAlign (2024, DPO on codec LM)
  → PreferenceAlignment (2024, DPO best practices for LM-based TTS)
  → MPO (2025, multidimensional preference set + CE regularization)
  → FPO (2025, token-level selective DPO for TTS)
  → DiffRO (2025, Token2Reward + Gumbel-Softmax + MTR 原始论文)
  → RRPO (2025, 鲁棒 reward via 三层正则化)
  → RL-for-Audio-LLM (2025, GRPO vs DiffRO 统一框架)
  → GSRM (2025, generative speech reward model)
  → RIO (2025, reverse inference optimization)
  → Multi-Reward GRPO (2025, 多奖励 GRPO for single-codebook TTS)
  → GRPO-TTS (2025, Whisper CER + NLL reward, 轻量级 GRPO)
  → Align2Speak (2025, 低资源 GRPO + 多目标 reward)
  → W3AR (2025, Whisper cross-attention word-level reward)
  DMOSpeech (2024, 端到端 CTC+SV direct metric optimization (作者 claim 首次), non-RL, DMD2 蒸馏打通梯度通路)
    → DMOSpeech 2 (2025, component-level GRPO 靶向 duration predictor)
  → F5R-TTS (2025, GRPO for NAR flow-matching TTS via output probabilization, RL 集成 NAR (作者 claim 首次))
  → DLPO (2025, RLHF for diffusion TTS)
  → ARDM-DPO (2025, DPO for autoregressive diffusion)
  → NoVerifiableRewardforProsody (2025, 韵律坍缩诊断 + iterative DPO 修复)
  → CosyEdit2 (2026, editing-oriented GRPO)
  → ConversationalTTS-RL (2026, ICL-based online RL for dialogue TTS)
  → EditContentPreserveAcoustics (2026, self-consistency rewards for speech editing)
  → Step-Audio-R1.5 (2026, "verifiable reward trap" 分析)

Token-level DPO/KTO 分支:
  TKTO (2025, token-level KTO) → FPO (2025, token-level selective DPO)

多模态 DPO/RL 分支:
  Step-Audio-AQAA (2025, masked DPO for LALM)
  → Step-Audio-R1 (2025, reasoning distillation for audio LM)
  → video-SALMONN-o1 (2025, process DPO for av-LLM)
  → video-SALMONN2 (2025, multi-round DPO for video captioning)
  → BridgingAudio-TextGap (2026, on-policy cross-modal GRPO distillation)
  → TARS (2026, on-policy RL for speech LLM reasoning)
  → MimicLM (2026, DPO for voice imitation)
```
