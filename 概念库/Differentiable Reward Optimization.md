---
type: concept
title: "Differentiable Reward Optimization"
aliases: [DiffRO]
category: "training-strategy"
tags: [reinforcement-learning, post-training, TTS, reward-model]
key_papers: ["[[论文笔记/DiffRO|DiffRO]]", "[[论文笔记/CosyVoice 3|CosyVoice 3]]", "[[论文笔记/GLM-TTS|GLM-TTS]]", "[[论文笔记/RL-for-Audio-LLM|RL-for-Audio-LLM]]", "[[论文笔记/SpeechAlign|SpeechAlign]]", "[[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]", "[[论文笔记/Multi-Reward GRPO|Multi-Reward GRPO]]", "[[论文笔记/Fish Audio S2|Fish Audio S2]]", "[[论文笔记/DMOSpeech 2|DMOSpeech 2]]", "[[论文笔记/DMOSpeech|DMOSpeech]]", "[[论文笔记/FPO|FPO]]", "[[论文笔记/FlexSpeech|FlexSpeech]]", "[[论文笔记/Koel-TTS|Koel-TTS]]", "[[论文笔记/F5R-TTS|F5R-TTS]]", "[[论文笔记/MPO|MPO]]", "[[论文笔记/DLPO|DLPO]]", "[[论文笔记/LatinX|LatinX]]", "[[论文笔记/TTS-1|TTS-1 (Inworld, 2025)]]", "[[论文笔记/TKTO|TKTO]]", "[[论文笔记/No Verifiable Reward for Prosody|No Verifiable Reward for Prosody]]", "[[论文笔记/RLAIF-SPA|RLAIF-SPA]]", "[[论文笔记/Vox-Evaluator|Vox-Evaluator]]"]
origin_paper: ""
related_concepts: ["[[Gumbel-Softmax]]", "[[Speech Tokenizer]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Differentiable Reward Optimization (DiffRO) 是 CosyVoice 3 提出的一种适用于 TTS 系统的 post-training 方法。其核心思想是: 在离散 speech token 层面直接计算 reward 并反向传播梯度,而非在最终音频层面做强化学习。

关键组件:
1. **Token2Text Reward Model**: 一个类 ASR 模型,输入 speech token 输出文本后验概率,作为内容一致性的 reward
2. **Gumbel-Softmax 采样**: 使 LLM 输出的离散 token 选择可微,梯度可回传
3. **Token-level KL 约束**: 在每个时间步的 token logits 上计算 KL 散度(而非 sequence-level),防止策略偏离参考模型
4. **Multi-task Reward (MTR)**: 可扩展到 SER、AED、MOS 等多个 reward 目标

## 在 TTS 中的应用

DiffRO 解决了 TTS RL 的两个核心难题:
- **计算成本**: 传统方法需通过 CFM + vocoder 生成完整音频后才能计算 reward;DiffRO 直接在 token 空间操作
- **正负样本区分度**: 生成的语音经过 downstream rendering 后高度相似,难以训练 reward model;DiffRO 在 token 层有更大区分度

实验显示 DiffRO 在 CosyVoice 2 和 CosyVoice 3 上均有效,WER 相对改进 20%~50%,低资源语言(如韩语)改进可达 68.7%。

## 关键论文

- [[论文笔记/Seed-TTS|Seed-TTS]] (ByteDance, 2024): DiffRO 的直接前驱——首次在 TTS 中系统性应用 RL post-training (REINFORCE),使用 SIM+WER 和 SER accuracy 作为 reward function,验证了 RL 对 TTS robustness/similarity/controllability 的有效性,同时发现 reward hacking 问题
- [[论文笔记/DiffRO|DiffRO]] (Gao et al., Tongyi Lab, 2507.05911, 2025): DiffRO 方法的原始独立论文,详细阐述 Token2Reward 机制 (SenseVoice→embedding layer 改造) 和 Gumbel-Softmax 可微采样,首次提出 Multi-Task Reward (MTR) 模型统一 ASR/SER/SQA/AED 多维度反馈; 在 CosyVoice 2.0 上验证,WER zh 1.56→0.78 (Seed-TTS-eval); MTR 实现零样本情感控制,accuracy 全面超越 F5-TTS/GPT-SoVITS; 揭示 FM+vocoder 去噪瓶颈限制 MOS/speaker 属性控制
- CosyVoice 3 (2025): 将 DiffRO 集成到完整 TTS 系统中,加入 token-level KL 约束

## 相关概念

- RLHF: NLP 中的对齐方法,DiffRO 可视为其在 TTS 的 token-level 适配
- [[Gumbel-Softmax]]: DiffRO 实现可微采样的关键技术
- KL Divergence: 约束策略不偏离参考模型
- [[Speech Tokenizer]]: DiffRO 优化的目标对象(token 选择)

## GRPO vs DiffRO 对比 (Tongyi, 2025)

Gao et al. (2025) [[论文笔记/RL-for-Audio-LLM|RL-for-Audio-LLM]] 首次在同一框架下公平对比 GRPO 和 DiffRO 用于 TTS RL:
- **DiffRO 降 WER 更强**: DiffRO R^1 WER 3.418 vs GRPO R^1 WER 3.710 (CosyVoice2, CV3-Eval)
- **DiffRO 可能伤 speaker similarity**: SS 77.00 vs GRPO SS 77.26
- **DiffRO 训练更稳定**: 超过 1500 步后 GRPO 迅速退化,DiffRO 保持稳定
- **组合方案**: 直接合并 GRPO+DiffRO loss 反而变差;通过 sample filter (仅对 positive samples 做 DiffRO) 解决兼容性问题,Combined+Filter R^{1,2,3} 达 WER 3.414

## 偏好优化路线 (SpeechAlign, 2024)

Zhang et al. (2024) [[论文笔记/SpeechAlign|SpeechAlign]] 是首次将偏好学习引入 codec language model 的工作,通过 golden vs synthetic AR tokens 构建偏好数据集。虽然使用 DPO 而非 DiffRO,但其 iterative self-improvement 思路与 DiffRO 的 RL 后训练目标一致: 让 TTS 输出更接近人类偏好的分布。

## Multi-Reward GRPO (Tencent, 2025)

Zhong et al. (2025) [[论文笔记/Multi-Reward GRPO|Multi-Reward GRPO]] 在单码本 TTS LLM (LLaSA) 上提出多奖励 GRPO 框架,包含 5 个奖励 (WER + SIM + length penalty + entropy regularization + LLM-annotated prosody alignment)。与 DiffRO 的核心区别: GRPO 在 audio-level 操作,不需要 Gumbel-Softmax 或 token-level reward model,与 ASR/speaker verification 工具链天然兼容。LLaSA+RL 在 SEED benchmark 达到 CER 1.10 (zh) / MOS 4.12,附加 FM 后 MOS 4.21。

## Industrial-scale GRPO for TTS (TTS-1, Inworld, 2025)

[[论文笔记/TTS-1|TTS-1]] (Inworld AI, 2025) 在 8.8B 参数 LLaMA-based TTS 上应用 GRPO,使用 WER + SIM + DNSMOS 三维 composite reward ($\alpha=\beta=\gamma=1.0$)。关键设计: (1) 禁用 reward scaling,使用 unscaled mean-centered advantages 避免 difficulty bias; (2) reward 在 48kHz 解码后的最终音频上计算,而非 token 空间。在 1000 小时英语数据上 GRPO 使 WER 相对降低 20-26% [TTS-1 Table 8]。与 Multi-Reward GRPO 的区别: TTS-1 使用更简单的 3 维 reward (无 length penalty / entropy / prosody),但模型规模更大 (8.8B vs LLaSA ~1B)。

## 演进

RLHF for NLP (2022) → RL for TTS on audio (Seed-TTS, 2024) → Preference optimization for codec LM (SpeechAlign, 2024) → Multidimensional preference set + CE regularization / MPO (NWPU, 2025) → Fine-grained token-level DPO / FPO (NWPU, 2025) → Token-level differentiable optimization / DiffRO (CosyVoice 3, 2025) → GRPO vs DiffRO 统一对比 + Combined (Tongyi, 2025) → Multi-Reward GRPO for single-codebook TTS (Tencent, 2025) → Industrial-scale GRPO (TTS-1, 2025) → Component-level GRPO for duration predictor (DMOSpeech 2, 2025) → GRPO for NAR flow-matching TTS via output probabilization (F5R-TTS, 2025) → Token-level KTO with contrastive LLM weight estimation / TKTO (SpiralAI, 2025) → GRPO prosody collapse diagnosis + iterative human-DPO (Channel Corp, 2026)

## GRPO Prosody Collapse + Iterative Human-DPO (Channel Corp, 2026)

Shin et al. (2026) [[论文笔记/No Verifiable Reward for Prosody|No Verifiable Reward for Prosody]] 提供了 GRPO 韵律坍缩的直接实验证据,并以 iterative DPO + 真人偏好标注作为修复方案。核心发现: (1) CER/NLL-driven GRPO 使 Llasa-1B 的 logF0 分布显著收窄,CER 降至 2.20% 但 ELO (human preference) 降至 753.7 (所有系统最低); (2) 加入 speaker-similarity reward 导致 CER 暴涨至 42.63% + EOS 失败; (3) iterative DPO (200 human pairs/round, moving reference) 在 Round 2 达到 ELO 1190.1 (最高) + CER 3.60%。与其他路线的区别: 本文不追求自动化 reward,而是论证"当 prosody 不可自动 reward 时,少量人类偏好是最实用的路径"。

## Fine-grained Preference Optimization / FPO (NWPU, 2025)

Yao et al. (2025) [[论文笔记/FPO|FPO]] 提出 token-level 选择性 DPO,在 utterance-level DPO 和 DiffRO 之间开辟第四条路线: 仍用 DPO 框架但将 loss 计算下沉到 error segment tokens。核心贡献: (1) 将 TTS segmental errors 分为 temporal modeling errors (局部标注) 和 semantic-phonetic alignment errors (级联标注至序列末尾); (2) 通过 indicator function 仅在 error tokens 上计算 DPO loss。在 CosyVoice/CosyVoice2/Llasa 三个 backbone 上,FPO 以 3-4x 数据效率 (200 utterances ≈ UNO 600 utterances) 将 bad case ratio 降低 40-56%。与 DiffRO 的核心区别: FPO 仍需显式偏好数据标注,而 DiffRO 通过可微 reward model 实现 end-to-end 优化; FPO 不使用 Gumbel-Softmax。

## Token-level KTO / TKTO (SpiralAI, 2025)

Kotoge & Sasaki (2025) [[论文笔记/TKTO|TKTO]] 提出第五条路线: 基于 prospect theory 的 token-level 无配对优化。核心方法: (1) 用 label-flipped KTO 训练 contrastive LLM pair (π+ 和 π-),其 log-ratio 自动估计 token-level importance weights; (2) 将 KTO 从 sequence-level 推广到 token-level weighted optimization。在 CosyVoice 2 (0.5B) 上验证,日语歧义发音 Acc 0.668→0.958, CER 降低 54%。与 FPO 的核心区别: TKTO 不需要 ASR forced alignment 标注 error segments,token importance 由 contrastive LLMs 自动估计; 同时消除了对配对数据的依赖 (KTO vs DPO),可利用 6x 更多数据。

## Component-level GRPO (DMOSpeech 2, Columbia/NewsBreak, 2025)

Li et al. (2025) [[论文笔记/DMOSpeech 2|DMOSpeech 2]] 开辟了 RL-for-TTS 的第三条路线: 不对整个 pipeline 做 RL,而是将 GRPO 精确靶向 duration predictor 这一单一组件。利用已有 4-step DMD-distilled student 生成样本计算 reward (SIM + WER),将 RL 计算成本压缩到传统方案的一小部分。Seed-TTS-en WER 1.752 (超越 ground truth duration 的 1.821), SIM 0.698。与 DiffRO/Multi-Reward GRPO 的核心区别: RL 作用对象是 duration predictor 而非 token/audio generator,问题空间更小,训练更高效 (仅需 1.5K GRPO steps)。
