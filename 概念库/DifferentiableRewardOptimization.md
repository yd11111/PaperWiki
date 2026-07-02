---
type: concept
title: "Differentiable Reward Optimization"
aliases: [DiffRO]
category: "training-strategy"
tags: [reinforcement-learning, post-training, TTS, reward-model]
key_papers: ["[[论文笔记/DiffRO|DiffRO]]", "[[论文笔记/CosyVoice3|CosyVoice 3]]", "[[论文笔记/GLM-TTS|GLM-TTS]]", "[[论文笔记/RL-for-Audio-LLM|RL-for-Audio-LLM]]", "[[论文笔记/SpeechAlign|SpeechAlign]]", "[[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]", "[[论文笔记/Multi-RewardGRPO|Multi-Reward GRPO]]", "[[论文笔记/FishAudioS2|Fish Audio S2]]", "[[论文笔记/DMOSpeech2|DMOSpeech 2]]", "[[论文笔记/DMOSpeech|DMOSpeech]]", "[[论文笔记/FPO|FPO]]", "[[论文笔记/FlexSpeech|FlexSpeech]]", "[[论文笔记/Koel-TTS|Koel-TTS]]", "[[论文笔记/F5R-TTS|F5R-TTS]]", "[[论文笔记/MPO|MPO]]", "[[论文笔记/DLPO|DLPO]]", "[[论文笔记/LatinX|LatinX]]", "[[论文笔记/TTS-1|TTS-1 (Inworld, 2025)]]", "[[论文笔记/TKTO|TKTO]]", "[[论文笔记/NoVerifiableRewardforProsody|No Verifiable Reward for Prosody]]", "[[论文笔记/RLAIF-SPA|RLAIF-SPA]]", "[[论文笔记/Vox-Evaluator|Vox-Evaluator]]", "[[论文笔记/ARDM-DPO|ARDM-DPO]]", "[[论文笔记/Align2Speak|Align2Speak]]", "[[论文笔记/GRPO-TTS|GRPO-TTS]]"]
origin_paper: ""
related_concepts: ["[[Gumbel-Softmax]]", "[[SpeechTokenizer]]"]
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
- [[SpeechTokenizer]]: DiffRO 优化的目标对象(token 选择)

## GRPO vs DiffRO 对比 (Tongyi, 2025)

Gao et al. (2025) [[论文笔记/RL-for-Audio-LLM|RL-for-Audio-LLM]] 首次在同一框架下公平对比 GRPO 和 DiffRO 用于 TTS RL:
- **DiffRO 降 WER 更强**: DiffRO R^1 WER 3.418 vs GRPO R^1 WER 3.710 (CosyVoice2, CV3-Eval)
- **DiffRO 可能伤 speaker similarity**: SS 77.00 vs GRPO SS 77.26
- **DiffRO 训练更稳定**: 超过 1500 步后 GRPO 迅速退化,DiffRO 保持稳定
- **组合方案**: 直接合并 GRPO+DiffRO loss 反而变差;通过 sample filter (仅对 positive samples 做 DiffRO) 解决兼容性问题,Combined+Filter R^{1,2,3} 达 WER 3.414

## 偏好优化路线 (SpeechAlign, 2024)

Zhang et al. (2024) [[论文笔记/SpeechAlign|SpeechAlign]] 是首次将偏好学习引入 codec language model 的工作,通过 golden vs synthetic AR tokens 构建偏好数据集。虽然使用 DPO 而非 DiffRO,但其 iterative self-improvement 思路与 DiffRO 的 RL 后训练目标一致: 让 TTS 输出更接近人类偏好的分布。

## Multi-Reward GRPO (Tencent, 2025)

Zhong et al. (2025) [[论文笔记/Multi-RewardGRPO|Multi-Reward GRPO]] 在单码本 TTS LLM (LLaSA) 上提出多奖励 GRPO 框架,包含 5 个奖励 (WER + SIM + length penalty + entropy regularization + LLM-annotated prosody alignment)。与 DiffRO 的核心区别: GRPO 在 audio-level 操作,不需要 Gumbel-Softmax 或 token-level reward model,与 ASR/speaker verification 工具链天然兼容。LLaSA+RL 在 SEED benchmark 达到 CER 1.10 (zh) / MOS 4.12,附加 FM 后 MOS 4.21。

## Industrial-scale GRPO for TTS (TTS-1, Inworld, 2025)

[[论文笔记/TTS-1|TTS-1]] (Inworld AI, 2025) 在 8.8B 参数 LLaMA-based TTS 上应用 GRPO,使用 WER + SIM + DNSMOS 三维 composite reward ($\alpha=\beta=\gamma=1.0$)。关键设计: (1) 禁用 reward scaling,使用 unscaled mean-centered advantages 避免 difficulty bias; (2) reward 在 48kHz 解码后的最终音频上计算,而非 token 空间。在 1000 小时英语数据上 GRPO 使 WER 相对降低 20-26% [TTS-1 Table 8]。与 Multi-Reward GRPO 的区别: TTS-1 使用更简单的 3 维 reward (无 length penalty / entropy / prosody),但模型规模更大 (8.8B vs LLaSA ~1B)。

## 演进

RLHF for NLP (2022) → RL for TTS on audio (Seed-TTS, 2024) → Preference optimization for codec LM (SpeechAlign, 2024) → Multidimensional preference set + CE regularization / MPO (NWPU, 2025) → Fine-grained token-level DPO / FPO (NWPU, 2025) → Token-level differentiable optimization / DiffRO (CosyVoice 3, 2025) → GRPO vs DiffRO 统一对比 + Combined (Tongyi, 2025) → Multi-Reward GRPO for single-codebook TTS (Tencent, 2025) → Industrial-scale GRPO (TTS-1, 2025) → Component-level GRPO for duration predictor (DMOSpeech 2, 2025) → GRPO for NAR flow-matching TTS via output probabilization (F5R-TTS, 2025) → Token-level KTO with contrastive LLM weight estimation / TKTO (SpiralAI, 2025) → GRPO prosody collapse diagnosis + iterative human-DPO (Channel Corp, 2026) → DPO for autoregressive diffusion models / ARDM-DPO (CUHK-SZ + ByteDance, 2025) → GRPO for low-resource multilingual TTS with IPA + unpaired data / Align2Speak (NVIDIA, 2025) → GRPO with CER+NLL composite reward, cross-architecture verification / GRPO-TTS (USTC/iFLYTEK, 2025) → LALM continuation-based style reward + gated GRPO / MCLP (StepFun/CASIA, ICML 2026) → Self-consistency rewards GRPO for speech editing (CASIA/Tsinghua, ICME 2026) → ODE→SDE Flow-GRPO for FM-based TTS (Tongyi Lab, Interspeech 2026) → Structured preference space + hierarchical progressive reward / HPRO (SCUT/Huya, 2026)

## GRPO Prosody Collapse + Iterative Human-DPO (Channel Corp, 2026)

Shin et al. (2026) [[论文笔记/NoVerifiableRewardforProsody|No Verifiable Reward for Prosody]] 提供了 GRPO 韵律坍缩的直接实验证据,并以 iterative DPO + 真人偏好标注作为修复方案。核心发现: (1) CER/NLL-driven GRPO 使 Llasa-1B 的 logF0 分布显著收窄,CER 降至 2.20% 但 ELO (human preference) 降至 753.7 (所有系统最低); (2) 加入 speaker-similarity reward 导致 CER 暴涨至 42.63% + EOS 失败; (3) iterative DPO (200 human pairs/round, moving reference) 在 Round 2 达到 ELO 1190.1 (最高) + CER 3.60%。与其他路线的区别: 本文不追求自动化 reward,而是论证"当 prosody 不可自动 reward 时,少量人类偏好是最实用的路径"。

## Fine-grained Preference Optimization / FPO (NWPU, 2025)

Yao et al. (2025) [[论文笔记/FPO|FPO]] 提出 token-level 选择性 DPO,在 utterance-level DPO 和 DiffRO 之间开辟第四条路线: 仍用 DPO 框架但将 loss 计算下沉到 error segment tokens。核心贡献: (1) 将 TTS segmental errors 分为 temporal modeling errors (局部标注) 和 semantic-phonetic alignment errors (级联标注至序列末尾); (2) 通过 indicator function 仅在 error tokens 上计算 DPO loss。在 CosyVoice/CosyVoice2/Llasa 三个 backbone 上,FPO 以 3-4x 数据效率 (200 utterances ≈ UNO 600 utterances) 将 bad case ratio 降低 40-56%。与 DiffRO 的核心区别: FPO 仍需显式偏好数据标注,而 DiffRO 通过可微 reward model 实现 end-to-end 优化; FPO 不使用 Gumbel-Softmax。

## Token-level KTO / TKTO (SpiralAI, 2025)

Kotoge & Sasaki (2025) [[论文笔记/TKTO|TKTO]] 提出第五条路线: 基于 prospect theory 的 token-level 无配对优化。核心方法: (1) 用 label-flipped KTO 训练 contrastive LLM pair (π+ 和 π-),其 log-ratio 自动估计 token-level importance weights; (2) 将 KTO 从 sequence-level 推广到 token-level weighted optimization。在 CosyVoice 2 (0.5B) 上验证,日语歧义发音 Acc 0.668→0.958, CER 降低 54%。与 FPO 的核心区别: TKTO 不需要 ASR forced alignment 标注 error segments,token importance 由 contrastive LLMs 自动估计; 同时消除了对配对数据的依赖 (KTO vs DPO),可利用 6x 更多数据。

## Component-level GRPO (DMOSpeech 2, Columbia/NewsBreak, 2025)

Li et al. (2025) [[论文笔记/DMOSpeech2|DMOSpeech 2]] 开辟了 RL-for-TTS 的第三条路线: 不对整个 pipeline 做 RL,而是将 GRPO 精确靶向 duration predictor 这一单一组件。利用已有 4-step DMD-distilled student 生成样本计算 reward (SIM + WER),将 RL 计算成本压缩到传统方案的一小部分。Seed-TTS-en WER 1.752 (超越 ground truth duration 的 1.821), SIM 0.698。与 DiffRO/Multi-Reward GRPO 的核心区别: RL 作用对象是 duration predictor 而非 token/audio generator,问题空间更小,训练更高效 (仅需 1.5K GRPO steps)。

## GRPO with ASR NLL Reward / GRPO-TTS (USTC/iFLYTEK, 2025)

Liu et al. (2025) [[论文笔记/GRPO-TTS|GRPO-TTS]] 提出用现成 ASR 模型 (Whisper) 的 CER + NLL 构建 composite reward,通过加权调和平均组合后用 GRPO 微调 TTS LLM。与 DiffRO 的核心区别: 完全在 audio-level 操作,不需要 Gumbel-Softmax 或 token-to-text reward model,仅依赖 off-the-shelf ASR。独特贡献: (1) NLL 作为 CER 互补信号 (r=0.3371),在 CER=0 时仍提供区分度; (2) 同时验证 semantic token 路线 (CosyVoice2) 和 acoustic token 路线 (Llasa-1B),发现 GRPO 对两类系统的 CER 均有效但仅对 semantic token 路线改善自然度。CosyVoice2+GRPO: CER zh 1.41→1.07, MOS zh 4.42→4.58; Llasa+GRPO: CER zh 7.73→1.30 但 MOS 无显著改善 [GRPO-TTS Table 1, Table 2]。

## LALM Continuation-based Style Reward / MCLP (StepFun/CASIA, ICML 2026)

Ren et al. (2026) [[论文笔记/MCLP|MCLP]] 将 LALM 的 continuation log-probability 重新定义为**风格一致性 reward**,用于 Role-Play TTS 的 GRPO 优化。与已有 GRPO 工作 (Multi-Reward GRPO/TTS-1/GRPO-TTS) 主要覆盖 CER/SIM/DNSMOS 等 content-fidelity 维度不同,MCLP 是首个 dense、continuous 的 style reward。关键设计: gated hybrid reward — CER 超过阈值 τ=0.2 时直接将 reward 置零,防止"expressive gibberish"式 reward hacking (消融显示纯 MCLP reward 导致 CER 暴涨至 61%)。基于 Step-Audio-2-mini (7B) + GRPO (8 rollouts, 1000 iterations),MOS 从 SFT 的 3.178 提升至 3.576,CER 从 3.334% 降至 1.130% [Table 2, 3]。与 RLAIF-SPA (属性级 AI 反馈) 的区别: MCLP 利用 LALM 的 ICL 能力在 latent style space 度量风格连续性,而非依赖离散标签匹配。

## DPO for Autoregressive Diffusion / ARDM-DPO (CUHK-SZ + ByteDance, 2025)

Liu et al. (2025) [[论文笔记/ARDM-DPO|ARDM-DPO]] 首次将 DPO 扩展到**连续 token 自回归扩散模型 (ARDM)**,填补了 next-token diffusion 范式下偏好对齐的空白。核心推导: 将 ARDM 采样视为 (token index n, diffusion time t) 双重索引的马尔可夫链,通过 Jensen 不等式近似将 DPO 目标分解为逐 token-timestep 的 denoising loss 比较。在 DiTAR (0.4B) 上验证: Task A (F0V 14.2→29.2 Hz, 表现力翻倍) + Task B (CER 8.37→6.32, 鲁棒性 25% 改善),均保持 SIM 无显著损失 (KL ~0.01)。与 DiffRO/GRPO 等路线的核心区别: 对象是连续 token ARDM 而非离散 token LM 或 NAR diffusion。

## Robust Reward Model for DiffRO / RRPO (Tongyi Lab, 2026)

Wang et al. (2026) [[论文笔记/RRPO|RRPO]] 识别并解决了 DiffRO 的 reward hacking 脆弱性: DiffRO 的全可微优化使 RM 缺陷被解析梯度精确放大,vanilla SER RM 的过度自信、脆弱决策边界和扰动敏感性使 policy 可通过生成声学伪影 (如不自然的嘴部咔嗒声) 骗取虚假奖励。核心方法: 三层混合正则化 fine-tune RM — (1) Label Smoothing 修正离散情感标签导致的过度自信; (2) Energy-Adaptive Mixup 基于语音能量自适应混合平滑决策边界; (3) Adversarial Training 在高层 embedding 上增强扰动鲁棒性。在 CosyVoice2 上验证: E-MOS 3.78 / N-MOS 3.81 (DiffRO: 3.65 / 3.61); SER WA 在 ESD 上 64.4→81.7%,跨语言 IEMOCAP 66.0→68.0% [RRPO Table 1, Table 2]。与 DiffRO 的核心区别: 不改 policy optimization 算法,仅强化 RM 质量;填补了 DiffRO 演进线中"RM 鲁棒性保障"的空白。

## ASR Cross-Attention Reward / W3AR (Melbourne/NJUPT, AAAI 2026)

Wang & Sun (2025) [[论文笔记/W3AR|W3AR]] 提出第六条路线: 利用 frozen ASR (Whisper) 的 cross-attention map 直接提取 word-level reward,无需额外训练 reward model 或偏好数据标注。两个互补指标: (1) Attention Purity — attention 在峰值周围的集中度,衡量发音清晰度; (2) Alignment Monotonicity — attention 峰值的单调前进程度,衡量韵律流畅性。使用 word-level group-relative policy optimization 更新 TTS。在 CosyVoice 上 OOD WER 8.92→4.54 (-49.1%),泛化到 VoiceCraft/MaskGCT 上同样有效。与 DiffRO 的核心区别: reward 在 audio-level 操作且来自 frozen ASR 内部表征 (cross-attention),不需要 Gumbel-Softmax 或 Token2Text reward model; 与 FPO 的区别: 不需要 error segment 标注,word-level importance 通过 group-relative advantage 自动发现。

## Self-Consistency Rewards GRPO for Speech Editing (CASIA/Tsinghua, ICME 2026)

Ren et al. (2026) [[论文笔记/EditContentPreserveAcoustics|Edit Content, Preserve Acoustics]] 首次将 GRPO 从 TTS 扩展到 **text-based speech editing** 领域。核心创新是 self-consistency reward: 用预训练 TTS 模型 (CosyVoice 3) 的条件 log-probability 作为编辑区与上下文融合质量的隐式评估,理论上等价于最小化 policy 与 TTS 先验的交叉熵 (Eq. 5)。配合 ASR WER + duration 一致性的 gated reward aggregation (不满足阈值的样本直接置零)。在 semantic token 空间做 editing + GRPO 对齐,Insertion WER 4.97% (vs VoiceCraft 12.94%), MOS 4.06 (vs 3.64) [Table I]。与 MCLP 的相似点: 两者独立提出了 gated reward 设计 (WER/CER 阈值硬门控); 与 MCLP 的区别: 本文的 self-consistency reward 衡量上下文融合 (editing coherence),MCLP 衡量风格一致性 (style continuity)。

## Editing-Oriented GRPO for End-to-End Speech Editing (Nankai, 2026)

Chen et al. (2026) [[论文笔记/CosyEdit2|CosyEdit2]] 在 CosyVoice2 backbone 上构建端到端 speech editing 系统,提出 teacher-free outcome-level GRPO。与 ECPA 的 self-consistency GRPO (依赖 TTS prior 作为隐式 critic) 不同,CosyEdit2 直接在解码波形上计算三维 editing-oriented reward: (1) WER 指数衰减 reward 作为粗粒度内容门控; (2) DTW-aligned MCD 在非编辑区域衡量声学保持 (含 δ=2dB 容忍度); (3) speaker embedding cosine similarity。三者通过乘性门控 + 加性排序的 coarse-to-fine 层次组合。target-speech-free 数据构造使 GRPO 无需人工目标录音,仅用 3000 句 GigaSpeech-XL 即完成训练。关键发现: editing-oriented GRPO 反哺 zero-shot TTS,SEED-TTS CER zh 1.36→1.16, WER en 3.10→1.95。与 DiffRO 的核心区别: 在 audio-level 计算 reward 但在 token-level 更新策略 (仅更新 LLM,冻结 Flow+BigVGAN),不需要 Gumbel-Softmax 或 token-level reward model。

## Flow-GRPO for FM-Based TTS (Tongyi Lab, Interspeech 2026)

Wang et al. (2026) [[论文笔记/FlowTTS-GRPO|FlowTTS-GRPO]] 首次将 GRPO 引入 flow-matching (FM) based TTS 的连续声学空间,通过 ODE→SDE 转换 (借鉴图像生成 Flow-GRPO [27]) 在保持边际分布不变的前提下为确定性 FM 推理注入 RL 所需的随机性。与 F5R-TTS 训练独立 Gaussian FM generator 不同,FlowTTS-GRPO 直接复用开源 FM 模型,支持多 rollout + 多目标 reward (SS + ASR + DNSMOS,std 归一化加权组合)。在 CosyVoice 3 上 SS1 test-zh 达 0.804,首次超越闭源 Seed-TTS (0.796) [Table 2]。关键发现: LLM+FM hybrid 系统中 FM-GRPO 主要改善 audio details (SS/DNSMOS),与 LM-RL 改善 intelligibility 功能解耦,两者可叠加。与 DiffRO 的核心区别: 作用于 FM 的连续声学空间而非 LM 的离散 token 空间,优化对象互补。

## ICL-Based Online RL for Conversational TTS (Meta AI, 2026)

Ouyang et al. (2026) [[论文笔记/ConversationalTTS-RL|ConversationalTTS-RL]] 在对话 TTS 场景中将 ICL (audio prompting) 与 online RL 耦合。核心方法: AR prosody model 以 human-curated audio prompt 作为 ICL 条件,用 AES-CE (aesthetic quality) reward + CTC alignment loss 正则化组成 composite reward (R = alpha_AES * AES - alpha_CTC * L_CTC),通过 KL-regularized policy gradient 在线优化。CTC 正则化有效抑制了 AES-only 优化导致的 text hallucination。RL-AES-CTC vs SFT-only: CMOS net win rate +7.1% (95% CI: 3.97-10.23%) [Table 2]。与已有路线的区别: (1) RL policy 与 ICL audio prompt 耦合训练 (非 unconditioned RL); (2) 在 audio-level 操作 (类似 Seed-TTS REINFORCE),不需要 token-level reward model; (3) CTC alignment 作为 text faithfulness 约束,与 MCLP/Edit Content 的 gated WER/CER reward 思路互补。

## Structured Preference Space + Hierarchical Progressive Reward / HPRO (SCUT/Huya, 2026)

Nie et al. (2026) [[论文笔记/HPRO|HPRO]] 从 reward 空间结构和优化粒度两个维度改进 DiffRO 在 emotional TTS 中的应用。核心方法: (1) HD-Emo codec — 双 FSQ 信息瓶颈将 speech tokens 分流为 content preference tokens (codebook=1296) 和 style preference tokens (codebook=64),用 stop-gradient 隔离 content branch 仅由 ASR 监督更新,style branch 由 SER + wVAD 层级监督; (2) 渐进式三阶段优化 — Frame warm-up (Lcp+Lsp) → Word refinement (+LwVAD+LASR) → Sentence alignment (+LSER),Gumbel 温度从 2→1→0.8 退火。在 CosyVoice2 上验证: WER 4.02% (vs CosyVoice2 SFT 5.45%), EMO-SIM 0.672 (vs 0.613), wVAD-CCC 0.339 (vs 0.307) [Table I]。消融 w/o frame&wvad (模拟 DiffRO) 的 WER 4.35% / wVAD-CCC 0.315 均劣于 HPRO,验证了层级 reward 对单尺度全局 reward 的改进 [Table III]。与 RRPO 的区别: RRPO 从 RM 鲁棒性切入 (加固 RM 本身),HPRO 从 reward 空间结构 (content/style 分离) 和优化粒度 (层级渐进) 切入,两者互补。

## GRPO-LoRA 作为可组合风格控制信号 / GLASS (2026)

[[论文笔记/GLASS|GLASS]] (Fang et al., 2026) 将 GRPO 的用途从"改善 TTS 生成质量"拓展到"学习模块化风格控制信号": 在冻结 CosyVoice2 backbone 上,为每个风格维度 (speed/pitch) 独立训练 GRPO-LoRA,SER/pitch/speed 评估器提供 reward,LoRA 权重更新方向被 reward 梯度引导为参数空间中的风格方向向量。运行时通过 LoRA 组合实现多维风格同时控制。Fast LoRA SPS 5.59 (vs baseline 3.65),S-MOS 4.72 vs DSP 3.08 [Table 1]。与 DiffRO/GRPO 在 TTS RL 中的传统角色 (优化 WER/SIM/MOS 等生成质量指标) 不同,GLASS 中 GRPO 的优化目标是**风格变化方向的准确性**而非整体生成质量,LoRA 的可组合性使 GRPO 产出的 reward 知识可模块化复用。与 FlowTTS-GRPO 的区别: FlowTTS-GRPO 在 FM 声学空间做全局质量优化,GLASS 在 LM 参数空间做维度化风格控制。
