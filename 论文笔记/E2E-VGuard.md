---
type: paper
tier: deep
title: "E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis"
arxiv_id: "2511.07099"
source: "Sources/E2E-VGuard.pdf"
authors: [Zhisheng Zhang, Derui Wang, Yifan Mi, Zhiyong Wu, Jie Gao, Yuxin Cao, Kai Ye, Minhui Xue, Jie Hao]
year: 2025
venue: "NeurIPS 2025"
tags: [voice-protection, adversarial-perturbation, voice-cloning, anti-spoofing, deepfake-defense, speaker-verification, LLM-TTS, end-to-end, psychoacoustic-model, ASR-attack]
concepts: ["[[Anti-spoofing and Deepfake Detection]]", "[[Speaker Verification]]", "[[Speaker Embedding]]", "[[Voice Cloning Taxonomy]]", "[[LLM-based TTS]]", "[[Speech Tokenizer]]"]
models: ["[[VITS]]", "[[CosyVoice]]", "[[WavLM]]", "[[Whisper]]", "[[wav2vec 2.0]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-04
updated: 2026-06-04
---

## KB 背景

> [!info] KB 背景 (基于 3 个已确认 + 3 个待确认实体页: [[Speaker Embedding]], [[LLM-based TTS]], [[Speech Tokenizer]], [[Anti-spoofing and Deepfake Detection]], [[Speaker Verification]], [[Voice Cloning Taxonomy]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文是 [[Anti-spoofing and Deepfake Detection]] 中"主动防御"分支的最新进展,由 [[论文笔记/SafeSpeech|SafeSpeech]] 同一一作(Zhisheng Zhang)推进。SafeSpeech (USENIX Security 2025) 通过 unlearnable examples + pivotal objective (mel L1) + SPEC (KL 散度引导输出趋近噪声) 实现 fine-tuning 阶段的语音保护,但其防护范围限于 DNN-based TTS(VITS/StyleTTS2 等连续 embedding 模型)。E2E-VGuard 将防护扩展到两个 SafeSpeech 未覆盖的关键场景: (1) LLM-based TTS(离散 speech token 架构)和 (2) ASR-driven 端到端攻击场景。
>
> **已有认知**: [[Speaker Embedding]] 页(confirmed)记录了 speaker encoder 在 TTS 中的注入方式和架构演进(d-vector → ECAPA-TDNN → CAM++),E2E-VGuard 正是利用这些 encoder 的 ensemble 来最大化 timbre 扰动。[[LLM-based TTS]] 页(confirmed)记录了 LLM-TTS 通过 speech tokenizer 将音频编码为离散 token 供 LM 建模的核心架构,这正是 E2E-VGuard 需要新防护机制的原因 -- SafeSpeech 的 mel L1 扰动无法有效作用于离散量化后的 token。[[Speech Tokenizer]] 页(confirmed)记录了语音离散化的三类方案(自监督/监督 semantic/声学),理解这些有助于理解 E2E-VGuard 为何引入 MFCC extractor 来补充对 LLM 组件的防护。
>
> **创新判断**: 相较 KB 中已有的 SafeSpeech 记录,E2E-VGuard 的核心新贡献在于: (a) 用 encoder ensemble + MFCC extractor 替代 SafeSpeech 的单代理模型 pivotal objective,实现对 LLM-based TTS(离散 token)的 timbre 防护; (b) 新增 ASR 对抗攻击维度,破坏 E2E 场景下的文本-发音对齐; (c) 用心理声学模型替代 SafeSpeech 的 STOI+STFT 感知优化。KB 中 [[Voice Cloning Taxonomy]] 的四分类(SA/FS/ZS/ML)下,E2E-VGuard 覆盖 fine-tuning + zero-shot 两种场景,且首次在 3 个商业 API 上验证有效性。
>
> 检索命中: [[Speaker Embedding]]✓, [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓, [[Anti-spoofing and Deepfake Detection]][待确认], [[Speaker Verification]][待确认], [[Voice Cloning Taxonomy]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过 encoder ensemble timbre 扰动 + ASR 对抗攻击发音破坏 + 心理声学模型感知优化,实现对 LLM-based TTS 和 ASR-driven 端到端语音克隆的双层主动防护
> - **路线**: 原始音频 → {6 encoder ensemble (VITS/GSV/CosyVoice/StyleTTS2 encoder + WavLM + MFCC) → timbre feature loss} + {ASR 模型 (Wav2vec2) → CTC targeted attack → pronunciation loss} + {心理声学模型 → 频域掩蔽阈值 + L2 norm → 感知 loss} → PGD 优化 500 步 → 受保护音频
> - **指标**: Fine-tuning: WER↑ 94.812% (T, GSV) / SIM↓ 0.082 (UT, StyleTTS2) [Table 1]; Zero-shot: SIM↓ 0.008 (UT, Step-Audio) [Table 2]; Commercial API: SIM 0.689→0.203 [Fig 2]; 感知: SNR 18.523-20.470, PESQ 1.949-2.324, MOS 3.522±0.218 [Table 1, Appendix G]; 真实世界: WER 72.615%, SIM 0.068 [Fig 3]
> - **可借鉴**: (1) Encoder ensemble + 异质 feature extractor(传统 MFCC + 预训练 SSL + TTS encoder)组合策略,可迁移到任何需要跨模型 transferable perturbation 的场景; (2) 双层防护思路(timbre + pronunciation),分别针对 TTS 的两个核心输入(参考音频特征 + 文本)施加扰动,比单层防护覆盖面更广
> - **局限**: (1) 扰动生成需 ~98-112 秒/样本(4090 GPU) [Appendix A],比 SafeSpeech 的 10.6s 慢约 10 倍; (2) 心理声学模型的 SNR 优于 baseline 但 PESQ 1.949 较低,说明在时频感知精细度上仍有提升空间; (3) 发音保护依赖特定 ASR 模型(非 universal),需依赖迁移性 [Appendix A]; (4) 代码开源但未提供预训练权重

## 核心问题

E2E-VGuard 要解决 SafeSpeech 等前序工作留下的两个关键盲区 [§1]:

1. **LLM-based TTS 盲区**: SafeSpeech/AntiFake/POP 等方法主要针对 DNN-based TTS(VITS/StyleTTS2 等),这些模型使用连续 embedding 编码 timbre。但 LLM-based TTS(CosyVoice/GPT-SoVITS/Llasa 等)通过 speech tokenizer 将音频编码为**离散 token** 供 LLM 建模,连续扰动在量化后可能被部分过滤。现有方法未探索如何保护离散 token 表示下的 timbre 信息 [§1, "a direction rarely explored"]。

2. **End-to-End 场景盲区**: 之前的工作假设攻击者已拥有手动标注的文本-音频对。但实际场景中(商业 API 如 ByteDance 声音克隆、开源 WebUI 如 GPT-SoVITS),攻击者通过 ASR 系统自动获取文本。这创造了一个新的攻击面: 如果能在 ASR 阶段就让文本转录出错,就能破坏下游 TTS 的文本-发音对齐,从根本上阻止高质量合成 [§1, §3.1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

E2E-VGuard 的优化目标由三项加权组成 [§3.2, Eq. 1]:

```
L(x') = L_asr(x') + alpha * L_fea(x') + beta * L_psy(x')
s.t. ||x' - x||_p <= epsilon,  x' in [-1, 1]^T
```

三项分别对应发音防护、音色防护、感知优化。扰动通过 PGD(Projected Gradient Descent)迭代 500 步生成,epsilon=8/255 [§4.1, Algorithm 1]。

### 关键设计选择

**1. Encoder ensemble + MFCC -- 为什么不像 SafeSpeech 那样用单一代理模型** [§3.3]

[论文原文] SafeSpeech 的 pivotal objective 依赖单个代理 TTS 模型(BERT-VITS2)的 mel L1 距离,其迁移性基于"所有 TTS 都要拟合 mel"的假设。但 LLM-based TTS 的核心组件是 speech tokenizer → LLM → 合成器的三段式架构,其中 LLM 操作的是离散 token 而非连续 mel。因此 SafeSpeech 的 mel-level 扰动无法直接影响 LLM 组件的学习 [§1, "The key distinction lies in decoding audio signals into discrete tokens rather than continuous embeddings"]。

[论文原文] E2E-VGuard 的策略转变为直接在 TTS 模型的 timbre 提取端施加扰动: 集成 6 个 encoder(VITS posterior encoder、GSV posterior encoder、CosyVoice CAM++ encoder、StyleTTS2 style encoder、WavLM speaker verification、MFCC extractor),通过最大化原始音频与扰动音频在所有 encoder 输出空间中的 cosine distance [Eq. 2]。MFCC extractor 的加入专门针对 LLM-based TTS: 因为 speech tokenizer 依赖的音频特征(如离散 token 前的声学表征)与 MFCC 相关,扰动 MFCC 可以改变 tokenizer 输出的 discrete tokens,从而影响 LLM 组件 [§3.3, "we consider to protect at the audio's original features by changing the discrete tokens obtained by the audio tokenizer"]。

[agent 解读] 这个设计选择的本质是从"代理模型 + 单一 loss"路线转向"多 encoder 特征空间 + ensemble"路线。SafeSpeech 的成功依赖于 mel L1 是所有 TTS 的"最低公分母",但 LLM-based TTS 打破了这个假设(LLM 不直接看 mel)。E2E-VGuard 的 ensemble 策略更直接: 不去找"通用 loss",而是在多个实际 TTS 模型的特征提取端同时施加扰动,相当于在多个模型的"入口"处设防,而非在某个共享的"中间层"设防。

**Untargeted vs Targeted timbre protection** [§3.3]:
- Untargeted [Eq. 2]: 最大化 x 与 x' 在所有 encoder 上的 cosine similarity(→ 0),使合成语音不像任何特定人
- Targeted [Eq. 3]: 最小化 x' 与预选目标说话人 x_t 的 cosine distance,使合成语音像另一个人。目标选择通过 speaker database 找特征距离最大的说话人 [§3.3],比传统的"随机选异性"更系统化

**2. ASR 对抗攻击 -- 为什么用 targeted attack 而非 untargeted** [§3.4]

[论文原文] 在 E2E 场景中,ASR 转录的文本会与音频配对用于 TTS 训练。如果 ASR 转录出错,错误的文本-音频对会破坏 TTS 模型的 text-pronunciation alignment 学习(如 VITS 的 monotonic alignment search)。Untargeted ASR attack 会产生无意义的乱码,容易被攻击者察觉并丢弃。Targeted attack 让 ASR 输出有意义但错误的文本,降低被检测的概率 [§3.4, "effectively reducing the adversary's awareness of the anomalous recognition text"]。

[论文原文] 目标文本的选择策略 [§3.4]:
- **Targeted timbre 模式**: 用目标说话人音频的 ASR 转录作为目标文本(因为 MFCC 优化已经倾向于目标音频的特征,ASR 也倾向于识别为目标音频的文本)
- **Untargeted timbre 模式**: 随机选择与原文同长度的其他文本

[agent 解读] 将 timbre 防护和 pronunciation 防护的目标对齐(都指向同一个 target speaker)是一个精巧的设计: 扰动后的音频在特征空间中"靠近"target speaker,同时 ASR 也"识别为"target speaker 的文本。两个方向的优化互相加强而非冲突,这也解释了为什么 targeted 模式在多数场景下 WER 更高。

**3. 心理声学模型 -- 为什么不用 SafeSpeech 的 STOI+STFT** [§3.5]

[论文原文] SafeSpeech 使用 STOI(时域可懂度)和 STFT L2(频域差异)作为感知代理指标。E2E-VGuard 转用心理声学模型(psychoacoustic model)的频率掩蔽效应: 设原始音频为 masker,计算每个频率点的掩蔽阈值 θ_x(f),要求扰动的功率谱密度不超过该阈值 [Eq. 5]。加上 L2 norm [Eq. 6] 进一步限制感知干扰。

[agent 解读] 心理声学模型比 STOI+STFT 更贴近人耳感知机制: 它直接建模了"掩蔽效应"(某些频率的声音会被旁边更强的声音遮盖),因此允许在低频被掩蔽的频段嵌入更多扰动,而在敏感频段严格限制。这解释了为什么 E2E-VGuard 的 SNR(18.5-20.5)显著高于 SafeSpeech(~17.8)和其他 baseline。

### 训练策略

**扰动生成** [Algorithm 1, Appendix B]:
1. 初始化 delta 在 [-epsilon, epsilon] 内
2. 迭代 500 步:
   - 计算 C1 = L_asr (CTC loss on Wav2vec2)
   - 计算 C2 = L_fea (encoder ensemble cosine similarity)
   - 计算 C3 = L_psy (psychoacoustic + L2)
   - 加权: C = C1 + 500*C2 + 5e-3*C3
   - 用 sign gradient (PGD) 更新 delta,clamp 到 [-epsilon, epsilon]
3. 输出 x' = x + delta,clamp 到 [-1, 1]

**超参数选择** [§4.5]: alpha=500 用于放大 speaker identity loss 使其与 ASR loss 同量级; beta=5e-3 在保护效果和感知质量间取折中(beta 更大→感知更好但防护更弱; beta 更小→SNR<15 感知差)。

## 实验

| 指标 | E2E-VGuard (UT) | E2E-VGuard (T) | SafeSpeech (best baseline) | Clean | 数据集/模型 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| WER (↑) | 66.471% | 94.812% | 44.777% | 3.434% | LibriTTS, GSV fine-tune | [Table 1] |
| SIM (↓) | 0.123 | 0.284 | 0.339 | 0.685 | LibriTTS, GSV fine-tune | [Table 1] |
| WER (↑) | 95.740% | 125.299% | 105.524% | 7.796% | LibriTTS, VITS fine-tune | [Table 1] |
| SIM (↓) | 0.106 | 0.245 | 0.180 | 0.710 | LibriTTS, VITS fine-tune | [Table 1] |
| WER (↑) | 45.836% | 54.732% | 7.770% | 1.895% | LibriTTS, StyleTTS2 fine-tune | [Table 1] |
| SIM (↓) | 0.082 | 0.229 | 0.298 | 0.731 | LibriTTS, StyleTTS2 fine-tune | [Table 1] |
| WER (↑, zero-shot) | 8.156% | 3.245% | 5.764% | 2.508% | Step-Audio-TTS | [Table 2] |
| SIM (↓, zero-shot) | 0.008 | 0.128 | 0.334 | 0.579 | Step-Audio-TTS | [Table 2] |
| WER (↑, zero-shot) | 33.357% | 72.522% | 23.866% | 1.341% | Spark-TTS | [Table 2] |
| SIM (↓, zero-shot) | 0.174 | 0.260 | 0.144 | 0.666 | Spark-TTS | [Table 2] |
| SIM (↓, ICL-based) | 0.053 | 0.319 | 0.282 (AntiFake) | 0.676 | VALLE-X | [Table 3] |
| SIM (↓, ICL-based) | 0.175 | 0.176 | 0.249 (AntiFake) | 0.519 | F5-TTS | [Table 3] |
| SIM (↓, commercial) | 0.203 (avg) | - | - | 0.689 (avg) | ByteDance/Alibaba/MiniMax | [Fig 2] |
| SNR (↑) | 18.523 | 20.470 | 7.647 (best baseline) | - | 感知 | [Table 1] |
| PESQ (↑) | 1.949 | 2.324 | 1.412 (SafeSpeech) | - | 感知 | [Table 1] |
| Protected MOS (↑) | 3.522±0.218 | - | - | 4.788±0.157 | Human study | [Table 10] |
| Real-world WER (↑) | 72.615% (avg) | - | 16.837% (clean) | - | GSV, over-the-air | [Fig 3] |
| Real-world SIM (↓) | 0.068 (avg) | - | 0.485 (clean) | - | GSV, over-the-air | [Fig 3] |

**鲁棒性实验** [§4.6, Table 5]:
- Mel extraction & inversion: WER 降幅最大(VITS: 96.7%→55.6%),但 SIM 仍低(0.122),说明 timbre 保护持续有效
- Spectral gating 去噪: VITS WER 31.958%, SIM 0.251; GSV WER 23.10%, SIM 0.243
- DNN denoiser: GSV WER 23.10%, SIM 0.243 — 即使去除可听噪声,speaker identity 仍受保护
- Data augmentation(13 种): E2E-VGuard 在大部分 augmentation 下保持 SIM < 0.25
- Over-the-air(真实环境, 22 dBA): WER 72.615%, SIM 0.068,保护效果显著

**Ablation** [§4.5, Table 4]:
- 去掉 L_psy & L2: 保护更强(WER 119.242% / SIM 0.101)但 SNR 仅 12.942(感知差)
- 只用 L_asr(无 timbre): SIM 0.409(timbre 泄露严重)
- 只用 L_fea(无 pronunciation): WER 49.059%(发音保护减弱)
- 完整 E2E-VGuard: 三项协同,在保护和感知之间取得最佳平衡

## 局限性

1. **时间开销**: 每个样本保护需 ~98-112 秒(4090 GPU) [Appendix A],远慢于 SafeSpeech 的 ~10.6 秒(A800)。[论文原文] 作者建议用 batching 和多 GPU 加速,但尚未验证
2. **ASR 迁移性限制**: 发音保护针对特定 ASR(Wav2vec2)优化,迁移到其他 ASR 效果有波动。例如 Whisper-large-v3 上 GSV WER 仅 66.534%(vs Wav2vec2 上 69.148%) [Table 9]。[论文原文] 作者承认未采用 universal ASR attack 是出于效率考虑
3. **PESQ 偏低**: 尽管 SNR 最优(18.5-20.5),PESQ 仅 1.949-2.324,低于无保护音频。[agent 解读] 心理声学模型优化的是频率掩蔽效应而非端到端感知质量,PESQ 作为端到端指标可能捕捉到了未被掩蔽模型覆盖的失真
4. **Mel augmentation 的脆弱性**: Mel extraction & inversion 可将 WER 降低 39%(VITS)和 20%(GSV) [Table 5],虽然 SIM 仍低,但发音保护被显著削弱
5. **无 universal 保护**: 每段音频需单独生成扰动(不像 SafeSpeech 的 single-sample universality),对大规模部署不够高效
6. **对手动标注的部分有效性**: 即使攻击者用正确手动标注文本(绕过 ASR 保护),timbre 保护仍然有效(SIM 0.161/0.278),但发音保护依赖 ASR 被干扰 [Appendix A, "Eliminating ASR System"]

## 点评

**优势**:
- 论文准确识别了 SafeSpeech 的两个盲区(LLM-based TTS + E2E scenario)并提出针对性方案,问题定义清晰且有实际意义 [§1]
- 实验规模远超前序工作: 16 个开源 + 3 个商业 TTS,7 个 ASR 系统,中英双语,fine-tuning + zero-shot + ICL-based 三类场景全覆盖 [§4]
- 商业 API 黑盒验证(ByteDance/Alibaba/MiniMax)和真实环境 over-the-air 测试增强了实用性证据 [§4.4, §4.6]
- 发表于 NeurIPS 2025,同一作者的第三篇 voice protection 工作(POP → SafeSpeech → E2E-VGuard),形成了从 fine-tuning UE 到 zero-shot AE 到 LLM+E2E 的完整防护体系

**不足**:
- 与 SafeSpeech 的对比不够充分: Table 1 中 SafeSpeech 仅在 fine-tuning 场景出现,未在 zero-shot 场景比较(仅 AntiFake 和 POP+ESP)。[agent 解读] 这可能因为 SafeSpeech 在 zero-shot 场景的 SIM 已经很低(F5-TTS 上 0.094),但缺少 LLM-based 模型上的对比
- 心理声学模型的贡献在 ablation 中体现为"去掉后 SNR 变差",但未与 SafeSpeech 的 STOI+STFT 直接对比,难以判断哪种感知优化策略更优
- 发音保护的 targeted ASR 依赖是设计性妥协,但在产品化场景中(不知道对方用什么 ASR)可能成为瓶颈

**在 KB 语境下的定位**:
从 [[Anti-spoofing and Deepfake Detection]] 的三层防线视角(unlearning + perturbation + watermark)看,E2E-VGuard 属于 perturbation 层的最新进展,将防护从 DNN-TTS 扩展到 LLM-TTS。与 SafeSpeech(数据端,mel-level 扰动)互补,E2E-VGuard(特征端,encoder-level 扰动)更适合对抗新一代 codec LM 架构。两者共同构成了语音主动防护领域最完整的方案组合。

从 [[Voice Cloning Taxonomy]] 的角度看,E2E-VGuard 首次在 19 个 TTS 模型(覆盖 SA/ZS 两大类 + ICL-based 子类)和 3 个商业 API 上验证了防护有效性,是目前覆盖面最广的 voice protection 工作。

## 可复用的 idea

1. **Encoder ensemble + 异质特征提取器**: 将 TTS encoder(VITS/GSV/CosyVoice/StyleTTS2) + SSL 模型(WavLM) + 传统特征(MFCC)组合成 ensemble,利用不同模型在不同特征空间中的互补性提升 transferability。这个策略可用于任何需要跨模型 adversarial transferability 的场景
2. **双层防护(timbre + pronunciation)**: 对 TTS 系统的两个核心输入维度(参考音频 timbre + 文本 pronunciation)分别施加扰动,比单维度防护更难被绕过。这个"多维度覆盖"思路可迁移到多模态防护场景
3. **心理声学频率掩蔽**: 用原始音频计算掩蔽阈值,在掩蔽区域嵌入更多扰动,在敏感区域严格限制,比 L_p norm 更贴近人耳感知。可用于任何需要嵌入不可感知信号的场景(水印、隐写术等)
4. **Targeted timbre + ASR 目标对齐**: 在 targeted 模式下,timbre 目标和 ASR 目标文本都指向同一个 target speaker,使两个优化方向协同而非冲突。这种"目标对齐"设计可迁移到其他多目标对抗优化场景

> [!review] 审阅摘要 (auto, 2026-06-04, checklist v1.1)
> **结论: pass-with-fixes**
> - 可复述: pass -- 方法节含 WHY 解释(为什么不用单代理模型、为什么用 targeted ASR attack、为什么用心理声学模型而非 STOI+STFT); 关键设计选择与 SafeSpeech 的差异分析清晰
> - 可信赖: pass-with-fixes -- 1 个 medium: 实验表格中数据丰富且标注覆盖率 ~90%,但部分 zero-shot 场景缺少 SafeSpeech baseline 对比(原文即如此)
> - 可区分: pass -- 因果解释均标注了 [论文原文] / [agent 解读]; 推断性分析与原文断言边界清晰
> - 可定位: pass -- KB 背景含与 SafeSpeech 的谱系对比和创新增量; frontmatter 字段完整
> - 不污染: pass -- 概念挂接合理(6 个实体页),无新建实体页需求,反向更新为纯 append(key_papers)
