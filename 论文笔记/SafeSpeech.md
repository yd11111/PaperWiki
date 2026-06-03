---
type: paper
tier: deep
title: "SafeSpeech: Robust and Universal Voice Protection Against Malicious Speech Synthesis"
arxiv_id: "2504.09839"
source: "Sources/SafeSpeech.pdf"
authors: [Zhisheng Zhang, Derui Wang, Qianyi Yang, Pengyang Huang, Junhan Pu, Yuxin Cao, Kai Ye, Jie Hao, Yixian Yang]
year: 2025
venue: "USENIX Security 2025"
tags: [voice-protection, adversarial-perturbation, unlearnable-examples, voice-cloning, anti-spoofing, deepfake-defense, data-poisoning, speaker-verification]
concepts: ["[[Anti-spoofing and Deepfake Detection]]", "[[Speaker Verification]]", "[[Speaker Embedding]]", "[[Mel Spectrogram]]", "[[Voice Cloning Taxonomy]]", "[[Speaker Adaptation]]"]
models: ["[[VITS]]"]
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 4 个待确认实体页: [[Speaker Embedding]], [[Anti-spoofing and Deepfake Detection]], [[Speaker Verification]], [[Voice Cloning Taxonomy]], [[Mel Spectrogram]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文属于 voice protection / proactive defense 方向,是 [[Anti-spoofing and Deepfake Detection]] 中"主动防御"分支的新进展。与 KB 中已有的 deepfake detection(被动检测)和 watermarking(可追踪)路线不同,SafeSpeech 走的是"让数据不可学"的主动扰动路线。KB 中已记录的 [[Voice Cloning Taxonomy]] 将 cloning 分为 Speaker Adaptation / Few-shot / Zero-shot / Multilingual 四类,SafeSpeech 的核心贡献在于从 zero-shot 防护扩展到 fine-tuning 防护,覆盖了 taxonomy 中更广的攻击面。
>
> **已有认知**: [[Speaker Verification]] 页记录了 ECAPA-TDNN 等 speaker encoder 用于计算 SECS/SIM 的标准做法,SafeSpeech 正是以此作为防护效果评估的核心指标。[[Speaker Embedding]] 页(confirmed)记录了 speaker embedding 在 TTS 中的注入方式,理解这些机制有助于理解 SafeSpeech 为何选择在 mel spectrogram 层面施加扰动。[[Mel Spectrogram]] 页记录了 mel 频谱作为 TTS 中间表示的标准流程,SafeSpeech 的 pivotal objective 正是利用了 mel 距离的通用性。
>
> **创新判断**: KB 中 [[Anti-spoofing and Deepfake Detection]] 页记录了 AntiFake 等前序工作仅覆盖 zero-shot 场景的局限,SafeSpeech 将防护场景扩展到 fine-tuning 是实质性推进。SPEC 技术(KL 引导输出趋近噪声)是该方向的新思路,KB 中尚无类似记录。
>
> 检索命中: [[Speaker Embedding]]✓, [[Anti-spoofing and Deepfake Detection]][待确认], [[Speaker Verification]][待确认], [[Voice Cloning Taxonomy]][待确认], [[Mel Spectrogram]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 通过在原始语音中嵌入不可感知扰动,使 TTS 模型在 fine-tuning 和 zero-shot 两种场景下均无法合成高质量仿冒语音,实现主动式语音隐私保护
> - **路线**: 原始音频 → 代理 TTS 模型(BERT-VITS2) → pivotal objective(mel L1) + SPEC(KL 散度引导输出趋近噪声) + 感知优化(STOI+STFT) → 受保护音频(不可感知扰动) → 攻击者用任意 TTS fine-tune/zero-shot 均生成低质量语音
> - **指标**: BERT-VITS2 上 WER 24.0%→99.6%, SIM 0.604→0.204 [Table 1]; 跨模型迁移到 GlowTTS WER 30.7%→102.4% [Table 1]; zero-shot F5-TTS SIM 0.885→0.094 [Fig 4]; 受保护音频 MOS naturalness 3.190±0.189 [Table 4]; 实时生成扰动仅需 10.6s [§7.3]
> - **可借鉴**: pivotal objective selection 思路 -- 在多目标 TTS 中找收敛最快且通用的单一目标(mel L1)作为扰动优化的锚点,可迁移到其他需要通用对抗扰动的场景
> - **局限**: 扰动依赖 L_p norm 约束(epsilon=8/255),极端降噪(DEMUCS)可部分削弱保护效果(WER 降至 57.3%); 物理世界需要额外播放设备; 对 FishSpeech 的 zero-shot 保护相对较弱(SIM=0.301)

## 核心问题

SafeSpeech 要解决的核心问题是: **现有的语音反克隆方法(AntiFake、VSMask、AttackVC)仅能防护 zero-shot 推理场景下的 timbre 相似度,无法防止攻击者通过 fine-tuning 获得高质量合成语音**。具体而言存在三个缺口 [§1]:

1. **场景缺口**: 之前的方法基于 adversarial examples,只在推理阶段干扰 speaker embedding 提取,一旦攻击者用 fine-tuning(更常见且质量更高)就失效
2. **质量缺口**: 之前的方法只降低 timbre similarity(SA1),但合成语音仍然可用(高质量但不像目标人),攻击者可用于其他目的(如搜索相似声音的受害者、语音助手诈骗)
3. **鲁棒性缺口**: 之前的 adversarial perturbation 方法在面对 robust training 技术(adversarial training、data augmentation)时效果大幅下降

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

SafeSpeech 的总体思路是: 在用户上传语音前,用一个代理 TTS 模型生成专门的扰动 delta,嵌入到原始音频中。当攻击者用受保护的音频去 fine-tune 任何 TTS 模型时,模型会"以为没什么可学的"(unlearnable),从而只能输出低质量噪声 [§4, Fig 2]。

核心优化目标 [§4, Eq. 2]:

```
arg min L(G(x+delta), x) + alpha * P(x+delta)
s.t. H(G(x+delta)) != H(G(x)),  // 合成语音质量下降
     SV(G(x+delta)) != SV(G(x)), // 合成语音 timbre 不同
     H(x+delta) ≈ H(x)           // 受保护音频听起来不变
```

### 关键设计选择

**1. Pivotal Objective Selection -- 为什么只优化 mel L1 而不优化全部 loss** [§4.1]

[论文原文] TTS 模型通常有多个 loss 函数(如 BERT-VITS2 有 8 个),直接用全部 loss 做扰动优化有两个问题: (a) 某些 loss 不依赖音频输入(如 duration loss 只依赖文本),无法通过音频扰动影响; (b) 多目标优化时梯度方向互相干扰,难以收敛。

[论文原文] 作者提出三个 pivotal function 选择原则 [§4.1]: (a) 可通过扰动优化; (b) 跨 TTS 模型通用; (c) 收敛速度快。实验显示 mel reconstruction loss(L1 距离)满足全部三条: 它与音频直接相关、所有生成式 TTS 都要拟合输入分布、且收敛最快(100 epoch 从 98.8 降到 16.7,其他 loss 几乎不下降) [Fig 3]。

[agent 解读] 这个设计选择之所以有效,本质上是因为 mel L1 是所有 TTS 模型共享的"最低公分母" -- 无论模型用什么架构(VAE/diffusion/flow/AR LM),最终都要让输出接近输入的声学特征。在这个共享目标上做"error minimization"等价于告诉模型"输入已经完美对齐了,不需要学习了"。

**2. Speech PErturbative Concealment (SPEC) -- 为什么要引入 KL 散度** [§4.1]

[论文原文] 仅优化 mel L1 可以使合成语音变得模糊,但仍然有微弱的可辨识发音。为了彻底破坏合成质量(SA2),作者希望模型输出趋近于 Gaussian 噪声,因此引入 KL 散度来度量模型输出与随机噪声分布的距离 [Eq. 6]:

```
L_noise = D_KL(x_hat_mel, z_mel) + ||x_hat_mel - z_mel||_1
```

[论文原文] KL 散度的不对称性在这里是有意义的: D_KL(output||noise) 低意味着输出更像噪声,这正是防护的目标。最终 SPEC 目标 [Eq. 7]:

```
L_SPEC = L_mel + beta * L_noise
```

[agent 解读] SPEC 相比纯 mel 优化的关键差异在于: mel L1 只让模型"学不到有用的东西",而 SPEC 进一步让模型"学到的东西是噪声"。这是从"消极防御"到"积极误导"的转变,类似于数据投毒中从 clean-label poisoning 到 targeted poisoning 的升级。

**3. 感知优化 -- 为什么不能只用 L_p norm** [§4.2]

[论文原文] L_p norm 限制了扰动的绝对幅值,但人耳感知不完全对应幅值大小。作者引入 STOI(时域语音可懂度)和 STFT L2 距离(频域差异)作为感知优化项 [Eq. 9]:

```
L_perception = L_stoi + L_stft
```

最终完整目标 [Eq. 10]:

```
L = L_SPEC + alpha * L_perception
```

[论文原文] alpha=0.05 时在防护效果和感知质量之间取得平衡: MCD=12.516, WER=84.709%, SIM=0.223, SNR=17.791,全部优于 PTA baseline [Fig 5c]。

### 训练策略

**代理模型选择**: 使用 BERT-VITS2 作为代理模型生成扰动 [§6.1],因为它性能好且支持 fine-tuning。生成的扰动迁移到其他 4 个 fine-tuning 模型(StyleTTS2, MB-iSTFT-VITS, VITS, GlowTTS)和 5 个 zero-shot 模型(TorToise-TTS, XTTS, OpenVoice, FishSpeech, F5-TTS)。

**扰动生成算法** [Algorithm 1]:
1. 初始化扰动 delta 在 [-epsilon, epsilon] 内
2. 迭代 max_epoch 步:
   - 用代理模型生成合成语音
   - 计算 L_SPEC + alpha * L_perception
   - 用 sign gradient 更新 delta,clamp 到 epsilon 范围
3. 输出受保护音频 x' = x + delta

**超参数**: epsilon=8/255, alpha=0.05, beta=10 [§5.4]。beta 对效果不敏感(0.01~100 范围内均有效) [Fig 5b],表明 SPEC 的鲁棒性好。

## 实验

| 指标 | SafeSpeech (SPEC) | PTA (best baseline) | Clean | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| WER (↑) | 99.610% | 57.921% | 24.024% | LibriTTS, BERT-VITS2 | [Table 1] |
| SIM (↓) | 0.204 | 0.286 | 0.604 | LibriTTS, BERT-VITS2 | [Table 1] |
| MCD (↑) | 14.771 | 11.193 | 5.444 | LibriTTS, BERT-VITS2 | [Table 1] |
| WER (↑) | 102.407% | 98.410% | 30.725% | LibriTTS, GlowTTS | [Table 1] |
| SIM (↓) | 0.081 | 0.144 | 0.311 | LibriTTS, GlowTTS | [Table 1] |
| SIM (↓, zero-shot) | 0.094 | - | 0.885 | F5-TTS | [Fig 4] |
| SIM (↓, zero-shot) | 0.301 | - | 0.578 | FishSpeech | [Fig 4] |
| Protected audio naturalness | 3.190±0.189 | - | - | User study (80人) | [Table 4] |
| Protected audio similarity to original | 98.333% judged same speaker | - | - | User study | [Table 4] |
| Synthesized MOS (↓) | 1.070±0.161 | 2.008±0.191 | 4.677±0.114 | User study | [Table 3] |

**鲁棒性实验** [§7, Table 6, Table 7]:

| 攻击手段 | WER (↑) | SIM (↓) | vs 无攻击 | 出处 |
| --- | --- | --- | --- | --- |
| 无攻击 (baseline) | 99.610% | 0.204 | - | [Table 6] |
| 频谱门控 (SG) 去噪 | 69.321% | 0.233 | WER 降 30pp | [§7.1.1] |
| DEMUCS 去噪 | 57.329% | 0.284 | WER 降 42pp | [§7.1.1] |
| AudioPure (diffusion-based) | 85.711% | 0.227 | WER 降 14pp | [Table 6] |
| MP3 压缩 | 98.082% | 0.252 | 基本不变 | [Table 6] |
| Adversarial training (rho_a=8/255) | 82.504% | 0.188 | WER 降 17pp, SIM 更低 | [Table 7] |
| Adversarial training (rho_a=16/255) | 113.238% | 0.193 | WER 反而更高 | [Table 7] |

**实时性** [§7.3]: 生成 speaker-specific 扰动仅需 10.606 秒(A800 GPU),之后可连续保护。物理世界测试中,50 dBA 音量下 FishSpeech SIM 降至 0.215。

## 局限性

1. **去噪对抗**: 高级去噪模型(DEMUCS)可将 WER 从 99.6% 降至 57.3%,虽然 SIM 仍低于阈值(0.284 > 0.25 但接近),但保护效果被削弱 [§7.1.1]。[论文原文] 作者认为去噪同时也会去除部分原始说话人信息,形成"双刃剑"效应
2. **物理世界局限**: 实时保护需要额外的 GPU 设备和播放设备(speaker),且需要 ~14 秒的启动时间 [§7.3]。在嘈杂环境中扰动效果可能下降
3. **FishSpeech 相对抗性**: zero-shot 评估中 FishSpeech 的 SIM 仍有 0.301(高于 0.25 阈值) [Fig 4],说明某些 codec-LM 架构对扰动有一定抗性
4. **epsilon 与感知的 trade-off**: 扩大 epsilon 可增强保护但降低感知质量。当前 epsilon=8/255 是折中点,极端场景可能需要更大的扰动 [Appendix D.3]
5. **代理模型依赖**: 虽然迁移性已经验证,但对未来架构(如 full audio LM)的迁移效果未知。作者建议用 model ensemble 但承认计算成本高 [§8]

## 点评

**优势**:
- 第一个将语音保护从 inference-only(adversarial examples)扩展到 training-stage(unlearnable examples)的系统,场景覆盖面大幅提升
- SPEC 的 KL 散度引导思路新颖,将防护目标从"降低相似度"升级为"让输出变成噪声",在 WER 指标上几乎达到 100%(完全不可用),远超 AntiFake 的 48.966%
- 实验设计全面: 10 个 TTS 模型(5 fine-tuning + 5 zero-shot)、2 个数据集、3 层鲁棒性(data/model/physical)、主观评估 80 人
- 效率突破: pivotal objective 选择将扰动生成时间从 10.3s 缩短到 4.0s(减少 61.2%),实现了实时保护的可能

**不足**:
- 对 codec LM 架构(如 FishSpeech)的保护效果不如传统 TTS,可能因为 codec LM 的 VQ 离散化天然过滤了连续扰动
- 感知优化依赖 STOI 和 STFT 两个代理指标,但未报告 PESQ 或 VISQOL 等更全面的感知质量指标
- 物理世界实验只在安静室内(22 dBA 背景噪声)测试,未验证嘈杂环境

**在 KB 语境下的定位**:
从 [[Anti-spoofing and Deepfake Detection]] 的视角看,SafeSpeech 代表了从"被动检测"到"主动防护"的范式转变。与 [[Voice Cloning Taxonomy]] 中四类 cloning 方法对应,SafeSpeech 是目前唯一同时覆盖 Speaker Adaptation(fine-tuning)和 Zero-shot VC 两种攻击场景的防护方案。

## 可复用的 idea

1. **Pivotal objective selection**: 面对多目标优化时,分析哪个 loss 对扰动最敏感(收敛快)且最通用,只优化它。这个策略可用于任何需要跨模型 transferable perturbation 的场景
2. **KL 散度引导输出趋近目标分布**: 不仅让模型"学不到东西",还引导它"学到噪声"。这个思路可迁移到其他数据保护场景(图像、文本)
3. **Perception-aware perturbation**: 用 STOI + STFT 替代纯 L_p norm 做感知约束,在保护效果和可用性之间找平衡
4. **Single-sample universality**: 只需一段目标说话人音频生成扰动,即可 pad/truncate 到其他样本,实现持续保护 [§7.3]

> [!review] 审阅摘要
> 待审阅。
