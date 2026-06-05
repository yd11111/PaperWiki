---
type: paper
tier: deep
title: "MelShield: Robust Mel-Domain Audio Watermarking for Provenance Attribution of AI Generated Synthesized Speech"
arxiv_id: "2605.01515"
source: "Sources/MelShield.pdf"
authors: [Yutong Jin, Qi Li, Lingshuang Liu, Jianbing Ni]
year: 2026
venue: "arXiv"
tags: [audio-watermarking, TTS, mel-spectrogram, spread-spectrum, copyright-attribution, provenance, security, vocoder, in-generation-watermarking]
concepts: ["[[Anti-spoofingandDeepfakeDetection]]", "[[MelSpectrogram]]", "[[NeuralVocoder]]", "[[Text-to-SpeechPipeline]]", "[[TTSEvaluation]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 3 个待确认实体页: [[NeuralVocoder]]✓, [[MelSpectrogram]][待确认], [[Anti-spoofingandDeepfakeDetection]][待确认], [[Text-to-SpeechPipeline]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[NeuralVocoder]]✓ | 参考: [[MelSpectrogram]](pending-review), [[Anti-spoofingandDeepfakeDetection]](pending-review), [[Text-to-SpeechPipeline]](pending-review) | 未命中但可能相关: Audio Watermarking(概念库中无独立页)

**谱系定位**: MelShield 处于 Neural Vocoder 与 Anti-spoofing/Deepfake Detection 的交叉地带,具体定位在"in-generation watermarking"子领域。NeuralVocoder 页(confirmed)记录了 HiFi-GAN(GAN-based,单次前传)和 DiffWave(diffusion-based,迭代去噪)作为两大主流 vocoder 范式,MelShield 正是以这两者为目标验证平台,利用它们共享的 mel spectrogram 输入接口进行水印注入。MelSpectrogram 页记录了 mel 作为"声学模型与声码器之间的桥梁"的定位,MelShield 的核心洞察正是将这个桥梁同时作为水印载体。Anti-spoofing 页记录了从被动检测到主动防护的演进趋势,其中 [[论文笔记/TraceableSpeech|TraceableSpeech]](Interspeech 2024)是直接前驱——将水印嵌入 codec LM 的 latent space 实现端到端联合训练,但局限于 codec-token 架构;MelShield 则选择了不同的嵌入点(mel spectrogram),实现了跨 vocoder 架构的通用性。

**已有认知**: NeuralVocoder 页明确了 HiFi-GAN(14M 参数,实时 13.4x @ V100)和 DiffWave(典型 50-200 步迭代)的特性差异。TraceableSpeech 笔记记录了其水印容量最高 4@64 ≈ 24 bit,且仅在 LibriTTS 上验证。

**创新判断**: 对比 TraceableSpeech(codec latent space 嵌入,需联合训练)和 GROOT(diffusion 初始噪声嵌入,限于 diffusion 架构),MelShield 的核心创新在于:(1)选择 mel spectrogram 作为嵌入点,实现 model-agnostic 的 plug-and-play 部署;(2)不依赖神经网络提取器,而是用经典 spread-spectrum + keyed correlation 验证,消除了提取器被逆向工程的风险;(3)reference-based 验证避免了训练额外解码模型。代价是需要存储 reference mel 用于验证。

> [!summary] 速查
> - **一句话**: 在 mel spectrogram 域通过 keyed spread-spectrum 扰动嵌入水印,实现跨 vocoder 架构的 plug-and-play 音频溯源,无需重训练或神经网络提取器
> - **路线**: Text → Acoustic Model → Mel Spectrogram X → [Spread-spectrum embedding: key K + payload m → spreading patterns → adaptive mask → low-energy perturbation] → Watermarked Mel X̃ → Vocoder (DiffWave/HiFi-GAN) → Watermarked Audio; 验证: Suspect Audio → Mel extraction → Residual vs Reference Mel → Keyed correlation test → Decoded payload
> - **指标**: HiFi-GAN @ 32-bit: ACC=1.0, PESQ=3.81 [Table 1]; DiffWave @ 32-bit: ACC=1.0, PESQ=3.63 [Table 1]; HiFi-GAN 鲁棒性: MP3/AAC/N20 全部 ACC=1.0 [Table 3]; DiffWave 鲁棒性: MP3/AAC ACC=1.0, N20 ACC=0.95 [Table 2]; LJSpeech 数据集
> - **可借鉴**: (1) 利用 mel spectrogram 作为 model-agnostic 的水印嵌入点——任何共享 mel 接口的 TTS pipeline 均可复用; (2) reference-based verification 无需训练提取器,可直接用于部署; (3) adaptive energy mask 根据帧能量自适应调节嵌入强度,平衡鲁棒性与不可察觉性
> - **局限**: reference-based 验证需存储每条语音的 reference mel,大规模部署存储开销大; 仅在 LJSpeech(单说话人)上验证; 未测试 codec-based TTS(如 VALL-E 系列); 未与 TraceableSpeech 直接对比; 噪声鲁棒性在低 SNR(5-10dB)下衰减明显

## 核心问题

AI 生成语音的 provenance attribution(溯源归属)面临三个层面的矛盾 [§1]:

1. **后处理水印可绕过** — 现有 post-hoc 方法(AudioSeal, WavMark, Timbre)在生成后作为独立模块叠加水印,部署时可被跳过、禁用或弱化 [§1]
2. **提取器暴露引发安全风险** — 公开或可访问的神经网络提取器便于验证,但也方便对手大规模探测,系统性分析提取器以开发定向移除攻击 [§1]
3. **架构耦合限制通用性** — GROOT 限于 diffusion 架构,TraceableSpeech 限于 codec-token 架构,缺乏跨 vocoder 架构的通用方案 [§1]

MelShield 的核心洞察: mel spectrogram 是几乎所有 mel-conditioned TTS pipeline 的共享中间表示,将水印嵌入在 mel → vocoder 之间,既实现 in-generation(在生成过程中),又保持 model-agnostic(不依赖特定 vocoder 架构)[§1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

MelShield 的工作流分为嵌入和验证两个阶段 [§4, Fig 2]:

**嵌入阶段**: 给定 TTS 前端生成的归一化 log-mel spectrogram X ∈ [0,1]^{C×M}(C 个 mel band,M 个时间帧),MelShield 在选定的中频区域 F = {c_min, ..., c_max-1} 上叠加 keyed spread-spectrum 扰动,生成水印化的 X̃,再送入 vocoder 合成波形 [§4.1]。

**验证阶段**: 合法所有者持有 secret key K 和 reference mel X_ref,对嫌疑音频重建 mel spectrogram,计算残差 Δ = X_det - X_ref,通过 keyed correlation test 逐 bit 解码 [§4.2]。

### 关键设计选择

#### 1. 为什么选择 mel domain 而非 waveform/latent space?

[论文原文] Mel spectrogram 是 Mel-conditioned TTS 架构中"共享的、model-agnostic 的接口" [§2.1]。在 mel 域嵌入水印后,vocoder 作为 black-box 处理,无需修改或重训练 [§4]。

[agent 解读] 这是一个 abstraction layer 的选择: waveform 域嵌入(post-hoc)可被绕过且独立于生成过程; latent space 嵌入(如 TraceableSpeech)需要与特定 codec 联合训练; mel 域处于两者之间——它是生成过程的一部分(in-generation),同时又足够通用(model-agnostic)。代价是依赖 vocoder 对 mel 扰动的容忍度,不同 vocoder 的敏感度不同(DiffWave α=0.025 vs HiFi-GAN α=0.25,差 10x [§5.1])。

#### 2. Spread-spectrum 编码机制

每个用户被分配 L-bit payload m 和 secret key K [§4.1]。编码过程:

1. 用 (K, j) 为种子,为每个 bit j 生成 ±1 的 spreading pattern S_j^{(K)} ∈ {-1,+1}^{|F|×M} [§4.1]
2. 将 bit 值映射为极性: d_j = 2m_j - 1 ∈ {-1, +1} [§4.1]
3. 叠加所有 bit 层为单个水印层: W = (1/√L) Σ_j d_j S_j^{(K)} [Eq. 16]
4. 嵌入: X̃_F = clip[0,1](X_F + α(A ⊙ W)) [Eq. 17]

[论文原文] 1/√L 归一化使得扰动能量不随 payload 长度变化 [§4.1]。

[agent 解读] 这是经典的 CDMA(码分多址)思路在信号处理中的应用: 每个 bit 用正交的伪随机序列编码,解码时通过相关检测恢复。key 的引入使得只有持有正确 key 的验证者才能重建 spreading patterns,无法通过"猜测"提取水印。

#### 3. Adaptive energy mask

[论文原文] 自适应掩码 A ∈ [0,1]^{|F|×M} 从干净区域 X_F 的帧级能量估计得到,沿频率轴广播。它在低能量帧降低嵌入强度,在归一化边界(0 和 1)附近减轻 clipping 伪影 [§4.1]。

[agent 解读] 这是 psychoacoustic masking 的简化版: 能量高的帧能"藏住"更多扰动,能量低的帧(如静音)中的扰动更容易被听到。实际实现中还在嵌入前预留 headroom margin,进一步减少 clipping。

#### 4. Mid-frequency band 选择

水印仅嵌入中频带 F = {20, ..., 55}(共 36 个 mel bin,总共 80 个)[§5.1]。

[agent 解读] 低频 mel bin 承载了语音的基本能量和音调信息,修改会显著影响感知质量; 高频 mel bin 的信息密度低且对噪声/压缩敏感,嵌入后鲁棒性差。中频带在 robustness-transparency trade-off 上最优,这与传统音频水印文献中的经验一致。

#### 5. Reference-based verification 的设计权衡

[论文原文] 验证需要存储的 reference mel X_ref 和 key K。验证者计算残差 Δ = X_det - X_ref(减去全局均值偏移后),用 keyed correlation 逐 bit 解码: s_j(K) = <Δ_F, A ⊙ S_j^{(K)}>,按符号判决 m̂_j = I[s_j(K) ≥ 0] [Eq. 18-19, §4.2]。

[论文原文] 这种设计消除了训练神经网络提取器的需要,降低了大规模探测和对抗逆向工程的风险 [§4.2]。

[agent 解读] Reference-based 验证是本文的核心设计权衡: 优点是完全不依赖可被攻击的神经网络提取器; 缺点是必须为每条生成语音存储 reference mel(约 C×M×4 bytes/utterance),且验证必须知道候选语音对应哪条 reference——这限制了"盲检测"能力(无法在不知道源头的情况下检测水印)。适用于 user-side 或 service-side 的 owner verification 场景,不适用于第三方盲检。

### 训练策略

MelShield **不涉及任何训练** [§4]。嵌入和提取完全基于 closed-form 的 spread-spectrum 信号处理操作。vocoder(DiffWave 或 HiFi-GAN)使用预训练权重,不做任何修改。

[agent 解读] 这是与 TraceableSpeech(需联合训练 codec)和 GROOT(需训练嵌入/提取 CNN)的本质区别。MelShield 是纯信号处理方案,部署成本极低。但也意味着无法通过学习优化水印嵌入策略以适配特定 vocoder 的频率响应特性。

## 实验

### 实验设置

- 数据集: LJSpeech 1.1(13100 条,单说话人英文,22.05kHz)[§5.1]
- Vocoder: 官方 DiffWave + HiFi-GAN(均配置为 22.05kHz)[§5.1]
- Mel 参数: f_min=20Hz, f_max=sr/2, C=80, log-mel → [0,1] 归一化 [§5.1]
- 水印配置: L=32 bits(鲁棒性评估); α=0.025(DiffWave), α=0.25(HiFi-GAN) [§5.1]
- 嵌入频带: F={20,...,55} [§5.1]
- Baseline: AudioSeal, WavMark, Timbre(post-hoc); GROOT(in-generation, diffusion-only)[§5.5]

### 保真度与容量

| 指标 | HiFi-GAN BM | HiFi-GAN @32bit | HiFi-GAN @256bit | HiFi-GAN @512bit | DiffWave BM | DiffWave @32bit | DiffWave @128bit | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PESQ(ref GT) | 3.849 | 3.813 | 3.650 | 3.596 | 3.700 | 3.635 | 3.501 | [Table 1] |
| STOI | 0.996 | 0.992 | 0.975 | 0.960 | 0.977 | 0.972 | 0.962 | [Table 1] |
| MOS(DNSMOS) | 3.799 | 3.751 | 3.671 | 3.651 | 3.657 | 3.652 | 3.624 | [Table 1] |
| ACC | — | 1.000 | 1.000 | 0.997 | — | 1.000 | 0.977 | [Table 1] |

关键发现 [§5.4]:
- HiFi-GAN 容忍 mel 域扰动的能力远强于 DiffWave(支持 1024-bit payload 在 PESQ≥3.5 约束下 ACC>95%)[Fig 5]
- DiffWave 在 128-bit 时 ACC 开始下降(0.977),HiFi-GAN 到 512-bit 仍为 0.997 [Table 1]
- α 增大→ACC 单调提升但 PESQ 下降,存在 elbow region [§5.4, Figs 6-7]

### 鲁棒性(32-bit payload)

**DiffWave 平台** [Table 2]:

| 方法 | MP3-128 | AAC-96 | Scaling | Resample 16k | Bandpass | Lowpass | N20dB | N10dB | N5dB | Echo | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AudioSeal | 0.993 | 0.811 | 0.998 | 0.999 | 0.998 | 0.974 | 0.954 | 0.764 | 0.615 | 0.999 | [Table 2] |
| WavMark | 0.872 | 0.875 | 0.893 | 0.869 | 0.864 | 0.864 | 0.603 | 0.519 | 0.494 | 0.851 | [Table 2] |
| Timbre | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.952 | 0.713 | 0.573 | 1.000 | [Table 2] |
| GROOT | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | — | 0.990 | 0.988 | 0.942 | — | [Table 2] |
| **MelShield** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 0.952 | 0.779 | 0.701 | **1.000** | [Table 2] |

**HiFi-GAN 平台** [Table 3]:

| 方法 | MP3-128 | AAC-96 | N20dB | N10dB | N5dB | Echo | 出处 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AudioSeal | 0.988 | 0.805 | 0.950 | 0.759 | 0.606 | 0.998 | [Table 3] |
| WavMark | 0.868 | 0.873 | 0.596 | 0.513 | 0.488 | 0.847 | [Table 3] |
| Timbre | 1.000 | 1.000 | 0.946 | 0.706 | 0.565 | 1.000 | [Table 3] |
| **MelShield** | **1.000** | **1.000** | **1.000** | 0.782 | 0.705 | **1.000** | [Table 3] |

关键发现 [§5.5]:
- MelShield 在所有非噪声攻击下 ACC=1.0(两个 vocoder 平台)[Table 2, 3]
- HiFi-GAN 平台上 MelShield 在 N20dB 达到完美 ACC=1.0,优于所有 baseline [Table 3]
- GROOT 在极端噪声(N5-N10dB)下鲁棒性更强,但基础 PESQ 明显更低(3.11 vs MelShield 3.41 on DiffWave)[Table 2]——以更大的质量代价换取噪声鲁棒性

### Key-based 安全性

H0(无水印/错误 key)和 H1(正确 key)的 BitAcc 分布清晰分离: H1 集中在 1.00 附近,H0 集中在 0.50 附近(近似高斯),τ=0.61 即可实现低 false positive rate [§5.3, Fig 3]。

## 局限性

1. **Reference storage 开销** — 每条生成语音需存储 reference mel spectrogram 用于验证,大规模部署(如百万级 TTS API)的存储和检索成本可能显著 [agent 解读]
2. **仅 LJSpeech 单说话人验证** — 未在多说话人、跨语言、或 zero-shot voice cloning 场景下验证 [§5.1]
3. **未覆盖 codec-based TTS** — 仅验证了 mel-conditioned vocoder(DiffWave, HiFi-GAN),未测试 VALL-E、SoundStorm 等 codec LM 架构,而这些架构可能不以 mel 为中间表示 [agent 解读]
4. **噪声鲁棒性有限** — N10dB 时 ACC 降至约 0.78,N5dB 降至约 0.70,在高噪声环境(如户外录音)下可靠性不足 [Table 2, 3]
5. **无盲检测能力** — reference-based 验证要求预先知道候选语音对应哪条 reference,无法在未知来源的大量音频中批量筛查水印 [agent 解读]
6. **未讨论二次合成攻击** — 如果对手对水印化音频进行 re-synthesis(如用另一个 vocoder 重新生成),水印存活性未知 [agent 解读]

## 点评

MelShield 的核心价值在于将水印嵌入点从 waveform 上移到 mel spectrogram——一个在 TTS 生态中几乎无处不在的中间表示。这使得它成为目前唯一真正 model-agnostic 的 in-generation 水印方案(TraceableSpeech 限于 codec LM,GROOT 限于 diffusion)。设计上的 elegance 在于完全不依赖神经网络: 嵌入是 spread-spectrum 信号处理,提取是 correlation test,vocoder 作为 black-box 处理。这带来了即插即用的部署便利性,但也限制了对特定 vocoder 频率响应的适应能力。

从实用角度看,MelShield 的 reference-based 验证机制是一把双刃剑: 消除了提取器被逆向工程的风险(对比 GROOT 的公开提取器),但引入了存储和检索的 operational burden,且无法做盲检测。这使得它更适合 owner-side verification(如 TTS 服务商验证自己的输出是否被盗用)而非 third-party detection(如平台扫描上传内容)。

实验设计的一个显著缺陷是仅在 LJSpeech(单说话人,朗读语音)上验证。现代 TTS 的核心应用是 zero-shot multi-speaker synthesis,水印在 speaker embedding conditioning 下的表现、在不同语言和说话风格下的鲁棒性均未知。此外,未与最直接的竞品 TraceableSpeech 正面对比(两者的目标场景高度重叠),是一个明显的遗漏。

## 可复用的 idea

1. **Mel spectrogram 作为 model-agnostic 嵌入点**: 任何需要在 TTS 生成过程中注入元信息(用户 ID、版权标记、情感标签等)的场景,都可以借鉴这一思路——在 mel 层面叠加结构化扰动,让下游 vocoder 自然"传递"这些信息
2. **Keyed spread-spectrum 保证安全性**: 传统 spread-spectrum 水印技术在 AI 音频水印中的应用尚未充分探索,MelShield 展示了经典信号处理方法在深度学习生成系统中仍有实用价值
3. **Adaptive energy mask**: 根据信号局部能量自适应调节嵌入强度的策略,可推广到任何需要在频域/时频域做信息嵌入的场景
4. **α-payload 联合调参策略**: Figs 6-7 展示的 elbow region 分析为实际部署中的参数选择提供了清晰的方法论——先固定质量约束(如 PESQ≥3.5),再找最小 α

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 5 个设计选择均有 WHY 解释,spread-spectrum 机制清晰 |
> | 可信赖 | pass | 数字标注覆盖率约 90%;GROOT 列对齐已修正 |
> | 可区分 | pass | [论文原文]/[agent 解读] 区分清晰,推断均有限定词 |
> | 可定位 | pass | 与 TraceableSpeech/GROOT 的对比有实质内容 |
> | 不污染 | pass | 不触发新建概念页,反向更新仅 append |
> 
> Issues: 3 (high: 0, medium: 1, low: 2)
> 详见 `_review/MelShield-review.yml`

