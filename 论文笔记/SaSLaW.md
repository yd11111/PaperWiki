---
type: paper
tier: deep
title: "SaSLaW: Dialogue Speech Corpus with Audio-visual Egocentric Information Toward Environment-adaptive Speech Synthesis"
arxiv_id: "2408.06858"
source: "Sources/SaSLaW.pdf"
authors: [Osamu Take, Shinnosuke Takamichi, Kentaro Seki, Yoshiaki Bando, Hiroshi Saruwatari]
year: 2024
venue: "Interspeech 2024"
tags: [TTS, corpus, dialogue, Lombard-effect, entrainment, environment-adaptive, speech-chain, prosody, style-adaptation]
concepts: ["[[GlobalStyleTokens]]", "[[ProsodyModeling]]", "[[StyleTransferinTTS]]", "[[F0Modeling]]", "[[NeuralVocoder]]"]
models: []
tasks: []
datasets: []
kb_context_sources: 5
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 2 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: 本文属于"环境自适应 TTS"方向,是 Prosody Modeling 中较少被探索的环境条件(噪声+对话者)驱动的风格适应分支。与 KB 中已有的 Style Transfer in TTS 和 Global Style Tokens 方向直接相关 --- GST 是本文 EA-TTS 模型的核心组件,用于从环境音频中预测说话风格向量。
>
> **已有认知**:
> - [[ProsodyModeling]] (confirmed): 将韵律变化源分为 text content / speaker / prosody-style-emotion / channel-noise 四类。本文聚焦第四类(环境噪声+对话者)对韵律的影响,这在 KB 中对应 Lombard effect (噪声引起的非自主音量提升) 和 entrainment (对话者间韵律趋同)。目前 KB 的 ProsodyModeling 页面未专门覆盖这两个现象。
> - [[GlobalStyleTokens]] [待确认]: GST 通过 reference encoder + style token bank 无监督发现风格维度。本文的 Env-to-style predictor 直接使用 GST 架构,但将 reference audio 替换为 hearing audio(对话者所听到的环境+对方语音混合信号),是 GST 从"风格迁移"到"环境适应"的功能扩展。
> - [[NeuralVocoder]] (confirmed): HiFi-GAN 作为本文所有 EA-TTS 模型的波形合成后端。
> - [[F0Modeling]] [待确认]: F0 是本文分析语料库中环境适应行为的关键特征之一,用于衡量对话者间的韵律趋同程度。
> - [[StyleTransferinTTS]] [待确认]: 本文的 EA-TTS 可视为 Style Transfer 的一种特殊形式 --- 风格不由参考音频或标签决定,而由实时环境感知驱动。
>
> **创新判断**: 本文的新颖性在于 (1) 构建了包含同步第一人称音视觉信息的自发对话语料库, (2) 将环境感知信号(hearing audio)作为 TTS 的风格条件输入,这在 KB 已有的风格控制方法中未见先例 --- 现有方法的风格条件来源为参考音频、文本描述或指令,而非实时环境感知。
>
> 检索命中: [[ProsodyModeling]]✓, [[NeuralVocoder]]✓ | 过滤: [[GlobalStyleTokens]](pending-review), [[F0Modeling]](pending-review), [[StyleTransferinTTS]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 构建了首个包含对话者第一人称听觉+视觉同步记录的自发对话语料库 SaSLaW,验证了将 hearing audio 作为环境条件输入的 EA-TTS 模型能生成适应噪声环境的语音
> - **路线**: Text + Hearing Audio → FastSpeech 2 (Encoder-Decoder) + GST Style-Token Layer + Env-to-style Predictor → Mel Spectrogram → HiFi-GAN → Waveform
> - **指标**: AB preference test 显示 FS2-predsty/FS2-predsty-ptrn 在 noisy 环境下显著优于 vanilla FS2 [Fig 6]; 客观评估显示 EA-TTS 模型的合成语音 RMS/F1/spectral tilt 随噪声等级变化 [Fig 5]
> - **可借鉴**: (1) 用 loudspeaker 阵列+真实噪声数据模拟扩散噪声环境的录制方法; (2) 收集脉冲响应(IR)+环境噪声以构建可复现的主观评估样本; (3) 用信号处理合成的伪环境自适应数据做预训练以增强小数据下的泛化
> - **局限**: 语料规模极小(4对说话人, ~30min/pair); 仅日语; 未探索视觉信息的建模; 安静环境下 EA-TTS 反而比 vanilla FS2 差; 无 MOS 绝对分数报告

## 核心问题

本文要解决的核心问题是: **如何让对话系统中的 TTS 像人类一样根据环境(噪声、对话者)自适应地调整说话方式?**

人类在面对面对话中会自然地根据环境因素调整语音特征 --- 在嘈杂环境中提高音量和清晰度(Lombard effect [§1]),根据对话者的语音特征调整自己的韵律(entrainment [§2.1])。现有 TTS 系统在安静环境下生成的"平均化"语音无法适应这些场景 [§1]。

实现 Environment-Adaptive TTS (EA-TTS) 面临的关键缺口是 **缺乏合适的训练数据** --- 现有语料库要么只关注朗读式 Lombard 语音 (Hurricane [19]) 而非自发对话,要么缺乏第一人称音视觉感知记录 (CEJC [22] 用第三人称),要么不面向 TTS 使用 (EgoCom [24], EasyCom [23]) [§2.2, Table 1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 语料库构建方法

**录制配置 [§3.2]**: 两名参与者在室内面对面坐着对话,间距 1.5-3 m。每人佩戴三种设备:
- Close-talking microphone: 记录说话者的语音(speak 通道)
- Ear-mounted binaural microphone: 记录说话者听到的声音(listen 通道,包含环境噪声+对方语音)
- Head-mounted camera: 记录说话者看到的画面(watch 通道)

六个传感器同步录制(44.1 kHz 采样率,30 fps 视频) [§3.2]。增益在所有录音中保持固定,以捕捉因环境引起的音量变化 [§3.2]。

**环境噪声模拟 [§3.2]**: 8 个扬声器环绕参与者放置,播放来自 DEMAND 数据集 [26] 的真实环境噪声的不同片段,模拟扩散性环境噪声。[论文原文] 作者选择这种方式而非简单混音,是为了让参与者真正"听到"并自然反应环境噪声,从而产生真实的 Lombard 效应和 entrainment。

**对话设计 [§3.2]**: 两名参与者根据给定的主题和角色(如旅游指南)进行 5-8 轮的即兴对话。语音使用 pyannote.audio [27] 自动分段,Whisper [28] 转写后人工校正。

**评估数据采集 [§3.3]**: 录制了说话者到听话者位置的脉冲响应(IR)和纯环境噪声音频。主观评估时,合成语音与 IR 卷积后加上录制的噪声,模拟听话者在实际环境中听到的效果。[论文原文] 这样做是因为评估者应基于"在听话者位置听到的声音"来判断语音在环境中的合理性,而非评估合成语音本身 [§3.3]。

### 语料库分析

**参与者 [§4]**: 4 对日语说话人(3 对男-男, 1 对女-女),每对约 30 分钟。论文分析了 spk01-spk02 (男) 和 spk05-spk06 (女) 两对。

**噪声适应 (Lombard effect) [§4.2, Fig 3]**: 
- F1 频率: 所有说话人从 quiet → noisy 显著增加
- RMS: 除 spk05 外,从 quiet → noisy 显著增加
- Spectral tilt: 女性说话人从 quiet → noisy 显著增加,男性趋势不同
- [论文原文] 这些特征在说话人间存在共性也存在差异,说明纯规则/信号处理方法难以准确模拟环境适应行为 [§4.2]

**对话者适应 (Entrainment) [§4.2, Table 2]**: 
- 在 moderate 和 noisy 环境中,说话者的 RMS 和 F0 与对话者上一句的 RMS 和 F0 显著相关 (spk01-spk02 noisy: RMS r=0.68, F0 r=0.47)
- Quiet 环境中相关性不显著
- [论文原文] 恶劣听觉环境可能激发了这种从 quiet → moderate/noisy 的相关性增强,这与先前关于对话兴奋度与 entrainment 关系的研究一致 [15]

### 整体架构: EA-TTS 模型 [§5.1, Fig 4]

基于 FastSpeech 2 [32] + HiFi-GAN [33],扩展为三种变体:

1. **FS2**: 标准 FastSpeech 2,在 SaSLaW 数据上 fine-tune。无环境输入。
2. **FS2-predsty**: 在 FS2 基础上增加:
   - **Style-Token Layer**: 使用 Global Style Token [35] 从 target utterance 提取固定长度的 style vector
   - **Env-to-style Predictor**: 4 层可训练卷积 + energy extractor,从 hearing audio(对话者上一轮的听觉输入,混合了环境噪声和对方语音)预测 style vector
3. **FS2-predsty-ptrn**: 与 FS2-predsty 相同架构,但增加了伪环境自适应数据的预训练

### 关键设计选择

**为什么用 hearing audio 而非分离的噪声/对话者信号作为输入?** [agent 解读] Hearing audio 是混合信号,同时包含环境噪声和对话者语音,这与人类的听觉感知一致 --- 人类并非先分离噪声和对话者信号再分别适应,而是对整体听觉感知做出反应。这种设计避免了信号分离的误差,也更符合 speech chain 框架 [4] 中"基于感知的产出调整"的理论。

**为什么选择 GST 而非其他风格建模方法?** [agent 解读] GST 的无监督特性使其不需要环境标签,适合建模难以显式标注的环境适应维度。GST 的 style token bank 可以自动发现与环境相关的风格维度(如音量、清晰度),而 Env-to-style predictor 学习从环境输入到这些维度的映射。

**伪数据预训练策略 [§5.1]**: 用 JSUT (TTS 语料) + DEMAND (噪声数据集) 通过信号处理合成伪环境自适应数据 --- 为每条语音分配随机噪声并提高其 spectral tilt 以增强在对应噪声中的可懂度。[论文原文] 虽然信号处理合成的语音特征偏离真实 Lombard 语音,但由于可扩展性,这些数据可有效预训练 EA-TTS 模型 [§5.1]。

### 训练策略

- 模型参数: FS2 35M, FS2-predsty/FS2-predsty-ptrn 61M [§5.1]
- 预训练: JSUT 上 900k 步 (FS2, FS2-predsty); 伪数据上 900k 步 (FS2-predsty-ptrn); 均 ≤ 3 天 [§5.1]
- Fine-tune: SaSLaW 上 100k 步, ≤ 12 小时 [§5.1]
- 数据划分: spk01 299/49 train/test, spk06 443/64 train/test; train/test 在 hearing audio 的环境噪声类型上无重叠 [§5.1]
- 训练目标: $L = L_{\text{TTS}} + L_{\text{sty}}$, 其中 $L_{\text{TTS}}$ 是 FastSpeech 2 原始损失, $L_{\text{sty}}$ 是 style vector 预测的 L1 损失 [§5.1]
- 硬件: 单张 NVIDIA GeForce RTX 4090 [§5.1]

## 实验

| 指标 | 本文 EA-TTS | Baseline (FS2) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| 客观: RMS/F1/spectral-tilt 随噪声等级变化 | FS2-predsty/predsty-ptrn: spk06 的 RMS/F1/spectral-tilt 在 noisy 显著高于 quiet; spk01 仅 RMS 显著 | 无显著差异 | SaSLaW test set | [§5.2, Fig 5] |
| 主观: AB preference (noisy) | FS2-predsty, FS2-predsty-ptrn 显著优于 FS2 (两位说话人) | - | SaSLaW test set | [§5.3, Fig 6] |
| 主观: AB preference (quiet) | FS2 显著优于 FS2-predsty, FS2-predsty-ptrn (两位说话人) | - | SaSLaW test set | [§5.3, Fig 6] |
| 主观: AB preference (FS2-predsty-ptrn vs FS2-predsty) | FS2-predsty-ptrn 等于或显著优于 FS2-predsty | - | SaSLaW test set | [§5.3, Fig 6] |

**评估方法 [§5.3]**: AB preference test, 每个 model-pair 配置 72 名评估者, 720 个回答, 通过 Lancers 众包平台招募。评估者选择在环境噪声中听起来更合理的语音。评估准则为自然度和可懂度的综合标准。

## 局限性

1. **语料规模极小**: 4 对说话人, 每对约 30 分钟, train/test 分别仅 299-443/49-64 句。这严重限制了模型泛化能力和实验结论的可靠性 [agent 解读]
2. **仅日语**: 限制了跨语言适用性
3. **安静环境下性能退化**: EA-TTS 在 quiet 环境下显著不如 vanilla FS2 [§5.3, Fig 6]。[论文原文] 认为 FS2 的"平均化"韵律特征在安静环境中反而获得了可懂度优势
4. **未建模视觉信息**: 虽然语料库收集了头戴摄像头的视频,但实验中未使用视觉模态 [§6]
5. **无 MOS 绝对分数**: 仅报告 AB preference 相对偏好,无法判断合成质量的绝对水平
6. **环境噪声为模拟而非真实**: 使用扬声器播放录制噪声,与真实环境仍有差距(论文标注为 near-real noise †) [Table 1]
7. **未探索更现代的 TTS 架构**: 2024 年的工作仍基于 FastSpeech 2,未尝试 LLM-based 或 diffusion-based TTS

## 点评

SaSLaW 的核心价值在于提出了一个新的研究问题 --- 让 TTS 根据环境感知自适应调整语音风格。这个问题在对话机器人的实际部署中确实存在且重要,但此前缺乏系统性的研究和数据支持。

**方法论贡献 > 技术贡献**: 本文的主要价值在于 (1) 定义了 EA-TTS 任务并提出了语料库构建方法论(同步第一人称多模态录制), (2) 通过语料库分析验证了人类语音在不同环境下确实存在可建模的适应行为。相比之下,EA-TTS 模型本身(FastSpeech 2 + GST + 简单的 Env-to-style predictor)在技术上较为朴素。

**数据规模是根本瓶颈**: ~30 分钟/说话人的数据量对于 TTS 来说极为有限(主流研究使用数百到数千小时数据)。这解释了为什么模型性能有限,也意味着在这个数据规模上的实验结论需谨慎解读。

**Quiet 环境退化问题值得关注**: EA-TTS 在安静环境下性能反而下降,说明当前的 Env-to-style predictor 尚未学会"不做适应" --- 理想的 EA-TTS 应在安静环境下退化为标准 TTS。这可能与 GST style vector 的空间结构有关,也可能是小数据导致的过拟合。

**与领域趋势的关系**: 从 KB 背景看,当前 TTS 领域的风格控制已从 GST 演进到 LLM in-context learning。本文的思路(环境感知驱动风格)若要扩展,更合适的框架可能是在 SpeechLM 的 prompt 中编码环境信息,而非依赖显式的 Env-to-style predictor。

## 可复用的 idea

1. **用扬声器阵列+真实噪声模拟扩散环境**: 相比直接混音,让说话人在真实噪声场中对话能诱发更自然的 Lombard effect 和 entrainment,产生更高质量的训练数据
2. **IR + 环境噪声构建可复现评估**: 收集脉冲响应和纯环境噪声,评估时将合成语音卷积后混合,模拟听话者真实听感 --- 这种评估方法可推广到任何环境相关的语音研究
3. **伪环境自适应数据预训练**: 用信号处理合成的"伪 Lombard 语音"预训练可缓解真实环境自适应数据不足的问题 (FS2-predsty-ptrn ≥ FS2-predsty) [§5.3]
4. **将 GST 的 reference 从"目标风格音频"替换为"环境感知信号"**: 这种"感知→风格"的映射思路可扩展到其他场景,如根据房间混响特性或背景音乐调整语音风格

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 方法节含因果解释,设计选择有 WHY 回答 |
> | 可信赖 | pass | 数字标注覆盖率 ~85%,指标名正确 |
> | 可区分 | pass | 来源标注覆盖率高,一处 [论文原文] 混入推断 |
> | 可定位 | pass | KB 背景定位具体,frontmatter models/datasets 为空 |
> | 不污染 | pass | 无新建页,无 factual-error |
> 
> Issues: 5 (high: 0, medium: 2, low: 3)
> 详见 `_review/SaSLaW-review.yml`
