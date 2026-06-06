---
type: paper
tier: deep
title: "BiSinger: Bilingual Singing Voice Synthesis"
arxiv_id: "2309.14089"
source: "Sources/BiSinger.pdf"
authors: [Huali Zhou, Yueqian Lin, Yao Shi, Peng Sun, Ming Li]
year: 2023
venue: "IEEE Conference 2023"
tags: [SVS, bilingual, code-switch, singing-voice-synthesis, multilingual, DiffSinger, language-independent-representation, dataset-adaptation, singing-voice-conversion]
concepts: ["[[SingingVoiceSynthesis]]", "[[PhonemeRepresentation]]", "[[MusicalScoreEncoder]]", "[[SVSEvaluationMetrics]]", "[[DiffusionModel]]"]
models: ["[[论文笔记/BiSinger|BiSinger]]"]
tasks: ["[[Cross-lingualVoiceCloning]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 5 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。

**谱系定位**: BiSinger 属于 [[SingingVoiceSynthesis]] 中 "Hi-Fidelity Synthesis" 大类的多语言扩展方向。在 SVS 演进谱系上,它位于 DiffSinger (2022) 之后,尝试将单语 SVS 系统扩展至中英双语 code-switch 场景。与 [[Cross-lingualVoiceCloning]] 任务有交叉 --- 后者聚焦 TTS 跨语言音色迁移,BiSinger 则聚焦 SVS 跨语言发音统一,二者共享 "语言无关表示" 这一技术路线。

**已有认知**:
- [[SingingVoiceSynthesis]] [待确认]: SVS 与 TTS 的本质差异在于乐谱约束 (绝对音高、节拍对齐),数据稀缺性更严重。级联管线 (Acoustic Model → Vocoder) 在低资源场景下仍占优 [Pan et al., 2026, §B.2]。
- [[PhonemeRepresentation]] [待确认]: 跨语言统一表示有多种方案: IPA、CMU dictionary、byte representation、phoneme embedding mapping。CMU dictionary 是英文标准,BiSinger 创新地将其扩展为中英共享音素集。
- [[MusicalScoreEncoder]] [待确认]: SVS 系统的乐谱编码需融合音素 + MIDI pitch + note duration,对齐方式有外部强制对齐 (MFA)、可学习单调对齐等。
- [[DiffusionModel]] [待确认]: DiffSinger 基于浅扩散 DDPM 生成 mel spectrogram,是级联 SVS 系统的标杆。
- [[SVSEvaluationMetrics]] [待确认]: SVS 评估五维度 --- 准确性 (FFE, F0 RMSE)、音质 (MCD)、自然度 (MOS)、相似度 (SIM)、可控性。
- [[Cross-lingualVoiceCloning]] ✓: 跨语言核心挑战是 "解耦音色与语言特征"。当前 TTS 领域主流方案是多语言 LLM + 统一 tokenizer;BiSinger 采用的 CMU 音素映射是更早期的路线。

**创新判断**: 相对于 KB 已知内容,BiSinger 的创新点在于: (1) 将 CMU dictionary 从 TTS 跨语言迁移到 SVS 场景 (KB 中尚无 SVS 领域的跨语言音素统一案例); (2) 用 SVC 技术做跨语言数据增强 (将少量英文歌声转换为中文歌手音色); (3) 用 pitch shift 将语音数据伪造为歌声数据。这三个方案都是工程导向的 data augmentation,理论创新有限但实用性强。

> 检索命中: [[SingingVoiceSynthesis]][待确认], [[PhonemeRepresentation]][待确认], [[MusicalScoreEncoder]][待确认], [[SVSEvaluationMetrics]][待确认], [[DiffusionModel]][待确认], [[Cross-lingualVoiceCloning]]✓ | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 基于 DiffSinger 的中英双语 SVS 系统,通过 CMU 音素统一表示 + SVC 数据增强 + 语音 pitch shift 伪歌声,在单模型中实现双语和 code-switch 歌声合成
> - **路线**: 乐谱 + 歌词 → CMU 音素转换 + Language/Style Token → Language-Style-Infused Encoder (DiffSinger + ESM) → Mel Spectrogram → HiFi-GAN → 波形
> - **指标**: System 5 (全数据) ALL MOS 3.24 [Table 7]; System 3 (SVC 增强) EN MOS 3.40, MIX MOS 3.32 [Table 7]; GT MOS 3.73 [Table 7]; 英文 WER 从 System 2 ~80% 降至 System 3/5 ~20% [Fig 3]
> - **可借鉴**: (1) CMU 音素 + MFA 比例分配时长的标注适配方案可迁移到其他双语 SVS; (2) SVC 做跨语言数据增强的思路可用于任何低资源歌声场景; (3) 语音 pitch shift 伪歌声思路可用于利用丰富的语音数据辅助 SVS 训练
> - **局限**: (1) GT 本身是 SVC 转换的,非真人歌声,MOS 上限受限 (GT ALL MOS 3.73); (2) 英文音素替代效果差 (某些音素如 /TH/, /V/, /DH/ 中文无对应); (3) 仅支持中英双语; (4) 未开源模型

## 核心问题

1. **没有中英双语歌声数据集,如何训练双语 SVS 模型?** BiSinger 不收集新数据,而是通过三种数据适配策略复用已有单语歌声数据和双语语音数据。
2. **中英两种语言的音素体系不同,如何在一个模型中统一?** 采用 CMU Pronunciation Dictionary 作为跨语言共享音素集,配合 language ID token 保留语言特性。
3. **语音数据缺少歌声特征 (旋律、节奏),如何让它辅助 SVS 训练?** 通过 pitch shift 合成伪歌声,增加音高变化,配合 style ID token 区分语音/歌声/伪歌声。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

BiSinger 基于 DiffSinger [10] 的级联 SVS 管线 [§3.1]:

```
乐谱 S → 转录 (L, Pnote, Dnote) → 变换处理器 T → (LGT, PHO, P̂note, D̂note)
→ Language-Style-Infused Encoder → Denoiser (DiffSinger) → Mel Spectrogram X̂
→ HiFi-GAN → 波形
```

**变换处理器 T** [§3.1, Eq.1]: 将歌词 L 转为 CMU 音素序列 PHO 和对应 language token LGT,同时将 note pitch 和 note duration 扩展到音素级对齐 (P̂note, D̂note)。

[agent 解读] 这是一个标准的级联 SVS 改造: 核心改动集中在编码器侧 (加入多语言/多风格信息),声学模型 (DiffSinger denoiser) 和声码器 (HiFi-GAN) 基本不变。

### 关键设计选择

#### 1. Language-Independent Representation (语言无关表示) [§3.2]

**选择 CMU dictionary 而非 IPA**: 将中文拼音和英文单词统一映射到 CMU 音素集 [§3.2, Table 1]。

**WHY**: [论文原文] 受 [14] (Cai et al., 2023 跨语言 TTS) 启发,CMU 音素是中英共享的最小公倍数 --- 绝大部分 CMU 音素在两种语言中都有对应发音,实现跨语言知识共享 [§3.2]。

**HOW**: 中文: 汉字 → Pypinyin → 拼音 → Pinyin-to-CMU 映射表 → CMU 音素。英文: 单词 → CMU dictionary 直接查询。歌唱中音高由乐谱决定而非声调/重音,因此不保留 tone/stress 信息 [§3.2]。

**局限**: 6 个英文音素 (/TH/, /Y/, /IH/, /DH/, /V/, /OY/) 在中文中完全缺失,只能用近似音素替代,效果较差 [§4.4, Fig 5]。[agent 解读] 这是 CMU 统一方案的根本瓶颈 --- 音素集的交集不够大,替代策略无法弥补跨语言语音学差异。

#### 2. Language-Style-Infused Encoder [§3.3]

在 DiffSinger 编码器基础上增加三个组件:

**Language Embedding + ESM (Embedding Strength Modulator)** [§3.3]: 
- Language ID token (0=英文, 1=中文) → 256 维 language embedding → 与音素/note/duration embedding 一起编码
- ESM [26]: 多头注意力 + FFN 融合模块,将静态 language embedding 转化为动态 language embedding,捕捉 "phonology 和 language 的动态强度" [论文原文, §3.3]

[agent 解读] ESM 的作用相当于让同一个 CMU 音素在中/英两种语言上下文中获得不同的嵌入强度,缓解 "同一音素在两种语言中发音不同" 的问题。

**Style Embedding** [§3.3]:
- Style ID token (0=语音, 1=歌声, 2=伪歌声) → 256 维 style embedding
- 与 speaker embedding 和编码器输出融合,送入辅助解码器或 denoiser

**WHY**: [论文原文] 语音数据和歌声数据在韵律特征上差异显著 (语音过于平稳、速度快),style token 区分三类数据以避免合成的歌声听起来像语音 [§3.3]。

#### 3. 数据集适配 [§3.4]

**方法 A: 音素级乐谱标注适配** [§3.4.1]

将 M4Singer 的拼音标注转换为 CMU 音素标注。关键问题是 "拼音分裂为多个 CMU 音素后,时长如何分配"。

- **均分策略**: 将原始音素时长平均分配给分裂后的 CMU 音素 → 简单但不自然 (元音在歌唱中通常比辅音长得多) [§3.4.1]
- **比例分配策略**: 用 MFA 预对齐获取辅音/元音比例,按比例分配原始时长 → 保持了辅音短/元音长的自然比例 [§3.4.1, Table 2]

[论文原文] 对拼音 initial (声母),保持 MFA 给出的绝对时长;对 final (韵母),分裂为 CMU 音素后按 MFA 比例分配 [§3.4.1]。

**方法 B: SVC 音色转换** [§3.4.2]

用 so-vits-svc 将 NUS-48E (小规模英文歌声, 1.91h, 12 歌手) 转换为 M4Singer 全部 20 个歌手的音色,扩增为 1.91h x 20 = 38.2h [§3.4.2]。

**WHY**: [论文原文] 英文歌声数据集规模太小且音色不一致,SVC 转换可以 (1) 大幅扩增英文歌声数据量; (2) 让每个歌手同时拥有中英歌声,保证音色一致性 [§3.4.2]。

转换时根据声部类型 (Bass/Baritone/Tenor/Alto/Soprano) 调整半音数以保持自然音域 [§3.4.2, Table 3]。

**方法 C: Pitch Shift 伪歌声** [§3.4.3]

利用双语语音数据集 DB-4 (中/英各 ~12h/6h):
1. MFA 对齐获取音素时长和词边界
2. Parselmouth 提取 F0,按词边界平均化得到音符级 pitch
3. 用 WORLD vocoder 替换原始 F0 为预定义的 10 种旋律频率,合成伪歌声 [§3.4.3]

**WHY**: [论文原文] 语音数据虽然缺乏歌唱特性,但双语语音数据容易获取;pitch shift 增加了音高变化,防止合成歌声 "过于平稳缺乏音乐表现力" [§3.4.3]。

### 训练策略

- 基于 DiffSinger 模型和 M4Singer 配置训练 [§4.2]
- 使用 M4Singer 预训练的 HiFi-GAN 作为声码器 [§4.2]
- 所有音频降采样至 24kHz [§4.1]
- 5 个系统配置逐步叠加组件/数据: Model 1/2 (原始/改进 DiffSinger) x Corpus 1-4 (不同标注+数据组合) [§4.2, Table 6]

## 实验

| 指标 | System 1 (基线) | System 3 (最佳英文) | System 5 (全数据) | GT | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| MOS ALL | 3.07 | 3.23 | 3.24 | 3.73 | 内部 25x3 test | [Table 7] |
| MOS EN | 2.75 | 3.40 | 3.42 | 3.80 | 内部 test | [Table 7] |
| MOS CN | 3.41 | 2.97 | 3.04 | 3.78 | 内部 test | [Table 7] |
| MOS MIX | 3.06 | 3.32 | 3.24 | 3.60 | 内部 test | [Table 7] |
| MCD ALL | 10.1941 | 9.5923 | 9.6407 | - | 内部 test | [Table 7] |
| F0 RMSE ALL | 0.1858 | 0.1770 | 0.1828 | - | 内部 test | [Table 7] |
| SIM ALL | 0.63 | 0.56 | 0.55 | - | 内部 test | [Table 7] |

**关键发现**:

1. **CMU 统一表示有效但不够**: System 2 (仅中文数据 + CMU 音素) 英文 MOS 仅 2.56,Whisper 将其发音识别为中文 → 共享音素只能部分迁移发音,真实英文数据不可缺少 [§4.3, Fig 3]。
2. **SVC 数据增强效果显著**: 加入 Corpus 3 后英文 MOS 从 2.56 → 3.40 (System 3),WER 大幅下降 [§4.3, Fig 3]。
3. **语音数据有辅助作用**: Corpus 4 (语音+伪歌声) 缓解了中文合成质量下降 --- System 3 CN MOS 2.97 → System 5 CN MOS 3.04 [§4.3]。
4. **比例分配优于均分**: System 2 (比例) vs System 1 (均分) 英文发音偏好测试 System 2 更优 [§4.3, Fig 4]。
5. **音素替代是瓶颈**: 缺失音素 (/TH/, /V/ 等) 的替代导致发音模糊,频繁替代使合成歌声 "不清晰且类似中文" [§4.4, Fig 5]。

**消融实验** [§4.5, Table 8]:
- SVC + LGT (ASM 3) MOS 3.96 vs NUS raw (ASM 1) MOS 3.83 → SVC 音色转换 + language ID 有效
- Pitch shift (ASM 6) MOS 3.87 vs 无 pitch shift (ASM 5) MOS 3.78 → pitch shift 有效

## 局限性

1. **GT 质量受限**: GT 歌声本身是 SVC 转换的而非真人录音,MOS 上限仅 3.73 (偏低),这意味着系统性能的绝对水平难以评判 [§4.3]。
2. **音素覆盖不完整**: 6 个英文音素在中文中无对应,替代策略效果差,是 code-switch 质量的根本瓶颈 [§4.4]。
3. **加入英文数据后中文下降**: System 3 CN MOS 2.97 < System 1 CN MOS 3.41,加入英文 SVC 数据后中文质量显著下降,多语言 trade-off 未解决 [Table 7]。
4. **仅支持中英**: 未验证三语以上场景。
5. **SIM 下降**: 全数据 System 5 SIM 0.55 < System 1 SIM 0.63,多数据多语言训练损害了说话人一致性 [Table 7]。
6. **评估规模小**: 75 测试用例,14 评估者,统计显著性有限。
7. **未开源**: 模型和代码均未公开。

## 点评

BiSinger 是 SVS 领域较早探索双语/code-switch 的工作,核心贡献是工程层面的数据适配三件套 (CMU 音素映射 + SVC 扩增 + 语音 pitch shift)。每个组件都不新 (CMU 统一来自 TTS 跨语言,SVC 和 pitch shift 都是成熟工具),但组合在 SVS 场景下有实用价值。

最大的遗憾是 GT 评估方案 --- 用 SVC 转换的歌声做 ground truth 本身引入了失真,导致 MOS 上限偏低,无法准确评估系统真实水平。此外,中英 trade-off (加英文则中文降) 是多语言系统的老问题,本文未提出有效解决方案。

与 KB 中的 [[Cross-lingualVoiceCloning]] 对比,BiSinger 采用的 CMU 映射路线属于较早期方案。当前 TTS 领域的跨语言主流已转向 LLM + unified tokenizer (如 CosyVoice 3, X-Voice),SVS 领域也逐渐跟进 (如 UniVocal)。BiSinger 的数据增强策略 (尤其 SVC 扩增) 在低资源歌声场景下仍有参考价值。

## 可复用的 idea

1. **SVC 音色转换做跨语言数据增强**: 将少量目标语言歌声 SVC 转换为所有源语言歌手的音色,同时解决数据量和音色一致性问题。可迁移到任何需要跨说话人/跨语言扩增歌声数据的场景。
2. **MFA 比例分配时长标注**: 当标注格式需要从一种音素集转换为另一种时,用 MFA 预对齐获取辅音/元音比例再按比例分配,比均分更符合发音规律。可迁移到任何涉及音素集转换的 SVS/TTS 工作。
3. **Style ID token 区分语音/歌声/伪歌声**: 简单的 embedding tag 即可让模型区分不同域的数据,防止域混淆。可迁移到任何混合不同类型音频数据训练的场景。
4. **Pitch shift 伪歌声**: 用 WORLD vocoder 替换语音 F0 为预定义旋律频率,快速生成音高丰富的伪歌声数据,利用大量语音数据辅助 SVS。

## 审阅

> [!review] 审阅 (2026-06-06, auto)
> **结论**: pass-with-fixes
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 3 个核心设计选择有 WHY/HOW 解释,速查可借鉴具体 |
> | 可信赖 | pass | 实验数字与 PDF 交叉验证一致,出处覆盖率 ~85% |
> | 可区分 | pass | [论文原文]/[agent 解读] 标签覆盖率 ~85% |
> | 可定位 | pass | KB 谱系定位具体,创新判断有对比基准 |
> | 不污染 | pass | 概念引用恰当,未提议不必要的新页 |
> 
> Issues: 3 (high: 0, medium: 1, low: 2)
> 详见 `_review/BiSinger-review.yml`
