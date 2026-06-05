---
type: concept
title: "Anti-spoofing and Deepfake Detection"
aliases: [反欺骗检测, Audio Deepfake Detection, Voice Anti-spoofing, 深伪语音检测, Speech Deepfake, 语音伪造检测]
category: "security"
tags: [voice-cloning, deepfake, anti-spoofing, ethics, safety, speaker-verification, TTS]
key_papers: ["[[论文笔记/Survey-VoiceCloning|Azzuni & El Saddik 2025]]", "[[论文笔记/TraceableSpeech|TraceableSpeech]]", "[[论文笔记/TITW|TITW]]", "[[论文笔记/SafeSpeech|SafeSpeech]]", "[[论文笔记/TraceableTTS|Traceable TTS]]", "[[论文笔记/SpeakerIdentityUnlearning|Speaker Identity Unlearning]]", "[[论文笔记/E2E-VGuard|E2E-VGuard]]", "[[论文笔记/Training-freeSpeakerUnlearning|TruS (Lee et al., 2026)]]", "[[论文笔记/SpeakerPoisoning|Speaker Poisoning (Trachu et al., 2026)]]", "[[论文笔记/MelShield|MelShield]]"]
origin_paper: ""
related_concepts: ["[[SpeakerVerification]]", "[[VoiceCloningTaxonomy]]", "[[SpeakerEmbedding]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-02
updated: 2026-06-02
---

## 定义

Anti-spoofing / Deepfake Detection 是指检测和防御通过 TTS、Voice Conversion (VC) 等技术生成的伪造语音的技术和方法。随着 voice cloning 技术的快速发展 (尤其是 zero-shot TTS),这一领域已成为语音安全研究的核心议题。

Azzuni & El Saddik (2025) 在 voice cloning survey 中将此定位为 cloning 技术的伦理对偶: 理解生成方法是发展检测算法的前提,反之检测研究也推动生成方法关注安全性。

## 威胁分类 (Survey Section VI)

### 已知滥用场景
- **身份冒充 (Impersonation)**: 用克隆语音冒充特定个人
- **金融诈骗**: 2019 年 CEO 深伪语音诈骗案,骗取 $243,000 (Forbes 报道)
- **Identity theft**: 训练和发布他人声音的 AI 模型
- **Data laundering**: 用合成语音洗白非法获取的数据
- **Coercion**: 用伪造语音进行威胁或胁迫

### Speech Generation Harm Taxonomy
Hutiri et al. (2024, ACM FAccT) 系统性地提出了语音生成的危害分类学,覆盖 voice cloning 的非伦理用途,是该方向的重要框架性工作。

## 检测方法概览

Survey 本身聚焦于生成端,但引用的相关 survey 覆盖了检测技术:

### 相关综述文献
| Survey | 年份 | 范围 |
|--------|------|------|
| Masood et al. [3] | 2023 | 视觉+音频深伪生成与检测,含 TTS 和 VC |
| Khanjani et al. [4] | 2023 | 音频、文本、视频、图像深伪综合调研 |
| Kadam et al. [2] | 2021 | 语音合成与唇同步 |

### 检测技术路线 (从引用文献推断)
1. **Speaker Verification-based**: 利用 SV 系统判断语音是否来自声称的说话人
   - SV-EER 作为核心指标 (Table X)
   - 局限: 高质量 voice cloning 可欺骗 SV 系统
2. **Artifact Detection**: 检测合成语音的伪影
   - 频谱不自然性、时域不连续性
   - 编解码器引入的量化噪声模式
3. **Adversarial Training Signal**: 部分 voice cloning 系统在训练中引入 real/fake 判别
   - Nakai et al.: 多任务对抗训练同时做真假判别和说话人验证
   - 表明生成和检测可以在同一框架内对抗学习

## Voice Cloning 中的安全意识设计

Survey 中提到的一些系统在设计时考虑了安全性:

- **Speaker verification as quality gate**: 多个系统用 SV 验证克隆质量,这同时可作为检测基线
- **Watermarking**: 部分商业系统 (如 Seed-TTS) 考虑在合成语音中嵌入水印; [[论文笔记/TraceableSpeech|TraceableSpeech]] (Zhou et al., Interspeech 2024) 将水印嵌入与 codec LM TTS 端到端联合训练,实现 proactive traceability; [[论文笔记/MelShield|MelShield]] (Jin et al., 2026) 在 mel spectrogram 域通过 keyed spread-spectrum 扰动嵌入水印,无需训练或修改 vocoder,实现跨架构(DiffWave/HiFi-GAN)的 plug-and-play in-generation watermarking
- **Proactive Voice Protection**: [[论文笔记/SafeSpeech|SafeSpeech]] (Zhang et al., USENIX Security 2025) 在上传前嵌入不可感知扰动,使 TTS 模型在 fine-tuning 和 zero-shot 场景下均无法合成高质量语音,代表从"被动检测"到"主动防护"的范式转变
- **ASVspoof Challenge 系列**: 推动 anti-spoofing 技术发展的标准化竞赛 (Survey 未展开但属于该领域核心)
- **Watermark-free Traceability**: [[论文笔记/TraceableTTS|Traceable TTS]] (Zhao et al., 2025) 提出不依赖显式水印的 TTS 模型溯源方案,通过反转 GAN generator loss 实现 TTS 模型与 discriminator (wav2vec 2.0 + LCNN) 的协同训练,使模型自然产生可追溯的隐式指纹。域外泛化 EER 11.5% vs baseline 18.99%
- **Machine Unlearning (模型级遗忘)**: [[论文笔记/SpeakerIdentityUnlearning|Speaker Identity Unlearning]] (Kim et al., ICML 2025) 首次在 ZS-TTS 中提出 speaker identity unlearning,通过 Teacher-Guided Unlearning (TGU) 直接修改模型权重使其丧失复制特定说话人的能力。与 SafeSpeech(数据端防护)和 Traceable TTS(事后溯源)互补,构成 ZS-TTS 安全的三层防线: 预防(unlearning) + 防护(perturbation) + 溯源(watermark/fingerprint)
- **LLM-TTS + E2E 防护**: [[论文笔记/E2E-VGuard|E2E-VGuard]] (Zhang et al., NeurIPS 2025) 将 SafeSpeech 的防护扩展到 LLM-based TTS 和 ASR-driven E2E 场景。通过 encoder ensemble(6 种 speaker encoder + MFCC)保护 timbre,同时用 ASR 对抗攻击破坏发音,加上心理声学模型保证扰动不可感知。在 19 个 TTS 模型(含 3 个商业 API)上验证有效
- **Training-free Inference-time Unlearning**: [[论文笔记/Training-freeSpeakerUnlearning|TruS]] (Lee et al., 2026) 将 speaker unlearning 从训练范式推向推理范式。通过在 F5-TTS DiT blocks 的 FFN 输出上做动态 activation steering,用 ID-prototype (retain speaker 均值) 定位 identity 方向后投影减法抑制 opt-out speaker 身份。0 GPU 小时训练成本,首次实现 unseen opt-out speaker 的 unlearning (SIM-UO 0.668→0.488),与 TGU (模型级) 互补
- **Continual Speaker Unlearning**: [[论文笔记/ContinualSpeakerUnlearning|CORTIS]] (Kim et al., 2026) 解决了 TGU 在持续增量遗忘场景下的"灾难性再学习"问题 -- 逐个遗忘 speaker 时,先前已遗忘的 speaker 身份会复活。通过 Fisher-information 对比参数定位 + 累积正交子空间投影,使每次新遗忘不干扰先前遗忘效果。VoiceBox 上 5 次顺序遗忘后所有 forget speaker SIM < 0.18,每请求仅 3.5 小时(恒定成本)。与 TGU(一次性遗忘)和 TruS(推理时遗忘)共同构成三种 unlearning 范式: 训练时一次性 / 训练时增量 / 推理时无训练
- **Speaker Poisoning (StyleTTS2)**: [[论文笔记/SpeakerPoisoning|Trachu et al. (2026)]] 将 Kim et al. 的 TGP 框架从 VoiceBox 迁移到 StyleTTS2,并提出 Encoder-Guided Poisoning (EGP) -- 用 style encoder 输出替代 teacher 生成的 mel 作为蒸馏目标,规避同容量 teacher-student 蒸馏效率低的问题。同时引入分布级评估框架 (AUC + FSSIM),揭示了 100 speaker 时 latent space crowding 导致的 scalability 限制 [Table 2]

## 在 TTS 中的应用

- **评估安全性**: Voice cloning 系统能否被 SV 系统检测出是合成语音
- **负责任发布**: 公开模型时附带检测工具或使用限制
- **对抗训练**: 在 TTS 训练中加入 discriminator 提升合成自然度的同时,本身也推进了检测技术
- **Benchmark 需求**: Survey 指出缺乏统一的 voice cloning benchmark,这同样制约了检测评估的标准化

## 关键论文

- Masood et al., "Deepfakes generation and detection: State-of-the-art" (Applied Intelligence, 2023): 视频+音频深伪综合 survey
- Khanjani et al., "Audio deepfakes: A survey" (Frontiers in Big Data, 2023): 音频深伪专题 survey
- Hutiri et al., "Not My Voice! A Taxonomy of Ethical and Safety Harms of Speech Generators" (ACM FAccT, 2024): 语音生成危害分类学
- Nakai et al., "Multi-Task Adversarial Training Algorithm for Multi-Speaker Neural TTS" (APSIPA ASC, 2022): 生成+判别联合训练
- Forbes, "A voice deepfake was used to scam a CEO out of $243,000" (2019): 标志性案例

## 相关概念

- [[SpeakerVerification]]: 检测的核心技术基础,同时是被攻击的对象
- [[VoiceCloningTaxonomy]]: 检测需要理解的生成方法分类
- [[SpeakerEmbedding]]: 检测系统和被检测系统共享的表示空间
- [[SpeechFactorization]]: 高质量 disentanglement 使伪造更难检测

## 演进

GMM-based spoofing detection (ASVspoof 2015) → DNN binary classification (2017) → End-to-end detection (2019) → Self-supervised feature-based detection (2021) → LLM-era: 生成质量逼近真实,检测难度陡增 (2023-) → Harm taxonomy 框架化 (Hutiri et al., 2024) → Watermark-free model attribution (Traceable TTS, 2025)
