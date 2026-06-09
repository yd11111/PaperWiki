# TTS 总览

<!-- scope: 本页是 TTS 知识库的顶层导航，汇总所有子主题 MOC 入口、跨领域论文及按任务/模型/概念的浏览视图。
     不包含: 纯 ASR/NLU、纯音乐生成(无语音)、纯语音增强/降噪。 -->

> [!note] 人工备注
> (此区域自动刷新时保留)

## 核心子主题

- [[_MOC/语音编码与量化|语音编码与量化]] — codec, tokenization, RVQ, FSQ, quantization
- [[_MOC/零样本语音合成|零样本语音合成]] — zero-shot TTS, voice cloning, LLM-based generation
- [[_MOC/语音表征与自监督学习|语音表征与自监督学习]] — SSL, wav2vec, HuBERT, speech representation
- [[_MOC/语音大模型与对话|语音大模型与对话]] — speech LLM, full-duplex dialogue, omni-model
- [[_MOC/TTS训练与评估|TTS 训练与评估]] — post-training, RL, evaluation, reward model
- [[_MOC/韵律与情感|韵律与情感]] — prosody, emotion, style

### 歌声合成 (SVS)
- [[论文笔记/TechSinger|TechSinger]] — 2025, Zhejiang Univ, flow matching 多技巧 SVS (作者 claim)

### 流式与实时
- [[论文笔记/LLMVoX|LLMVoX]] — 2025, MBZUAI, 30M streaming TTS, 475ms 延迟
- [[论文笔记/SMLLE|SMLLE]] — 2025, Microsoft+SJTU, 逐帧流式零样本 TTS (作者 claim)
- [[论文笔记/StreamMel|StreamMel]] — 2025, Nankai+Microsoft, 单阶段连续 mel 流式 TTS (作者 claim)

### TTS 应用与可访问性
- [[论文笔记/MathReader|MathReader]] — 2025, OCR+T5+VITS 数学文档朗读, WER 0.281
- [[论文笔记/MiSTR|MiSTR]] — 2025, iEEG-to-speech BCI 框架
- [[论文笔记/DiVISe|DiVISe]] — 2025, 端到端视频到语音, AV-HuBERT + Conformer

### 语音安全与隐私
- [[论文笔记/TraceableSpeech|TraceableSpeech]] — 2024, VALL-E+HiFiCodec 联合水印训练
- [[论文笔记/SafeSpeech|SafeSpeech]] — 2025, unlearnable perturbation 防护 voice cloning
- [[论文笔记/TraceableTTS|Traceable TTS]] — 2025, SJTU, watermark-free TTS traceability
- [[论文笔记/SpeakerIdentityUnlearning|Speaker Identity Unlearning]] — 2025, ZS-TTS speaker unlearning (作者 claim)
- [[论文笔记/E2E-VGuard|E2E-VGuard]] — 2025, encoder ensemble 防护端到端语音克隆

### 语音工具
- [[论文笔记/BFA|BFA]] — 2025, CTC 强制对齐, 比 MFA 快 240 倍, 多语言

## 按任务浏览

- [[Zero-shotSpeechSynthesis]] — 零样本语音合成
- [[Cross-lingualVoiceCloning]] — 跨语言语音克隆
- [[InstructedSpeechGeneration]] — 指令式语音生成
- [[NeuralAudioCompression]] — 神经音频压缩

## 按模型浏览

- [[CosyVoice3]] — Alibaba, 2025, 9 语种 zero-shot TTS (作者 claim SOTA)
- [[CosyVoice2]] — Alibaba, 2024, 流式零样本 TTS
- [[MinMo]] — Alibaba, 2025, 多模态语音 LLM
- [[EnCodec]] — Meta, 2022, neural audio codec
- [[SoundStream]] — Google, 2021, 端到端 neural audio codec (作者 claim 首个)
- [[BigVGAN]] — NVIDIA, 2023, 通用神经声码器
- [[论文笔记/PeriodWave|PeriodWave]] — Ajou Univ, 2024, 多周期 flow matching 声码器

## 按概念浏览

- [[SpeechTokenizer]] — 语音离散化
- [[ConditionalFlowMatching]] — 连续正规化流生成
- [[ResidualVectorQuantization]] — 多层级向量量化
- [[FiniteScalarQuantization]] — 无码本量化
- [[DifferentiableRewardOptimization]] — Token-level RL
- [[MaskedGenerativeModeling]] — 非自回归 mask-and-predict
- [[SpeechFactorization]] — 语音属性解耦
- [[Gumbel-Softmax]] — 可微离散采样
- [[GradientReversalLayer]] — 对抗解耦
- [[CodebookCollapse]] — 码本坍缩问题
- [[QuantizerDropout]] — 可变比特率训练
- [[SnakeActivation]] — 周期性激活函数
- [[Multi-scaleSTFTDiscriminator]] — 频域判别器
- [[DiffusionModel]] — 扩散模型 (DDPM, score-based, SDE/ODE)
- [[ScoreMatching]] — 分数匹配 (NCSN, score function)
- [[Classifier-FreeGuidance]] — 无分类器引导
- [[Diffusion-basedTTS]] — 扩散语音合成 (Grad-TTS, Diff-TTS, ProDiff)
- [[Diffusion-basedVocoder]] — 扩散声码器 (DiffWave, WaveGrad, BDDM)
- [[Next-TokenDiffusion]] — 逐 token 扩散头 (LatentLM, CLEAR, VibeVoice)

## 生成方法演进

Tacotron (2017) → Tacotron 2 (2018) → FastSpeech/NAR (2019)
GAN 声码器: MelGAN → HiFi-GAN (2020) → BigVGAN (2023)
扩散路线: Grad-TTS (2021) ⇢ Bridge-TTS (2023) ⇢ VoiceFlow (2023, flow matching)
Codec LM: AudioLM (2022) → VALL-E (2023) ⇢ CosyVoice (2024) → CosyVoice 3 (2025)
FM 加速: Matcha-TTS ⇢ RapFlow-TTS (consistency) ∥ ShallowFlowMatching (中间起点)
统一框架: Dragon-FM (chunk-AR + FM) ∥ MELA-TTS (AR + DiT, tokenizer-free)

### 代表论文
- [[论文笔记/VoiceFlow|VoiceFlow]] — 2023, SJTU, rectified flow matching TTS (作者 claim)
- [[论文笔记/Bridge-TTS|Bridge-TTS]] — 2023, Tsinghua/MSRA, Schrödinger bridge TTS
- [[论文笔记/VeryAttentiveTacotron|Very Attentive Tacotron]] — 2025, Google DeepMind, AR Transformer 鲁棒性
- [[论文笔记/Low-ResourceForwardTacotron|Low-Resource ForwardTacotron]] — 2025, Fraunhofer, 低资源 TTS
- [[论文笔记/TTS-Transducer|TTS-Transducer]] — 2025, NVIDIA, RNNT 单调对齐 codec TTS
- [[论文笔记/Koel-TTS|Koel-TTS]] — 2025, NVIDIA, AR TTS + DPO 偏好对齐
- [[论文笔记/UmbraTTS|UmbraTTS]] — 2025, aiOla, flow matching 环境感知 TTS (作者 claim)
- [[论文笔记/RapFlow-TTS|RapFlow-TTS]] — 2025, NAVER, consistency flow matching TTS (作者 claim)
- [[论文笔记/SpeechAccentLLM|SpeechAccentLLM]] — 2025, SEU, CTC-guided VQ 口音控制 TTS
- [[论文笔记/ShallowFlowMatching|Shallow Flow Matching]] — 2025, U. Tokyo, 中间起点加速 FM
- [[论文笔记/Dragon-FM|Dragon-FM]] — 2025, Microsoft, chunk-AR + FM 统一框架
- [[论文笔记/FNH-TTS|FNH-TTS]] — 2026, Megatronix, VITS + MoE Duration Predictor
- [[论文笔记/MELA-TTS|MELA-TTS]] — 2025, Alibaba, tokenizer-free AR + DiT
- [[论文笔记/TMD-TTS|TMD-TTS]] — 2026, UESTC, 藏语三方言统一 TTS
- [[论文笔记/MAVE|MAVE]] — 2025, MTS AI, Mamba codec LM speech editing (作者 claim)
- [[论文笔记/Semantic-VAE|Semantic-VAE]] — 2025, SJTU, VAE latent 语义对齐正则化
