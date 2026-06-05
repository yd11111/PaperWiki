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
- [[_MOC/韵律与情感|韵律与情感]] — prosody, emotion, style (33 篇)
  - [[论文笔记/EmoSteer-TTS|EmoSteer-TTS]] — 2025, HKUST-GZ/Tencent, training-free 情感可控: activation steering 迁移至 flow-matching DiT (作者 claim)
  - [[论文笔记/CapSpeech|CapSpeech]] — 2025, JHU/PKU/USC/MIT, 统一 9 种风格属性的 CapTTS 基准 (10M+ pairs) (作者 claim)
  - [[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]] — 2024, flow-matching zero-shot TTS + 帧级 arousal-valence, 时变情感控制
- 歌声合成 (SVS) — singing voice synthesis, technique control
  - [[论文笔记/TechSinger|TechSinger]] — 2025 (AAAI), Zhejiang Univ, flow matching 多语言多技巧 SVS (作者 claim 首个), CFG 技巧控制 + 自动技巧标注 + NL prompt, MOS-C 4.10
- 流式与实时 — streaming, low-latency
  - [[论文笔记/LLMVoX|LLMVoX]] — 2025, MBZUAI, 30M LLM-agnostic autoregressive streaming TTS, multi-queue, 475ms延迟
  - [[论文笔记/SMLLE|SMLLE]] — 2025, Microsoft+SJTU, 逐帧流式零样本 TTS (作者 claim 首个), Transducer→semantic tokens + AR mel, Delete <Bos> Mechanism
  - [[论文笔记/StreamMel|StreamMel]] — 2025, Nankai+Microsoft, 单阶段连续 mel 流式零样本 TTS (作者 claim 首个), text-mel interleaving (1:4), FPL-A 0.01s
- TTS 应用与可访问性 — accessibility, document reader, mathematical TTS, BCI speech prosthesis
  - [[论文笔记/MathReader|MathReader]] — 2025 (ICASSP), OCR+T5+VITS pipeline 实现数学文档正确朗读, WER 0.281 vs Edge 0.510
  - [[论文笔记/MiSTR|MiSTR]] — 2025, BME/UKIM, iEEG-to-speech BCI 框架: DWT 小波特征 + Transformer 韵律感知 mel 预测 + IHPR 谐波相位重建, PC 0.91 / HNR 12.7 dB
  - [[论文笔记/DiVISe|DiVISe]] — 2025, 端到端 V2S, AV-HuBERT + Conformer 直接从无声视频预测 mel, 无需 speaker embedding 保留说话人特性
- 语音安全与隐私 — voice protection, deepfake defense, watermarking
  - [[论文笔记/TraceableSpeech|TraceableSpeech]] — 2024, Interspeech, VALL-E+HiFiCodec 联合水印训练, proactive speech traceability
  - [[论文笔记/SafeSpeech|SafeSpeech]] — 2025, USENIX Security, unlearnable perturbation + SPEC(KL引导)防护 fine-tuning+zero-shot voice cloning, 10模型迁移, WER 24%→99.6%
  - [[论文笔记/TraceableTTS|Traceable TTS]] — 2025, SJTU, watermark-free TTS traceability, 反转 GAN loss 实现 F5-TTS + discriminator 协同训练, 域外 AUC 0.9421 / EER 11.50%
  - [[论文笔记/SpeakerIdentityUnlearning|Speaker Identity Unlearning]] — 2025 (ICML), ZS-TTS speaker unlearning (作者 claim 首次), Teacher-Guided Unlearning 使 VoiceBox 对 forget speaker 生成随机音色
  - [[论文笔记/E2E-VGuard|E2E-VGuard]] — 2025, encoder ensemble timbre 扰动 + ASR 对抗攻击 + 心理声学感知优化, 防护 LLM-based TTS 和端到端语音克隆
- 语音工具 — forced alignment, preprocessing
  - [[论文笔记/BFA|BFA]] — 2025, CTC + universal phoneme encoder 强制对齐, 比 MFA 快 240 倍, 多语言, 显式建模音素间隙

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

## 生成方法演进 (相关论文)

- [[论文笔记/VoiceFlow|VoiceFlow]] — SJTU, 2023 (ICASSP 2024), rectified flow matching 用于 TTS 声学模型 (作者 claim 首次), 2 步 MOS 3.92 vs GradTTS 2.98
- [[论文笔记/Bridge-TTS|Bridge-TTS]] — Tsinghua/MSRA, 2023, Schrödinger bridge data-to-data TTS, 2-step MOS 4.04 超越 CoMoSpeech/Grad-TTS
- [[论文笔记/VeryAttentiveTacotron|Very Attentive Tacotron]] — Google DeepMind, 2025, 用 IRPBs + latent alignment 解决 AR Transformer TTS 鲁棒性,实现无限长度泛化
- [[论文笔记/Low-ResourceForwardTacotron|Low-Resource ForwardTacotron]] — Fraunhofer IIS, 2025, 仅 4 个 HR 说话人 + 5min LR 数据,WGN 噪声增强 + binned sampling,speaker similarity 超越 HierSpeech++
- [[论文笔记/TTS-Transducer|TTS-Transducer]] — NVIDIA, 2025, RNNT transducer 单调对齐预测第一码本 + NAR Transformer 残余码本, 端到端 codec-agnostic TTS, CER 3.94% (challenging texts)
- [[论文笔记/Koel-TTS|Koel-TTS]] — NVIDIA, 2025, encoder-decoder AR TTS + DPO/RPO preference alignment + CFG for AR token prediction, CER 0.55% (benchmark: LibriTTS unseen), MOS 4.054
- [[论文笔记/UmbraTTS|UmbraTTS]] — aiOla, 2025 (ICML Workshop), flow matching 环境感知 TTS (作者 claim 首个), F5-TTS 框架 + SER 连续控制 + self-supervised 数据构建, WER 6.89% / 人类偏好 81.9%
- [[论文笔记/RapFlow-TTS|RapFlow-TTS]] — NAVER Cloud/Korea Univ, 2025, consistency flow matching TTS (作者 claim 首个), 2 步 MOS 4.01 超越 Matcha-TTS 10 步 3.83 / ComoSpeech 2 步 3.19
- [[论文笔记/SpeechAccentLLM|SpeechAccentLLM]] — SEU/LIGHTSPEED, 2025, CTC-guided VQ (SpeechCodeVAE) + FAC&TTS 联合训练 + BERT-style SpeechRestorer, accentedness 1.86 vs baseline 2.48
- [[论文笔记/ShallowFlowMatching|Shallow Flow Matching]] — U. Tokyo, NeurIPS 2025, shallow flow matching 将 FM 推理起点从纯噪声移至 coarse 表示中间状态, 正交投影+分段流, Matcha-TTS/CosyVoice/StableTTS 一致提升 + 自适应 ODE solver ~50% 加速
- [[论文笔记/Dragon-FM|Dragon-FM]] — Microsoft, 2025, chunk-AR + chunk 内 flow matching 统一框架 ("next-token denoising"), 12.5Hz FSQ codec (48kHz), FSQ embedding 直接作为 FM 连续目标, TNFE 2-4
- [[论文笔记/FNH-TTS|FNH-TTS]] — Megatronix/Newcastle, 2026, VITS + MoE Duration Predictor (Switch-Transformer 8 experts) + VOCOS vocoder + CoMBD/SBD, 揭示 duration-vocoder 耦合效应, MOS 4.48/4.63 (LJ/VCTK), 47.73M 参数
- [[论文笔记/MELA-TTS|MELA-TTS]] — Alibaba, 2025, Tokenizer-free joint AR transformer + DiT diffusion, ASR representation alignment 加速训练 3.3x + 提升内容一致性, CER 0.9% test-zh / WER 2.4% test-en (170K h)
- [[论文笔记/TMD-TTS|TMD-TTS]] — UESTC/Tibet Univ, 2026, 藏语三方言统一 TTS, Matcha-TTS + DSDR-Net (public/private FFN 条件路由) + dialect fusion, 构建 102h TMDD 数据集, nMOS 3.86 / DECS 88.09%
- [[论文笔记/MAVE|MAVE]] — MTS AI, 2025, Mamba SSM + cross-attention codec LM 用于 speech editing 和 zero-shot TTS (作者 claim 首个), 830M 参数, ~6x 内存优势超越 VoiceCraft, CM3 causal masking 实现 AR 框架双向上下文
- [[论文笔记/Semantic-VAE|Semantic-VAE]] — SJTU/Geely, 2025 (ICASSP 2026), VAE latent 语义对齐正则化 (WavLM cosine loss) 解决重建-生成困境, F5-TTS WER 2.23→1.95% / SIM 0.60→0.64
