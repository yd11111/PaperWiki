# TTS 总览

> [!note] 人工备注
> (此区域自动刷新时保留)

## 核心子主题

- [[_MOC/语音编码与量化|语音编码与量化]] — codec, tokenization, RVQ, FSQ, quantization
- [[_MOC/零样本语音合成|零样本语音合成]] — zero-shot TTS, voice cloning, LLM-based generation
- [[_MOC/语音表征与自监督学习|语音表征与自监督学习]] — SSL, wav2vec, HuBERT, speech representation
- [[_MOC/语音大模型与对话|语音大模型与对话]] — speech LLM, full-duplex dialogue, omni-model
- [[_MOC/TTS训练与评估|TTS 训练与评估]] — post-training, RL, evaluation, reward model
- 韵律与情感 — prosody, emotion, style
  - [[论文笔记/Survey-Controllable TTS|Survey-Controllable TTS]] — 2024, 首篇全面综述可控 TTS: 架构/控制策略/特征表示三轴分类, Gemini-based 可控性评估
  - [[论文笔记/Llama-VITS|Llama-VITS]] — 2024, Llama2 语义嵌入增强 VITS 情感表达
  - [[论文笔记/StoryTTS|StoryTTS]] — 2024, 61h 中文评书表现力数据集 + LLM 驱动五维度文本表现力标注
  - [[论文笔记/UMETTS|UMETTS]] — 2024, 多模态(视觉/音频/文本)对比学习情感对齐 + 多 TTS 后端情感合成
  - [[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]] — 2024, flow-matching zero-shot TTS + 帧级 arousal-valence + NV embedding, 时变情感控制
  - [[论文笔记/Daisy-TTS|Daisy-TTS]] — 2024, Plutchik 结构模型 + prosody embedding PCA 分解, 一/二级情感+强度+极性模拟
  - [[论文笔记/DiffCSS|DiffCSS]] — 2025, Tsinghua/Tencent, diffusion 韵律预测器 + ParlerTTS backbone 实现对话语音韵律多样性, NDB 4/JSD 0.036
  - [[论文笔记/PROEMO|PROEMO]] — 2025, FS2 + HuBERT emotion/intensity 双编码器 + GPT-4 prompt control, 多说话人情感强度可控
  - [[论文笔记/AutoStyle-TTS|AutoStyle-TTS]] — 2025, Tsinghua/Tencent, RAG 自动风格 prompt 选择 + CosyVoice backbone, 三维 embedding (profile+emotion+user) 风格匹配
  - [[论文笔记/RADKA-CSS|RADKA-CSS]] — 2025, Inner Mongolia Univ/CUHK-SZ, RAG + 多粒度异构图聚合对话风格知识, N-DMOS 3.904/S-DMOS 3.879
  - [[论文笔记/EmoVoice|EmoVoice]] — 2025 (ACM MM), Qwen2.5 LLM + freestyle NL emotion description + phoneme parallel output, 情感 MOS 3.507 接近 GPT-4o-mini-tts 3.598
  - [[论文笔记/FaceSpeak|FaceSpeak]] — 2025 (AAAI), 任意风格肖像驱动 TTS, FaRL+IAM/EAM+GRL+vCLUB 跨模态 identity-emotion 解耦, VITS2 backbone
  - [[论文笔记/Revival with Voice|Revival with Voice (RV-TTS)]] — 2025 (Interspeech), face image + descriptive text 双模态可控 TTS, contrastive face-audio alignment + alternating training + style augmentation, MusicGen codec LM, MOS 4.14
  - [[论文笔记/Multi-Step Hierarchical ED|Multi-Step Hierarchical ED]] — 2025, CUHK-SZ/Alibaba, 多步 utterance→word→phoneme 层级情感分布预测 + FastSpeech 2 集成, WER 2.45% vs 单步 4.61%
  - [[论文笔记/MAE Style-Rich TTS|MAE Style-Rich TTS]] — 2025, Zhejiang Univ/Tencent, MAE style-rich token + 两阶段 AR LM 细粒度离散标签可控 TTS, CFG on AR logit, 数据解耦缓解高质量语料稀缺
  - [[论文笔记/Prompt-Unseen-Emotion|Prompt-Unseen-Emotion]] — 2025, A*STAR, LLM emotion-guided prompt (百分比模板) 实现零样本混合情感合成, 基于 CosyVoice, AB Pref 68-87% vs VITS-based baseline
  - [[论文笔记/Multilingual TTS Accents Emotions|Multilingual TTS Accents Emotions]] — 2025, DJSCE Mumbai, Parler-TTS 三阶段 fine-tuning 实现 Hindi/Indian English 口音+情感控制, WER 11.8% / 情感识别 85.3%
  - [[论文笔记/UDDETTS|UDDETTS]] — 2025, USTC/Alibaba, 首个 LLM-TTS 引入 ADV 空间 + 非线性分箱 + 半监督训练, 三维解耦情感控制 SRC 0.85-0.92, ES 0.833
  - [[论文笔记/DiEmo-TTS|DiEmo-TTS]] — 2025 (Interspeech), Korea Univ, DINO 自监督蒸馏 + cluster-driven sampling + formant perturbation 实现跨说话人情感解耦, FastSpeech 2 backbone, nMOS 4.23/eMOS 4.07
  - [[论文笔记/Spotlight-TTS|Spotlight-TTS]] — 2025, Korea Univ, voiced-aware RVQ (rotation trick) + unvoiced filler (biased self-attention) + style direction adjustment (orthogonality+prosody loss), FastSpeech 2 backbone, nMOS 4.26/sMOS 3.84
  - [[论文笔记/CapSpeech|CapSpeech]] — 2025, JHU/PKU/USC/MIT, 首个统一覆盖 9 种风格属性的大规模 CapTTS 基准 (10M+ pairs), 5 个下游任务 (CapTTS/CapTTS-SE/AccCapTTS/EmoCapTTS/AgentTTS), NAR (F5-TTS) Style-ACC 66.0% > AR (Parler-TTS) 56.0%
  - [[论文笔记/TTS-CtrlNet|TTS-CtrlNet]] — 2025, Yonsei Univ, 首次 ControlNet 范式迁移至 flow-matching TTS (F5-TTS backbone), 冻结原模型+可训练副本+zero-conv, ~400h 公开数据, Emo-SIM 0.751/Aro-Val SIM 0.742 超越 EmoCtrl-TTS
- 歌声合成 (SVS) — singing voice synthesis, technique control
  - [[论文笔记/TechSinger|TechSinger]] — 2025 (AAAI), Zhejiang Univ, 首个 flow matching 多语言多技巧 SVS, CFG 技巧控制 + 自动技巧标注 + NL prompt, MOS-C 4.10
- 流式与实时 — streaming, low-latency
  - [[论文笔记/LLMVoX|LLMVoX]] — 2025, MBZUAI, 30M LLM-agnostic autoregressive streaming TTS, multi-queue, 475ms延迟
  - [[论文笔记/SMLLE|SMLLE]] — 2025, Microsoft+SJTU, 首个逐帧流式零样本 TTS, Transducer→semantic tokens + AR mel, Delete <Bos> Mechanism
  - [[论文笔记/StreamMel|StreamMel]] — 2025, Nankai+Microsoft, 首个单阶段连续 mel 流式零样本 TTS, text-mel interleaving (1:4), FPL-A 0.01s
- TTS 应用与可访问性 — accessibility, document reader, mathematical TTS
  - [[论文笔记/MathReader|MathReader]] — 2025 (ICASSP), OCR+T5+VITS pipeline 实现数学文档正确朗读, WER 0.281 vs Edge 0.510
- 语音安全与隐私 — voice protection, deepfake defense, watermarking
  - [[论文笔记/TraceableSpeech|TraceableSpeech]] — 2024, Interspeech, VALL-E+HiFiCodec 联合水印训练, proactive speech traceability
  - [[论文笔记/SafeSpeech|SafeSpeech]] — 2025, USENIX Security, unlearnable perturbation + SPEC(KL引导)防护 fine-tuning+zero-shot voice cloning, 10模型迁移, WER 24%→99.6%
  - [[论文笔记/Traceable TTS|Traceable TTS]] — 2025, SJTU, watermark-free TTS traceability, 反转 GAN loss 实现 F5-TTS + discriminator 协同训练, 域外 AUC 0.9421 / EER 11.50%

## 按任务浏览

- [[Zero-shot Speech Synthesis]] — 零样本语音合成
- [[Cross-lingual Voice Cloning]] — 跨语言语音克隆
- [[Instructed Speech Generation]] — 指令式语音生成
- [[Neural Audio Compression]] — 神经音频压缩

## 按模型浏览

- [[CosyVoice 3]] — Alibaba, 2025, 9 语种 zero-shot TTS SOTA
- [[CosyVoice 2]] — Alibaba, 2024, 流式零样本 TTS
- [[MinMo]] — Alibaba, 2025, 多模态语音 LLM
- [[EnCodec]] — Meta, 2022, neural audio codec
- [[SoundStream]] — Google, 2021, 首个端到端 neural audio codec
- [[BigVGAN]] — NVIDIA, 2023, 通用神经声码器
- [[论文笔记/PeriodWave|PeriodWave]] — Ajou Univ, 2024, 多周期 flow matching 声码器

## 按概念浏览

- [[Speech Tokenizer]] — 语音离散化
- [[Conditional Flow Matching]] — 连续正规化流生成
- [[Residual Vector Quantization]] — 多层级向量量化
- [[Finite Scalar Quantization]] — 无码本量化
- [[Differentiable Reward Optimization]] — Token-level RL
- [[Masked Generative Modeling]] — 非自回归 mask-and-predict
- [[Speech Factorization]] — 语音属性解耦
- [[Gumbel-Softmax]] — 可微离散采样
- [[Gradient Reversal Layer]] — 对抗解耦
- [[Codebook Collapse]] — 码本坍缩问题
- [[Quantizer Dropout]] — 可变比特率训练
- [[Snake Activation]] — 周期性激活函数
- [[Multi-scale STFT Discriminator]] — 频域判别器
- [[Diffusion Model]] — 扩散模型 (DDPM, score-based, SDE/ODE)
- [[Score Matching]] — 分数匹配 (NCSN, score function)
- [[Classifier-Free Guidance]] — 无分类器引导
- [[Diffusion-based TTS]] — 扩散语音合成 (Grad-TTS, Diff-TTS, ProDiff)
- [[Diffusion-based Vocoder]] — 扩散声码器 (DiffWave, WaveGrad, BDDM)
- [[Next-Token Diffusion]] — 逐 token 扩散头 (LatentLM, CLEAR, VibeVoice)

## 生成方法演进 (相关论文)

- [[论文笔记/VoiceFlow|VoiceFlow]] — SJTU, 2023 (ICASSP 2024), 首次 rectified flow matching 用于 TTS 声学模型, 2 步 MOS 3.92 vs GradTTS 2.98
- [[论文笔记/Bridge-TTS|Bridge-TTS]] — Tsinghua/MSRA, 2023, Schrödinger bridge data-to-data TTS, 2-step MOS 4.04 超越 CoMoSpeech/Grad-TTS
- [[论文笔记/Very Attentive Tacotron|Very Attentive Tacotron]] — Google DeepMind, 2025, 用 IRPBs + latent alignment 解决 AR Transformer TTS 鲁棒性,实现无限长度泛化
- [[论文笔记/Low-Resource ForwardTacotron|Low-Resource ForwardTacotron]] — Fraunhofer IIS, 2025, 仅 4 个 HR 说话人 + 5min LR 数据,WGN 噪声增强 + binned sampling,speaker similarity 超越 HierSpeech++
- [[论文笔记/TTS-Transducer|TTS-Transducer]] — NVIDIA, 2025, RNNT transducer 单调对齐预测第一码本 + NAR Transformer 残余码本, 端到端 codec-agnostic TTS, CER 3.94% (challenging texts)
- [[论文笔记/Koel-TTS|Koel-TTS]] — NVIDIA, 2025, encoder-decoder AR TTS + DPO/RPO preference alignment + CFG for AR token prediction, CER 0.55% LibriTTS unseen (SOTA), MOS 4.054
- [[论文笔记/UmbraTTS|UmbraTTS]] — aiOla, 2025 (ICML Workshop), 首个 flow matching 环境感知 TTS, F5-TTS 框架 + SER 连续控制 + self-supervised 数据构建, WER 6.89% / 人类偏好 81.9%
- [[论文笔记/RapFlow-TTS|RapFlow-TTS]] — NAVER Cloud/Korea Univ, 2025, 首个 consistency flow matching TTS, 2 步 MOS 4.01 超越 Matcha-TTS 10 步 3.83 / ComoSpeech 2 步 3.19
- [[论文笔记/SpeechAccentLLM|SpeechAccentLLM]] — SEU/LIGHTSPEED, 2025, CTC-guided VQ (SpeechCodeVAE) + FAC&TTS 联合训练 + BERT-style SpeechRestorer, accentedness 1.86 vs baseline 2.48
- [[论文笔记/Shallow Flow Matching|Shallow Flow Matching]] — U. Tokyo, NeurIPS 2025, shallow flow matching 将 FM 推理起点从纯噪声移至 coarse 表示中间状态, 正交投影+分段流, Matcha-TTS/CosyVoice/StableTTS 一致提升 + 自适应 ODE solver ~50% 加速
