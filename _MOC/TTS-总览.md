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
- 流式与实时 — streaming, low-latency
- TTS 应用与可访问性 — accessibility, document reader, mathematical TTS
  - [[论文笔记/MathReader|MathReader]] — 2025 (ICASSP), OCR+T5+VITS pipeline 实现数学文档正确朗读, WER 0.281 vs Edge 0.510

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
