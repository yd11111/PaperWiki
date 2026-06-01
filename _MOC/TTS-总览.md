# TTS 总览

> [!note] 人工备注
> (此区域自动刷新时保留)

## 核心子主题

- [[_MOC/语音编码与量化|语音编码与量化]] — codec, tokenization, RVQ, FSQ, quantization
- [[_MOC/零样本语音合成|零样本语音合成]] — zero-shot TTS, voice cloning, LLM-based generation
- 语音生成模型 — 架构范式: AR, NAR, diffusion, flow
- 韵律与情感 — prosody, emotion, style
- 流式与实时 — streaming, low-latency
- 语音大模型 — speech LLM, omni-model
- 评估与数据 — metrics, benchmarks, datasets

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
