---
type: concept
title: "Conditional Flow Matching"
aliases: [CFM, Flow Matching]
category: "generative-model"
tags: [generative-model, flow-based, diffusion-alternative, TTS]
key_papers: ["[[论文笔记/CosyVoice|CosyVoice]]", "[[论文笔记/CosyVoice2|CosyVoice 2]]", "[[论文笔记/CosyVoice3|CosyVoice 3]]", "[[论文笔记/IndexTTS2|IndexTTS2]]", "[[论文笔记/MaskGCT|MaskGCT]]", "[[论文笔记/Seed-TTS|Seed-TTS]]", "[[论文笔记/Survey-AudioDiffusionModels|Survey-Audio Diffusion Models]]", "[[论文笔记/GLM-TTS|GLM-TTS]]", "[[论文笔记/FireRedTTS|FireRedTTS]]", "[[论文笔记/Seed-VC|Seed-VC]]", "[[论文笔记/FELLE|FELLE]]", "[[论文笔记/FlowDec|FlowDec]]", "[[论文笔记/CLEAR|CLEAR]]", "[[论文笔记/FlexiVoice|FlexiVoice]]", "[[论文笔记/SiTok|SiTok]]", "[[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]", "[[论文笔记/FunAudioLLM|FunAudioLLM]]", "[[论文笔记/Multi-RewardGRPO|Multi-Reward GRPO]]", "[[论文笔记/VoxtralTTS|Voxtral TTS]]", "[[论文笔记/Qwen3-TTS|Qwen3-TTS]]", "[[论文笔记/VoxCPM|VoxCPM]]", "[[论文笔记/Cont-SPT|Cont-SPT]]", "[[论文笔记/E2TTS|E2 TTS]]", "[[论文笔记/TADA|TADA]]", "[[论文笔记/PilotTTS|PilotTTS]]", "[[论文笔记/DMOSpeech2|DMOSpeech 2]]", "[[论文笔记/VoiceFlow|VoiceFlow]]", "[[论文笔记/PeriodWave|PeriodWave]]", "[[论文笔记/EmoCtrl-TTS|EmoCtrl-TTS]]", "[[论文笔记/Bridge-TTS|Bridge-TTS]]", "[[论文笔记/EmoSphere++|EmoSphere++]]", "[[论文笔记/GOAT-TTS|GOAT-TTS]]", "[[论文笔记/FlexSpeech|FlexSpeech]]", "[[论文笔记/BreezyVoice|BreezyVoice]]", "[[论文笔记/LLaMA-Omni2|LLaMA-Omni 2]]", "[[论文笔记/F5R-TTS|F5R-TTS]]", "[[论文笔记/TechSinger|TechSinger]]", "[[论文笔记/OZSpeech|OZSpeech]]", "[[论文笔记/UmbraTTS|UmbraTTS]]", "[[论文笔记/RapFlow-TTS|RapFlow-TTS]]", "[[论文笔记/FMSD-TTS|FMSD-TTS]]", "[[论文笔记/TraceableTTS|Traceable TTS]]", "[[论文笔记/ZipVoice|ZipVoice]]", "[[论文笔记/TTS-CtrlNet|TTS-CtrlNet]]", "[[论文笔记/CapSpeech|CapSpeech]]", "[[论文笔记/UDDETTS|UDDETTS]]", "[[论文笔记/ShallowFlowMatching|Shallow Flow Matching]]", "[[论文笔记/Dragon-FM|Dragon-FM]]", "[[论文笔记/DeepDubbing|DeepDubbing]]", "[[论文笔记/DiFlow-TTS|DiFlow-TTS]]", "[[论文笔记/SpeakerIdentityUnlearning|Speaker Identity Unlearning]]", "[[论文笔记/DAIEN-TTS|DAIEN-TTS]]", "[[论文笔记/LibriQuote|LibriQuote]]", "[[论文笔记/ZipVoice-Dialog|ZipVoice-Dialog]]", "[[论文笔记/EmoSteer-TTS|EmoSteer-TTS]]", "[[论文笔记/Cross-LingualF5-TTS|Cross-Lingual F5-TTS]]", "[[论文笔记/TaDiCodec|TaDiCodec]]", "[[论文笔记/DiaMoE-TTS|DiaMoE-TTS]]", "[[论文笔记/TMD-TTS|TMD-TTS]]", "[[论文笔记/DialoSpeech|DialoSpeech]]", "[[论文笔记/Flamed-TTS|Flamed-TTS]]", "[[论文笔记/Ming-UniAudio|Ming-UniAudio]]", "[[论文笔记/WeSCon|WeSCon]]", "[[论文笔记/UniVoice|UniVoice]]", "[[论文笔记/Audiobook-CC|Audiobook-CC]]", "[[论文笔记/Semantic-VAE|Semantic-VAE]]", "[[论文笔记/DashengAudioGen|Dasheng AudioGen]]", "[[论文笔记/dots.tts|dots.tts]]", "[[论文笔记/VoxCPM2|VoxCPM2]]", "[[论文笔记/FlexiSLM|FlexiSLM]]"]
origin_paper: ""
related_concepts: ["[[FiniteScalarQuantization]]", "[[DiffusionModel]]", "[[ScoreMatching]]"]
status: confirmed
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---

## 定义

Conditional Flow Matching (CFM) 是一种基于连续正规化流 (Continuous Normalizing Flow, CNF) 的生成模型训练方法。它通过学习一个向量场将简单先验分布(如高斯)转换为目标数据分布,训练时仅需回归条件概率路径上的向量场,比传统 diffusion 模型推理步数更少、效率更高。

与 diffusion model 的关键区别: CFM 直接学习确定性 ODE 路径(flow),而非随机 SDE;可使用更少步数完成生成,且支持 rectified flow 等加速变体。

## 在 TTS 中的应用

在 coarse-to-fine TTS 系统(如 CosyVoice 系列、Voicebox、F5-TTS)中,CFM 用于将离散 speech token 序列转换为连续 Mel spectrogram。它作为 "fine stage" 渲染器,负责恢复 speech token 中被丢弃的声学细节(音色、韵律微观结构)。

CosyVoice 3 中 CFM 采用 DiT (Diffusion Transformer) 架构作为 backbone,参数从 100M 扩至 300M,去掉了 CosyVoice 2 的 text encoder 和 length regularization module。

## 关键论文

- Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
- CosyVoice (Du et al., 2024): 最早在 LLM-TTS 中采用 OT-CFM 替代 DDPM,使用 cosine scheduler + CFG (β=0.7) + masked mel conditioning 的工程组合
- CosyVoice 3 (2025): 使用 DiT-based CFM,300M 参数
- F5-TTS (2024): Flow matching for fluent and faithful speech
- [[论文笔记/VoiceFlow|VoiceFlow]] (Guo et al., ICASSP 2024): 首次将 rectified flow matching 引入 TTS 声学模型,与 GradTTS 保持相同 U-Net 架构仅替换生成算法,2 步 MOS 3.92 vs GradTTS 2.98 (LJSpeech); 通过 flow rectification 自蒸馏进一步拉直 ODE 轨迹,CMOS +0.78/+1.21
- Matcha-TTS (2024): CFM for fast TTS
- IndexTTS2 (Zhou et al., 2025): 在 S2M 模块中使用 flow matching 从 semantic tokens + speaker embedding 生成 mel spectrogram,并引入 GPT latent enhancement 融合上游 AR 隐状态以提升高情感语音的发音清晰度
- MaskGCT (Wang et al., 2024): 使用 flow matching 训练 total duration predictor (非 phone-level),12 层 Transformer + in-context learning + midpoint ODE solver (4 steps 推理)
- [[论文笔记/Seed-TTS|Seed-TTS]] (ByteDance, 2024): 使用 token diffusion model (diffusion transformer) 将 AR LM 生成的离散 token 转为连续声学表征;Seed-TTS_DiT 变体为完全 diffusion-based NAR TTS,直接从 Gaussian noise 预测 vocoder latent,无需 duration predictor;部署时使用 consistency distillation + modified flow matching 加速

## 相关概念

- [[DiffusionModel]]: CFM 的概念近亲,通过 SDE 而非 ODE; Zhang et al. (2023) audio diffusion survey 详细描述了 SDE → probability flow ODE → flow matching 的理论连接
- DiT (Diffusion Transformer): CosyVoice 3 CFM 的 backbone
- Vocoder: CFM 输出 Mel spectrogram 后仍需 vocoder 合成波形
- [[FiniteScalarQuantization]]: CFM 的输入(speech token 的条件)

- CosyVoice 2 (Du et al., 2024): 提出 chunk-aware causal flow matching,通过四种 attention mask (non-causal/full-causal/chunk-M/chunk-2M) 统一训练实现隐式自蒸馏,首次在 flow matching 框架下实现近无损流式 TTS; CFG β=0.7, NFE=10, cosine scheduler

## 演进

WaveNet (2016, autoregressive vocoder) → Diffusion-based TTS (Grad-TTS, 2021) → Flow Matching (Voicebox, 2023) → CFM + DiT (CosyVoice 3, 2025)

- [[论文笔记/CLEAR|CLEAR]] (Wu et al., 2025): 将 rectified flow 作为轻量 MLP head 直接挂在 AR language model 的每个 token 位置上,以 hidden state 为条件逐 token 生成连续 VAE latent。不同于 CosyVoice 系列将 CFM 作为独立的 second-stage renderer,CLEAR 实现了 **单阶段** AR + flow 联合训练。使用 logit-normal timestep sampling 和 auxiliary velocity direction loss 加速收敛。RTF 0.18, 仅需 78 步 AR decoding (对比 VALL-E 750 步)
- [[论文笔记/PeriodWave|PeriodWave]] (Lee et al., 2024): 首次将 OT-CFM 成功应用于 **波形级 (vocoder)** 生成,而非 mel spectrogram 级。提出 period-aware flow matching estimator,通过 multi-period reshaping (periods=[1,2,3,5,7]) 让 2D UNet 显式捕获不同周期特征。在 pitch/periodicity 指标上大幅超越 GAN vocoder (BigVGAN),且仅用单一 CFM loss 训练 3 天
- [[论文笔记/OZSpeech|OZSpeech]] (Huynh-Nguyen et al., 2025): 将 OT-CFM 的起始分布从 Gaussian noise 替换为 learned prior (Prior Codes Generator 生成),实现 **单步采样** (NFE=1)。在 FACodec 的 6 层离散码空间上操作,RTF 0.26 (比 F5-TTS 快近 3 倍),WER 仅 0.05 (3s prompt),但 UTMOS/SIM 有所牺牲
- [[论文笔记/RapFlow-TTS|RapFlow-TTS]] (Park et al., 2025): 首次将 **consistency flow matching** 引入 TTS,在 FM 的直 ODE 轨迹上施加 velocity consistency 约束,实现 2 步高保真合成。沿用 Matcha-TTS 架构 (18.2M),提出 multi-segment consistency FM + 对抗学习 + delta scheduling 等训练策略。LJSpeech 2 步 MOS 4.01 vs Matcha-TTS 10 步 3.83 / ComoSpeech 2 步 3.19
- [[论文笔记/ShallowFlowMatching|Shallow Flow Matching]] (Yang et al., NeurIPS 2025): 提出 **shallow flow matching (SFM)**,在 coarse-to-fine TTS 中将 FM 推理起点从纯噪声移至 coarse 表示构造的中间状态。通过正交投影自适应确定 CondOT 路径上的时间点,用单段分段流 (Theorem 2) 只训练后半段路径。在 Matcha-TTS / CosyVoice / StableTTS 上一致提升 CMOS,自适应步长 ODE solver 加速可达 ~50%
- [[论文笔记/DiFlow-TTS|DiFlow-TTS]] (Nguyen et al., 2025): 首个将 **discrete flow matching (DFM)** 应用于 TTS 的系统。不同于上述所有连续空间 FM,DiFlow-TTS 直接在离散 codec token 的概率分布上定义 flow,通过 factorized probability velocity field (prosody head + acoustic head) 在 FACodec 的多属性子空间上联合建模。122-164M 参数,UTMOS 3.98 (470h LibriTTS),RTF 0.03-0.07,但 SIM-O 仅 0.45
- [[论文笔记/Flamed-TTS|Flamed-TTS]] (Huynh-Nguyen et al., AAAI 2026): 延续 OZSpeech 的 learned prior 思路,提出 **attention-free flow matching**。假设 Code Generator 产出的离散码已包含语义信息,因此 Denoiser 中用 ConvNeXt 替代 self-attention,复杂度从 O(L^2*d) 降至 O(L*k*d)。同时引入 CFM 训练的 Probabilistic Duration & Silence Generator 增强时间多样性。143M 参数,WER 4% (best),RTF 0.028 (NFE=32),UTMOS 3.87 (500h LibriTTS)
- [[论文笔记/DSFlow|DSFlow]] (Lin et al., 2026): 提出模块化 **flow matching 蒸馏** 框架,通过 dual supervision (endpoint + JVP-free mean velocity alignment) + step-aware tokens (替代 adaLN-Zero,38M→1.5K 参数) 实现 **单步推理**。信息论动机: 蒸馏到 K 步后条件空间熵仅 log2(K) bits,模型容量应匹配。1-step MOS-N 4.32 vs teacher 10-step 4.43 (LibriSpeech, Emilia 95K hrs),RTF 0.012 (25x 加速),参数减少 24% (118M vs 154M)。跨架构验证: StepTTS/F5-TTS/CosyVoice2/E2-TTS [DSFlow Table 1]
- [[论文笔记/SplitMeanFlow|SplitMeanFlow]] (Guo et al., ByteDance, 2025): 从积分可加性推导纯代数的 **Interval Splitting Consistency** 恒等式 (t-r)u(zt,r,t) = (s-r)u(zs,r,s) + (t-s)u(zt,s,t),用于训练平均速度场。证明 MeanFlow 的微分恒等式是此代数恒等式在 s→t 极限下的特例。**JVP-free** (3 次标准前向 + 1 次反向,无需 Jacobian-vector product),训练更稳定且硬件兼容性更好。在 Seed-TTS 中验证: 1-step ICL WER 0.0286 = FM 10-step baseline, CMOS 0, 20x 加速 (无需 CFG) [SplitMeanFlow Table 2]
- [[论文笔记/LongCat-AudioDiT|LongCat-AudioDiT]] (Meituan, 2026): 首个公开的在 Seed 基准上 SOTA 的纯 CFM+DiT TTS 系统。独特之处在于直接在 **waveform latent space** (而非 mel spectrogram latent) 上做 flow matching,消除了 mel→waveform 的 compounding error。发现并修复了 VoiceBox-style masked conditioning 中 prompt noisy latent 在推理时 drift 的 training-inference mismatch 问题。3.5B 参数,SIM 0.818 (Seed-ZH),NFE=16
- [[论文笔记/AST-Edit|AST-Edit]] (Lv et al., 2026): 首次系统性地将 flow matching 的 **ODE 可逆性** 用于 training-free speech editing。通过 inverse Euler ODE solver 将 source mel 映射回 latent space,在 word-level 拼接 inverted latent (保留区) 与 Gaussian noise (编辑区),正向 ODE 合成编辑结果。提出 Adaptive Weak Fact Guidance (AWFG) 解决拼接边界 artifacts。基于 IndexTTS-2,SpkSim 0.986 / WDTW 0.2025 (SOTA) [AST-Edit Table 2]
- [[论文笔记/MAGIC-TTS|MAGIC-TTS]] (Mai et al., 2026): 在 F5-TTS Base CFM backbone 上增加 **token-level duration+pause 显式数值控制**。不修改 CFM 训练目标,仅在 text-side conditioning 路径注入 timing residual (zero-value correction + learnable gates)。展示了 CFM 架构对显式局部控制信号的良好兼容性: controlled C-Corr 0.918, spontaneous 模式保持与 F5 Base 相当的 C-Corr 0.588
- [[论文笔记/X-Voice|X-Voice]] (Xu et al., 2026): 基于 F5-TTS CFM backbone 扩展至 **30 语言 transcript-free 零样本跨语言克隆**。在 CFM 之上引入两项条件控制创新: (1) Dual-Level Language Injection — time-level concat LID embedding 约束 ODE 韵律轨迹 + textual-level FiLM 门控适配 IPA 表示到语言特异声学模式; (2) Decoupled Scheduled CFG + Asymmetric Warmup — 分离 acoustic/linguistic guidance 并在推理初期仅 warmup linguistic guidance 避免 integration shock。0.4B 参数,RTF 0.073,30 语言 WER 接近 GT [X-Voice Table 5, 9]
- [[论文笔记/KineticOptimalTTS|GibbsTTS]] (Yang et al., 2026): 首个将 **metric-induced discrete flow matching (MI-DFM)** 应用于 TTS。不同于 mask-source DFM (DiFlow-TTS),MI-DFM 使用 Gibbs 分布 softmax(-β_t·d(x,x_1)) 利用 codec embedding 几何结构。提出 training-free kinetic-optimal scheduler (恒速 Fisher-Rao 遍历,数值构建) + finite-step moment correction (保持 CTMC jump 方向只调概率)。399M DiT, full-codebook 联合预测,单阶段 text-to-codec。UTMOS 3.651 / WER 1.777% / SIM 0.743 (Seed test-en, 32 NFE),SIM 在 3/4 SOTA test set 最高 [KineticOptimalTTS Table 2,4]
- [[论文笔记/RobustSpeechFlow|RobustSpeechFlow]] (Yang et al., 2026): 将 **Contrastive Flow Matching** (Stoica et al., ICCV 2025) 从图像生成适配到 TTS,提出 TTS-specific hard negatives — 通过长度保持的 repeat/skip latent augmentation 构造 failure-mode 负样本。在 0.06B 的 SupertonicTTS 上,仅改训练目标(L = L_pos - λ_rand*L_rand - λ_aug*L_aug)即将 Seed-TTS-eval WER 从 1.44 降至 1.38 (benchmark 最低),证明 contrastive 正则化可在不增加模型/数据的条件下改善 FM-TTS 的对齐鲁棒性 [RobustSpeechFlow Table 1]
- [[论文笔记/UNISON|UNISON]] (Li et al., CUHK, 2026): 首个将 flow matching 扩展到 **统一音频生成+编辑** 的系统。提出 **layer-wise deep LLM fusion**: 将 frozen Qwen2.5-Omni-7B 各层 hidden states 通过 uniform sampling 映射到 MM-DiT 对应 block (浅层给词汇信息,深层给语义信息),替代传统的单层 embedding 广播。用 channel-mask (0/1/2) + VAE channel concatenation 统一 T2A/TTS/zero-shot cloning/audio editing/timed composition,无 phoneme encoder。621-732M 参数,FAD 1.558 / CLAP 0.503 (AudioCaps), WER 1.27% (Seed-TTS EN) [UNISON Table 1, 2]
- [[论文笔记/WavTTS|WavTTS]] (Chen et al., SJTU+ByteDance, 2026): 首个在 **原始波形空间** 上用 flow matching + DiT 实现接近 SOTA 的零样本 TTS。提出 signal-noise variance alignment (k=9 缩放波形使 σ≈1,消除 20dB SNR 失配) + x-prediction + multi-scale mel aux loss + PolyShift 推理时间表。673M 参数,100K hrs Emilia。Seed test-en WER **1.50%** (best) / UTMOS **3.92** (best) / SIM-o 0.65 (落后 latent 模型)。表示空间对比: 波形收敛快于 mel spectrogram 且 UTMOS 更高 (3.93 vs 3.68),优于 STFT 和 MDCT [WavTTS Fig 6, Table 1]
- [[论文笔记/FlowTTS-GRPO|FlowTTS-GRPO]] (Wang et al., Tongyi Lab, Interspeech 2026): 首次通过 **ODE→SDE 转换** 在 FM-based TTS 上实现 GRPO 训练。将确定性 ODE 转为等价反向 SDE (σ_t = a·√((1-t)/t)),在保持边际分布不变的前提下引入策略概率密度,使 GRPO importance ratio 可计算。窗口训练 (window training) 仅对早期步骤子集做 SDE,其余用 ODE,降低成本。在 CosyVoice 3 FM 和 F5-TTS 上验证,SS1 test-zh 0.804 首次超越闭源 Seed-TTS [FlowTTS-GRPO Table 2]
- [[论文笔记/Wan-Streamer|Wan-Streamer]] (Wan Team, Alibaba, 2026): 首次将 CFM 用于 **联合音视频 latent 生成**,在同一 clean context 下同时去噪 audio 和 video latent,使语音韵律与面部动作/唇形在 ODE solver 中天然同步。通过 rolling distillation + self-forcing 蒸馏 CFG 效果进 student 并减少 solver steps。160ms streaming unit, ~200ms model-side latency, 25 FPS。这是 CFM 从 TTS mel spectrogram/waveform 扩展到多模态交互生成的新应用场景 [Wan-Streamer §2.1, Eq 2-3]
