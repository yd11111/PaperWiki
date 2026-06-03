---
type: paper
tier: deep
title: "GLM-TTS Technical Report"
arxiv_id: "2512.14291"
source: "https://arxiv.org/abs/2512.14291"
authors: [Jiayan Cui, Zhihan Yang, Naihan Li, Jiankun Tian, Xingyu Ma, Yi Zhang, Guangyu Chen, Runxuan Yang, Zijian Huang, Yuqing Cheng, Yizhi Zhou, Guochen Yu, Xiaotao Gu, Jie Tang]
year: 2025
venue: "Technical Report (Zhipu AI)"
tags: [TTS, LLM, GRPO, reinforcement-learning, LoRA, vocoder, production-level, speech-tokenizer, phoneme, zero-shot, voice-cloning, emotion]
concepts: ["[[LLM-based TTS]]", "[[Speech Tokenizer]]", "[[Conditional Flow Matching]]", "[[Neural Vocoder]]", "[[Differentiable Reward Optimization]]", "[[Speaker Adaptation]]"]
models: []
tasks: []
datasets: ["[[SEED-TTS-Eval]]"]
kb_context_sources: 4
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 4 个已确认实体页: [[LLM-based TTS]], [[Speech Tokenizer]], [[Conditional Flow Matching]], [[Neural Vocoder]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[LLM-based TTS]]✓, [[Speech Tokenizer]]✓, [[Conditional Flow Matching]]✓, [[Neural Vocoder]]✓ | 过滤: [[Differentiable Reward Optimization]](pending-review), [[Speaker Adaptation]](pending-review) | 未命中但可能相关: 无

**[[LLM-based TTS]]**: GLM-TTS 遵循 CosyVoice 开创的 "Text-to-Token AR + Token-to-Wav Diffusion" 两阶段范式。1.5B 参数,100K 小时训练数据即达 SOTA,体现了工程优化的重要性。GRPO RL 对齐方法是 LLM-based TTS 中 RL post-training 的新探索,与 Seed-TTS 的 REINFORCE 和 CosyVoice 3 的 DiffRO 形成三条 RL 路线。

**[[Speech Tokenizer]]**: GLM-TTS 基于 Whisper-VQ 进行深度优化: token rate 从 12.5Hz 倍增至 25Hz、词表从 16K 扩至 32K、引入 Pitch Estimator 模块、改为非因果架构。这与 CosyVoice 系列使用 ASR encoder + FSQ 的路线不同,属于"从成熟 ASR 模型出发做 tokenizer 定制"的工程路线。

**[[Conditional Flow Matching]]**: GLM-TTS 的第二阶段使用 token-to-waveform diffusion model (FLOW) 从 speech tokens 合成 mel spectrogram,再经 vocoder 转为波形 [Fig 1]。与 CosyVoice 3 的 DiT-based CFM 属于同一设计范式。

**[[Neural Vocoder]]**: GLM-TTS 提出 Vocos2D vocoder,将 Vocos 的 1D 卷积改为 2D 卷积以改善频率子带建模,去除 MPD 仅保留 MRD,添加 discriminator augmentation (DA)。Vocos2D 在 NISQA/UTMOS/MOS 上全面优于原始 Vocos [Table 8]。

**[待确认]** [[Differentiable Reward Optimization]]: GLM-TTS 使用 GRPO (Group Relative Policy Optimization) 而非 CosyVoice 3 的 DiffRO。GRPO 在 audio level 操作 (生成完整语音后计算 CER/SIM/Emotion/Laughter 四维 reward),而 DiffRO 在 token level 操作。两种方法代表了 TTS RL 的不同粒度。

**[待确认]** [[Speaker Adaptation]]: GLM-TTS 的 LoRA 定制方案 (15% 参数、1 小时数据、80% 成本节省) 属于 parameter-efficient speaker adaptation 路线,与 AdaSpeech 的 CLN tuning 和 CosyVoice 3 的 LoRA 定制思路一致。

> [!summary] 速查
> - **一句话**: 1.5B 参数的生产级 LLM-based TTS, 通过 GRPO 多奖励 RL + 优化 Whisper-VQ tokenizer + Vocos2D vocoder + Phoneme-in 混合输入, 100K 小时训练达开源 SOTA
> - **路线**: Text/Phoneme → GLM-tokenizer (Whisper-VQ) + Prompt → GLM-TTS (AR) → Speech Tokens → FLOW (diffusion) → Mel → Vocos2D → Waveform
> - **指标**: test-zh CER 1.03% / SIM 76.1 (SFT), CER 0.89% / SIM 76.4 (RL); test-en WER 2.23% / SIM 67.2; Vocos2D MOS 4.16 vs Vocos 3.58 [Table 3, 8]
> - **可借鉴**: (1) GRPO 四维 reward (CER+SIM+Emotion+Laughter) 的 RL 框架; (2) Whisper-VQ tokenizer 的系统优化经验; (3) Vocos2D 的 2D 卷积设计; (4) Phoneme-in 混合输入处理多音字
> - **局限**: 英语数据不足 (不到中文一半); 情感 benchmark 为内部数据集; GRPO 的 Emotion reward 与 Dynamic Sampling 存在冲突

## 核心问题

GLM-TTS 的目标是构建生产级 TTS 系统,解决五大挑战 [§1]:
1. **低资源场景**: 高质量 voice cloning 需大规模数据,100K 小时是否足够? [§1]
2. **情感表现力**: 现有模型无法捕获文本蕴含的细微情感 [§1]
3. **发音精度**: 多音字、罕见词在中文等语言中尤为困难 [§1]
4. **RL 训练不稳定**: reward hacking 和梯度消失是 TTS RL 的核心难题 [§2.4]
5. **定制成本**: 全模型微调在生产环境中成本过高 [§2.5]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

GLM-TTS 采用两阶段架构 [Fig 1]:

**Stage 1 - Text-to-Token AR Model (GLM-TTS)**:
- 输入: Text tokens + Speech tokens (prompt from Whisper-VQ)
- 输出: Speech tokens 序列 (预测下一个 token)
- 特殊 token: Begin of Audio (B), End of Audio (E)
- 1.5B 参数, 100K 小时训练 [§1]

**Stage 2 - Token-to-Waveform**:
- FLOW: Speech tokens → Mel spectrogram (diffusion model)
- Vocos2D: Mel → Waveform (GAN vocoder)

### 关键设计选择

#### 1. GLM-TTS Speech Tokenizer 优化 [§2.3]

基于 GLM-4-Voice 的 Whisper-VQ,进行四项改进:

**WHY**: [论文原文] GLM-4-Voice tokenizer 在中文方言和高语速场景下发音精度不足 [§2.3]

(a) **Token Rate 倍增**: 12.5Hz → 25Hz, 词表 16K → 32K [§2.3]
- [论文原文] 低 token rate 在高语速时导致发音问题 (glitch),倍增后改善副语言特征 (笑声、呼吸声) 的自然度

(b) **Pitch Estimator (PE) 模块**: 新增音高估计模块优化韵律对齐 [§2.3]
- [论文原文] 解决 cloned TTS 与 reference 之间的韵律不匹配问题

(c) **非因果架构**: 去除原始因果约束,用标准卷积替代因果卷积 [§2.3]
- [论文原文] 消除 ASR 和 PE 模块的 sequential bottleneck

(d) **扩大训练数据**: 加入方言和高质量歌声数据 [§2.3]

**消融验证** [Table 1-2]:
- GLM-TTS-tokenizer 在四川方言 CER 上从 54.11 降至 24.40,台湾普通话从 49.09 降至 16.92 [Table 1]
- TTS 系统: SIM 75.2→76.1, CER 1.44→1.03 [Table 2]

#### 2. Text Tokenizer: Vocabulary Pruning [§2.2]

**WHY**: [论文原文] 长文本 token (超过两个汉字) 导致 speech-to-text 长度比方差大,一对多映射难学 [§2.2]

**HOW**: 修剪 tokenizer 词表,移除超过两个汉字的 token,将信息密度归一化,text-to-acoustic 对齐更稳定 [§2.2]

#### 3. GRPO-based Multi-Reward RL (SpeechLM RL-Alignment) [§2.4, Fig 3]

**WHY**: [论文原文] RL 在语音合成中未被充分探索,主要瓶颈在于 reward 设计复杂和梯度消失 [§2.4]

**HOW**: 引入 GRPO (Shao et al., 2024) 框架,三大创新:

**(a) 多维正则化 Reward** [§2.4, Fig 3]:
四个 reward 维度:
- **CER** (发音精度): 字符错误率
- **SIM** (音色保真): 说话人相似度
- **Emotion** (情感自然度): 情感分类得分
- **Laughter** (副语言): 笑声检测得分

层级处理: individual reward normalization → weighted fusion → group normalization [Fig 3]

**(b) Dynamic Sampling** [§2.4]:
- [论文原文] 当 batch 内 reward 变得均匀 (梯度消失风险) 时自动触发重采样 (最多三次)
- [论文原文] 限制重采样总次数,避免从低质量样本中负优化

**(c) Adaptive Gradient Clipping** [§2.4]:
- ε_high 和 ε_low 随训练步数动态调整
- [论文原文] 早期 tight clipping 防止 reward hacking; 后期放松以扩大探索
- [论文原文] 设置 ε_high > ε_low 鼓励低概率 token 的生成,提升人声自然度

**实验验证** [Table 4]:
- Pretrain-GRPO_c (Clip-Higher only): CER 1.93, SIM 80.4, EMO 0.660
- Pretrain-GRPO_d (+ Dynamic Sampling): CER 1.91, SIM 80.8, EMO 0.440
- [agent 解读] Dynamic Sampling 改善 CER/SIM 但降低 EMO,因 EMO 分布极端 (bimodal near 0/1)

**Laughter Reward** [§3.3, Table 6]:
- λ_laughter=5: CER 3.18, SIM 74.8, EMO 0.66
- λ_laughter=10: CER 3.06, SIM 74.8, EMO 0.72
- [论文原文] 笑声 reward 增加导致 CER/SIM 下降,因 ASR 将笑声识别为文本内容

**SFT-GRPO 消融** [Table 5]:
- 最佳配置: T=1.5, ε_h=0.5, ε_l=0.4 → CER 2.09, SIM 76.7, EMO 0.705
- 过度激进: T=3, ε_h=1, ε_l=0.4 → SIM 78.1 但 EMO 0.885 (reward hacking 倾向)

#### 4. LoRA for Premium Voice Customization [§2.5]

**WHY**: [论文原文] 全模型微调成本高、数据质量要求苛刻,不适合批量化生产 [§2.5]

**HOW**: LoRA 微调 ~15% backbone 参数,100 epochs [§2.5]
- [论文原文] 1 小时单说话人高质量音频即可完成定制
- [论文原文] 0.3%-5% 参数量的 LoRA 在风格和情感上提升有限;15% 是最优比例
- [论文原文] 控制微调参数比例 >15% 增强泛化性和跨场景稳定性

#### 5. Phoneme-in 混合输入 [§2.6]

**WHY**: [论文原文] 教育和标准化场景对多音字/罕见词发音精度要求极高,传统 G2P 或纯文本输入不可控 [§2.6]

**HOW**: 
- 构建多音字和罕见字的专用词表 [§2.6]
- 训练时使用 two-stage probabilistic replacement: 以 p=0.2 概率触发替换,随机选择比例 (0, 0.5) 的字符转为音素 [§2.6]
- 推理时: G2P 生成完整音素序列 → 遍历原文对多音字/罕见字替换为对应音素 → "hybrid phoneme+text" 混合输入 [§2.6]

**验证** [Table 7]:
- Text Only: PER 13.23%
- Hybrid (Text + Phoneme): PER **5.14%** (降低 61.1%)

#### 6. Vocos2D Vocoder [§2.7, Fig 4-5]

**WHY**: [论文原文] 原始 Vocos 的 1D 卷积处理所有频率子带,对特定频率子带建模不足 [§2.7]

**HOW**: 
- 2D 卷积替代 1D 卷积: PwConv1D → PwConv2D, DwConv1D → DwConv2D [Fig 4]
- 添加 per-frequency embeddings (因频率间不具备平移不变性) [§2.7]
- 修改 ConvNeXt(2D) block + DiT-style residual connections [§2.7]
- Generator Loss 改进 [Fig 5]:
  - 去除 Multi-Period Discriminator (MPD) — 在线性频谱输入上退化 [§2.7]
  - 添加 Discriminator Augmentation (DA): ±6dB loudness + random sample shifts + random phase rotations [§2.7]
- 混合训练: 加入高质量歌声数据扩展音域和适应能力 [§2.7]

### 训练策略

**数据处理 Pipeline** [§2.1, Fig 2]:
1. WAV 格式统一化 → VAD 切分 → 拼接为 ~10 分钟长段
2. 声源分离 (MelBand RoFormer) + 降噪
3. 说话人分割 (pyannote.audio) → 幅度归一化 → 拼接至 40 秒
4. WER 过滤 (<5%): 中文用 Paraformer + SenseVoice,英文用 Whisper + Reverb
5. 标点优化 (forced alignment + duration-based thresholding)
6. 特征提取: speaker embeddings + speech tokens
7. gRPC 分布式处理加速

**训练数据**: 100K 小时 (专有数据集),不到中文一半为英文 [§3.2]

## 实验

| 指标 | GLM-TTS (Ours) | Best Closed-Source | Best Open-Source | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| test-zh CER↓ | 1.03 / **0.89** (RL) | MiniMax-Speech **0.83** | IndexTTS2 1.03 | Seed-TTS-eval zh | [Table 3] |
| test-zh SIM↑ | 76.1 / 76.4 (RL) | Seed-TTS **79.6** | CosyVoice3 **78.0** | Seed-TTS-eval zh | [Table 3] |
| test-en WER↓ | 2.23 | MiniMax-Speech **1.65** | VoxCPM **1.85** | Seed-TTS-eval en | [Table 3] |
| test-en SIM↑ | 67.2 / 68.1 (RL) | Seed-TTS **76.2** | VoxCPM **72.9** | Seed-TTS-eval en | [Table 3] |
| Vocos2D NISQA↑ | **3.40** | - | Vocos 3.16 | Internal | [Table 8] |
| Vocos2D UTMOS↑ | **1.91** | - | Vocos 1.87 | Internal | [Table 8] |
| Vocos2D MOS↑ | **4.16** | - | Vocos 3.58 | Internal | [Table 8] |

**关键发现**:
1. GLM-TTS 在 1.5B 开源模型中达 top-tier 性能,与 closed-source 差距显著缩小 [§3.2]
2. GRPO RL 将 CER 从 1.03 进一步降至 0.89 [Table 3],验证了多奖励 RL 在 TTS 中的有效性
3. 英语性能受限于训练数据不足 (WER 2.23% vs 最优 1.65%) [§3.2]
4. Vocos2D 在所有指标上全面优于 Vocos (MOS 4.16 vs 3.58) [Table 8]
5. Phoneme-in 将 PER 从 13.23% 降至 5.14% [Table 7]

## 局限性

1. **英语数据不足**: 英语训练数据不到中文一半,导致英语指标落后于中文 [§3.2]
2. **EMO-Dynamic Sampling 冲突**: Dynamic Sampling 改善 CER/SIM 但损害 EMO,因 EMO 分布极端 [§3.3, Table 4]
3. **内部评估**: 情感 benchmark (CV3-eval-emotion) 和 Vocos2D 评估使用内部数据集,外部可复现性受限 [Table 4-8]
4. **Laughter Reward 副作用**: 增加笑声 reward 权重导致 CER/SIM 下降 [Table 6]
5. **缺少与 DiffRO/REINFORCE 的直接对比**: 与 CosyVoice 3/Seed-TTS 的 RL 方法仅在最终指标上间接对比,无 RL 方法本身的 controlled comparison

## 点评

GLM-TTS 是一个工程导向的系统论文,核心价值在于以有限资源 (100K 小时 vs CosyVoice 3 的 1M+ 小时) 达到可比 SOTA 性能。GRPO 四维 reward 框架是对 TTS RL post-training 的有意义探索,Clip-Higher + Dynamic Sampling + Adaptive Gradient Clipping 的组合解决了 RL 训练不稳定的实际问题。

Phoneme-in 混合输入是面向中文 TTS 部署的实用创新,精准解决了多音字/罕见词问题。Vocos2D 的 2D 卷积设计思路 (将频率维度作为卷积的第二维) 简单有效。

不足之处在于英语数据限制、情感评估的可复现性、以及与其他 RL 方法缺少 controlled comparison。论文结构清晰,消融实验充分 (tokenizer/RL/Phoneme-in/Vocos2D 分别消融),是 production-level TTS 的良好参考。

## 可复用的 idea

1. **GRPO 四维 Reward 框架**: CER + SIM + Emotion + Laughter 的多维 reward 设计,可扩展到其他 TTS/语音生成系统
2. **Adaptive Gradient Clipping**: ε_high > ε_low 鼓励低概率 token 生成,提升自然度
3. **Vocabulary Pruning**: 移除长文本 token 以稳定 text-to-speech 对齐
4. **Phoneme-in 混合输入**: 无需全音素化即可解决多音字问题,训练时概率替换增强鲁棒性
5. **Vocos2D**: 2D 卷积 + per-frequency embedding + DA 替代 MPD 的 vocoder 设计
6. **LoRA 15% 参数微调**: 1 小时数据实现 production-level voice customization

---

检索命中: [[LLM-based TTS]], [[Speech Tokenizer]], [[Conditional Flow Matching]], [[Neural Vocoder]] | 过滤: [[Differentiable Reward Optimization]](pending-review), [[Speaker Adaptation]](pending-review) | 未命中但可能相关: 无
