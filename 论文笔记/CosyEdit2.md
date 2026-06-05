---
type: paper
tier: deep
title: "CosyEdit2: Speech-Editing-Oriented RL Unlocks Better Zero-Shot TTS"
arxiv_id: "2605.25930"
source: "Sources/CosyEdit2.pdf"
authors: [Junyang Chen, Yuhang Jia, Hui Wang, Jiaming Zhou, Yongchang Gan, Yong Qin]
year: 2026
venue: "arXiv"
tags: [speech-editing, GRPO, reinforcement-learning, post-training, zero-shot-TTS, end-to-end, flow-matching, reward-design]
concepts: ["[[LLM-basedTTS]]", "[[ConditionalFlowMatching]]", "[[DifferentiableRewardOptimization]]"]
models: ["[[模型库/CosyVoice2]]", "[[论文笔记/CosyEdit|CosyEdit]]", "[[论文笔记/Ming-UniAudio|Ming-UniAudio]]"]
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: ["[[GigaSpeech]]"]
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页 + 1 个待确认实体页: [[模型库/CosyVoice2]], [[LLM-basedTTS]], [[ConditionalFlowMatching]], [[DifferentiableRewardOptimization]], [[SpeechLanguageModel]], [[Zero-shotSpeechSynthesis]])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **CosyVoice2 作为 backbone**: CosyEdit2 直接基于 CosyVoice2 进行 post-training。CosyVoice2 是阿里巴巴通义实验室的可扩展流式零样本 TTS 系统,使用 Qwen2.5-0.5B LLM + FSQ-SenseVoice semantic tokenizer + chunk-aware causal flow matching + HiFT-GAN vocoder。CosyEdit2 替换了 HiFT-GAN 为 BigVGAN,但保留 LLM 和 FSQ tokenizer 结构。
>
> **CosyEdit 谱系**: CosyEdit2 是 CosyEdit 的直接继承者。CosyEdit 基于原版 CosyVoice,提出了 GOT-CFM (Guided OT-CFM) 和训练-推理非对称设计,仅用 250h 数据实现端到端语音编辑。CosyEdit2 升级 backbone 到 CosyVoice2,引入 GRPO 替代纯 SFT,在编辑和零样本 TTS 上均有提升。
>
> **RL for TTS 演进定位**: 概念库中 [[DifferentiableRewardOptimization]] 页 [待确认] 记录了 TTS RL 的完整演进线: DiffRO (token-level 可微) → GRPO (audio-level, 无需 Gumbel-Softmax) → Multi-Reward GRPO → GRPO-TTS 等。CosyEdit2 首次将 GRPO 应用于端到端 speech editing,且同时提升了 zero-shot TTS,与 ECPA (Ren et al., 2026) 的 self-consistency GRPO 形成对比: ECPA 依赖 TTS prior 作为隐式 critic,CosyEdit2 使用 teacher-free outcome-level reward。
>
> **CFM 在 speech editing 中的角色**: CosyEdit2 沿用 CosyEdit 的 GOT-CFM 设计。Conditional Flow Matching 在此不仅是生成器,更是编辑时保持非编辑区域声学一致性的关键模块。通过完整原始语音的 mel + speech tokens 作为全局条件,使 flow matching 同时"看到"编辑区和非编辑区。
>
> **Zero-shot TTS ↔ Speech Editing 的统一观**: 论文提出 zero-shot TTS 可视为 speech editing 的特例(全段替换/全尾插入)。这个观点与 Zero-shotSpeechSynthesis 任务页记录的核心挑战(内容一致性 + 说话人相似度 + 泛化)高度一致,但增加了"声学保持"这一编辑特有的更强约束。
>
> 检索命中: [[模型库/CosyVoice2]]✓, [[LLM-basedTTS]]✓, [[ConditionalFlowMatching]]✓, [[SpeechLanguageModel]]✓, [[Zero-shotSpeechSynthesis]]✓ | 过滤: [[DifferentiableRewardOptimization]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 在 CosyVoice2 上构建两阶段 post-training 框架(SFT → editing-oriented GRPO),用无需目标录音的 target-speech-free 数据 + 三维 editing reward 突破 SFT 的 preservation-accuracy 天花板,同时提升 speech editing 和 zero-shot TTS
> - **路线**: (original text + target text + original speech) → BPE tokenizers + FSQ speech tokenizer → AR LLM 生成 target speech tokens → GOT-CFM flow matching (全局原始语音条件) → BigVGAN vocoder → edited speech
> - **指标**: Speech editing: WER 1.43-1.93% / SS 0.89-0.93 / MAE_DNSMOS 0.107-0.137 (Ming-Freeform-Audio-Edit, best acoustic consistency) [Table 1]; Zero-shot TTS: zh CER 3.52% / en WER 4.87% / hard-zh CER 8.06 / hard-en WER 5.93 (CV3-Eval, best across board) [Tables 3-5]; SEED-TTS: zh CER 1.16% / en WER 1.95% (Table 9)
> - **可借鉴**: (1) TTS-to-Edit prompt construction: 任何 TTS 语料对即可构造 editing 训练数据,消除人工目标录音需求; (2) Coarse-to-fine priority-aware reward composition: WER 作为粗粒度内容门控,MCD 细粒度声学保持,SIM 全局排序; (3) GRPO 仅更新 LLM 而冻结 Flow+BigVGAN 的模块化训练策略
> - **局限**: 语言覆盖受限于 CosyVoice2 (中英日韩); GRPO 仅用英文数据训练但声称跨语言迁移; 未开源; MCD reward 依赖 WhisperX forced alignment 的准确性; 情感/韵律等 paralinguistic editing 未探索

## 核心问题

1. **SFT-based speech editing 的本质瓶颈是什么?** 数据侧: 人工构造的目标录音存在边界模糊和声学不一致; 优化侧: token-level reconstruction loss 不区分编辑区和非编辑区,也不区分语义正确性和声学保持,导致 preservation-accuracy trade-off 成为性能天花板 [§1]
2. **为什么 speech editing 能反哺 zero-shot TTS?** 论文论证 zero-shot TTS 是 speech editing 的特例(全段替换或全尾插入),两者共享同一核心能力: prompt-conditioned in-context learning。Editing-oriented GRPO 强化这一共享能力,语义上鼓励更强的 speech-text alignment 减少 hallucination,声学上要求更精确地利用 prompt 中的 speaker/acoustic cues [§4.5]
3. **如何在无目标录音的情况下训练 speech editing?** 将任何 TTS 语料的 (speech, text) 对通过 rule-based NLP perturbation 转化为 (original text, target text, original speech) 编辑 prompt,GRPO 在最终波形上评估 reward 而非在 token 层面模仿目标录音 [§3.3]

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

CosyEdit2 复用 CosyVoice2 的四组件架构,针对 speech editing 重构输入接口 [§3.1]:

1. **Text Tokenizers**: 两个相同的 BPE tokenizer 分别编码 original text 和 target text,拼接后送入 LLM。[论文原文] 绕过显式 phoneme frontend 以实现端到端上下文发音学习 [§B.1]
2. **Speech Tokenizer**: 沿用 CosyVoice2 的 FSQ-SenseVoice supervised semantic tokenizer,低帧率提取离散 speech tokens [§B.2]
3. **AR Text-Speech LLM**: 基于 Qwen2.5-0.5B,输入 `[S, X_ori, X_tar, μ_ori, T]`,自回归生成 target speech tokens 直到 `E` [§B.3]
4. **GOT-CFM Flow Model**: 从 CosyEdit 继承的 Guided OT-CFM,将原始语音的完整 speech tokens + mel spectrogram 作为全局条件,引导 target speech mel 生成 [§B.4]
5. **BigVGAN Vocoder**: 替换 CosyVoice2 的 HiFT-GAN,训练于 clean + in-the-wild 混合数据,更好地保持多样声学环境 [§B.5]

### 关键设计选择

**1. 两阶段 post-training: SFT → GRPO [§3.2-3.3]**

[论文原文] SFT 的 token-level reconstruction loss 将编辑区和非编辑区同等对待,且依赖人工构造的不完美目标录音,导致固有的 preservation-accuracy trade-off。GRPO 在波形层面的 outcome-level 奖励替代了 token 层面的模仿信号,绕过了不完美监督的瓶颈 [§1, §3.3]。

- Stage 1 (SFT): LLM、Flow、BigVGAN 分别独立训练。LLM + Flow 在 GigaEdit-S (250h) 上训练 [§3.2],BigVGAN 在 625h clean+in-the-wild 混合数据上训练 [§3.2]
- Stage 2 (GRPO): 仅更新 LLM,Flow 和 BigVGAN 冻结。LLM 生成 speech tokens → 冻结的 Flow + BigVGAN 解码为波形 → 波形级 reward 计算 → 梯度仅回传 LLM [§3.3, Fig 3]

[agent 解读] 冻结 Flow+BigVGAN 的设计是关键的工程决策: 它将 RL 的搜索空间限制在语义 token 层面(LLM 输出),避免在高维声学空间中的不稳定优化。这与 CosyVoice 3 的 DiffRO 路线(token 层面 reward + Gumbel-Softmax)形成互补:CosyEdit2 在 audio 层面计算 reward 但在 token 层面更新策略。

**2. Target-speech-free 数据构造 [§3.3, §D.1]**

[论文原文] GRPO 不需要人工构造的目标录音。将 TTS 语料的 (speech, text) 对转化为编辑 prompt: 原始语音不变,通过 NLP perturbation 从原始文本生成目标文本 [§3.3]。

五种 perturbation 操作 [§D.1, Fig 5]:
- Insertion: 用 RoBERTa masked LM 插入词
- Deletion: 随机删除词
- Substitution: 用 RoBERTa masked LM 替换词
- Swap: 重排相邻词序
- Multi-edit: 以上操作的随机组合

编辑长度约束: `N_edit ≤ max(1, ⌊|X_ori|/2⌋)` [Eq. 18]

[agent 解读] 这个设计的深层意义在于: 传统 speech editing 的 SFT 数据需要"真实编辑后的语音",而 CosyEdit2 的 GRPO 数据只需要"编辑指令"(即文本变化),reward 计算代替了目标录音的监督角色。这使得训练数据可从任意 TTS 语料无成本生成,仅用 3000 句 GigaSpeech-XL 子集 [§D.4]。

**3. Editing-oriented 三维 Reward 设计 [§3.3, §D.5]**

三个 reward 按编辑偏好的优先级层次组合:

| Reward | 公式 | 角色 | 粒度 |
| --- | --- | --- | --- |
| r_wer | exp(-k_w * w^α), k_w=12, α=1.5 [Eq. 1] | 内容正确性门控 | 全句粗粒度 |
| r_mcd | exp(-k_m * max(m - δ, 0)), k_m=0.2, δ=2 dB [Eq. 3] | 非编辑区声学保持 | 局部细粒度 |
| r_sim | cosine similarity of speaker embeddings [Eq. 2] | 说话人一致性排序 | 全局 |

Coarse-to-fine 组合 [Eqs. 4-5]:
1. 先将 r_wer 和 r_mcd 乘性组合: `r_wer-mcd = r_wer * [(1-γ) + γ * r_mcd]`, γ=0.5
2. 再与 r_sim 加性组合: `r = λ_c * r_wer-mcd + λ_s * r_sim`, λ_c + λ_s = 1

[论文原文] r_wer 是粗粒度内容门控,在 content correctness 可比时 r_mcd 才有区分力; r_sim 在前两者都可比时做最终排序。这一层次结构反映了 speech editing 的实际偏好: 先正确编辑,再保持声学,最后保持 speaker 一致性 [§D.5]。

[论文原文] 引入 r_mcd 的必要性: 仅用 WER + speaker similarity (TTS 领域常用的两个 reward) 会导致 reward hacking——模型降低 WER 但以不自然的 word-by-word 韵律为代价 [§3.3]。

**4. MCD reward 的容忍度设计 [§D.5, Fig 6c]**

[论文原文] r_mcd 引入容忍度 δ=2 dB,使很小的 MCD 差异(通常感知不可辨)不产生 reward 梯度,优化集中在防止严重声学退化。MCD 通过 DTW 对齐计算,对 forced alignment 边界误差具有鲁棒性 [§D.5]。

**5. 动态 reward 权重调度 [§4.1]**

训练中 (λ_c, λ_s) 从 (0.9, 0.1) → (0.8, 0.2): 前 290 步优先保证内容正确性,最后 90 步加强 speaker consistency [§4.1]。

### 训练策略

**Stage 1 (SFT) [§C]:**
- LLM: GigaEdit-S 250h, lr=1e-6, warmup=2500 steps, 8 epochs, gradient clip=5
- Flow (GOT-CFM): GigaEdit-S 250h, lr=3e-5, 9 epochs
- BigVGAN: 625h (585h LibriTTS/LibriTTS-R + 40h YODAS2), 460k steps; 从 bigvgan_v2_22khz_80band_256x 初始化,复用 88.20% generator 参数 [§C.3]

**Stage 2 (GRPO) [§4.1]:**
- 仅更新 LLM (从 Stage-1 第 8 epoch checkpoint 初始化)
- 3000 句 GigaSpeech-XL,5 种 nlpaug perturbation
- G=4 rollouts/prompt, 380 steps total
- lr=3e-6, KL coefficient=0.001, batch size=64
- Temperature 0.8, top-p=0.95, top-k=25
- 2x NVIDIA H800 GPUs

**推理配置 [§4.1]:**
- Speech editing: GRPO-LLM + Stage-1 Flow + Stage-1 BigVGAN (完整编辑管线)
- Zero-shot TTS: GRPO-LLM + 原版 CosyVoice2 Flow + HiFT-GAN (仅替换 LLM,隔离 GRPO 对 LLM 的效果) [§4.1]

[agent 解读] 推理时 TTS 模式不用 BigVGAN 而用 HiFT-GAN 是一个重要的实验设计: 它确保 TTS 性能提升完全来自 LLM 策略变化而非声码器升级,严格隔离了 GRPO 的效果。

## 实验

### Speech Editing (Ming-Freeform-Audio-Edit, English) [Table 1]

| Edit Type | Model | WER↓ basic/full | SS↑ basic/full | MAE_DNSMOS↓ basic/full |
| --- | --- | --- | --- | --- |
| Insertion | SSR-Speech (cascade) | 1.75/2.03 | 0.94/0.94 | 0.139/0.128 |
| Insertion | **CosyEdit2** | **1.90/1.93** | 0.93/0.93 | **0.107/0.108** |
| Deletion | SSR-Speech | **5.22/5.29** | **0.91/0.91** | 0.132/0.134 |
| Deletion | **CosyEdit2** | 5.52/5.83 | 0.90/0.90 | **0.131/0.131** |
| Substitution | SSR-Speech | 1.90/1.95 | **0.89/0.90** | 0.146/0.140 |
| Substitution | **CosyEdit2** | **1.43/1.52** | **0.89/0.90** | **0.137/0.132** |

### Ablation (RealEdit, 310 in-the-wild samples) [Table 2]

| LLM | Flow | BigVGAN | WER↓ | SS↑ | MCD↓ | MAE_DNSMOS↓ |
| --- | --- | --- | --- | --- | --- | --- |
| CosyVoice2 (baseline) | x | x | 4.14 | 96.65 | 6.68 | 0.275 |
| SFT | x | x | 5.83 | 97.05 | 5.82 | 0.207 |
| GRPO | x | x | 4.71 | 97.23 | 5.50 | 0.210 |
| GRPO | ✓ | x | 4.34 | 97.79 | 4.07 | 0.134 |
| GRPO | ✓ | ✓ | **4.31** | **97.91** | **3.93** | **0.131** |

### Zero-Shot TTS (CV3-Eval) [Tables 3-5]

| 指标 | CosyVoice2 | CosyEdit2 | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| CER (%) zh | 3.77 | **3.52** | CV3-Eval multilingual | [Table 3] |
| WER (%) en | 5.45 | **4.87** | CV3-Eval multilingual | [Table 3] |
| WER (%) ja | 7.76 | **6.16** | CV3-Eval multilingual | [Table 3] |
| WER (%) ko | 6.89 | **5.14** | CV3-Eval multilingual | [Table 3] |
| CER (%) hard-zh | 15.70 | **8.06** | CV3-Eval hard | [Table 4] |
| WER (%) hard-en | 8.11 | **5.93** | CV3-Eval hard | [Table 4] |
| CER (%) zh | 1.36 | **1.16** | SEED-TTS-Eval | [Table 9] |
| WER (%) en | 3.10 | **1.95** | SEED-TTS-Eval | [Table 9] |

### Speech Preservation (Identity Editing, RealEdit) [Table 11]

| Method | SS↑ | MCD↓ |
| --- | --- | --- |
| CosyVoice2 (zero-shot TTS mode) | 96.92 | 6.24 |
| CosyEdit2 (identity edit) | **99.08** | **3.07** |
| BigVGAN oracle reconstruction | 99.25 | 2.81 |

## 局限性

1. **语言覆盖受限**: 受 CosyVoice2 限制仅支持中英日韩四种语言,低资源语言和方言未探索 [§Limitations]
2. **GRPO reward 设计空间未充分探索**: 当前 reward 基于任务理解和人工听评迭代调参,更细粒度的编辑区/非编辑区分离 reward 可能进一步改善 [§Limitations]
3. **仅限内容编辑**: 未探索情感转换、音高调节、说话风格控制等 paralinguistic 编辑 [§Limitations]
4. **未开源**: 论文发表时模型和代码均未公开
5. **MCD reward 依赖 forced alignment 质量**: WhisperX 的词级对齐误差会影响非编辑区域 Ω 的识别精度,进而影响 MCD reward 的可靠性 [§D.3]
6. **GRPO 仅用英文数据训练**: 跨语言迁移效果令人印象深刻但缺乏理论保证; 作者归因于"GRPO 强化的是 prompt-conditioned in-context learning 能力而非语言特定模式" [§4.5],这是合理的假说但未有 ablation 直接验证

## 点评

**核心贡献:**
- 首次在端到端 speech editing 模型上系统性引入 GRPO,并设计了编辑专用的三维分层 reward。与 ECPA 的 self-consistency GRPO 相比,CosyEdit2 的 teacher-free outcome-level 优化更直接地对准编辑偏好 [§2.2]
- 发现并实验验证了 speech editing 和 zero-shot TTS 的双向增益关系: editing-oriented GRPO 训练不仅提升编辑,还提升 TTS。hard-zh CER 从 15.70→8.06,hard-en WER 从 8.11→5.93 的改进尤为显著 [Table 4]
- Target-speech-free 数据构造完全消除了 SFT 数据中"不完美目标录音"的瓶颈,仅 3000 句即足够 GRPO 训练 [§D.4]

**SFT→GRPO 如何打破 preservation-accuracy trade-off:**
- Ablation [Table 2] 清晰展示了这一核心论点: SFT 将 SS 从 96.65 提升到 97.05 但 WER 从 4.14 恶化到 5.83; GRPO 同时改善两者 (WER 4.71, SS 97.23)。这不是边际改善,而是 trade-off 曲线的帕累托前沿推进

**Reward 设计的 engineering insight:**
- 乘性 WER-MCD 组合 (Eq. 4) 的直觉是让 WER 作为"准入门槛": 只有内容基本正确的样本,其声学保持质量才被考虑。这避免了"声学完美但内容错误"的样本获得高奖励
- 动态权重调度 (前 290 步 λ_c=0.9 → 后 90 步 λ_c=0.8) 体现了 curriculum 思想: 先学对内容,再学保持 speaker

**不足:**
- **跨语言迁移的因果解释缺乏 ablation 支持**: 论文称 GRPO 强化的是"shared in-context learning capability"[§4.5],但未提供消融实验(如在中文数据上也做 GRPO 对比纯英文 GRPO)来区分这一效果与简单的 LLM policy 改善
- **Deletion 场景仍有差距**: 在 Ming-Freeform-Audio-Edit 上 deletion WER (5.52/5.83) 明显高于 insertion (1.90/1.93) 和 substitution (1.43/1.52),且低于 cascade SSR-Speech。[论文原文] 归因于 cascade 系统的显式 speech-text alignment 简化了删除定位 [§4.2],但未提出针对性解决方案
- **DNSMOS 绝对值的解读局限**: 论文正确指出绝对 DNSMOS 在 speech editing 中可能具有误导性(高 DNSMOS 可能反映隐式去噪而非忠实保持)[§E.1],引入 MAE_DNSMOS 是合理改进,但该指标本身仍受 DNSMOS 模型偏差影响

## 可复用的 idea

1. **Target-speech-free 编辑数据构造**: 任何有转写的 TTS 语料都可通过 NLP perturbation 转化为 GRPO 编辑训练数据,完全消除人工目标录音的需求。这个方法可推广到 voice conversion、accent adaptation 等需要"编辑但无完美目标"的场景
2. **Priority-aware reward composition**: 乘性门控(WER → MCD) + 加性排序(→ SIM)的层次结构反映任务偏好优先级,可迁移到任何需要多维 reward 的 RL 任务(如视频编辑: 内容正确 → 时域一致 → 风格保持)
3. **模块化 RL**: GRPO 仅更新 LLM 而冻结下游 Flow + Vocoder,将 RL 搜索空间限制在语义 token 层面。这比在完整管线上做 RL 更稳定高效,可应用于任何 coarse-to-fine 生成系统
4. **Zero-shot TTS 作为 speech editing 特例**: 这个统一视角暗示 editing-oriented 训练可以是通用 TTS improvement 的有效路径。未来可探索: 先训 editing → 再做 TTS 部署,利用 editing 的更强约束作为"难题训练"提升模型能力
5. **MCD 容忍度设计**: r_mcd 的 δ=2 dB 阈值忽略感知不可辨的小差异,避免过度优化。这个"tolerance margin"思想可用于任何 perceptual metric 驱动的 RL(如图像编辑中的 SSIM reward)
