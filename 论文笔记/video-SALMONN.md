---
type: paper
tier: deep
title: "video-SALMONN: Speech-Enhanced Audio-Visual Large Language Models"
arxiv_id: "2406.15704"
source: "Sources/video-SALMONN.pdf"
authors: [Guangzhi Sun, Wenyi Yu, Changli Tang, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, Yuxuan Wang, Chao Zhang]
year: 2024
venue: "ICML 2024"
tags: [audio-visual, speech-LM, multimodal, Q-Former, LoRA, video-understanding, causal-attention, multi-resolution, diversity-loss, mixed-training, av-LLM]
concepts: ["[[SpeechLanguageModel]]", "[[ModalityAdaptationforSpeechLLM]]", "[[AudioUnderstanding]]", "[[Audio-LanguagePretraining]]", "[[Speech-LLMIntegrationTaxonomy]]"]
models: ["[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 5 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: video-SALMONN 是 [[论文笔记/SALMONN|SALMONN]] 的视频扩展版本,从纯音频理解扩展到音频-视觉联合理解。在 KB 的分类中,它属于 latent-representation-based Speech-LLM integration 路线 [Speech-LLMIntegrationTaxonomy],继承 SALMONN 的双编码器 (Whisper + BEATs) 设计并增加视觉编码器 (InstructBLIP)。在 ALM 分类中仍属于 "Two Heads" 架构 [Audio-LanguagePretraining],但从 two-head 扩展为 three-head (speech + audio + visual)。
>
> **已有认知**: KB 中 Modality Adaptation 页面记录了三种适配方法 (Conv downsampling, CTC compression, Q-Former),SALMONN 系列是 Q-Former 路线的代表。video-SALMONN 在此基础上提出 MRC Q-Former (Multi-Resolution Causal Q-Former),将 Q-Former 从单分辨率扩展到多分辨率 + 因果注意力结构,是对 Q-Former adapter 的重要架构创新。SALMONN 原版的 window-level Q-Former (每窗口 1 query, 0.33s) 在 video-SALMONN 中被替换为多分辨率版本 (0.5s 和 5s 两级)。Whisper encoder 是 SpeechLM 最流行的 speech encoder [Whisper]。
>
> **创新判断**: video-SALMONN 的核心创新在于 (1) MRC Q-Former 的多分辨率 + 因果结构; (2) diversity loss 鼓励 Q-Former 输出多样化; (3) unpaired audio-visual mixed training 避免模态主导。KB 中尚无讨论音频-视觉联合 Q-Former 或 diversity loss 的概念页。与 SALMONN 的 activation tuning 不同,video-SALMONN 的训练创新集中在避免模态/帧主导而非 task over-fitting。
>
> **与同系列论文的关系**: video-SALMONN (2024.06, ICML 2024) 是 SALMONN (2023.10, ICLR 2024) 的视频扩展。后续 [[论文笔记/SALMONN-omni|SALMONN-omni]] (2025.05) 转向全双工对话方向,采用完全不同的架构 (Mamba encoder + codec-free + CosyVoice2 synthesizer)。三篇论文共享核心团队 (Sun, Yu, Tang, Chen 等, 清华+字节)。
>
> 检索命中: [[SpeechLanguageModel]]✓, [[ModalityAdaptationforSpeechLLM]][待确认], [[AudioUnderstanding]][待确认], [[Audio-LanguagePretraining]][待确认], [[Speech-LLMIntegrationTaxonomy]][待确认], [[Whisper]][待确认] | 过滤: 无 | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 首个能同时理解视觉帧序列+语音+音频事件+音乐的端到端 av-LLM,通过多分辨率因果 Q-Former (MRC Q-Former) + diversity loss + unpaired 混合训练,在 Video QA 上比 InstructBLIP 提升 25%+,在含语音的 AVQA 上比 Video-LLaMA 提升 30%+
> - **路线**: Video frames (2Hz) → InstructBLIP ViT (冻结) + Audio → Whisper encoder + BEATs encoder (冻结) → 帧级时序同步 (0.5s) → MRC Q-Former (多分辨率滑窗: 0.5s + 5s, 因果自注意力, 可训练) → 投影层 → Vicuna 13B + LoRA → text response
> - **指标**: Video QA 49.6% vs InstructBLIP 24.7% (+25%); AVQA Presentation 70.5% vs Video-LLaMA 21.3% (+49%); ASR WER 2.6% (接近 Whisper 2.9%); AVM 79.7% (零样本跨模态推理); AVSR WER 7.7% vs Whisper 8.3% (-7.2% rel.) [Table 2, 3]
> - **可借鉴**: (1) 多分辨率 Q-Former: 高分辨率管语音/时序,低分辨率管语义/全局,互补后由 LLM 融合; (2) diversity loss: 用 cosine similarity 惩罚冗余 query,适用于所有 Q-Former 架构; (3) unpaired 混合训练: 将独立音频和视频任务混合为伪 AV 任务,迫使模型均衡关注双模态
> - **局限**: 仅支持短视频 (~25s); 视觉帧率仅 2Hz (低于人眼感知); LoRA 仅 0.4% 参数; 训练数据 <1M 样本; SAVE benchmark 自建,外部复现有限; 语音输出未涉及

## 核心问题

**想解决什么**: 当时 (2024 年) 的音频-视觉 LLM (av-LLM) 如 Video-LLaMA 和 Macaw-LLM 虽然支持视频和音频输入,但**不理解人类语音** -- 它们的音频编码器仅处理非语音音频事件 (如背景音乐、环境声),语音内容被完全忽略。然而语音是视频中最丰富的信息载体,提供语义、副语言 (语调/音高)、说话人属性 (年龄/性别/口音) 等信息 [§1]。

**为什么难**: 语音理解需要时间粒度极细的建模 (ASR 需要逐帧对齐),而视觉理解只需粗粒度全局信息 (Video QA 往往只需少数关键帧)。两者的时间分辨率需求存在根本矛盾: 单一分辨率的特征提取无法同时满足两者 [§1, Fig 4]。更进一步,将语音加入 av-LLM 后,模型容易出现帧主导 (某些帧霸占所有 query 输出) 或模态主导 (只关注视觉忽略音频) 的问题 [§3.3]。

**怎么切入**: (1) 用 MRC Q-Former 在多个时间尺度 (0.5s 精细 + 5s 粗粒) 同时提取信息,高分辨率负责语音,低分辨率负责视觉语义; (2) 用 diversity loss 惩罚 query 冗余,避免帧主导; (3) 用 unpaired 混合训练避免模态主导。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

video-SALMONN 由五个组件构成 [Fig 1]:

1. **Whisper encoder (冻结)**: Whisper-large-v2,编码语音信息,输出 50Hz 帧率 [§3.1]。

2. **BEATs encoder (冻结)**: 自监督音频编码器,编码非语音音频事件 (音乐、环境声) [§3.1]。

3. **InstructBLIP visual encoder (冻结)**: ViT + Q-Former 结构,以 2Hz 帧率编码视频帧,每帧输出 32 个 feature vector [§4.2]。

4. **MRC Q-Former (可训练)**: 核心创新。将时序同步后的三模态特征在多个分辨率下处理,输出固定数量的 query vector 送入 LLM [§3.2]。

5. **Vicuna 13B + LoRA (LLM 冻结, LoRA 可训练)**: LoRA rank=32,应用于 attention Q/K/V 和 FFN,占总参数 0.4% [§4.2]。

**时序同步** [§3.1]: 音频和视觉帧在每个视频帧处 (每 0.5s) 同步。Whisper 和 BEATs 的帧级输出 h^S_t, h^A_t 与 InstructBLIP 的帧输出 h^V_t 沿特征维度拼接:
```
h^SAV_t = Concat(s_t, a_t, v_t)   [Eq. 1]
```
缺失模态用零向量填充。单张图片被复制为匹配音频长度的视频帧 [§3.1]。

### 关键设计选择

**为什么需要多分辨率 Q-Former 而非单一 Q-Former?** [论文原文] 语音内容需要时间精细的建模 -- 当滑窗变小 (高分辨率) 时 ASR WER 降低,但 Video QA 准确率同时下降,因为每个窗口的 query 数减少无法捕获充分的视觉信息。反之,当滑窗变大 (低分辨率) 时 Video QA 改善但 ASR 恶化 [Fig 4]。这个 trade-off 验证了多分辨率的必要性 [§5.3]。

[agent 解读] 与 SALMONN 原版的 window-level Q-Former (固定 N=1, L=17) 相比,video-SALMONN 的 MRC Q-Former 是从"固定单分辨率"到"自适应多分辨率"的升级。原版每窗口 1 个 query 虽然极度压缩但时序保持好,适合纯音频; video-SALMONN 需要在语音和视觉之间平衡,因此采用两个分辨率层级分工互补。

**MRC Q-Former 的具体机制** [§3.2, Fig 2]:
- 将输入序列分为固定长度的滑窗,在每个分辨率 r 上用 N^(r) 个可学习 query vector 通过 cross-attention 从窗口内提取信息 [Eq. 2]:
  ```
  h^(r)_{w,1:N(r)} = Q-Former_MRC(h^SAV_{t:t+k(r)}; q^(r)_{1:N(r)})
  ```
- 两个分辨率: 高分辨率 (0.5s, k=1 帧, N=3 queries/窗) 和低分辨率 (5s, k=10 帧, N=30 queries/窗) [§4.2]
- 约束条件: W^(r) x N^(r) = C (总 query 数固定) [Eq. 3]。对 25s 视频,C=150
- 不同分辨率共享 Q-Former 参数 (除 query vector 外),因为对齐任务本质相同 [§3.2]
- 多分辨率输出通过投影层加权求和: H = W^(1)H^(1) + ... + W^(R)H^(R) [Eq. 4]

**因果自注意力** [§3.2.1, Fig 3]: 在标准 Q-Former 的 self-attention 和 cross-attention 之上,增加一个因果自注意力模块 (block-wise triangular mask)。编码某一帧时包含所有前序帧的信息 (自回归方式传递)。[论文原文] 这对因果推理问题 ("what happens next") 特别有益,因为这类时序依赖仅靠位置编码难以学习 [§3.2.1]。

**为什么 Q-Former 而非其他 adapter?** [agent 解读] SALMONN 系列一贯使用 Q-Former (源自 BLIP-2)。KB 已记录 Q-Former > CTC > Conv 的性能排序 [ModalityAdaptationforSpeechLLM]。video-SALMONN 的创新在于改造 Q-Former 内部结构 (多分辨率 + 因果),而非替换 Q-Former。

### 训练策略

**多任务指令微调** [§4.3]: MRC Q-Former + LoRA 联合训练。训练数据包含单模态和音频-视觉配对数据,总量 <1M 样本 (其中 <300k 视频样本),全部使用公开数据集:

| 模态 | 数据 | 规模 |
| --- | --- | --- |
| Audio-only | LibriSpeech (ASR), AudioCaps (AAC) | ~460h + 46k 样本 |
| Visual-only | LLAVA-150k, OCRVQA, TextCaps, NExT-QA, COCO spoken captions, VideoChat | ~170k 样本 |
| Audio-visual | Ego4D captioning, How2 AVSR, AVSD dialogue | ~600h + 300h + 对话集 |

**Diversity loss** [§3.3, Eq. 6]: 在低分辨率层级的 Q-Former 输出上施加 cosine similarity 惩罚:
```
L_diverse = ΣΣΣ sim(h^(r)_{w,i}, h^(r)_{w,j})   for r≥2, all windows w, all query pairs i≠j
```
[论文原文] 视频任务 (如 Video QA) 的训练数据通常只需要一两个关键帧,导致输出 queries 倾向于重复捕获相同信息。Diversity loss 鼓励提取更多样化的信息。仅在低分辨率层级应用,因为低分辨率窗口内有足够多的帧来提取多样信息 [§3.3]。

[agent 解读] 选择 cosine similarity 而非其他距离度量是合理的 -- Q-Former 输出经 layer normalization 后模值相近,cosine similarity 直接衡量语义相似度。且 Q-Former 输出最终被投影到 LLM 的 token embedding 空间,该空间中 cosine similarity 是标准语义度量。

**总 loss**: L = L_CE + λ · L_diverse [Eq. 7],λ 控制 diversity loss 权重。分析 [Fig 5] 显示 λ 过大会导致输出 queries 过于多样,混淆 LLM 并引发严重幻觉 (WER 中 insertion rate 飙升) [§5.4]。

**Unpaired audio-visual mixed training** [§3.3]: 除少量配对 AV 数据外,将一部分训练集用 unpaired 的音频和视频数据增强 -- 随机将不同来源的音频和视频组合,prompt 同时包含音频任务和视频任务。[论文原文] 这迫使模型必须从两个模态分别提取信息,不能只依赖主导模态。这是音频-视觉联合理解和 co-reasoning 能力的关键因素 [§3.3, §5.5]。

[agent 解读] 这与数据增强中的 CutMix/MixUp 思路异曲同工 -- 通过打破输入间的自然相关性,迫使模型学习独立的模态表征而非走捷径 (只看视觉)。简洁高效的设计。

### 分辨率功能分工分析

[§5.3, Table 5] 通过 zero-mask 实验揭示了两个分辨率层级的功能分化:
- **0.5s 高分辨率**: 负责语音内容 (ASR WER 2.6%),但几乎无法做 Video QA (14.4%) 和 IC (35.8)
- **5s 低分辨率**: 负责视觉语义 (Video QA 41.9%, IC 23.0),但完全无法做 ASR (WER >100%)
- **两级联合**: ASR 2.6%, Video QA 49.6%, IC 89.6 -- LLM 将两个分辨率的互补信息融合为最优结果

[agent 解读] 这是一个很漂亮的实验设计和结论: 模型自发学会了功能分工,无需显式监督。高分辨率窗口 (0.5s, 3 queries/窗) 每个 query 覆盖 ~167ms,接近语音的音素级粒度; 低分辨率窗口 (5s, 30 queries/窗) 每个 query 覆盖 ~167ms (同样的!),但 cross-attention 可以看到 5s 内的全局信息。关键区别不在 query 粒度而在感受野。

## 实验

### SAVE Benchmark

video-SALMONN 引入 SAVE (Speech-Audio-Visual Evaluation) 评测基准,包含 6 个单模态任务和 4 个跨模态任务 [Table 1]:

| 类别 | 任务 | 测试集 | 指标 |
| --- | --- | --- | --- |
| Single-modal | ASR, AAC, IC, OCR, VQA, Video QA | LibriSpeech, AudioCaps, Flickr30k, TextVQA, GQA, NExT-QA | WER, SPIDEr, CIDEr, Accuracy |
| Audio-visual | AVSR, AVQA, AVSSD, AVM | How2, Ego4D+Presentation-QA, VGGSS, SpokenCOCO+VGGSS | WER, Accuracy |

### 主要结果

| 指标 | 本文 (13B) | 最佳 Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| ASR WER | 2.6% | Whisper: 2.9% | LibriSpeech test-clean | [Table 2] |
| AAC SPIDEr | 49.7 | - | AudioCaps test | [Table 2] |
| Video QA Acc | 49.6% | InstructBLIP FT: 24.7% | NExT-QA | [Table 2] |
| IC CIDEr | 89.6 | InstructBLIP: 84.5 | Flickr30k | [Table 2] |
| OCR Acc | 37.8% | InstructBLIP: 36.5% | TextVQA | [Table 2] |
| VQA Acc | 44.8% | InstructBLIP: 48.9% | GQA | [Table 2] |
| AVSR WER | 7.7% | Whisper: 8.3% | How2 dev5 | [Table 3] |
| AVQA (Ego4D) | 49.8% | Video-LLaMA: 18.2% | Ego4D-QA | [Table 3] |
| AVQA (Presentation) | 70.5% | Video-LLaMA: 21.3% | Presentation-QA | [Table 3] |
| AVSSD Acc | 47.6% | InstructBLIP FT: 41.9% | VGGSS | [Table 3] |
| AVM Acc | 79.7% | Video-LLaMA: 52.3% | SpokenCOCO+VGGSS | [Table 3] |
| LRS2 WER | 4.9% | Whisper: 5.3% | LRS2 clean | [Table 9] |
| MUSIC-AVQA | 52.6% | AV-LLM: 45.2% | MUSIC-AVQA (zero-shot) | [Table 10] |

### Ablation 结果

| 消融变体 | Video QA | AVQA (avg) | AVM | 出处 |
| --- | --- | --- | --- | --- |
| Full model | 49.6% | 60.2% | 79.7% | [Table 4] |
| w/o 5s-resolution | 47.2% | 57.2% | 77.5% | [Table 4] |
| w/o 0.5s-resolution | 49.9% | 58.9% | 80.6% | [Table 4] |
| w/o mixed training | 46.9% | 54.0% | 75.3% | [Table 4] |
| w/o diversity loss | 49.3% | 53.5% | 78.6% | [Table 4] |
| w/o MRC Q-Former | 42.7% | 45.3% | 74.5% | [Table 4] |
| w/o MRC + sync + div | 36.0% | 44.6% | 72.0% | [Table 4] |

**关键观察**:

1. **MRC Q-Former 是核心**: 去除 MRC Q-Former 后 Video QA 从 49.6% 降到 42.7% (-6.9%),AVQA 从 60.2% 降到 45.3% (-14.9%)。进一步去除同步和 diversity 后 AVQA 降到 44.6%,近似 Video-LLaMA 的架构 [Table 4]。

2. **Mixed training 对跨模态至关重要**: 去除后 AVQA 从 60.2% 降到 54.0% (-6.2%),AVM 从 79.7% 降到 75.3%,说明 unpaired 混合训练是跨模态 co-reasoning 的关键 [Table 4]。

3. **两个分辨率互补**: 去除 5s 分辨率影响视觉相关任务 (Video QA -2.4%),去除 0.5s 分辨率影响语音相关任务 (ASR +0.3%, AVSR +0.6%) [Table 4]。

4. **7B vs 13B**: 13B 在大部分任务上优于 7B,但差距不大 (Video QA 49.6% vs 42.5%) [Table 2]。

5. **Vicuna vs Llama-2**: Vicuna-v1.5 在需要指令理解的任务上 (Video QA 49.6% vs 36.7%) 大幅优于 Llama-2,但 Llama-2 在 AC 和 IC 上略优 [Table 11, 12]。说明 instruction-tuned LLM backbone 对需要复杂推理的视频任务至关重要。

6. **Image spotlight**: 将图片分割为子区域序列送入 MRC Q-Former,OCR 从 37.8% 飙升至 56.1%,但略降 Video QA 和 IC [Table 13],表明视觉分辨率是 OCR 瓶颈。

## 局限性

1. **仅限短视频 (~25s)**: MRC Q-Former 的总 query 数 C 固定,25s 视频已产生 150 个 token。更长视频要么需更多 token (超出 LLM 上下文),要么需更低帧率 (更粗分辨率) [§4.2]。

2. **视觉帧率低 (2Hz)**: 0.5s 一帧无法捕捉快速视觉变化 (如体育动作、手势)。论文承认可以提高帧率,但计算和存储成本成正比 [§3.1]。

3. **VQA 未超越 InstructBLIP**: 在 GQA 上 video-SALMONN (44.8%) 低于 InstructBLIP (48.9%),说明多模态学习对纯视觉任务有负迁移 [Table 2]。

4. **仅支持理解,不支持生成**: 与 SALMONN 一样,只能输出文本,不能输出语音或视频。

5. **SAVE benchmark 局限**: 部分测试集由 GPT-4/ChatGPT 生成问题 (Ego4D-QA, Presentation-QA),评估中也使用 ChatGPT 辅助打分 (AVSSD),可复现性和评估可靠性受限 [Appendix B, C, D]。

6. **训练数据规模小**: <1M 样本,<300k 视频,远少于后续的 Gemini (数十亿级) 或 GPT-4V。且仅使用公开数据,覆盖场景有限。

7. **LoRA 容量约束**: 仅 0.4% 参数可训练 (LoRA rank=32),可能限制了模型对复杂跨模态关系的学习能力 [§4.2]。

## 点评

**从 SALMONN 到 video-SALMONN 的自然扩展**: video-SALMONN 的核心贡献不在于"把视频加入 SALMONN"这个方向 (这是显然的),而在于 MRC Q-Former 的设计和两个训练技巧的提出。MRC Q-Former 巧妙地将多分辨率问题转化为 Q-Former 的滑窗参数选择问题,复用了 Q-Former 的成熟架构而非从头设计新模块。共享 Q-Former 参数 (仅 query vector 分辨率独立) 进一步控制了参数量。

**Diversity loss 的普适性**: 这个 loss 不仅适用于 av-LLM,而是所有使用 Q-Former 或 learned queries 的场景的通用问题: 当输入序列存在冗余时 (视频中大部分帧内容相似),learned queries 容易学到相似的特征提取模式。Cosine similarity 惩罚简洁有效,但 λ 的调优 (过大导致幻觉) 需要注意 [Fig 5]。

**Mixed training 的洞察**: unpaired AV 混合训练本质上是一种对抗模态捷径 (modality shortcut) 的策略。在多模态学习中,模型倾向于只使用最容易的模态而忽略其他模态 (lazy multimodal learning)。将不相关的音频和视频配对,迫使模型不能通过模态间的自然相关性走捷径,必须分别理解每个模态。

**历史定位**: video-SALMONN 发表于 2024 年 6 月 (ICML 2024),处于 av-LLM 爆发的早期阶段。与同期的 Video-LLaMA, Macaw-LLM 相比,它的独特性在于理解语音 (而非仅非语音音频)。但后续的 Gemini, GPT-4o 等商业模型在更大规模数据和更强 backbone 下已大幅超越,video-SALMONN 的绝对性能已非 SOTA。其方法论贡献 (MRC Q-Former + diversity loss + mixed training) 仍有参考价值。

**与 SALMONN 系列的对比**: 有趣的是,原版 SALMONN 的核心贡献是 activation tuning (解决 task over-fitting),而 video-SALMONN 的核心贡献是 MRC Q-Former + diversity loss + mixed training (解决帧/模态主导)。两者面对的问题不同: SALMONN 的问题是"能力被压制",video-SALMONN 的问题是"注意力不均衡"。后续 SALMONN-omni 则转向完全不同的全双工方向,三篇论文共享团队但架构演化迅速。

## 可复用的 idea

1. **多分辨率 Q-Former**: 在任何需要处理不同时间粒度信息的场景中,可将 Q-Former 的滑窗设为多个分辨率。关键约束: 总 query 数 C 固定,不同分辨率的 queries/窗 x 窗数 = C [Eq. 3]。可推广到 TTS 中的 prompt encoding (短时音色 + 长时风格)、音频事件检测 (精确时间定位 + 事件类型识别) 等 [§3.2]。

2. **Diversity loss for Q-Former outputs**: 当使用 learned queries 从序列中提取信息时,施加 cosine similarity 惩罚可避免 query 冗余。适用条件: 输入序列中存在大量重复信息 (如视频帧)。不适用条件: 每个 query 有明确的预设功能 (如 token-level ASR)。需小心调 λ — 过大导致 query 语义过于分散,LLM 无法利用 [§3.3, Fig 5]。

3. **Unpaired multimodal mixed training**: 将不同模态的独立数据组合为伪多模态样本,迫使模型均衡关注各模态。这个思路可推广到任何多模态训练中存在模态主导问题的场景。成本极低 — 仅是训练时数据组合方式的改变,无需额外数据标注 [§3.3]。

4. **因果自注意力 for sequential tokens**: 在需要建模时序因果关系的场景中,在特征提取器内部添加 block-wise 因果 mask,使当前帧的表征包含历史信息。与位置编码互补,特别有利于 "what happens next" 类推理 [§3.2.1]。

5. **多分辨率功能分工的诊断方法**: 通过 zero-mask 某一分辨率的输出,可以精确测量各分辨率的功能贡献。这个诊断方法可推广到任何多头/多路径架构的功能分析 [§5.3, Table 5]。

## 审阅

> [!review] 审阅: pass (0 high, 0 medium, 0 low)
> 审阅日期: 2026-06-08 | checklist v1.1
> 详见 `_review/video-SALMONN-review.yml`
