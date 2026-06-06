---
type: paper
tier: deep
title: "ELLSA: End-to-End Listen, Look, Speak and Act"
arxiv_id: "2510.16756"
source: "Sources/ELLSA.pdf"
authors: [Siyin Wang, Wenyi Yu, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Lu Lu, Chao Zhang]
year: 2026
venue: "arXiv (Tsinghua / ByteDance)"
tags: [full-duplex, multimodal, speech-LM, embodied-AI, MoE, VLA, robot-manipulation, turn-taking, barge-in, streaming]
concepts: ["[[Full-duplexSpokenDialogue]]", "[[Turn-takinginSpokenDialogue]]", "[[StreamingSpokenDialogue]]", "[[ModalityAdaptationforSpeechLLM]]", "[[Speech-LLMIntegrationTaxonomy]]", "[[SpeechLanguageModel]]"]
models: ["[[CosyVoice2]]"]
tasks: []
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-06
updated: 2026-06-06
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认 + 5 个待确认实体页: [[SpeechLanguageModel]]✓, [[Full-duplexSpokenDialogue]], [[Turn-takinginSpokenDialogue]], [[StreamingSpokenDialogue]], [[ModalityAdaptationforSpeechLLM]], [[Speech-LLMIntegrationTaxonomy]])
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: [[SpeechLanguageModel]], [[Full-duplexSpokenDialogue]], [[Turn-takinginSpokenDialogue]], [[StreamingSpokenDialogue]], [[ModalityAdaptationforSpeechLLM]], [[Speech-LLMIntegrationTaxonomy]] | 过滤: 无 | 未命中但可能相关: 无

**谱系定位**: ELLSA 处于 full-duplex spoken dialogue 和 vision-language-action (VLA) 的交叉点。在 full-duplex 演进线上,已有系统 (Moshi, LSLM, Freeze-Omni, OmniFlatten 等) 解决了"边说边听"问题,但仅限于语音和文本模态,无法与物理环境交互。在 VLA 演进线上,pi0, OpenVLA 等模型实现了视觉语言驱动的机器人操控,但缺乏语音理解和全双工能力。ELLSA 试图将两条线合并: 一个模型同时支持 listen+look+speak+act。

**已有认知**:
- [[Full-duplexSpokenDialogue]][待确认]: 全双工从 dGSLM → Moshi → LSLM 演进,核心是同时处理输入输出流。ELLSA 在此基础上扩展到四模态 MIMO。
- [[Turn-takinginSpokenDialogue]][待确认]: 现有 turn-taking 方案 (IRQ token, chunk-level state prediction, SIL/BOW/BC) 聚焦语音轮次;ELLSA 新增"action turn-taking"和"action barge-in"概念。
- [[StreamingSpokenDialogue]][待确认]: 流式架构 (block-by-block, chunk-wise) 是全双工的前提;ELLSA 采用 1 秒 time block 的 interleaved sequence 方案。
- [[ModalityAdaptationforSpeechLLM]][待确认]: MLP adapter + LoRA 是标准方案;ELLSA 的 SA-MoE 通过 shared attention 实现跨模态融合,是一种新型 modality integration 架构。
- [[Speech-LLMIntegrationTaxonomy]][待确认]: ELLSA 不属于传统三分类中的任何单一类别 — 它同时使用 latent-representation (speech encoder → MLP → LLM) 和 audio-token (action tokenizer) 两种路线。

**创新判断**: ELLSA 的核心新意在于 SA-MoE 架构 — 不同于传统 MoE (在 FFN 层路由 token),SA-MoE 让不同 expert 共享 attention 的 KV cache,使专家在保持独立 FFN 的同时可以互相"看到"对方处理的模态信息。这是一种从 pi0 (VLM+action expert 共享 attention) 扩展而来的设计,但泛化到了任意多模态 expert 组合。

## 速查

> [!summary] 速查
> - **一句话**: ELLSA 是首个端到端全双工模型,通过 SA-MoE 架构将 speech expert 和 action expert 用 shared attention 连接,实现同时听+看+说+做的四模态 MIMO 交互
> - **路线**: Speech → Mamba encoder → MLP adapter → Speech Expert (LLaMA-3.1-8B) ←SA-MoE attention→ Action Expert (Emu3-Base) ← VisionVQ + FAST tokenizer; Speech Expert → Synthesizer Adapter → CosyVoice2-0.5B → 语音输出
> - **指标**: LIBERO 平均成功率 89.4% (超 pi0-FAST 85.5%); Llama Q. S2S 70.0 (超 Freeze-Omni 56.2); 对话 turn-taking 100%; Speaking-while-acting 时性能下降可控 (SPEAR encoder 版本下降 <3%)
> - **可借鉴**: SA-MoE 的"shared KV cache + 独立 FFN"设计可迁移到任何需要融合异构预训练模型的场景; 将 VLA 和 speech LLM 通过 attention 桥接的思路
> - **局限**: 仅在 LIBERO 仿真环境验证,未部署真实机器人; 全双工场景有限 (无 backchannel); speaking-while-acting 时性能有可见下降 (尤其难题); 2-expert 设计绑定了模态分组

## 核心问题

1. **为什么现有模型要么只能"说"要么只能"做"?** 全双工 speech LLM (Moshi, Freeze-Omni) 无法执行物理动作; VLA 模型 (pi0, OpenVLA) 无法处理语音输入,只能接受文本指令且采用半双工交互 [§1]。根本原因是这两类系统的模态处理和训练数据分属不同领域,直接用单一 dense model 训练所有模态会造成严重的 modality interference [§3.2, Table 7]。

2. **如何在一个模型中统一四种模态而不互相干扰?** ELLSA 提出 SA-MoE: 不同模态由专门 expert 处理 (speech expert 管 speech+text, action expert 管 vision+action),expert 之间通过 shared self-attention (共享 KV cache) 交换信息,但 FFN 独立 [§3.2]。

3. **如何实现全双工的四模态 MIMO streaming?** 将每个 time block 内的多模态数据按固定顺序交错排列 (speech input → image input → text output → action output),speech output 从 text output 的 hidden state 派生 [§3.1]。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

ELLSA 的架构由三层组成 [§3, Fig 1]:

**输入侧**:
- **Speech**: Mamba streaming encoder (32 blocks, hidden 2048, 25Hz → 下采样到 5Hz, 即每秒 5 个 embedding) + 2-layer MLP adapter [§4.1, Appendix A.2]
- **Vision**: Emu3-VisionTokenizer,单帧 ~300 tokens [§4.1]
- **Text**: 系统 prompt 通过 speech expert 处理 [Appendix A.1]

**骨干 (SA-MoE)**:
- **Speech Expert**: LLaMA-3.1-8B-Instruct (32 layers, hidden 4096, 32 heads, 8 KV heads) + LoRA (rank 256, scale 1.0) [§4.1]
- **Action Expert**: Emu3-Base (同样 32 layers, hidden 4096, 32 heads, 8 KV heads) + LoRA (rank 256) [§4.1]
- 两个 expert 层数相同,每一层对应连接,共享 attention KV cache [§3.2, Fig 2]

**输出侧**:
- **Text**: Speech Expert 的 LM head 输出 text tokens (每个 time block 8 tokens 或 `<silence>`) [Appendix A.1]
- **Action**: Action Expert 的 LM head 输出 FAST action tokens [§4.1]
- **Speech**: Speech Expert 的 last hidden states → Synthesizer Adapter (2-layer MLP) → CosyVoice2-0.5B 的 LM 部分 (25 speech codecs per 8 text embeddings) [§4.1]

### 关键设计选择

**SA-MoE 的工作机制** [§3.2, Fig 2]:

[论文原文] 在每个 transformer 层:
1. 每个模态的 token 被路由到对应 expert: speech/text → Speech Expert, vision/action → Action Expert
2. 每个 expert 独立计算自己的 Q, K, V
3. **关键**: 在做 attention 时,每个 expert 的 Q 不仅 attend 自己的 K/V,也 attend 另一个 expert 之前产生的 K/V (共享 KV cache)
4. FFN 层保持独立 — Speech Expert 和 Action Expert 各用各的 FFN
5. 在任意时刻,只有一个 expert 的权重被激活 (因为模态在序列中交错排列)

[agent 解读] 这个设计的巧妙之处在于: 从整体序列角度看,信息流等价于一个 vanilla transformer (所有 token 可以互相 attend); 从单步角度看,只有当前模态对应的 expert 处于激活状态。这样既利用了预训练 expert 的专业能力 (FFN 不变),又通过 attention 实现了跨模态信息流动。与标准 MoE (在 FFN 层路由) 的本质区别是: SA-MoE 在 attention 层融合、FFN 层分离; 标准 MoE 在 attention 层共享、FFN 层路由。

**Streaming Full-Duplex MIMO** [§3.1, Fig 1b]:

每个 1 秒 time block 内的固定序列: `<bos> speech_input <eos> <boi> image_input <eoi> <bot> text_output <eot> <boa> action_output <eoa>`。

[论文原文] Speech output 不在主序列中,而是从 text output 的 hidden states 派生 (通过 synthesizer adapter 送入 CosyVoice2)。这避免了在主序列中额外添加 speech output tokens 导致的序列膨胀。

两种工作模式 [Appendix A.1]:
- **Default mode**: 四模态全部激活 (用于有机器人操控的场景)
- **Speech-only mode**: 只有 speech+text 激活,vision 用 placeholder, action 输出 dummy tokens

**为什么 Speech+Text 在一个 Expert, Vision+Action 在另一个?** [§3.2]

[论文原文] 为了更好利用预训练知识: Speech Expert 基于 LLaMA-3.1-8B-Instruct (本身就是 text LM,加上 speech encoder 后获得语音理解能力); Action Expert 基于 Emu3-Base (本身就是 vision-language model,加上 action tokenizer 后获得动作预测能力)。论文也承认"mouth + hand"分属不同模块的分工方式并非唯一选择,理想的"brain-hand-mouth"架构可能更优 [Appendix F]。

**RoPE 处理** [Appendix A.2]: 两个 expert 保持各自的 RoPE 设置 (LLaMA-3.1 和 Emu3 的 RoPE 参数不同),但 token index 在整个多模态序列中共享。

### 训练策略

三阶段训练 [§3.3, Fig 3]:

**Stage 1: 训练个体 Expert**
- Speech Expert: Mamba encoder + MLP adapter + LLaMA-3.1-8B (冻结) + LoRA — 训练 ASR 和 speech QA 任务,40k steps, batch 512, lr=2e-4 [Appendix B]
- Action Expert: 直接使用预训练的 UniVLA (基于 Emu3-Base,经过 world model post-training + policy learning finetuning) [§3.3]
- Mamba encoder 先在 LibriHeavy + GigaSpeech 上预训练 300k steps [Appendix B]

**Stage 2: 训练 SA-MoE**
- 将两个 expert 整合到 SA-MoE 框架中
- 训练 diverse tasks: ASR, speech QA, speech-conditioned manipulation, speaking-while-acting, context-grounded VQA, defective instruction rejection, action barge-in
- 两个 expert 均用 LoRA 微调,500 steps, batch 1024, lr=4e-4 [Appendix B]

[agent 解读] Stage 2 仅训练 500 steps 却能有效整合两个 expert,这是 SA-MoE 数据效率的直接证据。与 dense model 需要 3k steps 全参数微调 (仍大幅落后) 形成对比 [Table 7]。SA-MoE 之所以高效,是因为 expert 的 FFN 已有强大的单模态预训练能力,Stage 2 只需通过 LoRA 微调让 attention 层学会跨模态路由。

**Stage 3: 连接 Speech Synthesizer**
- CosyVoice2-0.5B 的 LM 部分被微调
- 随机初始化的 2-layer MLP adapter 连接 speech expert hidden states 和 synthesizer
- 训练任务与 Stage 2 类似 (去掉 speech-conditioned manipulation,因为该任务只输出 `<silence>`)
- 20k steps, batch 256, lr=2e-4 [Appendix B]

**训练数据规模** [Table 6]: ASR ~481k samples (LibriSpeech + GigaSpeech), QA ~728k samples (7 个 QA 数据集), LIBERO manipulation ~3.4k + defective instruction ~1.7k。数据规模不大,尤其 manipulation 数据仅数千条。

## 实验

| 指标 | 本文 (ELLSA) | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| Llama Q. Acc (S2T) | 74.7 | Freeze-Omni 74.2, Moshi 60.8 | Llama Questions | [Table 1] |
| Llama Q. Acc (S2S) | 70.0 | Freeze-Omni 56.2, Moshi 54.5 | Llama Questions | [Table 1] |
| TriviaQA Acc (S2T) | 45.2 | Freeze-Omni 45.1, Moshi 25.6 | TriviaQA | [Table 1] |
| TriviaQA Acc (S2S) | 41.7 | Freeze-Omni 28.5, Moshi 16.7 | TriviaQA | [Table 1] |
| LIBERO Average SR | 89.4% | pi0-FAST 85.5%, CoT-VLA 81.1% | LIBERO | [Table 2] |
| LIBERO LONG SR | 84.4% | pi0-FAST 60.2%, CoT-VLA 69.0% | LIBERO LONG | [Table 2] |
| Dialogue turn-taking | 100% | Freeze-Omni 72-99.8%, Moshi 37-85% | 4 datasets | [Table 3a] |
| Action turn-taking | 96.4-100% | N/A (首次测试) | LIBERO | [Table 3b] |
| Defective instruction rejection | 100% | N/A (首次测试) | LIBERO | [Table 3b] |
| Speaking-while-acting (Llama Q. S2T) | 68.9 (-7.8%) | 单独 speaking 74.7 | Llama Q. | [Table 4a] |
| Speaking-while-acting (LIBERO LONG SR) | 73.2% (-13.3%) | 单独 acting 84.4% | LIBERO LONG | [Table 4b] |
| Context-grounded VQA | 82.5% (manual) / 83.3% (Gemini) | N/A (首次测试) | 12 questions | [Table 5] |
| SA-MoE vs Dense (Llama Q. S2T) | 74.7 | Dense (from speech) 62.7, Dense (from action) 32.7 | Llama Q. | [Table 7a] |
| SA-MoE vs Dense (LIBERO Avg) | 89.4% | Dense (from speech) 2.2%, Dense (from action) 70.4% | LIBERO | [Table 7b] |
| SA-MoE vs Individual Expert (Speech) | 74.7 (-3.9%) | Speech expert alone 77.7 | Llama Q. | [Table 8a] |
| SA-MoE vs Individual Expert (Action) | 89.4% (-6.4%) | Action expert alone 95.4% avg | LIBERO | [Table 8b] |
| SPEAR encoder: speaking-while-acting gap | -2.5% (Llama Q.), -2.9% (LONG) | Mamba: -7.8%, -13.3% | Multiple | [Table 11] |
| CALVIN Avg Len | 4.43 | RoboVLMs 4.49, UP-VLA 4.42 | CALVIN | [Table 12b] |
| Per-block latency (1s block) | 854ms (S2S), 786ms (S2A) | N/A | A100 | [Table 9d] |

**关键发现**:

1. **SA-MoE >> Dense model**: Dense model 从 speech expert 初始化时几乎无法做 manipulation (SR 2.2%),从 action expert 初始化时 speech 能力极差 (Llama Q. 32.7)。SA-MoE 500 steps LoRA 就远超 dense 3k steps 全参数微调 [Table 7]。

2. **ELLSA 在 LIBERO 上超过所有 text-conditioned VLA**: 平均 89.4% 超过 pi0-FAST (85.5%),特别是 LIBERO LONG (84.4% vs 60.2%)。注意 ELLSA 是 speech-conditioned 且需自行判断何时行动 (更难的设置) [Table 2]。

3. **Speaking-while-acting 有可见性能下降**: 同时说+做时,speech 性能下降 ~8-22%,action 性能下降 0-13% (取决于任务难度)。用更强的 SPEAR encoder 替换 Mamba encoder 后,下降幅度显著缩小 (Llama Q. -2.5%, LIBERO LONG -2.9%),说明瓶颈在模型容量而非架构 [Table 11]。

4. **Dialogue turn-taking 100%**: ELLSA 在所有数据集上 100% turn-taking 成功,远超 Moshi 和 Freeze-Omni。论文假设这归因于 ELLSA 更长的 time block (1s vs Freeze-Omni 0.16s),使全双工动态更容易学习 [§5.2.1]。

## 局限性

1. **仅在仿真环境验证** [Appendix J]: LIBERO 是仿真 benchmark,未在真实机器人上测试。真实世界的噪声、延迟和物理不确定性可能带来额外挑战。

2. **全双工场景有限** [Appendix J]: 仅支持 turn-taking 和 barge-in,不支持 backchannel (用户回传信号如"uh-huh")、不支持系统主动打断用户等更复杂的对话动态。

3. **Speaking-while-acting 性能下降明显**: 尤其在困难任务和知识密集型 QA 上。虽然更强的 encoder 可缓解,但这表明当前模型容量在多任务并发时捉襟见肘。

4. **Expert 分组固定**: 将 speech+text 绑定、vision+action 绑定是基于预训练模型的便利选择,不一定是最优分组。论文自己也指出 "brain-hand-mouth" 架构可能更优 [Appendix F]。

5. **评估局限**: Context-grounded VQA 仅 12 个问题,defective instruction 仅 160 样本,样本量偏小。

6. **训练数据量极小**: Manipulation 数据仅 ~3.4k samples,SA-MoE 仅训练 500 steps。这是数据效率的正面证据,但也意味着模型泛化能力未经充分验证。

## 点评

**核心贡献的价值**: ELLSA 解决了一个真实且重要的问题 — 现有系统要么是"只说不做的聊天机器人",要么是"只做不说的机器人",没有系统能同时具备这两种能力。SA-MoE 提供了一种简洁的解决方案: 让预训练好的 expert 各司其职,只通过 attention 层的 KV cache 共享来交换跨模态信息,不需要重新训练整个模型。

**SA-MoE vs pi0**: ELLSA 明确承认 SA-MoE 的灵感来自 pi0 [§3.2]。pi0 将 VLM backbone 和 action expert 通过 attention 连接;ELLSA 将这个思想泛化到 interleaved multimodal sequence 和任意多 expert 组合。但基本机制相同: 共享 KV cache。ELLSA 的新意在于证明了这一机制在四模态全双工场景下也能工作。

**1 秒 time block 是双刃剑**: 1 秒的粗粒度简化了全双工建模 (100% turn-taking),但也引入了 ~1 秒的理论最小响应延迟 (854ms 实测)。对话场景中,1 秒的延迟可能影响自然度。Moshi 的 160ms 和 Freeze-Omni 的 160ms time block 虽然 turn-taking 更难学,但延迟更低。

**与 KB 中现有系统的定位**: 在 [[Full-duplexSpokenDialogue]] 的演进线中,ELLSA 开辟了新方向 — 从"边听边说"扩展到"边听边看边说边做"。但在纯 speech 交互维度上,ELLSA 并没有超越专门的全双工系统 (如不支持 backchannel)。ELLSA 的真正贡献是跨域融合,而非在任何单一维度上的突破。

## 可复用的 idea

1. **SA-MoE "shared KV cache + independent FFN"**: 当需要融合多个预训练好的异构模型时,不要试图合并成 dense model (会严重互相干扰),而是让每个模型保持自己的 FFN,只通过 attention 层共享 KV cache。这比 LoRA merge 或 adapter 更优雅,因为跨模态信息流是通过 attention 自然发生的。

2. **Interleaved temporal multimodal sequence**: 将不同模态的输入输出在 time block 内按固定顺序交错,是实现 MIMO streaming 的简洁方案。每个 time block 内的顺序固定 (input modalities → output modalities),避免了复杂的 scheduling。

3. **Speech output 从 text hidden state 派生**: 不在主序列中放 speech output tokens,而是将 text output 的 hidden state 通过 adapter 送入外部 speech synthesizer。这避免了主序列膨胀,同时让 text 作为 speech 的"inner monologue"。

4. **Action barge-in 设计**: 通过训练模型在收到中断命令时输出 "Action Cancelled" 文本来停止动作,将 barge-in 转化为一个文本生成问题,简单有效。

5. **Defective instruction rejection via cross-expert attention**: Speech expert 本身从未见过视觉数据,但通过 SA-MoE 的 shared attention 可以理解视觉场景并拒绝不合理的语音指令。这证明了 SA-MoE 的跨模态理解能力。

## 审阅

(待独立审阅 agent 填充)
