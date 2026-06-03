---
type: paper
tier: deep
title: "RWKVTTS: Yet another TTS based on RWKV-7"
arxiv_id: "2504.03289"
source: "Sources/RWKVTTS.pdf"
authors: [Lin Yueyu, Liu Xiao]
year: 2025
venue: "arXiv preprint"
tags: [TTS, RWKV, RNN, LLM-based-TTS, efficiency, zero-shot, CosyVoice]
concepts: ["[[LLM-based TTS]]", "[[Speech Tokenizer]]", "[[Codec Language Model]]"]
models: ["[[模型库/CosyVoice 2|CosyVoice 2]]"]
tasks: ["[[Zero-shot Speech Synthesis]]"]
datasets: []
kb_context_sources: 6
status: draft
created: 2026-06-03
updated: 2026-06-03
---

## KB 背景

> [!info] KB 背景 (基于 5 个已确认实体页 + 1 个待确认实体页: [[LLM-based TTS]]✓, [[模型库/CosyVoice 2|CosyVoice 2]]✓, [[Speech Tokenizer]]✓, [[Zero-shot Speech Synthesis]]✓, [[Residual Vector Quantization]]✓, [[Codec Language Model]][待确认])
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: RWKVTTS 处于 LLM-based TTS 的"效率优化"分支,与 [[论文笔记/MamTra|MamTra]] (Mamba-Transformer 混合架构替换 CosyVoice 2 的 Transformer 层) 属于同一方向但不同技术路线。当前 LLM-based TTS 主流架构 (VALL-E, CosyVoice 系列, Llasa, Fish-Speech) 均使用 Transformer backbone,RWKVTTS 是首个将 RWKV-7 (RNN-based 线性复杂度模型) 作为 LLM backbone 完整替换到 CosyVoice 2.0 框架中的工作。另一个相关工作 Lina-Speech (Lemerle et al., 2024) 已验证 gated linear attention (RWKV-related) 在 TTS 中的可行性,但未直接使用 RWKV-7 且未集成进主流 TTS 框架。
>
> **已有认知**: KB 中 [[LLM-based TTS]] 已记录 LLM-based TTS 的演进路线: 传统 CNN/RNN → Flow-based → LLM-based → Hybrid; KB 中 [[模型库/CosyVoice 2|CosyVoice 2]] 详细记录了 CosyVoice 2 的架构 (FSQ-SenseVoice tokenizer + text-based LLM 初始化 + 双向流式), 其 baseline 性能 (CER 1.45% test-zh, WER 2.57% test-en on SEED-TTS-Eval)。[[Speech Tokenizer]] 中覆盖了 VQ-VAE 到 continuous tokenizer 的完整演进。RWKVTTS 使用 CosyVoice 2.0 的 VQ-VAE tokenizer,属于 acoustic codec token 路线。
>
> **创新判断**: RWKVTTS 的核心卖点是效率: 用 RWKV-7 (O(L) 复杂度, 无 KV cache) 替换 Transformer (O(L^2)),声称在保持质量的同时降低计算成本。与同属 CosyVoice backbone 替换路线的 MamTra 相比,RWKVTTS 是完整替换 (不是混合比例),但论文的实验部分显著薄弱: 仅与 FireRedTTS-1S 做对比,无标准 benchmark (如 SEED-TTS-Eval),评估指标 (Production Quality / Content Enjoyment 等) 非领域标准 (WER/CER/Speaker Similarity/MOS),且缺少消融实验和效率数据。
>
> 检索命中: [[LLM-based TTS]](confirmed), [[模型库/CosyVoice 2|CosyVoice 2]](confirmed), [[Speech Tokenizer]](confirmed), [[Zero-shot Speech Synthesis]](confirmed), [[Residual Vector Quantization]](confirmed) | 过滤: [[Codec Language Model]](pending-review) | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 将 CosyVoice 2.0 中的 Transformer LLM backbone 完整替换为 RWKV-7 (RNN-based 线性复杂度模型),声称在 TTS 质量上与 Ground Truth 和 Transformer baseline 相当
> - **路线**: 参考音频 → VQ-VAE → 音频 token; 文本 → text token; [audio embedding + text embedding] → RWKV-7 LLM → audio head → 音频 token → 解码
> - **指标**: Production Quality 7.73 (vs GT 7.80, FireRedTTS-1S 7.82); Content Enjoyment 6.11 (vs GT 6.20, FireRedTTS-1S 6.06) [Fig 1]; 无 WER/CER/MOS/Speaker Similarity 标准指标
> - **可借鉴**: RWKV-7 直接 drop-in 替换 Transformer LLM 的集成方式 (保持 data layout / embedding / audio head 不变,仅换中间的 LM); PISSA 微调策略复用 LM 参数同时做语音和文本任务的思路 (§5.3,未实现)
> - **局限**: 论文完成度低 -- 无标准 TTS 评估指标 (WER/CER/Speaker Similarity/MOS),仅用非标准主观评分; 仅与 FireRedTTS-1S 一个 baseline 对比; 无效率数据 (推理速度/显存/RTF); 无消融实验; 未开源训练细节; 多处提到的增强 (§3.5) 均为 future work

## 核心问题

1. **RWKV-7 能否作为 LLM-based TTS 中 Transformer 的 drop-in replacement?** 当前 TTS 的 LLM backbone 几乎全部是 Transformer,RWKV-7 的线性复杂度 (O(L) vs O(L^2)) 和无 KV cache 特性理论上适合长序列语音生成,但实际效果如何?

2. **替换后的 speech quality 能否保持?** Transformer 的全局注意力被认为对语音的长程韵律建模至关重要,RWKV 的 recurrent state 机制能否捕捉同等的长程依赖?

3. **效率提升有多大?** 论文的核心卖点是效率,但实际推理速度/显存/RTF 数据缺失,无法量化评估。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

RWKVTTS 采用与主流 LLM-based TTS 相同的两阶段架构 [§3.1]:

1. **VQ-VAE 阶段**: 音频 tokenization -- 将参考音频和目标语音编码为离散 audio tokens
2. **LLM 阶段**: 音频 token 生成 -- 用 RWKV-7 替代 Transformer 做 audio token 的自回归预测

Pipeline 流程 [§3.1, Fig 2]:
```
参考音频 → VQ-VAE → audio tokens → audio embedder → ┐
                                                       ├→ 拼接 → RWKV-7 → hidden states → audio head → audio tokens
文本 → text tokens → text embedder → ─────────────────┘
```

论文强调 RWKV-7 是一个"expressive dynamic state based"模型,既保留 RNN 的顺序处理效率,又具备类 Transformer 的长上下文记忆能力 [论文原文, §1]。

### 关键设计选择

**1. CosyVoice 2.0 特定的 data layout [§3.2, Fig 3]**

在 CosyVoice 2.0 框架中,RWKV-7 的 forward pass 处理以下输入序列 [论文原文]:
- SOS embedding
- Text embedding
- Task ID embedding (任务指令)
- Audio embedding (参考音频 tokens)
- Last audio embedding

特殊 token `<|endofprompt|>` 标记 prompt text 作为指令的边界 [§3.2]。

**为什么保持 CosyVoice 2.0 的 data layout 不变?** 论文未给出显式解释。[agent 解读] 这是 drop-in replacement 的核心策略: 保持 tokenizer、embedding 层、audio head 等组件不变,仅替换中间的 LM 部分,最大化复用已有框架的工程投入和数据处理 pipeline。

**2. 完整替换而非混合 [§1, §3]**

与 MamTra 的混合 Mamba-Transformer 策略不同,RWKVTTS 完全替换 LLM 部分为 RWKV-7。[agent 解读] 这意味着放弃了 Transformer 的全局 attention 能力,完全依赖 RWKV-7 的 recurrent state 来建模长程依赖。论文声称 RWKV-7 "具备类 Transformer 的长上下文记忆" [§1],但未提供与 attention 机制的定量对比。

**3. PISSA 微调方案 (§5.3, Future Work)**

论文提出一个有趣但**尚未实现**的增强方案: 保持 RWKV-7 的 LM 参数不变,仅在 MLP 层使用 Parameter-Efficient Sparse Adaptation (PISSA) 微调,加上额外的 text embedding 层和 audio head [§5.3]。这样大部分参数在语音合成和文本对话之间共享,实现高效的端侧部署。[agent 解读] 这一思路类似于 adapter/LoRA 系列的参数高效微调,但将文本 LM → 语音 LM 的 adaptation 定位为稀疏适配而非全量微调,降低多任务部署成本。

### 训练策略

论文对训练细节的描述极为稀少 [§3.3-3.4]:

- **数据准备** [§3.3]: 参考音频来自公开 TTS 数据集; 文本来自 Wikipedia 等多语言大规模语料 (中英文); 通过 VQ-VAE 将文本+参考音频处理为 audio token pairs
- **训练** [§3.4]: 使用分布式多 GPU 训练; 其余细节 (学习率、batch size、训练步数、模型大小、数据量) 均未给出
- **推理** [§3.4]: 通过零样本合成评估 (给定 prompt text + reference audio 生成语音)

[agent 解读] 训练细节的缺失是本文的重大缺陷。无法判断 RWKV-7 的 TTS 性能是源于架构优势还是训练数据/超参的偶然选择,也无法复现实验。

## 实验

| 指标 | RWKVTTS | FireRedTTS-1S | Ground Truth | 数据集 | 出处 |
| --- | --- | --- | --- | --- | --- |
| Production Quality (0-10) | 7.73 | 7.82 | 7.80 | 未明确 | [§4.2, Fig 1] |
| Production Complexity (0-10) | 1.53 | 1.51 | 1.53 | 未明确 | [§4.2, Fig 1] |
| Content Enjoyment (0-10) | 6.11 | 6.06 | 6.20 | 未明确 | [§4.2, Fig 1] |
| Content Usefulness (0-10) | 6.46 | 6.61 | 6.52 | 未明确 | [§4.2, Fig 1] |

**评估设置** [§4.1]:
- 评估数据: 多语言、多说话风格的 prompt text + reference audio (具体数据集未明确)
- 评估方式: 人类标注者对生成语音进行主观评分 (0-10)
- 仅对比 Ground Truth 和 FireRedTTS-1S (Guo et al., 2025) 一个 baseline

**关键观察**:

1. **Production Quality**: RWKVTTS (7.73) 与 GT (7.80) 和 FireRedTTS-1S (7.82) 差距微小 (< 0.1 分),论文声称可比 [§4.2]
2. **Content Enjoyment**: RWKVTTS (6.11) 略优于 FireRedTTS-1S (6.06),论文认为这反映了 RWKV-7 更好的时序建模带来的表现力优势 [论文原文, §4.3]
3. **Production Complexity**: 所有模型得分极低 (1.51-1.53),论文承认当前 TTS 系统在复杂韵律建模上仍有限制 [§4.3]

## 局限性

1. **评估严重不足**: 未使用 TTS 领域标准指标 (WER/CER/Speaker Similarity/MOS/DNSMOS),仅用 Production Quality 等非标准主观评分; 评估数据集未明确; 人类标注者数量/协议未交代 [agent 解读]

2. **Baseline 单一**: 仅与 FireRedTTS-1S 对比,缺少与 CosyVoice 2.0 (原始 Transformer backbone) 的直接对比,无法回答"替换 RWKV-7 相比保留 Transformer 到底如何"这一核心问题 [agent 解读]

3. **效率数据完全缺失**: 论文的核心卖点是 RWKV-7 的效率优势,但无 RTF、推理延迟、显存占用、吞吐量等任何量化数据 [agent 解读]

4. **训练细节缺失**: 模型大小、数据量、训练步数、学习率等关键超参均未报告 [§3.3-3.4]

5. **Production Complexity 低分**: 所有模型的 Production Complexity 得分仅约 1.5/10,论文承认在复杂韵律建模上存在挑战 [§4.3],但未分析 RWKV-7 是否比 Transformer 在这方面有结构性劣势

6. **泛化性未验证**: 仅展示了 CosyVoice 2.0 一个框架的集成,论文标题提到的 Fish-Speech 和 ChatTTS 集成均未有实验数据 [agent 解读]

7. **PISSA 方案和流式生成均为 future work**: 论文 §3.5 和 §5.3 提出的增强方案 (控制 token、方言支持、流式生成、PISSA 微调) 均未实现 [§3.5, §5.3]

## 点评

RWKVTTS 在方向上有价值 -- 探索非 Transformer 架构作为 LLM-based TTS 的 backbone 是重要的研究问题,RWKV-7 的线性复杂度理论上很有吸引力。但论文的执行质量显著低于领域标准:

**方向正确但论证薄弱**: 核心贡献是"证明 RWKV-7 可以替换 Transformer 做 TTS",但缺少与原始 Transformer backbone (CosyVoice 2.0 原版) 的直接对比,相当于缺少了最关键的消融实验。不与被替换的对象对比,替换的意义难以评估。

**与 MamTra 的对比**: 同为 CosyVoice backbone 替换, MamTra 提供了详细的 (1) 替换策略消融 (哪些层换/换多少), (2) 与原始 CosyVoice 2 的直接对比, (3) 标准 SEED-TTS-Eval 评估, (4) 显存/速度效率数据。RWKVTTS 在实验严谨性上远不及 MamTra。

**效率承诺未兑现**: 论文反复强调 RWKV-7 的"效率优势"、"资源受限环境适用性",但无任何效率数据支撑。这是论文最大的缺陷。

**值得关注的信号**: 尽管论文本身质量有限,RWKV 社区在 TTS 方向的探索 (ARWKV, Lina-Speech, RWKVTTS) 构成了一条"RNN 回归 TTS"的研究线,其核心价值主张 (线性推理 + 无 KV cache + 高效端侧部署) 在实际应用场景中有真实需求。

## 可复用的 idea

1. **Drop-in LM replacement 策略**: 保持 TTS 框架的 tokenizer / embedding / audio head 不变,仅替换中间 LM 部分。这是一种低风险的架构探索方式,可快速验证不同 LM backbone 的效果 [§3.1-3.2]

2. **PISSA 参数共享部署 (未实现)**: 文本对话和语音合成共享大部分 LM 参数,仅通过稀疏适配 (MLP 层 + 额外 embedding/head) 切换任务。适用于端侧多任务部署场景 [§5.3]

3. **RNN-based LM 用于 TTS 的可行性信号**: 尽管本文证据不充分,但 RWKV-7 + Lina-Speech + MamTra 的混合证据表明线性复杂度模型在 TTS 中有潜力,值得在更严格的实验条件下进一步验证

---

检索命中: [[LLM-based TTS]], [[模型库/CosyVoice 2|CosyVoice 2]], [[Speech Tokenizer]], [[Zero-shot Speech Synthesis]], [[Residual Vector Quantization]] | 过滤: [[Codec Language Model]](pending-review) | 未命中但可能相关: 无
