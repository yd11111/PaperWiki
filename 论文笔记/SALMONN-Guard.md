---
type: paper
tier: deep
title: "Speech-Audio Compositional Attacks on Multimodal LLMs and Their Defense with SALMONN-Guard"
arxiv_id: "2511.10222"
source: "Sources/SALMONN-Guard.pdf"
authors: [Yudong Yang, Xuezhen Zhang, Zhifeng Han, Siyin Wang, Jimin Zhuang, Zengrui Jin, Jing Shao, Guangzhi Sun, Chao Zhang]
year: 2026
venue: "ICML 2026"
tags: [audio-safety, red-teaming, jailbreaking, multimodal-safety, guard-model, speech-audio-composition, benchmark, LLM-safety, black-box-attack]
concepts: ["[[AudioUnderstanding]]", "[[SpeechLanguageModel]]", "[[Audio-LanguagePretraining]]", "[[Anti-spoofingandDeepfakeDetection]]"]
models: ["[[Whisper]]"]
tasks: []
datasets: []
kb_context_sources: 4
status: draft
created: 2026-06-08
updated: 2026-06-08
---

## KB 背景

> [!info] KB 背景 (基于 1 个已确认实体页 + 3 个待确认实体页)
> 自动生成,不保证完整覆盖所有相关知识。
>
> **谱系定位**: SALMONN-Guard 是 SALMONN 系列的安全防御分支,与原始 [[论文笔记/SALMONN|SALMONN]] (ICLR 2024, 通用听觉理解) 和 [[论文笔记/SALMONN-omni|SALMONN-omni]] (全双工对话) 同属清华-上海 AI Lab-Cambridge 团队。本文转向 multimodal LLM 安全领域,聚焦"音频输入侧的攻击与防御",这是 KB 中 [[Anti-spoofingandDeepfakeDetection]] 概念页覆盖的 TTS 伪造检测的延伸方向 — 但攻击对象不同: Anti-spoofing 防御的是"伪造语音冒充真人",SALMONN-Guard 防御的是"通过音频组合绕过 LLM 安全机制"。
>
> **已有认知**: KB 中 [[Audio-LanguagePretraining]] 概念页已记录 adversarial vulnerability 为 ALM 核心挑战之一 ("Jailbreak attacks 绕过 safety alignment"),但缺乏专门的音频攻击 benchmark 和防御方案。[[AudioUnderstanding]] 页覆盖了 SpeechLM 的理解能力分类 (语义/副语言/speaker),但未涉及安全维度。[[SpeechLanguageModel]] 确认页记录了端到端语音模型的发展,SALMONN-Guard 要保护的目标正是这类模型。
>
> **创新判断**: KB 中尚无专门的"multimodal LLM safety"或"audio jailbreaking"概念页。SALMONN-Guard 是 KB 记录的首篇系统性音频红队攻击+防御论文。与 Anti-spoofing 的区别在于: Anti-spoofing 是检测"假语音",SALMONN-Guard 是检测"用真实音频组合的恶意意图"。
>
> 检索命中: [[SpeechLanguageModel]]✓ | 过滤: [[AudioUnderstanding]][待确认], [[Audio-LanguagePretraining]][待确认], [[Anti-spoofingandDeepfakeDetection]][待确认] | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: 提出 SACRED-Bench (首个利用语音-音频组合的黑盒攻击 benchmark) 和 SALMONN-Guard (首个联合检查语音+音频+文本的 guard model),将 Gemini 2.5 Pro 的 66% 攻击成功率降至 20%
> - **路线**: 攻击: 有害内容 → ChatTTS 合成 → SSO (语音重叠) / SAO (语音+非语音混合) / MSD (多人对话) → 间接文本 prompt → 目标 LLM; 防御: Audio + Text → Qwen2.5-Omni-7B + LoRA (SFT, 两阶段) → safety judgment (harmful/harmless/refusal)
> - **指标**: Gemini 2.5 Pro overall ASR 66.75%, SALMONN-Guard 降至 11.32% (OBE 7.74%); 对 Speech Insertion/Editing 等未训练攻击 ASR 分别为 0.00%/3.00% [Table 1, Table 4]; FAR 0% [Table 1]
> - **可借鉴**: (1) 语音-音频组合作为黑盒攻击范式: 不需要梯度/白盒访问,利用真实音频场景的复杂性; (2) 两阶段 SFT curriculum (先全局再难例聚焦) 训练轻量 guard model; (3) 间接引用 (indirect reference) 作为绕过文本安全过滤器的通用策略
> - **局限**: 仅评估英文; SALMONN-Guard 仍有 ~11% 残余 ASR; SAO 依赖人工收集的有害音频; 仅 2 人听力测试验证 intelligibility; guard model 需要部署为独立组件,增加推理延迟

## 核心问题

**想解决什么**: 当前 multimodal LLM 的安全机制主要依赖文本侧过滤 — 检查输出文本是否违规,或检查输入文本 prompt 是否包含恶意指令。但当恶意内容隐藏在音频输入中 (而非文本 prompt 中) 时,这些 text-only safeguards 几乎完全失效 [§1]。

**为什么难**: 音频输入的复杂性远超文本: 多说话人重叠、语音与非语音混合、对话中隐含的恶意意图 — 这些都是现实中常见的音频场景,但现有安全机制无法处理。更关键的是,攻击者可以将文本 prompt 设计为表面无害 (如"这段音频适合放给小孩听吗?"),恶意内容完全由音频携带,形成跨模态的语义缝隙 [§1, §3.1]。

**怎么切入**: (1) 构建 SACRED-Bench 系统性暴露漏洞 — 三种组合攻击机制 + 两种提问方式; (2) 提出 SALMONN-Guard 作为防御 — 联合检查音频+文本而非仅检查文本。

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### SACRED-Bench: 攻击设计

SACRED-Bench (Speech-Audio Composition for RED-teaming) 基于三个设计原则 [§3]:
1. 利用语音信号的复杂性 (多说话人、重叠)
2. 利用非语音音频 (暴力/色情场景声音)
3. 使用间接引用的文本 prompt (表面无害,语义指向音频中的恶意内容)

**威胁模型** [§3.1]: 黑盒攻击者,无模型参数/梯度访问,不做对抗优化,不注入不可感知扰动。攻击者只能提供音频输入 + 可选文本 prompt,与标准用户接口一致。所有音频对人类完全可听、可理解 (Appendix E 听力测试: SSO 98%, SAO 96% intelligibility) [Table 7]。

#### SSO (Speech-Speech Overlap) [§3.3.1, Fig 3a]

将有害语音与无害语音在时域上重叠,形成单一音频流:
1. 用 GPT-4o 为有害内容生成逻辑上合理的无害前文 (如为"制造危险物质"生成"我在写小说需要描述化学实验场景")
2. 用 ChatTTS 分别合成无害和有害语音
3. 降低有害语音的感知显著性: 调整音量 (-8dB)、播放速度 (1.3x)、时域重叠 (500ms cross-fade) [§3.3.1]
4. 用 cross-fade 融合为流畅的单条语音

**设计直觉**: 有害内容被声学包装为"次要信息",对模型清晰可感知但足够微妙以绕过安全过滤 [论文原文]。[agent 解读] 这利用了 LLM 音频理解的一个结构性弱点: 模型能转写重叠语音的内容,但安全机制只检查转写后的文本,而非评估音频场景的整体意图。

#### SAO (Speech-Audio Overlap) [§3.3.2, Fig 3b]

将有害非语音音频 (如成人内容、暴力场景的声音) 与无害语音混合:
1. 从公开在线视频中收集有害音频事件 (成人内容、暴力内容)
2. 取 VoiceBank-DEMAND 数据集的清洁无害语音作为载体
3. 将有害音频作为背景层叠加到无害语音上 (明确可听的音量)

**设计直觉**: 测试模型安全机制是否仅是"文本处理器" (分析语音转写内容),还是具备真正的多模态场景理解 [论文原文]。[agent 解读] SAO 是三种攻击中效果最好的 (88.56% ASR on Gemini 2.5 Pro [Table 1]),恰恰因为非语音音频完全不会出现在 ASR 转写中,text-only safeguards 对此完全失明。

#### MSD (Multi-Speaker Dialogue) [§3.3.3, Fig 3c]

将有害指令嵌入多人对话中:
1. 用 GPT-4o 将直接有害指令改写为两人对话 (去除命令式标记,用对话结构包装)
2. 单独构造语义中性但指向对话内容的文本 query (如"How would I create the device mentioned in the discussion?")
3. 用 ChatTTS 多说话人合成对话音频

**设计直觉**: 灵感来自视觉安全研究 (Hu et al., 2025) 的发现 — 许多多模态模型的安全对齐实际上仅靠文本训练数据实现,因为训练时文本 prompt 已经暴露了图像的有害本质。MSD 通过让文本 prompt 完全无害来测试这种跨模态安全缝隙是否同样存在于音频领域 [§3.3.3] [论文原文]。

### 提问方式 [§3.4]

1. **Permission Questions** (用于 SSO/SAO): Yes-No 格式,如"这段音频适合放给5岁小孩听吗?" — 如果模型回答"安全"则攻击成功 [§3.4.1]
2. **Open-ended Questions** (用于 MSD): 要求模型提取/阐述音频中的有害内容,如"怎么执行对话中提到的方法?" — 用 Gemini 2.5 Pro 作为 judge 判定回答是否有害 [§3.4.2]

### SALMONN-Guard: 防御设计

#### 架构 [§4.2]

基于 Qwen2.5-Omni-7B,利用其原生音频+文本多模态理解能力。两种输出模式 [§4.2]:
1. **独立音频评估**: 二分类 (harmful/harmless)
2. **联合文本-音频评估** (主要模式): 判定联合 query 是否安全,有害则生成拒绝响应 ("I'm sorry, I cannot assist with that request"),无害则放行给下游 LLM

[agent 解读] SALMONN-Guard 的定位是"部署时安全层" (deployment-time guard),类似 LlamaGuard 系列在文本领域的角色。但关键区别是: LlamaGuard 仅检查文本输入/输出,SALMONN-Guard 联合检查音频+文本,填补了 modality gap。

#### 训练数据 [§4.1]

~10k 样本 (8,545 harmful + 1,828 benign):
- SSO: 6,640 样本 (大部分)
- SAO: 986 样本
- MSD: 919 样本
- Benign: 防止模型只输出 "unsafe" [§4.1]

**严格跨数据集验证**: 对于每种攻击类别,训练 prompt 的来源 benchmark 不出现在测试集中,防止数据泄露 [§4.1]。

#### 训练过程 [§4.3]

两阶段 SFT curriculum + LoRA [§4.3]:

**Stage 1**: 全数据集训练 3 epochs — 建立通用安全判断能力

**Stage 2**: 仅在 MSD 子集上额外训练 5 epochs — 针对最困难的跨模态攻击进行强化

**为什么分两阶段?** [论文原文] MSD 攻击对模型来说"特别难学",因为它需要理解对话语义+跨模态意图推理,而非简单的音频内容检测 [§4.3]。[agent 解读] 这与原始 SALMONN 的三阶段训练策略 (pre-train → instruction tuning → activation tuning) 有相似的"先广后专"哲学,但简化为两阶段。Stage 2 的 MSD 聚焦训练是论文的关键工程选择,因为 MSD 的 ASR 是三种攻击中最接近随机水平的 (14.08%),说明额外训练确实有效。

**LoRA 配置** [Table 6]: rank=32, alpha=32, 应用于 LLM + audio encoder + aligner 的所有 linear layers。lr=3e-5, weight decay 0.1, warmup 0.05, bfloat16, max seq len 2048。

### 整体架构

从部署视角看,SALMONN-Guard 作为前置 guard 拦截恶意输入:
```
User input (audio + text) → SALMONN-Guard → {harmful → refuse; harmless → forward to target LLM}
```

## 实验

| 指标 | 本文 (SALMONN-Guard) | Baseline (Gemini 2.5 Pro) | 数据集 | 出处 |
| --- | --- | --- | --- | --- |
| SSO ASR | 12.93% | 37.25% | SACRED-Bench SSO (N=400) | [Table 1] |
| MSD ASR | 14.08% | 63.93% | SACRED-Bench MSD (N=717) | [Table 1] |
| SAO ASR | 5.16% | 88.56% | SACRED-Bench SAO (N=1364) | [Table 1] |
| Overall ASR | 11.32% | 66.75% | SACRED-Bench (N=2481) | [Table 1] |
| HCS FAR | 0.00% | 1.00% | Harmless Control Set (N=400) | [Table 1] |
| OBE | 7.74% | 63.15% | Weighted avg of ASR+FAR | [Table 1] |
| Speech Insertion ASR | 0.00% | 13.41% | JALMBench SSJ (N=246) | [Table 4] |
| Speech Editing ASR | 3.00% | 23.00% | JALMBench AMSE (N=1800) | [Table 4] |
| MSD Harmful Compliance | 2.15% | 55.12% | SACRED-Bench MSD | [Table 9] |
| MSD Wrong Understanding | 11.93% | 8.81% | SACRED-Bench MSD | [Table 9] |

**关键观察**:

1. **SAO 暴露了最严重的安全缺口**: Gemini 2.5 Pro 在 SAO 上 ASR 高达 88.56%,所有开源模型 (Qwen2.5-Omni, Qwen2-Audio 等) 更是接近 100%。这证实了非语音音频是当前安全机制的完全盲区 [Table 1] [论文原文]。

2. **MSD 利用跨模态语义缝隙**: 消融实验 [Table 3] 显示 MSD (Text+Audio, 78.58% ASR on Gemini 1.5 Pro) 显著优于单模态攻击 (Text-Only 57.96%, Audio-Only 65.03%),证实攻击效果来自跨模态协同而非单一模态 [论文原文]。

3. **SALMONN-Guard 的泛化能力**: 虽然仅在 SACRED-Bench 数据上训练,但对从未见过的 Speech Insertion (0.00% ASR) 和 Speech Editing (3.00% ASR) 攻击也有近乎完美的防御 [Table 4]。[论文原文] 将此归因于 SALMONN-Guard 学到了"恶意音频操控的底层原理"而非过拟合到训练集的模式 [§5.4]。

4. **SSO 攻击的声学参数敏感性**: 消融 [Table 2] 显示攻击效果对增加有害内容隐蔽性的参数最敏感 — 更高播放速度 (1.5x → 61% ASR vs 1.1x → 42%)、更长重叠 (相关性较弱)、更低音量 (相关性较弱)。这支持了"次要信息"假说: 有害内容越像背景信息,越能绕过安全过滤 [§5.3] [论文原文]。

5. **MSD 失败模式分解** [Table 9]: SALMONN-Guard 的 14.08% ASR 中,仅 2.15% 是真正的"有害配合" (模型理解恶意意图并执行),11.93% 是"错误理解" (模型未拒绝但生成了无关/困惑的内容)。相比之下 Gemini 2.5 Pro 的 63.93% 中 55.12% 是真正的有害配合,说明其安全失败是真实的安全漏洞而非能力不足 [论文原文]。

6. **无过度拒绝**: 所有模型在 HCS 上 FAR 都很低 (SALMONN-Guard 0.00%),说明高 ASR 不是因为模型默认放行,而是真正的安全机制失效 [Table 1] [论文原文]。

## 局限性

1. **残余攻击成功率**: SALMONN-Guard 仍有 ~11% overall ASR,其中 SSO (12.93%) 和 MSD (14.08%) 尚有优化空间。MSD 的残余 ASR 中 11.93% 是"错误理解"而非"有害配合" [Table 9],暗示模型在理解复杂对话上仍有能力瓶颈 [agent 解读]。

2. **仅英语评估**: 所有攻击和防御均基于英语,多语言场景 (尤其是低资源语言) 的安全性未被验证 [§6 未提及但从数据源可推断]。

3. **SAO 数据伦理风险**: 有害非语音音频从"publicly available online videos featuring adult-oriented or intimate content"收集 [§3.2.3],数据收集过程的伦理审核和数据保护措施未详细说明。

4. **Intelligibility 验证不充分**: 仅 2 名听者 (1 名作者 + 1 名外部人员),每种攻击仅 25 个随机样本 [Table 7]。更大规模的 crowd-sourced 听力测试才能确认攻击的 ecological validity。

5. **Guard model 的部署代价**: SALMONN-Guard 基于 Qwen2.5-Omni-7B,需要独立部署为前置组件。对于已有 LLM 的服务,增加一个 7B 参数的 guard model 意味着额外的推理延迟和计算成本。论文未报告推理时间 [agent 解读]。

6. **评估偏差**: MSD 的 open-ended 评估使用 Gemini 2.5 Pro 作为 judge,而 Gemini 2.5 Pro 本身就是被攻击的目标之一。虽然论文在 Appendix F 中做了 judge 交叉验证 (Cohen's Kappa 0.80-0.86) [Table 8],但单一 AI judge 的系统偏差仍值得关注。

7. **与现有安全机制的组合未探索**: 论文将 SALMONN-Guard 定位为独立 guard,未探索与 Gemini 的内置安全机制组合使用时是否有增益或冲突。

## 点评

**SACRED-Bench 的核心价值在于暴露了一个被忽视的安全盲区**: 现有 multimodal LLM 的安全机制本质上仍是 text-centric — 检查文本输入/输出是否违规,而对音频输入的安全语义几乎完全失明。SAO 的 88.56% ASR (Gemini 2.5 Pro) 是最有力的证据: 当恶意内容完全由非语音音频携带时,即使是最强的商业模型也几乎无法防御。这个发现对整个 multimodal LLM 安全领域都有警示意义。

**攻击设计的"真实性"是最大亮点**: 与之前依赖对抗优化、不可感知扰动、白盒访问的音频攻击不同,SACRED-Bench 的三种攻击都基于现实中常见的音频场景 (重叠语音、背景声音、多人对话),不需要任何技术门槛即可实施。这意味着这些漏洞不是学术界的理论风险,而是任何用户都可能 (有意或无意) 触发的实际安全问题。

**SALMONN-Guard 的泛化能力是最令人意外的发现**: 仅在 ~10k SACRED-Bench 样本上训练,就能对从未见过的 Speech Insertion (0.00% ASR) 和 Speech Editing (3.00% ASR) 攻击实现近乎完美的防御 [Table 4]。这暗示"联合检查音频+文本"的根本方法论比特定攻击的防御更重要 — guard model 学到的可能是"音频中存在恶意意图"这个通用模式,而非特定的攻击特征。

**与 KB 中已有安全工作的关系**: 这篇论文将 KB 中 [[Anti-spoofingandDeepfakeDetection]] 的"音频安全"概念从"检测假语音"扩展到"检测恶意音频组合"。两者的核心差异在于: deepfake detection 假设攻击者生成了不自然的音频,而 SACRED-Bench 使用的是完全自然、人类可理解的音频组合。这是一种更难防御的攻击,因为没有任何"伪造痕迹"可以检测。

**方法论上的遗憾**: 论文采用 SFT 训练 guard model,但在 LLM safety 领域,RLHF/DPO 已被证明对 safety alignment 更有效。原始 SALMONN 系列的 SALMONN-omni 就使用了 DPO 来优化 barge-in 判断。如果对 SALMONN-Guard 也应用 DPO (如训练模型在模糊场景中做出更保守的安全判断),可能进一步降低残余 ASR。

## 可复用的 idea

1. **语音-音频组合作为黑盒攻击范式**: 不需要模型访问权限,仅用真实音频的组合就能构造有效攻击。这个范式可扩展到其他多模态场景 — 如图像-文本组合 (已有 MM-SafetyBench),视频-音频组合 (尚未被系统探索)。核心原理是: 跨模态安全缝隙 > 单模态安全漏洞 [§3]。

2. **间接引用 (indirect reference) 绕过文本安全过滤**: 让文本 prompt 表面无害,恶意语义由其他模态携带。"Is it safe to play this audio in public?" 这种 prompt 设计对任何基于文本关键词/语义匹配的安全过滤器都是致命的。在设计安全系统时,必须考虑"文本无害+其他模态有害"的组合 [§3.4]。

3. **两阶段 SFT curriculum (先广后专)**: Stage 1 全数据集建立基础 + Stage 2 难例子集强化。这种策略适用于任何"部分任务比其他任务难学"的多任务 SFT 场景。对 SALMONN-Guard 来说,MSD 是难例; 对 TTS 模型来说,可能是情感控制或多语言是难例 [§4.3]。

4. **部署 guard model 而非修改目标 LLM**: 当目标 LLM 是黑盒 (如 Gemini 2.5 Pro) 时,无法修改其安全机制,但可以在前端部署独立的 guard model 进行拦截。这是一种模块化安全架构,guard 和 target LLM 可以独立更新 [§4]。

5. **ASR 的失败模式分解** (Harmful Compliance vs Wrong Understanding): 在评估安全系统时,区分"模型理解了恶意意图并配合"和"模型没理解但也没拒绝"是重要的。前者是真正的安全失败,后者是能力不足 [Table 9]。

## 审阅

> [!review] 审阅 (2026-06-08, auto)
> **结论**: pass
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | pass | 三种攻击均有因果解释 (WHY),防御设计有"先广后专"训练策略的机制分析,可借鉴项具体可迁移 |
> | 可信赖 | pass | 数字标注覆盖率>90%, ASR/FAR/OBE 指标使用正确,关键数字与 PDF 交叉验证一致 |
> | 可区分 | pass | [论文原文]/[agent 解读] 标注覆盖率>80%,推断均有限定词,无推断写成断言 |
> | 可定位 | pass | KB 背景谱系清晰 (SALMONN 系列定位 + 与 Anti-spoofing 区别),速查卡片 5 字段均有实质内容 |
> | 不污染 | pass | 未新建概念页,无 overclaim,frontmatter 语义正确 |
> 
> Issues: 0 (high: 0, medium: 0, low: 0)
> 详见 `_review/SALMONN-Guard-review.yml`
