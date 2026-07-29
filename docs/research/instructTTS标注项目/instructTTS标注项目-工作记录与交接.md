---
title: "instruct TTS 标注项目 · 工作记录与交接"
created: 2026-07-29
tags: [research, tts, instruct, annotation, handoff, worklog]
---

# instruct TTS 标注项目 · 工作记录与交接

> **本文是整个项目的入口**。目标:为 instruct TTS 设计一套真实语音的**多维标注方案**(speech→labels),产出带多维标签 + NL 指令的训练语料。
> 阶段:设计 + 方案文档已成型(v0.1),**尚未进入验证/落地**。

## 0. 一句话现状

维度体系、taxonomy、数据格式、pipeline、候选模型、验证方案**已全部落成文档(v0.1)**;下一步是**建 Gold Set + 跑工具验证选型**。核心待拍板:gender/age 许可缺口、SER 选型。

---

## 1. 文档清单(全部在 `docs/research/`)

| 文档 | 内容 | git |
|---|---|---|
| **[[instructTTS标注方案设计]]** | 主方案:维度体系/情感三层/两阶段/工具选型/一二期划分 | ✅ 已提交 83a5f9c |
| **[[instructTTS结构化标签维度参考]]** | 各家维度设置汇总 + 数据集视角 + taxonomy 拍板 | ✅ 已提交 83a5f9c |
| [[instructTTS工具验证与交叉验证方案]] | 验证实验设计(Gold Set + Part A/B/C)| ⏳ 未提交 |
| [[instructTTS成品数据格式规范]] | 单条样本 JSON schema + 字段定义 + gate | ⏳ 未提交 |
| [[instructTTS标注pipeline与算子]] | 字段→算子映射 + 算子清单 + 流程 | ⏳ 未提交 |
| [[instructTTS候选标注模型清单]] | 按维度列候选模型 + 许可 + 核实状态 | ⏳ 未提交 |
| instructTTS_dataflow.html | 流程图(Emilia 蛇形风)| ⏳ 未提交 |
| instructTTS_pipeline_phased.html | 流程图(MOSS 两阶段风,**推荐**)| ⏳ 未提交 |
| instructTTS_pipeline.html | 流程图(泳道风)| ⏳ 未提交 |

**依赖的 vault 现成资产**(设计地基):[[可控TTS综述]]、[[DSP-LLM韵律评估可行性验证方案]](含内部 WordVoice-5A + "DSP+LLM>直接听"假设)、[[TTS基础模型训练数据调研]]、[[TTS评估方法与前沿进展综述]]、概念页 EmotionControlinTTS / NaturalLanguageDescriptionforTTS / ProsodyModeling / StyleTransferinTTS。
**外部参考**:内部生成项目 `~/Downloads/Instruct0706/instruct_data_generator`(44 词中文情绪表 + 14 style,已借鉴)。

---

## 2. 已定稿的关键决策

### 2.1 架构
- **维度关系模型**:prosody(物理基质)→ emotion(状态轴)/ style(情境轴)**同级正交** → persona(组合顶层)。style ≠ 在 emotion 之上。
- **两阶段**:第一阶段 结构化标注(labels)→ 第二阶段 NL 扩写(nl)。
- **一期/二期(均必要,非优先级)**:第一期=可直接标注层(有现成工具);第二期=语义推理/派生层(intent/persona,无现成模型、依赖第一期输出)。
- **情感三层级联**:L1 SER 基础类 → L2 audio-Omni 细类+强度(输入=音频+文本+L1)→ L3 text-LLM 自由描述(输入=文本+L1+L2)。

### 2.2 Taxonomy(第一期)
| 维度 | 定稿 |
|---|---|
| emotion 基础类 / age | **随所选模型类别集** |
| pitch-level / speed | **5 档** |
| pitch-range / energy | 3 档 |
| pause | b0-b4(5 级,WordVoice-5A)|
| pitch-tone | 7 类轮廓 |
| emotion 细类(L2)| ~44 词中文起步池(源自 instruct_data_generator)+ 受控增长 |
| emotion 强度 | 3 档(轻/中/强,相对排序)|
| register | 一级 7 类(断句破碎/自然对话/影视娱乐/播报朗读/教学解说/游戏电竞/其他无效);**二级不启用** |
| accent | **第一期不纳入**(无好开源中文方言分类器)|

- **社交态度类(安抚/鼓励/关切/抱歉)保留为情感标签** → 第一期即消解大量"安慰型"指令;纯言语行为(命令/请求/质问)归第二期意图。

### 2.3 工具
- 物理层:**复用内部 WordVoice-5A**(词级 Duration/Boundary/Energy/Pitch/Tone);
- SER:emotion2vec+ large(9 类)vs SenseVoice —— 待 Part A 定;
- gender/age:**商用友好开源缺口**,待拍板;
- L2/校验:Qwen3-Omni;register/NL:Qwen3-32B;质量:DNSMOS+SNR。

### 2.4 数据策略
- 腿 A 真实挖掘(主)+ 腿 B style-guided mining(补情感长尾)+ 腿 C 合成(后期,仅强度梯度/稀有组合);
- **交叉验证**:emotion(SER↔Omni roll-up)、register(文本↔语音)双标;放量决策:Gold agreement >90% 全自动 / 80-90% human-in-loop / <80% 返工。

---

## 3. 待拍板 / 阻断项

| # | 事项 | 影响 | 归属文档 |
|---|---|---|---|
| 1 | **gender/age 许可缺口**:自训轻量 head / Omni 弱标 / 内部模型 三选一 | 决定 A2 验证路径 | 方案 §9.3 |
| 2 | **SER 选型**:emotion2vec+ vs SenseVoice = 定 L1 taxonomy | linchpin | 验证方案 Part A |
| 3 | Gold Set 谁标、标多少维、多少条(建议 ~300)| 验证前提 | 验证方案 §1 |
| 4 | 情感细类清单**审批流程**(受控增长如何运作)| L2 一致性 | 方案 §3 L2 |
| 5 | NC 模型用于内部标注是否算商用(法务)| 工具可用性 | 方案 §9.3 |
| 6 | 流程图定稿用哪张(或都留)| 展示 | 3 张 HTML |

---

## 4. 下一步(建议顺序)

1. **建 Gold Set(~300 人工)+ Silver Set(~2000)**,覆盖多 register + 非中性情感;
2. **跑 Part A 验证选型**:SER(定 L1)、gender/age(+许可)、register prompt、WordVoice-5A 输出质量;
3. **Part C 消融**:Qwen3-Omni 输入配方(audio / +text / +L1)量化文本增益;
4. **建轻量算子**:分位档聚合器、text_challenges 规则、交叉验证 reconciler;
5. **写 prompt**:emotion L2、L3 描述、global_desc/指令(四阶段);
6. **Pilot 100-500 条**跑通全链路 → 定分位阈值 + 测一致性 → 放量。

---

## 5. 过程备忘(经验/教训)

- **调研前先扫 vault**:本项目一开始只从论文笔记+联网起步,漏了 [[可控TTS综述]] 和 [[DSP-LLM韵律评估可行性验证方案]](内含内部 WordVoice-5A 工具),被指出"调研起点不完善"。已存入 memory [[feedback_vault_sweep_before_research]]。
- **WordVoice-5A 是关键复用**:词级韵律标注已有,不要重造 parselmouth/librosa 轮子;也解决了词级时间戳依赖。
- **"结构化→NL扩写"路线有内外部双重印证**:内部 DSP-LLM 假设 + 外部 BatonVoice/SpeechCraft/CapSpeech,不是拍脑袋。
- **Playwright 当前被残留实例锁死**,联网核实用 WebFetch 兜底(WebSearch 在本端点禁用);候选模型清单里 📚 项需最终核实。

---

## 6. git 状态

- 已提交:`83a5f9c` = 方案设计 + 维度参考;
- 未提交:验证方案 / 数据格式规范 / pipeline与算子 / 候选模型清单 / 3 张 HTML / 本交接文档。
- memory 已存:[[feedback_vault_sweep_before_research]]。
