---
title: "instruct TTS 工具验证与交叉验证方案 v0.1"
created: 2026-07-28
tags: [research, tts, instruct, annotation, validation, experiment]
---

# instruct TTS 工具验证与交叉验证方案 v0.1

> 配套 [[instructTTS标注方案设计]] §9(工具选型)/§10(执行计划)。把"每维工具选择 / 交叉验证逻辑 / Omni+文本表达扩充"三件事落成可执行的验证实验。
> **方法论骨架借鉴 [[DSP-LLM韵律评估可行性验证方案]]**(小样本 + 多路径对比 + Agreement 指标)——那份已验证"DSP/文本 + LLM 推理 > LLM 直接听音频"。

## 0. 目标(三条主线)

1. **每维工具验证选择**:候选工具在小样本上跑 → 量化 → 选定,不拍脑袋。
2. **交叉验证逻辑设计**:低置信维度(emotion/register)双标交叉,定一致/不一致处置 + 阈值 + 跨层一致性。
3. **Omni+文本表达扩充消融**:量化"文本信息加入"对 Omni 标注/描述的增益,定 L2/L3 输入配方。

## 1. 数据准备

| 数据集 | 规模 | 标注 | 用途 |
|---|---|---|---|
| **Gold Set(人工)** | ~300 条几十秒片段 | 人工标 emotion(主类+强度)、register、gender/age;抽样标细类 + 描述质量参考 | 测工具 accuracy、交叉一致率上界 |
| **Silver Set(无标注)** | ~2000 条 | 无 | 测 pipeline 内部工具间 agreement、失败率、分布 |

- Gold Set 需覆盖**多 register**(对话/播报/影视/教学…)和**非中性情感**(否则 90% 中性,情感维度测不出区分力);
- 统一 16kHz mono;记录标注者人数与一致性(kappa)。

---

## 2. Part A · 每维工具验证

按置信度分三档,验证方法不同:

### A1 · 物理量(WordVoice-5A,单源确定性)

维度:pitch(level+range)/ energy / rate / pause(b0-b4)/ tone(7 类)。
- **无需工具间交叉**(DSP 确定性)。验证 = ① 词级对齐**失败率** ② 各档**分布合理性**(不塌到单档)③ 抽样人工听感一致。
- **决策**:失败率 < 5%、分布无异常 → 采用;分位阈值在 Gold Set 上算 5 档边界并固化。

### A2 · 分类器(gender / age)

- **候选**:见 [[instructTTS标注方案设计]] §9.3 —— 商用友好开源缺口是主要障碍(audeering NC / Vox-Profile NC)。本验证顺带**筛出许可可用的候选**。
- **验证**:Gold Set accuracy + 置信度分布;低置信样本比例。
- **决策**:accuracy > 阈值(gender >95% / age >70%)**且**许可可商用 → 采用;否则触发"轻量自训 / Omni 弱标 / 内部模型"决策。

### A3 · 语义维度(emotion / register)—— 重点,进 Part B 交叉

| 维度 | 候选工具 | 验证指标 | 决策阈值 |
|---|---|---|---|
| emotion 基础类 | emotion2vec+(9 类)vs SenseVoice | 各自 vs Gold accuracy、中文表现、类别覆盖 | accuracy 最高者 + 类别集适配 |
| emotion 细类+强度 | audio-OmniLLM(见 Part C) | vs Gold 抽样一致率 | 进 Part C 消融定 |
| register | text-LLM prompt(7 类)+ audio-Omni 校验 | vs Gold accuracy、**【断句破碎】召回** | accuracy >85%、断句破碎召回 >90% |

> **emotion SER 选型是 linchpin**:选定即固化 L1 类别集。SenseVoice 附带副语言事件 + BGM(顺带 environment),emotion2vec+ 类别更清晰——本验证给出量化对比后再定(参考 §9.2 架构权衡)。

---

## 3. Part B · 交叉验证逻辑

### B1 · 分层策略(哪些维度交叉)

| 层 | 维度 | 是否交叉 |
|---|---|---|
| 物理量 | pitch/energy/rate/pause/tone | ✗ 单源(确定性)|
| 分类器 | gender/age | 单源 + Gold 抽检;低置信样本 Omni 复核 |
| **语义** | **emotion / register** | ✓ **强制双标交叉** |

### B2 · 双标 reconciliation 规则

**emotion**:标注器 A = SER(出 L1 主类);标注器 B = audio-OmniLLM(出 L2 细类,roll-up 到主类)。
- 主类一致 → 高置信入库(细类/强度取 B);
- 主类不一致 → 按数据预算三选一:① 人工仲裁(小样本)② 置信度加权(取高置信源)③ 丢弃(数据充足时最省事)。
- **不一致率本身是质量信号**:若某 register 下不一致率异常高,提示该场景 SER/Omni 有系统性偏差。

**register**:标注器 A = text-LLM(文本主判);标注器 B = audio-Omni(语音校验)。
- 一致 → 入库;不一致 → 以语音为准 或 人工(尤其【断句破碎】需听声学确诊)。

### B3 · 跨层一致性(emotion 专属)

- **roll-up 校验**:L2 细类必须能归到 L1 主类(暴怒 ⊆ 愤怒)。若 L2 audio-LLM 出"暴怒"但 L1 SER 出"happy" → 冲突,flag 进 B2 不一致流程。
- **L3 ⊆ L1/L2**:L3 自由描述以 L1/L2 为事实锚(方案已定),校验描述不与结构化标签矛盾。

### B4 · 一致率指标与放量决策

- **指标**:工具间 agreement rate、Cohen's kappa;跨层 roll-up 一致率。
- **放量决策规则**:
  - Gold Set 上 pipeline 与人工 agreement **> 90%** → 该维度可**无人工全自动放量**;
  - 80–90% → **human-in-loop**(只人工核不一致的部分);
  - < 80% → 工具或 prompt 需返工,不放量。

---

## 4. Part C · Omni + 文本表达扩充消融

**目标**:量化"文本信息加入"对 audio-OmniLLM 的增益,定 L2/L3 的输入配方。仿 [[DSP-LLM韵律评估可行性验证方案]] 的 P1/P2/P3 三路对比。

| 条件 | 输入 | 对应层 |
|---|---|---|
| **C1 audio-only** | 仅语音 | 下界基线 |
| **C2 audio + ASR 文本** | 语音 + 转写 | 文本消歧 |
| **C3 audio + 文本 + L1 标签** | 语音 + 转写 + SER 主类 | 我方 L2 设计 |

- **测什么**:① emotion 细类 vs Gold accuracy ② L3 自由描述质量(LLM-as-judge 1-5 + 人工抽样)③ 与声学的一致性(有无音文矛盾)。
- **假设**:C3 > C2 > C1(文本消歧 + L1 锚点各有增益)。
- **产出**:文本/标签的增益量化 → **确定 L2/L3 到底喂多少上下文**(若 C2≈C3,可省 L1 注入;若 C3 显著更好,坐实"带标签作条件"的必要)。

---

## 5. 评估指标汇总

| 指标 | 用于 | 定义 |
|---|---|---|
| accuracy(vs Gold)| A2/A3/C | 与人工标签一致率 |
| agreement / kappa | B | 工具间一致率(去偶然)|
| roll-up 一致率 | B3 | L2 细类归主类正确率 |
| 描述质量 | C | LLM-judge 1-5 + 人工抽样 |
| 失败率 / 覆盖率 | A1 | 对齐失败、档位塌缩 |
| 断句破碎召回 | A3 register | 质量门有效性 |

## 6. 实施步骤与执行计划

| 阶段 | 任务 | 依赖 | 预计 |
|---|---|---|---|
| 准备 | 构建 Gold(~300 人工)+ Silver(~2000)| 数据源 | 视标注人力 |
| A1 | 跑 WordVoice-5A,验失败率/分布/分位阈值 | Gold+Silver | 1–2h |
| A2 | 候选 gender/age checkpoint vs Gold | 许可确认 | 1–2h |
| A3 | emotion2vec+ vs SenseVoice vs Gold;register prompt vs Gold | Gold | 2–3h |
| C | Omni C1/C2/C3 三路消融 | A3 + Gold | 2–4h |
| B | 用 A/C 结果算交叉一致率,定 reconciliation 阈值与放量规则 | 全部 | 2–3h |
| 汇总 | 出验证报告,回填主方案 §9/§10 | 全部 | 2–3h |

## 7. 风险与应对

| 风险 | 应对 |
|---|---|
| Gold Set 情感偏中性,情感区分力测不出 | 定向从表达性来源(影视/对话)采;style-guided mining 预挖 |
| gender/age 无可商用 checkpoint | 触发 §9.3 决策(轻量自训 / Omni 弱标 / 内部模型)|
| SER 中文 accuracy 不足 | 双标交叉兜底 + Omni 复核;必要时换 checkpoint |
| Omni JSON 输出不稳 | few-shot + retry + 正则兜底(仿 DSP-LLM 风险表)|
| ~300 Gold 样本量小 | 关注效应方向而非显著性;case 级错误分析 |

## 8. 通过标准与产出

- **每维**:选定工具 + 决策阈值(A1/A2/A3);
- **交叉验证**:reconciliation 规则 + 放量决策(全自动 / human-in-loop / 返工)固化;
- **Omni+文本**:增益量化结论,L2/L3 输入配方定稿;
- **回填**:结果更新回 [[instructTTS标注方案设计]] §9(选定工具)/§10(流水线参数)。

---

## 待确认

1. **Gold Set 谁来标、标多少维**(至少 emotion 主类 + register;细类/描述可抽样)?
2. **gender/age 许可缺口的决策**(自训 / Omni 弱标 / 内部模型)——影响 A2 是否需要"自训 baseline"路径。
3. **SER 候选是否就锁 emotion2vec+ 和 SenseVoice 两个**,还是要纳入更多(如内部 SER)?
