# Learning Signals 摘要报告 (2026-06-09)

> 来源: 57 条未消化的审阅报告建议
> **操作**: 勾选 ☑ = 采纳 (我来实施)，不勾 = 搁置。标注完告诉我。

---

## 1. Reader Skill 改进建议 (17 条)

### 1.1 数据准确性: 表格数字交叉验证

> 核心问题: reader 从 PDF 提取表格数字可能有行偏移/模型映射错误，当前无校验。

- [ ] **1.1a** 提取多模型比较表时，对每个 baseline 至少做一项交叉验证 (文中叙述 vs 表格数字)
- [ ] **1.1b** pdftotext 表格数据应与正文定性描述对比 + 同数据在不同表格中核对
- [ ] **1.1c** benchmark 论文统计数字应做数学验证 (如 total = categories × per_category)
- [x] **1.1d** 消融实验数字在方法节引用时应写完整比较 (A=X vs B=Y)，不用"略高于"
- [ ] **1.1e** 速查卡片指标应校验方向 (WER 越低越好 vs MOS 越高越好)

### 1.2 可区分性: 论文原文 vs agent 解读

> 核心问题: 笔记中因果解释混淆论文原文与 agent 推断。

- [x] **1.2a** 方法节因果解释后标注 `[论文原文]` 或 `[agent 解读]`
- [x] **1.2b** "为什么能 work" 模板中强制标注来源
- [x] **1.2c** 论文描述模糊的关键组件插入 `[⚠️ 论文未详述]` 标记
- [x] **1.2d** 提取 SoundStream 笔记的显式标记法为模板参考

### 1.3 Frontmatter 语义正确性

> 核心问题: 字段填了但语义可能不对 (如 models 列了后继系统)。

- [x] **1.3a** models 必须含本文模型; concepts 含标题核心概念; datasets 含实验所有数据集
- [x] **1.3b** models 应含实验表中的主要 baseline
- [x] **1.3c** reader 自动检查方法基于的 baseline 是否在 models 中引用
- [x] **1.3d** models 字段规则: 本文模型 + 对比基准 + 关键组件，后继系统仅正文讨论时才列

### 1.4 速查卡片与溯源

- [ ] **1.4a** 速查中每个数字必须附带 `[Table/Fig/§]` 标注
- [x] **1.4b** repro 笔记增加代码引用准确性验证

### 1.5 范式感知

- [x] **1.5a** 范式开创性论文应自动检查新概念是否需创建概念页
- [x] **1.5b** 提取 DAC repro 笔记结构为 repro-tier 模板

---

## 2. 审阅检查项改进 (18 条)

### 2.1 从"形式完备"到"语义正确" ⭐

> 当前检查只验证"有没有"，不验证"对不对"。

- [x] **2.1a** frontmatter_complete 拆为 `fields_present` + `fields_semantically_correct`
- [x] **2.1b** 增加"比较表非本文模型数字准确性" (抽样 ≥2 baseline 与 PDF 核对)
- [x] **2.1c** 增加"消融数据交叉一致性" — 方法节引用数字与实验表一致
- [x] **2.1d** 增加"因果解释来源标注覆盖率" (≥80%)

### 2.2 实体页 key_papers 一致性

- [x] **2.2a** frontmatter key_papers 与 body 关键论文 section 一致性检查
- [x] **2.2b** frontmatter 论文在 body 中须有至少一句关系说明
- [x] **2.2c** 每条 key_papers 是否对概念有直接贡献 (vs 仅下游使用)
- [x] **2.2d** 追加表格行时验证字段与列标题语义一致

### 2.3 论文特殊类型

- [x] **2.3a** 论文缺乏定量评估时须显式声明"本文无标准定量指标"
- [x] **2.3b** 声称系统/框架论文须检查有无端到端验证结果
- [x] **2.3c** benchmark 论文: models 仅列本文模型，被评估系统放正文

### 2.4 MOC 检查项

- [x] **2.4a** 按模型/概念浏览区链接优先指向实体页; 演进 section 须用关系符号
- [x] **2.4b** 自动刷新后运行 year-mismatch 验证; 超 80 条触发拆分建议
- [x] **2.4c** inline 描述年份与 frontmatter year 一致; 超 30 篇按子路线分组

### 2.5 其他精细化

- [x] **2.5a** 速查指标名与论文原文表头一致
- [ ] **2.5b** 实验结果表每行来源精确到单一 Table
- [x] **2.5c** 关键统计量数学一致性 (total = categories × per_category)
- [x] **2.5d** 消融总结数字与笔记内详细表格一致

---

## 3. 规则盲区 (10 条)

### 3.1 key_papers 增量膨胀

> ⚠️ lint L18 已调整 (概念页取消上限)。以下关注反向更新流程。

- [x] **3.1a** 概念页 key_papers > 20 时，新论文改追加到正文而非 frontmatter
- [x] **3.1b** append 时做准入判断: 论文是否对概念有直接贡献
- [x] **3.1c** key_papers 只保留有笔记的论文 (禁止 plain text 条目)

### 3.2 特殊论文类型

- [x] **3.2a** datasets 字段模板要求至少填写核心数据集
- [x] **3.2b** 论文无定量指标时实验节的标注规范
- [x] **3.2c** feasibility study 类论文审阅增加"完整性"维度

### 3.3 MOC 规则补充

- [x] **3.3a** benchmark 论文 models 字段语义指导
- [x] **3.3b** R4 每条描述增加字数建议 (当前 15~90+ 字差异过大)
- [x] **3.3c** R4 增加"何时触发拆分"操作指南

### 3.4 年份一致性

- [x] **3.4a** R5 扩展: inline 年份文本也须与 frontmatter year 一致

---

## 4. 新问题类型 — 纳入审阅词汇表? (9 条)

### 4.1 数据错误子类型

- [x] **4.1a** `baseline-number-swap` — 比较表多 baseline 数字整体互换 (行偏移)
- [x] **4.1b** `metric-naming-ambiguity` — 同论文中名称相近但语义不同的指标需区分
- [x] **4.1c** `column-mismatch` — 追加表格行时字段与列定义不匹配

### 4.2 实体页结构退化

- [x] **4.2a** `frontmatter-body-desync` — frontmatter key_papers 与 body 严重不一致
- [x] **4.2b** `key-papers-inflation` — key_papers 退化为全量引用列表

### 4.3 MOC 结构问题

- [x] **4.3a** `missing-evolution-markers` — 演进脉络缺少关系符号
- [x] **4.3b** `year-section-batch-drift` — 批量刷新时年份映射系统性偏移

### 4.4 Frontmatter 语义

- [x] **4.4a** `successor-as-model` — models 字段误列后继系统

---

## 5. 检查项盲区 ⭐ (3 条)

> 检查给出"通过"但实际违反核心原则。

- [x] **5.1** frontmatter_complete 通过但 models 内容错误 — 只查"存在"不查"正确"
- [x] **5.2** claim_coverage 93% 通过但 distinguishable 仅 7 分 — 不验证因果解释来源
- [x] **5.3** claim_coverage 87% 通过但 trustworthy 仅 3/10 — 不验证数字对不对

---

## 建议优先级 (供参考)

| 优先级 | 条目 | 影响 | 难度 |
|--------|------|------|------|
| P1 | 2.1a — frontmatter_complete 拆两层 | 消除盲区 #1 | 中 |
| P2 | 3.1a — key_papers > 20 改追加正文 | 阻止膨胀 | 低 |
| P3 | 2.1b — 比较表数字交叉验证 | 防 baseline-number-swap | 高 |
| P4 | 1.2a — 因果解释强制标注来源 | 提升可区分性 | 低 |
| P5 | 1.4a — 速查数字附带溯源标注 | 速查可信度 | 低 |
