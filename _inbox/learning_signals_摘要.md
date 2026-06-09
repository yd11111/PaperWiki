# Learning Signals 摘要报告 (2026-06-09)

> 来源: 57 条未消化的审阅报告建议 (`_review/*.yml`)
> 目的: 供人工评估，决定哪些应纳入规则/检查项/skill 改进

---

## 1. Reader Skill 改进建议 (system_upgrade_suggestion) — 17 条

### 1.1 数据准确性: 表格数字交叉验证 (5 条)

**核心问题**: reader agent 从 PDF 提取的表格数字可能有行偏移、模型身份映射错误，当前无校验机制。

- **TASTE-Streaming**: 提取多模型比较表时，应对每个 baseline 至少做一项交叉验证（文中叙述 vs 表格数字）
- **XTTS**: pdftotext 提取的表格数据应交叉验证：(1) 与正文定性描述对比；(2) 同数据在不同表格中核对
- **NV-Bench**: benchmark 论文的统计数字应做数学验证（如 total = categories x per_category）
- **WavTokenizer**: 消融实验数字在方法节引用时应写完整比较（A=X vs B=Y），不用模糊的"略高于"
- **CosyVoice**: 速查卡片指标应校验方向（WER 越低越好 vs MOS 越高越好），比较词与数值方向须一致

### 1.2 可区分性: 论文原文 vs agent 解读标注 (4 条)

**核心问题**: 笔记中的因果解释经常混淆论文原文与 agent 推断，违反可区分原则。

- **DAC-v2**: 方法节每个 WHY/因果解释后应标注 [论文原文] 或 [agent 解读]
- **IndexTTS2**: "为什么能 work" 模板中应强制标注来源
- **Seed-TTS**: 遇到论文描述模糊的关键组件时，应主动插入 [论文未详述] 标记
- **SoundStream**: SoundStream 的显式标记法是满分范例，可提取为模板

### 1.3 Frontmatter 完整性与语义正确性 (4 条)

**核心问题**: frontmatter 字段虽然填了，但语义可能不对（如 models 列了后继系统而非对比 baseline）。

- **MaskGCT**: models 必须含本文模型；concepts 必须含标题核心概念；datasets 必须含实验所有数据集
- **IndexTTS**: models 应含实验表中的主要 baseline
- **Cont-SPT**: reader 应自动检查方法基于的 baseline 系统是否在 models 中引用
- **CosyVoice-v2**: models 字段规则应明确：列本文模型 + 对比基准 + 关键组件，后继系统仅论文正文讨论时才列

### 1.4 速查卡片与溯源 (2 条)

- **CosyVoice3**: 速查中每个数字必须附带 [Table/Fig/section] 标注
- **OmniVoice**: repro 笔记应增加代码引用准确性验证（笔记描述的代码逻辑 vs 实际仓库代码）

### 1.5 范式感知 (2 条)

- **VALL-E**: 处理范式开创性论文时，应自动检查新定义的概念是否需要创建概念页
- **DAC**: DAC repro 笔记的模块细节/训练配置/复现要点结构可提取为 repro-tier 模板

---

## 2. 审阅检查项改进 (checklist_upgrade_suggestion) — 18 条

### 2.1 从"形式完备"到"语义正确" (4 条)

**最高优先级主题**。当前检查项只验证"字段是否存在"和"标注是否有"，不验证"内容是否正确"。

- **CosyVoice-v2**: 将 frontmatter_complete 拆为 fields_present + fields_semantically_correct
- **TASTE-Streaming**: 增加"比较表非本文模型数字准确性"（抽样 >= 2 个 baseline 与 PDF 核对）
- **WavTokenizer**: 增加"消融数据交叉一致性"——方法节/局限性节引用的消融数字与实验表一致
- **DAC-v2**: 增加"因果解释来源标注覆盖率"（>= 80%），与 claim 标注率并列

### 2.2 实体页 key_papers 一致性 (4 条)

**核心问题**: 反向更新不断 append key_papers，导致 frontmatter 膨胀但 body section 未同步。

- **entity-CodecLM**: 增加 frontmatter key_papers 与 body 关键论文 section 一致性检查；plain text 条目占比（理想为 0%）
- **entity-CFM**: 同上——确保 frontmatter 论文在 body 中有至少一句关系说明
- **entity-SpeechTokenizer**: 增加检查：每条 key_papers 是否对该概念有直接贡献（vs 仅作为下游使用者）
- **WavTokenizer-kb**: 追加表格行时验证字段与列标题的语义一致性

### 2.3 论文特殊类型的检查项缺失 (3 条)

- **HumaneSpeech**: 当论文缺乏定量评估时，笔记须显式声明"本文无标准定量指标"
- **JointDialogueSpeech**: 对声称系统/框架论文，是否有端到端验证结果
- **MINT-Bench**: benchmark 论文 models 字段语义指导：models 仅列本文模型，被评估系统放正文

### 2.4 MOC 检查项 (3 条)

- **moc-TTS-总览**: 按模型/概念浏览区链接优先指向实体页；按概念浏览须有分组；演进 section 须用关系符号
- **moc-零样本-AR**: 自动刷新后运行 year-mismatch 验证；超 80 条触发 sub-MOC 拆分；演进脉络自动匹配验证
- **moc-零样本-自回归**: inline 描述年份与 frontmatter year 一致；超 30 篇建议按技术子路线分组

### 2.5 其他精细化检查 (4 条)

- **SpeakerPoisoning**: 速查卡片指标名是否与论文原文表头一致
- **JoyVoice**: 实验结果表每行来源标注是否精确到单一 Table
- **NV-Bench**: "关键统计量数学一致性"（如 total = categories x per_category）
- **OmniVoice**: 消融总结中的数字与笔记内详细表格一致

---

## 3. 规则盲区 (rule_gap) — 10 条

### 3.1 key_papers 增量膨胀无控制 (3 条)

**最严重的系统性盲区**。反向更新每次 append 一条 key_papers，无上限警告，无质量过滤，导致概念页从"代表作列表"退化为"全量引用列表"。

- **entity-CFM**: 67 条 key_papers，逐篇精读累积的结果。需要触发阈值检查机制（如 > 12 时 lint 报警）
- **entity-SpeechTokenizer**: 同上。append 不触发警告导致增量膨胀无人发现
- **entity-CodecLM**: key_papers 的"仅保留有笔记的论文"要求未明确。plain text 条目处理规则也未明确

> **注**: lint.py L18 已实现 key_papers 数量检查（上限 12），这 3 条建议已部分被 lint 覆盖，但 lint 只在事后报警，未在反向更新时阻断。

### 3.2 特殊论文类型的规则缺失 (3 条)

- **CoCoEmo**: datasets 字段在模板中为空列表，未明确要求至少填写核心数据集
- **HumaneSpeech**: 论文无定量指标时，笔记实验节的处理无标注规范
- **JointDialogueSpeech**: feasibility study 类论文（无完整系统/无端到端验证），审阅标准缺少"完整性"维度

### 3.3 MOC 规则不够精确 (3 条)

- **MINT-Bench**: benchmark 论文的 models 字段语义对 benchmark 适用性有限
- **moc-TTS-总览**: R4 "每条描述 <= 1 行"缺少字数建议（实际从 15 字到 90+ 字）
- **moc-零样本**: R4 的 50 条阈值无"何时触发拆分"的操作指南

### 3.4 年份一致性规则不完整 (1 条)

- **moc-零样本-自回归**: R5 只规定 section 归属匹配 frontmatter year，未规定 inline 描述中的年份文本也须一致

---

## 4. 新问题类型 (new_issue_type) — 9 条

### 4.1 数据错误的新子类型 (3 条)

| 类型 | 描述 | 来源 |
|---|---|---|
| **baseline-number-swap** | 比较表中多个 baseline 数字整体互换（模型身份映射错误），每个数字单独看都是 PDF 真实数字但归到了错误模型名下。根因: agent 读 PDF 表格行偏移 | TASTE-Streaming |
| **metric-naming-ambiguity** | 同一论文中名称相近但语义不同的指标（如 SSIM-F vs FSSIM），笔记需首次出现时明确区分 | SpeakerPoisoning |
| **column-mismatch** | 追加表格行时字段语义与列定义不匹配，是 factual-error 的子类型 | WavTokenizer-kb |

### 4.2 实体页结构退化 (3 条)

| 类型 | 描述 | 来源 |
|---|---|---|
| **frontmatter-body-desync** | frontmatter key_papers（37 条）与 body 关键论文 section（6 条）严重不一致，反复 append 导致 frontmatter 膨胀但 body 未同步 | entity-CodecLM |
| **frontmatter-body-mismatch** | 同上，67 vs 9 | entity-CFM |
| **key-papers-inflation** | key_papers 从代表作列表退化为"所有提及该概念的论文全集"，准入标准缺失 | entity-SpeechTokenizer |

### 4.3 MOC 结构问题 (2 条)

| 类型 | 描述 | 来源 |
|---|---|---|
| **missing-evolution-markers** | 标题为"演进脉络"的 section 缺少实际演进关系标注（箭头/符号），仅为论文列表 | moc-TTS-总览 |
| **year-section-batch-drift** | 批量精读后自动刷新 MOC 时，因 frontmatter year 与 arXiv 发布年份不同，整批论文被错放到错误年份 section。是系统性问题 | moc-零样本-AR |

### 4.4 Frontmatter 语义错误 (1 条)

| 类型 | 描述 | 来源 |
|---|---|---|
| **successor-as-model** | models 字段误列后继系统而非论文实际涉及模型，是 bad-linking 的子类型 | CosyVoice-v2 |

---

## 5. 检查项盲区 (checks_passed_but_principle_failed) — 3 条

这是最值得关注的类别: 说明现有自动检查给出了"通过"但实际上违反了核心原则。

| 检查项 | 通过条件 | 实际问题 | 盲区本质 | 来源 |
|---|---|---|---|---|
| **frontmatter_complete** | 所有字段已填 | models 字段列了后继系统而非对比模型 | 只验证"字段存在"不验证"字段语义正确" | CosyVoice-v2 |
| **claim_coverage 93%** | 标注覆盖率高 | distinguishable 仅 7 分 | claim 标注只验证数据溯源，不验证因果解释的来源标注 | DAC-v2 |
| **claim_coverage 87%** | 标注覆盖率高 | trustworthy 仅 3/10 | 覆盖率只衡量"有没有标来源"，不衡量"标的数字对不对" | TASTE-Streaming |

**共性**: 当前检查体系偏重形式完备性（有没有填/有没有标），缺少内容准确性验证。这是最根本的系统性 gap。

---

## 建议优先级

按影响面和实施难度排序，以下 5 项建议最值得优先纳入:

### P1: 将 frontmatter_complete 拆为 fields_present + fields_semantically_correct
- **影响**: 消除检查项盲区 #1，防止 models/concepts/datasets 语义错误
- **难度**: 中（需定义语义规则）
- **来源**: CosyVoice-v2, MaskGCT

### P2: 反向更新增加 key_papers 阈值阻断 + 准入过滤
- **影响**: 阻止 30+ 个概念页继续膨胀（当前最多已达 91 条）
- **难度**: 低（lint L18 已有检测，需在反向更新流程中加 pre-check）
- **来源**: entity-CFM, entity-SpeechTokenizer, entity-CodecLM

### P3: 比较表数字交叉验证步骤
- **影响**: 消除检查项盲区 #3（claim_coverage 高但 trustworthy 低），防止 baseline-number-swap
- **难度**: 高（需 PDF 原表回查）
- **来源**: TASTE-Streaming, XTTS, NV-Bench

### P4: 因果解释强制标注 [论文原文]/[agent 解读]
- **影响**: 消除检查项盲区 #2，提升可区分性
- **难度**: 低（在 reader prompt 中加指令即可）
- **来源**: DAC-v2, IndexTTS2, Seed-TTS

### P5: 速查卡片数字必须附带 [Table/Fig/section] 溯源标注
- **影响**: 速查卡片是最常被引用的部分，数字无溯源则无法验证
- **难度**: 低（在速查模板中加硬性要求）
- **来源**: CosyVoice3
