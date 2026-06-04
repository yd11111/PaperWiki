# PaperWiki Harness 升级报告

> 对照 Harness 设计规范，系统性分析 PaperWiki 各层差距，给出升级方向和优先级排序。
>
> 日期: 2026-06-04
> 输入: harness-design-spec.md + harness-case-studies.md + AGENTS.md + MOC 审阅意见 + review-checklist.md + 社区调研 + harness 模板
> 深度: 诊断 + 方向指引（不含具体规格草稿）

---

## 0. Executive Summary

**当前成熟度: Level 1.5** — 拥有 AGENTS.md + 模板 + 审阅框架，但三者之间缺乏机械化连接。

**一句话诊断**: PaperWiki 的架构设计（信任层级、反向更新、两层审阅）在社区中几乎独一无二，但约束停留在"文档约束"层面、审阅覆盖只有笔记一种、闭环机制完全缺失 — 相当于有骨架但缺少肌肉和神经。

**核心定理** (Spec §0.1):
> Agent 的产出质量不取决于 agent 的智力，取决于围绕 agent 的约束和反馈机制的质量。

对照此定理，PaperWiki 目前的问题不是 agent "不够小心"，而是约束和反馈机制有结构性缺口。

### 关键设计决策: 信任机制转型

本次升级确认了一个根本性的设计哲学变化：

**旧模式**: AI 生成 → pending-review → 等人确认 → confirmed → 进入可信层（人是必经门）

**新模式**: AI 生成 → 自动化质量门（reviewer agent + lint + 反幻觉检查）→ 达标即自动进入可信层。人偶尔审阅发现问题 → 纠正本身成为 few-shot 信号 → 反哺自动化质量门的规则/检查项 → 下次自动更准

**本质变化**: 人从"必经审批者"变为"偶尔介入的校准者"。人的每次介入不是在做 review 工作，而是在训练系统。

AGENTS.md 原则 1 当前写的是"AI 生成内容在未经人类确认前，不得进入可信层"，需要修正为"AI 生成内容通过自动化质量门后可自动进入可信层。人的纠正是系统校准信号，不是必经审批。"

### 总览: 15 个差距

| 层 | 差距数 | P0 | P1 | P2 | P3 |
|---|--------|----|----|----|----|
| 治理层 | 4 | 1 | 1 | 1 | 1 |
| 执行层 | 3 | 1 | — | 1 | 1 |
| 验证层 | 4 | 1 | 2 | 1 | — |
| 闭环 | 2 | — | 1 | — | 1 |
| 熵管理 | 1 | — | — | 1 | — |
| 度量 | 1 | — | — | — | 1 |
| **合计** | **15** | **3** | **4** | **4** | **4** |

---

## 1. 治理层 (Governance) — 4 个差距

### G1. AGENTS.md 是百科全书，不是目录

| 项 | 内容 |
|---|---|
| **现状** | ~500 行, 17 个 section，每个 agent 加载全量 |
| **违反** | Spec §0.7 — "Tier 1 是目录，不是百科全书"；OpenAI 教训："巨大的 AGENTS.md 会失败" |
| **证据** | 上下文利用率超 40% 后质量急剧下降；当前 AGENTS.md 单文件就可能占 3-5K tokens |
| **方向** | 重构为 ~100 行 TOC，详细规则拆到独立文件（moc-rules.md, kb-rules.md, review-rules.md 等），每个 agent 按角色加载相关 Tier 2 文件 |
| **参考案例** | OpenAI "AGENTS.md as TOC not encyclopedia"；Ghostty AGENTS.md 的"每次犯错加一行" |
| **优先级** | **P1** — 力乘数效应：修此一项改善所有 agent 的上下文质量 |

### G2. MOC 规则几乎空白（§11 = 7 行）

| 项 | 内容 |
|---|---|
| **现状** | §11 仅有 7 条规则，主要管 lifecycle（deprecated/merged 处理），对 MOC 的内容结构、边界、分层、准入完全没有约束 |
| **违反** | Spec §0.4 — "如果一条约束不能被机械化验证，agent 就会偏离" |
| **证据** | MOC 审阅暴露 15 类问题，其中 11 类根因是"上游规则缺失" — 教科书级的"文档约束不够 → agent 偏离" |
| **15 类问题的根因归属** | 混层(无分层规则)、边界不清(无 scope 声明)、粒度不一致(无准入分层)、代表模型混入 backbone(无准入规则)、论文区失控(无容量阈值)、年份错乱(无从 frontmatter 取值规则)、SOTA 表条件不一致(无实验协议规则)、演进箭头语义不明(无关系类型规则)、任务页/MOC 角色重叠(无职责边界)、强判断无来源(无标注规则)、alias 未充分使用(执行问题) |
| **方向** | 需要完整 MOC 规则体系：(1) MOC 宪法原则（可导航/不混层/可溯源/不腐烂/不过载）；(2) 具体规则条目（scope boundary、concept 分层、model 准入、paper 容量限制、SOTA 表条件标注、关系类型枚举、年份取值规则）；(3) MOC 审阅 checklist |
| **参考案例** | GitHub Always/Ask/Never 三层边界；van Krieken "不维护 MOC"反证（规则不够好则维护成本 > 价值） |
| **优先级** | **P0** — MOC 是用户的主导航界面，当前问题已影响使用体验 |

### G3. 三层治理只覆盖笔记审阅，其他 4 个领域空白

| 项 | 内容 |
|---|---|
| **现状** | review-checklist.md 实现了宪法/法律/判例三层，但只用于笔记审阅。概念提取、MOC、实体页内容、系统同步这 4 个领域没有三层治理 |
| **违反** | Spec §0.2 — 三层治理应覆盖所有知识生产环节 |
| **证据** | upgrade-context 已识别 5 个审阅模块(A-E)，当前只有 A 有 checklist |
| **方向** | 为 B(KB 提取审阅)、C(MOC 分类审阅)、D(实体页内容审阅) 分别建立三层治理：宪法原则 + 具体检查项 + 判例存放路径。E(系统同步) 已有 AGENTS.md §12 覆盖但可强化 |
| **参考案例** | academic-research-skills 的 7-agent Paper Reviewer（每个审阅维度独立 agent） |
| **优先级** | B=**P1**, C=**P0**(与 G2 配套), D=**P3**, E=已有 |

### G4. 约束普遍停留在"文档约束"层面

| 项 | 内容 |
|---|---|
| **现状** | 大部分规则靠 agent "读 AGENTS.md 后自觉遵守"（文档约束 = 最低可靠度）|
| **违反** | Spec §0.4 — 约束三形态（文档→模板→脚本），应按强度逐步升级 |
| **应升级的高频违反项** | (1) 年份一致性 → 脚本检查 frontmatter year vs MOC section；(2) 概念去重 → 脚本检查 title/aliases 重叠；(3) frontmatter 完整性 → 脚本验证 schema；(4) MOC 内重复条目 → 脚本检测同页多次出现的 wikilink |
| **方向** | 将上述 4 项从文档约束升级为脚本约束。脚本错误信息应是 agent 可读的修复指令（如"Sources/Dragon-FM.pdf 应重命名为 ..." 而非 "naming violation detected"） |
| **参考案例** | OpenAI "lint error = agent prompt for self-correction"；Hashimoto 两种修复：隐式 prompt vs 程序化工具 |
| **优先级** | **P2** — 前提是先有规则(G2)，再有脚本验证规则 |

---

## 2. 执行层 (Execution) — 3 个差距

### E1. 单 Agent 承担 7 步 Pipeline 全部角色

| 项 | 内容 |
|---|---|
| **现状** | paperwiki-reader 一个 skill 完成：PDF 阅读 → KB 检索 → 笔记生成 → 审阅 → commit → 反向更新 → 自动检查。涵盖 Research + Execution + Review + Cleanup 四种角色 |
| **违反** | Spec §0.5 — "角色不混合"；"最小上下文：每个 agent 只接收与当前任务相关的规则和文件" |
| **证据** | 到 Step 6（反向更新）时，上下文已包含：完整 PDF 文本 + 6 个 KB 页面 + 完整笔记草稿 + 审阅报告 + 所有规则。质量在后半段下降是必然的 |
| **方向** | 分阶段拆分，渐进实施：**Phase 1**（最小改动）— 把审阅拆为独立 subagent dispatch（解决 E2）；**Phase 2** — 把反向更新拆为独立 Updater agent（scoped write 概念库/模型库）；**Phase 3** — KB 检索作为 Initializer 阶段（Anthropic 模式），输出结构化 JSON 传给后续 agent |
| **参考案例** | Anthropic "Initializer + Coding agent" 两阶段；Carlini 16 个 parallel Claudes 各有独立 Docker；statewright 状态机约束不同阶段可用工具集 |
| **优先级** | **P3** — 当前 pipeline 能跑，拆分是优化。但 Phase 1（审阅独立 = E2）提前到 P0 |

### E2. 审阅不是真正独立的 Agent

| 项 | 内容 |
|---|---|
| **现状** | paperwiki-reviewer 作为 Step 4 在同一 session 中运行。reviewer 能看到 reader 的全部生成过程、中间推理、甚至失败重试 |
| **违反** | Spec §0.5 — "生成者不审自己"是最核心的 agent 分离原则 |
| **为什么关键** | 信任机制转型后，自动审阅成为唯一门控。同一上下文中的 reviewer 会继承 reader 的偏见，不够可靠承担"自动门"角色 |
| **方向** | Reviewer 必须是独立 subagent dispatch：(1) 独立 session（不共享生成上下文）；(2) 只接收：生成的笔记 + checklist + 相关 KB 页面；(3) 不接收：generation prompt、中间推理、PDF 原文；(4) 输出：_review/*.yml + pass/revise 结论。pass 则自动进入可信层 |
| **参考案例** | OpenAI agent-to-agent review；Carlini 通过 git 而非共享内存协调 agent |
| **优先级** | **P0** — 自动门的核心组件，信任机制转型的前提 |

### E3. MOC 生成 Agent 缺位

| 项 | 内容 |
|---|---|
| **现状** | generate-mocs skill 存在但指向不存在的脚本。MOC 刷新靠 reader pipeline Step 7 顺带完成 |
| **违反** | Spec §0.5 — 最小权限 + 角色不混合 |
| **证据** | MOC 审阅 15 类问题中很多因为 MOC 更新被当作 pipeline "附带产出"而非独立任务 |
| **方向** | 建立独立 MOC Agent：(1) scoped write 仅限 _MOC/；(2) 输入：待收录论文列表 + MOC 规则文件 + 当前 MOC 内容；(3) 不接收 pipeline 上下文；(4) 可被 per-ingest 触发（增量）或手动触发（重构） |
| **参考案例** | Carlini task locking via filesystem；LangChain middleware hooks |
| **优先级** | **P2** — 与 G2（MOC 规则）配套，先有规则再有专门 agent |

---

## 3. 验证层 (Verification) — 4 个差距

### V1. 概念提取（反向更新）无审阅 — 最高污染风险

| 项 | 内容 |
|---|---|
| **现状** | Step 6 反向更新直接修改概念库/模型库，没有任何审阅环节。唯一保护是 append/substantive 规则（§7），但这是文档约束 |
| **违反** | Spec §0.3 — 双向反压模型的"下游兜底"缺失；review-checklist 原则"不污染"在实际流程中无检查点 |
| **为什么最高风险** | 概念页是 KB 检索的数据源 → 污染的概念页影响后续所有精读的 KB 背景 → 错误扩散。整个系统中唯一一个"错误会被放大"的环节 |
| **方向** | 新建 KB 更新审阅模块：(1) 触发时机：每次反向更新前；(2) 宪法原则：不污染知识库；(3) 检查维度：factual accuracy、overclaim、正确分类 append vs substantive、去重检查、准入规则遵守；(4) 输出：_review/{论文}-kb-review.yml；(5) 审阅不通过 → 跳过反向更新 + 记录日志 |
| **参考案例** | Eric Ma "衍生物必须引用原文——没有证据不允许综合"；academic-research-skills 反幻觉门（不可跳过） |
| **优先级** | **P1** — 污染风险随 vault 规模线性增长，90+ 实体页已进入风险区 |

### V2. MOC 无审阅机制

| 项 | 内容 |
|---|---|
| **现状** | MOC 生成后无任何质量检查。用户 MOC 审阅是第一次人工全面检查，发现 15 类问题 |
| **违反** | Spec §0.3 — 下游反馈完全缺失；Spec §0.2 — 三层治理中 MOC 领域法律层为空 |
| **证据** | 15 类问题、年份错乱、重复条目、backbone 混入代表模型 — 全部是审阅本应捕获的问题 |
| **方向** | 新建 MOC 审阅模块：(1) 触发时机：定期批量；(2) 宪法原则：可导航/不混层/可溯源/不腐烂/不过载；(3) 检查维度：scope boundary、concept 分层一致、model 准入、年份与 frontmatter 一致、无重复条目、强判断有标注、论文区不超容量阈值；(4) 输出：_review/moc-review-YYYY-MM-DD.yml |
| **参考案例** | OpenAI agent-to-agent review；Fowler 2x2 "推理性下游"象限 |
| **优先级** | **P0** — 与 G2（MOC 规则）同步实施，规则 + 审阅构成完整反压 |

### V3. Lint 覆盖严重不足

| 项 | 内容 |
|---|---|
| **现状** | Step 5 per-ingest 检查仅查死链。§12 全面系统检查定义了更多项目，但是列表不是脚本 — 靠 agent 手动执行，不是确定性自动化 |
| **违反** | Spec §0.4 — "脚本约束"是最高可靠度，当前 §12 是"文档约束"伪装成检查项；Fowler 2x2 中"计算性下游"象限应优先投资 |
| **可机械化的检查项（至少 8 个）** | (1) frontmatter 字段存在性验证；(2) frontmatter year vs MOC section year 一致；(3) 同 MOC 内重复 wikilink 检测；(4) 概念页 title/aliases 重叠检测；(5) deep/repro 笔记 MOC 覆盖率；(6) 审阅 callout 存在性；(7) source 字段指向的 PDF 是否存在；(8) wikilink 目标文件存在性（已有） |
| **方向** | 将上述检查项实现为 bash/python 脚本：(1) 每条 lint rule 的错误信息 = agent 可直接执行的修复指令；(2) 所有 rule 设为 error（不用 warn）；(3) 禁用 inline-disable；(4) 集成到 per-ingest（轻量子集）和 full-check（完整集） |
| **参考案例** | OpenAI "lint error message = agent prompt"；Hashimoto "programmatic tool for deterministic constraints"；claude-obsidian 8 类 lint |
| **优先级** | **P1** — 信任机制转型后，机械化检查是自动门的第一道防线 |

### V4. 审阅不区分计算性 vs 推理性检查

| 项 | 内容 |
|---|---|
| **现状** | review-checklist.md 所有检查项都由审阅 agent 执行（推理性检查），包括本可机械化的项目（如"frontmatter 字段存在且非空"） |
| **违反** | Fowler 2x2 矩阵 — 计算性检查不应消耗推理 agent 的上下文和注意力 |
| **方向** | 将 review-checklist 中的计算性检查项抽离到 lint 脚本（V3），审阅 agent 只负责推理性检查（因果解释质量、claim 与证据对应、KB 背景定位质量等）。审阅 agent 上下文更聚焦，推理质量更高 |
| **参考案例** | Fowler "Feedforward/feedback × computational/inferential 2x2"；LangChain "仅改 harness 不换模型，从 rank 30 升到 Top 5" |
| **优先级** | **P2** — 与 V3 同步实施，拆分后两端都更高效 |

---

## 4. 闭环机制 (Closed Loop) — 2 个差距

### L1. 四步闭环缺失（有判例无消化）

| 项 | 内容 |
|---|---|
| **现状** | 174 份 _review/*.yml 已积累大量判例（步骤①），但步骤②③④完全不存在。判例在磁盘上沉睡 |
| **违反** | Spec §0.6 — "闭环是 harness 与静态规则系统的本质区别" |
| **量化损失** | review-checklist.md 更新记录：v1 → v1.1 只发生过一次（2026-06-02），此后 170+ 次审阅没有产生任何规则更新 — 170 次学习机会被浪费 |
| **信任机制转型后的新角色** | 闭环不仅消化审阅报告，还要消化人工纠正 — 人修改了某个 confirmed 页面 = 最高优先级学习信号 |
| **方向** | 建立 Pattern Analyzer 能力：(1) 触发：每 N 次审阅或手动；(2) 输入：最近 _review/*.yml + 人工纠正记录；(3) 分析：问题类型频率 → 高频问题 → 根因归属（模板/规则/skill/检索哪层缺失）；(4) 输出：_review/pattern-analysis-YYYY-MM-DD.md 含规则修改建议；(5) 人确认后才执行规则修改；(6) 每次更新追加到 checklist 更新记录表 |
| **人校准捕获机制** | 人修改 confirmed 页面时，系统记录 correction entry：what was wrong + what it should be + which quality gate should have caught this → 模式分析的最高优先级输入 |
| **参考案例** | Hashimoto "every mistake → engineer a solution"；OpenAI "doc-gardening agent" |
| **优先级** | **P1** — 人校准信号的落地通道，信任机制转型的必要组件 |

### L2. 无闭环质量信号

| 项 | 内容 |
|---|---|
| **现状** | 没有跟踪任何闭环效果指标 |
| **违反** | Spec §0.6 闭环质量信号表 — 三个关键信号未被追踪 |
| **方向** | 在 pattern-analysis 输出中加入与上一次分析的对比：(1) 同一问题类型频率趋势（规则生效 = 递减）；(2) 新问题类型出现率（系统扩展 = 偶尔出现）；(3) 规则更新频率（系统收敛 = 稳定） |
| **参考案例** | Augment Code PEV Loop |
| **优先级** | **P3** — 依赖 L1 先运行 |

---

## 5. 熵管理 (Entropy Management) — 1 个差距

### N1. 四种熵已出现，无系统性清理机制

| 项 | 内容 |
|---|---|
| **现状** | 无专门熵管理机制。清理依赖人工发现或 per-ingest 死链检查 |
| **违反** | Spec §0.8 — "清理吞吐量应与生成吞吐量成比例" |
| **四种熵的实例** | **冗余**: Dragon-FM/DiaMoE-TTS 重复出现在同一 MOC；**漂移**: §11 定义的 MOC 规则 vs MOC 实际内容严重偏离；**不一致**: 不同 MOC 对"核心概念"粒度标准不同；**过时**: SOTA 表无更新时间戳 |
| **方向** | 不需要独立 Cleanup Agent（当前规模不需要）。将熵检测分散到 lint（计算性：重复检测、年份一致性）和审阅（推理性：漂移、不一致）中。当 vault 规模翻倍后再评估 |
| **参考案例** | Karpathy "Lint 是一等公民"；OpenAI "garbage collection agent"（PaperWiki 当前规模不需要） |
| **优先级** | **P2** — 作为 V3（lint 扩展）和 V2（MOC 审阅）的附带产出 |

---

## 6. 度量与可观测性 (Metrics) — 1 个差距

### M1. 无系统级度量

| 项 | 内容 |
|---|---|
| **现状** | CLAUDE.md 有手动 vault 状态快照；log.md 有操作记录；§17 有 backlog 告警阈值。这些是快照不是趋势，是告警不是度量 |
| **违反** | Spec §0.10 — 至少需要 4 个指标 |
| **方向** | 不建议搭建复杂度量系统。在 pattern-analysis 输出中记录简单指标：(1) 审阅积压趋势；(2) 问题类型频率变化；(3) 可信层增长率。格式用 markdown 表格 |
| **参考案例** | Microsoft Azure SRE Agent metric-driven 迭代（但 PaperWiki 不需要同等复杂度） |
| **优先级** | **P3** — 前提是闭环机制先运行 |

---

## 7. 自动门设计方向

信任机制转型后，自动门是系统可信度的核心保障。

### 自动门工作流

```
AI 生成内容
    ↓
[Lint 脚本] ── fail → 自动修复或标注 + 降级 ──→ 不入可信层
    ↓ pass
[Reviewer Agent (独立 session)] ── high issue → 标注 + 不入可信层
    ↓ pass / pass-with-fixes
自动进入可信层 (status: confirmed)
    ↓
人偶尔审阅发现问题
    ↓
纠正 = few-shot 信号 (结构化记录)
    ↓
反哺: 新 lint 规则 / 新 checklist 项 / 规则修正
    ↓
下次自动门更准
```

### 自动门对各项差距的影响

| 差距 | 原优先级 | 转型后 | 原因 |
|------|---------|--------|------|
| E2 审阅独立 | P1 | **P0** | 自动审阅成为唯一门控，必须可靠 |
| V3 Lint 扩展 | P2 | **P1** | 机械化检查是自动门第一道防线 |
| L1 闭环机制 | P2 | **P1** | 人校准信号必须有通道反哺系统 |
| V1 KB 审阅 | P1 | 不变 | pass/fail 现在直接控制 auto-promotion |
| pending-review 状态 | 必须人工解锁 | 自动审阅未通过的标记，通过后自动升级 | — |

---

## 8. 优先级总表

| 优先级 | ID | 改什么 | 层 | 核心原则 | 参考案例 |
|--------|-----|--------|-----|---------|---------|
| **P0** | G2 | MOC 规则补充（§11 从 7 行扩展为完整规则体系） | 治理 | §0.4 机械化 + §0.2 三层 | GitHub Always/Ask/Never |
| **P0** | V2 | MOC 审阅标准（新建 moc-review-checklist） | 验证 | §0.3 双向反压 | OpenAI agent-to-agent |
| **P0** | E2 | 审阅拆为独立 subagent（自动门核心组件） | 执行 | §0.5 生成者不审自己 | Carlini Docker 隔离 |
| **P1** | V1 | KB 更新审阅（新建 kb-review-checklist） | 验证 | §0.3 + 不污染原则 | Eric Ma 衍生物引用 |
| **P1** | G1 | AGENTS.md 重构为 TOC + 拆分专用规则文件 | 治理 | §0.7 上下文管理 | OpenAI TOC 模式 |
| **P1** | V3 | Lint 脚本扩展（8 项机械化检查） | 验证 | §0.4 脚本约束 | OpenAI lint=prompt |
| **P1** | L1 | 闭环机制（pattern analysis + 人校准捕获） | 闭环 | §0.6 闭环演进 | Hashimoto error→rule |
| **P2** | V4 | 审阅拆分计算性 vs 推理性 | 验证 | Fowler 2x2 | LangChain harness-only 提升 |
| **P2** | G4 | 高频违反项升级为脚本约束 | 治理 | §0.4 升级路径 | Hashimoto 程序化工具 |
| **P2** | E3 | MOC Agent 独立化 | 执行 | §0.5 角色不混合 | Carlini task locking |
| **P2** | N1 | 熵管理（分散到 lint + 审阅中） | 验证 | §0.8 熵管理 | Karpathy Lint 一等公民 |
| **P3** | E1 | Pipeline 多 agent 分阶段 | 执行 | §0.5 + §0.7 | Anthropic 两阶段 |
| **P3** | G3-D | 实体页内容审阅标准 | 治理 | §0.2 三层 | — |
| **P3** | L2 | 闭环质量信号追踪 | 闭环 | §0.6 质量信号 | Augment PEV Loop |
| **P3** | M1 | 简单度量（markdown 表格级） | 度量 | §0.10 | — |

---

## 9. 依赖关系与实施批次

### 依赖图

```
G2 (MOC 规则) ──→ V2 (MOC 审阅) ──→ E3 (MOC Agent)
                                      ↓
G1 (AGENTS.md TOC) ──→ 所有 agent 性能提升
                                      
E2 (审阅独立) ──→ V1 (KB 审阅)
              ──→ V2 (MOC 审阅, 需要独立 reviewer 模式)
              
V3 (Lint 扩展) ──→ V4 (审阅拆分计算/推理)
              ──→ G4 (脚本约束升级)
              ──→ N1 (熵管理, 计算性部分)
              
V1 + V2 积累判例 ──→ L1 (闭环) ──→ L2 (质量信号) ──→ M1 (度量)
```

### 建议实施批次

| 批次 | 包含 | 预期产出 | 前置条件 |
|------|------|---------|---------|
| **Batch 1** | G2 + V2 + E2 | MOC 规则体系 + MOC 审阅标准 + 审阅独立化 | 无 |
| **Batch 2** | V1 + G1 + V3 | KB 审阅 + AGENTS.md 瘦身 + Lint 脚本 | Batch 1（验证审阅模式可行） |
| **Batch 3** | L1 + V4 + G4 | 闭环机制 + 审阅拆分 + 约束升级 | Batch 1-2 积累判例 |
| **Batch 4** | E3 + N1 | MOC Agent + 熵管理 | Batch 1（MOC 规则就绪） |
| **Batch 5** | E1 + L2 + M1 + G3-D | Pipeline 拆分 + 质量信号 + 度量 + 实体审阅 | Batch 3-4 验证闭环可行 |

---

## 10. 交叉洞察

**1. PaperWiki 的护城河是"自动化质量门 + 人校准闭环"。** 社区项目要么靠人审批（不可扩展），要么靠纯自动化（无学习能力）。PaperWiki 的目标是：自动门足够强使得日常不需要人介入，但当人介入时每次纠正都让系统变得更好。

**2. G2 是此次升级的支点。** MOC 审阅 15 类问题中 11 类指向 §11 规则缺失。补上 MOC 规则后，现有 pipeline 在不做其他改动的情况下就能改善 MOC 质量。投入产出比最高。

**3. "不独立的审阅"是系统性弱点。** E2 不仅影响笔记审阅，还决定了 V1、V2 能否真正发挥作用。如果新增审阅模块仍在同一 session 中运行，添加再多 checklist 也会被同一上下文的偏见削弱。信任机制转型后，这个弱点从"影响质量"升级为"影响可信度"。

**4. 闭环机制决定长期 ROI。** 目前所有改进都是"一次性注入"。只有 L1 能让系统自我改进。正确顺序：先扩审阅覆盖 → 积累判例 → 再建闭环。人校准信号是闭环的最高价值输入。

**5. 当前规模不需要 Level 3-4 能力。** 100 篇笔记 + 90 实体页，不需要无人值守并行、自动熵管理等 Level 4 特征。Batch 1-3 将系统推进到 Level 2.5 即可满足未来 6-12 个月需求。过早追求 Level 3+ 违反 Spec §0.11 "避免过度设计"。

---

## 附录: 参考来源映射

| 差距 | 主要参考 |
|------|---------|
| G1 (AGENTS.md TOC) | OpenAI "Harness Engineering"; Spec §0.7 |
| G2 (MOC 规则) | GitHub 2500+ 仓库分析; MOC 审阅 15 类问题 |
| G3 (三层治理扩展) | Spec §0.2; academic-research-skills |
| G4 (约束升级) | Hashimoto Ghostty; OpenAI lint |
| E1 (Pipeline 拆分) | Anthropic 两阶段; Carlini 并行 agent |
| E2 (审阅独立) | Spec §0.5; Carlini Docker 隔离 |
| E3 (MOC Agent) | Carlini task locking; LangChain middleware |
| V1 (KB 审阅) | Eric Ma 衍生物引用; Spec §0.3 |
| V2 (MOC 审阅) | OpenAI agent-to-agent; Fowler 2x2 |
| V3 (Lint 扩展) | OpenAI lint=prompt; claude-obsidian 8 类 lint |
| V4 (计算/推理拆分) | Fowler 2x2; LangChain harness-only 提升 |
| L1 (闭环) | Hashimoto error→rule; OpenAI doc-gardening |
| L2 (质量信号) | Augment Code PEV Loop; Spec §0.6 |
| N1 (熵管理) | Karpathy Lint; Spec §0.8 |
| M1 (度量) | Microsoft Azure SRE; Spec §0.10 |
