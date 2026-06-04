# PaperWiki Harness 升级计划 v2

> v1 (harness-upgrade-design.md) 是诊断报告,识别了 15 个差距。
> v2 记录已完成工作、当前系统状态、剩余问题和执行计划。
>
> 日期: 2026-06-04
> 基于: v1 诊断 + Batch 1/2 实施 + pipeline 端到端验证 + MOC 审阅实战

---

## 1. 15 个差距当前状态

### 完成 (7/15)

| ID | 差距 | 产出物 |
|---|------|--------|
| G1 | AGENTS.md 瘦身 | 653→199 行; `docs/rules/` 6 个 Tier 2 文件 (sources/notes/kb/moc/review/checks) |
| G2 | MOC 规则 | §11 宪法 5 原则 + R1-R8 具体规则 + 运维规则 |
| E2 | 审阅独立化 | paperwiki-reader Step 3.5 改为 subagent dispatch; paperwiki-reviewer 3 模式 |
| V1 | KB 更新审阅 | `_templates/kb-review-checklist.md`; pipeline 第 6 步 |
| V2 | MOC 审阅 | `_templates/moc-review-checklist.md` (5 原则 + 17 检查项); 零样本语音合成已审阅并修复 |
| V3 | Lint 扩展 | `scripts/lint.py` L1-L9 (含 L9 vault stats 自动更新); `--fix` 模式 |
| — | MOC 修复 | 零样本语音合成拆为 3 个 sub-MOC; 全部 9 个 MOC 补 scope 声明; 去重完成 |

### 部分完成 (3/15)

| ID | 差距 | 已做 | 未做 |
|---|------|------|------|
| G3 | 三层治理扩展 | 笔记✅ KB✅ MOC✅ | 实体页内容审阅❌ |
| G4 | 约束升级为脚本 | lint L1-L9 覆盖 9 项 | checks.md 5 项人工检查未脚本化 |
| N1 | 熵管理 | 分散在 lint(去重/死链) + MOC 审阅(过载/腐烂) | 无独立熵管理机制(当前规模不需要) |

### 未做 (5/15)

| ID | 差距 | 原因 | 触发条件 |
|---|------|------|----------|
| E1 | Pipeline 多 agent | 当前 pipeline 能跑,规模未到瓶颈 | vault ≥ 400 篇或上下文质量明显下降 |
| E3 | MOC Agent 独立 | sub-MOC 未到阈值 | 任一 sub-MOC ≥ 50 篇或 MOC 刷新出错 ≥ 2 次 |
| V4 | 审阅拆分计算/推理 | 依赖 L1 识别哪些检查项该抽离 | L1 产出后 |
| L1 | 闭环机制 | 新规则下报告仅 1 份,数据不足 | 新规则下报告 ≥ 30 |
| L2 | 闭环质量信号 | 依赖 L1 | L1 跑过 2 轮 |

**M1 (度量)**: L9 vault stats 已部分覆盖,完整度量待闭环稳定后再建。

---

## 2. 当前系统架构

### 文件结构

```
AGENTS.md (Tier 1, 199 行)       ← 系统总纲,所有 agent 加载
docs/rules/ (Tier 2, 6 文件)     ← 按角色加载
_templates/ (3 个审阅 checklist)  ← review / moc-review / kb-review
scripts/lint.py (L1-L9)          ← 自动化检查
```

### Pipeline (8 步)

```
① 读原文 PDF
② KB 检索(confirmed 实体页)
③ 生成笔记草稿
④ 笔记审阅(独立 subagent dispatch)
⑤ Git commit 草稿 + 审阅报告
⑥ KB 更新审阅(独立 subagent,审阅变更计划)
⑦ 反向更新(仅 KB 审阅通过后)
⑧ 自动检查(lint.py) + Git commit + log
```

### 审阅系统 (3 模式)

| 模式 | checklist | 触发 | 结论 |
|------|-----------|------|------|
| 笔记审阅 | review-checklist.md (5 原则) | 每次精读自动 | pass / pass-with-fixes / revise / reject-as-deep |
| KB 更新审阅 | kb-review-checklist.md (3 原则) | 每次反向更新前自动 | pass / pass-with-fixes / block |
| MOC 审阅 | moc-review-checklist.md (5 原则 + 17 检查项) | 手动触发 | pass / pass-with-fixes / revise / restructure |

### Lint (9 项)

L1 frontmatter schema / L2 年份一致 / L3 MOC 重复 / L4 概念去重 / L5 MOC 覆盖 / L6 审阅 callout / L7 PDF 存在 / L8 死链 / L9 vault stats

### 信任机制

**当前: 人审批制。** 实体页创建后 status=pending-review,等人 batch review 确认。71 个积压。

### 关键指标

| 指标 | 数值 |
|------|------|
| 论文笔记 | 257 篇 (242 deep + 2 repro + 2 enhanced-card + 11 card) |
| 实体页 | 93 个 (22 confirmed / 71 pending-review) |
| MOC | 9 个 (6 主题 + 3 sub-MOC) |
| 审阅报告 | 173 个 (新规则下仅 1 个) |
| 死链 | 0 |

---

## 3. 剩余问题 (5 个)

### P1. 自动晋升缺失

**现状**: 实体页 pending-review 永远等人 batch review。71 个积压。
**根因**: AGENTS.md 原则 1 要求"AI 生成内容未经人类确认前不得进入可信层"。
**方案**: 信任机制转型 — reviewer pass + lint pass → 自动 confirmed。人从"必经审批者"变为"偶尔介入的校准者"。
**改动**: AGENTS.md 原则 1 + paperwiki-reader 反向更新逻辑。

### P2. 人工检查项未脚本化

**现状**: checks.md 5 项标注"人工检查" — 孤儿页、概念过时(>3 月未更新)、审阅积压、可信层进度、log 完整性。
**根因**: 和 CLAUDE.md 统计一样,靠人记得跑就会被忽略。
**方案**: 加入 lint.py (L10-L14),full-check 时自动执行。
**改动**: scripts/lint.py + docs/rules/checks.md。

### P3. 实体页无内容审阅

**现状**: 概念/模型/任务/数据集页被多篇论文反复追加后,整体质量从无检查。G3 的最后一块。
**根因**: 升级报告标 P3 推迟,但信任机制转型(P1)后自动晋升的前提是审阅覆盖完整。
**方案**: 新建 entity-review-checklist.md + paperwiki-reviewer 第 4 模式。定期或按积累量触发。
**改动**: _templates/entity-review-checklist.md + paperwiki-reviewer skill + docs/rules/review.md。

### P4. MOC 审阅无自动触发

**现状**: 论文数逼近 50 阈值无人知道,靠人记得说"审阅 MOC"。
**方案**: lint full-check 时检测每个 MOC 论文数,>40 输出 warning,>50 输出 error。
**改动**: scripts/lint.py 新增检测项。

### P5. 审阅 learning_signals 无消化通道

**现状**: 三种审阅产出的 learning_signals(new_issue_type / rule_gap / checklist_upgrade_suggestion)沉睡在 _review/*.yml 中。零样本语音合成 MOC 审阅已产出 3 条建议,未被处理。
**方案**: lint 或 pattern analysis 扫描所有审阅报告 → 汇总未处理建议 → 输出待执行清单。
**改动**: scripts/lint.py + docs/rules/checks.md。

---

## 4. 执行计划

### 第一波: 自动化补全 (现在可做)

**目标**: 消除所有强制性人工介入,系统转为"自动门 + 人校准"模式。

| 项 | 做什么 | 涉及文件 |
|---|--------|----------|
| P1 | 信任机制转型: 自动门通过即 confirmed | AGENTS.md, paperwiki-reader skill |
| P2 | 5 项人工检查脚本化 (L10-L14) | lint.py, checks.md |
| P3 | 实体页内容审阅 | entity-review-checklist.md(新建), paperwiki-reviewer skill, review.md |
| P4 | MOC 论文数阈值检测 | lint.py, moc.md |
| P5 | learning_signals 消化机制 | lint.py, checks.md |

**验证**: 精读一篇论文,确认自动晋升 + lint 全量通过 + 无人工介入。

### 第二波: 闭环 (新规则下 30+ 报告后)

**触发检查**:
```bash
find _review -name "*-review.yml" -newer _review/moc-review-零样本语音合成-2026-06-04.yml | wc -l
```

| 项 | 做什么 | 来源 |
|---|--------|------|
| L1 | pattern analysis: 高频问题 → 根因归属 → 规则修改建议 | 升级报告 L1 |
| V4 | 从 checklist 抽离计算性检查项到 lint | L1 产出指导 |
| L2 | 闭环质量信号: 同类问题频率趋势 / 新问题出现率 / 规则更新频率 | 升级报告 L2 |

### 第三波: 规模优化 (按需)

| 项 | 触发条件 | 做什么 |
|---|----------|--------|
| E3 | sub-MOC ≥ 50 篇或 MOC 刷新出错 ≥ 2 次 | MOC Agent 独立化 |
| E1 | vault ≥ 400 篇或上下文质量明显下降 | pipeline 多 agent 分阶段 |
| M1 | 闭环稳定运行 | 系统级度量(问题逃逸率/规则有效率/闭环周期) |

---

## 5. 触发条件速查

| 事件 | 检查命令 | 阈值 | 动作 |
|------|----------|------|------|
| 新规则报告数 | `find _review -name "*-review.yml" -newer _review/moc-review-零样本语音合成-2026-06-04.yml \| wc -l` | ≥ 30 | 启动第二波 (L1) |
| MOC 论文数 | `grep -c "^\- \[\[论文笔记/" _MOC/xxx.md` | > 40 warn, > 50 error | 拆分 sub-MOC |
| sub-MOC 论文数 | 同上 | ≥ 50 | 评估 E3 |
| vault 总规模 | `ls 论文笔记/*.md \| wc -l` | ≥ 400 | 评估 E1 |
| 闭环轮次 | `ls _review/pattern-analysis-*.md \| wc -l` | ≥ 2 | 启动 L2, 评估 M1 |

---

## 6. 与 v1 的变化

| 方面 | v1 (诊断时) | v2 (当前) |
|------|------------|-----------|
| 成熟度 | Level 1.5 | **Level 2.5** |
| AGENTS.md | 653 行百科全书 | 199 行 TOC + 6 个 Tier 2 |
| 审阅 | 1 种(笔记,内联) | 3 种(笔记/KB/MOC,独立 subagent) |
| Lint | 仅死链 | 9 项自动化 (L1-L9) |
| Pipeline | 7 步,审阅内联 | 8 步,审阅+KB审阅独立 |
| MOC | 7 行规则 | 宪法 5 原则 + R1-R8 + 审阅 + 已实战 |
| 信任机制 | 人审批制 | 人审批制(转型方案已设计,待第一波实施) |
| 闭环 | 无 | 无(数据积累中,第二波启动) |
| vault | 105 篇笔记 | 257 篇笔记 |
