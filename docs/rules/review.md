# 审阅规则

> Tier 2 规则文件 | 加载者: reviewer
> 父文档: AGENTS.md 审阅系统 section
> 更新: 2026-06-04

---

## 单篇审阅

每次 deep/repro 笔记生成后、反向更新前自动触发。

- 输出: `_review/{论文名}-review.yml` (YAML) + 笔记末尾 `[!review]` callout
- 结论: pass(直接反向更新) / pass-with-fixes(建议修正,可放行) / revise(有 high 问题,修正后重审) / reject-as-deep(降为 enhanced-card)
- 原则: 只诊断不重写,不阻断 pipeline,每个问题有 location+detail+suggestion

**审阅标准**: `_templates/review-checklist.md`（论文笔记审阅）

## KB 更新审阅

每次反向更新前自动触发。审阅反向更新的变更计划。

- 输出: `_review/{论文名}-kb-review.yml`
- 结论: pass / pass-with-fixes / block(阻断反向更新)
- 原则: 不污染知识库,不膨胀实体数量,更新可溯源

**审阅标准**: `_templates/kb-review-checklist.md`（KB 更新审阅）

## MOC 审阅

定期批量触发。审阅 MOC 导航页的质量。

- 输出: `_review/moc-review-{MOC名}-YYYY-MM-DD.yml`
- 结论: pass / pass-with-fixes / revise / restructure
- 原则: 可导航/不混层/可溯源/不腐烂/不过载

**审阅标准**: `_templates/moc-review-checklist.md`（MOC 审阅）

## 实体页审阅

定期批量触发或用户手动触发(如"审阅这个概念页")。审阅实体页(概念/模型/任务/数据集)的内容质量。

- 输出: `_review/entity-review-{页名}-YYYY-MM-DD.yml`
- 结论: pass(可进 confirmed) / pass-with-fixes(修正后可晋升) / revise(需重构) / restructure(需拆分)
- 原则: 不混层/不过载/可溯源/可导航/不污染
- 自动晋升: pass → status 自动变为 confirmed

**审阅标准**: `_templates/entity-review-checklist.md`（实体页审阅）

## 跨篇模式分析

每 5-10 篇手动触发。统计高频问题 → 判断根因(模板/规则/skill/检索) → 产出系统升级建议。
输出: `_review/pattern-analysis-YYYY-MM-DD.md`

## 审阅独立性原则

- **生成者不审自己**: reviewer 必须是独立 subagent,不共享生成上下文
- reviewer 只接收: 生成的产出 + checklist + 相关 KB 页面
- reviewer 不接收: generation prompt、中间推理、PDF 原文(除非交叉验证数字)

**Tier 2 规则加载(按审阅模式):**
- Reviewer (笔记): review.md
- Reviewer (KB): review.md + kb.md
- Reviewer (MOC): review.md + moc.md
- Reviewer (实体页): review.md + kb.md
