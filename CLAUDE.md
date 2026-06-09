# PaperWiki — TTS 方向个人知识库

本 vault 是一个 AI+人共同维护的 TTS 论文知识库。**操作前必须读 `AGENTS.md`(系统总纲)+ 按任务加载 `docs/rules/` 下的 Tier 2 规则文件。**

## Skill 使用声明 (严格遵守)

**⚠️ 论文精读必须使用 `paperwiki-reader` skill。绝对禁止使用通用 `paper-reader` skill。**

通用 paper-reader 的路径指向旧 vault `/Users/xiangshu/DailyPaper/DailyPaper/`,会把笔记写到错误位置。如果你看到系统自动选择了 paper-reader,必须拒绝并手动调用 paperwiki-reader。

| 用户说 | 必须用的 skill | 禁止用的 skill |
|---|---|---|
| "精读这篇"/"读一下"/"帮我读"/"深读" | **paperwiki-reader** | ~~paper-reader~~ |
| "审阅这篇笔记"/"审阅" | **paperwiki-reviewer** | — |
| "审阅 MOC"/"审阅这个 MOC" | **paperwiki-reviewer** (MOC mode) | — |
| "复现分析" | **paperwiki-reader** (repro mode) | ~~paper-reader~~ |

**如何确认用对了 skill:** 检查输出路径是 `/Users/xiangshu/PaperWiki/论文笔记/`,不是 `/Users/xiangshu/DailyPaper/`。

## 关键约束 (不可违反)

1. **不得自动升级笔记层级** — 只有用户明确说"精读/复现/升级"才能改 tier
2. **自动化质量门** — AI 生成内容通过 reviewer + lint 后自动进入可信层; 人的纠正是校准信号,不是必经审批
3. **一篇论文 = 一个文件** — 永远在 `论文笔记/` 下,DailyPapers/ 只是索引(title+link)
4. **失败不阻断** — KB 检索无结果就标注继续,反向更新失败就跳过记 log,不要中断流程
5. **所有网络访问用 Playwright** — WebFetch/WebSearch 不稳定,必须用 mcp__playwright 工具
6. **PDF 文本提取** — 优先用 `pdftotext` 预处理,fallback 到 Read 工具 OCR 模式

## 精读论文的完整流程

当用户说"精读这篇"/"读一下这篇论文"/"帮我读 [PDF/URL]"时,必须执行以下全部步骤:

```
① 读原文 PDF
② KB 检索（搜索 confirmed 实体页）
③ 生成笔记草稿（含速查卡片 + KB 背景）
④ 审阅（独立 subagent dispatch）: dispatch reviewer agent,输出 _review/xxx-review.yml
⑤ Git commit 草稿 + 审阅报告
⑥ KB 更新审阅（独立 subagent dispatch）: 审阅反向更新变更计划
⑦ 反向更新（仅 KB 审阅通过后）
⑧ 自动检查 (lint script) + Git commit + log
```

## 概念页更新规则

- **追加更新**(加 key_papers、末尾追加一行): 不改 status
- **实质修改**(重写定义、改 related_concepts): status → pending-review
- **用论文重写**: 用户说"用这篇重写 [[概念名]]" → 整体重写 → pending-review
- **搜索补充**: 用户说"搜索补充 [[概念名]]" → 用 Playwright 搜索 → 补充 → pending-review

## 每日推荐规则

- 只收当天 arXiv 发布的论文,HF trending ≠ 当天
- 当天无论文可以不生成,不可混入非当天论文
- Card 笔记在 论文笔记/ 下,DailyPapers/ 只是索引

## Commit 格式

`[type/subtype] 页面 — 描述`

Types: ingest|update|create|check|review|alert|skip|moc|kb|lifecycle|init

## 文件结构

```
AGENTS.md          ← 系统总纲 (Tier 1, ~200 行)
docs/rules/        ← Tier 2 详细规则 (sources/notes/kb/moc/review/checks)
docs/使用手册.md    ← 用户操作速查
_templates/        ← 页面模板 + 审阅标准 (review/moc-review/kb-review checklist)
scripts/           ← 自动化脚本 (lint.py)
论文笔记/          ← 所有论文笔记(4 层共存)
概念库/            ← 概念实体页(KB 检索数据源)
模型库/            ← 模型实体页
任务库/            ← 任务实体页
数据集/            ← 数据集实体页
_MOC/              ← 导航页(主视图)
DailyPapers/       ← 每日推荐索引
_inbox/            ← 零承诺缓冲
_lint/             ← 系统检查报告
_review/           ← 审阅报告 (笔记/KB/MOC)
log.md             ← 操作日志
Sources/           ← PDF(gitignored)
```

## 当前 vault 状态 (2026-06-09)

- 论文笔记: 529 篇 (513 deep + 3 repro + 2 enhanced-card + 11 card)
- 实体页: 96 个 (62 概念 + 20 模型 + 4 任务 + 10 数据集), 其中 22 confirmed / 74 pending-review
- 可信层: 22 confirmed 实体 + 2 reviewed 笔记
- MOC: 16 个
- MOC 覆盖: 516/516 deep/repro
- 审阅覆盖: 516/516 deep/repro 有 review callout
- 审阅报告: 560 个 (_review/*.yml)

