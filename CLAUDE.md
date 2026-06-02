# PaperWiki — TTS 方向个人知识库

本 vault 是一个 AI+人共同维护的 TTS 论文知识库。**操作前必须读 `AGENTS.md` 了解完整规则。**

## Skill 使用声明

- **论文精读**: 使用 `paperwiki-reader` skill(项目专属),**不要**使用通用 `paper-reader` skill(路径指向旧 vault)
- **笔记审阅**: 使用 `paperwiki-reviewer` skill（精读后自动触发，或手动"审阅这篇笔记"）
- **其他 skill**: 按需使用,但输出路径必须在 `/Users/xiangshu/PaperWiki/` 内

## 关键约束 (不可违反)

1. **不得自动升级笔记层级** — 只有用户明确说"精读/复现/升级"才能改 tier
2. **AI 生成内容不得进入可信层** — 新建实体页 status=pending-review,笔记 status=draft,只有用户确认后才能变 confirmed/reviewed
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
④ 审阅（反向更新前）: 对照 checklist 检查,输出 _review/xxx-review.yml
⑤ Git commit 草稿 + 审阅报告
⑥ 反向更新（仅审阅通过后）
⑦ 局部 lint + Git commit + log
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

Types: ingest|update|create|lint|review|alert|skip|moc|kb|lifecycle|init

## 文件结构

```
AGENTS.md          ← 完整规则(必读)
docs/使用手册.md    ← 用户操作速查
_templates/        ← 页面模板(生成时参考)
论文笔记/          ← 所有论文笔记(4 层共存)
概念库/            ← 概念实体页(KB 检索数据源)
模型库/            ← 模型实体页
任务库/            ← 任务实体页
数据集/            ← 数据集实体页
_MOC/              ← 导航页(主视图)
DailyPapers/       ← 每日推荐索引
_inbox/            ← 零承诺缓冲
_lint/             ← lint 报告
log.md             ← 操作日志
Sources/           ← PDF(gitignored)
```

## 当前 vault 状态 (2026-06-01)

- 5 deep + 1 repro + 1 enhanced-card + 2 card
- 29 实体页(4 confirmed: CFM, Speech Tokenizer, RVQ, Zero-shot TTS)
- 3 MOCs
- Git tags: p0-foundation → p5-specification-ready
