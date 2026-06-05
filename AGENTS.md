# AGENTS.md — PaperWiki 系统规范

> 本文件是 Tier 1 系统总纲。详细规则按角色拆分到 `docs/rules/` 下的 Tier 2 文件。
> Agent 读本文件获得全局上下文,再按任务加载对应 Tier 2 文件。

---

## 1. 系统概览

**目的**: 以 TTS/语音合成为核心的个人知识库，AI agent 辅助人类进行论文阅读、概念整理与知识积累。

| 角色 | 职责 |
|------|------|
| 你（人类） | 决策者。确认知识可信度、触发升级、执行 merge/deprecate |
| LLM（Agent） | 执行者。论文摘录、笔记生成、概念页维护、系统检查、MOC 刷新 |
| 概念页 | 知识载体。实体化的概念/模型/数据集，是知识库的主视图单元 |

---

## 2. 目录语义

| 目录 | 用途 | 谁写入 | 是否入 git |
|------|------|--------|-----------|
| `论文笔记/` | 一篇论文一个文件，所有层级笔记 | Agent + 人类 | Yes |
| `概念库/` | 概念实体页 | Agent + 人类 | Yes |
| `模型库/` | 模型实体页 | Agent + 人类 | Yes |
| `数据集/` | 数据集实体页 | Agent + 人类 | Yes |
| `任务库/` | 任务实体页 | 人类 | Yes |
| `_MOC/` | Map of Content 导航页 | Agent | Yes |
| `DailyPapers/` | 每日论文推荐索引（仅 title + link） | Agent | Yes |
| `Sources/` | PDF 原文 | 人类/Agent | **No** (gitignored) |
| `_inbox/` | 零承诺收件箱 | 人类/Agent | Yes |
| `_templates/` | Obsidian 模板 | 人类 | Yes |
| `_lint/` | 系统检查报告 | Agent | Yes |
| `_review/` | 审阅报告 | Agent | Yes |
| `docs/` | 系统文档 | 人类 | Yes |
| `docs/rules/` | Tier 2 规则文件 | 人类 | Yes |
| `scripts/` | 自动化脚本(lint 等) | 人类 | Yes |
| `log.md` | Agent 操作日志 | Agent | Yes |

---

## 3. Sources/ 存储规则

PDF 存放于 Sources/,命名必须与论文笔记文件名一致。

**完整规则**: [docs/rules/sources.md](docs/rules/sources.md)

---

## 4-6. 笔记规则

一篇论文 = 一个文件,四级笔记体系(card → enhanced-card → deep → repro),只有用户可触发升级。

**完整规则**: [docs/rules/notes.md](docs/rules/notes.md)

---

## 5. 六大原则

| 原则 | 核心 |
|------|------|
| **可信知识层** | AI 生成内容通过自动化质量门(reviewer + lint)后自动进入可信层; 人的纠正是系统校准信号,不是必经审批 |
| **主视图 = 实体页 + MOC** | 论文笔记是输入端,实体页是沉淀端 |
| **Inbox 零承诺** | 放入不代表要处理,处理完即删 |
| **失败降级** | KB 无结果标注继续,反向更新失败跳过记 log,不阻断 |
| **默认不升级** | Agent 不得未经用户指令升级笔记层级 |
| **允许不完美** | 局部完整性优先,全局一致性逐步达成 |

---

## 7-8. KB 规则 (实体页 + 检索)

Append(不改 status) vs Substantive(→ pending-review)。实体页生命周期。概念页准入与去重。KB 检索触发条件与排序。

**完整规则**: [docs/rules/kb.md](docs/rules/kb.md)

---

## 9. Inbox 规则

- **零承诺**: 放入不代表处理承诺
- **最小字段**: `title` + `source` + `why`
- **处理后删除**
- **无时间压力**

---

## 11. MOC 规则

宪法 5 原则(可导航/不混层/可溯源/不腐烂/不过载) + R1-R8 具体规则 + 运维规则。

**完整规则**: [docs/rules/moc.md](docs/rules/moc.md)

---

## 12. 系统检查

自动化: `python3 scripts/lint.py`(18 项机械化检查,含孤儿页/概念过时/审阅积压/可信层进度/MOC 容量/key_papers 上限等)。

**完整规则**: [docs/rules/checks.md](docs/rules/checks.md)

---

## 审阅系统

四种审阅模式: 笔记审阅 / KB 更新审阅 / MOC 审阅 / 实体页审阅。独立 subagent 运行,两层评估。

自动晋升: entity-review pass → status 自动升为 confirmed(唯一路径)。

**完整规则**: [docs/rules/review.md](docs/rules/review.md)

---

## 13. Link 约定

- 所有跨页引用使用 `[[wikilink]]` 格式
- Frontmatter 中的链接使用 `"[[Page Name]]"` 格式（带引号）
- **同名文件消歧**: `[[论文笔记/CosyVoice3|CosyVoice 3]]`。实体页 `key_papers` 始终使用 `[[论文笔记/xxx|显示名]]`
- **核心链接(must-link)**: 实体页必须包含的关系链接
- **相关引用(optional)**: 可选的延伸链接

---

## 14. Commit 格式

```
[type/subtype] pages — description
```

| Type | 含义 | 常用 Subtype |
|------|------|-------------|
| `ingest` | 新论文入库 | deep, card, enhanced-card, repro |
| `update` | 更新已有页面 | concept, model |
| `create` | 创建新实体页 | concept |
| `check` | 系统检查 | auto, full |
| `review` | 审阅操作 | batch, auto, kb |
| `moc` | MOC 刷新 | — |
| `kb` | KB 检索 | search |
| `lifecycle` | 生命周期变更 | merge, deprecate |
| `skip` | 跳过操作 | update, ingest |
| `alert` | 告警 | backlog |
| `init` | 初始化 | — |

失败的 ingest 不产生 commit,仅在 log.md 记录 `[skip/ingest]`。

---

## 16. 失败降级总表

| 场景 | 降级策略 | 阻塞 |
|------|----------|------|
| KB 检索无匹配 | 标注 `[未找到相关知识库背景]` | 否 |
| 仅命中 pending-review | 标注 `[基于未确认概念页]` | 否 |
| 概念页过时(>3 月) | 标注 `[可能过时]` | 否 |
| 无法确定是否独立实体 | 记录 `[待决]`,不创建 | 否 |
| 反向更新失败 | 跳过 + 记录日志 | 否 |
| PDF 无法解析 | 记录 `[skip/ingest]` | 是(本次) |
| Wikilink 目标不存在 | 系统检查报告 | 否 |
| MOC 刷新失败 | 日志记录 | 否 |

---

## 17. Review Backlog 告警

| 条件 | 动作 |
|------|------|
| pending-review 实体页 ≥ 10 | `[alert/backlog]` 提醒 review |
| draft deep/repro ≥ 5 | `[alert/backlog]` 提醒确认 |

告警仅为提醒,不阻塞操作。

---

## 附: 快速参考

```
Agent 精读 pipeline:
① 读原文
② KB 检索
③ 生成笔记草稿 (status: draft)
④ 审阅 (独立 subagent,对照 checklist)
⑤ Git commit 草稿 + 审阅报告
⑥ KB 更新审阅 (独立 subagent,审阅变更计划)
⑦ 反向更新 (仅 KB 审阅通过后)
⑧ 自动检查 (lint script) + Git commit + log

Tier 2 规则加载:
- Reader: notes.md + kb.md + sources.md
- Reviewer (笔记): review.md
- Reviewer (KB): review.md + kb.md
- Reviewer (MOC): review.md + moc.md
- Reviewer (实体页): review.md + kb.md
- MOC Agent: moc.md
- System Check: checks.md

自动晋升:
- entity-review pass → status 自动升为 confirmed(唯一路径)
- 存量页: 批量审阅 → pass 则晋升, revise 则标注待重构

其他规则:
- 新概念出现 → 创建实体页(pending-review) → entity-review → pass 则 confirmed
- Append → 保持 status; Substantive → status=pending-review
- 每次 ingest 后 → 检查 MOC 是否需刷新
- 永不: 自行升级层级 / 自行 merge/deprecate
```
