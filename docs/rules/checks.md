# 系统检查规则

> Tier 2 规则文件 | 加载者: lint agent, system check
> 父文档: AGENTS.md §12
> 更新: 2026-06-04

---

## Per-ingest 自动检查 (pipeline 最后一步)

每次精读/复现完成后自动执行。使用 lint 脚本:

```bash
python3 scripts/lint.py --per-ingest "论文笔记/{笔记名}.md"
```

检查项: L1(frontmatter schema) + L6(审阅 callout) + L7(PDF 存在性) + L8(wikilink 目标)

另外手动确认:
- [ ] 本篇涉及的 MOC 已更新(论文 tags 对应的 MOC 包含本篇)
- [ ] 实体页引用存在
- [ ] log.md 已更新
- [ ] Git commit 已完成

有遗漏当场补完,不留到以后。

## 全面系统检查 (用户说"系统检查" / 每周 / 批量操作后)

使用 lint 脚本执行全量检查:

```bash
python3 scripts/lint.py
```

18 项自动化检查:
- L1: Frontmatter 字段存在性验证
- L2: Frontmatter year vs MOC section year 一致性
- L3: 同 MOC 内重复 wikilink 检测
- L4: 概念页 title/aliases 重叠检测
- L5: Deep/repro 笔记 MOC 覆盖率
- L6: 审阅 callout 存在性
- L7: source 字段指向的 PDF 存在性
- L8: Wikilink 目标文件存在性
- L9: CLAUDE.md vault 状态自动检查(--fix 可自动更新)
- L10: 孤儿页 — 没有入链的实体页
- L11: 概念过时 — active 概念页超过 90 天未更新
- L12: 审阅积压 — pending-review ≥ 10 或 draft deep/repro ≥ 5
- L13: 可信层进度 — confirmed 占 active 实体页比例(信息输出,不报 error)
- L14: Log 完整 — 论文笔记数 vs log.md ingest 条目数差异 > 5
- L15: MOC 容量 — MOC 内论文引用数 > 40(接近 50 阈值)
- L16: Stale pending — pending-review 且 updated > 30 天(提醒批量审阅)
- L17: Learning signals 消化 — 审阅报告中未处理的建议汇总(info,不报 error)
- L18: key_papers 上限 — 实体页 key_papers > 12 条(应精简为奠基/代表/转折级)

按系统四条底线组织:

### 可导航
| 检查项 | Lint ID | 说明 |
|--------|---------|------|
| MOC 覆盖 | L5 | deep/repro 被至少一个 MOC 引用 |
| 死链 | L8 | wikilink 目标文件存在 |
| 孤儿页 | L10 | 实体页至少有一个入链 |
| MOC 容量 | L15 | MOC 内论文数不超 40(warning) |

### 可溯源
| 检查项 | Lint ID | 说明 |
|--------|---------|------|
| 审阅覆盖 | L6 | deep/repro 有 review callout |
| Frontmatter 一致性 | L1 | schema 验证 |
| 可信层进度 | L13 | confirmed/total 比例(info) |

### 不腐烂
| 检查项 | Lint ID | 说明 |
|--------|---------|------|
| 概念过时 | L11 | 90 天未更新的 active 概念页 |
| 概念去重 | L4 | title/aliases 重叠 |
| 审阅积压 | L12 | pending-review ≥ 10 或 draft deep ≥ 5 |
| Stale pending | L16 | pending-review 超 30 天(warning) |

### 系统完整性
| 检查项 | Lint ID | 说明 |
|--------|---------|------|
| CLAUDE.md 状态 | L9 | vault 数据快照(--fix 自动更新) |
| Log 完整 | L14 | ingest 条目数 ≈ 论文笔记数 |
| Learning signals | L17 | 审阅报告中未消化建议汇总(info) |

产出: `_lint/YYYY-MM-DD-system-check.md`

### learning_signals 消化流程

lint L17 自动扫描所有审阅报告 (`_review/*.yml`) 中的 learning_signals,输出未消化建议汇总。

```bash
python3 scripts/lint.py --check L17
```

处理步骤:
1. 运行上述命令获取汇总
2. 逐条评估:
   - **采纳** → 更新对应规则文件 (docs/rules/*.md) 或检查项,在审阅报告中标记 `digested: true`
   - **拒绝** → 在审阅报告中标记 `digested: rejected`,附理由
   - **延后** → 不标记,下次扫描仍会出现
3. 产出合并到系统检查报告 `_lint/YYYY-MM-DD-system-check.md`

## 深度分析 (手动触发,成本高)

| 检查项 | 说明 |
|--------|------|
| 矛盾检测 | 不同页面对同一概念的描述冲突 |
| Stale SOTA | 标记的 SOTA 结果已被超越 |
| Missing links | 语义相关但缺少 wikilink 的页面对 |

**规则**: 所有结果标记 `[疑似]`,永不自动修改内容。
