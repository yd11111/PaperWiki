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

8 项自动化检查:
- L1: Frontmatter 字段存在性验证
- L2: Frontmatter year vs MOC section year 一致性
- L3: 同 MOC 内重复 wikilink 检测
- L4: 概念页 title/aliases 重叠检测
- L5: Deep/repro 笔记 MOC 覆盖率
- L6: 审阅 callout 存在性
- L7: source 字段指向的 PDF 存在性
- L8: Wikilink 目标文件存在性

按系统四条底线补充人工检查:

### 可导航
| 检查项 | 说明 |
|--------|------|
| MOC 覆盖 | L5 自动化 |
| 死链 | L8 自动化 |
| 孤儿页 | 没有入链的实体页(人工检查) |

### 可溯源
| 检查项 | 说明 |
|--------|------|
| 审阅覆盖 | L6 自动化 |
| Frontmatter 一致性 | L1 自动化 |
| 可信层进度 | confirmed/reviewed 比例是否在增长(人工检查) |

### 不腐烂
| 检查项 | 说明 |
|--------|------|
| 概念过时 | active 概念页超过 3 月未更新(人工检查) |
| 概念去重 | L4 自动化 |
| 概念溯源 | key_papers ≥ 3 且 origin_paper 为空(人工检查) |
| 审阅积压 | pending-review ≥ 10 或 draft deep ≥ 5(人工检查) |

### 系统完整性
| 检查项 | 说明 |
|--------|------|
| CLAUDE.md 状态 | vault 数据快照是否反映当前(人工检查) |
| Log 完整 | ingest 条目数 ≈ 论文笔记数(人工检查) |

产出: `_lint/YYYY-MM-DD-system-check.md`

## 深度分析 (手动触发,成本高)

| 检查项 | 说明 |
|--------|------|
| 矛盾检测 | 不同页面对同一概念的描述冲突 |
| Stale SOTA | 标记的 SOTA 结果已被超越 |
| Missing links | 语义相关但缺少 wikilink 的页面对 |

**规则**: 所有结果标记 `[疑似]`,永不自动修改内容。
