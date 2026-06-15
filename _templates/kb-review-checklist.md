# KB 更新审阅标准

> 原则是宪法,检查项是法律,案例是判例法。
> 原则不变,检查项随系统演进更新。
>
> 本 checklist 用于反向更新(概念库/模型库/数据集/任务库的增量修改)的质量审阅。
> 论文笔记审阅使用 `review-checklist.md`,MOC 审阅使用 `moc-review-checklist.md`。

---

## 第一层: 评估原则 (稳定,极少变动)

从知识库下游使用场景推导。反向更新的错误会通过 KB 检索放大到后续所有精读,是系统中唯一"错误扩散"的环节。

| 原则 | 使用场景 | 核心问题 |
|------|----------|----------|
| **不污染** | 下次精读时 KB 检索命中这些更新内容 | 这些更新进入知识库后,会让未来的 KB 背景更好还是更差? |
| **不膨胀** | 浏览概念库寻找相关概念 | 新建实体页真的需要独立存在吗?还是可以合并到已有页? |
| **可溯源** | 引用概念页内容写报告或做决策 | 每条更新的内容都能追溯到论文原文吗? |

---

## 第二层: 当前检查项 (v1, 动态更新)

每条检查项标注它服务的原则。检查项会随系统架构演进而增删。

### 不污染

- [ ] 反向更新内容无 factual-error(数字/名称/指标与论文原文一致)
- [ ] 反向更新内容无 overclaim(未将 agent 推测写成论文结论)
- [ ] 已有概念页的定义/核心描述未被实质性改写(除非笔记审阅明确建议)

### 不膨胀

- [ ] 新建实体页满足准入规则(多篇引用/前置知识/连接论文,至少 2 of 3)
- [ ] 无语义相近的已有页(搜索 title + aliases 确认)
- [ ] 反向更新正确分类: append(加 key_papers / 末尾追加)不改 status; substantive(重写定义/改关系) → pending-review

### 可溯源

- [ ] 每条追加到实体页的描述含论文 wikilink 或 section 引用
- [ ] 新建实体页的 key_papers 包含触发创建的论文

---

## 问题词汇表 (命名用,不是 checklist)

审阅发现问题时用以下类型命名,便于跨篇统计:

| type | 含义 |
|------|------|
| factual-error | 更新内容与论文原文事实不符 |
| overclaim | 更新内容超出论文证据 |
| unauthorized-rewrite | 非必要的定义/核心描述重写 |
| admission-violation | 新实体页不满足准入规则 |
| dedup-miss | 存在语义相近的已有页未合并 |
| classification-error | append/substantive 分类错误 |
| missing-provenance | 更新缺少论文来源标注 |

**如果发现新类型的问题(不在此表中),直接命名并记录。** 这个表会在模式分析时扩展。

---

## 严重度

| 级别 | 含义 | 行动 |
|------|------|------|
| high | 会污染知识库或扩散错误(factual-error/overclaim/dedup-miss) | 阻断反向更新 |
| medium | 影响质量但不直接扩散(classification-error/missing-provenance) | 建议修正,可标注后放行 |
| low | 可读性/规范性 | 可忽略或后续批量改善 |

---

## 审阅结论

| 结论 | 判定条件 |
|------|----------|
| pass | 所有原则基本满足,无 high issue |
| pass-with-fixes | 有 medium issue 但不直接污染知识库 |
| block | ≥ 1 个 high issue,阻断本次反向更新 |

block 结论时: 跳过反向更新,记录 `[skip/update]` 到 log.md,附带具体 issue 供用户后续手动处理。

---

## 审阅报告格式

输出到 `_review/{论文名}-kb-review.yml`:

```yaml
paper: "论文名"
reviewer: agent
date: "YYYY-MM-DD"
conclusion: pass | pass-with-fixes | block

principles:
  no_pollution:
    verdict: pass | pass-with-fixes | block
    notes: ""
  no_bloat:
    verdict: pass | pass-with-fixes | block
    notes: ""
  traceable:
    verdict: pass | pass-with-fixes | block
    notes: ""

change_plan_summary:
  total_targets: N
  appends: N
  substantive: N
  new_pages: N

issues:
  - severity: high | medium | low
    type: (问题词汇表中的类型)
    principle: (服务哪条原则)
    target: "概念库/X.md"
    action: append | substantive | new
    detail: "问题描述"
    suggestion: "修正建议"

learning_signals:
  new_issue_type: ""
  rule_gap: ""
  checklist_upgrade_suggestion: ""
```

---

## 检查项更新记录

| 日期 | 变更 | 来源 |
|------|------|------|
| 2026-06-04 | v1 初始检查项 | Harness 升级报告 V1 + AGENTS.md §7 append/substantive 规则 |
| 2026-06-09 | v1.1: 对应 kb.md 规则更新 — key_papers >20 改追加正文; append 准入判断(直接贡献 vs 下游使用); 禁止 plain text 条目 | 51 条 learning signals 批量消化 |

*每次模式分析更新检查项时,在此表追加记录。*
