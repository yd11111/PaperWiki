# MOC 审阅标准

> 原则是宪法,检查项是法律,案例是判例法。
> 原则不变,检查项随系统演进更新。
>
> 本 checklist 用于 MOC 导航页的质量审阅,与论文笔记审阅 (`review-checklist.md`) 互为配套。

---

## 第一层: 评估原则 (稳定,极少变动)

从用户导航场景推导,不从文档属性推导。

| 原则 | 使用场景 | 核心问题 |
|------|----------|----------|
| **可导航** | 新人第一次打开这个 MOC | 3 秒内能理解页面结构并找到入口吗? |
| **不混层** | 浏览"核心概念"区 | 同一 section 内的条目是同一抽象层级吗? |
| **可溯源** | 引用 MOC 中的判断写报告 | 每个判断性描述(SOTA/首创/最优)有标注来源吗? |
| **不腐烂** | 3 个月后再看这个 MOC | 年份、条目、演进关系与当前 vault 状态一致吗? |
| **不过载** | 找某条路线的代表工作 | 信息密度在可维护范围内吗?还是已变成论文大全? |

---

## 第二层: 当前检查项 (v1, 动态更新)

每条检查项标注它服务的原则和对应的 AGENTS.md 规则编号。

### 可导航

- [ ] MOC 文件包含 scope 声明(注释或 frontmatter),说明覆盖什么、不包含什么 `[R1]`
- [ ] MOC 身份明确: 是导航页(不是任务定义页),与同名任务页(如有)通过 wikilink 互引 `[R1]`
- [ ] 页面结构层次清晰: 核心概念 → 代表模型 → 相关论文,用户无需滚屏即可理解骨架 `[R1]`

### 不混层

- [ ] 核心概念区按抽象层级分组(范式/模块/技巧),不平铺 `[R2]`
- [ ] 代表模型区只含主线系统,backbone/vocoder/上游模型不与主线系统平级 `[R3]`
- [ ] 如有 backbone 或通用组件,放在独立子节("相关上游模型"/"关键组件") `[R3]`

### 可溯源

- [ ] 强判断词("首创/SOTA/唯一/全面优于/远超/开创/填补空白/最佳/最优")均有来源标注 `[R8]`
- [ ] 来源标注使用规定格式: `(作者 claim)` / `(benchmark: XX)` / `(agent 整理)` `[R8]`
- [ ] 演进脉络使用规定关系符号(`→/⇢/∥/⊃`),不同团队的工作未使用 `→` `[R7]`
- [ ] SOTA 表(如有)满足 R6 全部标注要求(benchmark/metric/数据来源/日期/可比性声明) `[R6]`

### 不腐烂

- [ ] 每条论文的年份 section 归属与该论文 frontmatter `year` 字段一致 `[R5]`
- [ ] 同一 MOC 内无重复 wikilink(同一论文/实体只出现一次) `[R5]`
- [ ] 所有 wikilink 指向存在的文件(无死链) `[通用]`
- [ ] 代表模型的描述(年份/机构/一句话)与对应实体页/笔记 frontmatter 一致 `[通用]`

### 不过载

- [ ] 论文区总条目数 ≤ 50;超过时已拆为 sub-MOC `[R4]`
- [ ] 每条论文描述 ≤ 1 行 `[R4]`
- [ ] 人工备注区块存在且未被覆盖 `[运维]`

---

## 问题词汇表 (命名用,不是 checklist)

审阅发现问题时用以下类型命名,便于跨篇统计:

| type | 含义 |
|------|------|
| layer-mixing | 不同抽象层级的条目平铺在同一 section |
| scope-violation | 收录了 scope 声明之外的内容 |
| granularity-inconsistency | 同一 section 内条目粒度不一致 |
| model-admission-error | 非主线系统放入代表模型区 |
| paper-overflow | 论文区超过容量阈值 |
| year-mismatch | 年份 section 与 frontmatter year 不一致 |
| duplicate-entry | 同一条目在同一 MOC 中出现多次 |
| sota-protocol-missing | SOTA 表缺少必要标注 |
| relation-type-error | 演进关系符号使用不当 |
| unattributed-judgment | 强判断无来源标注 |
| role-boundary-blur | MOC 内容侵入任务页职责 |
| dead-link | wikilink 指向不存在的文件 |
| stale-description | 描述与当前 vault 状态不一致 |

**如果发现新类型的问题(不在此表中),直接命名并记录。** 这个表会在模式分析时扩展。

---

## 严重度

| 级别 | 含义 | 行动 |
|------|------|------|
| high | 破坏导航或传播错误(死链/年份错挂/SOTA 误导) | 必须修正 |
| medium | 影响质量但不阻塞(混层/缺标注/粒度不一致) | 建议修正 |
| low | 可读性/规范性(描述过长/格式不统一) | 可忽略或批量改善 |

---

## 审阅结论

| 结论 | 判定条件 |
|------|----------|
| pass | 所有原则基本满足,无 high issue |
| pass-with-fixes | 有 medium issue 但不破坏导航 |
| revise | ≥ 1 个 high issue,需修正后重审 |
| restructure | ≥ 2 个原则明显不满足,需要结构性重组 |

---

## 审阅报告格式

输出到 `_review/moc-review-{MOC名}-YYYY-MM-DD.yml`:

```yaml
moc: "MOC名"
file: "_MOC/xxx.md"
date: "YYYY-MM-DD"
conclusion: pass | pass-with-fixes | revise | restructure

principles:
  navigable:
    score: N
    judgment: ""
  no_mixing:
    score: N
    judgment: ""
  traceable:
    score: N
    judgment: ""
  no_decay:
    score: N
    judgment: ""
  no_overload:
    score: N
    judgment: ""

stats:
  total_papers: N
  total_concepts: N
  total_models: N
  duplicate_entries: N
  year_mismatches: N
  dead_links: N
  unattributed_judgments: N

issues:
  - severity: high | medium | low
    type: (问题词汇表中的类型)
    principle: (服务哪条原则)
    location: "具体位置(section + 行内容)"
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
| 2026-06-04 | v1 初始检查项 | MOC 审阅反馈(15 类问题) + AGENTS.md §11 规则体系 |

*每次模式分析更新检查项时,在此表追加记录。*
