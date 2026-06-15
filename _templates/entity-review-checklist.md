# 实体页审阅标准

> 原则是宪法,检查项是法律,案例是判例法。
> 原则不变,检查项随系统演进更新。
>
> 本 checklist 用于实体页(概念/模型/任务/数据集)的内容质量审阅。
> 论文笔记审阅使用 `review-checklist.md`,KB 更新审阅使用 `kb-review-checklist.md`,MOC 审阅使用 `moc-review-checklist.md`。

---

## 第一层: 评估原则 (稳定,极少变动)

从实体页的下游使用场景推导。实体页是 KB 检索的数据源,质量问题会通过检索扩散到后续所有精读。

| 原则 | 使用场景 | 核心问题 |
|------|----------|----------|
| **不混层** | 浏览概念库寻找某概念的定义和边界 | 页面内容全部属于"概念定义+机制+边界"层,还是混入了论文级/survey 级/MOC 级内容? |
| **不过载** | 快速提取核心定义和关键路线 | 用户能在 2 分钟内提取核心信息吗?还是页面已变成持续堆积的研究池? |
| **可溯源** | 引用概念页内容做判断或写报告 | 页面中的强判断("首个"/"最优"/"核心")是否都有来源锚点? |
| **可导航** | 通过概念页找到核心代表工作 | key_papers 和链接能帮助区分核心/代表/边缘吗?还是信号已被稀释? |
| **不污染** | 下次精读时 KB 检索命中此页 | 页面作为 KB 检索背景时,会给下游笔记提供准确定位还是引入偏见? |

---

## 第二层: 当前检查项 (v1, 动态更新)

每条检查项标注它服务的原则。检查项会随系统架构演进而增删。

### 不混层
- [ ] 页面不含逐论文长段摘要(>3 行的论文专属段落 ≤ 3 个)
- [ ] 页面职责单一(概念页不兼做 topic page / mini-MOC / survey)
- [ ] 无"研究演进"时间线式内容(属于 MOC)
- [ ] 模型级/系统级细节归入模型页或论文笔记,不在概念页展开

### 不过载
- [ ] key_papers ≤ 12 条(奠基/代表/转折级)
- [ ] 一级标题 ≤ 6 个
- [ ] 页面总行数 ≤ 200 行(不含 frontmatter)
- [ ] 超出阈值的内容有明确拆分去向(子概念页/taxonomy 页/MOC)

### 可溯源
- [ ] 强判断触发词("首个/首次/开创/最优/全面超越/代表了/核心/关键发现")有来源标注
- [ ] 来源标注格式规范("X 论文声称/Survey Y 指出/在…实验中")
- [ ] 无方向性错误(指标方向、时间先后、因果关系)
- [ ] 无将 agent 推测写成事实断言的句子

### 可导航
- [ ] key_papers 仅含奠基/代表/转折级文献(非"相关论文全集")
- [ ] key_papers 格式统一(全部使用 `[[论文笔记/xxx|显示名]]`,禁止 plain text 条目)
- [ ] key_papers 中每篇论文对概念有直接贡献(提出/改进/验证),非仅"下游使用"
- [ ] frontmatter key_papers 与 body 关键论文 section 一致(frontmatter 中列的论文在 body 中有至少一句关系说明)
- [ ] related_concepts 与本页有明确语义区分(不是子概念/同义词)
- [ ] 链接密度适中(不因过多链接导致信号稀释)
- [ ] 追加表格行时字段与列标题语义一致

### 不污染
- [ ] 定义段不含未验证的综合性结论
- [ ] aliases 仅含同一概念的不同名称(不含近义概念/子概念/相关任务)
- [ ] origin_paper 语义明确(触发创建本页的来源论文,大概念可空)
- [ ] category 使用受控枚举(technique/component/model-family/representation/problem/design-choice/taxonomy/trade-off)

---

## 问题词汇表 (命名用,不是 checklist)

| type | 含义 |
|------|------|
| scope-drift | 概念页职责漂移(变 survey/topic/MOC) |
| overload | 内容过载(key_papers/行数/结构超限) |
| unsourced-claim | 强判断无来源锚点 |
| signal-dilution | key_papers/链接噪声导致导航失效 |
| layer-mixing | 论文级细节混入概念层 |
| boundary-blur | 概念边界不清(与相邻概念未区分) |
| frontmatter-drift | frontmatter 字段语义不稳定(category/aliases/origin_paper) |
| frontmatter-body-desync | frontmatter key_papers 与 body 关键论文严重不一致 |
| key-papers-inflation | key_papers 退化为全量引用列表,丧失导航功能 |
| column-mismatch | 追加表格行时字段与列定义不匹配 |

**如果发现新类型的问题(不在此表中),直接命名并记录。** 这个表会在模式分析时扩展。

---

## 严重度

| 级别 | 含义 | 行动 |
|------|------|------|
| high | 影响概念页作为 KB 数据源的可靠性(污染下游) | 不能进 confirmed,需重构 |
| medium | 影响质量但不直接污染下游 | 修正后可晋升 |
| low | 可读性/规范性 | 可后续批量改善 |

---

## 审阅结论

| 结论 | 判定条件 |
|------|----------|
| pass | 所有原则基本满足,无 high issue → 可自动晋升 confirmed |
| pass-with-fixes | 有 medium issue,修正后可晋升 |
| revise | ≥ 1 个 high issue,需内容重构 |
| restructure | ≥ 2 个原则明显不满足,需拆分/合并/降级 |

---

## 审阅报告格式

输出到 `_review/entity-review-{页名}-YYYY-MM-DD.yml`:

```yaml
entity: "页名"
file: "概念库/xxx.md"
type: concept | model | task | dataset
reviewer: agent
date: "YYYY-MM-DD"
conclusion: pass | pass-with-fixes | revise | restructure

principles:
  no_layer_mixing:
    verdict: pass | pass-with-fixes | revise
    notes: ""
  no_overload:
    verdict: pass | pass-with-fixes | revise
    notes: ""
  traceable:
    verdict: pass | pass-with-fixes | revise
    notes: ""
  navigable:
    verdict: pass | pass-with-fixes | revise
    notes: ""
  no_pollution:
    verdict: pass | pass-with-fixes | revise
    notes: ""

structure_metrics:
  key_papers_count: N
  heading_count: N
  line_count: N
  long_paper_paragraphs: N

issues:
  - severity: high | medium | low
    type: (问题词汇表中的类型)
    principle: (服务哪条原则)
    location: "具体位置(section/行)"
    detail: "问题描述"
    suggestion: "最小修正建议"

restructure_plan:
  split_targets: []
  merge_into: ""
  demote_to: ""

learning_signals:
  new_issue_type: ""
  rule_gap: ""
  checklist_upgrade_suggestion: ""
```

---

## 检查项更新记录

| 日期 | 变更 | 来源 |
|------|------|------|
| 2026-06-04 | v1 初始检查项 | 用户三页概念审阅案例(Emotion Control/Speech Tokenizer/LLM-based TTS) + harness 升级设计 |
| 2026-06-09 | v1.1: +key_papers 准入质量检查(直接贡献 vs 下游使用); +frontmatter-body 一致性; +禁止 plain text 条目; +表格列一致性; +3 新 issue types (frontmatter-body-desync, key-papers-inflation, column-mismatch) | 51 条 learning signals 批量消化 |

*每次模式分析更新检查项时,在此表追加记录。*
