# KB 规则 (实体页 + 检索)

> Tier 2 规则文件 | 加载者: reader, kb-reviewer
> 父文档: AGENTS.md §7 + §8 + §10
> 更新: 2026-06-04

---

## Append vs Substantive Update 规则

### Append（不改变 status）

| 操作 | 示例 |
|------|------|
| 向 `key_papers` 列表追加条目 | 新增一篇引用此概念的论文 |
| 向列表型 section 末尾追加一行 | `## 相关论文` 末尾加一条 |
| 向表格末尾追加一行 | 对比表新增一行 |

**判定标准**: 仅在列表/表格末尾追加，不修改已有内容。

**status 行为**: "保持当前 status 不变" — confirmed 的页面 append 后仍为 confirmed，pending-review 的页面 append 后仍为 pending-review。

**key_papers 上限守卫**: 追加 key_papers 前**必须**检查当前条数:
- 已达 12 条上限时,**跳过追加**,改为在正文"相关工作"段追加一行引用,并在 log 中记录 `[skip/update] [[页名]] — key_papers 已达上限(12),改追加到正文`。
- 已超过 20 条(遗留页面): 所有新论文一律追加到正文,不再追加 frontmatter key_papers。

**追加准入判断**: 追加 key_papers 前须判断论文对概念的贡献类型。仅"对概念有直接贡献"(提出/改进/实验验证该概念)的论文可追加;仅"下游使用"(在系统中使用了该概念但无新见解)的论文改追加到正文引用段。

### Substantive Update（→ 触发自动审阅）

| 操作 | 示例 |
|------|------|
| 修改 `## 定义` section | 重写概念定义 |
| 修改 `## 在TTS中的应用` 中已有文本 | 改写已有段落 |
| 修改 frontmatter 关系字段 | 改 `category`、`related_concepts`、`aliases`、`supersedes` |

**判定标准**: 修改已有 prose 或关系型 frontmatter。

**status 行为**: 实质性修改后 status → `pending-review`,等待定期实体页审阅或用户手动触发审阅。

---

## 可信层与自动晋升

### 自动晋升规则

AI 生成内容通过自动化质量门后自动进入可信层。人的纠正是系统校准信号,不是必经审批。

| 操作 | 质量门 | status 变更 |
|------|--------|-------------|
| 新建实体页 | entity-review pass + lint pass | → confirmed |
| Append 更新 | 不改 status(已有规则) | 保持不变 |
| Substantive 更新 | — | → pending-review(等待定期审阅) |
| 定期实体页审阅 | entity-review pass | pending-review → confirmed |
| 定期实体页审阅 | entity-review pass-with-fixes | 保持当前 status,标注待修正 |
| 定期实体页审阅 | entity-review revise/restructure (对 pending-review 页) | 保持 pending-review |
| 定期实体页审阅 | entity-review revise/restructure (对 confirmed 页) | **confirmed → pending-review** |

**自动晋升的唯一路径**: entity-review pass → auto-confirmed。
**降级路径**: confirmed 页面 entity-review 结论为 revise/restructure → 降级为 pending-review,log 记录 `[lifecycle/downgrade] [[页名]] — entity-review {结论}`。

### status 含义

| status | 含义 |
|--------|------|
| `confirmed` | 通过自动质量门或人工确认,参与 KB 检索 |
| `pending-review` | 自动审阅未通过,需修正; KB 检索可参考但标注 `[待确认]` |
| `reviewed` | 人工审核确认(仅笔记,最高可信) |
| `draft` | 初始草稿(仅笔记) |

### 人校准机制

人手动修改 confirmed 页面时,修改本身是最高优先级的学习信号:
- 系统记录 diff(什么被改了)
- 闭环(L1)分析时作为优先输入
- 不影响 status(人修改后仍为 confirmed)

---

## 实体页生命周期

### 状态

| lifecycle | 含义 |
|-----------|------|
| `active` | 当前有效，正常参与检索和 MOC |
| `deprecated` | 已过时，保留但标记 |
| `merged` | 已合并入另一页，本页变为重定向 |

### Merge 操作（仅用户可执行）

1. 用户指令: "把 [[A]] 合并到 [[B]]"
2. Agent 执行:
   - 将 A 的独有内容迁移至 B
   - 将 A 的 title 加入 B 的 `aliases`（保证旧名可被 KB 检索匹配）
   - `grep -rl "[[A]]"` 全 vault 替换为 `[[B]]`
   - **删除 A.md**（不留 redirect 文件）
3. Commit: `[lifecycle/merge] [[A]] → [[B]]`

### 概念库的定位

概念库是**论文之间的连接节点 + KB 检索的知识缓存层**。主要服务"懂"(带着已有知识读新论文)和"连"(把论文连成网)。

### 概念页准入规则

一个东西值得独立成页,**必须满足至少两条**:
1. **会被多篇论文引用** — 是共享节点,不是单篇论文的一次性术语
2. **知道它能帮理解下一篇论文** — 是前置知识
3. **它连接不同论文** — 多篇论文通过它产生关联

不满足的 → 内嵌到相关概念页的某一节,或放在综述笔记/MOC 中。

### 概念页类型(用 category 字段区分)

| category | 含义 | KB 检索命中频率 | 价值方式 |
|---|---|---|---|
| technique | 可复用方法 | 高 | 自动辅助"懂" |
| component | 构建模块 | 高 | 自动辅助"懂" |
| model-family | 系统范式 | 高 | 辅助"懂"+"连" |
| representation | 数据表示 | 中 | 辅助"懂" |
| problem | 领域问题 | 中 | 辅助"懂" |
| design-choice | 路线对比/取舍 | 低 | 浏览时辅助"连" |
| taxonomy | 分类框架 | 低 | 浏览时辅助"连" |
| trade-off | 设计权衡 | 低 | 浏览时辅助"连" |

所有类型共存于概念库/,不分目录。低频命中类型的价值在主动浏览时兑现,不在自动检索时。

### 创建新概念前的去重检查

创建新概念页之前**必须**:
1. 搜索 概念库/ 中所有页面的 `title` 和 `aliases`
2. 如果找到语义相近的已有页 → 追加到那个页 + 把新术语加入 aliases
3. 如果不确定是否同一概念 → 先追加到最相近的页（拆比合容易）
4. 确实没有且满足准入规则 → 才新建

## 概念页结构规则

### 职责限定

概念页只保留:
1. 定义(是什么)
2. 重要性(为什么重要)
3. 核心机制/分类(主流技术路线)
4. 与相邻概念的边界(怎么区分)
5. 少量代表工作(≤ 12 key_papers)
6. 指向更细页面的链接

不允许:
- 无限追加"逐论文摘要"(>3 行的论文专属段落 ≤ 3 个)
- 兼做 topic page / mini-MOC / 研究综述
- 包含"研究演进时间线"(属于 MOC)

### key_papers 规则

- 硬上限: ≤ 12 条
- 必须是"奠基/代表/转折"级文献
- 格式统一: 全部使用 `[[论文笔记/xxx|显示名]]`
- **禁止 plain text 条目**: 所有 key_papers 必须是指向已有笔记的 wikilink,不可使用纯文本论文名
- 超出限额 → 多余的放到正文"相关工作"段或 MOC

### 强判断来源化

以下触发词要求来源标注:
- 首个 / 首次 / 开创 / 最优 / 全面超越 / 代表了 / 核心 / 主流 / 关键发现

改写为:
- "X 论文声称……" / "Survey Y 指出……" / "在作者报告的实验中……"

### 结构阈值

| 指标 | 阈值 | 超出处理 |
|------|------|----------|
| 一级标题数 | ≤ 6 | 拆分子概念页 |
| 论文专属段落(>3行) | ≤ 3 | 精简到 1-2 行或移除 |
| 页面行数(不含 frontmatter) | ≤ 200 | 拆分 |
| key_papers | ≤ 12 | 精简 |

### frontmatter 规范

- `origin_paper`: "触发创建本页的来源论文",非必填(大概念可空)
- `category`: 受控枚举 — technique / component / model-family / representation / problem / design-choice / taxonomy / trade-off
- `aliases`: 仅限同一概念的不同名称,不含近义概念/子概念/相关任务

---

### Deprecate 操作（仅用户可执行）

1. 用户指令: "废弃 [[A]]"
2. Agent 执行:
   - A 的 frontmatter 设置 `lifecycle: deprecated`
   - 在相关 MOC 中将 A 移至 `## 历史参考` section
3. Commit: `[lifecycle/deprecate] [[A]]`

---

## KB 检索规则

### 触发条件

仅在 **deep** 或 **repro** 层级 ingest 时触发 KB 检索。card/enhanced-card 不触发。

### 匹配条件（满足任一即命中）

1. 关键词匹配实体页 title 或 aliases
2. tags 交集 ≥ 2
3. 论文 frontmatter 中 `concepts` / `models` / `tasks` 字段直接引用实体

### 过滤规则

- 只检索 `lifecycle: active` 的实体页
- `status: confirmed` → 作为主要参考
- `status: pending-review` → 作为参考，标注 `[待确认]`

### 排序规则

优先级: direct-ref > tag match > alias match，同级按 `updated` 日期降序。

命中数 > 6 时取 Top 6。

### 日志格式

```
[kb/search] [[paper]] — 命中 N 页(取 Top M): [[A]]✓, [[B]]✓ | 过滤: [[C]](pending-review)
```

### 笔记 footer

在生成的笔记末尾附加：

```
检索命中: [[A]], [[B]] | 过滤: [[C]](pending-review) | 未命中但可能相关: 无
```

### 降级策略

| 情况 | 处理 |
|------|------|
| 无匹配 | 标注 "未找到相关知识库背景" |
| 仅有 pending-review 命中 | 标注 "基于未确认概念页，仅供参考" |
| 命中页 >3 月未更新 | 标注 "相关概念页可能过时" |

### Alias 反馈机制

用户 review 时若发现遗漏的概念关联，可为实体页添加 alias → 未来检索自动改善。
