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

### Substantive Update（→ pending-review）

| 操作 | 示例 |
|------|------|
| 修改 `## 定义` section | 重写概念定义 |
| 修改 `## 在TTS中的应用` 中已有文本 | 改写已有段落 |
| 修改 frontmatter 关系字段 | 改 `category`、`related_concepts`、`aliases`、`supersedes` |

**判定标准**: 修改已有 prose 或关系型 frontmatter。

**status 行为**: 无论当前 status 是什么，实质性修改后 status → `pending-review`。

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
