# AGENTS.md — PaperWiki 系统规范

> 本文件是 PaperWiki 的唯一权威规范。Agent 只需阅读此文件即可完整运行。

---

## 1. 系统概览

**目的**: 以 TTS/语音合成为核心的个人知识库，AI agent 辅助人类进行论文阅读、概念整理与知识积累。

**三个角色**:

| 角色 | 职责 |
|------|------|
| 你（人类） | 决策者。确认知识可信度、触发升级、执行 merge/deprecate |
| LLM（Agent） | 执行者。论文摘录、笔记生成、概念页维护、lint 检查、MOC 刷新 |
| 概念页 | 知识载体。实体化的概念/模型/数据集，是知识库的主视图单元 |

---

## 2. 目录语义

| 目录 | 用途 | 谁写入 | 是否入 git |
|------|------|--------|-----------|
| `论文笔记/` | 一篇论文一个文件，所有层级笔记的存放地 | Agent（生成）+ 人类（确认） | Yes |
| `概念库/` | 概念实体页（technique、loss、architecture 等） | Agent（创建/更新）+ 人类（确认） | Yes |
| `模型库/` | 模型实体页（具名模型如 VITS、CosyVoice） | Agent（创建/更新）+ 人类（确认） | Yes |
| `数据集/` | 数据集实体页 | Agent（创建/更新）+ 人类（确认） | Yes |
| `_MOC/` | Map of Content 导航页 | Agent（自动刷新） | Yes |
| `DailyPapers/` | 每日论文推荐索引（仅 title + link），不含正文 | Agent | Yes |
| `Sources/` | PDF 原文存放 | 人类/Agent 下载 | **No**（gitignored） |
| `_inbox/` | 零承诺收件箱 | 人类/Agent | Yes |
| `_templates/` | Obsidian 模板 | 人类 | Yes |
| `_lint/` | Lint 报告输出 | Agent | Yes |
| `任务库/` | 任务/项目追踪 | 人类 | Yes |
| `docs/` | 系统文档（非知识内容） | 人类 | Yes |
| `log.md` | Agent 操作日志 | Agent | Yes |

---

## 3. Sources/ 存储规则

- PDF 文件存放于 `Sources/`，已被 `.gitignore` 排除
- 可追溯性通过 frontmatter 保证：
  - `arxiv_id`: arXiv 论文 ID（如 `"2301.12345"`）
  - `source`: 论文来源 URL 或标识
- Agent 引用论文时使用 frontmatter 中的标识符，不依赖本地 PDF 存在

---

## 4. 笔记存储模型

- **一篇论文 = 一个文件**，存放于 `论文笔记/`
- `DailyPapers/` 是纯索引：只含 title + wikilink，不含论文正文内容
- **每日推荐日期规则**: 只收录当天 arXiv 发布的论文。HuggingFace trending ≠ 当天论文。当天无新论文可不生成推荐，不可混入非当天论文。
- **禁止双重真相**：论文的实质内容只存在于 `论文笔记/` 对应文件中，`DailyPapers/` 通过 `[[wikilink]]` 指向它
- 笔记文件名 = 论文简称或模型名（如 `CosyVoice.md`、`NaturalSpeech 3.md`）

---

## 5. 六大原则

### 原则 1: 可信知识层

| 内容 | 是否属于可信层 | 准入条件 |
|------|----------------|----------|
| 精读笔记 (status: reviewed) | **是** | 人类审核并标记 `status: reviewed` |
| 复现笔记 (status: reviewed) | **是** | 人类审核并标记 `status: reviewed` |
| 概念页 (status: confirmed) | **是** | 人类批量确认后标记 `status: confirmed` |
| 模型页 (status: confirmed) | **是** | 人类批量确认后标记 `status: confirmed` |
| 任务页 (status: confirmed) | **是** | 人类批量确认后标记 `status: confirmed` |
| 数据集页 (status: confirmed) | **是** | 人类批量确认后标记 `status: confirmed` |
| 精读/复现笔记 (status: draft) | **否** | 未审核，参考级 |
| 实体页 (status: pending-review) | **否** | Agent 刚更新，等待确认 |
| 卡片 / 加强卡片 | **否，永远不是** | 线索级，不可引用 |

**规则**:
- KB 检索以可信层(confirmed/reviewed)为主依据，pending-review 可作为参考依据但必须标注 `[待确认]`
- AI 生成内容在未经人类确认前，**不得**进入可信层
- `status` 字段记录于 frontmatter

### 原则 2: 主视图 = 实体页 + MOC

知识库的主视图不是论文笔记列表，而是**实体页**（概念库/模型库/任务库/数据集）+ **MOC 导航页**。论文笔记是输入端，实体页是沉淀端。

### 原则 3: Inbox — 零承诺

`_inbox/` 是零承诺的收件箱：
- 放进去不代表要处理
- 处理完即删除
- 没有时间压力

### 原则 4: 失败降级

| 场景 | 降级策略 |
|------|----------|
| KB 检索无匹配 | 标注 `[未找到相关知识库背景]`，继续处理 |
| 概念页内容过时（>3 月未更新） | 标注 `[相关概念页可能过时]`，仍可引用 |
| 无法确定是否为独立实体 | 记录 `[待决]`，不创建新页 |
| 反向更新失败 | 跳过 + 记录日志，不阻塞主流程 |

### 原则 5: 默认不升级

Agent **必须不**在没有用户显式指令的情况下升级笔记层级。用户说"升级"才升级。

### 原则 6: 允许不完美但可运行

局部完整性优先，全局一致性逐步达成。一篇笔记可以独立成立，不必等待所有关联页就绪。

---

## 6. 四级笔记体系

### 层级定义

| 层级 | 触发条件 | 成本 | KB 检索 | 深度 | 信任级别 |
|------|----------|------|---------|------|----------|
| **card**（卡片） | 每日推荐 / 快速浏览 | 1-3 min | 不触发 | 标题+一句话+标签 | draft |
| **enhanced-card**（增强卡片） | 用户要求"快速看一下" | 5-10 min | 不触发 | 摘要+核心方法+关键结果 | draft |
| **deep**（精读） | 用户要求"精读"/"深读" | 30-60 min | **触发** | 完整机制分析+实验解读 | pending-review |
| **repro**（复现） | 用户要求"复现分析" | 60+ min | **触发** | 代码级分析+复现要点 | pending-review |

### 升级规则

- **只有用户可以触发升级**：card → enhanced-card → deep → repro
- Agent 不得自行判断某篇论文"值得精读"而升级
- 升级时保留原有内容，在其基础上扩展

### 精读必须体现机制理解

deep 及以上层级笔记**必须**包含：
- 方法的 WHY（为什么这样设计）
- 关键模块的 HOW（具体怎么做）
- 与已有方法的差异点
- 实验中能佐证设计选择的 evidence

### 速查卡片

精读/复现笔记在 KB 背景节之后、核心问题之前包含速查卡片：
```markdown
> [!summary] 速查
> - **一句话**: (核心贡献)
> - **路线**: (输入→模块→输出)
> - **指标**: (关键数字 + 对比基准 + 数据集)
> - **可借鉴**: (可迁移的 idea)
> - **局限**: (不 work 的 / 复现难点 / 未开源)
```

---

## 7. Append vs Substantive Update 规则

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

## 8. 实体页生命周期

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

## 9. Inbox 规则

- **零承诺**: 放入 `_inbox/` 不代表任何处理承诺
- **最小字段**: 只需 `title` + `source` + `why`（为什么感兴趣）
- **处理后删除**: 一旦处理完成（生成笔记 or 决定不看），删除 inbox 文件
- **无时间压力**: 可以一直放着，不产生任何告警

---

## 10. KB 检索规则

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

---

## 11. MOC 规则

- **自动创建**: 某主题下 ≥ 3 个 active 实体页时，自动创建该主题的 sub-MOC
- **收录标准**: 只有 deep + repro 层级笔记可出现在 MOC 论文列表中（card/enhanced-card 不收录）
- **deprecated 处理**: 移至 MOC 底部 `## 历史参考` section
- **merged 处理**: 从 MOC 中移除（不显示）
- **"被 MOC 包含"定义**: 至少一个 MOC 中有直接 `[[wikilink]]` 指向该页
- **自动刷新**: 每次 ingest 后检查并更新相关 MOC
- **人工备注保护**: MOC 中 `> [!note] 人工备注` 区块在刷新时**必须保留**，不得覆盖

---

## 12. Lint 规则

### Local Lint（每次 ingest 时执行）

| 检查项 | 说明 |
|--------|------|
| Wikilink 有效性 | 所有 `[[link]]` 指向存在的文件 |
| Frontmatter 完整性 | 必要字段齐全（title, status, tags 等） |
| 实体存在性 | 引用的概念/模型页存在 |
| MOC 覆盖（仅 P4+） | deep/repro 笔记是否被至少一个 MOC 包含 |

### Full Lint（每周执行）

| 检查项 | 说明 |
|--------|------|
| Dead links | 指向不存在页面的 wikilink |
| Orphan pages | 没有任何 incoming link 的页面 |
| Frontmatter 一致性 | 字段格式、枚举值合规 |
| MOC 覆盖 | 所有 deep/repro 笔记被 MOC 包含 |
| 概念页溯源提醒 | `key_papers` ≥ 3 且 `origin_paper` 为空 → 提示"建议找原始论文完善此概念页" |
| Review backlog | 10 个 pending-review 或 5 个 draft deep → alert |
| 概念页 staleness | active 页超过 3 月未更新 |
| 概念去重检查 | 扫描 title/aliases 是否存在语义重叠的页面对,提示 merge 候选 |

### Cognitive Lint（P5，仅手动触发）

| 检查项 | 说明 |
|--------|------|
| 矛盾检测 | 不同页面对同一概念的描述冲突 |
| Stale SOTA | 标记的 SOTA 结果已有更新论文超越 |
| Missing links | 语义相关但缺少 wikilink 的页面对 |

**规则**: 所有 cognitive lint 结果标记 `[疑似]`，**永不**自动修改内容。

---

## 审阅系统

### 单篇审阅
每次 deep/repro 笔记生成后、反向更新前自动触发。
- 输出: `_review/{论文名}-review.yml` (YAML) + 笔记末尾 `[!review]` callout
- 结论: pass(直接反向更新) / pass-with-fixes(建议修正,可放行) / revise(有 high 问题,修正后重审) / reject-as-deep(降为 enhanced-card)
- 原则: 只诊断不重写,不阻断 pipeline,每个问题有 location+detail+suggestion

### 跨篇模式分析
每 5-10 篇手动触发。统计高频问题 → 判断根因(模板/规则/skill/检索) → 产出系统升级建议。
输出: `_review/pattern-analysis-YYYY-MM-DD.md`

---

## 系统同步

### 单次同步 (每次 ingest 后自动)

精读 pipeline 完成后检查:
- [ ] 本篇 frontmatter 字段齐全
- [ ] 本篇涉及的 MOC 是否已更新(论文 tags 对应的 MOC 是否包含本篇)
- [ ] 审阅 callout 已追加
- [ ] log.md 已更新
- [ ] Git commit 已完成

有遗漏当场补完。

### 全面同步 (每周 / 批量操作后 / "跑一下系统同步")

| 检查项 | 说明 |
|--------|------|
| MOC 覆盖 | 所有 deep/repro 是否被至少一个 MOC 收录 |
| 审阅覆盖 | 所有 deep/repro 是否有审阅 callout |
| Frontmatter 一致性 | 所有笔记是否符合当前模板 schema |
| 死链 | 全 vault wikilink 有效性 |
| 孤儿页 | 没有入链的实体页 |
| 概念去重 | title/aliases 语义重叠 |
| 可信层进度 | confirmed/reviewed 比例是否在增长 |
| 概念溯源 | key_papers ≥ 3 且 origin_paper 空 |
| 系统文档 | CLAUDE.md vault 状态是否反映当前数据 |

产出: `_lint/YYYY-MM-DD-system-sync.md`

---

## 13. Link 约定

- 所有跨页引用使用 `[[wikilink]]` 格式
- Frontmatter 中的链接使用 `"[[Page Name]]"` 格式（带引号）
- **同名文件消歧**: 当论文笔记和实体页同名时（如论文笔记/CosyVoice 3.md 和模型库/CosyVoice 3.md），使用 folder-qualified path：`[[论文笔记/CosyVoice 3|CosyVoice 3]]`。实体页的 `key_papers` 始终使用 `[[论文笔记/xxx|显示名]]` 格式。
- 模板区分：
  - **核心链接（must-link）**: 实体页必须包含的关系链接（如 `related_concepts`）
  - **相关引用（optional）**: 可选的延伸链接（如 `see_also`）

---

## 14. Commit 格式

```
[type/subtype] pages — description
```

### Type 列表

| Type | 含义 |
|------|------|
| `ingest` | 新论文笔记入库 |
| `update` | 更新已有页面 |
| `create` | 创建新实体页 |
| `lint` | Lint 检查与修复 |
| `review` | 批量 review 操作 |
| `alert` | 告警（如 backlog 超限） |
| `skip` | 跳过操作（失败/主动放弃） |
| `moc` | MOC 刷新 |
| `kb` | KB 检索相关 |
| `lifecycle` | 生命周期变更 |
| `init` | 初始化操作 |

### Subtype 列表

| Subtype | 用于 |
|---------|------|
| `ingest/deep` | 精读笔记入库 |
| `ingest/card` | 卡片笔记入库 |
| `ingest/enhanced-card` | 增强卡片入库 |
| `ingest/repro` | 复现笔记入库 |
| `update/concept` | 更新概念页 |
| `update/model` | 更新模型页 |
| `create/concept` | 新建概念页 |
| `lint/local` | 局部 lint |
| `lint/full` | 全量 lint |
| `review/batch` | 批量 review |
| `skip/update` | 跳过更新 |
| `skip/ingest` | 跳过入库 |
| `kb/search` | KB 检索记录 |
| `lifecycle/merge` | 实体合并 |
| `lifecycle/deprecate` | 实体废弃 |
| `alert/backlog` | Backlog 告警 |

### 失败处理

失败的 ingest **不产生 commit**，仅在 `log.md` 记录 `[skip/ingest]`。

---

## 15. Claim 标注格式

论文中的具体 claim 必须标注出处：

| 格式 | 用途 | 示例 |
|------|------|------|
| `[§X.X]` | 章节引用 | "采用 flow matching 训练 [§3.2]" |
| `[Table N]` | 表格引用 | "MOS 达到 4.2 [Table 3]" |
| `[Fig N]` | 图片引用 | "架构如 [Fig 2] 所示" |

---

## 16. 失败降级总表

| 场景 | 降级策略 | 是否阻塞 |
|------|----------|----------|
| KB 检索无匹配 | 标注 `[未找到相关知识库背景]`，继续 ingest | 否 |
| 仅命中 pending-review | 标注 `[基于未确认概念页，仅供参考]` | 否 |
| 概念页过时（>3 月） | 标注 `[相关概念页可能过时]` | 否 |
| 无法确定是否独立实体 | 记录 `[待决]`，不创建新页 | 否 |
| 反向更新失败 | 跳过 + 记录日志 | 否 |
| PDF 无法解析 | 记录 `[skip/ingest]`，不产生 commit | 是（本次） |
| Wikilink 目标不存在 | Lint 报告，不自动创建 | 否 |
| MOC 刷新失败 | 日志记录，不阻塞 ingest | 否 |

---

## 17. Review Backlog 告警阈值

| 条件 | 动作 |
|------|------|
| pending-review 实体页 ≥ 10 | 在 `log.md` 记录 `[alert/backlog]`，提醒用户 review |
| draft 层级的 deep/repro 笔记 ≥ 5 | 在 `log.md` 记录 `[alert/backlog]`，提醒用户确认 |

告警仅为提醒，不阻塞任何操作。用户可选择忽略。

---

## 附: 快速参考

```
Agent 精读 pipeline:
① 读原文
② KB 检索
③ 生成笔记草稿 (status: draft)
④ 审阅 (反向更新前,对照 checklist 检查)
⑤ Git commit 草稿 + 审阅报告
⑥ 反向更新 (仅审阅通过后)
⑦ 局部 lint
⑧ Git commit + log
⑨ 审核积压检查

其他规则:
- 新概念出现 → 创建实体页(pending-review) → 更新 MOC
- Append → 保持 status; Substantive → status=pending-review
- 每次 ingest 后 → 检查 MOC 是否需刷新
- 每周 → full lint → 报告 + backlog alert
- 永不: 自行升级层级 / 自行 merge/deprecate / 修改 trusted 内容
```
