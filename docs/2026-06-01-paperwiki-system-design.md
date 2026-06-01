# PaperWiki 系统设计 — TTS 方向个人知识库

> 用最小的注意力成本,把论文沉淀成可信、可懂、可复用、不会烂的知识网络。

---

## 一、系统定位

### 三个职责
1. **筛** — 从海量论文里选出值得花时间的,不浪费注意力
2. **懂** — 对值得读的,真正搞懂它怎么 work,而不是知道它"存在"
3. **连** — 把单篇的理解接进已有认知,形成可导航的网

### 四条底线
- **可理解**: 讲清原理,合上能复述
- **可溯源**: 每个判断能追回原文出处
- **可导航**: 概念互链、有结构
- **不腐烂**: 能发现并修正过期、矛盾、死链

### 设计哲学 (参考 Karpathy LLM Wiki)
- **Obsidian 是 IDE,LLM 是程序员,Wiki 是代码库**
- 三层: 原始源(不可变) / Wiki(LLM 写,你读审) / Schema(规则配置)
- LLM 做总结、交叉引用、归档、记账 — 因为 LLM 不会累
- **概念页是知识缓存层** — 既是导航入口,也是 KB 检索的压缩知识源
- **Vault = Git Repo** — 每次 ingest 操作自动 commit,支持回滚、blame、diff 审计

### 典型复用场景(系统目标边界)
本系统面向以下使用场景:
- 快速定位某篇论文属于哪条技术路线
- 做某主题的综述/调研
- 对比两个模型或方法的差异
- 回答"某概念在 TTS 里怎么用、谁用了、效果如何"
- 为新论文做知识库背景定位和创新性评价
- 选复现对象时参考已有分析

---

## 二、系统运行原则

### 原则 1: 可信知识层正式定义

**可信知识层** = 系统中可以被引用、被 KB 检索依赖、被当作"事实"使用的内容。

| 内容 | 是否属于可信层 | 准入条件 |
|---|---|---|
| 精读笔记 (status: reviewed) | **是** | 你人工审核并标记 `status: reviewed` |
| 复现笔记 (status: reviewed) | **是** | 你人工审核并标记 `status: reviewed` |
| 概念页 (status: confirmed) | **是** | 你批量确认后标记 `status: confirmed` |
| 模型页 (status: confirmed) | **是** | 你批量确认后标记 `status: confirmed` |
| 任务页 (status: confirmed) | **是** | 你批量确认后标记 `status: confirmed` |
| 数据集页 (status: confirmed) | **是** | 你批量确认后标记 `status: confirmed` |
| 精读笔记 (status: draft) | **否** | 未审核,参考级 |
| 概念页 (status: pending-review) | **否** | LLM 刚更新,等待确认 |
| 卡片 / 加强卡片 | **否,永远不是** | 线索级,不可引用 |

**KB 检索以可信层为主依据。** pending-review 内容可作为参考依据读取,但必须标注"此页尚未确认",不得作为事实断言的来源。

**核心约束: AI 生成但未经人工确认的内容,不得进入可信层。**

### 原则 2: 主视图声明

**系统的主视图是: 实体页(概念库/模型库/任务库/数据集) + MOC。**

含义:
- 你日常浏览和检索的主入口是 MOC 和实体页,不是论文笔记列表
- Agent 更新优先级: 实体页 > MOC > 论文笔记
- KB 检索默认读实体页
- 首页/仪表盘应导向 MOC

论文笔记是"生产车间",实体页和 MOC 是"展厅"。你主要逛展厅。

### 原则 3: 低承诺缓冲层 (Inbox)

系统承认"先收着、暂不判断"这个状态的合法性。

**Inbox 机制:**
- 任何输入可以先进入 `inbox` 状态,不触发任何 pipeline
- Inbox 条目只需最少信息: 标题 + 来源 + 为何收藏(一句话)
- Inbox 没有时间压力,不算审核积压
- 你随时可以从 inbox 中选取条目,指定层级后进入对应 pipeline

**与四层笔记的关系:**
```
inbox(零承诺) → 卡片(自动) → 加强卡片(选读) → 精读(明确指定) → 复现(明确指定)
```

Inbox 在卡片之前,是整个系统的最低承诺入口。

### 原则 4: 失败降级策略

系统不要求所有流程都完美运行。定义"不顺时怎么办":

| 场景 | 降级策略 |
|---|---|
| KB 检索失败(搜不到相关实体页) | 正常生成笔记,KB 背景节标注"未找到相关知识库背景",不阻断流程 |
| 概念页质量差或过时 | KB 背景节标注"相关概念页可能过时(上次更新: YYYY-MM-DD)",不阻断流程 |
| Agent 无法判断是否该创建新实体页 | 标记为 `[待决]` 写入 log,不自动创建,等你下次审核时决定 |
| Lint 结论不确定(如矛盾检测 false positive) | 标记为 `[疑似]` 写入 lint 报告,不自动修改 |
| 反向更新失败(概念页格式异常等) | 跳过该步,在 log 中记录 `[skip/update]`,不阻断整个 ingest |

**核心哲学: 宁可产出带标注的不完美结果,也不因局部失败而中断整个流程。**

### 原则 5: 默认不升级

**笔记层级只能由你主动升级,系统不得自动升级。**

- Agent 不得自行将卡片升为加强卡片
- Agent 不得自行将加强卡片升为精读
- Agent 不得自行将精读升为复现
- 每次升级必须有你的明确指令

防止: 精读笔记膨胀 → 审核队列过长 → 审核被跳过 → 可信层注水。

### 原则 6: 系统哲学

> **系统默认允许"不完美但可运行"的状态。局部完备优先,全局逐步修复。**

含义:
- 不要求 vault 始终全局一致
- 有死链、有孤儿页、有待审内容是正常状态
- lint 的作用是"让你知道哪里不完美",不是"强制修复到完美"
- 新增内容的质量优先,存量修复按优先级渐进

---

## 三、架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    AGENTS.md (总纲)                       │
│  目录规范 · 操作接口 · 链接规则 · lint 规则 · 6条原则    │
└─────────────────────────────────────────────────────────┘

┌──────────┐    ┌──────────────────┐    ┌──────────────┐
│  Sources │    │      Wiki        │    │  _templates  │
│ (不可变) │───▶│ (LLM写, 你读审)  │◀───│  (页面模板)  │
└──────────┘    └──────────────────┘    └──────────────┘
                         │
                    Git (版本控制)
                 每次 ingest 自动 commit

Wiki 内部:
┌────────────┬─────────────┬──────────────┬────────────┐
│  论文笔记   │   概念库     │   模型库      │  任务库    │
│ (4 层笔记) │ (知识缓存层) │ (系统演进线)  │ (问题维度) │
└────────────┴─────────────┴──────────────┴────────────┘
                        ▲
                        │ KB 检索只读可信层
┌────────────┬──────────┴───┬────────────┐
│ DailyPapers│    _MOC      │   _lint    │
│ (推荐清单) │ (主题导航)   │ (健康报告) │
└────────────┴──────────────┴────────────┘
```

**三个角色分工:**
| 角色 | 职责 |
|---|---|
| **你** | 筛选、审核、确认可信层、策展方向、决定升级 |
| **LLM** | 拓取、打分、生成笔记、维护实体页(pending-review)、链接、lint |
| **概念页** | 既是导航入口,也是 KB 检索的知识缓存层(确认后才可信) |

---

## 四、目录结构

```
Vault/
├── AGENTS.md                    ← 系统总纲(目录规范+操作接口+6条原则)
├── _templates/                  ← 页面模板
│   ├── paper-card.md            ← 卡片级
│   ├── paper-enhanced-card.md   ← 加强卡片
│   ├── paper-deep.md            ← 精读级
│   ├── paper-repro.md           ← 复现级
│   ├── concept.md               ← 概念/方法
│   ├── model.md                 ← 模型/系统
│   ├── task.md                  ← 任务/问题
│   └── dataset.md               ← 数据集与指标
├── Sources/                     ← 原始 PDF(本地存储,gitignore 排除)
├── _inbox/                      ← 低承诺缓冲区(先收着,暂不处理)
├── 论文笔记/                    ← 所有论文笔记(4层共存,frontmatter 区分)
├── 概念库/                      ← 概念/方法实体页
├── 模型库/                      ← 模型/系统实体页
├── 任务库/                      ← 任务/问题实体页
├── 数据集/                      ← 数据集与指标页
├── DailyPapers/                 ← 每日推荐 + 周报
│   ├── YYYY-MM-DD.md
│   └── Weekly/
├── _MOC/                        ← Map of Content 导航页(按主题聚合)
│   ├── TTS-总览.md
│   ├── 语音编码.md
│   ├── 韵律与情感.md
│   └── ...
├── _lint/                       ← lint 报告
└── log.md                       ← 时间线(append-only)
```

**P5 阶段预留(早期不创建):**
```
├── _创新亮点/                   ← 跨论文的设计亮点归档(按主题)
├── _对比报告/                   ← 论文对比分析
```

### Sources/ 存储语义

**PDF 存于 Sources/ 内,但 gitignore 排除,不进 git。**

- Obsidian 可直接打开 vault 内的 PDF
- 溯源靠论文笔记 frontmatter 中的 `arxiv_id` + `source: "Sources/xxx.pdf"`
- 换电脑/clone 后需重新下载 PDF(通过 arxiv_id 可自动化)
- .gitignore 规则: `Sources/*.pdf`

### 论文笔记存储模型

**一篇论文 = 一个文件,永远在 `论文笔记/` 下。**

- 无论 tier 是 card/enhanced-card/deep/repro,都是同一个文件
- 升级只改 frontmatter `tier` + 补内容,文件不动,链接不断
- **DailyPapers/ 是索引页**,只列标题 + 一句话 + `[[论文笔记/xxx]]` 链接,不存笔记内容
- 不存在"双重真相":论文笔记是唯一主记录,DailyPapers 是视图

### Inbox 处理后

**处理后直接删除 inbox 条目。**

- 从 inbox 选取论文进入精读 → 在 `论文笔记/` 创建笔记 → 删除 `_inbox/` 中的条目
- "当初为什么收"如有价值,自然体现在笔记内容中
- Inbox 始终只包含未处理的条目,无需额外状态字段

---

## 五、四层笔记体系

| 层级 | 触发场景 | 成本 | KB 上下文 | 内容深度 | 可信度 |
|---|---|---|---|---|---|
| **卡片** | 每日推荐自动生成 | 极低 | 不调 | 一句话定位 + 方法关键词 + 为何值得关注 | 线索级,不可引用 |
| **加强卡片** | 你从推荐中选出"感兴趣" | 低 | 不调 | 方法概述 + 关键结果 + 与已知方法简单对比 | 参考级,需验证才可引用 |
| **精读** | 你明确说"精读" / 外部导入 | 中 | **调 KB** | 原理 + 方法 + 实验 + 点评 + KB 背景定位 | 人工审核后可引用 |
| **复现** | 你明确指定 | 高 | **调 KB** | 精读全部 + 架构细节 + 训练配置 + loss + 代码级理解 | 最高,但需验证 |

**升级路径:** 只改 frontmatter `tier` 字段 + 补充内容,文件不动,链接不断。升级只能由你主动触发。

**精读产出原则:**
> 精读笔记必须体现**机制理解**,而不是更长的摘要。标准:合上笔记后能向同事解释"它为什么 work",而不只是"它做了什么"。

**可信度规则:**
- 卡片/加强卡片 = 线索和参考,永远不进入可信层
- 精读 (status: reviewed) = 可信层,可作为引用来源
- 复现 (status: reviewed) = 最高可信度
- 实体页 (status: confirmed) = 可信层,KB 检索可依赖

---

## 六、Frontmatter 规范

### 论文笔记

```yaml
---
type: paper
tier: card | enhanced-card | deep | repro
title: "Paper Title"
arxiv_id: "2406.xxxxx"
source: "Sources/paper.pdf"       # 或 URL
authors: [Author1, Author2]
year: 2026
venue: ICASSP
tags: [flow-matching, zero-shot-tts, codec]
concepts: ["[[Flow Matching]]", "[[Neural Codec]]"]
models: ["[[CosyVoice]]"]
tasks: ["[[Zero-Shot TTS]]"]
datasets: ["[[LibriTTS]]"]
status: draft | reviewed           # 你审核过没有
created: 2026-06-01
updated: 2026-06-01
kb_context_sources: 3              # KB 检索时读了几个概念页(精读/复现才有)
---
```

### 概念页

```yaml
---
type: concept
title: "Flow Matching"
aliases: [flow matching, conditional flow matching, CFM]
category: generative-method
tags: [generation, continuous-flow, ODE]
key_papers: ["[[Paper1]]", "[[Paper2]]"]
related_concepts: ["[[Diffusion]]", "[[ODE Solver]]"]
status: pending-review | confirmed  # LLM 实质修改后为 pending-review,你确认后为 confirmed
lifecycle: active | deprecated | merged  # 实体生命周期状态
merged_into: ""                    # 如 lifecycle=merged,指向目标页 "[[Target Concept]]"
deprecated_reason: ""              # 如 lifecycle=deprecated,说明原因
created: 2026-06-01
updated: 2026-06-01
---
```

### 模型页

```yaml
---
type: model
title: "CosyVoice"
aliases: [CosyVoice, CosyVoice2]
org: Alibaba
year: 2024
tags: [tts, flow-matching, zero-shot]
key_concepts: ["[[Flow Matching]]", "[[LLM-based TTS]]"]
tasks: ["[[Zero-Shot TTS]]", "[[Voice Cloning]]"]
key_papers: ["[[CosyVoice Paper]]"]
supersedes: []                     # 它替代了什么
superseded_by: []                  # 被什么替代了
status: pending-review | confirmed
lifecycle: active | deprecated | merged
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---
```

### 任务页

```yaml
---
type: task
title: "Zero-Shot TTS"
aliases: [zero-shot speech synthesis, 零样本语音合成]
tags: [tts, zero-shot, generalization]
key_approaches: ["[[In-Context Learning]]", "[[Speaker Embedding]]"]
key_models: ["[[VALL-E]]", "[[CosyVoice]]"]
benchmarks: ["[[LibriSpeech Test Clean]]"]
metrics: [MOS, SIM, WER]
status: pending-review | confirmed
lifecycle: active | deprecated | merged
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---
```

### 数据集页

```yaml
---
type: dataset
title: "LibriTTS"
aliases: [LibriTTS, LibriTTS-R]
domain: English speech
scale: "585 hours"
tags: [english, tts, multi-speaker]
used_by: ["[[VITS]]", "[[NaturalSpeech]]"]
metrics_reported_on: [MOS, WER]
url: "https://www.openslr.org/60/"
status: pending-review | confirmed
lifecycle: active | deprecated | merged
merged_into: ""
created: 2026-06-01
updated: 2026-06-01
---
```

### Inbox 条目

```yaml
---
type: inbox
title: "Some Interesting Paper"
source: "https://arxiv.org/abs/..."
why: "看到有人提到它在 zero-shot 上效果很好"
created: 2026-06-01
---
```

---

## 七、三条入口的 Ingest 流程

### 入口 0: Inbox (先收着)

```
你随时丢一个链接/标题/想法进来
    ↓
创建 _inbox/ 下的条目(最少信息: 标题+来源+一句话原因)
    ↓
不触发任何 pipeline,零成本
    ↓
你随时可以从 inbox 选取,指定层级后进入对应流程
```

### 入口 1: 每日推荐 → 选读

```
arXiv / HuggingFace 自动拓取(TTS 相关)
    ↓
LLM 按兴趣画像打分排序
    ↓
生成 DailyPapers/YYYY-MM-DD.md(含所有卡片级条目)
    ↓
你浏览,标记感兴趣的
    ↓
你明确指定 → 生成加强卡片(tier 升级为 enhanced-card)
    ↓
[可选] 你说"这篇精读" → 进入精读 pipeline
```

### 入口 2: 外部导入

```
你丢 PDF / URL / 博客链接
    ↓
PDF 存入 Sources/(URL 记录在 frontmatter)
    ↓
默认进入精读 pipeline
```

### 入口 3: 指定复现

```
你指定某篇(可能已有笔记,可能是新论文)
    ↓
进入复现 pipeline(复现模板 + KB 上下文)
```

### 精读/复现 Pipeline (核心流程)

```
① 读原文(PDF / URL)
    ↓
② KB 上下文检索(只读可信层实体页)
   - 从新论文提取关键 tags/concepts
   - 在 概念库/模型库/任务库 搜 frontmatter(tags + aliases 匹配)
   - 只读 status: confirmed 的实体页
   - 如果命中 pending-review 页,标注"此页尚未确认"
   - 命中 3-6 个实体页,读取正文(总计 5-10k tokens)
   - 综合: 谱系定位 + 已有认知 + 新在哪
   - [降级] 如搜不到相关页 → 标注"未找到相关知识库背景",继续
    ↓
③ 生成笔记(按模板,含「KB 背景」节)
   - 笔记 status: draft
   - 必须体现机制理解,不是更长的摘要
    ↓
④ 反向更新(区分追加 vs 实质修改)
   - 追加更新(加 key_papers、追加贡献描述): 不改变 status,保持当前状态不变
   - 实质修改(重写定义、改变关系结构): status → pending-review
   - 如涉及新概念/模型/任务: 创建新实体页, status: pending-review
   - [降级] 如无法判断是否该创建 → 记入 log [待决],不自动创建
   - Git: 自动 commit,message 标注操作类型
    ↓
⑤ 局部 lint(基础健康)
   - 所有 [[wikilink]] 指向真实文件
   - frontmatter 完整性
   - 新实体页是否已创建
   - 新页面至少被一个 MOC 覆盖(P4 启用后生效,P1-P3 不检查此项)
    ↓
⑥ 更新 log.md + 触发 MOC 刷新(如涉及新主题)
    ↓
⑦ 审核积压检查
   - 统计当前 pending-review 实体页数量 + draft 精读笔记数量
   - 如超过阈值(建议: 10 个 pending-review 或 5 篇 draft 精读) → 提醒
```

---

## 八、KB 上下文检索机制

### 核心原则

**实体页(概念库/模型库/任务库/数据集)是知识缓存层,KB 检索以可信层(status: confirmed)实体页为主依据,不读论文笔记。**

### 为什么不读论文笔记

| 策略 | 读取量 | 质量 | 成本 |
|---|---|---|---|
| 读所有命中的论文笔记 | 40-100k tokens | 信息多但噪声大 | 高,不可持续 |
| **只读已确认的实体页** | **5-10k tokens** | **信息已压缩,信噪比高** | **低,可持续** |

### 检索流程(形式化)

**Step 1: 提取查询向量**

从新论文的 title/abstract/正文中提取:
- 方法名(如 flow matching, diffusion)
- 任务名(如 zero-shot TTS, voice cloning)
- 模型名(如 CosyVoice, VALL-E)
- 论文 frontmatter 中的 tags/concepts/models/tasks

**Step 2: 匹配(满足任一条件即命中)**

```
条件 1: 查询关键词精确匹配实体页 title 或任一 alias
条件 2: 新论文 tags 与实体页 tags 交集 ≥ 2
条件 3: 新论文 frontmatter 中 concepts/models/tasks 直接引用了该实体页
```

**Step 3: 筛选**

- 只保留 lifecycle=active 的页面
- status: confirmed → 正常参与
- status: pending-review → 可读取但标注"此页尚未确认"

**Step 4: 排序(命中 > 6 个时取 Top 6)**

```
排序优先级:
1. 直接引用(条件3) > tag匹配(条件2) > alias匹配(条件1)
2. 同优先级内,按 updated 日期降序(越新越优先)
```

**Step 5: 读取并综合**

读取 Top 3-6 个实体页正文(总计 5-10k tokens),生成「KB 背景」节:
- 相关概念: 库里已有什么认知
- 谱系定位: 它沿着什么路线走,前面是谁
- 创新判断: 相对已知方法,新在哪(或不新)

### 检索日志

**log.md 记录:**
```
- [kb/search] [[新论文]] — 命中 5 页(取 Top 4): [[A]]✓, [[B]]✓, [[C]]✓, [[D]]✓ | 过滤: [[E]](pending-review)
```

**笔记 KB 背景节底部标注:**
```
> 检索命中: [[A]], [[B]], [[C]], [[D]] | 过滤: [[E]](未确认) | 未命中但可能相关: 无
```

**召回反馈回路:** 如果你审核时发现 KB 背景遗漏了明显相关的概念,手动给该实体页补 alias → 下次检索即可命中。

### 降级场景

| 场景 | 处理 |
|---|---|
| 搜不到相关实体页 | 标注"未找到 KB 背景",正常生成笔记 |
| 只找到 pending-review 页 | 可读取但标注"基于未确认概念页,仅供参考" |
| 实体页明显过时(updated > 3个月) | 标注"相关概念页可能过时" |

### 单点脆弱性声明

**实体页(概念库/模型库/任务库/数据集)是 KB 检索的数据源。** 如果实体页质量下降,KB 检索质量随之下降。

缓解:
- pending-review 机制防止未审核内容直接成为检索依据
- 基础 lint 检查概念页 updated 日期
- 当概念页不够新时,系统降级运行(标注不确定性),不阻断

### KB 背景节的标注

```markdown
> [!info] KB 背景 (基于 4 个已确认实体页: [[Flow Matching]], [[Neural Codec]], [[Zero-Shot TTS]], [[CosyVoice]])
> 自动生成,不保证完整覆盖所有相关知识。
```

---

## 九、审核与可信层管理

### 概念页更新的两类操作

**核心规则:区分"追加更新"和"实质修改",只有后者触发 pending-review。**

| 更新类型 | 定义 | 是否触发 pending-review | 示例 |
|---|---|---|---|
| **追加更新** | 往已有结构中加条目,不改变核心语义 | **否,保持当前 status 不变** | 往 key_papers 加一篇;在正文末追加"XX 也用了此方法,贡献是..." |
| **实质修改** | 改变页面的核心定义、描述、关系 | **是,status → pending-review** | 重写概念定义;改变 category;修改 related_concepts 结构 |

**形式化判定规则(可编程):**

追加更新(不触发 pending-review):
- 往 `key_papers` 列表追加条目
- 往正文的列表区(如"关键论文")末尾追加一行
- 往表格末尾追加一行数据
- **判定标准: 只动列表/表格的末尾追加**

实质修改(触发 pending-review):
- 修改 `## 定义` 节的任何内容
- 修改 `## 在 TTS 中的应用` 节的已有描述(不是追加)
- 修改 frontmatter 中的 `category` 字段
- 增删 `related_concepts` 字段
- 增删 `aliases` 字段
- 修改 `supersedes` / `superseded_by` 字段
- **判定标准: 动任何已有正文描述或 frontmatter 关系字段**

**注意: "保持当前 status 不变"意味着:**
- 如果页面是 confirmed → 追加后仍是 confirmed
- 如果页面是 pending-review → 追加后仍是 pending-review(不会因为追加而变成 confirmed)

**为什么这样设计:**
活跃概念(如 Flow Matching)可能每周被多篇新论文引用。如果每次追加都触发 pending-review,该概念页几乎永远不在 confirmed 状态 → KB 检索最需要的概念反而不可信 → 复利效果被严重削弱。

追加更新风险极低(只是"多了一条记录"),不值得打断可信状态。实质修改才需要你验证。

### pending-review 流程

只有实质修改触发此流程:

```
LLM 对概念页做实质修改
    ↓
status 标记为 pending-review
    ↓
累积若干次实质修改
    ↓
你批量审阅(可按 Dataview 查询 status: pending-review)
    ↓
确认 → status 改为 confirmed,进入可信层
拒绝 → git revert 回滚到上次 confirmed 状态
```

### 审核积压告警

系统在每次 ingest 结束后统计:
- `pending-review` 实体页数量
- `draft` 精读笔记数量

**告警阈值(建议):**
- pending-review 实体页 > 10 个 → 提醒"有 N 个概念页等待你确认"
- draft 精读笔记 > 5 篇 → 提醒"有 N 篇精读笔记等待你审核"

**告警只提醒,不阻断。** 你可以选择忽略,但系统确保你知道积压状态。

### 审核的具体内容

| 审核对象 | 重点检查什么 |
|---|---|
| 精读笔记 | 机制理解是否到位;事实 claim 是否有原文标注;KB 背景节是否合理 |
| 概念页更新 | 新增内容是否准确;与已有描述是否一致;粒度是否合适 |
| 新实体页 | 是否真的需要独立成页(vs 合并到已有页);定义是否准确 |

---

## 十、Lint 系统

### 分级设计

Lint 能力分两级,对应不同实施阶段:

#### 基础健康 lint (P4 早期)

每次 ingest 局部触发 + 每周全量:

| 检查项 | 说明 | 难度 |
|---|---|---|
| 死链 | `[[X]]` 指向不存在的页面 | 低,纯文件检查 |
| frontmatter 完整性 | 符合模板 schema 的必填字段 | 低,规则匹配 |
| 孤儿页 | 无任何入链的笔记 | 低,反向链接统计 |
| 实体存在性 | frontmatter 引用的实体在对应库中存在 | 低,文件查找 |
| MOC 覆盖 | 每个实体页至少被一个 MOC 包含 | 低,遍历检查 |
| 概念页 updated 日期 | 有新论文引用但概念页未更新 | 低,日期比较 |
| 审核积压统计 | pending-review 和 draft 数量 | 低,状态统计 |

#### 高级认知 lint (P5 增强)

需要 LLM 理解能力,成本较高,非常规检查:

| 检查项 | 说明 | 难度 |
|---|---|---|
| 过期 SOTA | 笔记中标注的"最优结果"被后续论文超越 | 高,需比对数据 |
| 矛盾检测 | 两篇笔记对同一概念的描述互相矛盾 | 高,需语义理解 |
| 缺失关联 | A 和 B 都涉及概念 C 但未互相链接 | 中,需交叉比对 |
| 近义概念检测 | 两个概念页其实在说同一件事 | 高,需语义判断 |

**高级 lint 标注为 `[疑似]`,不自动修改,只报告。**

### Lint 输出

```
_lint/YYYY-MM-DD-lint-report.md
```

报告格式:
```markdown
# Lint Report 2026-06-01

## Summary
- Dead links: 2
- Orphan pages: 1
- Pending review backlog: 8 concept pages, 3 deep notes
- Schema violations: 0

## Details
### Dead Links
- [[论文笔记/PaperX]] → [[不存在的概念]] (line 42)

### Action Items
- [ ] 创建 [[不存在的概念]] 或修正链接
- [ ] 审核 8 个 pending-review 概念页
```

---

## 十一、实体页生命周期

### 状态定义

| lifecycle | 含义 | 在 MOC 中 | KB 检索 |
|---|---|---|---|
| **active** | 当前有效的实体 | 正常显示 | 正常参与 |
| **deprecated** | 已淘汰/不再使用的概念或模型 | 移到"历史"区 | 不参与检索,但保留以免死链 |
| **merged** | 合并到另一个实体页 | 不显示(redirect) | 不参与,检索自动指向目标页 |

### 操作流程

**合并(两个概念页其实说同一件事):**
```
你决定: [[Audio Codec]] 应该合并进 [[Neural Codec]]
    ↓
[[Audio Codec]] 的内容迁移到 [[Neural Codec]]
    ↓
[[Audio Codec]] frontmatter: lifecycle: merged, merged_into: "[[Neural Codec]]"
    ↓
[[Audio Codec]] 正文替换为: "此页已合并至 [[Neural Codec]]"
    ↓
所有指向 [[Audio Codec]] 的链接保持有效(页面仍存在,作为 redirect)
    ↓
lint 不再报告 [[Audio Codec]] 为孤儿页
```

**退役(某概念/模型已完全过时):**
```
你决定: [[某旧方法]] 已被淘汰
    ↓
frontmatter: lifecycle: deprecated, deprecated_reason: "已被 [[新方法]] 完全取代"
    ↓
MOC 中从活跃区移到"历史参考"区
    ↓
KB 检索不再读取此页(避免推荐过时方法)
    ↓
反向链接保持有效
```

### 触发条件
- lint 检测到近义概念时建议合并(你决定)
- 你手动标记退役
- Agent 不得自行合并或退役实体页

---

## 十二、版本控制 (Git)

### 基本配置

```
Remote: git@github.com:yd11111/PaperWiki.git
User: yd11111
Email: 1784578480@qq.com
```

### Git 在系统中的角色

- **每次 ingest 操作自动 commit** — commit message 包含操作类型和涉及页面
- **审核确认产生 commit** — 标记为 `[review]` 类型
- **支持回滚** — 如果实质修改被拒绝,`git revert` 到上次 confirmed 状态
- **支持审计** — `git blame` 追溯每行内容是哪次 ingest 产生的
- **支持 diff** — 审核时可以 `git diff` 查看自上次确认以来的变更

### Commit 格式约定

```
[ingest/deep] CosyVoice 2 — 精读笔记 + Flow Matching 概念页追加
[ingest/card] DailyPapers 2026-06-01 — 14 篇卡片
[review/batch] 确认 5 概念页 + 2 精读笔记
[update/concept] Flow Matching — 实质修改: 重写核心定义
[lint/full] 周报 + 2 死链修复
[lifecycle/merge] Audio Codec → Neural Codec
```

### 推送策略
- 本地 commit 频繁(每次操作)
- 推送到 remote 可按你的节奏(每日/每周)
- 不强制自动推送

---

## 十三、MOC 导航系统

### 设计原则

- **自动生成 + 人工备注分区**: `## 人工备注` 区域在自动刷新时保留不覆盖
- **按研究主题组织**: 不是字母表,是你思考问题的方式
- **层级结构**: 总览 → 子主题,支持 vault 增长
- **MOC 是主视图的一部分**: 日常浏览从这里开始

### MOC 页面结构

```markdown
# 语音编码

> [!note] 人工备注
> 2026 年的主要趋势是从 RVQ 转向单码本 + 连续特征...
> (此区域自动刷新时保留)

## 核心概念
- [[Neural Codec]] — 离散语音表示的通用框架
- [[RVQ]] — 残差向量量化,EnCodec/SoundStream 的基础
- [[Semantic Token]] — 语义层编码,HuBERT/W2V-BERT 提取

## 代表模型
- [[EnCodec]] — Meta, 2022, 多带宽 RVQ
- [[SoundStream]] — Google, 2021, 端到端编解码
- [[DAC]] — Descript, 2023, 高保真音频编码

## 相关论文 (按时间倒序)
- [[论文A]] — 2026, ...
- [[论文B]] — 2025, ...

## 相关任务
- [[Audio Compression]]
- [[Speech Tokenization]]
```

### MOC 生成规则(形式化)

**自动创建条件:** 某个 tag/主题下有 ≥ 3 个 active 实体页时,自动创建该主题的子 MOC。低于 3 个挂在上级 MOC 即可。

**纳入范围:**
- 实体页: 所有 lifecycle=active 的概念/模型/任务/数据集页
- 论文笔记: **仅 deep + repro 层级**(card/enhanced-card 不进入 MOC)
- deprecated 实体: 移到页面底部 `## 历史参考` 区(可折叠)
- merged 实体: 不显示(已被 redirect)

**"被 MOC 包含"的定义:** 至少一个 MOC 页面正文中存在直接 `[[wikilink]]` 指向该实体页。

### MOC 刷新触发条件
- 每次精读/复现 ingest 后,如果涉及的主题 MOC 存在 → 自动更新
- 如果涉及的主题下 active 实体页达到 3 个但无 MOC → 自动创建
- 每周基础 lint 时检查 MOC 覆盖完整性

---

## 十四、log.md 格式

```markdown
# Log

## 2026-06-01
- [ingest/deep] [[CosyVoice 2]] — 外部导入精读, KB 上下文: Flow Matching, Neural Codec, Zero-Shot TTS
- [ingest/enhanced-card] [[Paper A]], [[Paper B]], [[Paper C]] — 每日推荐选读
- [ingest/card] 每日推荐 14 篇
- [update/concept] [[Flow Matching]] — +CosyVoice2 贡献 (pending-review)
- [update/model] [[CosyVoice]] — +v2 对比数据 (pending-review)
- [create/concept] [[Speech Language Model]] — 新概念页 (pending-review)
- [lint/local] CosyVoice 2 笔记通过, 0 问题
- [review/batch] 确认 5 个概念页, 2 篇精读笔记
- [skip/update] [[某概念]] — 反向更新失败,格式异常,已跳过

## 2026-05-25
- [lint/full] 周报: 2 死链修复, 1 孤儿页处理
- [ingest/repro] [[VITS]] — 指定复现级阅读
- [alert] 审核积压: 12 pending-review, 4 draft — 建议抽空审核
```

**格式约定:**
- 前缀: `[操作类型/子类型]`
- 操作类型: `ingest` | `update` | `create` | `lint` | `review` | `alert` | `skip` | `moc` | `kb` | `lifecycle` | `init`
- 常见子类型: `ingest/deep` `ingest/card` `ingest/enhanced-card` `ingest/repro` `update/concept` `update/model` `create/concept` `lint/local` `lint/full` `review/batch` `skip/update` `skip/ingest` `kb/search` `lifecycle/merge` `lifecycle/deprecate` `alert/backlog`
- 每行可用 grep 过滤

**失败事务规则:** 失败的 ingest 不产生 git commit,只写 log `[skip/ingest]` 条目。

---

## 十五、风险与缓解

| 风险 | 严重度 | 缓解措施 | 降级路径 |
|---|---|---|---|
| **LLM 幻觉破坏可溯源** | 高 | 模板要求 claim 标注原文位置;精读级必须审核;卡片永不可信 | 未审核内容不进可信层 |
| **概念页质量下降(单点脆弱性)** | 高 | pending-review 机制;审核积压告警;updated 日期检查 | KB 检索标注"概念页可能过时",不阻断 |
| **审核积压导致信任塌方** | 中-高 | 积压告警;默认不升级;你控制输入节奏 | 积压期间 KB 检索降级为"仅供参考" |
| **概念页粒度判断困难** | 中 | AGENTS.md 定义粒度规则;aliases;lint 检测近义 | agent 无法判断时记 log [待决],不强行创建 |
| **MOC 腐化** | 中 | 自动重生成 + 人工备注保留;lint 检查覆盖 | MOC 过时不影响实体页使用 |
| **过度链接 → 噪声** | 中 | 模板区分核心关联 vs 相关参考 | — |
| **规模瓶颈 (500+ 篇)** | 低 | log 按年/季分;MOC 层级化;lint 增量 | — |
| **Schema 演化需迁移** | 低 | AGENTS.md 版本化;迁移脚本 | — |

---

## 十六、实施阶段

| 阶段 | 内容 | 交付物 |
|---|---|---|
| **P0: 地基** | 目录结构 + AGENTS.md(含 6 条原则) + 全部模板 + log.md + _inbox/ | 空但完整的 vault 骨架,原则已立 |
| **P1: 核心引擎** | 精读 ingest pipeline(入口2: 外部导入 → 精读 → 概念页 pending-review → 局部 lint → 审核积压检查) | 能跑通完整链路,包含 pending-review 和降级 |
| **P2: 筛的入口** | 每日推荐(拓取 → 打分 → 卡片 → 加强卡片升级) + inbox 流转 | 每日推荐 + inbox 作为缓冲 |
| **P3: 复利兑现** | KB 上下文检索(只读可信层) + 复现级模板 + 反向更新 | 精读时自动带上知识库背景(含降级标注) |
| **P4: 基础防腐** | 基础健康 lint(死链/孤儿/frontmatter/MOC覆盖/积压统计) + MOC 自动生成 | 每周基础健康报告 + MOC 自维护 |
| **P5: 高级增强** | 高级认知 lint(矛盾/stale SOTA/缺失关联) + 创新亮点 + 对比报告 | 认知层面的健康检查 + 高级复用功能 |

**每个阶段独立可用,不依赖后续阶段。**

---

## 十七、评价标准 (自检清单)

系统上线后,定期用这 6 个问题自检:

1. **我今天丢一篇论文进去,成本低吗?** → 如果麻烦,获取层失败
2. **我一周后回来看,还知道为什么记它吗?** → 如果不知道,筛选层失败
3. **我一个月后还能快速恢复这篇论文的核心机制吗?** → 如果不能,理解层失败
4. **我三个月后还能把它和旧知识连起来吗?** → 如果不能,连接层失败
5. **我半年后还敢引用里面的结论吗?** → 如果不敢,可信层失败
6. **我读得越多,这个库是越清楚还是越混乱?** → 如果越混乱,复利失败
