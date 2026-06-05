# Harness Upgrade Batch 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the three P0 items from the harness upgrade design — MOC rules (G2), MOC review checklist (V2), and reviewer independence (E2) — to establish the foundation for the "automatic quality gate + human calibration" trust model.

**Architecture:** G2 creates comprehensive MOC governance rules in AGENTS.md §11. V2 creates a three-layer MOC review checklist (mirroring `review-checklist.md`'s structure). E2 restructures `paperwiki-reader` to dispatch the reviewer as an independent subagent instead of running it in the same session. G2 and V2 are independently developable; E2 depends on neither but benefits from V2 being available.

**Tech Stack:** Markdown (AGENTS.md rules, checklist templates), Claude Code skills (SKILL.md files), Bash (git operations)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `AGENTS.md` §11 | Expand MOC rules from 7 lines to complete governance system |
| Create | `_templates/moc-review-checklist.md` | Three-layer MOC review checklist (宪法/法律/判例) |
| Modify | `_templates/review-checklist.md` | Add cross-reference to new MOC checklist |
| Modify | `~/.claude/skills/paperwiki-reader/SKILL.md` | Change Step 3.5 from inline review to subagent dispatch |
| Modify | `~/.claude/skills/paperwiki-reviewer/SKILL.md` | Add standalone invocation mode + MOC review mode |
| Modify | `CLAUDE.md` | Update vault state section to reflect new files |

---

## Task 1: G2 — MOC Rules Expansion (AGENTS.md §11)

**Files:**
- Modify: `AGENTS.md:333-341` (current §11, 7 lines)

### Context for Implementer

Current §11 is only 7 bullet points covering: auto-create threshold, inclusion criteria (deep/repro only), deprecated handling, merged handling, "被 MOC 包含" definition, auto-refresh, and human notes protection. The MOC review (see `/Users/xiangshu/PaperWiki-log/moc审阅`) identified 15 problem categories, 11 of which trace to missing rules in §11. The new rules must address all 11.

### Source Material

The 15 problem categories from MOC review, mapped to rules needed:

1. **混层** → Need: concept layering rule (范式/模块/技巧 分层)
2. **边界不清** → Need: scope boundary declaration per MOC
3. **粒度不一致** → Need: concept admission tiering within MOC
4. **代表模型混入 backbone** → Need: model admission rule (代表模型 vs 组件 vs 上游)
5. **论文区失控** → Need: paper section capacity rule + overflow to sub-MOC
6. **年份错乱** → Need: year-from-frontmatter rule
7. **SOTA 表条件不一致** → Need: SOTA table protocol rule (or ban)
8. **演进箭头语义不明** → Need: relationship type enumeration
9. **任务页/MOC 角色重叠** → Need: role boundary declaration
10. **强判断无来源** → Need: claim attribution rule
11. **alias 未充分使用** → Already covered by §8 dedup rules
12. **重复条目** → Need: no-duplicate-wikilink rule
13. **命名一致性** → Already covered by §3/§8
14. **缺边界声明** → Same as #2
15. **过满** → Same as #5

- [ ] **Step 1: Read current §11 in full**

Read `AGENTS.md` lines 333-341 to confirm exact content and surrounding section boundaries.

- [ ] **Step 2: Write expanded §11**

Replace the existing §11 (7 bullet points) with the following complete section. Use the Edit tool with `old_string` = the entire current §11 content and `new_string` = the expanded version below.

```markdown
## 11. MOC 规则

### 宪法原则 (5 条,极少变动)

MOC 的设计服务于一个目标: **让人在 3 秒内找到想要的东西**。

| 原则 | 核心问题 | 违反信号 |
|------|----------|----------|
| **可导航** | 新人打开 MOC 能在 3 秒内理解结构吗? | 需要滚屏才能找到入口 |
| **不混层** | 每个 section 内的条目是同一抽象层级吗? | 范式概念和训练技巧平铺 |
| **可溯源** | 每个判断性描述有来源吗? | "首创"/"SOTA" 无出处 |
| **不腐烂** | 所有信息与当前 vault 状态一致吗? | 年份错挂、重复条目、已删论文仍在 |
| **不过载** | 单页条目数在可维护范围内吗? | 论文列表 > 50 条 |

### 具体规则

#### R1. MOC 身份与边界

- 每个 MOC 文件**必须**以 frontmatter 或注释声明 scope:
  ```markdown
  <!-- scope: 本页覆盖零样本语音合成(zero-shot TTS / voice cloning)的概念、代表模型和相关论文。
       不包含: 纯 VC、纯歌声生成、通用音频生成、ASR-only、纯 vocoder、纯 watermark/safety。 -->
  ```
- MOC 是**导航页**,不是任务定义页。任务定义放 `任务库/`。两者通过 wikilink 互引,不合并。
- MOC 与同名任务页的分工: MOC 回答"这个领域有什么",任务页回答"这个任务是什么"。

#### R2. 概念分层 (核心概念区)

核心概念区**必须**按抽象层级分组,不得平铺:

| 层级 | 含义 | 示例 |
|------|------|------|
| 范式概念 | 领域级方法论 | LLM-based TTS, Non-autoregressive TTS |
| 表征/模块概念 | 可复用的技术模块 | Speech Tokenizer, Duration Predictor |
| 训练/优化技巧 | 通用底层技巧 | Gumbel-Softmax, Gradient Reversal Layer |

分组格式:
```markdown
## 核心概念

### 范式
- [[LLM-basedTTS]] — ...

### 表征与模块
- [[SpeechTokenizer]] — ...

### 训练技巧
- [[Gumbel-Softmax]] — ...
```

#### R3. 代表模型准入

"代表模型"区只放**该任务的主线系统**。准入规则:

| 类别 | 放在 | 判定 |
|------|------|------|
| 主线系统 | 代表模型 | 论文核心任务 = 本 MOC 主题 |
| 上游 backbone / tokenizer | "相关上游模型"子节 | 为主线系统提供组件,但自身不是该任务的系统 |
| 通用组件 (vocoder 等) | "关键组件"子节或不收录 | 跨多任务使用的通用工具 |

示例: MinMo 是 CosyVoice 3 的 tokenizer backbone → 放"相关上游模型",不放"代表模型"。BigVGAN 是通用 vocoder → 放"关键组件"或不收录。

#### R4. 论文区容量与溢出

- 单个 MOC 的"相关论文"区 **≤ 50 条**。超过时拆为 sub-MOC。
- 拆分策略: 按技术路线拆(如 `零样本语音合成-AR与Codec LM.md`),总览页保留 sub-MOC 链接 + 每路线 Top 3 代表作。
- 每条论文描述 **≤ 1 行**(论文名 + 年份 + 机构 + 一句话贡献)。

#### R5. 年份取值规则

- 论文在 MOC 中的年份 section 归属**必须**从该论文笔记 frontmatter 的 `year` 字段取值。
- 禁止手动判断年份。如 frontmatter 无 `year` 字段,先补 frontmatter 再收录。
- 同一论文在同一 MOC 中**只出现一次**。禁止重复 wikilink。

#### R6. SOTA 表规则

MOC 中的 SOTA/性能对比表**必须**标注:

| 必须标注 | 示例 |
|----------|------|
| benchmark 名称 + split | SEED-TTS-Eval test-zh |
| metric 实现 | WER by Paraformer, SS by ERes2Net |
| 数据来源 | "作者报告值" 或 "复现值" |
| 更新日期 | `(截至 2026-06)` |
| 可比性声明 | "以下结果来自不同实验设置,不可直接横比" 或 "以下结果基于统一 benchmark" |

不满足标注要求的 SOTA 表**禁止保留**。如无法补全,替换为"代表结果参考(非严格对比)"并降低结构层级。

#### R7. 演进脉络关系类型

MOC 中的演进关系**必须**使用以下类型标注:

| 符号 | 含义 | 示例 |
|------|------|------|
| `→` | 直接后继(同团队/明确声明) | VITS → VITS2 |
| `⇢` | 借鉴/启发(不同团队,局部复用) | VALL-E ⇢ CosyVoice |
| `∥` | 并行/互补(同体系,不同方向) | AudioLM ∥ SoundStorm |
| `⊃` | 包含/集成(B 把 A 作为组件) | MinMo ⊃ CosyVoice 3 |

禁止对不同团队、无明确声明的工作使用 `→`。不确定时用 `⇢` 并附注理由。

#### R8. 强判断标注

MOC 中的判断性描述**必须**标注来源类型:

| 来源类型 | 标注格式 | 示例 |
|----------|----------|------|
| 作者 claim | `(作者 claim)` | "首个端到端零样本 TTS (作者 claim)" |
| 基于 benchmark | `(benchmark: XX)` | "SOTA (benchmark: SEED-TTS-Eval)" |
| agent 整理 | `(agent 整理)` | "代表连续值 AR 路线 (agent 整理)" |

以下词汇视为强判断,必须标注: "首创"、"首个"、"SOTA"、"唯一"、"全面优于"、"远超"、"开创"、"填补空白"、"最佳"、"最优"。

### 运维规则 (保留并扩展原有条目)

- **自动创建**: 某主题下 ≥ 3 个 active 实体页时,自动创建该主题的 sub-MOC
- **收录标准**: 只有 deep + repro 层级笔记可出现在 MOC 论文列表中(card/enhanced-card 不收录)
- **deprecated 处理**: 移至 MOC 底部 `## 历史参考` section
- **merged 处理**: 从 MOC 中移除(不显示)
- **"被 MOC 包含"定义**: 至少一个 MOC 中有直接 `[[wikilink]]` 指向该页
- **自动刷新**: 每次 ingest 后检查并更新相关 MOC
- **人工备注保护**: MOC 中 `> [!note] 人工备注` 区块在刷新时**必须保留**,不得覆盖
- **scope 声明保护**: MOC 的 scope 注释在刷新时**必须保留**,不得覆盖
```

- [ ] **Step 3: Apply the edit**

Use the Edit tool to replace the current §11 content in `AGENTS.md`. The `old_string` is:

```
## 11. MOC 规则

- **自动创建**: 某主题下 ≥ 3 个 active 实体页时，自动创建该主题的 sub-MOC
- **收录标准**: 只有 deep + repro 层级笔记可出现在 MOC 论文列表中（card/enhanced-card 不收录）
- **deprecated 处理**: 移至 MOC 底部 `## 历史参考` section
- **merged 处理**: 从 MOC 中移除（不显示）
- **"被 MOC 包含"定义**: 至少一个 MOC 中有直接 `[[wikilink]]` 指向该页
- **自动刷新**: 每次 ingest 后检查并更新相关 MOC
- **人工备注保护**: MOC 中 `> [!note] 人工备注` 区块在刷新时**必须保留**，不得覆盖
```

The `new_string` is the full expanded §11 from Step 2 above.

- [ ] **Step 4: Verify the edit**

Run: `grep -n "## 11\. MOC" AGENTS.md` to confirm §11 now starts at the expected line.
Run: `grep -c "^####" AGENTS.md` to confirm the R1-R8 rules are present.
Run: `grep "R[1-8]\." AGENTS.md` to confirm all 8 rules exist.

Expected: 8 rule headers (R1 through R8) present in §11.

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md
git commit -m "[update/system] AGENTS.md §11 — MOC 规则从 7 行扩展为完整治理体系 (宪法 5 原则 + R1-R8 规则 + 运维规则)"
```

---

## Task 2: V2 — MOC Review Checklist

**Files:**
- Create: `_templates/moc-review-checklist.md`
- Modify: `_templates/review-checklist.md` (add cross-reference only)

### Context for Implementer

This checklist mirrors the three-layer structure of the existing `_templates/review-checklist.md` (宪法层 → 法律层 → 判例层), but scoped to MOC review instead of paper note review. The 5 宪法 principles come directly from the G2 rules written in Task 1. The 检查项 (法律层) operationalize those principles into verifiable items. The checklist will be used by the reviewer agent (Task 3's E2 upgrade adds MOC review mode).

- [ ] **Step 1: Read the existing review-checklist for structural reference**

Read `/Users/xiangshu/PaperWiki/_templates/review-checklist.md` to confirm the three-layer structure pattern:
- 第一层: 原则 (table with 原则/使用场景/核心问题)
- 第二层: 检查项 (grouped by principle, checkbox format)
- 问题词汇表
- 严重度
- 审阅结论
- 更新记录

- [ ] **Step 2: Create `_templates/moc-review-checklist.md`**

Write the following file:

```markdown
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
```

- [ ] **Step 3: Add cross-reference in existing review-checklist.md**

Edit `/Users/xiangshu/PaperWiki/_templates/review-checklist.md` to add a note after the opening quote block. The `old_string` is:

```
> 原则是宪法,检查项是法律,案例是判例法。
> 原则不变,检查项随系统演进更新。

---
```

The `new_string` is:

```
> 原则是宪法,检查项是法律,案例是判例法。
> 原则不变,检查项随系统演进更新。
>
> 本 checklist 用于论文笔记审阅。MOC 审阅使用 `moc-review-checklist.md`。

---
```

- [ ] **Step 4: Verify**

Run: `wc -l _templates/moc-review-checklist.md` — expect ~170 lines.
Run: `grep -c "^\- \[ \]" _templates/moc-review-checklist.md` — expect 16 checkbox items.
Run: `grep "moc-review-checklist" _templates/review-checklist.md` — expect 1 match.

- [ ] **Step 5: Commit**

```bash
git add _templates/moc-review-checklist.md _templates/review-checklist.md
git commit -m "[create/spec] moc-review-checklist.md — MOC 三层审阅标准 (5 原则 + 16 检查项 + 13 问题类型)"
```

---

## Task 3: E2 — Reviewer Independence (Subagent Dispatch)

**Files:**
- Modify: `~/.claude/skills/paperwiki-reader/SKILL.md` (lines 91-119, Step 3.5)
- Modify: `~/.claude/skills/paperwiki-reviewer/SKILL.md` (add standalone dispatch mode)

### Context for Implementer

Currently, `paperwiki-reader` Step 3.5 runs the review inline — the same agent that generated the note also reviews it, sharing the full generation context (PDF text, KB pages, intermediate reasoning). This violates the "生成者不审自己" principle (Spec §0.5).

The fix: Step 3.5 must dispatch `paperwiki-reviewer` as an independent subagent using Claude Code's `Agent` tool. The subagent receives ONLY: (1) the generated note file path, (2) the review checklist path, (3) relevant KB page paths. It does NOT receive: the PDF text, the generation prompt, intermediate reasoning, or any other pipeline context.

The `paperwiki-reviewer` skill itself also needs a small update: it currently assumes it's invoked inline and references the caller's context. It needs a clear "standalone mode" where it works entirely from the file paths provided.

Additionally, the reviewer skill gains a new MOC review mode, triggered by user command "审阅 MOC" or programmatically for periodic MOC audits. This mode uses `moc-review-checklist.md` instead of `review-checklist.md`.

### Important: How Subagent Dispatch Works in Claude Code

The `Agent` tool creates a fresh session with no shared context. Communication is via the `prompt` parameter. The subagent has access to Read/Write/Edit/Bash/Grep/Glob tools. It does NOT share the parent agent's conversation history.

- [ ] **Step 1: Read current paperwiki-reader Step 3.5**

Read `/Users/xiangshu/.claude/skills/paperwiki-reader/SKILL.md` lines 91-119 to confirm the current inline review implementation.

- [ ] **Step 2: Rewrite Step 3.5 in paperwiki-reader to use subagent dispatch**

Edit `/Users/xiangshu/.claude/skills/paperwiki-reader/SKILL.md`. The `old_string` is the entire Step 3.5 section (from `## Step 3.5: 审阅` to the end of step 5's commit block, before `## Step 4`):

```
## Step 3.5: 审阅 (反向更新前自动触发)

生成笔记后,在反向更新前,触发审阅:

1. 先 commit 草稿(不含反向更新):
   ```bash
   git add "论文笔记/xxx.md"
   git commit -m "[ingest/deep/draft] xxx — 草稿,待审阅"
   ```

2. 对照 `_templates/review-checklist.md` 检查 5 维度 × 10 问题类型:
   - 可理解性(因果解释 vs 组件罗列)
   - 可溯源性(claim 标注覆盖率)
   - 严谨性(过强断言/指标名)
   - 可导航性(挂接/KB 背景/速查卡片)
   - 知识库安全性(概念准入/反向更新风险)

3. 输出 `_review/xxx-review.yml` + 笔记末尾 `[!review]` callout

4. 根据审阅结论:
   - pass / pass-with-fixes → 继续 Step 4 反向更新
   - revise → 提示用户"有 high 问题,是否仍继续反向更新?"
   - reject-as-deep → 建议降为 enhanced-card,不做反向更新

5. commit 审阅报告:
   ```bash
   git add "_review/xxx-review.yml" "论文笔记/xxx.md"
   git commit -m "[review/auto] xxx — conclusion: {结论}, {N} issues"
   ```
```

The `new_string` is:

```
## Step 3.5: 审阅 (独立 subagent dispatch)

**原则: 生成者不审自己。** 审阅在独立 session 中运行,不共享生成上下文。

1. 先 commit 草稿(不含反向更新):
   ```bash
   git add "论文笔记/xxx.md"
   git commit -m "[ingest/deep/draft] xxx — 草稿,待审阅"
   ```

2. 收集 subagent 需要的信息(仅文件路径,不传内容):
   - `note_path`: 刚生成的笔记文件路径 `论文笔记/xxx.md`
   - `checklist_path`: `_templates/review-checklist.md`
   - `kb_paths`: Step 2 中命中的 KB 页面路径列表(最多 6 个)
   - `review_output`: `_review/xxx-review.yml`

3. 使用 Agent 工具 dispatch 独立审阅 subagent:
   ```
   Agent({
     description: "论文笔记审阅",
     prompt: `你是 PaperWiki 的独立审阅 agent。
   
   ## 任务
   审阅论文笔记,输出审阅报告。
   
   ## 输入
   - 待审笔记: {note_path}
   - 审阅标准: {checklist_path}
   - KB 参考页: {kb_paths 逐个列出}
   - 报告输出: {review_output}
   
   ## 执行步骤
   1. 读取审阅标准 ({checklist_path})
   2. 读取待审笔记 ({note_path})
   3. 读取 KB 参考页(用于交叉验证定位质量)
   4. 如果 Sources/ 下有同名 PDF,读取用于交叉验证关键数字
   5. 执行两层审阅(原则判断 + 检查项验证)
   6. 输出 _review/{论文名}-review.yml
   7. 在笔记末尾追加 [!review] callout
   
   ## 约束
   - 不重写笔记正文,只在末尾追加 callout
   - 每个问题必须有 location + detail + suggestion
   - 原则判断给 1-10 分 + 理由
   - 发现新问题类型记入 learning_signals
   
   ## 输出
   完成后报告: 审阅结论(pass/pass-with-fixes/revise/reject-as-deep) + 问题数量摘要`
   })
   ```

4. 解析 subagent 返回的审阅结论:
   - pass / pass-with-fixes → 继续 Step 4 反向更新
   - revise → 提示用户"有 high 问题,是否仍继续反向更新?"
   - reject-as-deep → 建议降为 enhanced-card,不做反向更新

5. commit 审阅报告:
   ```bash
   git add "_review/xxx-review.yml" "论文笔记/xxx.md"
   git commit -m "[review/auto] xxx — conclusion: {结论}, {N} issues (independent reviewer)"
   ```
```

- [ ] **Step 3: Update paperwiki-reviewer skill description and add MOC review mode**

Edit `/Users/xiangshu/.claude/skills/paperwiki-reviewer/SKILL.md`. There are two changes:

**Change A:** Update the YAML frontmatter `description` to reflect both invocation modes.

The `old_string` for the description is:

```yaml
name: paperwiki-reviewer
description: |
  PaperWiki 论文笔记审阅 skill。在精读笔记生成后、反向更新前运行。
  两层评估: 原则层(稳定锚点) + 检查项层(动态演进)。

  触发: paperwiki-reader skill 精读完成后自动调用,或用户说"审阅这篇笔记"
```

The `new_string` is:

```yaml
name: paperwiki-reviewer
description: |
  PaperWiki 审阅 skill。支持两种审阅模式:
  1. 论文笔记审阅 — 精读后自动 dispatch 或用户说"审阅这篇笔记"
  2. MOC 审阅 — 用户说"审阅 MOC"/"审阅这个 MOC"

  两层评估: 原则层(稳定锚点) + 检查项层(动态演进)。
  设计为独立 subagent 运行,不共享生成上下文。

  触发: paperwiki-reader 以 subagent dispatch 调用,或用户手动调用
```

**Change B:** After the existing "## 约束 (红线)" section at the end of the file, append a new MOC review mode section.

The `old_string` is the last constraint:

```
6. **检查项通过但原则不满足时要标记** — 这是系统进化的关键信号
```

The `new_string` is:

```
6. **检查项通过但原则不满足时要标记** — 这是系统进化的关键信号

---

## MOC 审阅模式

当用户说"审阅 MOC"/"审阅这个 MOC"时,切换到 MOC 审阅模式。

### Step 0: 读取 MOC 审阅标准

读取 `/Users/xiangshu/PaperWiki/_templates/moc-review-checklist.md` 获取:
- 5 条 MOC 评估原则(可导航/不混层/可溯源/不腐烂/不过载)
- 当前检查项(对应 AGENTS.md §11 R1-R8 规则)
- MOC 问题词汇表

### Step 1: 读取待审 MOC

读取指定的 `_MOC/*.md` 文件。

### Step 2: 收集验证数据

为检查项验证收集必要数据:
- 论文年份: 对 MOC 中引用的论文笔记,读取 frontmatter `year` 字段(抽样 10 篇或全部)
- 重复检测: 提取所有 wikilink,检查同一 link 是否出现多次
- 死链检测: 检查所有 wikilink 目标文件是否存在
- 论文区计数: 统计"相关论文"区的条目总数

```bash
# 提取所有 wikilink 并检查重复
grep -oP '\[\[([^\]|]+)' "_MOC/xxx.md" | sort | uniq -d

# 统计论文区条目数
grep -c "^\- \[\[论文笔记/" "_MOC/xxx.md"
```

### Step 3: 两层审阅

与论文笔记审阅相同的两层结构,但使用 MOC 原则和 MOC 检查项。

### Step 4: 输出

- YAML 报告: `_review/moc-review-{MOC名}-YYYY-MM-DD.yml`(格式见 moc-review-checklist.md)
- 不修改 MOC 文件本身(审阅只诊断不修改)

### Step 5: 行动建议

- **pass**: "MOC 审阅通过,结构健康。"
- **pass-with-fixes**: "有 N 个 medium 问题,建议修正。"
- **revise**: "有 N 个 high 问题(如年份错挂/死链),需修正。"
- **restructure**: "多条原则不满足,建议结构性重组(如拆分 sub-MOC)。"
```

- [ ] **Step 4: Verify skill file integrity**

Run: `grep "独立 subagent" ~/.claude/skills/paperwiki-reader/SKILL.md` — expect 1 match in Step 3.5 title.
Run: `grep "MOC 审阅模式" ~/.claude/skills/paperwiki-reviewer/SKILL.md` — expect 1 match as section header.
Run: `grep "moc-review-checklist" ~/.claude/skills/paperwiki-reviewer/SKILL.md` — expect 1 match.
Run: `grep "Agent(" ~/.claude/skills/paperwiki-reader/SKILL.md` — expect 1 match in the dispatch block.

- [ ] **Step 5: Commit**

```bash
git add ~/.claude/skills/paperwiki-reader/SKILL.md ~/.claude/skills/paperwiki-reviewer/SKILL.md
git commit -m "[update/system] E2: 审阅拆为独立 subagent dispatch — reader 不再内联审阅, reviewer 新增 MOC 审阅模式"
```

---

## Task 4: Integration — Update CLAUDE.md + Cross-Verification

**Files:**
- Modify: `CLAUDE.md` (vault state section)

### Context for Implementer

CLAUDE.md contains a "当前 vault 状态" section that needs to reflect the new files. Also need to verify that all three changes (G2, V2, E2) are internally consistent — the rule numbers referenced in the checklist match the rules in AGENTS.md, and the reviewer skill references the correct checklist file.

- [ ] **Step 1: Update CLAUDE.md vault state**

Read `CLAUDE.md` and find the "当前 vault 状态" section. Edit to add:

After the line about 审阅报告, add:
```
- MOC 审阅标准: moc-review-checklist.md (5 原则 + 16 检查项)
- MOC 规则: AGENTS.md §11 (宪法 5 原则 + R1-R8 具体规则 + 运维规则)
```

- [ ] **Step 2: Update CLAUDE.md pipeline description**

In the "精读论文的完整流程" section, update step ④ description:

The `old_string` is:
```
④ 审阅（反向更新前）: 对照 checklist 检查,输出 _review/xxx-review.yml
```

The `new_string` is:
```
④ 审阅（独立 subagent dispatch）: dispatch reviewer agent,输出 _review/xxx-review.yml
```

- [ ] **Step 3: Cross-verify rule references**

Run the following to verify consistency:

```bash
# Verify R1-R8 in AGENTS.md exist
grep "^#### R[0-9]" AGENTS.md

# Verify checklist references R1-R8
grep "\[R[0-9]\]" _templates/moc-review-checklist.md

# Verify reviewer references correct checklist
grep "moc-review-checklist" ~/.claude/skills/paperwiki-reviewer/SKILL.md

# Verify reader references Agent dispatch
grep -A2 "Step 3.5" ~/.claude/skills/paperwiki-reader/SKILL.md
```

Expected: R1 through R8 present in both AGENTS.md and moc-review-checklist.md. Reviewer references moc-review-checklist.md. Reader Step 3.5 mentions "独立 subagent".

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "[update/system] CLAUDE.md — 同步 Batch 1 变更 (MOC 规则 + MOC 审阅 + 审阅独立化)"
```

---

## Task 5: Validation — End-to-End Dry Run

**Files:**
- No new files created. This task validates Tasks 1-4.

### Context for Implementer

Before declaring Batch 1 complete, run a dry verification that the three changes work together. This is NOT a full review — it's a structural check.

- [ ] **Step 1: Validate G2 — MOC rules are parseable**

```bash
# Count rules in §11
grep "^#### R[0-9]" AGENTS.md | wc -l
# Expected: 8

# Verify 宪法 principles table
grep "可导航\|不混层\|可溯源\|不腐烂\|不过载" AGENTS.md | head -5
# Expected: 5 lines from the principles table
```

- [ ] **Step 2: Validate V2 — Checklist structure**

```bash
# Verify three-layer structure
grep "^## 第一层\|^## 第二层\|^## 问题词汇表\|^## 严重度\|^## 审阅结论" _templates/moc-review-checklist.md
# Expected: 5 section headers

# Verify all 5 principles in checklist match AGENTS.md
for p in 可导航 不混层 可溯源 不腐烂 不过载; do
  agents_count=$(grep -c "$p" AGENTS.md)
  checklist_count=$(grep -c "$p" _templates/moc-review-checklist.md)
  echo "$p: AGENTS=$agents_count, checklist=$checklist_count"
done
# Expected: all principles present in both files
```

- [ ] **Step 3: Validate E2 — Skill file structure**

```bash
# Reader has Agent dispatch
grep -c "Agent(" ~/.claude/skills/paperwiki-reader/SKILL.md
# Expected: 1

# Reviewer has MOC mode
grep -c "MOC 审阅模式" ~/.claude/skills/paperwiki-reviewer/SKILL.md
# Expected: 1

# Reviewer is still marked as fork context (independent session)
grep "context: fork" ~/.claude/skills/paperwiki-reviewer/SKILL.md
# Expected: 1 match
```

- [ ] **Step 4: Spot-check one MOC against new rules**

Read `_MOC/零样本语音合成.md` and manually verify:
1. Does it have a scope declaration? (Expected: NO — this is a known gap the new rules will catch)
2. Are concepts layered by abstraction? (Expected: NO — flat list, which R2 now requires fixing)
3. Are there backbone models in "代表模型"? (Expected: YES — MinMo, BigVGAN, which R3 now catches)

This confirms the new rules would catch the problems identified in the MOC review. Report findings but do NOT modify the MOC — that is a separate task.

- [ ] **Step 5: Final commit message**

No code change. Report: "Batch 1 validation complete. G2 (8 rules), V2 (16 checks), E2 (subagent dispatch) are structurally consistent. Existing MOCs will fail new checks as expected — MOC remediation is a separate task."
