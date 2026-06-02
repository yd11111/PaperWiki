# Paper Note Review Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a quality diagnosis layer to the paper reading pipeline — review notes BEFORE they touch the knowledge skeleton, accumulate findings, and use patterns to improve the system.

**Architecture:** A reviewer sub-agent runs after note generation but before reverse update. It produces structured YAML output stored in `_review/`. Every 5-10 notes, a pattern analysis produces system upgrade suggestions. The pipeline changes from "generate → update → lint" to "generate → review → (human decides) → update → lint".

**Tech Stack:** Claude Code skills (reviewer agent), Markdown + YAML (output format), Git (versioning)

---

## Pipeline Change (Before → After)

```
BEFORE:                          AFTER:
① 读原文                         ① 读原文
② KB 检索                        ② KB 检索
③ 生成笔记                       ③ 生成笔记草稿
④ 反向更新                       ④ 审阅子 agent ← NEW
⑤ 局部 lint                     ⑤ Git commit 草稿 (不含反向更新)
⑥ Git commit                    ⑥ 人工看(草稿+审阅报告)决定是否通过
⑦ log                           ⑦ 通过后才做反向更新
                                 ⑧ 局部 lint
                                 ⑨ Git commit + log
```

关键变化: **审阅在反向更新之前,防止错误污染概念页。**

## File Structure

```
需要创建:
├── _review/                           ← 审阅报告存放
│   ├── {论文名}-review.yml            ← 单篇审阅结果(YAML)
│   └── pattern-analysis-YYYY-MM-DD.md ← 跨篇模式分析报告
├── _templates/
│   └── review-checklist.md            ← 审阅 checklist 模板
├── ~/.claude/skills/paperwiki-reviewer/
│   └── SKILL.md                       ← 审阅子 agent skill

需要修改:
├── AGENTS.md                          ← 更新 pipeline 流程 + 审阅规则
├── CLAUDE.md                          ← 更新 pipeline 摘要
├── ~/.claude/skills/paperwiki-reader/
│   └── SKILL.md                       ← pipeline 插入审阅步骤
```

---

## Task 1: Define review checklist template

**Files:**
- Create: `_templates/review-checklist.md`

- [ ] **Step 1: Write the review checklist template**

```markdown
# 审阅 Checklist

## 问题分类体系 (10 类)

| type | 含义 | 典型表现 |
|------|------|----------|
| factual-error | 事实/数字错误 | 指标名写错(CER↔WER),数字抄错,模型名写错 |
| traceability-gap | claim 无出处 | 关键数字没有 [§X.X]/[Table N] 标注 |
| overclaim | 结论超出证据 | "唯一""全面优于""SOTA"但证据不支持 |
| fact-inference-mixing | 事实与推断混写 | 把 agent 的解释写成论文的结论,没区分 |
| summary-without-mechanism | 像摘要不像 deep | "方法"节只描述 WHAT(组件列表),不解释 WHY(因果链) |
| missing-lineage | 缺谱系定位 | KB 背景节没有具体定位,或完全缺失 |
| weak-reusability | 可借鉴太泛 | "提出了新方法"(贡献声明)而非"X trick 可迁移到 Y 场景" |
| bad-linking | 概念挂接不合理 | 遗漏明显相关概念,或挂接了不相关概念 |
| template-compliance | 格式不合规 | frontmatter 缺字段,缺 section,速查卡片字段空 |
| kb-safety-risk | 可能污染知识库 | 新建概念不满足准入规则,或反向更新内容有事实错误 |

## 严重度定义

| 级别 | 含义 | 行动 |
|------|------|------|
| high | 影响可信性,会误导后续使用 | 必须修正才能反向更新 |
| medium | 影响质量,但不阻塞 | 建议修正,可标注后放行 |
| low | 影响可读性/复用性 | 可忽略或后续批量改善 |

## 审阅结论

| 结论 | 含义 |
|------|------|
| pass | 可直接反向更新 |
| pass-with-fixes | 有 medium 问题,建议修正但可放行 |
| revise | 有 high 问题,需修正后重新审阅 |
| reject-as-deep | 质量不达 deep 标准,降为 enhanced-card |

## 审阅维度 (5 维)

### 1. 可理解性 (understandability)
- "方法:它怎么 work" 是否包含因果解释(WHY)
- 检查: 有无"之所以/因为/这使得/从而" vs 仅有"使用了/采用了/包含"
- 检查: 每个关键设计选择是否回答"为什么选这个"

### 2. 可溯源性 (traceability)
- 数字型 claim 是否都有 [§X.X]/[Table N] 标注
- 计算: 标注覆盖率 = 有标注 claim 数 / 总数字 claim 数
- 检查: 是否存在事实与推断混写(agent 解释 vs 作者原文)

### 3. 严谨性 (rigor)
- 指标名是否正确(CER/WER/MOS/SIM 不混淆)
- baseline 对比措辞是否适当("优于"需要证据,"领先"需要具体数字)
- "SOTA/唯一/全面" 等强断言是否有充分支撑

### 4. 可导航性 (navigability)
- concepts/models/tasks/datasets 挂接是否合理
- KB 背景是否有具体谱系定位(不是"与已有工作相关")
- 速查卡片 5 字段是否都有实质内容

### 5. 知识库安全性 (kb-safety)
- 新建概念页是否满足准入规则(多篇引用/前置知识/连接论文 ≥2)
- 反向更新内容是否包含 factual-error 或 overclaim
- 如果直接进入可信层,是否会造成误导
```

- [ ] **Step 2: Commit**

```bash
git add _templates/review-checklist.md
git commit -m "[init] review checklist template — 10 问题类型 + 3 严重度 + 5 审阅维度"
```

---

## Task 2: Create review output directory

**Files:**
- Create: `_review/.gitkeep`

- [ ] **Step 1: Create directory**

```bash
mkdir -p _review
touch _review/.gitkeep
```

- [ ] **Step 2: Commit**

```bash
git add _review/
git commit -m "[init] _review/ directory for structured review reports"
```

---

## Task 3: Create paperwiki-reviewer skill

**Files:**
- Create: `~/.claude/skills/paperwiki-reviewer/SKILL.md`

- [ ] **Step 1: Write the reviewer skill**

```markdown
---
name: paperwiki-reviewer
description: |
  PaperWiki 论文笔记审阅 skill。在精读笔记生成后、反向更新前运行。
  对照 checklist 检查笔记质量,输出结构化 YAML 审阅报告。

  触发: paperwiki-reader skill 精读完成后自动调用,或用户说"审阅这篇笔记"
context: fork
allowed-tools: Bash, Read, Grep, Glob
---

# PaperWiki 论文笔记审阅

你是独立于生产 agent 的**审阅 agent**。你的职责是找问题、分类问题、评估严重度。
**你不是第二个写作 agent** — 不要重写笔记,只做诊断。

## Step 0: 读取审阅标准

读取 `/Users/xiangshu/PaperWiki/_templates/review-checklist.md` 获取:
- 10 类问题分类
- 3 级严重度定义
- 5 个审阅维度的具体检查方法
- 4 种审阅结论

## Step 1: 读取待审笔记

读取被指定的论文笔记文件(由调用者告知路径)。

## Step 2: 逐维度检查

### 2.1 可理解性 (understandability)

扫描 "## 方法: 它怎么 work" 节:

**检查因果解释:**
```bash
# 因果连接词(好的信号)
grep -c "之所以\|因为\|这使得\|从而\|目的是\|解决了\|原因\|所以\|导致\|才能" 论文笔记/xxx.md

# 组件罗列词(差的信号)
grep -c "使用了\|采用了\|包含了\|由.*组成\|利用了" 论文笔记/xxx.md
```

如果 组件罗列词 > 因果连接词 × 2 → warning "方法节偏向组件罗列"

**检查设计选择解释:**
扫描 "### 关键设计选择" 节,每个选择是否有"为什么选这个而不是那个"。

### 2.2 可溯源性 (traceability)

**Claim 标注覆盖率:**
```bash
# 提取所有包含数字的句子(潜在 claim)
grep -n "[0-9]\+\.\?[0-9]*[%kKMBh]" 论文笔记/xxx.md | wc -l

# 提取有标注的 claim
grep -n "\[§\|Table\|\[Fig\|\[Eq\|\[section" 论文笔记/xxx.md | wc -l
```

覆盖率 < 80% → high "标注覆盖率不足"

**事实/推断区分:**
扫描是否存在 agent 的推测性解释被写成断言(没有"可能/推测/作者认为"等限定词)。

### 2.3 严谨性 (rigor)

**指标名核对:**
检查常见混淆: CER↔WER, MOS↔DNSMOS, SIM↔SIM-O, SS↔Speaker Similarity

**过强断言检测:**
```bash
grep -n "唯一\|首个\|全面优于\|SOTA\|state-of-the-art\|最优" 论文笔记/xxx.md
```
每个匹配项检查: 附近是否有支撑证据(数字+出处)?

### 2.4 可导航性 (navigability)

**概念挂接检查:**
- frontmatter concepts/models/tasks/datasets 是否都有对应实体页?
- 论文明显讨论了某概念但 frontmatter 未列?

**KB 背景质量:**
- kb_context_sources > 0?
- "谱系定位"是否有具体的论文/模型名(不是泛泛而谈)?

**速查卡片:**
逐字段检查: 一句话(>15字+含方法名?)、路线(有→?)、指标(有数字?)、可借鉴(无"提出/贡献"?)、局限(有具体限制?)

### 2.5 知识库安全性 (kb-safety)

**新建概念检查:**
对 git diff 中新创建的概念页:
- 是否满足准入规则(多篇引用/前置知识/连接论文 ≥2)?
- 已有概念库是否有语义相近页?

**反向更新风险:**
如果审阅发现 factual-error 或 overclaim → 标记 "反向更新前需修正此问题"

## Step 3: 生成审阅报告

输出到 `/Users/xiangshu/PaperWiki/_review/{论文名}-review.yml`:

```yaml
paper: "论文名"
file: "论文笔记/xxx.md"
date: "YYYY-MM-DD"
conclusion: pass | pass-with-fixes | revise | reject-as-deep

scores:
  understandability: 1-10
  traceability: 1-10
  rigor: 1-10
  navigability: 1-10
  kb_safety: 1-10

issues:
  - severity: high | medium | low
    type: factual-error | traceability-gap | overclaim | fact-inference-mixing | summary-without-mechanism | missing-lineage | weak-reusability | bad-linking | template-compliance | kb-safety-risk
    location: "具体位置(节名/段落)"
    detail: "问题描述"
    suggestion: "最小修正建议"

risk_assessment:
  safe_for_reverse_update: true | false
  safe_for_trusted_layer: true | false
  reason: "如果 false,说明原因"

learning_signals:
  new_pattern: true | false
  matches_known_pattern: ""
  system_upgrade_suggestion: ""
```

同时在笔记末尾追加人类可读摘要:

```markdown
> [!review] 自动审阅 (YYYY-MM-DD)
> **结论:** pass-with-fixes
> **评分:** 理解 8 | 溯源 7 | 严谨 8 | 导航 9 | 安全 9
> **问题:** 2 medium, 1 low
> - ⚠️ [traceability-gap] 实验节第 3 段: 3 处数字无标注
> - ⚠️ [weak-reusability] 速查"可借鉴": "提出了新方法" → 应改为具体可迁移 trick
> - 💡 [template-compliance] 局限节只有 2 条,建议补充
> **反向更新:** ✅ 安全
> 详细报告: [[_review/xxx-review.yml]]
```

## Step 4: 报告行动建议

根据 conclusion:
- **pass**: "可直接进行反向更新"
- **pass-with-fixes**: "建议修正 N 个 medium 问题后反向更新,或标注后放行"
- **revise**: "有 N 个 high 问题必须修正。修正后重新审阅。"
- **reject-as-deep**: "质量不达 deep 标准,建议降为 enhanced-card"

## 约束

- **不重写笔记** — 只诊断,不替代
- **不阻断 pipeline** — 即使 revise 也只是建议,最终由人决定
- **输出必须结构化** — YAML 为主,callout 为辅
- **每个问题必须有 location + detail + suggestion** — 不给模糊评语
```

- [ ] **Step 2: Commit**

```bash
git add ~/.claude/skills/paperwiki-reviewer/SKILL.md
git commit -m "[init] paperwiki-reviewer skill — 结构化论文笔记审阅"
```

---

## Task 4: Update paperwiki-reader skill (insert review step)

**Files:**
- Modify: `~/.claude/skills/paperwiki-reader/SKILL.md`

- [ ] **Step 1: Modify pipeline in paperwiki-reader**

在 paperwiki-reader skill 的 Step 3 (生成笔记) 之后、Step 4 (反向更新) 之前,插入审阅步骤:

原来的 Step 3 → Step 4 之间加:

```markdown
## Step 3.5: 审阅 (自动触发)

生成笔记后,在反向更新前,调用 paperwiki-reviewer skill 审阅草稿:

1. 先 commit 草稿 (不含反向更新):
   ```bash
   git add "论文笔记/xxx.md"
   git commit -m "[ingest/deep/draft] xxx — 草稿,待审阅"
   ```

2. 运行审阅:
   - 读取 _templates/review-checklist.md
   - 对照 5 维度 + 10 问题类型检查
   - 输出 _review/xxx-review.yml + 笔记末尾 callout

3. 根据审阅结论决定:
   - pass / pass-with-fixes → 继续 Step 4 反向更新
   - revise → 标记问题,提示用户 "有 high 问题需修正,是否继续反向更新?"
   - reject-as-deep → 建议降为 enhanced-card,不做反向更新

4. commit 审阅报告:
   ```bash
   git add "_review/xxx-review.yml" "论文笔记/xxx.md"
   git commit -m "[review/auto] xxx — conclusion: pass-with-fixes, 2 issues"
   ```
```

- [ ] **Step 2: Commit**

```bash
git add ~/.claude/skills/paperwiki-reader/SKILL.md
git commit -m "[update/skill] paperwiki-reader — 插入审阅步骤(反向更新前)"
```

---

## Task 5: Update AGENTS.md (pipeline + review rules)

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Update pipeline flow in AGENTS.md**

将第七章"精读/复现 Pipeline"更新为:

```markdown
### 精读/复现 Pipeline (核心流程)

① 读原文(PDF / URL)
    ↓
② KB 上下文检索(只读可信层实体页)
    ↓
③ 生成笔记草稿(按模板,含 KB 背景 + 速查卡片)
   - status: draft
    ↓
④ 审阅(自动触发,反向更新前)
   - 对照 checklist 检查 5 维度 × 10 问题类型
   - 输出 _review/xxx-review.yml + 笔记末尾 callout
   - 结论: pass / pass-with-fixes / revise / reject-as-deep
    ↓
⑤ Git commit 草稿 + 审阅报告
    ↓
⑥ 反向更新(仅审阅通过后执行)
   - revise → 提示用户修正后重审
   - reject-as-deep → 不做反向更新
    ↓
⑦ 局部 lint
    ↓
⑧ Git commit + log
    ↓
⑨ 审核积压检查
```

- [ ] **Step 2: Add review rules section to AGENTS.md**

在 Lint 章节之后新增:

```markdown
## 审阅系统

### 单篇审阅

每次 deep/repro 笔记生成后、反向更新前自动触发。

**审阅输出:** `_review/{论文名}-review.yml` (结构化 YAML)
**笔记标注:** 笔记末尾追加 `> [!review]` callout 摘要

**审阅结论与行动:**
| 结论 | 行动 |
|------|------|
| pass | 直接反向更新 |
| pass-with-fixes | 建议修正 medium 问题,可标注后放行 |
| revise | 有 high 问题,修正后重审,不自动反向更新 |
| reject-as-deep | 建议降为 enhanced-card,不反向更新 |

**审阅原则:**
- 审阅 agent 只做诊断,不做重写
- 审阅不阻断 pipeline — 最终由人决定
- 每个问题必须有 location + detail + suggestion

### 跨篇模式分析

每 5-10 篇审阅后手动触发。

**输入:** `_review/` 下所有 .yml 文件
**输出:** `_review/pattern-analysis-YYYY-MM-DD.md`

**分析内容:**
- 高频问题类型统计
- 根因判断(来自模板/AGENTS.md/skill/检索?)
- 系统升级建议(具体指出改什么文件的什么规则)
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "[update/convention] AGENTS.md — pipeline 加入审阅步骤 + 审阅系统规则"
```

---

## Task 6: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update pipeline summary in CLAUDE.md**

将精读流程摘要更新:

```markdown
## 精读论文的完整流程

① 读原文 PDF
② KB 检索(搜索 confirmed 实体页)
③ 生成笔记草稿(含速查卡片 + KB 背景)
④ 审阅(自动,反向更新前): 对照 checklist 检查,输出 _review/xxx-review.yml
⑤ Git commit 草稿 + 审阅报告
⑥ 反向更新(仅审阅通过后): 追加已有页 / 创建新实体页
⑦ 局部 lint + Git commit + log
```

- [ ] **Step 2: Add reviewer skill to skill declaration**

```markdown
## Skill 使用声明

- **论文精读**: 使用 `paperwiki-reader` skill
- **笔记审阅**: 使用 `paperwiki-reviewer` skill(精读后自动触发,或手动"审阅这篇笔记")
- **不要**使用通用 `paper-reader` skill
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "[update/bootstrap] CLAUDE.md — pipeline 加入审阅步骤"
```

---

## Task 7: Update 系统总览

**Files:**
- Modify: `docs/系统总览.md`

- [ ] **Step 1: Add review module to system overview**

在"四、核心流程"中 4.8 之后添加:

```markdown
### 4.9 笔记审阅 (生成后 → 反向更新前)

**单篇审阅(每次 deep/repro 自动触发):**
1. 审阅子 agent 对照 checklist 检查 5 维度 × 10 问题类型
2. 输出 _review/xxx-review.yml (结构化) + 笔记末尾 callout (人类可读)
3. 结论决定是否允许反向更新(pass/revise/reject)

**跨篇模式分析(每 5-10 篇手动触发):**
1. 统计高频问题类型
2. 判断根因(模板/规则/skill/检索)
3. 产出系统升级建议
```

在"三、目录结构"中加入:

```markdown
├── _review/            审阅报告(YAML + 模式分析)
```

- [ ] **Step 2: Commit**

```bash
git add docs/系统总览.md
git commit -m "[docs] 系统总览 — 加入审阅模块描述"
```

---

## Task 8: End-to-end validation

- [ ] **Step 1: Pick a test paper**

使用 vault 中已有的一篇 draft 笔记(如 IndexTTS2 或 MaskGCT)做审阅测试。

- [ ] **Step 2: Run paperwiki-reviewer on the note**

```
审阅这篇笔记 [[论文笔记/IndexTTS2]]
```

- [ ] **Step 3: Verify output**

```bash
# YAML 报告存在
ls _review/IndexTTS2-review.yml

# YAML 格式正确(有 conclusion + scores + issues)
grep "conclusion:" _review/IndexTTS2-review.yml
grep "scores:" _review/IndexTTS2-review.yml
grep "issues:" _review/IndexTTS2-review.yml

# 笔记末尾有 callout
grep "\[!review\]" "论文笔记/IndexTTS2.md"
```

- [ ] **Step 4: Verify review quality**

人工检查:
- [ ] 每个 issue 都有 severity + type + location + detail + suggestion?
- [ ] 没有模糊评语("写得不错")?
- [ ] conclusion 是否合理(跟你自己的判断一致)?
- [ ] 没有重写笔记内容(只诊断)?

- [ ] **Step 5: Commit + tag**

```bash
git add -A
git commit -m "[review/auto] IndexTTS2 — 审阅模块 end-to-end 验证"
git tag review-module-v1
```

---

## Completion Criteria

| 检查项 | 验证方法 |
|---|---|
| _review/ 目录存在 | `ls _review/` |
| checklist 模板存在 | `ls _templates/review-checklist.md` |
| reviewer skill 被系统识别 | 出现在 available skills 列表 |
| AGENTS.md pipeline 更新 | grep "审阅" AGENTS.md |
| CLAUDE.md pipeline 更新 | grep "审阅" CLAUDE.md |
| 审阅报告 YAML 可解析 | 手动检查结构 |
| 笔记末尾有 callout | grep "[!review]" 论文笔记/xxx.md |
| 审阅质量: 无模糊评语 | 人工检查 |
| 审阅质量: 每个 issue 有完整字段 | 人工检查 |
| 审阅不重写笔记 | diff 验证笔记主体未变 |
