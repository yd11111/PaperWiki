# PaperWiki Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a TTS-focused personal knowledge base that turns papers into compounding, trustworthy cognitive assets with minimal attention cost.

**Architecture:** Obsidian vault managed by git. AGENTS.md as system schema. Four-tier notes (card/enhanced-card/deep/repro) with single-file-per-paper model. Concept pages as KB retrieval cache layer. All rules live in AGENTS.md, not in separate config docs.

**Tech Stack:** Obsidian (vault), Git (version control), Claude Code skills (automation), Markdown + YAML frontmatter (data format)

**Spec:** `docs/2026-06-01-paperwiki-system-design.md`

**Key design decisions baked in:**
- Sources/: PDFs in vault, gitignored. Traceability via frontmatter.
- One file per paper, always in `论文笔记/`. DailyPapers is index-only.
- Inbox items deleted after processing.
- Append (list/table tail) = no status change. Substantive (prose/relations) = pending-review.
- MOC: ≥3 active entities → auto-create. Only deep/repro notes in MOC.
- KB retrieval: 3 match conditions, priority sort, log hits/misses.

---

## Completion criteria philosophy

A phase is **complete** when:
1. The capability runs successfully on **≥ 3 diverse inputs**
2. At least **1 failure/edge case** has been triggered and handled correctly
3. The output matches spec (verifiable by schema check, not "looks reasonable")

A phase is **NOT** complete when:
- A specification document has been written
- It worked once on a happy path
- Files exist but haven't been exercised

---

## Phase P0: Foundation

**Delivers:** Empty but complete vault skeleton. All rules in AGENTS.md. All templates with correct schema. Git initialized.

**Complete when:** `AGENTS.md` can be read by an agent and used to produce a correctly-formatted note without additional instructions.

---

### Task 1: Git + directory structure + .gitignore

- [ ] **Step 1: Create .gitignore**

```gitignore
.obsidian/
.DS_Store
.trash/
Sources/*.pdf
```

- [ ] **Step 2: Create all directories**

```bash
mkdir -p Sources _inbox 论文笔记 概念库 模型库 任务库 数据集 DailyPapers/Weekly _MOC _lint _templates docs
```

- [ ] **Step 3: Create log.md**

```markdown
# Log

## 2026-06-01
- [init] Vault 骨架创建
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "[init] vault directory structure + .gitignore + log.md"
```

---

### Task 2: Write AGENTS.md

AGENTS.md is the **single source of truth** for all system rules. It must contain everything an agent needs to operate correctly — no separate config docs.

- [ ] **Step 1: Write AGENTS.md with full content from spec**

The file must include (in this order):
1. System overview (one-liner + three roles)
2. Directory semantics (what each dir is for, who writes, what goes in git)
3. Sources/ storage rule (PDF local + gitignore, traceability via frontmatter)
4. Note storage model (one file per paper in 论文笔记/, DailyPapers = index)
5. Six principles (verbatim from spec)
6. Four-tier note system (table + upgrade rules)
7. Append vs substantive update rules (formalized, with examples)
8. Entity page lifecycle (active/deprecated/merged + operations)
9. Inbox rules (zero-commitment, delete after processing)
10. KB retrieval rules (3 match conditions, sort priority, log format)
11. MOC rules (≥3 triggers creation, only deep/repro, deprecated/merged handling)
12. Lint rules (local checks list, what each checks)
13. Link conventions (wikilink format, core vs related distinction)
14. Commit format (prefix table + examples)
15. Claim annotation format (`[§X.X]`, `[Table N]`, `[Fig N]`)
16. Failure degradation table
17. Review backlog alert thresholds (10 pending-review, 5 draft deep)

- [ ] **Step 2: Verify completeness**

Read AGENTS.md and answer: could a fresh agent, given only this file and the templates, correctly:
- Create a deep note with proper frontmatter?
- Decide whether an update is append or substantive?
- Know when to create a new concept page vs mark [待决]?
- Know what to do when KB retrieval finds nothing?

If any answer is "no", fix the gap.

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "[init] AGENTS.md — complete system schema"
```

---

### Task 3: Write all templates (9 files)

Each template must have:
- Complete frontmatter with all fields from spec (correct types, correct enum values)
- Section structure matching spec
- Placeholder markers (`{{...}}`) for variable content
- No ambiguity about what goes where

- [ ] **Step 1: Write `_templates/paper-card.md`**

Frontmatter fields: type, tier(=card), title, arxiv_id, source, authors, year, venue, tags, concepts(=[]), models(=[]), tasks(=[]), datasets(=[]), status(=draft), created, updated

Sections: 一句话定位 / 方法关键词 / 为何值得关注

- [ ] **Step 2: Write `_templates/paper-enhanced-card.md`**

Same frontmatter as card but tier=enhanced-card.

Sections: 一句话定位 / 方法概述 / 关键结果 / 与已知方法简单对比 / 为何值得关注

- [ ] **Step 3: Write `_templates/paper-deep.md`**

Frontmatter adds: kb_context_sources

Sections: KB 背景 / 核心问题 / 方法:它怎么 work (整体架构 + 关键设计选择 + 训练策略) / 实验(含表格,每行有出处列) / 局限性 / 点评 / 可复用的 idea

- [ ] **Step 4: Write `_templates/paper-repro.md`**

Extends deep with: 模块细节(逐模块) / Loss 设计(公式) / 训练配置(optimizer/lr/batch/hardware) / 数据处理 / 推理流程 / 消融实验 / 复现要点

- [ ] **Step 5: Write `_templates/concept.md`**

Frontmatter: type, title, aliases, category, tags, key_papers, related_concepts, status(=pending-review), lifecycle(=active), merged_into, deprecated_reason, created, updated

Sections: 定义 / 在 TTS 中的应用 / 关键论文 / 相关概念 / 演进

- [ ] **Step 6: Write `_templates/model.md`**

Frontmatter: type, title, aliases, org, year, tags, key_concepts, tasks, key_papers, supersedes, superseded_by, status, lifecycle, merged_into, created, updated

Sections: 概述 / 核心方法 / 性能(表格) / 演进线 / 关键贡献

- [ ] **Step 7: Write `_templates/task.md`**

Frontmatter: type, title, aliases, tags, key_approaches, key_models, benchmarks, metrics, status, lifecycle, merged_into, created, updated

Sections: 问题定义 / 主流方法 / 代表模型 / 评估(Benchmarks + Metrics + 当前 SOTA 表) / 开放问题

- [ ] **Step 8: Write `_templates/dataset.md`**

Frontmatter: type, title, aliases, domain, scale, tags, used_by, metrics_reported_on, url, status, lifecycle, merged_into, created, updated

Sections: 概述 / 用途 / 使用此数据集的模型 / 注意事项

- [ ] **Step 9: Write `_templates/inbox-item.md`**

Frontmatter: type(=inbox), title, source, why, created

Sections: 备注(optional)

- [ ] **Step 10: Schema validation**

For each template, verify:
- [ ] All frontmatter fields match spec section 六 exactly
- [ ] Enum values match (tier options, status options, lifecycle options)
- [ ] No field from spec is missing
- [ ] No extra field not in spec exists

- [ ] **Step 11: Commit**

```bash
git add _templates/
git commit -m "[init] all 9 templates — schema-validated against spec"
```

---

### Task 4: Initial MOC + P0 completion

- [ ] **Step 1: Create `_MOC/TTS-总览.md`**

Top-level MOC with links to sub-topics (as placeholders for future MOCs) and pointers to entity libraries.

- [ ] **Step 2: Final log update + commit + tag**

```bash
# Update log
echo "- [init] P0 完成: AGENTS.md + 9 模板 + MOC 骨架" >> log.md
git add -A
git commit -m "[init] P0 complete — vault skeleton ready"
git tag p0-foundation
```

- [ ] **Step 3: P0 acceptance test**

Verify by answering these questions (all must be "yes"):
1. Does `AGENTS.md` contain all 6 principles? → grep for each principle header
2. Does every template have the exact frontmatter fields from spec? → diff check
3. Is Sources/*.pdf in .gitignore? → `git check-ignore Sources/test.pdf`
4. Is git configured correctly? → `git config user.name && git config user.email`
5. Can you open the vault in Obsidian and navigate TTS-总览? → manual check

---

## Phase P1: Core Engine

**Delivers:** Working capability: give it a PDF → get a deep note → entity pages created/updated → local lint validates. Proven on ≥ 3 papers including 1 failure scenario.

**Complete when:** You have 3+ deep notes, 5+ entity pages, all local lint passing, at least one edge case (e.g., PDF parse failure, concept already exists) handled correctly.

---

### Task 5: First deep-read ingest (happy path)

**Input:** A well-known TTS paper you've already read (for verifiable correctness).

- [ ] **Step 1: Place PDF in Sources/**

```bash
# Copy or download a test paper
cp ~/path/to/cosyvoice.pdf Sources/CosyVoice.pdf
```

- [ ] **Step 2: Run paper-reader**

```
读一下这篇论文 Sources/CosyVoice.pdf
```

- [ ] **Step 3: Validate note output (objective checks)**

Run these checks on the produced file in `论文笔记/`:

```bash
# Check file exists
ls 论文笔记/CosyVoice*.md

# Check required frontmatter fields present
grep "^type: paper" 论文笔记/CosyVoice*.md
grep "^tier: deep" 论文笔记/CosyVoice*.md
grep "^arxiv_id:" 论文笔记/CosyVoice*.md
grep "^source:" 论文笔记/CosyVoice*.md
grep "^status: draft" 论文笔记/CosyVoice*.md
grep "^concepts:" 论文笔记/CosyVoice*.md

# Check key sections exist
grep "## 核心问题" 论文笔记/CosyVoice*.md
grep "## 方法" 论文笔记/CosyVoice*.md
grep "## 实验" 论文笔记/CosyVoice*.md
grep "## 点评" 论文笔记/CosyVoice*.md

# Check claim annotations exist (at least some [§] or [Table] markers)
grep -c "\[§\|Table\|\[Fig" 论文笔记/CosyVoice*.md
# Should be > 0

# Check KB 背景 says P1 mode (no KB yet)
grep "KB 检索未启用\|未找到" 论文笔记/CosyVoice*.md
```

- [ ] **Step 4: Validate entity pages created**

```bash
# At least 1 concept page should exist
ls 概念库/
# At least 1 should have pending-review status
grep -l "status: pending-review" 概念库/*.md
# Check frontmatter completeness
grep "^type: concept" 概念库/*.md
grep "^lifecycle: active" 概念库/*.md
```

- [ ] **Step 5: Validate log entry**

```bash
grep "\[ingest/deep\]" log.md
```

- [ ] **Step 6: Validate git commit**

```bash
git log --oneline -1 | grep "\[ingest/deep\]"
```

- [ ] **Step 7: Run local lint manually**

Check all [[wikilinks]] in the new note resolve to real files:
```bash
grep -oP '\[\[([^\]|]+)' 论文笔记/CosyVoice*.md | sed 's/\[\[//' | while read link; do
  found=$(find . -name "${link}.md" -not -path "./.git/*" | head -1)
  if [ -z "$found" ]; then echo "DEAD LINK: $link"; fi
done
```

Expected: 0 dead links, OR dead links are logged as [待决] in log.md.

Note: MOC 覆盖检查不在 P1 范围内(P4 启用后生效)。P1 局部 lint 只检查:链接有效性 + frontmatter 完整性 + 实体存在性。

---

### Task 6: Second ingest (tests reverse update to existing entity)

**Input:** A paper that shares concepts with paper 1 (e.g., another flow-matching TTS paper).

- [ ] **Step 1: Ingest second paper**

```
读一下这篇论文 [second paper URL/PDF]
```

- [ ] **Step 2: Validate append update (not substantive)**

The concept page created in Task 5 (e.g., Flow Matching) should:
```bash
# Still be confirmed (if it was confirmed) or pending-review (if never confirmed)
# Key: status should NOT have changed due to append
grep "status:" 概念库/Flow\ Matching.md

# Should have new paper in key_papers
grep "CosyVoice\|second_paper" 概念库/Flow\ Matching.md
```

- [ ] **Step 3: Validate new entity pages for new concepts**

If paper 2 introduces concepts not in paper 1, verify new pages created with `status: pending-review`.

- [ ] **Step 4: Local lint on paper 2**

Same wikilink check as Task 5 Step 7.

---

### Task 7: Third ingest (different sub-area, tests breadth)

**Input:** A paper from a different TTS sub-area (e.g., codec-based if first two were flow-based).

- [ ] **Step 1: Ingest**

- [ ] **Step 2: Validate creates new concept pages in different area**

```bash
# Should see concept pages related to the new area
ls 概念库/
# Count should be higher than after Task 6
```

- [ ] **Step 3: Validate cross-links between shared concepts**

If papers share any concept (e.g., both mention "zero-shot TTS"), verify that concept page's `key_papers` list contains both papers.

---

### Task 8: Failure scenario — missing/corrupt PDF

- [ ] **Step 1: Test with a URL that might fail or a truncated PDF**

```
读一下这篇论文 https://arxiv.org/abs/0000.00000
```
(or provide a corrupted/empty file)

- [ ] **Step 2: Validate degradation behavior**

Expected:
- System does NOT crash or produce an empty note silently
- Error is logged as `[skip/ingest]` in log.md with reason
- No garbage entity pages created
- **No git commit for failed ingest** (only log.md is updated, then committed separately as `[skip/ingest] reason`)

---

### Task 9: Batch review + deep note review

- [ ] **Step 1: List all pending-review entity pages**

```bash
grep -rl "status: pending-review" 概念库/ 模型库/ 任务库/ 数据集/ 2>/dev/null
```

- [ ] **Step 2: Review each — change to confirmed or fix content**

For each page:
- Read content, verify accuracy against your knowledge
- If correct: change `status: pending-review` → `status: confirmed`
- If wrong: fix content, then confirm

- [ ] **Step 3: Review deep notes — change draft to reviewed**

```bash
grep -rl "status: draft" 论文笔记/ | head -5
```

For each deep note:
- Verify "方法: 它怎么 work" explains mechanism (not just restates abstract)
- Verify claims have `[§X.X]` or `[Table N]` annotations
- If acceptable: change `status: draft` → `status: reviewed`

- [ ] **Step 4: Commit review**

```bash
git add -A
git commit -m "[review/batch] 确认 N 概念页 + M 精读笔记 — P1 验证"
```

- [ ] **Step 5: Validate confirmed pages are now in trusted layer**

```bash
# Count confirmed entity pages
grep -rl "status: confirmed" 概念库/ 模型库/ 任务库/ 数据集/ | wc -l
# Should be > 0

# Count reviewed deep notes
grep -rl "status: reviewed" 论文笔记/ | wc -l
# Should be > 0
```

---

### Task 10: P1 completion validation

- [ ] **Step 1: Objective acceptance criteria**

All must pass:

| Check | Command | Expected |
|---|---|---|
| ≥ 3 deep notes exist | `grep -rl "tier: deep" 论文笔记/ \| wc -l` | ≥ 3 |
| ≥ 1 reviewed note | `grep -rl "status: reviewed" 论文笔记/ \| wc -l` | ≥ 1 |
| ≥ 5 entity pages | `find 概念库/ 模型库/ 任务库/ 数据集/ -name "*.md" \| wc -l` | ≥ 5 |
| ≥ 1 confirmed entity | `grep -rl "status: confirmed" 概念库/ 模型库/ 任务库/ 数据集/ \| wc -l` | ≥ 1 |
| 0 unaccounted dead links | wikilink check: 0 dead links OR all logged as [待决] | 0 unaccounted |
| 1 failure handled | `grep "\[skip/ingest\]" log.md` | ≥ 1 entry |
| git history clean | `git status` | nothing to commit |

- [ ] **Step 2: Tag**

```bash
git tag p1-core-engine
```

---

## Phase P2: Filter Entry (Daily Papers + Inbox)

**Delivers:** Working daily recommendation → card → enhanced-card pipeline. Inbox flow. DailyPapers as index-only view.

**Complete when:** 3+ daily runs produced, cards upgraded to enhanced-cards, at least 1 paper promoted from enhanced-card to deep via explicit command.

---

### Task 11: Daily papers → cards in 论文笔记/

- [ ] **Step 1: Run daily papers**

```
今日论文推荐
```

- [ ] **Step 2: Validate output structure**

DailyPapers file check:
```bash
# DailyPapers/YYYY-MM-DD.md exists
ls DailyPapers/$(date +%Y-%m-%d).md

# It's an INDEX: contains [[论文笔记/...]] links, NOT full note content
grep "\[\[论文笔记/" DailyPapers/$(date +%Y-%m-%d).md | wc -l
# Should be > 0

# It does NOT contain full frontmatter blocks
grep -c "^---" DailyPapers/$(date +%Y-%m-%d).md
# Should be exactly 2 (its own frontmatter), not 2*N
```

Card files check:
```bash
# Card notes created in 论文笔记/
grep -rl "tier: card" 论文笔记/ | wc -l
# Should match number of recommendations

# Each card has required sections
for f in $(grep -rl "tier: card" 论文笔记/); do
  grep -q "## 一句话定位" "$f" && grep -q "## 方法关键词" "$f" && echo "OK: $f" || echo "FAIL: $f"
done
```

- [ ] **Step 3: Validate git commit**

```bash
git log --oneline -1 | grep "\[ingest/card\]"
```

---

### Task 12: Enhanced-card upgrade

- [ ] **Step 1: Select 2-3 cards for upgrade**

From the daily papers, explicitly request enhanced cards for specific papers.

- [ ] **Step 2: Validate upgrade (same file, changed tier)**

```bash
# The file should now say enhanced-card, NOT card
grep "tier: enhanced-card" 论文笔记/[selected-paper].md

# New sections present
grep "## 方法概述" 论文笔记/[selected-paper].md
grep "## 关键结果" 论文笔记/[selected-paper].md

# No KB retrieval triggered (no kb_context_sources field)
grep -c "kb_context_sources" 论文笔记/[selected-paper].md
# Should be 0

# No entity pages created/modified (check git diff)
git diff --name-only HEAD~1 | grep -c "概念库\|模型库\|任务库"
# Should be 0
```

---

### Task 13: Promote enhanced-card to deep-read

- [ ] **Step 1: Explicitly request deep-read**

```
这篇精读 [[论文笔记/selected-paper]]
```

- [ ] **Step 2: Validate full deep-read pipeline triggered**

```bash
# Tier changed
grep "tier: deep" 论文笔记/[selected-paper].md

# All deep sections now present
grep "## 核心问题" 论文笔记/[selected-paper].md
grep "## 方法" 论文笔记/[selected-paper].md
grep "## 实验" 论文笔记/[selected-paper].md

# Entity pages updated (reverse update happened)
git diff --name-only HEAD~1 | grep "概念库\|模型库\|任务库"
# Should be > 0
```

---

### Task 14: Inbox flow

- [ ] **Step 1: Create an inbox item**

Manually create `_inbox/test-blog.md`:
```yaml
---
type: inbox
title: "Some TTS Blog Post"
source: "https://example.com/blog"
why: "有人推荐,跟 voice cloning 相关"
created: 2026-06-01
---
```

- [ ] **Step 2: Process inbox item → deep read**

```
读一下这篇 https://example.com/blog
```

- [ ] **Step 3: Validate inbox item deleted**

```bash
ls _inbox/test-blog.md
# Should: No such file or directory

# But a note exists in 论文笔记/
ls 论文笔记/*blog* || ls 论文笔记/*Blog*
```

- [ ] **Step 4: P2 acceptance + tag**

| Check | Expected |
|---|---|
| ≥ 1 DailyPapers index page | exists, is index-only |
| ≥ 5 card-tier notes in 论文笔记/ | exist |
| ≥ 2 enhanced-card notes | exist, have method overview |
| ≥ 1 card→enhanced→deep promotion | single file, tier changed twice |
| inbox item processed + deleted | _inbox/ empty |
| no KB triggered for card/enhanced | no kb_context_sources in those notes |

```bash
git tag p2-filter-entry
```

---

## Phase P3: Compound Interest (KB Context)

**Delivers:** Deep-read notes automatically contain KB background based on existing confirmed entity pages. Proven richer output when vault has relevant knowledge vs when it doesn't.

**Complete when:** Side-by-side comparison shows KB-enriched note has substantively better positioning than a note without KB. Degradation scenarios handled.

**Prerequisite:** ≥ 5 confirmed entity pages from P1.

---

### Task 15: KB retrieval — happy path

- [ ] **Step 1: Pick a paper that overlaps heavily with confirmed concepts**

Choose a paper whose topic (methods, models, tasks) maps to ≥ 3 already-confirmed entity pages.

- [ ] **Step 2: Ingest with KB retrieval active**

```
精读这篇论文 [URL/PDF]
```

- [ ] **Step 3: Validate KB 背景节 content**

```bash
# KB 背景节 exists and references confirmed pages
grep "KB 背景" 论文笔记/[new-paper].md
grep "检索命中:" 论文笔记/[new-paper].md

# kb_context_sources > 0
grep "kb_context_sources:" 论文笔记/[new-paper].md
# Value should be 3-6

# Log entry shows hits
grep "\[kb/search\]" log.md | tail -1
# Should show hits with ✓ marks
```

- [ ] **Step 4: Qualitative check — KB background adds value**

Read the KB 背景节. Does it:
- [ ] Position the paper in a lineage? (not just "related to X")
- [ ] Say what's already known about the method from confirmed pages?
- [ ] Make a judgment about what's new vs known?

If any "no": fix the prompt/rules and re-run.

---

### Task 16: KB retrieval — degradation scenarios

- [ ] **Step 1: Test with zero matching concepts**

Pick a paper in an area with NO entity pages yet (e.g., audio watermarking if you have no pages on that).

```
精读这篇论文 [unrelated-area paper]
```

Validate:
```bash
# KB 背景 says no match found
grep "未找到相关知识库背景" 论文笔记/[new-paper].md
# Note is still complete (all sections present)
grep "## 核心问题" 论文笔记/[new-paper].md
grep "## 方法" 论文笔记/[new-paper].md
```

- [ ] **Step 2: Test with only pending-review matches**

If all relevant concept pages are pending-review (not confirmed):

Validate:
```bash
# KB 背景 marks as "未确认"
grep "未确认\|尚未确认" 论文笔记/[paper].md
```

- [ ] **Step 3: Test with stale concept page (>3 months old)**

Manually backdate a concept page's `updated` field to 3+ months ago, then ingest a paper that matches it.

Validate:
```bash
grep "可能过时" 论文笔记/[paper].md
```

---

### Task 17: KB retrieval — alias feedback loop

- [ ] **Step 1: Ingest a paper that uses a different term for an existing concept**

E.g., paper says "rectified flow" but your concept page is titled "Flow Matching" with aliases that don't include "rectified flow".

- [ ] **Step 2: Verify it was NOT matched (expected miss)**

```bash
# KB 背景 should NOT reference Flow Matching (because alias wasn't there)
grep "Flow Matching" 论文笔记/[new-paper].md | grep -v "KB 背景"
# The paper may mention it in its own content but KB didn't provide it
```

- [ ] **Step 3: Add alias and verify future match**

Add "rectified flow" to Flow Matching's aliases. Then ingest another paper that uses "rectified flow".

Validate it now matches:
```bash
grep "Flow Matching" 论文笔记/[second-paper].md
grep "\[kb/search\].*Flow Matching" log.md | tail -1
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "[init] KB retrieval alias feedback verified"
```

---

### Task 18: Reproduction-level ingest

**Input:** A paper you intend to reproduce or deeply study (architecture-level detail needed).

- [ ] **Step 1: Designate a paper for repro-level reading**

```
这篇复现级阅读 [[论文笔记/existing-paper]] 
```
(or provide a new paper)

- [ ] **Step 2: Validate repro template output**

```bash
# Tier is repro
grep "tier: repro" 论文笔记/[paper].md

# Repro-specific sections exist
grep "## 模块细节" 论文笔记/[paper].md
grep "## 复现要点" 论文笔记/[paper].md
grep "### Loss 设计" 论文笔记/[paper].md
grep "### 训练配置" 论文笔记/[paper].md
grep "### 推理流程" 论文笔记/[paper].md

# KB 背景 present (since P3 is active)
grep "KB 背景" 论文笔记/[paper].md
grep "kb_context_sources:" 论文笔记/[paper].md
```

- [ ] **Step 3: Validate repro note is substantively deeper than deep note**

A repro note for the same paper should contain:
- [ ] Specific module descriptions with input/output shapes
- [ ] Loss formula or detailed loss description
- [ ] Training hyperparameters (lr, batch, optimizer)
- [ ] Inference pipeline step-by-step
- [ ] ≥ 3 "复现要点" with specific section references

- [ ] **Step 4: P3 acceptance + tag**

| Check | Expected |
|---|---|
| KB 背景 populated with real references | ≥ 2 notes with substantive KB background |
| Degradation: no match → labeled | ≥ 1 note with "未找到" |
| Degradation: pending-only → labeled | ≥ 1 note with "未确认" |
| Degradation: stale → labeled | ≥ 1 note with "可能过时" |
| Log records hits/misses | grep [kb/search] shows details |
| Alias feedback works | adding alias improves future retrieval |
| Repro note produced with full depth | template sections + KB background present |

```bash
git tag p3-compound-interest
```

---

## Phase P4: Basic Health (Lint + MOC)

**Delivers:** Automated structural health checks + MOC auto-generation. Runs reliably, catches real issues, doesn't false-positive excessively.

**Complete when:** Full lint runs, finds ≥ 2 real issues, those issues get fixed. MOC auto-generates correctly for topic with ≥ 3 entities.

**Prerequisite:** Vault has 10+ notes, 10+ entity pages.

---

### Task 19: Full vault lint

- [ ] **Step 1: Intentionally introduce 3 defects for testing**

```bash
# 1. Create a dead link in a note
echo "See also [[不存在的概念]]" >> 论文笔记/CosyVoice*.md

# 2. Remove a required frontmatter field from one entity
sed -i '' '/^tags:/d' 概念库/[some-concept].md

# 3. Create an orphan page (no links pointing to it)
cat > 概念库/Orphan-Test.md << 'EOF'
---
type: concept
title: "Orphan Test"
aliases: []
category: test
tags: [test]
key_papers: []
related_concepts: []
status: confirmed
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-06-01
updated: 2026-06-01
---
## 定义
Test page that nothing links to.
EOF
```

- [ ] **Step 2: Run full lint**

```
跑一下全量 lint
```

- [ ] **Step 3: Validate lint report catches all 3 defects**

```bash
# Report exists
ls _lint/$(date +%Y-%m-%d)*.md

# Contains all 3 issues
grep "不存在的概念" _lint/*.md        # dead link found
grep "tags" _lint/*.md                 # missing field found  
grep "Orphan" _lint/*.md               # orphan page found
```

- [ ] **Step 4: Validate review backlog reporting**

```bash
# Report includes pending-review count
grep -i "pending.review\|backlog\|积压" _lint/*.md
```

- [ ] **Step 5: Fix defects + re-run lint**

Fix the 3 intentional defects. Run lint again. Verify 0 issues (or only pre-existing issues).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "[lint/full] first validated lint run — 3 defects caught + fixed"
```

---

### Task 20: MOC auto-generation

- [ ] **Step 1: Identify a topic with ≥ 3 active entity pages**

```bash
# Find tags that appear in ≥ 3 entity pages
grep -h "^tags:" 概念库/*.md 模型库/*.md 任务库/*.md | tr ',' '\n' | sed 's/.*\[//;s/\].*//;s/^ //' | sort | uniq -c | sort -rn | head
```

- [ ] **Step 2: Trigger MOC generation for that topic**

```
更新 MOC
```

- [ ] **Step 3: Validate MOC content**

```bash
# MOC file exists for the topic
ls _MOC/[topic].md

# Contains only deep/repro notes (not cards)
# Check: any link to a card-tier note is a violation
for link in $(grep -oP '\[\[论文笔记/[^\]]+' _MOC/[topic].md | sed 's/\[\[//'); do
  tier=$(grep "^tier:" "${link}.md" 2>/dev/null | awk '{print $2}')
  if [ "$tier" = "card" ] || [ "$tier" = "enhanced-card" ]; then
    echo "VIOLATION: $link is tier=$tier but appears in MOC"
  fi
done

# Contains entity pages with lifecycle=active
grep "\[\[概念库/" _MOC/[topic].md

# Does NOT contain deprecated or merged entities in main section
# (they should only be in ## 历史参考 if present at all)
```

- [ ] **Step 4: Validate 人工备注 preservation**

Add a note to the MOC:
```markdown
> [!note] 人工备注
> Test: this should survive refresh.
```

Re-run MOC generation. Verify the note survives:
```bash
grep "Test: this should survive" _MOC/[topic].md
```

- [ ] **Step 5: P4 acceptance + tag**

| Check | Expected |
|---|---|
| Lint catches dead links | proven |
| Lint catches missing frontmatter | proven |
| Lint catches orphan pages | proven |
| Lint reports review backlog | proven |
| MOC auto-generates for ≥3 entities | proven |
| MOC only includes deep/repro notes | verified |
| MOC preserves 人工备注 | verified |
| Lint + MOC run without errors | clean run |

```bash
git tag p4-basic-health
```

---

## Phase P5: Advanced Enhancement (Specification + Scaffold)

**Delivers:** Specification for advanced cognitive lint. Directory scaffold for innovation highlights and comparison reports. **Does NOT deliver fully operational advanced lint** — that requires significantly more content density and is better specified than prematurely implemented.

**Complete when:** Specification is written, scaffold exists, and the system is aware of P5 capabilities as future options.

**Honesty:** P5 tag means "ready to implement when vault is mature enough (30+ deep notes, 50+ entities)", not "capability operational."

---

### Task 21: Advanced lint specification + scaffold

- [ ] **Step 1: Add advanced lint rules to AGENTS.md**

Append to the lint section of AGENTS.md:

```markdown
## 高级认知 lint (P5, 手动触发, 需要 LLM 推理)

以下检查成本高(需要读取和比较多个页面内容),仅在手动触发时运行。
所有结论标记为 [疑似],不自动修改任何页面。

### 矛盾检测
- 比较共享 tags 的实体页/论文笔记,检查描述是否互相矛盾
- 输出: "[疑似矛盾] [[A]] 说X, [[B]] 说Y"

### 过期 SOTA
- 扫描包含"SOTA/最优/state-of-the-art"的笔记
- 检查是否有更新的论文在同一数据集+指标上报告了更好数字
- 输出: "[疑似过期] [[PaperX]] 声称 MOS=4.2 为 SOTA,但 [[PaperY]] 报告 4.3"

### 缺失关联
- 对每对都引用同一概念但未互链的论文笔记,报告
- 输出: "[疑似缺失] [[A]] 和 [[B]] 都涉及 [[C]] 但未互链"

这些能力是启发式的,会有 false positive。设计目标是"提醒你可能有问题",不是"断定一定有问题"。
```

- [ ] **Step 2: Create scaffold directories**

```bash
mkdir -p _创新亮点 _对比报告
```

- [ ] **Step 3: Commit + tag**

```bash
git add -A
git commit -m "[init] P5 scaffold — advanced lint spec + highlight/comparison directories

P5 标记为 specification-ready,不是 capability-operational。
高级 lint 实现需要 vault 达到 30+ deep notes / 50+ entities 后再启动。"
git tag p5-specification-ready
```

---

## Summary: What each tag means

| Tag | Meaning | Proven by |
|---|---|---|
| `p0-foundation` | Vault skeleton correct, rules complete | Schema check passes |
| `p1-core-engine` | Ingest pipeline works reliably | 3+ papers ingested, 1 failure handled, lint passes |
| `p2-filter-entry` | Daily recommendation → card → upgrade works | Full flow exercised, single-file model validated |
| `p3-compound-interest` | KB retrieval enriches new notes | Side-by-side comparison, degradation proven |
| `p4-basic-health` | Lint + MOC self-maintain structural integrity | Real defects caught + fixed, MOC auto-generates |
| `p5-specification-ready` | Advanced features specified, ready for future implementation | Spec in AGENTS.md, scaffold exists |
