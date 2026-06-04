# PaperWiki Harness 升级计划 v3

> v1: 诊断报告(15 个差距)
> v2: Batch 1+2 完成后的进度记录
> v3: 概念页升级 + MOC 升级完成后的最终状态,系统从 Level 1.5 → Level 3
>
> 日期: 2026-06-04
> 历程: v1 诊断 → Batch 1 (G2+V2+E2) → Batch 2 (G1+V1+V3) → 信任机制转型 → 概念页升级 (G3-D+G4+P1) → MOC 升级 (P4+P5)

---

## 1. 15 个差距最终状态

### 完成 (13/15)

| ID | 差距 | 产出物 |
|---|------|--------|
| G1 | AGENTS.md 瘦身 | 206 行 Tier 1 + `docs/rules/` 6 个 Tier 2 |
| G2 | MOC 规则 | §11 宪法 5 原则 + R1-R8 + 运维规则 |
| G3 | 三层治理全覆盖 | 4 个 checklist: 笔记 / KB / MOC / 实体页 |
| G4 | 约束升级为脚本 | L1-L17,0 项人工必须 |
| E2 | 审阅独立化 | 4 模式均为独立 subagent dispatch |
| V1 | KB 更新审阅 | kb-review-checklist + pipeline 第 6 步 |
| V2 | MOC 审阅 | moc-review-checklist (17 检查项) + L15 阈值检测 |
| V3 | Lint 扩展 | 17 项自动化检查 (L1-L17) |
| N1 | 熵管理 | lint (去重/死链/孤儿/过时/积压) + MOC 审阅 (过载/腐烂) |
| M1 | 度量 (部分) | L9 vault stats 自动更新 + L13 可信层进度 |
| — | 信任机制转型 | 原则 1: "自动门通过即 confirmed"; entity-review pass = 唯一晋升路径 |
| — | MOC 修复 | 零样本语音合成拆 3 sub-MOC; 全部 MOC 补 scope; 去重完成 |
| — | learning_signals 消化 | L17 扫描 + digested 标记机制 |

### 部分完成 (1/15)

| ID | 差距 | 已做 | 未做 |
|---|------|------|------|
| L1 | 闭环机制 | L17 消化通道已建; 审阅报告持续积累 | pattern analysis 未跑(新规则下报告仅 1 篇,需 ≥ 30) |

### 未做 (4/15) — 全部等信号触发

| ID | 差距 | 触发条件 | 当前距离 |
|---|------|----------|----------|
| V4 | 审阅拆分计算/推理 | L1 pattern analysis 产出后 | 依赖 L1 |
| L2 | 闭环质量信号 | L1 跑过 2 轮 | 依赖 L1 |
| E3 | MOC Agent 独立 | sub-MOC ≥ 50 或 MOC 刷新出错 ≥ 2 次 | 最大 46,接近 |
| E1 | Pipeline 多 agent | vault ≥ 400 或上下文质量下降 | 当前 257 |

---

## 2. 当前系统架构

### 文件结构

```
AGENTS.md (Tier 1, ~200 行)
docs/rules/ (Tier 2, 6 文件)
    sources.md / notes.md / kb.md / moc.md / review.md / checks.md
_templates/ (4 个审阅标准)
    review-checklist.md        论文笔记 (5 原则)
    kb-review-checklist.md     KB 更新 (3 原则)
    moc-review-checklist.md    MOC (5 原则 + 17 检查项)
    entity-review-checklist.md 实体页 (5 原则)
scripts/lint.py (L1-L17)
~/.claude/skills/
    paperwiki-reader/    精读 pipeline (8 步 + Step 4.5)
    paperwiki-reviewer/  审阅 agent (4 模式)
```

### Pipeline (8 + 1 步)

```
① 读原文 PDF
② KB 检索 (confirmed 实体页 Top 6)
③ 生成笔记草稿
④ 笔记审阅 (独立 subagent)
⑤ Git commit 草稿 + 审阅报告
⑥ KB 更新审阅 (独立 subagent)
⑦ 反向更新 (KB 审阅通过后)
  ④.5 新建页 → entity-review → pass 则 auto-confirmed
⑧ 自动检查 (lint.py) + Git commit + log
```

### 审阅系统 (4 模式)

全部独立 subagent,不共享生成上下文。

| 模式 | checklist | 触发 | 结论 |
|------|-----------|------|------|
| 笔记 | review-checklist.md | 精读自动 | pass / pass-with-fixes / revise / reject-as-deep |
| KB 更新 | kb-review-checklist.md | 反向更新前自动 | pass / pass-with-fixes / block |
| MOC | moc-review-checklist.md | 手动 / L15 阈值触发 | pass / pass-with-fixes / revise / restructure |
| 实体页 | entity-review-checklist.md | 新建页自动 / 手动 | pass / pass-with-fixes / revise / restructure |

### 信任机制

```
AI 生成 → lint (L1-L17) → reviewer (独立 subagent) → pass → auto-confirmed
                                                    → fail → pending-review
人偶尔纠正 → 校准信号 → L17 消化 → 规则改进
```

自动晋升唯一路径: entity-review pass → confirmed。

### Lint (17 项)

| 编号 | 检查项 | 级别 |
|------|--------|------|
| L1 | frontmatter schema | error |
| L2 | 年份一致 (MOC vs frontmatter) | error |
| L3 | MOC 重复 wikilink | error |
| L4 | 概念去重 (title/aliases) | error |
| L5 | MOC 覆盖 (deep/repro) | error |
| L6 | 审阅 callout | error |
| L7 | PDF 存在 | error |
| L8 | 死链 | error |
| L9 | CLAUDE.md vault 统计 | error (--fix 可修复) |
| L10 | 孤儿页 | error |
| L11 | 概念过时 (>90 天) | error |
| L12 | 审阅积压 | error |
| L13 | 可信层进度 | info |
| L14 | log 完整性 | warning |
| L15 | MOC 论文数阈值 | warning >40 / error >50 |
| L16 | Stale pending (>30 天) | warning |
| L17 | learning_signals 未消化建议 | info |

### 关键指标

| 指标 | 数值 |
|------|------|
| 论文笔记 | 257 篇 (242 deep + 2 repro + 2 enhanced-card + 11 card) |
| 实体页 | 93 个 (22 confirmed / 71 pending-review) |
| MOC | 10 个 (6 主题 + 3 sub-MOC + 1 总览) |
| 审阅报告 | 173 个 (新规则下 1 个) |
| 审阅 checklist | 4 个 |
| Lint 检查项 | 17 项 |
| 人工必须项 | 0 |

---

## 3. 升级历程

| 阶段 | 完成项 | 关键产出 |
|------|--------|----------|
| Batch 1 | G2, V2, E2 | MOC 规则 + MOC 审阅 + 审阅独立化 |
| Batch 2 | G1, V1, V3 | AGENTS.md 瘦身 + KB 审阅 + lint L1-L9 |
| 收尾 | — | MOC 审阅实战 + MOC 修复 + docs 清理 + L9 |
| 信任机制转型 | — | 原则 1 改写 + kb.md 自动晋升规则 |
| 概念页升级 | G3, G4, P1-P3 | entity-review-checklist + 第 4 审阅模式 + L10-L16 + Step 4.5 自动晋升 |
| MOC 升级 | P4, P5 | L15 阈值检测 + L17 learning_signals 消化 |

成熟度: Level 1.5 → **Level 3**

---

## 4. 剩余工作

### 唯一的主动项: L1 Pattern Analysis

**当前状态**: 消化通道 (L17) 已建,但 pattern analysis 本身未跑。新规则下审阅报告仅 1 篇。

**触发条件**:
```bash
find _review -name "*-review.yml" -newer _review/moc-review-零样本语音合成-2026-06-04.yml | wc -l
# ≥ 30 时启动
```

**到达后执行**:
1. 统计新规则下审阅报告的高频问题类型
2. 根因归属(模板/规则/skill/检索哪层缺失)
3. 产出 `_review/pattern-analysis-YYYY-MM-DD.md` 含规则修改建议
4. 确认后更新 checklist (法律层演进)
5. V4: 从 checklist 抽离计算性检查项到 lint

### 等信号触发的 3 项

| 项 | 信号 | 检查命令 |
|---|------|----------|
| E3 MOC Agent | sub-MOC ≥ 50 | `python3 scripts/lint.py --check L15` |
| E1 Pipeline 拆分 | vault ≥ 400 | `ls 论文笔记/*.md \| wc -l` |
| L2 + M1 度量 | L1 跑过 2 轮 | `ls _review/pattern-analysis-*.md \| wc -l` |

---

## 5. 与 v1 的变化

| 方面 | v1 (诊断时) | v3 (当前) |
|------|------------|-----------|
| 成熟度 | Level 1.5 | **Level 3** |
| AGENTS.md | 653 行百科全书 | 206 行 Tier 1 + 6 Tier 2 |
| 审阅 | 1 种(笔记,内联) | **4 种**(笔记/KB/MOC/实体页,独立 subagent) |
| Lint | 仅死链 | **17 项**(L1-L17) |
| Pipeline | 7 步 | **8+1 步**(含 KB 审阅 + entity-review + lint) |
| MOC | 7 行规则 | 宪法 5 原则 + R1-R8 + 审阅 + L15 阈值 |
| 信任机制 | 人审批制 | **自动门制**(entity-review pass = auto-confirmed) |
| 闭环 | 无 | **L17 消化通道就绪**,等数据积累 |
| 人工必须项 | 5 项 | **0 项** |
| Checklist | 1 个 | **4 个**(三层结构,覆盖所有知识产出) |
| vault | 105 篇 | 257 篇 |

---

## 6. 设计原则对照

| Harness 原则 | 状态 |
|---|---|
| §0.1 核心定理(约束+反馈) | ✅ 17 项 lint 约束 + 4 模式审阅反馈 + L17 消化通道 |
| §0.2 三层治理 | ✅ 4 个 checklist 覆盖全部知识产出环节 |
| §0.3 双向反压 | ✅ 上游(规则+模板) + 下游(审阅+lint) |
| §0.4 机械化 | ✅ 17 项脚本,0 项人工 |
| §0.5 Agent 分离 | ✅ 生成/审阅/lint 独立 |
| §0.6 闭环 | ⏳ 通道就绪,等数据积累后跑 pattern analysis |
| §0.7 上下文管理 | ✅ Tier 1 (206 行) + Tier 2 按需加载 |
| §0.8 熵管理 | ✅ lint 孤儿/过时/重复 + MOC 审阅过载/腐烂 |
| §0.9 失败降级 | ✅ 全链路降级不阻断 |
| §0.10 度量 | ⏳ L9 stats + L13 进度,完整度量待闭环后 |
| §0.11 演进不重建 | ✅ 全部基于现有系统扩展 |

**结论: 系统升级基本完成。唯一待触发项是闭环(L1 pattern analysis),靠日常精读自然积累数据。**
