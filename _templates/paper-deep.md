---
type: paper
tier: deep
title: "{{title}}"
arxiv_id: "{{arxiv_id}}"
source: "{{source}}"
authors: [{{authors}}]
year: {{year}}
venue: "{{venue}}"
tags: [{{tags}}]
concepts: []
models: []
tasks: []
datasets: []
kb_context_sources: {{N}}
status: draft
created: {{date}}
updated: {{date}}
---

## KB 背景

> [!info] KB 背景 (基于 {{N}} 个已确认实体页: {{entity_list}})
> 自动生成,不保证完整覆盖所有相关知识。
> 检索命中: ... | 过滤: ... | 未命中但可能相关: 无

## 速查

> [!summary] 速查
> - **一句话**: {{核心贡献,一句话}}
> - **路线**: {{输入→模块→输出}}
> - **指标**: {{关键数字 + 对比基准 + 数据集}}
> - **可借鉴**: {{可迁移到你工作中的 idea}}
> - **局限**: {{不 work 的地方 / 复现难点 / 未开源}}

## 核心问题

## 方法: 它怎么 work

> [!important] 区分来源
> 本节中的因果解释须标注来源:[论文原文] 表示论文作者的解释,[agent 解读] 表示基于论文内容的推断性分析。

### 整体架构

### 关键设计选择

### 训练策略

## 实验

| 指标 | 本文 | Baseline | 数据集 | 出处 |
| --- | --- | --- | --- | --- |

## 局限性

## 点评

## 可复用的 idea

## 审阅

> [!review] 审阅 ({{date}}, auto)
> **结论**: {{pass / pass-with-fixes / revise / reject-as-deep}}
> 
> | 原则 | 状态 | 备注 |
> |------|------|------|
> | 可复述 | {{pass/fail}} | {{简述}} |
> | 可信赖 | {{pass/fail}} | {{简述}} |
> | 可区分 | {{pass/fail}} | {{简述}} |
> | 可定位 | {{pass/fail}} | {{简述}} |
> | 不污染 | {{pass/fail}} | {{简述}} |
> 
> Issues: {{N}} ({{high: X, medium: Y, low: Z}})
> 详见 `_review/{{论文名}}-review.yml`
