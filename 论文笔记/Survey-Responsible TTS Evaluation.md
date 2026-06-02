---
type: paper-note
title: "Position: Towards Responsible Evaluation for Text-to-Speech"
authors: [Yifan Yang, Hui Wang, Bing Han, Shujie Liu, Jinyu Li, Yong Qin, Xie Chen]
year: 2025
venue: "ICML 2026 (arXiv:2510.06927)"
arxiv_id: "2510.06927"
tier: card
tags: [survey, TTS-evaluation, MOS, responsible-AI, standardization, fairness, metrics, cold-start]
concepts: ["[[TTS Evaluation]]", "[[Speaker Verification]]", "[[LLM-based TTS]]"]
created: 2026-06-02
updated: 2026-06-02
---

## 概要

首篇系统性审视 TTS 评估实践的 position paper,提出 **Responsible Evaluation** 三层框架。论文从 ICML 2026 发表,涵盖 TTS 评估方法的历史演进 (统计参数时代→深度学习时代→扩散/基础模型时代),深入分析当前主流指标 (MOS, WER, SIM, Predicted MOS, F0) 的系统性缺陷,并提出可操作建议。

## 核心贡献

1. **Responsible Evaluation 三层框架**:
   - Level 1: Fidelity & Accuracy (指标忠实反映能力)
   - Level 2: Comparability, Standardization, Transferability (跨系统可比)
   - Level 3: Governance, Fairness, Security (伦理/安全纳入评估)

2. **系统性缺陷诊断**:
   - MOS ceiling effect + 不可迁移性
   - WER 与感知可懂度非线性关联
   - SIM 协议不统一 (SIM-o ± prompt, 差异达 0.151)
   - LibriSpeech test-clean 版本混乱 (同一模型 WER 2.63 vs 4.22)

3. **评估不足维度识别**: 数学/公式朗读、长篇章合成、情感表现力、标点敏感性

4. **新兴方法**: LLM-as-a-Judge (SpeechLLM-as-Judges), Audio Turing Test

## 知识提取成果

本文用于概念库冷启动 T2 补充。

### 新建概念页 (1 个)

| 概念 | 核心内容 |
|------|----------|
| [[TTS Evaluation]] | TTS 评估方法论: 指标体系、局限诊断、新兴方法、Responsible Evaluation 框架 |

### 追加更新 (1 个)

| 页面 | 更新内容 |
|------|----------|
| [[SVS Evaluation Metrics]] | 补充与 TTS Evaluation 的交叉引用 |

## 关键观点

- "We argue that Responsible Evaluation is essential and urgent for the next phase of TTS development" [§1]
- MOS 在高质量系统间区分力饱和,需要 Audio Turing Test 等替代 [§3.2]
- 不同论文使用不同版本 LibriSpeech test-clean (40/1127/1234 utterances),同一模型 WER 变化 60% [Appendix A]
- SIM-o 包含/排除 prompt 段时差异 0.151 [Appendix B]
- LLM-as-a-Judge 可提供 transferable scores + interpretable reasoning [§4.4]
