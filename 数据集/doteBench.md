---
type: dataset
title: "doteBench"
aliases: [dote Bench, dots.tts.edit Benchmark]
domain: "Precisely controlled speech editing evaluation"
scale: "1,781 cases across 5 categories (bilingual EN/ZH)"
tags: [benchmark, speech-editing, evaluation, bilingual, instruction-following, preservation]
used_by: ["[[论文笔记/dots.tts.edit|dots.tts.edit]]"]
metrics_reported_on: [Instruction Following, Local Preservation, Audio Quality, WER/CER, WDTW-Dur, WDTW-F0, SpkSim, UTMOS]
url: "huggingface.co/spaces/dots-studio/dots.tts.edit"
status: pending-review
lifecycle: active
merged_into: ""
created: 2026-08-05
updated: 2026-08-05
---

## 概述

doteBench 是随 [[论文笔记/dots.tts.edit|dots.tts.edit]] (Wang et al., 2026) 发布的双语 (英/中) **精确语音编辑**评测套件,为 typed、可定位的语音控制定义了 category-specific、scope-aware 的评测协议 [dots.tts.edit §3.2]。

**五个类别** [§3.2, Fig 2]:
- **Text** (569 例): 含 Easy / Hard split,主对比用 Hard;
- **Emotion** (312 例): utterance / local-span;
- **Prosody** (360 例): pitch / rate;
- **Pause** (300 例): insert / reduce;
- **Compositional** (240 例): 2/3/4 操作组合,报 component-wise 与 all-component 成功率。

单任务子套件 (text+emotion+prosody+pause) 共 1,541 例,加组合 240 例 = **1,781 例** [Fig 2]。

## 三个评估维度

评测契约对应"控制契约" [Table 1]:
1. **Instruction Following** (执行请求的操作): text 用编辑区 WER/CER; emotion 用 Gemini 评的情感准确率; prosody 用请求的 duration/pitch 变化误差; pause 用插入/缩短方向准确率; compositional 用 component / all-component 成功率。
2. **Local Preservation** (定位之外的语音): text 用指令派生编辑邻域**补集**上的 Qwen3-ASR WER/CER (排除区向两侧各扩 3 token);其余任务保留转写故用全句 WER/CER。声学保留另比对 WDTW-Dur、WDTW-F0、SpkSim (WavLM-large / ECAPA-TDNN 余弦相似度)。
3. **Audio Quality**: UTMOS;并用 SEED-TTS-Eval 测基座合成能力是否保持。

## 新提出的保留度量: WDTW-F0

WDTW-Dur (词级 duration DTW) 对 pitch 漂移不敏感,故 doteBench 提出 **WDTW-F0** [§A.3]: 对每个合格保留词,用 Praat/Parselmouth 抽取对齐源/输出区间的 min/max/mean voiced F0,计半音误差 `e = 12·log2(F0_out/F0_src)`,只在双侧都有有效 voiced 估计的词上平均,报 eligible/valid/skipped 词数 (不同 valid 数的比较不等价)。其名虽含 DTW 但**不是对完整 F0 轨迹做 DTW**,而是词对齐摘要,补 WDTW-Dur 的缺口。相比此前 benchmark (如 SpeechEditBench) 对非内容任务只用 ASR WER/CER 把关保留度,doteBench 的声学保留度量更细。

## 注意事项

- 所有系统对同一 frozen manifest 评测;生成失败按固定惩罚留在分母;WDTW 仅当 manifest 无合格保留区时才跳过 (skip set 与候选系统无关) [§6.1.4]。
- Instruction Following 的情感准确率依赖 Gemini 判定,受判官感知能力约束。
- 目前 (v1) 尚**不覆盖多说话人编辑**任务 [Limitations]。
