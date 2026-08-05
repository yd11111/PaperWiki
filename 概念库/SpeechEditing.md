---
type: concept
title: "Speech Editing"
aliases: [语音编辑, Text-based Speech Editing, Speech Attribute Editing, 语音内容编辑, Precise Speech Editing]
category: "problem"
tags: [speech-editing, TTS, controllability, content-editing, attribute-editing, preservation]
key_papers: ["[[论文笔记/dots.tts.edit|dots.tts.edit]]", "[[论文笔记/AST-Edit|AST-Edit]]", "[[论文笔记/CosyEdit|CosyEdit]]", "[[论文笔记/CosyEdit2|CosyEdit2]]", "[[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]", "[[论文笔记/Ming-UniAudio|Ming-UniAudio]]", "[[论文笔记/SonoEdit|SonoEdit]]", "[[论文笔记/EditContentPreserveAcoustics|EditContentPreserveAcoustics]]"]
origin_paper: "[[论文笔记/dots.tts.edit|dots.tts.edit]]"
related_concepts: ["[[Instruction-GuidedSpeechSynthesis]]", "[[ConditionalFlowMatching]]", "[[EmotionControlinTTS]]", "[[ProsodyModeling]]", "[[Zero-shotSpeechSynthesis]]"]
status: pending-review
lifecycle: active
merged_into: ""
deprecated_reason: ""
created: 2026-08-05
updated: 2026-08-05
---

## 定义

Speech Editing 指在保留一段已有录音的说话人身份、内容与声学环境的前提下,对其**局部**属性做受控修改的任务。修改可作用于:
- **词汇内容 (lexical/content)**: 插入、删除、替换词或短语 (text-based speech editing);
- **副语言属性 (paralinguistic/attribute)**: 情感、韵律 (pitch / 语速)、停顿、说话风格等。

核心难点是**双重约束**: (i) 精确执行请求的修改 (target execution); (ii) 保住未被修改的内容与声学属性 (local preservation),同时整句自然连贯——尤其当编辑改变时长时,同一语言学 span 在源/目标占据不同时间线 [[论文笔记/dots.tts.edit|dots.tts.edit]] §3.1]。

## 为什么重要

语音编辑是内容创作 (配音、有声书、播客后期) 的核心需求:比"重新合成整句"更省、更可控,且能保留原始录音的真实感。它也是 agent-mediated 音频创作工作流里的可调用工具单元。评估必须区分**执行 / 保留 / 音质**三维,单看 ASR WER 无法衡量韵律或音色是否被破坏。

## 技术路线分类

| 路线 | 代表 | 机制 |
|------|------|------|
| **mask-predict / infill** | A3T、CampNet、Voicebox、VoiceCraft | 显式 aligner 把编辑 span 映射到被 mask 的声学区,从文本 + 周边语音补全缺失区 |
| **flow-inversion (training-free)** | [[论文笔记/AST-Edit|AST-Edit]] | 用 flow matching 的 ODE 可逆性把 source 反演到 latent,在保留区拼接 inverted latent + 编辑区噪声,正向 ODE 合成 |
| **CFM 端到端内容编辑** | [[论文笔记/CosyEdit|CosyEdit]] / [[论文笔记/CosyEdit2|CosyEdit2]] | 从零样本 TTS 模型解锁编辑能力 |
| **codec-LM 属性编辑** | [[论文笔记/Step-Audio-EditX|Step-Audio-EditX]]、[[论文笔记/Ming-UniAudio|Ming-UniAudio]] | codec language model 迭代式改情感/风格/副语言,多为整句级 |
| **连续 AR 骨干适配** | [[论文笔记/dots.tts.edit|dots.tts.edit]] | 保持连续自回归 TTS 基座不变,仅改条件序列 + 配对数据学会编辑,支持局部 text+emotion+prosody+pause 联合编辑 |

## 与相邻概念的边界

- vs [[Instruction-GuidedSpeechSynthesis]]: 指令引导语音**合成**从零生成语音; speech editing 以一段已有录音为输入并保留其大部分。两者可共享指令接口 (自由文本 or 结构化 tag)。
- vs [[Zero-shotSpeechSynthesis]]: 零样本 TTS 用参考音克隆音色重新生成整句; editing 保留原录音、只改局部。[[论文笔记/dots.tts.edit|dots.tts.edit]] Table 8 显示专用编辑相比全句零样本重合成在局部保留上更好,但后者 SpkSim/UTMOS 更高。

## 评估

- 内容: 编辑区 / 非编辑区 WER/CER (区分执行与保留)。
- 声学保留: 词对齐 duration DTW (WDTW-Dur,源自 [[论文笔记/AST-Edit|AST]])、词对齐半音 F0 漂移 (WDTW-F0,由 [[数据集/doteBench|doteBench]] 提出)、SpkSim。
- 音质: UTMOS 等。
- benchmark: [[数据集/doteBench|doteBench]] (双语,scope-aware),此前另有 RealEdit、LibriSpeech-Edit、SpeechEditBench、MMAE 等。

## 指令接口取向

编辑请求的表示存在**自由文本 vs 结构化**之争: 自由文本 (如 MMAE) 灵活但有歧义 (操作/参数/定位可能欠规定); 结构化指令 (如 [[论文笔记/dots.tts.edit|dots.tts.edit]] 的 transcript-grounded XML typed tag) 显式、可外部检查、可组合,适合专业创作与 agent 调用。两种取向的正面对照实验目前仍缺失。
