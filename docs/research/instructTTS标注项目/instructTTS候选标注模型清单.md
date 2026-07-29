---
title: "instruct TTS 候选标注模型清单 v0.1"
created: 2026-07-29
tags: [research, tts, instruct, annotation, models, tooling]
---

# instruct TTS 候选标注模型清单 v0.1

> 配套 [[instructTTS标注pipeline与算子]] / [[instructTTS工具验证与交叉验证方案]]。按维度列出**可用/可选模型**,供 Part A 验证选型。
> **核实状态**:✅ = 已 WebFetch 核实(2026-07);📚 = 基于知识,**需最终核实**(许可/中文表现以官方为准);🏠 = 内部工具。
> **许可警示**:标注是否属"商用",NC 模型能否用于内部标注训练数据 —— 走法务(见 [[instructTTS标注方案设计]] §9.3)。

---

## 1. 韵律 / 物理量(pitch / energy / rate / pause / tone)

| 模型/工具 | 产出 | 许可 | 状态 | 备注 |
|---|---|---|---|---|
| **WordVoice-5A** | 词级 Duration/Boundary(b0-b4)/Energy/Pitch/Tone(7 类)| 内部 | 🏠 | **首选**,MFA+Qwen3FA 对齐,一站式 |
| CREPE / torchcrepe | F0(pitch)| MIT | 📚 | 神经网 F0,精度高,需 GPU |
| pYIN / librosa.pyin | F0 | ISC | 📚 | 轻量 |
| Praat / parselmouth | F0/energy/共振峰 | GPL-3 | 📚 | GPL 注意商用 |
| pyworld(WORLD)| F0/包络 | MIT 类 | 📚 | TextrolSpeech/Spark 用过 |
| librosa RMS / pyloudnorm | energy/LUFS | ISC / MIT | 📚 | 音量 |
| MFA / WhisperX / Qwen3FA | 词级对齐→rate/pause | MIT / BSD / 内部 | 📚🏠 | rate=字/音素率;pause=停顿检测 |

**结论**:物理层直接**复用 WordVoice-5A**(已覆盖全部 5 项),DSP 备选仅在需补充时用。

---

## 2. 说话人属性(gender / age)—— ⚠️ 商用友好开源缺口

| 模型 | 产出 | 许可 | 状态 | 备注 |
|---|---|---|---|---|
| audeering wav2vec2-age-gender | gender 3类 + age 连续 | **cc-by-nc-sa** 禁商用 | ✅ | 最好用但 NC;英文为主 |
| Vox-Profile wavlm-sex-age | sex/age | **RAIL 禁商用 + 仅英文** | ✅ | 覆盖全但不可用 |
| 3D-Speaker(CAM++/ERes2Net)| 说话人/语种 | Apache-2.0 ✅商用 | ✅ | **无 gender/age 模型** |
| wav2vec2/WavLM 性别分类(社区)| gender 2类 | 各异(多 MIT/Apache)| 📚 | HF 有多个,需核许可/中文表现 |
| **自训轻量 head**(WavLM/CAM++ + 分类头)| gender/age | 自有 | 📚 | Spark 用 WavLM(AISHELL-3 gender 99.4%);数据易得 |

**结论**:**商用友好的现成 age/gender 基本缺**。三选一(见 §9.3):① 轻量自训 head(gender 易、age 难)② audio-OmniLLM 弱标 ③ 内部已有模型。**待拍板**。

---

## 3. 情感基础类 SER(L1,linchpin)

| 模型 | 类别 | 中英 | 许可 | 状态 | 备注 |
|---|---|---|---|---|---|
| **emotion2vec+ large** | 9 类(angry/disgusted/fearful/happy/neutral/other/sad/surprised/unknown)| 语言无关 | 自定义 model-license | ✅ | 纯音频、专用、清晰 taxonomy;FunASR |
| emotion2vec+ base/seed | 同上 | 语言无关 | 同上 | 📚 | 轻量版 |
| **SenseVoice-Small** | ASR+SER+AED+LID 一体 | 中英粤优 | 自定义 model-license | ✅ | 顺带副语言事件+BGM;情感类未逐一枚举;70ms/10s |
| wav2vec2/HuBERT-SER(IEMOCAP/MELD 微调)| 多为 4-7 类 | 多英文 | 各异 | 📚 | 中文表现存疑 |

**结论**:主选 **emotion2vec+ large**(类别清晰=定 L1 taxonomy);**SenseVoice** 作备选/兼副语言+环境。二者中文 accuracy + 一致率进 Part A 对比。

---

## 4. 情感细类/强度(L2)+ 混合分布 —— audio-OmniLLM

| 模型 | 用途 | 许可 | 状态 | 备注 |
|---|---|---|---|---|
| **Qwen3-Omni** | 听音频+读文本+L1 → 细类+强度 | Apache/Qwen 类 | 📚🏠 | 内部在用;多模态输入 |
| Qwen2-Audio | 同上(旧版)| Apache | 📚 | OV-InstructTTS 用其标副语言 |
| Kimi-Audio / SALMONN | audio understanding | 各异 | 📚 | 备选 |
| Gemini-2.5-Pro(闭源 API)| audio caption/情感 | 商用 API | ✅ | 音频 caption 事实标准,成本高 |
| GPT-4o-audio | 同上 | 商用 API | 📚 | 备选 |

**结论**:主力 **Qwen3-Omni**(内部可控、可批量);高质量抽检/难例可用 Gemini。输入配方(audio / +text / +L1)由验证方案 **Part C** 消融定。

---

## 5. register(场景,文本主判 + 语音校验)

| 模型 | 用途 | 许可 | 状态 | 备注 |
|---|---|---|---|---|
| **Qwen3-32B** / Qwen3.7-max | ASR 文本 → 7 类分类 | Apache/内部 | 📚🏠 | MOSS-VG 用 Qwen3-32B;prompt 已定稿 |
| GPT-4o / GPT-5 | 文本分类 | 商用 API | 📚 | 备选/高精度 |
| DeepSeek / Claude | 文本分类 | API | 📚 | 备选 |
| Qwen3-Omni | 语音校验(听) | 同上 | 📚🏠 | 复核文本分不开的类 |

**结论**:text-LLM(Qwen3-32B)主判 + Qwen3-Omni 语音校验。

---

## 6. 质量 / 环境 / 音频事件

| 模型 | 产出 | 许可 | 状态 | 备注 |
|---|---|---|---|---|
| **DNSMOS**(P.808/P.835)| MOS | 开源(MS)| 📚 | MOSS 用 ≥3.0 门 |
| Brouhaha | SNR/C50/VAD | MIT | 📚 | 信噪比 |
| UTMOS / NISQA | MOS | 各异 | 📚 | MOS 备选 |
| Silero VAD | 语音活动 | MIT | 📚 | 已有上游可复用 |
| **SenseVoice AED** | BGM/掌声/笑/哭/咳/嚏 | 自定义 | ✅ | 顺带 environment(BGM)+ 副语言起点 |
| PANNs / BEATs / YAMNet | 音频事件 tagging | Apache/MIT | 📚 | 环境/事件备选 |

**结论**:质量门 DNSMOS+SNR;environment 用 SenseVoice BGM 检测(顺带);副语言二期用 SenseVoice AED / PANNs。

---

## 7. NL 扩写(L3 + global_desc + 指令)—— text-LLM

| 模型 | 用途 | 状态 | 备注 |
|---|---|---|---|
| Qwen3-32B / Qwen3.7-max | L3 描述、global_desc、四阶段指令扩写 | 📚🏠 | 内部主力 |
| GPT-4o/5、DeepSeek、Claude | 同上 | 📚 | 多样性/高质量备选 |

**结论**:内部 Qwen 系列为主;输入 = 文本 + L1/L2 标签(以标签为锚)。

---

## 8. 上游(已有,列出常见实现供参考)

| 环节 | 常见模型 | 备注 |
|---|---|---|
| VAD | Silero / FunASR fsmn-vad / pyannote | 已有 |
| ASR(词级)| Qwen3-Omni-ASR / Whisper-large-v3 / FunASR | 已有,出词级时间戳 |
| 说话人分离 | pyannote / 3D-Speaker CAM++ / DiariZen | 已有 |
| 源分离/去噪 | MossFormer2_SE / Demucs | 可选补充 |

---

## 9. 第二期维度(intent / persona / accent / 副语言)候选

| 维度 | 候选 | 状态 | 备注 |
|---|---|---|---|
| intent | text-LLM(Qwen3/GPT)+ 片段上下文 | 📚 | 无专用模型,重文本语义 |
| persona | 低阶标签聚合 + LLM 合成 | 📚 | bottom-up 派生,无专用模型 |
| accent/方言 | 元数据 / Qwen3-Omni 弱标 | 📚 | **无好开源中文方言分类器**;3D-Speaker 仅语种 zh/en |
| 副语言(span)| SenseVoice AED / PANNs / NVSpeech 类 | 📚 | 带时间戳;NVSpeech 18 类为标注起点 |

---

## 选型收敛路径

1. **确定即固化**:SER(定 emotion L1 类别集)、age 模型(定 age 类别集)——第一期第一步;
2. **走 Part A 验证**:gender/age/SER/register 在 Gold Set 上比 accuracy + 许可,选定;
3. **走 Part C 消融**:Qwen3-Omni 的输入配方(audio/+text/+L1);
4. **许可法务**:emotion2vec+/SenseVoice 自定义 license、以及 NC 模型用于内部标注是否可行。

> 本清单为候选池,**非最终选型**;选定结果回填 [[instructTTS标注方案设计]] §9。
