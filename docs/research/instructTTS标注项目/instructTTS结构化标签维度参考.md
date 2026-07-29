---
title: "instruct TTS 结构化标签维度 · 类别设置参考"
created: 2026-07-28
tags: [research, tts, instruct, annotation, taxonomy, reference]
---

# 结构化标签维度 · 类别设置参考

> 配套 [[instructTTS标注方案设计]] 第一阶段(结构化标注)。为每个维度**定类别数 + 取值集**提供选型依据:逐维汇总现有方案的设置,末附本方案建议起点。
> 引用格式:`文件:行号`(vault 笔记)/ HF·GitHub 模型卡 / 原文章节。标注"原文未枚举"处表示论文只给数量未给清单。

## 0. 速览:各维度类别数对照

| 维度 | 现有方案的类别数(区间)| 本方案建议起点 | 标签形态 |
|---|---|---|---|
| gender | 2(TextrolSpeech)~ 3(audeering,含 child)| **随分类器类别集** | 封闭 |
| age | 连续(audeering)/ 5 档(Spark)| **随年龄模型类别集** | 封闭 |
| pitch-level | 3(TextrolSpeech)~ 5(Spark)| **5 档(性别内归一)** ✅定 | 封闭 |
| pitch-range 起伏 | 少见单列 | **3 档(平/中/起伏大)** | 封闭 |
| energy/volume | 3(TextrolSpeech)| **3 档** | 封闭 |
| **pause/boundary** | **5(WordVoice-5A b0-b4)** | **直接采用 5 级** | 封闭 |
| rate/speed | 3(TextrolSpeech)~ 5(Spark)| **5 档** ✅定 | 封闭 |
| pitch-tone 轮廓 | **7(WordVoice-5A)** | **直接采用 7 类** | 封闭 |
| **emotion 基础类(L1)** | 5 ~ 9(Step/Textrol/emotion2vec)| **随 SER 模型类别集** ✅定 | 封闭 |
| emotion 强度 | 3(EmoInstruct)| **3 档(轻/中/强)** | 序数 |
| **emotion 细类(L2)** | 27(EmoInstruct/GoEmotions)| **折中:~20 精选起步池 + 增长** ✅定 | 半封闭 |
| register(一级)| 7(本方案)| **7 类(已定稿,二级先不启用)** | 封闭 |
| accent/方言 | 6(CosyVoice2)~ 19(CosyVoice3)| **第一期不纳入**(后期弱标)| 封闭 |
| environment/事件 | 6(SenseVoice AED)| **净/噪/乐 3 + 事件表** | 封闭 |
| 副语言(二期)| 10(Step)~ 13(FireRed)| 二期定 | 标签+时间戳 |

---

## 1. gender(性别)

| 方案 | 类别数 | 具体类别 | 来源 |
|---|---|---|---|
| TextrolSpeech | 2 | male / female | TextrolSpeech.pdf:133 |
| audeering age-gender | 3 | child / female / male | HF audeering/wav2vec2-age-gender |
| Spark-TTS(WavLM 分类器)| 2 | male / female(AISHELL-3 99.4%)| Spark-TTS.pdf §gender |
| InstructTTSEval(评测特征)| — | gender 作 12 特征之一 | InstructTTSEval.pdf:333 |

**建议**:male / female / **unknown**(低置信兜底)。是否单列 child 取决于所选分类器(audeering 有 child,但禁商用)。

## 2. age(年龄段)

| 方案 | 类别数 | 具体设置 | 来源 |
|---|---|---|---|
| Spark-TTS | 5 | Child / Teenager / Young Adult / Middle-aged / Elderly | Spark-TTS.pdf:2403 |
| audeering | 连续 | 0~1 回归(×100=岁)| HF 模型卡 |
| InstructTTSEval | 参考 | child / teen / young adult / middle-aged / elderly | InstructTTSEval.pdf:1299 |

**建议(已定)**:**由所选年龄打标模型的类别集决定**(与 gender 分类器同思路);若模型可选,优先对齐 Spark 5 档。选型是第一期工程件。

## 3. pitch(音高:level + range 两轴)

| 方案 | 档数 | 设置 | 来源 |
|---|---|---|---|
| TextrolSpeech | 3 | high / medium / low,**按数据整体分布切**(无公开阈值)| TextrolSpeech.pdf:157 |
| Spark-TTS | 5 | PyWorld 提 F0,5 级 | Spark-TTS.pdf |
| InstructTTSEval | free | 相对描述(如 "high female pitch, rising sharply")| InstructTTSEval.pdf:1299 |

**建议(已定)**:**level 5 档 + range 3 档**;level 必须**性别内归一化**(男女分开切),否则男声恒"低"。5 档阈值 pilot 定分位。
- **复用内部 WordVoice-5A**(词级 Pitch + Tone),**tone 轮廓 7 类现成**:flat / rise / rrise / fall / ffall / peak / valley [DSP-LLM韵律评估可行性验证方案.md:311]。词级 pitch/tone 可聚合成句级档,也支持后期 span 级。

## 3b. pause / boundary(停顿 —— 补漏维度)

之前遗漏。韵律核心维度,内部 WordVoice-5A 已产出。

| 方案 | 档数 | 具体 | 来源 |
|---|---|---|---|
| **WordVoice-5A(内部)** | **5** | b0 无停顿 / b1 ≤0.05s / b2 ≤0.18s 正常词间 / b3 ≤0.4s 逗号级 / b4 >0.4s 句号级 | DSP-LLM韵律评估可行性验证方案.md:303-308 |

**建议**:直接采用 WordVoice-5A 的 **b0-b4 5 级**,词级标注,可判停顿合理性(逗号后 ≥b3、句末 ≥b4、词中不应 b4)。

## 4. energy / volume(音量)

| 方案 | 档数 | 设置 | 来源 |
|---|---|---|---|
| TextrolSpeech | 3 | high / normal / low(librosa 能量,按分布切)| TextrolSpeech.pdf:157 |
| InstructTTSEval | 语义锚 | whisper → normal → shouting | InstructTTSEval.pdf:1299 |

**建议**:3 档(轻/中/响),分位切,语义锚对齐 whisper/normal/shout。

## 5. rate / speed(语速)

| 方案 | 档数 | 设置 | 来源 |
|---|---|---|---|
| TextrolSpeech | 3 | fast / normal / slow(forced-align 均时长)| TextrolSpeech.pdf:155 |
| Spark-TTS | 5 | 5 级 | Spark-TTS.pdf |

**建议(已定)**:**5 档**(极慢/慢/中/快/极快),分位切。复用 WordVoice-5A 词级 Duration 算字/音素率。

## 6. emotion 基础类(L1)—— 核心

| 方案 | 类别数 | 具体类别 | 来源 |
|---|---|---|---|
| **emotion2vec+ large** | **9** | angry / disgusted / fearful / happy / neutral / other / sad / surprised / unknown | HF emotion2vec_plus_large |
| TextrolSpeech | 8 | angry / contempt / disgusted / fear / happy / sad / surprised / neutral | TextrolSpeech.pdf:146 |
| Step-Audio-EditX | 5 | happiness / anger / sadness / fear / surprise | Step-Audio-EditX.pdf §5.1 |
| IndexTTS2 | 7 | 7 基础情感(文本→7 维概率)| IndexTTS2.pdf:80 |
| SenseVoice | 有 | SER 能力,官方卡**未逐一枚举** | HF SenseVoiceSmall |

**建议**:**由所选 SER 决定**。若用 emotion2vec+ = 9 类(有效控制约 7,去 other/unknown);为可评测,建议对齐"7 主类"(中性/高兴/悲伤/愤怒/恐惧/惊讶/厌恶)。

## 7. emotion 强度(intensity)

| 方案 | 档数 | 设置 | 来源 |
|---|---|---|---|
| EmoInstruct-TTS | 3 | low / medium / high,**序数排序约束**(非绝对阈值),IOA 0.78→0.91 | EmoInstruct-TTS.pdf:104 |
| Step-Audio-EditX | 连续 | large-margin 合成,margin 即强度 | Step-Audio-EditX.pdf §3.1 |

**建议**:3 档(轻/中/强),用**相对排序 / 配对**标,不用绝对打分。

## 8. emotion 细类(L2)—— 受控增长

| 方案 | 类别数 | 说明 | 来源 |
|---|---|---|---|
| **instruct_data_generator(内部生成项目)** | **~44** | **中文语音验证过的 multi-label(max 3)情绪表**:开心/兴奋/好笑/得意/温柔/撩/愤怒/烦躁/轻蔑/难过/悲痛/自怜/孤独/害怕/焦虑/紧张/惊慌/愣住/困惑/疲惫/无聊/认命/无助/内疚/尴尬/别扭/阴阳怪气/嘲讽/不服/委屈/期待/释然/感动/好奇/俏皮/温暖/关切/抱歉/鼓励/安抚/平静/从容/专注/嫌弃/惊讶 | `instruct_data_generator/schema/dimensions.json` |
| EmoInstruct-TTS | 27 | 27 细粒度 + 7 主类×3 强度=48 态;**原文未逐一枚举 27 类** | EmoInstruct-TTS.pdf:73 |
| GoEmotions(外部 NLP 参考)| 27 | admiration/amusement/anger/…/neutral(通用文本情感集)| 通用知识,非本 vault |
| Vox-Profile | 8+other | 8 情感 + arousal/valence/dominance 维度 | github vox-profile(禁商用/英文)|

**建议(已定 · 折中)**:**起步池 = instruct_data_generator 的 ~44 词中文情绪表**(中文语音场景验证过,优于 GoEmotions 西式集),映射到 7 主类下 + **受控增长清单**(选不到才提议新类、人工审批)。v0.1 起步池见 [[instructTTS标注方案设计]] §3 L2。**"社交态度类"(安抚/鼓励/关切/抱歉)保留为情感标签**——第一期即消解大量"安慰型"指令;**只有纯言语行为(命令/请求/质问/催促)归意图(二期)**。

## 9. register(场景,一级)—— 已定稿

**本方案 7 类(见主方案 §3,兼质量门)**:断句破碎 / 自然对话 / 影视娱乐 / 播报朗读 / 教学解说 / 游戏电竞 / 其他无效。

其他方案的场景/风格设置(**均非封闭 register 集,仅供二级细拆参考**):

| 方案 | 设置 | 来源 |
|---|---|---|
| **instruct_data_generator(内部,style 14)** | **general / casual_chat / news_broadcast / customer_service / teaching / storytelling / livestream / advertising / short_video / recitation / navigation / emotion_confide / speech / audiobook**;原则=只留"媒介/场合"、砍伪 style(comforting/persuasive/reasoning 等由 emotion+scene 承载)| `instruct_data_generator/schema/dimensions.json` style |
| Step-Audio-EditX(style 7)| childlike/elderly/exaggerated/recitative/passionate/coquettish/whisper | Step-Audio-EditX.pdf §5.1 |
| MOSS-VoiceGenerator | 影视来源域(storytelling/game/role-play/assistant)| MOSS-VoiceGenerator.pdf:13 |
| InstructAudio | style 为自由 NL 属性,无闭集 | InstructAudio.pdf:25 |
| VoxInstruct | scenario background 自由文本 | VoxInstruct.pdf:620 |
| CosyVoice2 role-play | 神秘/凶猛/好奇/优雅/孤独/机器人/佩奇 等(开放)| CosyVoice2.pdf:459 |

> **register 二级细拆的首选参考**:instruct_data_generator 的 14 style 是中文场景验证过、且已按"媒介/场合"清洗过伪 style 的成熟集——二级启用时优先在它基础上映射(其"scene_category 15 类话题维度"也可参考,若未来引入话题维度)。

## 10. accent / 方言

| 方案 | 类别数 | 具体 | 来源 |
|---|---|---|---|
| CosyVoice3 | 19 中文口音 | sichuan/hubei/cantonese/wuzhong/shan1xi/suhang/shanghai/hunan/shan3xi/minnan/henan/shandong/jiangxi/ningxia/gansu/yunnan/dongbei/guizhou/tianjin | CosyVoice3.pdf:764 |
| CosyVoice3 | +英语口音 | chinese-english / indian-english / russian-english | CosyVoice3.pdf:582 |
| CosyVoice2 | 6 方言 | 粤语/四川话/上海话/郑州话/长沙话/天津话 | CosyVoice2.pdf:453 |
| VoxCPM2 | 9 中文方言 | 未逐一枚举 | VoxCPM2.pdf:21 |

**建议(已定)**:**第一期不纳入**(无好开源中文方言分类器)。后期做时以 **CosyVoice3 19 口音**为超集,靠元数据 + OmniLLM 弱标。

## 11. environment / 音频事件

| 方案 | 类别数 | 具体 | 来源 |
|---|---|---|---|
| SenseVoice AED | 6 | BGM / applause / laughter / crying / coughing / sneezing | HF SenseVoiceSmall |

**建议**:environment 门用 **净/噪/乐 3 类**(BGM 检测复用 SenseVoice);音频事件表(6)为副语言二期起点。

## 12. 副语言事件(二期,span 级)

| 方案 | 类别数 | 具体 | 来源 |
|---|---|---|---|
| **NVSpeech** | **18** | 18 类 paralinguistic vocalizations(标注 48K 话语)| 可控TTS综述 §3.3 |
| **NVV-SuperBench** | **45** | 45 类 NVV(副语言评测,双语 4500 实例)| 可控TTS综述 §5 |
| Affectron | 15 | 15 类 NV(笑/叹/咳等,emotion-driven Top-K)| 可控TTS综述 §3.2 |
| FireRedTTS | 13 | 13 种副语言行为(笔记未逐一列)| FireRedTTS.pdf:31 |
| Step-Audio-EditX | 10 | breathing/laughter/surprise-oh/confirmation-en/uhm/surprise-ah/surprise-wa/sigh/question-ei/dissatisfaction-hnn | Step-Audio-EditX.pdf §5.1 |
| CosyVoice(标记)| — | [laughter]/[breath]/<strong> 等 | CosyVoice3.pdf |

> 二期若做副语言,建议以 **NVSpeech 18 类** 为标注起点、**NVV-SuperBench 45 类** 为评测对齐。

## 13. 数据集视角:各标注数据集的维度设置

(来源:[[可控TTS综述]] §5.1,聚合自各数据集论文)

| 数据集 | 维度数 | 维度清单 | 标注方式 | 规模 / 语种 |
|---|---|---|---|---|
| **SpeechCraft** | **8** | gender / age / pitch / energy / speed / emotion / **topic** / **emphasis** | 自动 pipeline + LLaMA2 rewrite 成 NL | 2391h / 2.25M clips,EN+ZH |
| **CapSpeech** | **9** | 固有 I1-5:gender/age/pitch/speed/expressiveness;表达 E1-4:emotion/accent/sound-event/chat-agent | Mistral-7B 自动生成 | 10M+ 机标 + 358K 人工,EN |
| TextrolSpeech | 5 | gender/pitch/speed/volume/emotion | 模板 × 432 组合 | 330h,EN |
| PromptSpeech | 5 | gender/pitch/speed/volume/emotion | 手动标注 | 330h,EN |
| StoryTTS | 5 | 句式 / 修辞 / 场景 / 角色 / 情感 | GPT-4 + Claude-2 | 60.9h,ZH 评书 |
| ESD | 5 情感 | neutral/angry/happy/sad/surprise | 专业录制 | 29h,EN+ZH |
| IEMOCAP | 离散+VAD | 离散情感 + arousal/valence/dominance | 演员录制 | 12h,EN |

**给本方案的启发**:
- **SpeechCraft 8 维与我方第一期高度重合**(多出 topic/emphasis),可作维度完整性的对照基准;
- **CapSpeech 的 I/E 分组(固有属性 vs 表达属性)** 是个好的组织思路——和我们"原子物理阶 vs 情感语用阶"的分阶异曲同工;
- 主流"NL 描述"数据集(SpeechCraft/CapSpeech)都走 **结构化维度标注 → LLM rewrite 成 NL** 两步,**印证我方"第一阶段结构化 + 第二阶段扩写"的路线**。

## 附:InstructTTSEval 评测 schema(12 特征 × 4 层)—— 评测锚点

| 层 | 特征 |
|---|---|
| 生理 | gender / pitch / texture |
| 语言 | clarity / fluency / speed |
| 社会 | accent / age / volume |
| 心理/语用 | emotion / tone / personality |

来源:InstructTTSEval.pdf:333-335。我方结构化维度应能映射到此 12 项,便于用其 APS/DSD/RP 三层评测。

---

## 拍板结果(2026-07-28)

1. ✅ **emotion 基础类**:随所选 SER 模型(SenseVoice / emotion2vec 等)的类别集。
2. ✅ **emotion 细类初始池**:折中——~20 个精选起步池(GoEmotions/Plutchik 中文高频细类,映射 7 主类)+ 增长式扩充;语气态度项归意图(二期)。
3. ✅ **age**:随所选年龄打标模型的类别集。
4. ✅ **pitch/speed**:**5 档**。
5. ✅ **accent**:第一期不纳入,后期弱标。
6. ✅ **register 二级**:先不启用,只用一级 7 类。

**连带结论**:emotion 基础类、age 均"随模型"——**SER 模型 + 年龄模型的选型是第一期第一步工程件,选完 taxonomy 即固化**。


1.emotion种类，使用sencevoice emotion2vec等SER模型支持的种类。
2.没看懂第二个问题。
3.age也是看 年龄打标模型支持几类。
4.五档
5.先不纳入。
6.register二级先不启用。
