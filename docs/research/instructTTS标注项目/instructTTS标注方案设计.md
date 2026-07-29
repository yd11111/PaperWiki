---
title: "instruct TTS 标注方案设计 v0.1"
created: 2026-07-27
tags: [research, tts, instruct, annotation, data-pipeline, voice-design]
---

# instruct TTS 标注方案设计 v0.1

> **文档状态:暂定方案(v0.1),支持后续持续修正**。维度体系、类别数、工具选型均可随 pilot 结果和讨论迭代。
> 姊妹篇:[[TTS基础模型训练数据调研]](训练数据规模视角)。本篇是"指令控制数据怎么标"的方法论视角。
> 所有可复用的档位规则 / 标签集 / 流水线算子均来自本 vault 已精读论文,后附 `文件:行号` 或 PDF 章节。

## 0. 目标与范围

- **终极目标**:通用 instruct(自然语言指令控制音色/情感/风格/韵律的全维度)。
- **数据策略**:真实挖掘打底 + 合成补长尾,两条腿。
- **交付标准**:够跑 100–500 条 pilot,验证标注一致性和档位阈值后再放量。

**第一期 / 第二期的划分(重要:两期都必要,非优先级/版本)**:

划分依据是**标注方法的性质 + 依赖关系**,不是"重要程度"或"v1/v2 迭代":

| | **第一期 · 可直接标注层** | **第二期 · 语义推理/派生层** |
|---|---|---|
| 有无现成工具 | 有(DSP 物理特征 / SER / gender-age 分类器 / register 文本分类)| **无现成模型** |
| 怎么得到 | 从信号或文本**直接标** | **重语义推理**(意图)或**跨句聚合派生**(人设)|
| 依赖 | 独立可跑 | **以第一期的结构化标签为输入** |
| 维度 | gender/age/pitch/energy/rate/pause/tone/emotion(L1/L2)/register/environment | intent(意图)/persona(人设)/accent/副语言 span |

两期共同构成完整 instruct 系统。**第二期不是"以后有空再做的增强",而是必须等第一期结构化标签就位、且需新建标注能力的必经环节。**

## 1. 三条设计原则

1. **属性分阶(order)决定一切**:一个维度距离原始声学信号越近,越能用自动标注器 + 封闭标签 + 越该进第一期;越是高阶语义组合,越需要 audio-LLM 出 NL 描述、越该往后放。
2. **高阶模糊桶在标注层拆成可测原子,在使用层再组合**:"音色""风格""人设"不作独立标注维度——它们各执一词、一致性必崩。音色 = gender+age+texture+pitch 的感知合成;风格 = register+情感+persona 的组合。
3. **先钉评测锚点再定维度**:以 InstructTTSEval 的 **12 特征 × 4 层**为公共坐标系(生理:gender/pitch/texture;语言:clarity/fluency/speed;社会:accent/age/volume;心理:emotion/tone/personality)[InstructTTSEval.md:81-85]。标了的维度必须落在这张评测图上,否则无法验证。

**维度关系模型(prosody 基质 → emotion/style 同级正交 → persona 顶层)**:

```
        persona(人设 · 组合顶层,由下面派生)
  ┌───────────┴───────────┐
emotion(状态轴)      style/register(情境轴)   ← 同级正交,互不包含
  └───────────┬───────────┘
     prosody(pitch/energy/duration/pause/tone · 物理实现基质)
```

- prosody 是**物理实现基质**(emotion 和 style 都通过它实现,清晰的底层)。
- **emotion(状态,逐句可变)与 style/register(情境,较稳定)是两条正交轴,不是上下层**——同一情境配不同情绪、同一情绪跨情境;style 只**弱约束** emotion 的取值范围(envelope),不包含它。佐证:[[ProsodyModeling]] 的 Variation Information 四分类把 prosody-style-emotion 归为一桶(信号层纠缠,非堆叠)。
- **persona 才是真正的组合顶层**(emotion 倾向 + style + 音色 + 性别年龄的派生)。
- 因此阶编号仅表"距声学信号的抽象距离",**不表示 style 高于 emotion**。

## 2. 维度体系(总矩阵)

| 阶 | 维度 | 标注器 | 标签形态 | 置信 | 粒度 | 第一期 |
|---|---|---|---|---|---|---|
| **0·原子物理** | 性别 gender | 开源分类器 checkpoint | 封闭 male/female/unknown | ★★★ | 句级 | ✅ |
| | 年龄段 age | 开源分类器 checkpoint | 封闭(取决于模型) | ★★☆ | 句级 | ✅ |
| | 语速 rate | forced-align 音素/字率 | 档位(分位切) | ★★★ | 句级 | ✅ |
| | 音高 pitch | F0 统计(level + range) | 档位 × 2 轴 | ★★★ | 句级 | ✅ |
| | 音量 energy | RMS/响度 | 档位 | ★★★ | 句级 | ✅ |
| **1·感知音质** | 音质纹理 texture | audio-LLM caption | NL + 可选粗档 | ★☆ | 句级 | ⚠️半 |
| | 口音/方言 accent | 元数据 / audio-LLM 弱标 | 封闭(方言表) | ★☆ | 句/说话人级 | ⏳后期(无好开源) |
| | 清晰度/流利度 clarity/fluency | 信号 + ASR 置信 | 档位 | ★★☆ | 句级 | ⚠️半(先作过滤) |
| **2·解释轴<br>(状态+情境,正交)** | 情感 emotion(**核心,三层级联**) | L1 SER + L2 audio-LLM + L3 text-LLM | 基础类 → 细类+强度 → 自由 NL | ★★☆ | 句级(未来 span) | ✅ |
| | 意图 intent(吸收 tone) | text-LLM(片段作上下文)+ 声学校验 | 半封闭 taxonomy + NL | ★☆ | 句级 | 二期(必要)|
| (同上·情境轴) | 场景 register/style | **ASR 文本 → LLM 分类 + 语音校验**(兼质量门) | 封闭 7 大类(**二级先不启用**) | ★★☆ | 文本/句级 | ✅ |
| | 背景环境 environment | 音频事件检测 | 封闭 净/噪/乐 | ★★☆ | 句级 | ✅(作过滤) |
| **3·组合顶层** | 人设 persona | emotion/style/音色/属性 聚合 + LLM | 片段级派生 NL | — | 片段级 | 二期(必要)|
| **横切·副语言** | 笑/叹气/呼吸/犹豫… | 事件检测 + audio-LLM | 标签 **+ 时间戳** | ★☆ | **span 级** | ⏳后期(先标"有无") |

**第一期维度清单**:gender / age / rate / pitch / energy / pause / tone / environment / **emotion** / **register**。(accent 第一期不纳入)
`⚠️半`:texture、intensity、clarity/fluency（先作数据过滤,不进控制维度）。
`⏳后期`:tone、persona、副语言 span。

> **关键 schema 决策**:tone 与 emotion 合成一个「情感/语气」维度族,但内部保留两个 facet(emotion 槽用 SER 分类器高置信标、第一期;tone 槽用 audio-LLM 半开放标、后期)。对使用层是一个标签,对标注管线是两个互不拖累的槽。副语言即使第一期只标整句"有无",schema 也必须预留时间戳字段,否则升 span 级要重标。

## 3. 第一期逐维标注规范

### gender（性别）
- **标注器**:**直接用开源分类器 checkpoint(第一期不自研)**。选型可参考 Spark-TTS 的 WavLM-large recipe(AISHELL-3 上 99.4%)[Spark-TTS.pdf §gender classifier],但优先找现成开源权重。
- **取值**:male / female / unknown。
- **质控**:分类器置信度 < 阈值 → unknown,不强判。

### age（年龄段）
- **标注器**:**开源分类器 checkpoint,不自研**。
- **取值**:由所选 checkpoint 的类别集决定;若可选,优先对齐 Spark-TTS 5 档(Child / Teenager / Young Adult / Middle-aged / Elderly)[Spark-TTS.pdf]。
- **粒度**:句级,但同说话人可聚合投票稳定化。

### rate（语速)
- **标注器**:forced alignment 后取 音素率 / 字率(TextrolSpeech 用 forced-align 平均时长 [TextrolSpeech.pdf:155-157])。
- **离散化**:**5 档**(极慢/慢/中/快/极快),按数据分位切(原文无公开阈值,pilot 后定死)。
- **未来 span**:同一句内变速留给后期。

### pitch（音高)
- **标注器**:F0 提取(WORLD / PyWorld,TextrolSpeech 用 WORLD [TextrolSpeech.pdf:155-157];Spark-TTS 用 PyWorld 5 级)。
- **取值(两轴)**:① level **5 档**(均值,分位切)② range 3 档(起伏)。level 按数据分位切。
- **说明**:pitch 与 gender 强相关,档位应**性别内归一化**(男女分别切),否则男声永远"低"。

### energy（音量)
- **标注器**:RMS / 响度(TextrolSpeech 用 librosa 提能量 [TextrolSpeech.pdf:155-157])。
- **取值**:3 档(轻/中/响),分位切。参考 InstructTTSEval 语义锚点 whisper→normal→shouting [InstructTTSEval.pdf:1299-1325]。

### accent（口音/方言)—— ⏳ 第一期不纳入(无好开源中文方言分类器,后期弱标)
- **标注器**:优先来源元数据;无则分类器 / audio-LLM。
- **取值(直接抄 CosyVoice3 的 18-19 中文口音 + 英语口音变体)**:sichuan, hubei, cantonese, wuzhong, shan1xi, suhang, shanghai, hunan, shan3xi, minnan, henan, shandong, jiangxi, ningxia, gansu, yunnan, dongbei, guizhou, tianjin;英语口音:chinese-english / indian-english / russian-english [CosyVoice3.pdf:764-786,582-634]。CosyVoice2 更小的 6 方言集(粤/川/沪/郑/长沙/津)可作 MVP 子集 [CosyVoice2.pdf:453]。
- **质控盲区**:方言的**客观指标不可靠**(Paraformer 无法识别方言,CER 失效 [CosyVoice2.pdf:1156]),只能主观评估——第一期方言标签仅作**属性标记**,不进 CER 门。

### environment（背景环境,作过滤)
- **标注器**:音频事件检测 + 去噪链。
- **取值**:净 / 有噪 / 带音乐。
- **用途**:第一期主要用于**数据清洗**——嘈杂/带 BGM 样本先标记,决定去噪或剔除(见 §6 DNSMOS 门)。

### emotion（情感 —— 核心可控维度,三层级联标注)

情感是后续主控维度,目标是**细粒度可控**。按"**信号源递进 + 粒度递增 + 结构递减**"分三层级联标注;每层只处理上一层筛出的表达性数据,天然省算力(90% 中性在 L1 就被拦下,不进昂贵的 L2/L3)。

**L1 · 基础类别(信号源=音频/SER)**
- 工具:开源 SER(emotion2vec+ 9 类 / SenseVoice),第一期不自研。
- 输出:封闭基础类(约 7 类)**+ 中性门**——中性直接归档,不进后续层。
- 定位:全量可比,**评测 + RL reward 锚点**。参考锚点:emotion2vec+ 9 类 [hf emotion2vec_plus_large]、Step-Audio-EditX 5 类 [Step-Audio-EditX.pdf §5.1]。
- ⚠️ **SER 的固有局限**:分类器只衡量"情感类别对不对",**不衡量"表达是否充分/自然"** [[TTS评估方法与前沿进展综述]]。这正是 L1 只能当锚点、必须靠 L2(细类+强度)和 L3(自由描述)补足表达力的根本原因——别指望 L1 承载细粒度。

**L2 · 细粒度类别(信号源=音频 + 文本 + L1 标签/audio-LLM)**
- 工具:audio-LLM,只对 L1=表达性 的样本再细分。
- **输入 = 三路融合**:① 语音(听声学)② ASR 转写文本(给语境)③ L1 基础类(给锚点)。文本消歧 + L1 约束,细分更准、不跑偏基础类。
- 输出:在基础类下**扩展细类**(愤怒→暴怒/恼怒/冷漠的怒/克制的怒…)+ 顺带 **强度正交档**(轻/中/强)。
- **这是情感类目增长的地方**。
- ⚠️ **防类目膨胀**:audio-LLM 从**受控增长的细类清单**里选,选不到才提议新类、**人工审批入库**——不是自由造词。否则近义词泛滥、一致性崩、无法评测。
- **初始池策略(折中,已定)**:用一份**中文语音场景验证过的情绪词表**作起步池(源自内部 instruct 数据生成项目 `instruct_data_generator` 的 44 词中文 emotion 表,远优于 GoEmotions 的西式集),映射到 7 主类下,再增长式扩充。**v0.1 起步池(~44,主类随 SER 最终确定,以下为占位映射)**:
  - 中性/平静 → 平静 / 从容 / 专注 / 疲惫 / 无聊 / 认命
  - 高兴 → 开心 / 兴奋 / 好笑 / 得意 / 俏皮 / 撩 / 期待 / 释然 / 感动 / 温暖 / 好奇
  - 悲伤 → 难过 / 悲痛 / 自怜 / 孤独 / **委屈** / 无助 / 内疚
  - 愤怒 → 愤怒 / 烦躁 / 轻蔑 / 阴阳怪气(讽刺) / 嘲讽 / 不服
  - 恐惧 → 害怕 / 焦虑 / 紧张 / 惊慌
  - 惊讶 → 惊讶 / 愣住 / 困惑
  - 厌恶 → 嫌弃
  - **社交态度类(作情感标,吸收大量"安慰型"指令)** → 温柔 / 关切 / **安抚** / **鼓励** / 抱歉 / 尴尬 / 别扭
- **与意图维度的边界**:上面的"社交态度类"(安抚/鼓励/关切/抱歉)**作为情感标签保留**——这正好在第一期就消解了大量"安慰的话"类指令(参考该项目做法:安慰 = scene 话题 + reassuring/tender 情感)。**只有纯言语行为**(命令 / 请求 / 质问 / 催促)才留给第二期意图维度,避免与情感重叠。
- 强度按正交档而非塞进类名(避免情感质地与强度耦合导致类目爆炸);可参考 EmoInstruct 的序数排序思路 [EmoInstruct-TTS.pdf:104-108]。

**L3 · 自由描述(信号源=文本 + 已有音频标注/text-LLM)**
- 工具:text-LLM 生成完全自由的情感/语气 NL 描述("嘴上说没事,语气里全是委屈")。
- **输入 = 三路融合**:① ASR 转写文本 ② L1 基础类 ③ L2 细类 + 强度。即在已知声学情绪标签的约束下,结合文本语义扩写自由描述。
- 定位:开放词汇,**训练信号**,捕捉语义/语用 nuance。
- **为何必须带 L1/L2**:纯文本会与声学矛盾(同一句"我没事"可轻松可委屈);以音频标签为事实锚,text-LLM 只负责把"克制的愤怒 + 这句话的内容"扩写成自然描述,不臆测情绪本身。

**(正交)混合分布**:混合情感用 7 维概率(SER logits / LLM 赋分),表达"六成悲三成怒"(参考 IndexTTS2 T2E 7 维分布)。零成本,可选。

**控制粒度 = 层的组合**:同一份数据支持多档控制——"生气"(L1)、"轻微生气"(L1+强度)、"强忍的克制愤怒"(L1+L3)、"六成悲三成怒"(正交)。**一次标注,多档控制**,是分层的核心收益。

**数据来源**:L1 粗类真实挖掘够(腿 A);L3 细描述真实挖掘 + LLM(腿 A);细类/表达性长尾靠 **style-guided mining**(腿 B,真实数据 90%+ 中性 [InstructAudio.pdf:287]);**合成(腿 C)仅在需要严格强度梯度或真实缺失的稀有组合时启用,保持后期**。

### register（场景,兼质量门)
- **标注器(双通道)**:**文本主判 + 语音校验**——① 主:对 ASR 转录文本做 LLM 7 类分类(基于文本内容判定);② 校验:audio-LLM 听语音复核,纠正纯文本分不开的类(如"新闻播报 vs 有声书叙述"、部分【断句破碎】需听声学确诊劣质合成)。两者不一致时以语音为准或进人工。放在流水线 ASR 步骤之后(见 §4 腿 A)。
- **设计要点**:体系把**数据质量过滤合进了 register 分类**——【断句破碎】和【其他无效】两类直接命中 ASR 错误 / 劣质合成 / 乱码 / 纯歌词,分类顺手清洗。
- **粒度**:文本/句级(每条转录判一次);同来源可辅助先验。

**一级封闭标签集(7 类,LLM 分类 prompt 已定稿)**:

> 请根据以下 7 个类别对输入的文本进行分类(必须严格使用以下分类名称之一):
> 1. **【断句破碎】**:文本中存在大量反直觉、不符合语法的异常标点(如频繁的逗号、句号将正常词语生硬切断),韵律极其奇怪。这类通常是 ASR 识别错误或劣质机器合成音。例:"那。剩余。气,体体。积,就。为零毫升。是吧。"或"这个我,不太。熟柴郡,猫"。
> 2. **【自然对话】**:无固定剧本、即兴交流。含口语化表达、正常停顿、重复。如日常闲聊、播客访谈、直播互动。
> 3. **【影视娱乐】**:有剧本支撑、具表演性质。如影视剧台词、综艺节目对话、广播剧。
> 4. **【播报朗读】**:纯书面语、单向输出、极少口语瑕疵。如新闻联播、有声书、天气预报、公共提示音。
> 5. **【教学解说】**:知识传递或事件叙述。如课程讲座、纪录片旁白、影视解说、科技科普。
> 6. **【游戏电竞】**:含游戏术语、游戏实况情绪化发音、游戏角色配音。
> 7. **【其他无效】**:乱码、无意义拼凑、纯歌词、无法归入以上类的无价值文本。

**二级细拆(v0.1 提议,一级稳定不变,二级供下游风格控制;待确认)**:

| 一级 | 用途 | 二级细分(提议) |
|---|---|---|
| 断句破碎 | ✂️ 剔除/降权 | 不细拆 |
| 自然对话 | ✅ 表达性来源 | 日常闲聊 / 播客访谈 / 直播互动 / 电话客服 |
| 影视娱乐 | ✅ 高表演性 | 影视剧台词 / 综艺对话 / 广播剧 |
| 播报朗读 | ✅ 高质量净音 | 新闻播报 / 有声书叙述 / 公共提示音(天气/通知) |
| 教学解说 | ✅ 叙述性 | 课程讲座 / 纪录片旁白 / 影视解说 / 科普 |
| 游戏电竞 | ✅ 情绪化 | 实况解说 / 游戏角色配音 |
| 其他无效 | ✂️ 剔除 | 不细拆 |

- **实现**:一级用给定 prompt 单次分类;二级可在一级命中后追加一次细分类(或让一级 prompt 直接吐"一级/二级"两段)。

### 第二期维度(语义推理/派生层):意图 · 人设

**两者都是完整 instruct 的必要维度**,归第二期不是因为次要,而是因为**无现成模型 + 需语义推理/派生 + 依赖第一期的结构化标签作输入**(见 §0 两期划分)。数据 schema 需在第一期就预留字段。当前几十秒片段数据即可支撑(几十秒 = 足够的上下文窗口 / persona 素材),无需先补跨片段说话人聚类。

> **典型例子:一句"安慰的话"**。它的核心是**意图=安慰安抚**(第二期),情感维度只能给"中性/温和"。第一期靠 **L3 自由描述**("温柔地安慰")先兜住语义;第二期补上结构化的 intent=安慰标签。两期配合才完整——这正说明第二期不可省。

**意图 intent(语用层,吸收原 tone facet)**
- **定义**:一句话的交际目的(安抚 / 说服 / 质问 / 请求 / 命令 / 调侃 / 道歉 / 催促…)。语气态度并入此维。
- **标注器**:text-LLM 为主,输入 = 句子文本 + **所在几十秒片段作上下文**;声学可选校验。复用已有 ASR 文本。
- **单元**:句 / turn 级。
- **取值**:借 dialogue-act / speech-act taxonomy 裁剪出半封闭意图集 + NL 补充。
- **数据分流**:独白片段(有声书/播报)意图弱(多为陈述/叙述),按 register **只在【自然对话】【影视娱乐】等交互片段重点标**。
- **上下文限制**:几十秒局部上下文足够判意图,不需整场对话历史。

**人设 persona(片段级派生描述)**
- **定义**:片段说话人的角色画像(温柔知性主播 / 毒舌游戏主播 / 沉稳权威男主播…),是 gender+age+音色+情感倾向+register+说话习惯 的高阶组合。
- **标注器**:**bottom-up 派生**——聚合片段内 L1/L2 情感分布 + register + 性别/年龄/音色标签,加一次 holistic(整段音频+文本)pass,LLM 合成 persona 描述。**本质是"片段级的 L3"**,契合"高阶维度=低阶组合"原则(§1 原则 2)。
- **单元**:**片段级(几十秒 = 一个 persona)**。注意是片段级人设,**非稳定说话人身份**(无跨片段聚类);对片段式训练无碍——每样本自带 persona 描述即可。
- **评测**:RP / role-play 式(生成语音是否贴合 persona 指令),难度最高。

### 腿 A · 真实挖掘 + 全 LLM 打标(对标 MOSS-VoiceGenerator)

逐算子(可直接照搬)[MOSS-VoiceGenerator.pdf §2.2]:

1. **说话人分离** — DiariZen
2. **去噪 + 质量过滤** — MossFormer2_SE_48K 去噪(擅长 BGM/音乐叠加)→ **DNSMOS ≥ 3.0** 硬门。关键数据:不去噪仅 ~5% 过阈,去噪后升到 **45–50%**
3. **单说话人过滤** — 防同性别混淆
4. **ASR 转录** — Qwen3-Omni-30B(强于数字/符号/标点,标点承载情感线索);规则过滤空/重复条目防 collapse
5. **register 分类 + 文本质量门** — 对转录文本跑 §3 的 7 类 LLM 分类:命中【断句破碎】/【其他无效】**直接剔除**(ASR 错误/劣质合成/乱码/歌词),其余打上 register 标签继续
6. **语言过滤** — Whisper-large-v3 剔非中英
7. **属性标注** — 开源 gender/age/SER checkpoint 打结构化标签(第一期不自研)
8. **Captioning** — Gemini-2.5-Pro 出详细 speech caption(覆盖 age/emotion-tone/texture,对标 InstructTTSEval APS 12 属性)
9. **Instruction 生成** — Qwen3-32B 把 caption + 结构化标签转成 NL 指令

产出参考量:Phase1 ~5000h;英文不足时**每条生成 2 个语义等价指令变体**翻倍信号。

### 腿 B · style-guided mining 补长尾(解决情感/风格中性偏斜)

[MOSS-VoiceGenerator.pdf:244-262] 这是"从中性池挖表达性长尾"最清晰的方案:
1. 微调专用 caption 模型(Qwen3-Omni-Thinking)降本;
2. 训 **Speech-CLAP** 把 style 指令与语音映射到共享空间;
3. **GPT-5 生成多样化 style query** 作检索种子;
4. 每 query 取 **cosine top-50**,**取后即从候选池移除**保证不重复;
5. 补 ~10000h 表达性数据。

### 腿 C · 合成配对补稀缺(对标 Step-Audio-EditX)—— ⏳ 第一期不上,后期启用

第一期先纯真实挖掘(腿 A + 腿 B);待真实数据长尾仍不足时再启用合成。情感/风格/副语言的稀缺组合靠合成 [Step-Audio-EditX.pdf §3.1]:
- **large-margin 三元组** ⟨text_prompt, audio_neutral, audio_emotion⟩:演员每情感/风格录 ~10s → StepTTS 克隆出同说话人中性+情感配对,**两条用同一文本 prompt**逼模型只学情感变化;
- **margin scoring model**:小规模人工标注训练,音频对 1-10 打分,**保留 ≥6**;
- **副语言四元组**:借 NVSpeech 副语言标注,把去 tag 的克隆音频作输入、原音频作目标——**时域大 margin,免打分**。

## 5. 标签 → 自然语言指令(封闭标签与 NL 双轨)

第一期产出**结构化标签**(便于评测/RL),再用 LLM 扩成 NL 指令(便于训练)。复用 TextrolSpeech 的**四阶段 prompt programming**,一组属性可扩 500 种表述 [TextrolSpeech.pdf:160-236]:
1. **Base**:给关键词(如 male / high pitch / fast / normal energy)要求生成一句自然描述;
2. **增多样性**:允许同义词/自由发挥(用 tone/key/volume 描述 pitch);
3. **减无关**:禁止无关内容(如场景"教堂");
4. **few-shot 模板**:给手工范例定风格。

## 6. 质控与评测

- **双标交叉**:分类器 + audio-LLM 一致才入可信;不一致进人工小样本核验(EmoInstruct 的 Dataset-Annotation 即人工核验层 [EmoInstruct-TTS.pdf:226-230])。
- **质量硬门**:DNSMOS ≥ 3.0(过滤)、ASR 置信(clarity/fluency)。
- **评测锚点**:对标 InstructTTSEval 三抽象层 [InstructTTSEval.pdf:349-373]——
  - **APS**:逐维 free-form 指令,测精确映射;
  - **DSD**:LLM 把 APS 改写成自然段并随机丢部分特征(dropout 权重集);
  - **RP**:只给角色/场景,测世界知识推理。
  - 用 Gemini-as-judge True/False 评判。caption 生成禁用 "moderate"、要求描述时变 [InstructTTSEval.pdf:1342]。

## 7. 中文特化 gap(机会点)

现有工作**几乎没人处理**:语气词(呢/吧/啊)、儿化、句末语气、方言腔的标注 [各篇均未提及]。CosyVoice3 仅做多音字 Pronunciation Inpainting。**这是本方案可以做出差异化的空间**——第一期先在 schema 里预留字段,后期专门设计。

## 8. 第一期 pilot 建议

0. **选型工程件(第一期第一步)**:定开源 gender / age / SER checkpoint —— SER 的类别集直接固化为情感 taxonomy;
1. 选 1-2 个高质量来源域(如有声书 + 客服),先跑腿 A 全流程 100–500 条;
2. **定死档位阈值**:在这批数据上算 pitch/rate/energy 的 33/67 分位,固化为档位边界;
3. **测标注一致性**:SER vs audio-LLM 的 emotion 一致率、gender 分类器准确率;register 7 类 LLM 分类的抽检准确率(尤其【断句破碎】召回);
4. 通过后接腿 B(style-guided mining)放量;腿 C 合成后期再评估。

## 已决(v0.1 定稿)

- **emotion(核心)**:三层级联标注——L1 SER 基础类(评测/RL 锚点)→ L2 audio-LLM 细类+强度(受控增长清单)→ L3 text-LLM 自由描述(带音频标签作条件)。目标细粒度可控。见 §3。
- **emotion 基础类**:随所选 SER 模型(SenseVoice/emotion2vec 等)类别集(L1)。
- **age**:随所选年龄打标模型类别集。
- **pitch level / speed**:均 **5 档**;pitch-range 3 档;energy 3 档;pause b0-b4;tone 7 类。
- **accent**:第一期不纳入,后期弱标。
- **register**:7 类文本分类 + 语音校验双通道,兼质量门;**二级先不启用**。
- **意图 / 人设(二期规划)**:意图 = 句级、text-LLM + 片段上下文、吸收 tone、按 register 分流;人设 = 片段级 bottom-up 派生(=片段版 L3)。几十秒片段数据即可支撑,见 §3。
- **合成腿**:第一期不上,先纯真实挖掘 + audio-LLM 细标;合成仅为强度梯度/稀有组合,后期启用。
- **分类器**:第一期尽量用开源 checkpoint;gender/age 因商用友好开源缺口,处理方式待拍板(见 §9.3)。

## 9. 工具选型(仅新增模块)

> **现有 pipeline 已有 VAD / ASR / 说话人分离,不重选**。以下只覆盖为 instruct 新增的标注模块,标清挂载点。
> 选型信息经 WebFetch 核对(HuggingFace/GitHub 模型卡,2026-07);Playwright 当时被残留实例锁死,故用 WebFetch 兜底,**许可条款以官方为准**。

**前置依赖**:你们 ASR 是否输出**词级时间戳**?
- 输出 → rate / pitch 直接复用,不加工具;
- 不输出 → 补 WhisperX(BSD)或 MFA 做对齐。

### 9.1 选型表

| 模块 | 推荐工具 | 输出/类别 | 许可(商用) | 中英 | 挂载点 | 置信 |
|---|---|---|---|---|---|---|
| pitch | parselmouth(Praat)/ pyworld | F0 → level+range 档 | GPL/MIT 类,✅ | ✅ | 音频段 | ★★★ |
| energy | librosa RMS / pyloudnorm | LUFS/RMS → 档 | ISC/MIT,✅ | ✅ | 音频段 | ★★★ |
| rate | 复用 ASR 词级时间戳 / WhisperX | 字·音素率 → 分位档 | BSD,✅ | ✅ | ASR 输出 | ★★★ |
| **emotion(SER)** | **emotion2vec+ large**(主)/ SenseVoice(备) | **9 类**(见下)| 自定义 model-license,**待核** | ✅ | 音频段 | ★★☆ |
| register | Qwen3-32B / GPT LLM 文本分类 | 7 大类(+二级)| 视所选 LLM | ✅ | **ASR 文本** | ★★★ |
| quality 门 | DNSMOS(MOS)+ SNR 估计 | MOS/SNR | 开源,✅ | ✅ | 音频段 | ★★★ |
| 音频事件(副语言,后期) | SenseVoice AED | BGM/掌声/笑/哭/咳/嚏 | 待核 | ✅ | 音频段 | ★★☆ |
| **gender + age** | ⚠️ 见 §9.3 许可缺口 | gender 3 类 / age | **最优选禁商用** | — | 单人段 | ★☆ 待定 |
| accent/方言 | 元数据 + audio-LLM 弱标 | CosyVoice3 口音表 | — | ✅ | 音频段 | ★☆ 后期 |
| audio caption | Gemini-2.5-Pro / Qwen3-Omni | 12 属性 free-form | API / 开源 | ✅ | 音频段 | ★★☆ |

### 9.2 emotion taxonomy(选 SER = 定类别)

- **emotion2vec+ large**(FunASR,`iic/emotion2vec_plus_large`,~300M,42.5K h 训练):**9 类 = angry / disgusted / fearful / happy / neutral / other / sad / surprised / unknown**。有效控制类约 7(去 other/unknown)。纯音频、无语义。
- **SenseVoice-Small**(FunAudioLLM):**ASR + 语种 + 情感 + 音频事件一体**,中英粤优化,70ms/10s(比 Whisper 快 15×);音频事件 = BGM/掌声/笑/哭/咳/嚏。情感标签官方卡未逐一枚举。

**架构建议**:你们已有 ASR,SER 的边际价值在情感 + 副语言事件。两条路——
1. **taxonomy 清晰优先**:emotion2vec+ 定情感(9 类固化),SenseVoice 供音频事件/BGM(顺带环境门)后期接;
2. **省算力一体**:纯 SenseVoice 出情感 + 事件 + 语种。
推荐 **路 1**——情感类别可控可评测更重要;SenseVoice 的 BGM 检测还能白送 environment 质量门。

### 9.3 ⚠️ 许可红线(生产必看)

| 工具 | 许可 | 结论 |
|---|---|---|
| audeering wav2vec2-age-gender | cc-by-nc-sa-4.0 | **禁商用** |
| Vox-Profile(age/sex/accent/emotion 一体) | RAIL,No commercial + 仅英文 | **禁商用 + 不支持中文** |
| 3D-Speaker | Apache-2.0 | 可商用,但**只有语种 ID(中/英),无 gender/age/方言** |
| emotion2vec+ / SenseVoice | 自定义 "model-license" | **需法务核** |

- **核心灰区判断**:用 NC 许可模型做"内部标注训练数据"这一用途,是否构成"商用"?建议走**法务确认**,别默认可用。
- **gender/age 缺口**:商用友好的开源 age/gender 分类器**基本没有**(gender 尚可拼,age 难)。这与"第一期不自研"原则冲突——**要么破例自训一个轻量 gender/age head(数据易得),要么用 audio-LLM 弱标,要么走内部已有模型**。需拍板。

### 9.4 缺口小结

1. **gender/age**:无商用友好开源 → 三选一(轻量自训 / audio-LLM / 内部模型)。
2. **中文方言/口音分类**:无好开源 checkpoint → 元数据 + audio-LLM 弱标,建议后期。
3. **SER/SenseVoice 许可**:自定义 license 需法务核可否用于商用标注。

---

## 10. 执行计划

挂载在现有 **VAD / ASR / 说话人分离** 之后。总原则:**先结构化、后自然语言**;结构化内部按级联依赖排序,能并行的并行。

> **架构依据(内部 + 外部双重印证)**:
> - **内部**:[[DSP-LLM韵律评估可行性验证方案]] 已确立核心假设——**"DSP 词级客观标注 + LLM 推理" > "LLM 直接听音频"**(Gemini 评韵律 PCC=-0.05,比随机差;LLM 不擅长从波形感知细粒度韵律)。本方案"先结构化(DSP/分类器)、后 NL 扩写(LLM)"正是这一假设的落地。
> - **外部**:BatonVoice(2025)Conductor LLM 把指令解析成结构化声学特征 JSON → TTS 执行;SpeechCraft/CapSpeech 也都是"结构化维度标注 → LLM rewrite 成 NL"两步(见 [[可控TTS综述]] §2.4.3 / §5.1)。
> **结论:"结构化打底 + NL 扩写"是内外部共同验证的路线。**

### 10.1 工具 → 维度映射

| 工具类别 | 负责维度 | 阶段 |
|---|---|---|
| **内部 WordVoice-5A**(词级 Duration/Boundary/Energy/Pitch/Tone,MFA+Qwen3FA)| pitch · pitch-tone · energy · rate · **pause/boundary** | S1 |
| **开源专用标注**(SER / gender-age / DNSMOS-VAD) | emotion L1 基础类 · gender · age · environment/quality 门 | S1 |
| **文本 LLM** | register 主判 · **L3 情感自由描述** · NL 指令扩写 · 意图(二期) | S1(register)/ S2 |
| **语音 OmniLLM** | emotion L2 细类+强度 · register 语音校验 · accent 弱标 · audio caption | S1 / S2 |

### 10.2 第一阶段 · 结构化标注

产出:**每片段/句一份结构化标签 JSON**。按依赖分四步:

1. **物理特征(复用内部 WordVoice-5A,词级)** — 直接产出词级 Duration/Boundary/Energy/Pitch/Tone;聚合成句级档(pitch level+range、energy、rate 按 **33/67 分位**固化),pause 用 b0-b4、tone 用 7 类现成 taxonomy。**词级时间戳依赖已由 WordVoice-5A(MFA+Qwen3FA)解决,无需另接 WhisperX**。
2. **开源专用维度(并行)** — ① gender/age 分类器;② SER → **emotion L1 基础类 + 中性门**(中性归档,不进后续);③ DNSMOS + VAD → 音频质量/environment。
3. **register(文本主判 → 语音校验)** — 文本 LLM 跑 7 类分类,**【断句破碎】【其他无效】直接剔除**(兼质量门);OmniLLM 听音复核文本分不开的类。
4. **emotion L2 细分(OmniLLM,依赖 L1)** — 输入 = 语音 + ASR 文本 + L1 标签 → 细类 + 强度档;细类从**受控增长清单**选,新类走人工审批。

> 级联依赖:step4 需 step2 的 L1;register 与情感可并行。中性门在 step2 就把 ~90% 数据挡在 L2 之外,省 OmniLLM 算力。

### 10.3 第二阶段 · 自然语言扩写

输入 = 第一阶段的全部结构化标签。产出:**(音频, 文本, 结构化标签, NL 指令)训练样本**。

1. **L3 情感自由描述** — 文本 LLM,输入 = ASR 文本 + L1 + L2 标签 → 开放 NL 情感描述(以音频标签为事实锚,不臆测)。
2. **整合 NL 指令生成** — 用 **四阶段 prompt programming**(TextrolSpeech 法 [TextrolSpeech.pdf:160-236])把全维度结构化标签 + L3 描述扩成多样化自然语言指令,一组属性可扩数百种表述。

### 10.4 推进节奏

- **Pilot(先)**:1-2 来源域、100–500 条,跑通 S1+S2 全链路 → 定分位阈值 + 初始化情感细类清单 + 测标注一致性(SER vs OmniLLM 情感一致率、register 抽检、断句破碎召回)。
- **放量**:接 style-guided mining(腿 B)补情感/表达性长尾。
- **二期**:意图、人设;合成腿(腿 C)仅为强度梯度/稀有组合。

### 10.5 关键依赖 / 待决(阻断项)

| 项 | 影响 | 状态 |
|---|---|---|
| ~~ASR 词级时间戳~~ | 已由内部 WordVoice-5A(MFA+Qwen3FA)解决 | ✅ 已解 |
| gender/age 商用许可缺口 | 自训轻量 head / OmniLLM 弱标 / 内部模型 三选一 | 待拍板(§9.3) |
| SER checkpoint 选型 | = 定 emotion L1 基础类集 | 待选(§9.2) |
| 情感细类清单初始化 + 审批流 | L2 能否防类目膨胀 | 待建 |
| register 二级细拆 | 是否需要更细场景 | 待确认 |

---

## 附录:数据源引用

- 档位离散化规则:TextrolSpeech(按分布切,无公开阈值)[TextrolSpeech.pdf:155-159]
- 情感 taxonomy + 强度序数:EmoInstruct-TTS(27 细粒度 + 7×3,ranking loss)[EmoInstruct-TTS.pdf:73-108]
- 评测 12×4 schema + APS/DSD/RP:InstructTTSEval [InstructTTSEval.pdf:333-373,1293-1558]
- 真实挖掘流水线 + DNSMOS 门 + style-guided mining:MOSS-VoiceGenerator [MOSS-VoiceGenerator.pdf §2.2]
- 合成三元组 + margin 打分 + 副语言四元组:Step-Audio-EditX [Step-Audio-EditX.pdf §3.1]
- gender/age 分类器 recipe:Spark-TTS(WavLM-large,age 5 档)[Spark-TTS.pdf]
- 方言/口音标签集:CosyVoice3(18-19 中文口音)[CosyVoice3.pdf:764-786]、CosyVoice2(6 方言)[CosyVoice2.pdf:453]
- 情感中性偏斜:InstructAudio(90%+ 中性)[InstructAudio.pdf:287]、VoxCPM2(分类器预筛)[VoxCPM2.pdf:884-907]
- 工具选型(WebFetch 核对,2026-07):emotion2vec+ 9 类 [hf.co/emotion2vec/emotion2vec_plus_large]、SenseVoice 一体+AED [hf.co/FunAudioLLM/SenseVoiceSmall]、audeering age-gender cc-by-nc-sa [hf.co/audeering/wav2vec2-large-robust-24-ft-age-gender]、Vox-Profile RAIL/英文 [github tiantiaf0627/vox-profile-release]、3D-Speaker Apache-2.0/仅语种 [github modelscope/3D-Speaker]
