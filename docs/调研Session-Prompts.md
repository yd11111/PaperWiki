# TTS 行业调研 — 各 Session Prompt

> 复制粘贴到新 session 即可执行。每个 session 调研 3-4 个团队。

---

## 通用前置 Prompt（每个 session 开头先发这段）

```
你现在是 TTS 行业研究分析师。请用以下"5层搜索法"系统调研指定团队在 2024-2026 年的语音/TTS/音频 全部研究工作。

## 5层搜索法

Layer 1 组织搜索:
- 访问 GitHub org → 列出所有 speech/audio 相关 repo (含 star 数)
- 访问 HuggingFace org → 列出所有模型/数据集 (含下载量)
- 搜索官方技术博客/公众号

Layer 2 核心人搜索:
- 从已知论文找到 2-3 个核心人
- 用 arXiv 搜索每个核心人的 2024-2026 全部论文
- 从合作者网络扩展发现更多工作

Layer 3 arXiv affiliation 搜索:
- 搜索 "[公司/机构名]" 组合以下关键词: speech, audio, voice, TTS, text-to-speech, codec, tokenizer, vocoder, synthesis, singing, dialogue, spoken language model
- 不限论文标题,看作者 affiliation 字段

Layer 4 产品/竞赛反推:
- 他们的语音产品叫什么,背后可能用了什么技术
- 参加了哪些竞赛 (Blizzard/VoiceMOS/Interspeech Challenge/ICASSP Challenge)
- 招聘 JD 中暴露的研究方向

Layer 5 引用网络:
- 从已知论文的 references 找同团队早期工作
- 从 cited by 找后续迭代

## 输出格式（每个团队必须填完）

### [团队名]

**基本信息**
- 公司/机构:
- 团队名/实验室名:
- GitHub org:
- HuggingFace org:
- 核心人物 (2-3人, 含职位):
- 产品线:

**论文时间线 (2024-2026, 按时间倒序)**
| 时间 | 论文 | arXiv/会议 | 方向标签 |

**技术栈全景**
- Tokenizer/Codec:
- 生成模型:
- Post-training:
- 流式/效率:
- 评估:
- 其他(VC/SVS/Edit/Safety/数据):

**架构演进** (用箭头画代际关系)

**独特技术赌注** (他们押注了什么别人没押的)

**开源情况** (repo名, star数, 下载量)

**判断**
- 优势:
- 短板:
- 下一步方向推测:
- 与行业其他玩家的差异化:

## 重要规则
- 所有网络访问用 Playwright (mcp__playwright 工具)，不用 WebFetch/WebSearch
- 宁可多搜不可漏搜,关键词要覆盖 speech/audio/voice/TTS/codec/tokenizer/vocoder/ASR/singing
- 如果某个 Layer 搜不到结果,明确标注"未找到",不要跳过
- 论文时间线要尽可能完整,不只列最重要的3-5篇

现在开始调研以下团队:
```

---

## Session 1: 阿里双团队 + 字节 + 阶跃 + 智谱

```
[粘贴上面的通用前置 Prompt]

现在开始调研以下团队:

### 团队1a: 通义语音/FunAudioLLM (阿里巴巴-通义实验室-语音团队)
- 已知 GitHub: FunAudioLLM, modelscope
- 已知 HuggingFace: FunAudioLLM
- 已知核心人: 叶杰平(Jieping Ye, VP), Shiliang Zhang(技术Lead), 杜智昊(Zhihao Du, CosyVoice一作)
- 注意: 武执政(Zhi-Zheng Wu)是港中文(深圳)副教授, 非阿里员工, 是外部合作者
- 注意: 前负责人鄢志杰(Zhijie Yan) 2025.02 离职加入腾讯
- 已知工作: CosyVoice 1/2/3, SenseVoice, Fun-Audio-Chat, FunASR, ThinkSound, FunMusic
- 重点关注: CosyVoice 3 (2025.05) 之后有没有新工作? 鄢志杰离职后团队方向有无变化?

### 团队1b: Qwen语音 (阿里巴巴-通义实验室-千问团队)
- 已知 GitHub: QwenLM
- 已知 HuggingFace: Qwen
- 已知核心人: 周靖人(Jingren Zhou, CTO), Hangrui Hu(Qwen3-TTS一作, 曾参与CosyVoice 1)
- 注意: 前Tech Lead林俊旸(Junyang Lin) 2026.03 离职
- 已知工作: Qwen-Audio, Qwen2-Audio, Qwen3-TTS, Qwen3-Omni
- 重点关注: Qwen3-TTS的技术细节, 与CosyVoice的tokenizer/数据是否独立? 林俊旸离职后团队变化?
- 注意: 通义语音和Qwen语音同属通义实验室但人员几乎零重叠, 技术路线独立, 需分别调研并对比

### 团队2: 字节跳动/Seed-TTS (ByteDance)
- 已知核心人: 从 Seed-TTS/Seed-VC 论文找
- 无公开 GitHub org (需要从论文作者找)
- 已知工作: Seed-TTS, Seed-VC, TLC, M3-TTS, ProsodyEval, Attention Guidance, JoyVoice
- 重点关注: 2026年有无新论文? "Seed-TTS 2"? 产品侧(豆包/剪映)技术演进?

### 团队3: 阶跃星辰/StepAudio (StepFun)
- 已知 GitHub: stepfun-ai
- 已知核心人: 从 Step-Audio 论文找
- 已知工作: Step-Audio, Step-Audio-EditX, StepAudio 2.5
- 重点关注: 2.5之后有无新工作? RLHF三模式的后续?

### 团队4: 智谱/GLM-4-Voice (Zhipu AI)
- 已知 GitHub: THUDM
- 已知核心人: 从 GLM-4-Voice 论文找
- 已知工作: GLM-4-Voice
- 重点关注: GLM-5有无语音能力? 2026年后续论文?

调研完后,在最后给出五个团队的横向对比矩阵。特别要对比通义 vs Qwen 的分工和路线差异。
```

---

## Session 2: 国内大厂第二梯队

```
[粘贴通用前置 Prompt]

现在开始调研以下团队:

### 团队1: MiniMax
- 已知工作: MiniMax-Speech (2505.07916)
- 产品: 海螺AI
- 重点: Learnable Speaker Encoder技术细节, 后续迭代, 其他音频工作

### 团队2: 小红书/FireRedTTS (REDNote/Xiaohongshu)
- 已知工作: FireRedTTS, FireRedTTS-1S, FireRedASR, SoCodec, PodAgent
- 搜索: "小红书" OR "REDNote" OR "Xiaohongshu" + speech/audio
- 重点: SoCodec技术细节, 有无更多codec/TTS工作, 团队核心人

### 团队3: 科大讯飞 (iFlytek)
- 已知工作: SparkAudio, PhonemeVec, 语音同传2.0
- 注意: 与中科大深度绑定,区分讯飞独立工作 vs 合作工作
- 搜索: "iFlytek" OR "科大讯飞" + TTS/speech, 也搜中科大语音实验室
- 重点: 2025-2026完整论文列表, 竞赛成绩

### 团队4: 蚂蚁集团/Ming (Ant Group)
- 已知工作: Ming-UniAudio, DualSpeechLM, ReStyle-TTS, AT-ADD
- 搜索: "Ant Group" OR "蚂蚁" + speech/audio
- 重点: Ming-UniAudio完整能力, 统一音频模型架构细节

调研完后,给出四家的横向对比矩阵。
```

---

## Session 3: 国内新锐+游戏/社交

```
[粘贴通用前置 Prompt]

现在开始调研以下团队:

### 团队1: 京东 (JD AI)
- 已知工作: JoyVoice, Ham-TTS, JoyHallo, JoyAvatar, Ditto
- 搜索: "JD" OR "Jingdong" OR "京东" + speech/audio/TTS
- 重点: JoyVoice长上下文技术, 数字人+TTS联动

### 团队2: 天工/昆仑万维 (Kunlun/Tiangong)
- 已知工作: MoE-TTS, Mureka TTS, SkyMusic, Skyo
- 搜索: "Kunlun" OR "天工" OR "Tiangong" + speech/audio/TTS/music
- 重点: MoE-TTS论文详情(如果有arXiv), Voice Design技术细节

### 团队3: Soul App (Soulgate)
- 已知工作: SoulX-Singer, OpenTalking, Teller, SoulX-Duplug
- 已知关联: X-LANCE (SoulX-Duplug是Interspeech 2026投稿)
- 搜索: "Soul" OR "Soulgate" + speech/audio/singing
- 重点: SoulX-Singer 42000小时训练细节, 全双工模块

### 团队4: 米哈游 (miHoYo/HoYoverse)
- 已知工作: SpeechRole, EmoMix, 与高校合作
- 搜索: "miHoYo" OR "HoYoverse" OR "米哈游" + speech/voice/emotion
- 重点: 游戏配音AI化进展, 情绪表达技术

调研完后,给出四家的横向对比矩阵,重点标注各家的场景差异化。
```

---

## Session 4: 国际公司 A（语音AI创业公司）

```
[粘贴通用前置 Prompt]

现在开始调研以下团队:

### 团队1: ElevenLabs
- 产品页: elevenlabs.io
- 搜索: 有无发表论文(arXiv "ElevenLabs"), 博客技术文章
- 重点: 技术架构(如果公开), 产品矩阵演进, 融资和估值时间线

### 团队2: Cartesia
- 已知 GitHub: cartesia-ai
- 已知技术: State Space Model (SSM) for speech
- 搜索: "Cartesia" + speech/audio, 搜 SSM for TTS 相关论文
- 重点: Sonic系列模型技术细节, SSM vs Transformer对比

### 团队3: Sesame AI
- 已知 GitHub: sesame (CSM)
- 已知 HuggingFace: sesame
- 已知工作: CSM-1B (Conversational Speech Model)
- 搜索: "Sesame" + speech, 从CSM论文/博客找技术细节
- 重点: CSM架构细节, 后续迭代(CSM-2?), 对话韵律技术

### 团队4: Fish Audio
- 已知 GitHub: fishaudio
- 已知工作: Fish-Speech, Fish-Speech S2
- 搜索: "Fish Audio" OR "fishaudio" + speech/TTS
- 重点: Fish-Speech系列技术演进, 开源生态, 与其他开源TTS对比

调研完后,给出四家的横向对比,重点对比商业模式+技术路线差异。
```

---

## Session 5: 国际公司 B（大厂语音能力）

```
[粘贴通用前置 Prompt]

现在开始调研以下团队:

### 团队1: Voxtral/Mistral AI
- 已知 HuggingFace: mistralai (Voxtral-4B-TTS-2603)
- 已知工作: Voxtral Small/Mini (理解), Voxtral TTS (生成), Voxtral Transcribe
- 搜索: "Mistral" + speech/voice/audio, arXiv搜Voxtral
- 重点: 4B TTS模型架构, LLM-native语音方案

### 团队2: Meta
- 已知工作: Seamless系列, MOSHI(合作?), 收购PlayHT
- GitHub: facebookresearch
- 搜索: "Meta" OR "Facebook" + TTS/speech synthesis 2025-2026
- 重点: 收购PlayHT后做了什么, Seamless后续, 有无新TTS系统

### 团队3: Google DeepMind
- 已知工作: SoundStorm, AudioPaLM, Gemini Live语音
- 搜索: "Google" OR "DeepMind" + speech synthesis/TTS 2025-2026
- 重点: Gemini语音能力背后技术, SoundStorm后续, 战略投资Hume AI

### 团队4: OpenAI
- 已知: GPT-4o Voice Mode, 但技术细节不公开
- 搜索: "OpenAI" + speech/voice/TTS, 看有无技术博客
- 重点: Voice Mode技术架构(如果有公开信息), 对行业的影响

调研完后,给出大厂语音能力对比,重点分析开源vs闭源策略。
```

---

## Session 6: 学术界 A（亚洲实验室）

```
[粘贴通用前置 Prompt]

现在开始调研以下实验室:

### 实验室1: 李宏毅组 (NTU, 台湾)
- 核心人: Hung-yi Lee
- 已知方向: Spoken LM, Full-Duplex-Bench, TASTE, Align-SLM
- 搜索: arXiv author "Hung-yi Lee" 2025-2026全部论文
- 重点: 2026年最新工作, SLM方向细分, 评测benchmark产出

### 实验室2: 中科大语音实验室 (USTC)
- 关联: 科大讯飞
- 搜索: "University of Science and Technology of China" + speech/TTS/audio
- 核心人: 需要从论文中发现
- 重点: 区分与讯飞合作 vs 独立学术工作, 学生毕业去向

### 实验室3: 西工大 ASLP (NPU)
- 搜索: "Northwestern Polytechnical" + speech/audio
- GitHub: NPU-ASLP
- 重点: 2025-2026是否有重要TTS论文, 声纹/安全方向进展

### 实验室4: 清华语音组 (非VoxCPM的其他组)
- 已知: VoxCPM是面壁/清华NLP
- 搜索: "Tsinghua" + speech synthesis/TTS, 区分不同实验室
- 重点: 清华有几个做语音的组? 各自方向?

调研完后,对比四个实验室的研究方向分化和重叠。
```

---

## Session 7: 学术界 B（更多实验室+国际）

```
[粘贴通用前置 Prompt]

现在开始调研以下实验室:

### 实验室1: 浙江大学 (ZJU)
- 关联: 与阿里CosyVoice团队合作
- 搜索: "Zhejiang University" + speech/TTS/codec/audio
- 重点: 独立于阿里合作的自主工作, codec方向

### 实验室2: 港中文 MMLab + 语音相关组
- 已知工作: DualSpeechLM (AAAI 2026), OmniCharacter
- 搜索: "Chinese University of Hong Kong" + speech/audio/TTS
- 重点: 与蚂蚁合作的完整工作链

### 实验室3: CMU (卡内基梅隆)
- 搜索: "Carnegie Mellon" + speech synthesis/TTS 2025-2026
- 已知: CMU有 Language Technologies Institute
- 重点: 北美顶校在TTS方向还活跃吗? 做什么?

### 实验室4: 其他值得关注的组
- Haizhou Li (新加坡国立/深圳)
- Xu Tan (离开微软后去了哪里?)
- 韩国 KAIST/SNU 语音组
- 搜索各自最新工作

调研完后,画出全球学术版图:谁在做什么,谁退出了,新兴力量在哪。
```

---

## Session 8: 会议/竞赛/趋势汇总

```
你现在是 TTS 行业研究分析师。这是最后一个调研 session,需要:

## Part 1: 会议热点统计

用 Playwright 搜索以下会议的 2025-2026 accepted papers / program:

1. **ICASSP 2026**: 搜索 speech synthesis / TTS 相关 session,统计 topic 分布
2. **Interspeech 2025/2026**: 搜索 special sessions, 看新增了什么方向
3. **NeurIPS 2025 + ICLR 2026**: 搜索 audio/speech 相关 papers,统计主题
4. **ACL/EMNLP 2025-2026**: spoken language / speech 相关论文

输出: 各会议 TTS/speech 方向的 topic 频率排名 (前10)

## Part 2: 竞赛方向变化

1. **Blizzard Challenge 2024/2025**: 任务设计变化, 参赛系统特点
2. **VoiceMOS Challenge 2025**: track 设计, 新增维度
3. **Interspeech 2026 Challenges**: 小米主办的 Audio Encoder Challenge 等
4. **其他新竞赛**: 有无新的 TTS/speech 评测竞赛出现

## Part 3: 投资与产业信号

搜索 2025-2026 语音AI领域:
1. 融资事件 (金额/轮次/投资方)
2. 收购事件
3. 团队变动 (核心人跳槽/创业)
4. 开源项目影响力排名 (GitHub star)

## Part 4: 汇总分析

综合前 7 个 session 的调研结果(我会提供摘要),输出:

1. **技术路线图**: 各流派+代表团队 (一张大表)
2. **能力矩阵**: 30+团队 × 10个维度 (tokenizer/生成/RL/流式/可控/评估/安全/开源/数据/产品)
3. **趋势判断**: 上升/稳定/下降 方向
4. **空白地带**: 无人做或做得少的方向
5. **2024→2025→2026 关键时间线**: 行业 milestone

注意: 用 Playwright 搜索,不用 WebFetch/WebSearch。
```

---

## 补充: 调研质量自检 Prompt（每个 session 结束时追问）

```
请对照以下清单自检你的调研质量:

- [ ] 每个团队的 GitHub/HuggingFace org 都实际访问了?
- [ ] 核心人的名字都在 arXiv 上搜过了?
- [ ] 搜索关键词覆盖了 speech/audio/voice/TTS/codec/tokenizer/vocoder/ASR/singing/dialogue?
- [ ] 找到了他们的"非TTS标题"但相关的工作? (如 Borderless Long Speech ≠ 标题含TTS)
- [ ] 论文时间线是否完整 (不只列top 5)?
- [ ] 有无遗漏的开源项目?
- [ ] 确认了竞赛参与情况?

如果有遗漏,请补充搜索后更新报告。
```

---

## 最终输出位置

所有 session 结果汇总后,保存为:
- `docs/2026-TTS行业全景报告.md` — 最终汇总报告
- `docs/2026-TTS行业全景报告-原始数据/` — 各 session 的原始调研结果(可选)
