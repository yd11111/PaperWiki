# TTS 行业调研方法论与执行计划

> 2026-06-04 | 目标: 系统性摸底 2024-2026 TTS 全行业动向

---

## 一、调研方法论（5层搜索法）

### 核心经验

1. **不能只搜"[公司名] TTS"** — 很多团队的论文标题不含TTS(如"Borderless Long Speech Synthesis")
2. **必须找到团队内部名称** — 对外品牌≠研究团队名(如小米语音叫MiSpeech/Horizon Team)
3. **GitHub/HuggingFace org 是最全的入口** — 比搜论文更不易遗漏
4. **核心人的作者网络能串起所有工作** — 从一篇已知论文找lead，再搜lead全部发表
5. **有些团队分层发布而非发系统论文** — encoder→tokenizer→生成器各一篇,不看全就低估其能力
6. **同一公司可能有多个独立语音团队** — 如阿里的"通义语音(FunAudioLLM)"和"Qwen语音(Qwen-Audio/Qwen3-TTS)"是两个团队

### 5层搜索法

```
Layer 1: 组织搜索
  → GitHub org (xiaomi-research, AlibabaResearch, bytedance, etc.)
  → HuggingFace org (mispeech, FunAudioLLM, etc.)
  → 官方技术博客

Layer 2: 核心人搜索
  → 找到团队 lead (通过一篇已知论文的作者列表)
  → 用 lead 的名字搜 arXiv (author:xxx)
  → 从 lead 的合作者网络扩展

Layer 3: arXiv affiliation 搜索
  → 搜 "affiliation:Xiaomi" 或论文中标注的机构
  → 不只搜 "TTS"，要搜 "speech" "audio" "voice" "codec" "tokenizer"

Layer 4: 产品/竞赛反推
  → 看他们参加了什么竞赛 (Blizzard/VoiceMOS/Interspeech Challenge)
  → 看他们的产品技术栈用了什么 (小爱同学/豆包/通义千问语音版)
  → 招聘 JD 暴露研究方向

Layer 5: 引用网络
  → 从已知论文的 references 找同团队的早期工作
  → 从 cited by 找后续迭代
```

### 调研模板（每个团队必须填完）

```markdown
## [团队名]

### 基本信息
- 公司/机构:
- 团队名/实验室名:
- GitHub org:
- HuggingFace org:
- 核心人物 (2-3人):
- 产品线:

### 论文时间线 (2024-2026)
| 时间 | 论文 | 会议/期刊 | 方向 |

### 技术栈全景
- Tokenizer/Codec:
- 生成模型:
- Post-training:
- 流式/效率:
- 评估:
- 其他(VC/SVS/Edit/Safety):

### 架构演进
[用箭头画出代际关系]

### 独特技术赌注
[他们押注了什么别人没押的]

### 开源情况
[哪些开源了, star数, 下载量]

### 判断
- 优势:
- 短板:
- 下一步方向推测:
```

---

## 二、调研对象清单

### A. 国内工业界 (14家/15个团队)

| # | 团队 | 已知入口 | 调研深度 |
|---|------|----------|----------|
| 1a | **通义语音/FunAudioLLM** (阿里) | GitHub: FunAudioLLM; HF: FunAudioLLM | 需更新(CosyVoice 3后有无新工作) |
| 1b | **Qwen语音** (阿里) | GitHub: QwenLM; HF: Qwen | 需独立调研: Qwen-Audio, Qwen3-TTS, 与通义的关系和分工 |
| 2 | 字节/Seed-TTS | 无公开org, 搜作者 | 需补: 2026新论文, 产品侧(豆包语音) |
| 3 | 阶跃星辰/StepAudio | GitHub: stepfun-ai | 需补: StepAudio 2.5后续 |
| 4 | 智谱/GLM-4-Voice | GitHub: THUDM | 需补: 2026有无GLM-5-Voice |
| 5 | VoxCPM/面壁 | GitHub: OpenBMB | 需补: VoxCPM后续迭代 |
| 6 | MiniMax | 搜作者/产品(海螺AI) | 需补: MiniMax-Speech后续 |
| 7 | **小红书/FireRedTTS** | GitHub: RedNoteAI? 搜作者 | 需深挖: SoCodec/PodAgent详情 |
| 8 | **科大讯飞** | 搜作者(中科大合作) | 需深挖: 完整论文列表 |
| 9 | **蚂蚁/Ming** | 搜作者 | 需深挖: Ming-UniAudio完整能力 |
| 10 | **京东** | GitHub: jd-ai? 搜作者 | 需深挖: JoyVoice/JoyHallo系列 |
| 11 | **天工/昆仑** | 搜作者 | 需深挖: MoE-TTS论文详情 |
| 12 | **Soul** | GitHub: SoulApp? | 需深挖: SoulX-Singer/Duplug |
| 13 | **小米/MiSpeech** | GitHub: xiaomi-research; HF: mispeech | ✅ 已完成(可作为模板) |
| 14 | **米哈游** | 搜作者(高校合作为主) | 需深挖: 游戏配音AI |

> **注意**: 阿里有两个独立语音团队(同属通义实验室,但人员几乎零重叠):
> - **通义语音 (FunAudioLLM)**: VP叶杰平, 技术Lead张士良(Shiliang Zhang), CosyVoice/SenseVoice/FunASR, 偏语音专用工具链。前负责人鄢志杰2025.02离职(加入腾讯)
> - **Qwen语音**: CTO周靖人统管, Qwen-Audio/Qwen3-TTS/Qwen3-Omni, 偏LLM多模态语音能力。前Tech Lead林俊旸2026.03离职
> - **武执政(Zhi-Zheng Wu)**: 港中文(深圳)副教授,非阿里员工,是外部学术合作者(NaturalSpeech/Amphion)
> 两者技术路线不同(tokenizer/数据/LLM backbone均独立),需分别调研并对比。

### B. 国际工业界 (10家)

| # | 团队 | 已知入口 | 调研深度 |
|---|------|----------|----------|
| 1 | ElevenLabs | 博客+产品页 | 需补: 技术论文(如果有) |
| 2 | Cartesia | GitHub: cartesia-ai | 需深挖: SSM架构论文详情 |
| 3 | Hume AI | GitHub: hume-ai? | 需深挖: EVI技术细节 |
| 4 | Sesame | GitHub: sesame-ai; HF: sesame | 需深挖: CSM后续迭代 |
| 5 | **Voxtral/Mistral** | HF: mistralai | 需深挖: Voxtral TTS 4B技术 |
| 6 | **Fish Audio** | GitHub: fishaudio | 需深挖: Fish-Speech系列演进 |
| 7 | **Meta** | 收购PlayHT后做了什么; Seamless系列 | 需深挖 |
| 8 | **Google** | SoundStorm/AudioPaLM/Gemini语音 | 需深挖 |
| 9 | **Apple** | WWDC语音功能反推 | 公开信息可能少 |
| 10 | **OpenAI** | Voice Mode技术细节(如果有论文) | 公开信息少 |

### C. 学术实验室 (8个)

| # | 实验室 | 核心人 | 调研深度 |
|---|--------|--------|----------|
| 1 | X-LANCE (上交) | 余凯, 陈谢, 郭依蔚 | ✅ 已完成 |
| 2 | 李宏毅组 (NTU) | Hung-yi Lee | 需补: 2026最新 |
| 3 | 中科大语音 | — | 需深挖: 除讯飞合作外的独立工作 |
| 4 | 西工大ASLP | — | 需深挖: 核心人+最新方向 |
| 5 | 港中文MMLab | — | 需补: DualSpeechLM后续 |
| 6 | **清华 (非VoxCPM)** | — | 需查: 清华其他语音组 |
| 7 | **浙大** | — | 需深挖: 与阿里合作之外的独立工作 |
| 8 | **CMU/Stanford/MIT** | — | 需查: 北美顶校语音方向 |

### D. 会议/竞赛/投资 (补充维度)

| # | 来源 | 获取什么 |
|---|------|----------|
| 1 | ICASSP 2026 accepted papers | 热点topic统计 |
| 2 | Interspeech 2026 特别session | 新兴方向 |
| 3 | NeurIPS/ICLR 2026 audio track | ML社区对speech的关注点 |
| 4 | Blizzard Challenge 2025 | 参赛系统+任务设计变化 |
| 5 | VoiceMOS Challenge 2025 | 评估方向 |
| 6 | 投资数据 (Crunchbase/IT桔子) | 钱往哪走 |

---

## 三、执行计划（新 session 用）

### Session 结构建议

每个 session 调研 3-4 个团队（避免上下文过长），按以下流程：

```
Session 1: 国内大厂 A (通义更新 + 字节更新 + 阶跃更新 + 智谱更新)
Session 2: 国内大厂 B (MiniMax + 小红书 + 讯飞 + 蚂蚁)
Session 3: 国内新锐 (京东 + 天工 + Soul + 米哈游)
Session 4: 国际公司 A (ElevenLabs + Cartesia + Sesame + Fish Audio)
Session 5: 国际公司 B (Voxtral + Meta + Google + OpenAI)
Session 6: 学术界 A (李宏毅2026 + 中科大 + 西工大 + 清华)
Session 7: 学术界 B (浙大 + 港中文 + CMU/Stanford + 其他)
Session 8: 会议/竞赛/投资 + 汇总
```

### 每个团队的调研 prompt 模板

```
按以下5层搜索法调研 [团队名] 在 2024-2026 的语音/TTS 全部工作:

Layer 1 组织搜索:
- 访问 GitHub org: [xxx]
- 访问 HuggingFace org: [xxx]
- 搜索官方博客/技术公众号

Layer 2 核心人搜索:
- 已知核心人: [xxx]
- 用 arXiv author search 找他们所有 2024-2026 论文
- 从合作者扩展

Layer 3 arXiv affiliation:
- 搜 "[公司名]" + speech/audio/voice/TTS/codec/tokenizer/synthesis
- 不限标题,看 affiliation 字段

Layer 4 产品/竞赛:
- 他们的语音产品叫什么
- 参加了哪些竞赛/workshop

Layer 5 引用网络:
- 从已知论文的 reference 找早期工作
- 从 cited by 找后续

输出格式: [用上面的调研模板]
```

### 汇总输出

所有 session 完成后，最终输出:

```
docs/reports/2026-TTS行业全景报告.md
├── 一、技术路线图 (各流派+代表团队)
├── 二、团队能力矩阵 (30+团队 × 10个能力维度)
├── 三、技术趋势判断 (上升/稳定/下降)
├── 四、空白地带 (无人做或做得少的方向)
├── 五、时间线 (2024→2025→2026 关键节点)
└── 六、个人规划建议 (基于全景分析)
```

---

## 四、调研质量自检清单

每个团队调研完后对照:

- [ ] GitHub/HuggingFace org 都访问了？有无遗漏的 repo？
- [ ] 核心人的 arXiv 主页都翻了？不止看最近5篇？
- [ ] 搜索关键词覆盖了 speech/audio/voice/TTS/codec/tokenizer/vocoder/ASR？
- [ ] 找到了他们的"非TTS"但相关工作？(如编码器/评估/数据集/安全)
- [ ] 画出了技术代际演进线？
- [ ] 确认了开源情况和影响力指标(star/download)？
- [ ] 找到了他们参加的竞赛和获得的名次？
- [ ] 从招聘 JD 推测了未公开方向？
