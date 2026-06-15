# Speech LLM / Omni / 全双工对话 行业调研方法论与执行计划

> 2026-06-08 | 目标: 系统性摸底 2024-2026 Speech LLM / Omni / 全双工对话 全行业动向

---

## 一、调研方法论（5 层搜索法）

### 核心经验

1. **不能只搜"[公司名] Speech LLM"** — 很多团队的论文标题用 "omni"、"multimodal"、"voice agent"、"spoken dialogue" 等不同术语
2. **必须找到团队内部名称** — 对外品牌 ≠ 研究团队名（如阿里的"通义语音(FunAudioLLM)"和"Qwen语音(Qwen-Audio/Qwen3.5-Omni)"是两个团队）
3. **GitHub/HuggingFace org 是最全的入口** — 比搜论文更不易遗漏
4. **核心人的作者网络能串起所有工作** — 从一篇已知论文找 lead，再搜 lead 全部发表
5. **有些团队分层发布** — 理解模块、生成模块、对话模块各一篇论文，不看全就低估能力
6. **同一公司可能有多个独立团队** — 人员零重叠但领域相同
7. **Speech LLM 与 TTS 有重叠但不同** — 关注的是端到端语音理解+生成+对话能力，不只是语音合成

### 5 层搜索法

```
Layer 1: 组织搜索
  → GitHub org (搜公司名/团队名)
  → HuggingFace org
  → 官方技术博客

Layer 2: 核心人搜索
  → 找到团队 lead (通过一篇已知论文的作者列表)
  → 用 lead 的名字搜 arXiv (author:xxx)
  → 从 lead 的合作者网络扩展

Layer 3: arXiv affiliation 搜索
  → 搜 "affiliation:XXX" 或论文中标注的机构
  → 搜索关键词: speech language model, audio language model, omni model, full-duplex, spoken dialogue, voice agent, speech-to-speech, multimodal speech, end-to-end speech, duplex conversation, real-time voice, voice chat, audio LLM, speech understanding, streaming speech interaction
  → 不限标题，看 affiliation 字段

Layer 4: 产品/竞赛反推
  → 看他们参加了什么竞赛 (VoiceBench, AudioBench, Dynamic-SUPERB, AIShell, SUPERB)
  → 看他们的产品技术栈用了什么
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
- 语音编码器 (Speech Encoder):
- LLM 骨干 (LLM Backbone):
- 语音解码器 (Speech Decoder):
- 对话策略 (Turn-taking / Full-duplex):
- 训练数据规模:
- 推理延迟:
- 多语言支持:
- 情感/副语言 (Paralinguistics):
- 其他:

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

### 国内大厂 (8)
| 团队 | 已知入口 | 代表工作 |
|------|----------|----------|
| 阿里通义 (FunAudioLLM / Qwen-Audio) | FunAudioLLM, Qwen | Qwen2.5-Omni, Qwen3.5-Omni, FunAudioChat |
| 字节豆包 | ByteDance | Seed-ASR, 豆包语音对话 |
| 智谱 | THUDM | GLM-4-Voice |
| 阶跃星辰 | StepFun | Step-Audio, Step-Audio 2.5 |
| 百度 | PaddlePaddle | 文心大模型语音能力 |
| 讯飞 | iFlytek | 星火语音交互 |
| 腾讯 | Tencent | 混元语音 |
| 小米 | MiSpeech | MiSpeech 团队 |

### 国内新锐 (4)
| 团队 | 已知入口 | 代表工作 |
|------|----------|----------|
| 月之暗面 | Moonshot AI | Kimi-Audio, MoonVoice |
| MiniMax | MiniMax | MiniMax-Speech |
| 面壁智能 | ModelBest | VoxCPM |
| 蚂蚁集团 | Ant Group | Ming-UniAudio |

### 国际创业公司 (12)
| 团队 | 已知入口 | 代表工作 |
|------|----------|----------|
| Kyutai | kyutai-labs | Moshi |
| Sesame | sesame | CSM (Conversational Speech Model) |
| Thinking Machine | — | 待调研 |
| Cartesia | cartesia-ai | Sonic (SSM架构, 40ms延迟) |
| Fixie AI | fixie-ai | Ultravox |
| Hume AI | hume-ai | EVI (Empathic Voice Interface) |
| ElevenLabs | elevenlabs | Conversational AI |
| Deepgram | deepgram | Nova, Aura |
| AssemblyAI | assemblyai | Universal 语音理解 |
| Play.ht | playht | PlayDialog |
| Resemble AI | resemble-ai | 语音克隆/安全 |
| LiveKit / Pipecat | livekit, pipecat-ai | 开源 voice agent 框架 |
| Boson AI | — | 待调研 |

### 国际大厂 (5)
| 团队 | 已知入口 | 代表工作 |
|------|----------|----------|
| OpenAI | openai | GPT-4o, Realtime API |
| Google | google | Gemini Live, Project Astra |
| Meta | facebookresearch | Spirit-LM, Seamless |
| Microsoft | microsoft | Azure Speech, VALL-E |
| Apple | apple | Apple Intelligence 语音 |

### 学术实验室 (6)
| 团队 | 已知入口 | 代表工作 |
|------|----------|----------|
| CMU | — | Ichigo, LLaMA-Omni |
| X-LANCE (上海交大) | — | WavSLM, DualSpeechLM |
| 港中深 | — | MaskGCT 团队, OpenOmni |
| THU (清华) | — | SpeechGPT |
| CUHK (港中文) | — | SALMONN |
| Kyoto / NII | — | 日本语音 LM |

---

## 三、执行计划

### Session 结构

| Session | 描述 | 团队 |
|---------|------|------|
| 1 | 国内大厂 A | 阿里通义 + 字节豆包 + 智谱 + 阶跃 |
| 2 | 国内大厂 B | 百度 + 讯飞 + 腾讯 + 小米 |
| 3 | 国内新锐 | 月之暗面 + MiniMax + 面壁 + 蚂蚁 |
| 4 | 国际创业 A | Kyutai + Sesame + Thinking Machine + Cartesia |
| 5 | 国际创业 B | Fixie + Hume AI + ElevenLabs + Deepgram |
| 6 | 国际创业 C | AssemblyAI + Play.ht + Resemble AI + LiveKit/Pipecat + Boson AI |
| 7 | 国际大厂 | OpenAI + Google + Meta + Microsoft + Apple |
| 8 | 学术 A | CMU + X-LANCE + 港中深 |
| 9 | 学术 B | THU + CUHK + Kyoto/NII |
| **10** | **会议/竞赛/趋势汇总** | **跨 session 综合** |

### 执行策略

- 每轮并行 2 个 session agent
- 执行顺序: (1,2) → (3,4) → (5,6) → (7,8) → (9) → (10 汇总)
- Session 10 依赖前 9 个 session 数据

### 汇总输出

所有 session 完成后，最终输出:

```
docs/reports/2026-SpeechLLM行业全景报告.md
├── 一、技术路线图 (Speech LLM 架构流派 + 关键技术层分歧)
├── 二、团队能力矩阵 (35 个团队 × 8 个能力维度)
├── 三、技术趋势判断 (上升/稳定/下降)
├── 四、空白地带 (无人做或做得少的方向)
├── 五、时间线 (关键节点)
├── 六、投资与市场
├── 七、竞赛与评估演进
├── 八、关键人才流动
├── 九、开源策略分类
└── 十、个人规划建议
```

---

## 四、调研质量自检清单

- [ ] GitHub/HuggingFace org 都访问了？有无遗漏的 repo？
- [ ] 核心人的 arXiv 主页都翻了？不止看最近 5 篇？
- [ ] 搜索关键词覆盖了所有相关术语？（speech LLM, omni, full-duplex, voice agent 等）
- [ ] 找到了他们的"非核心但相关"工作？
- [ ] 画出了技术代际演进线？
- [ ] 确认了开源情况和影响力指标 (star/download)？
- [ ] 找到了他们参加的竞赛和获得的名次？
- [ ] 从招聘 JD 推测了未公开方向？
- [ ] 区分了 Speech LLM（端到端语音对话）和 TTS（纯合成）的工作？
