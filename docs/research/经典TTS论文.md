# 经典 TTS 论文谱系

> 29 篇定义 TTS 发展脉络的经典论文。按技术纪元排列。
> 更新: 2026-06-10

---

## 第一纪 — 神经 TTS 奠基 (2016-2021)

| # | 论文 | 年份 | 贡献 |
|---|------|------|------|
| 1 | **WaveNet** | 2016 | 自回归逐采样点生成波形，证明神经网络可以合成自然语音 |
| 2 | **Tacotron 2** | 2017 | 定义了 "text → mel → vocoder" 标准 pipeline，统治 TTS 3+ 年 |
| 3 | **FastSpeech 2** | 2020 | 非自回归 + 显式 duration/pitch/energy 预测，让 TTS 快了 100 倍 |
| 4 | [[论文笔记/VITS|VITS]] | 2021 | VAE + normalizing flow + GAN 端到端；GPT-SoVITS 和整个 SVC 生态的地基 |

## 第二纪 — Codec 基础设施 (2021-2022)

| # | 论文 | 年份 | 贡献 |
|---|------|------|------|
| 5 | [[论文笔记/SoundStream|SoundStream]] | 2021 | RVQ neural audio codec — 把语音变成离散 token 序列的第一篇 |
| 6 | [[论文笔记/EnCodec|EnCodec]] | 2022 | Meta 的开源 codec，让所有后续 codec LM 论文有了可复现的基础 |

## 第三纪 — Codec Language Model 开范式 (2023)

| # | 论文 | 年份 | 贡献 |
|---|------|------|------|
| 7 | [[论文笔记/AudioLM|AudioLM]] | 2022 | "用 LM 预测音频 token"的概念起源，VALL-E 显式继承 |
| 8 | [[论文笔记/VALL-E|VALL-E]] | 2023 | **开创 codec language model for TTS**，证明零样本 3 秒克隆可行 |
| 9 | [[论文笔记/Voicebox|Voicebox]] | 2023 | **Flow matching 进入语音生成**的奠基；CosyVoice 系列 CFM 模块的源头 |
| 10 | [[论文笔记/SoundStorm|SoundStorm]] | 2023 | 并行解码 codec tokens — MaskGCT 的直系先驱 |

## 第四纪 — 百花齐放 (2024)

| # | 论文 | 年份 | 贡献 |
|---|------|------|------|
| 11 | [[论文笔记/NaturalSpeech3|NaturalSpeech 3]] | 2024 | Factorized codec 四路分离 + factorized diffusion，微软 TTS 巅峰 (ICML 2024) |
| 12 | [[论文笔记/Seed-TTS|Seed-TTS]] | 2024 | 字节旗舰；定义了 SEED-TTS-Eval — 2025-2026 几乎所有零样本 TTS 的评测标准 |
| 13 | [[论文笔记/MaskGCT|MaskGCT]] | 2024 | NAR masked generative codec transformer，非自回归路线的最强代表 |
| 14 | [[论文笔记/MELLE|MELLE]] | 2024 | 第一个正面挑战 VQ 必要性的 AR TTS — 连续 token 先驱 |
| 15 | [[论文笔记/Moshi|Moshi]] | 2024 | 全双工对话式语音交互先驱，inner monologue + 多流并行 |
| 16 | [[论文笔记/CosyVoice|CosyVoice]] | 2024 | 阿里 TTS 系列开篇，LLM + CFM 两阶段工业范式 |
| 17 | **GPT-SoVITS** | 2024 | GPT 预测 + VITS 解码，开源社区最广泛使用的零样本 TTS |
| 18 | [[论文笔记/E2TTS|F5-TTS]] | 2024 | Voicebox 极简化 (纯 DiT + flow matching)，2025 年 fork 量最大的开源 TTS |

## 第五纪 — 连续 AR 革命 + 工业竞赛 (2025)

| # | 论文 | 年份 | 贡献 |
|---|------|------|------|
| 19 | [[论文笔记/DiTAR|DiTAR]] | 2025 | **连续 AR 的起点** — AR 预测连续 latent + in-context flow matching |
| 20 | [[论文笔记/CosyVoice2|CosyVoice 2]] | 2025 | 有限标量量化 + chunk-aware 流式，工业级流式 TTS 标杆 |
| 21 | [[论文笔记/CosyVoice3|CosyVoice 3]] | 2025 | 多 codebook masked generation + LLM 初始化，NAR 路线的阿里版 |
| 22 | [[论文笔记/IndexTTS2|IndexTTS2]] | 2025 | AAAI 2026，工业级零样本 TTS，开源生态重要一环 |
| 23 | [[论文笔记/VoxCPM|VoxCPM]] | 2025 | Tokenizer-free：端到端 diffusion AR，无需外部 codec |

## 第六纪 — 当前前沿 (2026)

| # | 论文 | 年份 | 贡献 |
|---|------|------|------|
| 24 | [[论文笔记/dots.tts|dots.tts]] | 2026 | 2B 全连续 AR + per-patch FM + SOAR 自纠错，小红书 |
| 25 | [[论文笔记/WavTTS|WavTTS]] | 2026 | 跳过 mel/token 中间表示，直接建模原始波形 latent |
| 26 | [[论文笔记/SemaVoice|SemaVoice]] | 2026 | 语义感知连续 AR，在连续 latent 中保留语义可控性 |
| 27 | [[论文笔记/OmniVoice|OmniVoice]] | 2026 | 单阶段 NAR text-to-multi-codebook + LLM 初始化进 NAR |
| 28 | [[论文笔记/VoxCPM2|VoxCPM2]] | 2026 | concat-projection 融合 + NoPE RALM + AudioVAE V2 |
| 29 | [[论文笔记/Qwen3-TTS|Qwen3-TTS]] | 2026 | 阿里统一 TTS 旗舰，AR LM + flow matching |

---

## 谱系图

```
WaveNet ──→ Tacotron 2 ──→ FastSpeech 2 ──→ VITS ──→ GPT-SoVITS
                                                 └──→ IndexTTS2

SoundStream ──→ EnCodec ──→ AudioLM ──→ VALL-E ──┬→ CosyVoice 1/2
                                                   └→ Seed-TTS

Voicebox ──→ E2 TTS ──→ F5-TTS ──→ CosyVoice 3
         └──────────────────────→ CosyVoice 1/2 (CFM stage)

SoundStorm ──→ MaskGCT ──→ OmniVoice

NaturalSpeech 3 (factorized codec 分支)

MELLE ──→ DiTAR ──→ dots.tts / WavTTS / SemaVoice

VoxCPM ──→ VoxCPM2 (tokenizer-free 分支)

Moshi (全双工对话分支)

Qwen3-TTS (AR LM + FM 工业集成)
```

---

## 按技术路线索引

### 自回归离散 token
VALL-E → Seed-TTS → CosyVoice 1/2 → Qwen3-TTS

### 自回归连续 latent
MELLE → DiTAR → dots.tts / WavTTS / SemaVoice / VoxCPM / VoxCPM2

### 非自回归 (NAR)
FastSpeech 2 → SoundStorm → MaskGCT → OmniVoice / CosyVoice 3

### Flow matching
Voicebox → F5-TTS → CosyVoice (CFM stage) → WavTTS

### 端到端
WaveNet → VITS → GPT-SoVITS → IndexTTS2

### Codec 基础设施
SoundStream → EnCodec

### 全双工/对话
Moshi

### Factorized 路线
NaturalSpeech 3
