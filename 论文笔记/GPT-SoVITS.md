---
type: paper
tier: deep
title: "GPT-SoVITS: Few-Shot Voice Conversion and Text-to-Speech"
source: ""
authors: [RVC-Boss community]
year: 2024
venue: "GitHub"
tags: [TTS, zero-shot, few-shot, GPT, VITS, voice-conversion, open-source, community]
concepts: ["[[SpeechTokenizer]]", "[[LLM-basedTTS]]"]
models: []
tasks: ["[[Zero-shotSpeechSynthesis]]"]
datasets: []
kb_context_sources: 0
status: draft
created: 2026-06-10
updated: 2026-06-10
---

## 速查

> [!summary] 速查
> - **一句话**: 开源社区驱动的两阶段 TTS 系统:GPT 自回归预测 HuBERT semantic tokens + VITS 解码为波形,以极低门槛实现少样本语音克隆
> - **路线**: Text (phoneme) + BERT → GPT (AR, 12-layer Transformer) → semantic tokens (HuBERT VQ, 1024 codebook) → MRTE + VITS (posterior encoder + flow + HiFi-GAN decoder) → Waveform; v3/v4 引入 CFM (DiT) 替代 VITS decoder
> - **指标**: 无标准化 benchmark 评估 (社区项目); RTF 0.028 (4060Ti) / 0.014 (4090)
> - **可借鉴**: (1) BERT feature 注入 GPT 输入增强语义理解; (2) MRTE (Multi-Reference Timbre Encoder) 实现多参考音频融合; (3) 完整的 WebUI + 数据处理 pipeline 降低使用门槛; (4) v3 引入 F5-TTS 的 DiT CFM 架构实现质量跃升
> - **局限**: (1) 无正式论文,设计选择缺乏消融验证; (2) 依赖 Chinese-HuBERT 提取 semantic tokens,非中文场景表现较弱; (3) 训练/推理参数散落各处,调参困难

## 核心方法

GPT-SoVITS 是一个社区开源项目,没有正式论文,以下分析完全基于代码。

### 整体架构 (v2)

系统分为两个阶段:

**Stage 1 (s1_train.py): Text-to-Semantic GPT**
- 输入: phoneme IDs + BERT feature (chinese-roberta-wwm-ext-large)
- 输出: semantic token IDs (HuBERT VQ codebook, 1024 entries)
- 架构: 12-layer Transformer Encoder (非 decoder-only), 512 dim, 8 heads
- 训练: 交叉熵 loss + DPO (reference-free, beta=0.2)

**Stage 2 (s2_train.py): Semantic-to-Waveform VITS**
- 输入: semantic tokens (quantized SSL features) + phoneme text + reference audio
- 输出: 波形
- 架构: 标准 VITS (TextEncoder + PosteriorEncoder + ResidualCouplingFlow + HiFi-GAN Generator)
- 关键改进: MRTE (Multi-Reference Timbre Encoder) 融合 SSL 和 text 信息 + ReferenceEncoder 提取说话人风格

### 版本演进

| 版本 | 核心变化 | 代码位置 |
|-----|---------|---------|
| v1 | 基础 GPT + VITS | `SynthesizerTrn` |
| v2 | 改进 ReferenceEncoder (704-dim spec), 更大预训练 | `SynthesizerTrn` (version="v2") |
| v2Pro/v2ProPlus | 添加 SV embedding (ERes2Net, 20480-dim), 更多判别器 | `SynthesizerTrn` (is_v2pro) |
| v3 | 用 CFM (DiT) 替代 VITS decoder | `SynthesizerTrnV3` |
| v4 | v3 + 更高采样率 (32kHz) | `SynthesizerTrnV3` (version="v4") |

## 代码级分析

> [!info] 代码来源
> - 仓库: https://github.com/RVC-Boss/GPT-SoVITS
> - commit: 08d627c3338173c3229286d8787060d6559fe0f8
> - 分析日期: 2026-06-10

### 架构验证

#### Stage 1: Text2SemanticDecoder (GPT)

核心类 `Text2SemanticDecoder` (`AR/models/t2s_model.py:260`):

```
phoneme_ids → ar_text_embedding (TokenEmbedding) → ar_text_position (SinePosEmb)
                + bert_proj(bert_feature) ← BERT 特征注入
                
semantic_ids → ar_audio_embedding (TokenEmbedding) → ar_audio_position (SinePosEmb)

[text_pos, audio_pos] → TransformerEncoder (12 layers, 512 dim, 8 heads)
                       → ar_predict_layer (Linear → 1025) → CE loss
```

关键设计:
- **BERT feature 注入** (`t2s_model.py:357`): `x = x + self.bert_proj(bert_feature.transpose(1,2))`,将 1024-dim BERT embedding 通过线性层投影到 512-dim 后直接加到 phoneme embedding 上
- **Attention mask** (`t2s_model.py:377-392`): text 部分全双向,audio 部分因果(上三角 mask),text→audio 可见但 audio→text 不可见
- **DPO 训练** (`t2s_model.py:408-448`): 通过 `make_reject_y` 构造负样本(打乱 semantic tokens 顺序),reference-free DPO (beta=0.2)
- **推理优化** (`t2s_model.py:73-222`): `T2SBlock`/`T2STransformer` 是 torch.jit.script 优化版本,手动实现 KV-cache,prompt 一次性处理 + 逐 token 解码

#### Stage 2: SynthesizerTrn (VITS)

核心类 `SynthesizerTrn` (`module/models.py:829`):

```
semantic_tokens → ssl_proj (Conv1d stride=2 downsample if 25Hz)
               → ResidualVectorQuantizer (1 codebook, 1024 bins)
               → quantized (768-dim)
               → TextEncoder (ssl_proj → encoder_ssl → MRTE → encoder2 → proj)
               → m_p, logs_p (prior distribution)

reference_audio → spec → ReferenceEncoder (6-layer CNN + GRU) → ge (speaker embedding, 512-dim)

training:
  spec → PosteriorEncoder → z, m_q, logs_q
  z → ResidualCouplingBlock → z_p
  z_slice → Generator (HiFi-GAN) → waveform
  Loss: recon + KL + discriminator + feature matching

inference:
  m_p + noise * exp(logs_p) → z_p
  z_p → flow.reverse() → z
  z → Generator → waveform
```

关键设计:
- **MRTE** (`module/mrte_model.py`): Multi-Reference Timbre Encoder,将 SSL 特征和 text 特征通过 cross-attention 融合,输出条件特征给后续 encoder
- **RVQ 仅 1 层** (`models.py:925`): `ResidualVectorQuantizer(dimension=768, n_q=1, bins=1024)`,只使用 1 个 codebook,与 EnCodec/SoundStream 的 8-32 层 RVQ 形成鲜明对比
- **25Hz semantic rate** (`models.py:921-922`): 如果 semantic_frame_rate="25hz",用 stride=2 的 Conv1d 下采样 HuBERT 特征(原始 50Hz)
- **v2Pro SV embedding** (`models.py:930-932`): 引入 ERes2Net speaker verification embedding (20480-dim→512-dim),与 ReferenceEncoder 的 ge 相加

#### Stage 2 v3: SynthesizerTrnV3 (CFM)

核心变化 (`module/models.py:1215`):

```
quantized → TextEncoder → enc_p 输出 → bridge (Conv1d) 
         → interpolate(1.875x) → WNEncoder → condition feature (512-dim)
         → CFM (DiT, dim=1024, depth=22, heads=16) → mel (100-dim)
         → BigVGAN / HiFi-GAN → waveform
```

- **去掉了 VITS 的 decoder/posterior/flow** (`models.py:1272-1276`): v3 注释掉了这些模块
- **DiT 配置** (`models.py:1294`): `DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)`,来自 F5-TTS 的实现
- **CFM 训练** (`models.py:1174-1207`): 标准 flow matching + 30% 概率使用 consistency distillation (两步预测取平均作为 target velocity)
- **Mel 预测** (`models.py:1291`): 输出 100-dim mel (24kHz, hop=256, win=1024); v4 改为 100-dim mel (32kHz, hop=320, win=1280)

### 论文未写的实现细节 (代码独有发现)

1. **GPT 的 semantic token 来源** (`prepare_datasets/2-get-hubert-wav32k.py`, `prepare_datasets/3-get-semantic.py`): 使用 Chinese-HuBERT-Base 提取特征,然后通过预训练的 SynthesizerTrn.extract_latent() 做 RVQ 编码得到 semantic codes。token rate 为 25Hz (50Hz 下采样)。

2. **GPT 推理的早停和重复惩罚** (`t2s_model.py:713-715,903`): 前 11 个 token 不允许 EOS (至少生成 0.44 秒); 使用 `repetition_penalty=1.35` 抑制重复; 最大生成 1500 个 token。

3. **流式推理** (`t2s_model.py:931-948`): GPT 支持 streaming 模式,使用 `mute_emb_sim_matrix` 在静音位置切分 chunk,阈值 `chunk_split_threshold=0.3`,每 chunk 至少 `chunk_length=24` 个 token。

4. **多参考音频支持** (`models.py:1011-1015`): `decode` 方法支持传入 refer 列表,对每个参考音频分别提取 ge,然后取平均作为最终 speaker embedding。

5. **v2ProPlus 的判别器扩展** (`models.py:627-630`): v2Pro 使用 7 个周期判别器 (periods=[2,3,5,7,11,17,23]),比标准 VITS 的 5 个 (periods=[2,3,5,7,11]) 多 2 个。

6. **CodePredictor** (`models.py:771-826`): 存在一个未在主训练流程中使用的 `CodePredictor` 类,可以从 VQ 第 1 层预测后续 7 层的 codes (n_q=8),类似 VALL-E 的 NAR 解码。这暗示项目早期可能考虑过多 codebook 方案但最终只用了 1 层。

7. **速度控制** (`models.py:259`): 通过 `F.interpolate(y, size=int(y.shape[-1]/speed)+1, mode="linear")` 在 latent 空间做简单线性插值实现变速。

8. **CUDA graph 加速** (`AR/models/t2s_model_cudagraph.py`): 最新代码加入了 CUDA graph 支持,推理速度提升约 2x。

### 训练 pipeline 拆解

```
数据准备:
  音频 → slicer2.py 切分 → 降噪 (UVR5)
  音频 → funasr/fasterwhisper ASR → 文本标注
  音频 → Chinese-HuBERT → SSL 特征 → SynthesizerTrn.extract_latent → semantic codes
  文本 → G2P → phoneme IDs
  文本 → chinese-roberta-wwm-ext-large → BERT features

Stage 1 训练 (s1_train.py):
  phoneme_ids + BERT + semantic_ids → Text2SemanticDecoder
  Loss: CE + DPO (reference-free)
  Framework: PyTorch Lightning
  
Stage 2 训练 (s2_train.py):
  semantic_codes + spectrogram + phoneme → SynthesizerTrn
  Loss: recon + KL + GAN (discriminator + feature matching)
  freeze_quantizer=True 时冻结 ssl_proj + quantizer
  Framework: 原生 PyTorch DDP
```

### 推理 pipeline 拆解

```
输入文本 + 参考音频 (5s~1min)
  → 参考音频 → HuBERT → SSL feat → RVQ → prompt semantic codes
  → 参考音频 → spec → ReferenceEncoder → ge (speaker style)
  → 输入文本 → G2P → phoneme_ids
  → 输入文本 → BERT → bert_feature
  
  Stage 1 (GPT):
    [phoneme+BERT, prompt_codes] → TransformerEncoder (causal for audio)
    → 自回归生成 semantic codes
    top-k, top-p, temperature, repetition_penalty=1.35
    max 1500 tokens, 前 11 个不允许 EOS
    
  Stage 2 (VITS/CFM):
    v1/v2: semantic_codes → VQ decode → TextEncoder+MRTE → prior → flow.reverse → HiFi-GAN
    v3/v4: semantic_codes → TextEncoder → bridge → WNEncoder → CFM (DiT, 32 steps) → BigVGAN
    
  输出: waveform (v1/v2: 32kHz, v3: 24kHz, v4: 32kHz)
```

### 关键超参数表

| 参数 | 值 | 代码位置 | 备注 |
|------|------|---------|------|
| GPT layers | 12 | config YAML | `n_layer` |
| GPT dim | 512 | `hidden_dim` | embedding + model |
| GPT heads | 8 | `head` | |
| Semantic codebook | 1024 + 1 (EOS) | `vocab_size=1025` | 1 层 RVQ |
| Phoneme vocab | 512 | `phoneme_vocab_size` | |
| BERT dim | 1024 → 512 | `bert_proj` | |
| Semantic rate | 25Hz | `semantic_frame_rate` | 50Hz 下采样 |
| VITS channels | 192 | `inter_channels` | |
| VITS hidden | 192 | `hidden_channels` | |
| ReferenceEncoder | 704-dim spec (v2) | `MelStyleEncoder(704)` | |
| SV embedding (v2Pro) | 20480 → 512 | `sv_emb` | ERes2Net |
| DiT (v3) | dim=1024, depth=22, heads=16 | `CFM.__init__` | F5-TTS |
| Repetition penalty | 1.35 | inference default | |
| Max generation | 1500 tokens | `range(1500)` | ~60s |
| Min tokens before EOS | 11 | hardcoded | ~0.44s |

### 与 CosyVoice 的关键对比

| 维度 | GPT-SoVITS | CosyVoice |
|------|-----------|-----------|
| Tokenizer | HuBERT (自监督) + 1层 RVQ | ASR (监督式) + VQ/FSQ |
| Text encoder | BERT + phoneme embedding | BPE + Conformer/Qwen2 |
| Stage 1 | Transformer Encoder (非 decoder-only) | LLM (decoder-only) |
| Stage 2 | VITS (VAE+flow) / CFM (v3) | OT-CFM |
| Speaker cond | ReferenceEncoder (训练时学习) | x-vector (预训练冻结, v1) / 无 (v2/3) |
| 流式 | GPT 支持, VITS 不支持 | 全链路支持 (v2/3) |
| 训练门槛 | 低 (WebUI, 1min 数据) | 高 (大规模预训练) |
| 设计哲学 | 社区迭代,工程优先 | 学术驱动,科学验证 |

### 复现 checklist (基于代码)

- [ ] 环境依赖: PyTorch, transformers (BERT), pytorch-lightning (s1), peft (LoRA for v3), f5-tts
- [ ] 数据准备: WebUI 提供一站式工具 (切分+降噪+ASR+标注+特征提取)
- [ ] 预训练模型: chinese-hubert-base, chinese-roberta-wwm-ext-large, 预训练 GPT+SoVITS ckpt
- [ ] 训练命令: s1_train.py (GPT) → s2_train.py (VITS) 或 s2_train_v3.py (CFM)
- [ ] 推理命令: `TTS_infer_pack/TTS.py` 或 WebUI
- [ ] 已知坑: (1) 依赖 Chinese-HuBERT,英文效果明显弱于中文; (2) v3 LoRA 微调需要额外安装 peft; (3) BERT 模型较大 (1.3GB)

### 代码质量与可复现性评估

- **工程质量**: 3/5 - 功能丰富但代码风格不统一,大量全局变量和 hardcode path,s1/s2 使用不同训练框架
- **文档完善度**: 4/5 - 有详细的中文文档和 WebUI,对非开发者友好;但代码层面注释稀少
- **社区活跃度**: 5/5 - GitHub 40k+ stars,极其活跃,版本迭代快 (v1→v2→v2Pro→v3→v4)
- **复现难度**: 2/5 - 提供了完整的 WebUI 和预训练模型,1 分钟数据即可微调;是 TTS 领域复现门槛最低的系统之一
