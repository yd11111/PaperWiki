---
title: "instruct TTS 成品数据格式规范 v0.1"
created: 2026-07-28
tags: [research, tts, instruct, annotation, schema, spec]
---

# instruct TTS 成品数据格式规范 v0.1

> 配套 [[instructTTS标注方案设计]] / [[instructTTS标注pipeline与算子]]。定义标注管线**每条成品样本**的字段结构。
> 分区:`labels`(第一阶段结构化)/ `labels_phase2`(第二期预留)/ `nl`(第二阶段扩写)/ `word_prosody`(词级韵律轨)/ `annotation_meta`(溯源+交叉验证)/ `context` / `quality`。

## 1. 规范 JSON(单条样本)

```jsonc
{
  "id": "comfort_014_turn3",
  "audio_path": "segments/comfort_014_turn3.wav",
  "duration_s": 4.6,
  "text": "别担心，一切都会好起来的，我一直在你身边。",

  "labels": {
    "gender": "female",
    "age": "young_adult",
    "prosody": {
      "pitch_level": "low",
      "pitch_range": "narrow",
      "pitch_tone": "level_low",
      "energy": "soft",
      "rate": "slow"
    },
    "emotion": {
      "l1_base": "neutral",
      "is_expressive": true,
      "l2_fine": ["安抚", "温柔"],
      "intensity": "mid",
      "distribution": null
    },
    "register": "自然对话",
    "environment": "clean",
    "text_challenges": []
  },

  "labels_phase2": {
    "intent": null,
    "persona": null,
    "accent": null,
    "paralinguistics": null
  },

  "nl": {
    "l3_emotion_desc": "以温柔轻缓的语气安慰对方，语气里带着关切和笃定",
    "global_desc": "一个年轻女性，用轻柔缓慢的语气温柔地安慰、宽慰对方，声音低而稳，充满关切",
    "instructions": [
      "温柔地安慰她，语速放慢，声音轻一点",
      "用充满关切的语气宽慰对方"
    ]
  },

  "word_prosody": [
    {"word": "别担心", "start": 0.10, "end": 0.72, "duration": 0.62, "boundary": "b3", "energy": 0.41, "pitch": -0.18, "tone": "falling"},
    {"word": "一切都会好起来的", "start": 0.95, "end": 2.30, "duration": 1.35, "boundary": "b3", "energy": 0.45, "pitch": -0.10, "tone": "level_low"}
  ],

  "annotation_meta": {
    "gender": {"tool": "wavlm-gender", "confidence": 0.98},
    "age": {"tool": "age-clf", "confidence": 0.81},
    "prosody": {"tool": "WordVoice-5A", "align_ok": true},
    "emotion_l1": {"tool": "SER:emotion2vec+", "confidence": 0.62},
    "emotion_l2": {"tool": "omni-llm", "input": "audio+text+l1"},
    "register": {"tool_a": "text-llm", "tool_b": "omni-audio"},
    "cross_validation": {
      "emotion": {"ser_l1": "neutral", "omni_rollup": "neutral", "agree": true, "resolution": "accept"},
      "register": {"text_pred": "自然对话", "audio_pred": "自然对话", "agree": true, "resolution": "accept"}
    }
  },

  "context": {
    "source_id": "comfort_014",
    "segment_start_s": 12.4,
    "segment_end_s": 17.0,
    "preceding_text": "对方：我真的撑不下去了……",
    "scene_summary": "朋友倾诉低谷，说话人出言安慰",
    "speaker_role": "friend"
  },

  "quality": {
    "asr_cer": 0.0,
    "snr_db": 28.5,
    "dnsmos": 3.6,
    "silence_ratio": 0.08,
    "pass": true
  }
}
```
> `//` 注释仅说明用,实际 JSON 去掉。

## 2. 字段规范

### 2.1 顶层

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| id | string | ✓ | 唯一标识 `{source}_turn{N}` |
| audio_path | string | ✓ | 切句后音频相对路径 |
| duration_s | float | ✓ | 时长,质量下限 0.5s |
| text | string | ✓ | ASR 转写 |

### 2.2 `labels`(第一阶段 · 结构化)

| 字段 | 类型 | 取值 | 备注 |
|---|---|---|---|
| gender | string | 随分类器类别集(如 male/female/unknown)| — |
| age | string | 随年龄模型类别集 | — |
| prosody.pitch_level | string | 5 档 very_low/low/mid/high/very_high | **性别内归一** |
| prosody.pitch_range | string | 3 档 narrow/mid/wide | — |
| prosody.pitch_tone | string | 7 类 flat/falling/rising/varied/dramatic/rising_falling/level_low | WordVoice-5A tone |
| prosody.energy | string | 3 档 soft/normal/loud | — |
| prosody.rate | string | 5 档 very_slow/slow/normal/fast/very_fast | — |
| emotion.l1_base | string | 随 SER 模型类别集 | 基础类锚点 |
| emotion.is_expressive | bool | — | 中性门,false→不进 L2/L3 |
| emotion.l2_fine | string[] | ≤3,受控增长清单 | 含社交态度类(安抚/鼓励…)|
| emotion.intensity | string | low/mid/high | 序数 |
| emotion.distribution | object\|null | 7 维软分布(可选)| 混合情感 |
| register | string | 一级 7 类 | 二级先不启用 |
| environment | string | clean/noisy/music | 兼过滤 |
| text_challenges | string[] | code_switch/numbers/units/… | 命中才填 |

### 2.3 `labels_phase2`(第二期 · 预留,现 null)

| 字段 | 类型 | 说明 |
|---|---|---|
| intent | string\|null | 纯言语行为(命令/请求/质问);安慰类由 emotion 承载 |
| persona | string\|null | 片段级 bottom-up 派生描述 |
| accent | string\|null | 口音/方言(无好开源,后期)|
| paralinguistics | array\|null | 副语言事件,带时间戳(span 级)|

### 2.4 `nl`(第二阶段 · 自然语言扩写)

| 字段 | 类型 | 说明 |
|---|---|---|
| l3_emotion_desc | string | L3 情感自由描述,以 L1/L2 为事实锚 |
| global_desc | string | 整体 NL 控制描述(≤200 字),训练输入信号 |
| instructions | string[] | 四阶段 prompt programming 扩出的多样指令变体 |

### 2.5 `word_prosody`(词级韵律轨,WordVoice-5A 原始)

数组,每元素:`word / start / end / duration / boundary(b0-b4) / energy / pitch / tone`。句级 `prosody` 是其聚合,保留原始轨以支持后期 span 级。

### 2.6 `annotation_meta`(溯源 + 交叉验证)

- 每个语义字段记 `{tool, confidence}` 或 `{tool_a, tool_b}`;
- `cross_validation.{emotion,register}`:记双标结果 `{两标值, agree, resolution}`,resolution ∈ accept/human/weighted/drop。

### 2.7 `context` / `quality`

- context:`source_id / segment_start_s / segment_end_s / preceding_text / scene_summary / speaker_role`(几十秒片段上下文,供意图/人设二期用);
- quality:`asr_cer / snr_db / dnsmos / silence_ratio / pass`(质量门)。

## 3. 校验规则(gate)

1. `emotion.l2_fine` ⊆ 受控清单(非清单词拒收,走人工审批新增);
2. **roll-up**:`l2_fine` 必须能归到 `l1_base` 主类,冲突→ `cross_validation` 标 disagree;
3. `register` ∈ 一级 7 类;命中【断句破碎】【其他无效】→ 整条剔除;
4. `nl.*` 不得与 `labels` 矛盾(L3 以 L1/L2 为锚);
5. `quality.pass=false` → 不进训练集。
