# 审阅 Checklist

## 问题分类体系 (10 类)

| type | 含义 | 典型表现 |
|------|------|----------|
| factual-error | 事实/数字错误 | 指标名写错(CER↔WER),数字抄错,模型名写错 |
| traceability-gap | claim 无出处 | 关键数字没有 [§X.X]/[Table N] 标注 |
| overclaim | 结论超出证据 | "唯一""全面优于""SOTA"但证据不支持 |
| fact-inference-mixing | 事实与推断混写 | 把 agent 的解释写成论文的结论,没区分 |
| summary-without-mechanism | 像摘要不像 deep | "方法"节只描述 WHAT(组件列表),不解释 WHY(因果链) |
| missing-lineage | 缺谱系定位 | KB 背景节没有具体定位,或完全缺失 |
| weak-reusability | 可借鉴太泛 | "提出了新方法"(贡献声明)而非"X trick 可迁移到 Y 场景" |
| bad-linking | 概念挂接不合理 | 遗漏明显相关概念,或挂接了不相关概念 |
| template-compliance | 格式不合规 | frontmatter 缺字段,缺 section,速查卡片字段空 |
| kb-safety-risk | 可能污染知识库 | 新建概念不满足准入规则,或反向更新内容有事实错误 |

## 严重度定义

| 级别 | 含义 | 行动 |
|------|------|------|
| high | 影响可信性,会误导后续使用 | 必须修正才能反向更新 |
| medium | 影响质量,但不阻塞 | 建议修正,可标注后放行 |
| low | 影响可读性/复用性 | 可忽略或后续批量改善 |

## 审阅结论

| 结论 | 含义 |
|------|------|
| pass | 可直接反向更新 |
| pass-with-fixes | 有 medium 问题,建议修正但可放行 |
| revise | 有 high 问题,需修正后重新审阅 |
| reject-as-deep | 质量不达 deep 标准,降为 enhanced-card |

## 审阅维度 (5 维)

### 1. 可理解性 (understandability)
- "方法:它怎么 work" 是否包含因果解释(WHY)
- 检查: 有无"之所以/因为/这使得/从而" vs 仅有"使用了/采用了/包含"
- 检查: 每个关键设计选择是否回答"为什么选这个"

### 2. 可溯源性 (traceability)
- 数字型 claim 是否都有 [§X.X]/[Table N] 标注
- 计算: 标注覆盖率 = 有标注 claim 数 / 总数字 claim 数
- 检查: 是否存在事实与推断混写(agent 解释 vs 作者原文)

### 3. 严谨性 (rigor)
- 指标名是否正确(CER/WER/MOS/SIM 不混淆)
- baseline 对比措辞是否适当("优于"需要证据,"领先"需要具体数字)
- "SOTA/唯一/全面" 等强断言是否有充分支撑

### 4. 可导航性 (navigability)
- concepts/models/tasks/datasets 挂接是否合理
- KB 背景是否有具体谱系定位(不是"与已有工作相关")
- 速查卡片 5 字段是否都有实质内容

### 5. 知识库安全性 (kb-safety)
- 新建概念页是否满足准入规则(多篇引用/前置知识/连接论文 ≥2)
- 反向更新内容是否包含 factual-error 或 overclaim
- 如果直接进入可信层,是否会造成误导
