# 超级精读 Prompt (复制到新 session 使用)

---

## 使用方式

复制下面分隔线内的 prompt，在新 Claude Code session 中粘贴，然后附上具体论文名。

例: `{粘贴 prompt} 目标论文: dots.tts, VoxCPM2`

---

## Prompt

```
你是 TTS 方向的论文精读专家。本次任务是"超级精读" — 结合论文 PDF + GitHub 开源代码，做到实现级理解。

## 工作目录

/Users/xiangshu/PaperWiki

这是一个 Obsidian 知识库，规则见 AGENTS.md。本次产出的笔记写入 论文笔记/ 目录。

## 超级精读的定义

比 repro (复现分析) 更深一层:
- repro = 读论文，提取复现要点
- **超级精读 = 下载代码仓库，逐模块对照论文分析，验证论文 claim，发现论文没写的实现细节**

## 工作流程

### Phase 1: 准备
1. 读 AGENTS.md 获取 vault 规则
2. 读已有笔记 (论文笔记/{论文名}.md)，了解已有分析
3. 从论文 PDF 提取 GitHub/代码仓库链接
4. git clone 代码到 /tmp/{论文名}-code/

### Phase 2: 代码分析 (核心)
逐模块对照论文和代码，重点关注:

**架构验证**
- 论文图和代码是否一致？有没有论文没画的模块？
- 各模块的实际输入输出 shape
- 论文说 "我们使用 X"，代码里实际用的是什么？

**训练细节**
- 实际 loss 函数（论文可能简化了）
- learning rate schedule, warmup, batch size
- 数据预处理 pipeline（采样率、分帧、归一化）
- 有没有论文没提的 trick（gradient clipping, EMA, 特殊初始化）

**推理细节**
- 实际推理流程 vs 论文描述
- 采样策略（temperature, top-k, top-p, cfg scale）
- 有没有后处理（vocoder, denoising, silence trimming）

**关键数字验证**
- 论文说的参数量 vs 代码实际参数量
- 论文说的 FLOPs/速度 vs 代码 profile

**隐藏 know-how**
- 代码注释里的 insight
- 被注释掉的代码（说明尝试过但放弃了）
- config 文件里的关键超参数
- README 或 issue 里的已知问题

### Phase 3: 产出

在已有笔记基础上，追加 `## 代码级分析` 大节，包含:

```markdown
## 代码级分析

> [!info] 代码来源
> - 仓库: {GitHub URL}
> - commit: {hash}
> - 分析日期: {date}

### 架构验证
(论文图 vs 代码实际，标注差异)

### 论文未写的实现细节
(按重要性排序，每条标注代码位置 file:line)

### 训练 pipeline 拆解
(数据流: 原始音频 → ... → loss，标注每步的实际实现)

### 推理 pipeline 拆解
(输入 → ... → 输出波形，标注关键参数)

### 关键超参数表
| 参数 | 论文值 | 代码实际值 | 备注 |
|------|--------|-----------|------|

### 复现 checklist (基于代码)
- [ ] 环境依赖 (Python/CUDA/关键库版本)
- [ ] 数据准备 (格式、预处理脚本)
- [ ] 预训练模型依赖
- [ ] 训练命令
- [ ] 推理命令
- [ ] 已知坑 (来自 issue/注释)

### 代码质量与可复现性评估
(工程质量、文档完善度、社区活跃度、复现难度 1-5)
```

### Phase 4: KB 更新
- 如果代码分析发现了概念页需要更新的信息，按 kb.md 规则执行反向更新
- 如果发现论文 claim 和代码不一致，在笔记中显式标注

## 约束
- 所有网络访问用 Playwright (WebFetch 不稳定)
- 笔记文件名不含空格
- 不自行升级笔记层级
- git clone 到 /tmp/，不要污染 vault
- commit 格式: [ingest/super] {论文名} — 代码级分析
```

---

## 变体: 多论文并行

如果要同时超级精读多篇，在 prompt 末尾加:

```
目标论文: {论文1}, {论文2}, {论文3}
请用多 agent 并行处理，每个 agent 负责一篇论文的 clone + 分析。最后统一 commit。
```

## 变体: 对比精读

如果要对比同一路线的多个实现:

```
对比超级精读: {论文1} vs {论文2} vs {论文3}
重点: 同一技术路线的不同实现选择。产出对比表，标注每个实现的 trade-off。
```
