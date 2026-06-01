# 跨 Session 一致性测试

> 目标:验证新 session 在只有 CLAUDE.md 自动加载的情况下,能否正确操作 PaperWiki。

---

## 测试 1: 精读一篇论文 (核心流程)

**输入:** 对新 session 说:
```
精读这篇论文 Sources/SoundStream.pdf
```

(先把 SoundStream 论文放到 Sources/: `cp "/Users/xiangshu/高德工作/文献/TTS-LLM/SoundsStream.pdf" Sources/SoundStream.pdf`)

**期望产出:**

| 检查项 | 期望值 | 验证命令 |
|---|---|---|
| 笔记位置 | `论文笔记/SoundStream.md` | `ls 论文笔记/SoundStream*.md` |
| tier | deep | `grep "^tier:" 论文笔记/SoundStream*.md` |
| status | draft | `grep "^status:" 论文笔记/SoundStream*.md` |
| KB 背景节 | 存在,引用 confirmed 页(RVQ 应该命中) | `grep "KB 背景" 论文笔记/SoundStream*.md` |
| kb_context_sources | > 0 | `grep "kb_context_sources" 论文笔记/SoundStream*.md` |
| 速查卡片 | 存在(一句话/路线/指标/可借鉴/局限) | `grep "可借鉴" 论文笔记/SoundStream*.md` |
| Claim 标注 | > 0 个 [§] 或 [Table] | `grep -c "\[§\|\[Table\|\[Fig" 论文笔记/SoundStream*.md` |
| 反向更新 | RVQ 概念页被追加(status 不变) | `grep "SoundStream" 概念库/Residual\ Vector\ Quantization.md` |
| Commit 格式 | `[ingest/deep] SoundStream — ...` | `git log --oneline -1` |
| log.md | 有 [kb/search] + [ingest/deep] 条目 | `grep "SoundStream" log.md` |

**失败标志(任一出现 = session 一致性有问题):**
- 笔记没有 KB 背景节,或写着"KB 检索未启用"
- 笔记没有速查卡片
- 没做反向更新(RVQ 页面没变化)
- commit message 格式不对
- 笔记输出到了其他位置(如旧 vault)
- 没有 claim 标注

---

## 测试 2: 收 inbox + 处理

**步骤 1:** 对新 session 说:
```
收一下这个到 inbox: https://arxiv.org/abs/2301.02111 ,是 VALL-E 的论文,想了解最早的 codec LM TTS
```

**期望:**
- `_inbox/` 下出现一个 .md 文件
- 内容含 title + source + why

**步骤 2:** 然后说:
```
精读 inbox 里的 VALL-E
```

**期望:**
- `_inbox/` 中 VALL-E 条目被删除
- `论文笔记/VALL-E.md` 被创建(tier: deep)
- 完整精读流程(KB+反向更新+lint+commit)

**验证:**
```bash
ls _inbox/          # 应为空
ls 论文笔记/VALL-E*  # 应存在
grep "tier: deep" 论文笔记/VALL-E*.md
git log --oneline -2  # 应有 inbox 和 ingest 两个 commit
```

---

## 测试 3: 概念页搜索补充

**输入:**
```
搜索补充一下 [[Codebook Collapse]] 的背景知识
```

**期望:**
- Agent 用 Playwright 搜索(不是 WebFetch)
- `概念库/Codebook Collapse.md` 内容丰富化
- status 变为 pending-review(实质修改)
- commit 格式: `[update/concept] Codebook Collapse — ...`

**验证:**
```bash
grep "^status:" "概念库/Codebook Collapse.md"  # pending-review
git log --oneline -1  # [update/concept]
```

---

## 测试 4: 违规检测 (agent 不应做的事)

**输入:**
```
这篇 IndexTTS2 看着不错,帮我升级为复现级
```

**期望:** Agent 应该执行升级(因为你明确说了"升级")。

**然后测试违规场景:**
```
帮我看看今天的推荐,把最好的那篇直接精读
```

**期望:** Agent 应该:
1. 生成推荐(card 级)
2. 展示给你看
3. **不自动精读** — 必须等你明确说"这篇精读"才能升级

如果 agent 自动把某篇推荐升级为精读 → 违反原则 5(默认不升级)→ 测试失败

---

## 测试 5: Full Lint

**输入:**
```
跑一下全量 lint
```

**期望:**
- `_lint/` 下生成新报告
- 报告包含: dead links / orphan pages / frontmatter / review backlog
- log.md 有 [lint/full] 条目
- 如果 Codebook Collapse 有 3+ key_papers 且 origin_paper 空 → 报告中有溯源提醒

---

## 评分标准

| 等级 | 条件 |
|---|---|
| **A: 完全一致** | 5 个测试全部通过,产出格式/流程与当前 session 一致 |
| **B: 基本一致** | 测试 1 和 2 通过(核心流程对),3-5 有小偏差(可接受) |
| **C: 部分一致** | 测试 1 通过但缺少 KB 背景或速查卡片(CLAUDE.md 规则被部分遗漏) |
| **D: 不一致** | 测试 1 失败(笔记格式错误/位置错误/不做反向更新) |
| **F: 完全失败** | Agent 不知道这是 PaperWiki,按默认行为操作 |

---

## 测试前准备

```bash
# 确保 SoundStream PDF 就位
cp "/Users/xiangshu/高德工作/文献/TTS-LLM/SoundsStream.pdf" /Users/xiangshu/PaperWiki/Sources/SoundStream.pdf
```

然后开新 session,cd 到 `/Users/xiangshu/PaperWiki/`,开始测试。
