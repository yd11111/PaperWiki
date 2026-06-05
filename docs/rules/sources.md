# Sources/ 存储规则

> Tier 2 规则文件 | 加载者: reader
> 父文档: AGENTS.md §3
> 更新: 2026-06-04

---

- PDF 文件存放于 `Sources/`，已被 `.gitignore` 排除
- 可追溯性通过 frontmatter 保证：
  - `arxiv_id`: arXiv 论文 ID（如 `"2301.12345"`）
  - `source`: 指向本地 PDF 的相对路径 `"Sources/{笔记名}.pdf"`
- Agent 引用论文时使用 frontmatter 中的标识符，不依赖本地 PDF 存在

## 命名规范

PDF 文件名 **必须** 与论文笔记文件名一致（不含 .md 后缀）。文件名**禁止包含空格**。

| 笔记文件 | PDF 文件 | 正确 |
|----------|----------|------|
| 论文笔记/CosyVoice3.md | Sources/CosyVoice3.pdf | ✓ |
| 论文笔记/Dragon-FM.md | Sources/Dragon-FM.pdf | ✓ |
| 论文笔记/BASETTS.md | Sources/BASETTS.pdf | ✓ |
| 论文笔记/CosyVoice 3.md | Sources/CosyVoice 3.pdf | ✗ (含空格) |
| 论文笔记/Dragon-FM.md | Sources/2507.22746.pdf | ✗ (arXiv ID) |

**规则**：
1. 文件名 = 笔记简称（与 .md 文件名相同），**不含空格**，保留大小写和连字符
2. 版本号紧跟名称不加空格（CosyVoice2, IndexTTS2, DMOSpeech2）
3. 多词名称直接拼接（BASETTS, ConditionalFlowMatching, ShallowFlowMatching）
4. 连字符保留（Cross-LingualF5-TTS, Step-Audio-EditX）
5. 禁止使用 arxiv ID 作为文件名
6. 禁止使用下划线替代空格
7. 禁止附加会议名后缀（EmotionThinker-AAAI.pdf ✗）
8. `source` frontmatter 字段必须指向实际存在的文件

## 下载规则

| 输入类型 | 操作 |
|----------|------|
| 用户给本地 PDF | 如不在 Sources/，复制到 `Sources/{笔记名}.pdf` |
| 用户给 arXiv URL | `curl -L -o "Sources/{笔记名}.pdf" "https://arxiv.org/pdf/{arxiv_id}.pdf"` |
| 用户给其他 URL | 用 Playwright 下载; 失败则在 source 中记录 URL 并在 log 中标注 `[pdf/missing]` |

**时机**：在精读 Step 1（获取论文）完成后、Step 2（KB 检索）开始前，确认 PDF 已存在于 Sources/ 且命名正确。
