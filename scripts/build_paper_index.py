#!/usr/bin/env python3
"""从论文笔记 frontmatter + 速查卡片生成静态 HTML 索引页。

用法: python3 scripts/build_paper_index.py
输出: docs/paper-index.html
"""

import json
import re
import sys
from pathlib import Path

import yaml

VAULT = Path(__file__).resolve().parent.parent
NOTES_DIR = VAULT / "论文笔记"
OUTPUT = VAULT / "docs" / "paper-index.html"

PREPRINT_RE = re.compile(r"^(arXiv|Preprint)", re.IGNORECASE)


def parse_frontmatter(text: str) -> dict | None:
    m = re.match(r"^---\n(.*?\n)---\n", text, re.DOTALL)
    if not m:
        return None
    try:
        return yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None


def extract_card_field(text: str, label: str) -> str:
    pattern = rf"\*\*{re.escape(label)}\*\*:\s*(.+)"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def arxiv_to_date(arxiv_id: str) -> str:
    if not arxiv_id:
        return ""
    m = re.match(r"(\d{2})(\d{2})\.", arxiv_id)
    if m:
        yy, mm = m.groups()
        year = 2000 + int(yy)
        return f"{year}-{mm}"
    return ""


def extract_paper(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8")
    fm = parse_frontmatter(text)
    if not fm or fm.get("type") != "paper":
        return None

    name = path.stem
    arxiv_id = fm.get("arxiv_id", "").strip('"').strip("'")
    date = arxiv_to_date(arxiv_id)
    if not date and fm.get("year"):
        date = str(fm["year"])

    venue = fm.get("venue", "").strip('"').strip("'")
    tags = fm.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]

    concepts = fm.get("concepts", [])
    if isinstance(concepts, list):
        concepts = [c.strip().strip("[").strip("]") for c in concepts]

    return {
        "name": name,
        "title": fm.get("title", "").strip('"').strip("'"),
        "arxiv_id": arxiv_id,
        "date": date,
        "venue": venue,
        "org": fm.get("org", ""),
        "tier": fm.get("tier", "card"),
        "tags": tags[:8],
        "summary": extract_card_field(text, "一句话"),
        "route": extract_card_field(text, "路线"),
        "metrics": extract_card_field(text, "指标"),
        "takeaway": extract_card_field(text, "可借鉴"),
        "limitation": extract_card_field(text, "局限"),
        "concepts": concepts,
        "source": fm.get("source", ""),
        "github": fm.get("github", ""),
        "demo": fm.get("demo_page", ""),
    }


def build_html(papers: list[dict]) -> str:
    data_json = json.dumps(papers, ensure_ascii=False, indent=None)

    deep = sum(1 for p in papers if p["tier"] == "deep")
    repro = sum(1 for p in papers if p["tier"] == "repro")
    card = sum(1 for p in papers if p["tier"] in ("card", "enhanced-card"))
    years = len({p["date"][:4] for p in papers if p["date"]})

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PaperWiki — 论文索引</title>
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --surface2: #1c2129;
    --surface3: #21262d;
    --border: #30363d;
    --border-light: #3d444d;
    --text: #e6edf3;
    --text2: #8b949e;
    --text3: #656d76;
    --accent: #58a6ff;
    --accent2: #3fb950;
    --tag-bg: #1f2a3a;
    --tag-text: #79c0ff;
    --hover: #1c2433;
    --deep: #3fb950;
    --repro: #d2a8ff;
    --card: #8b949e;
    --enhanced-card: #f0883e;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    min-height: 100vh;
  }}
  .header {{ padding: 40px 48px 32px; border-bottom: 1px solid var(--border); }}
  .header h1 {{ font-size: 28px; font-weight: 600; margin-bottom: 6px; }}
  .header h1 span {{ color: var(--accent); }}
  .header .subtitle {{ color: var(--text2); font-size: 14px; }}
  .stats {{
    display: flex; gap: 32px; padding: 20px 48px;
    border-bottom: 1px solid var(--border); background: var(--surface);
  }}
  .stat-item {{ display: flex; flex-direction: column; align-items: center; }}
  .stat-num {{ font-size: 24px; font-weight: 700; color: var(--accent); }}
  .stat-label {{ font-size: 12px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; }}
  .controls {{
    display: flex; gap: 12px; padding: 16px 48px;
    align-items: center; flex-wrap: wrap; border-bottom: 1px solid var(--border);
  }}
  .search-box {{ flex: 1; min-width: 240px; max-width: 400px; position: relative; }}
  .search-box input {{
    width: 100%; padding: 8px 12px 8px 36px;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; color: var(--text); font-size: 14px; outline: none;
    transition: border-color 0.2s;
  }}
  .search-box input:focus {{ border-color: var(--accent); }}
  .search-box svg {{
    position: absolute; left: 10px; top: 50%; transform: translateY(-50%); color: var(--text2);
  }}
  .filter-group {{
    display: flex; gap: 4px; background: var(--surface);
    border-radius: 6px; padding: 2px; border: 1px solid var(--border);
  }}
  .filter-btn {{
    padding: 6px 14px; font-size: 13px; border: none; background: none;
    color: var(--text2); border-radius: 4px; cursor: pointer; transition: all 0.15s;
  }}
  .filter-btn:hover {{ color: var(--text); background: var(--surface2); }}
  .filter-btn.active {{ background: var(--accent); color: #fff; }}
  .sort-select {{
    padding: 7px 12px; background: var(--surface); border: 1px solid var(--border);
    border-radius: 6px; color: var(--text); font-size: 13px; cursor: pointer; outline: none;
  }}
  .result-count {{ font-size: 13px; color: var(--text2); margin-left: auto; }}
  .paper-list {{ padding: 8px 48px 48px; }}
  .paper-item {{
    border: 1px solid var(--border); border-radius: 8px; margin-bottom: 8px;
    background: var(--surface); transition: border-color 0.15s; overflow: hidden;
  }}
  .paper-item:hover {{ border-color: var(--border-light); }}
  .paper-main {{
    display: flex; align-items: flex-start; padding: 16px 20px; cursor: pointer; gap: 16px;
  }}
  .expand-icon {{
    color: var(--text3); font-size: 11px; margin-top: 7px;
    transition: transform 0.2s; flex-shrink: 0; width: 12px;
  }}
  .paper-item.expanded .expand-icon {{ transform: rotate(90deg); }}
  .paper-head {{ flex: 1; min-width: 0; }}
  .title-line {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
  .paper-name {{ font-size: 20px; font-weight: 700; color: var(--text); white-space: nowrap; flex-shrink: 0; }}
  .paper-title {{ font-size: 14px; color: var(--text2); font-weight: 400; line-height: 1.4; }}
  .meta-line {{
    display: grid;
    grid-template-columns: 64px 180px 80px 56px 1fr;
    align-items: center; gap: 8px; margin-top: 8px;
  }}
  .meta-date {{ font-size: 13px; color: var(--text2); font-variant-numeric: tabular-nums; }}
  .meta-org {{
    font-size: 12px; color: var(--accent2); font-weight: 500;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }}
  .meta-venue {{
    font-size: 12px; color: var(--text3); padding: 1px 8px;
    background: var(--surface3); border-radius: 4px; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis;
  }}
  .tier {{
    display: inline-block; padding: 1px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px;
  }}
  .tier-deep {{ background: rgba(63,185,80,0.15); color: var(--deep); }}
  .tier-repro {{ background: rgba(210,168,255,0.15); color: var(--repro); }}
  .tier-card {{ background: rgba(139,148,158,0.15); color: var(--card); }}
  .tier-enhanced-card {{ background: rgba(240,136,62,0.15); color: var(--enhanced-card); }}
  .tags-inline {{ display: flex; flex-wrap: wrap; gap: 4px; }}
  .tag {{
    display: inline-block; padding: 1px 8px;
    background: var(--tag-bg); color: var(--tag-text);
    border-radius: 10px; font-size: 11px;
    cursor: pointer; transition: all 0.15s; white-space: nowrap;
  }}
  .tag:hover {{ background: rgba(88,166,255,0.25); }}
  .tag.active {{ background: var(--accent); color: #fff; }}
  .paper-links {{
    display: flex; gap: 6px; flex-shrink: 0; align-items: center; margin-top: 4px;
  }}
  .link-btn {{
    display: inline-flex; align-items: center; gap: 4px;
    padding: 5px 10px; border-radius: 5px; font-size: 12px;
    text-decoration: none; transition: all 0.15s; white-space: nowrap;
    border: 1px solid var(--border); color: var(--text2); background: none;
  }}
  .link-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
  .link-btn.disabled {{ opacity: 0.25; pointer-events: none; }}
  .link-btn svg {{ width: 14px; height: 14px; }}
  .paper-detail {{ display: none; padding: 0 20px 20px 44px; }}
  .paper-item.expanded .paper-detail {{ display: block; }}
  .card {{
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px 24px;
  }}
  .card-row {{ display: flex; gap: 8px; margin-bottom: 10px; font-size: 13px; line-height: 1.6; }}
  .card-row:last-child {{ margin-bottom: 0; }}
  .card-label {{ flex-shrink: 0; font-weight: 600; color: var(--accent); min-width: 56px; }}
  .card-value {{ color: var(--text); }}
  .card-route {{
    color: var(--text2); font-family: "SF Mono", "Fira Code", monospace;
    font-size: 12px; word-break: break-all;
  }}
  .card-concepts {{
    display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px;
    padding-top: 14px; border-top: 1px solid var(--border);
  }}
  .card-concepts-label {{
    font-size: 11px; color: var(--text3); text-transform: uppercase;
    letter-spacing: 0.5px; width: 100%; margin-bottom: 2px;
  }}
  .concept-chip {{
    padding: 2px 10px; background: rgba(210,168,255,0.1);
    color: var(--repro); border-radius: 10px; font-size: 12px;
  }}
  .empty-state {{ text-align: center; padding: 64px 0; color: var(--text2); font-size: 14px; display: none; }}
  @media (max-width: 900px) {{
    .header, .stats, .controls, .paper-list {{ padding-left: 16px; padding-right: 16px; }}
    .stats {{ flex-wrap: wrap; gap: 16px; }}
    .paper-main {{ flex-direction: column; }}
    .paper-links {{ margin-top: 8px; }}
    .title-line {{ flex-direction: column; gap: 4px; }}
    .paper-name {{ font-size: 17px; }}
    .meta-line {{ grid-template-columns: 64px 1fr; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Paper<span>Wiki</span> 论文索引</h1>
  <div class="subtitle">TTS / Speech / Audio 方向论文知识库</div>
</div>

<div class="stats">
  <div class="stat-item"><div class="stat-num">{len(papers)}</div><div class="stat-label">论文总数</div></div>
  <div class="stat-item"><div class="stat-num">{deep}</div><div class="stat-label">Deep</div></div>
  <div class="stat-item"><div class="stat-num">{repro}</div><div class="stat-label">Repro</div></div>
  <div class="stat-item"><div class="stat-num">{card}</div><div class="stat-label">Card</div></div>
  <div class="stat-item"><div class="stat-num">{years}</div><div class="stat-label">跨越年份</div></div>
</div>

<div class="controls">
  <div class="search-box">
    <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M11.5 7a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0Zm-.82 4.74a6 6 0 1 1 1.06-1.06l3.04 3.04a.75.75 0 1 1-1.06 1.06l-3.04-3.04Z"/></svg>
    <input type="text" id="searchInput" placeholder="搜索论文名、标签…">
  </div>
  <div class="filter-group" id="tierFilter">
    <button class="filter-btn active" data-tier="all">全部</button>
    <button class="filter-btn" data-tier="deep">Deep</button>
    <button class="filter-btn" data-tier="repro">Repro</button>
    <button class="filter-btn" data-tier="card">Card</button>
  </div>
  <select class="sort-select" id="sortSelect">
    <option value="date-desc">日期 ↓ 新→旧</option>
    <option value="date-asc">日期 ↑ 旧→新</option>
    <option value="name-asc">名称 A→Z</option>
    <option value="name-desc">名称 Z→A</option>
  </select>
  <div class="result-count" id="resultCount"></div>
</div>

<div class="paper-list" id="paperList"></div>
<div class="empty-state" id="emptyState">没有匹配的论文</div>

<script>
const PAPERS = {data_json};

const ICONS = {{
  arxiv: `<svg viewBox="0 0 16 16" fill="currentColor"><path d="M2 1.75C2 .784 2.784 0 3.75 0h6.586c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v9.586A1.75 1.75 0 0 1 13.25 16h-9.5A1.75 1.75 0 0 1 2 14.25Zm1.75-.25a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h9.5a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688l-.011-.013-2.914-2.914-.013-.011Z"/></svg>`,
  github: `<svg viewBox="0 0 16 16" fill="currentColor"><path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/></svg>`,
  demo: `<svg viewBox="0 0 16 16" fill="currentColor"><path d="M0 2.75C0 1.784.784 1 1.75 1h12.5c.966 0 1.75.784 1.75 1.75v8.5A1.75 1.75 0 0 1 14.25 13H1.75A1.75 1.75 0 0 1 0 11.25Zm1.75-.25a.25.25 0 0 0-.25.25v8.5c0 .138.112.25.25.25h12.5a.25.25 0 0 0 .25-.25v-8.5a.25.25 0 0 0-.25-.25ZM3.5 14.5a.75.75 0 0 1 .75-.75h7.5a.75.75 0 0 1 0 1.5h-7.5a.75.75 0 0 1-.75-.75Z"/></svg>`,
  pdf: `<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4.5 0A2.5 2.5 0 0 0 2 2.5v11A2.5 2.5 0 0 0 4.5 16h7a2.5 2.5 0 0 0 2.5-2.5v-8.072a2.5 2.5 0 0 0-.732-1.768l-2.928-2.928A2.5 2.5 0 0 0 8.572 0H4.5Zm0 1.5h3.572a1 1 0 0 1 .707.293l2.928 2.928a1 1 0 0 1 .293.707V13.5a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1v-11a1 1 0 0 1 1-1ZM6 7a.75.75 0 0 0 0 1.5h4A.75.75 0 0 0 10 7H6Zm0 2.5a.75.75 0 0 0 0 1.5h4a.75.75 0 0 0 0-1.5H6Z"/></svg>`,
  note: `<svg viewBox="0 0 16 16" fill="currentColor"><path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0 1 14.25 16H1.75A1.75 1.75 0 0 1 0 14.25ZM1.75 1.5a.25.25 0 0 0-.25.25v12.5c0 .138.112.25.25.25h12.5a.25.25 0 0 0 .25-.25V1.75a.25.25 0 0 0-.25-.25ZM4 4h8v1.5H4Zm0 3h8v1.5H4Zm0 3h5v1.5H4Z"/></svg>`
}};

function esc(s) {{ return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }}
function arxivPDF(id) {{ return id ? `https://arxiv.org/pdf/${{id}}` : ""; }}
function arxivAbs(id) {{ return id ? `https://arxiv.org/abs/${{id}}` : ""; }}
function displayVenue(v) {{ return /^(arXiv|Preprint)/i.test(v) ? "Preprint" : v; }}

let currentTier = "all", currentSearch = "", currentSort = "date-desc", activeTag = null;

function dateSortKey(d) {{
  const p = d.split("-");
  return parseInt(p[0]) * 100 + parseInt(p[1] || "0");
}}

function render() {{
  let list = PAPERS.filter(p => {{
    if (currentTier !== "all" && p.tier !== currentTier) return false;
    if (activeTag && !p.tags.includes(activeTag)) return false;
    if (currentSearch) {{
      const q = currentSearch.toLowerCase();
      if (!(p.name + " " + p.title + " " + p.tags.join(" ") + " " + (p.org||"")).toLowerCase().includes(q)) return false;
    }}
    return true;
  }});

  list.sort((a, b) => {{
    switch (currentSort) {{
      case "date-desc": return dateSortKey(b.date) - dateSortKey(a.date) || a.name.localeCompare(b.name);
      case "date-asc": return dateSortKey(a.date) - dateSortKey(b.date) || a.name.localeCompare(b.name);
      case "name-asc": return a.name.localeCompare(b.name);
      case "name-desc": return b.name.localeCompare(a.name);
    }}
  }});

  const container = document.getElementById("paperList");
  container.innerHTML = "";

  list.forEach(p => {{
    const absUrl = arxivAbs(p.arxiv_id);
    const pdfUrl = p.source || arxivPDF(p.arxiv_id);
    const noteUrl = `论文笔记/${{p.name}}.md`;
    const venue = displayVenue(p.venue);

    const div = document.createElement("div");
    div.className = "paper-item";

    const hasCard = p.summary || p.route;
    const cardHtml = hasCard ? `
      <div class="paper-detail">
        <div class="card">
          ${{p.summary ? `<div class="card-row"><span class="card-label">一句话</span><span class="card-value">${{esc(p.summary)}}</span></div>` : ""}}
          ${{p.route ? `<div class="card-row"><span class="card-label">路线</span><span class="card-route">${{esc(p.route)}}</span></div>` : ""}}
          ${{p.metrics ? `<div class="card-row"><span class="card-label">指标</span><span class="card-value">${{esc(p.metrics)}}</span></div>` : ""}}
          ${{p.takeaway ? `<div class="card-row"><span class="card-label">可借鉴</span><span class="card-value">${{esc(p.takeaway)}}</span></div>` : ""}}
          ${{p.limitation ? `<div class="card-row"><span class="card-label">局限</span><span class="card-value">${{esc(p.limitation)}}</span></div>` : ""}}
          ${{p.concepts && p.concepts.length ? `
          <div class="card-concepts">
            <div class="card-concepts-label">关联概念</div>
            ${{p.concepts.map(c => `<span class="concept-chip">${{esc(c)}}</span>`).join("")}}
          </div>` : ""}}
        </div>
      </div>` : "";

    div.innerHTML = `
      <div class="paper-main">
        <span class="expand-icon">${{hasCard ? "▶" : ""}}</span>
        <div class="paper-head">
          <div class="title-line">
            <span class="paper-name">${{esc(p.name)}}</span>
            <span class="paper-title">${{esc(p.title)}}</span>
          </div>
          <div class="meta-line">
            <span class="meta-date">${{p.date}}</span>
            <span class="meta-org" title="${{esc(p.org || "")}}">${{esc(p.org || "")}}</span>
            <span class="meta-venue">${{esc(venue)}}</span>
            <span class="tier tier-${{p.tier}}">${{p.tier}}</span>
            <div class="tags-inline">${{p.tags.slice(0, 5).map(t =>
              `<span class="tag${{activeTag === t ? ' active' : ''}}" data-tag="${{t}}">${{t}}</span>`
            ).join("")}}</div>
          </div>
        </div>
        <div class="paper-links">
          <a class="link-btn" href="${{noteUrl}}" title="阅读报告">${{ICONS.note}} 笔记</a>
          <a class="link-btn${{pdfUrl ? '' : ' disabled'}}" href="${{pdfUrl}}" title="PDF">${{ICONS.pdf}} PDF</a>
          <a class="link-btn${{absUrl ? '' : ' disabled'}}" href="${{absUrl}}" target="_blank" title="arXiv">${{ICONS.arxiv}} arXiv</a>
          <a class="link-btn${{p.github ? '' : ' disabled'}}" href="${{p.github || '#'}}" target="_blank" title="GitHub">${{ICONS.github}}</a>
          <a class="link-btn${{p.demo ? '' : ' disabled'}}" href="${{p.demo || '#'}}" target="_blank" title="Demo">${{ICONS.demo}}</a>
        </div>
      </div>
      ${{cardHtml}}
    `;

    if (hasCard) {{
      div.querySelector(".paper-main").addEventListener("click", (e) => {{
        if (e.target.closest("a")) return;
        if (e.target.classList.contains("tag")) return;
        div.classList.toggle("expanded");
      }});
    }}

    div.querySelectorAll(".tag").forEach(tag => {{
      tag.addEventListener("click", (e) => {{
        e.stopPropagation();
        const t = tag.dataset.tag;
        activeTag = activeTag === t ? null : t;
        render();
      }});
    }});

    container.appendChild(div);
  }});

  document.getElementById("resultCount").textContent = `显示 ${{list.length}} 篇`;
  document.getElementById("emptyState").style.display = list.length === 0 ? "block" : "none";
}}

document.getElementById("tierFilter").addEventListener("click", e => {{
  const btn = e.target.closest(".filter-btn");
  if (!btn) return;
  document.querySelectorAll("#tierFilter .filter-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  currentTier = btn.dataset.tier;
  render();
}});
document.getElementById("searchInput").addEventListener("input", e => {{ currentSearch = e.target.value; render(); }});
document.getElementById("sortSelect").addEventListener("change", e => {{ currentSort = e.target.value; render(); }});

render();
</script>
</body>
</html>"""


def main():
    if not NOTES_DIR.is_dir():
        print(f"Error: {NOTES_DIR} not found", file=sys.stderr)
        sys.exit(1)

    papers = []
    errors = []
    for path in sorted(NOTES_DIR.glob("*.md")):
        try:
            p = extract_paper(path)
            if p:
                papers.append(p)
        except Exception as e:
            errors.append(f"{path.name}: {e}")

    papers.sort(key=lambda p: dateSortKey(p["date"]), reverse=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_html(papers), encoding="utf-8")

    print(f"Built {OUTPUT}")
    print(f"  {len(papers)} papers extracted")
    print(f"  {sum(1 for p in papers if p['summary'])} with 速查 card")
    print(f"  {sum(1 for p in papers if p['org'])} with org info")
    if errors:
        print(f"  {len(errors)} errors:")
        for e in errors[:5]:
            print(f"    {e}")


def dateSortKey(d):
    if not d:
        return 0
    parts = d.split("-")
    return int(parts[0]) * 100 + int(parts[1]) if len(parts) > 1 else int(parts[0]) * 100


if __name__ == "__main__":
    main()
