#!/usr/bin/env python3
"""PaperWiki vault linter — 18 mechanizable checks.

Usage:
    python3 scripts/lint.py                       # full check (L1-L18)
    python3 scripts/lint.py --per-ingest FILE      # per-ingest subset (L1,L6,L7,L8)
    python3 scripts/lint.py --check L1,L3,L5       # specific checks only
    python3 scripts/lint.py --check L9 --fix       # update CLAUDE.md vault stats

Exit code: 0 = all pass, 1 = errors found.
Each error line: ERROR [Lx] file: agent-readable fix instruction.
Each warning line: WARNING [Lx] file: advisory (does not affect exit code).
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

VAULT = Path(__file__).resolve().parent.parent

DIRS = {
    "notes": VAULT / "论文笔记",
    "concepts": VAULT / "概念库",
    "models": VAULT / "模型库",
    "datasets": VAULT / "数据集",
    "tasks": VAULT / "任务库",
    "moc": VAULT / "_MOC",
    "sources": VAULT / "Sources",
}

# ---------------------------------------------------------------------------
# Frontmatter parsing (no external deps)
# ---------------------------------------------------------------------------

def parse_frontmatter(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return None
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    block = text[4:end]
    fm = {}
    for line in block.split("\n"):
        m = re.match(r'^(\w[\w_-]*)\s*:\s*(.*)', line)
        if m:
            key = m.group(1)
            val = m.group(2).strip()
            if val.startswith("["):
                items = re.findall(r'"([^"]*)"', val) or re.findall(r"'([^']*)'", val)
                if not items and val not in ("[]",):
                    bare = val.strip("[]")
                    items = [x.strip() for x in bare.split(",") if x.strip()]
                fm[key] = items
            elif val in ('""', "''", ""):
                fm[key] = ""
            else:
                val_unquoted = val.strip('"').strip("'")
                try:
                    fm[key] = int(val_unquoted)
                except ValueError:
                    fm[key] = val_unquoted
    return fm


def read_content(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Wikilink extraction
# ---------------------------------------------------------------------------

WIKILINK_RE = re.compile(r'\[\[([^\]]+)\]\]')

def extract_wikilinks(text: str, skip_frontmatter: bool = True) -> list[tuple[str, int]]:
    """Return list of (target, line_number) for all wikilinks in text.

    When skip_frontmatter is True, ignores wikilinks inside the YAML frontmatter block.
    """
    results = []
    lines = text.split("\n")
    in_frontmatter = False
    fm_count = 0
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if skip_frontmatter:
            if stripped == "---":
                fm_count += 1
                if fm_count == 1:
                    in_frontmatter = True
                    continue
                elif fm_count == 2:
                    in_frontmatter = False
                    continue
            if in_frontmatter:
                continue
        for m in WIKILINK_RE.finditer(line):
            raw = m.group(1)
            target = raw.split("|")[0].strip()
            results.append((target, i))
    return results


def resolve_wikilink(target: str) -> Path | None:
    """Resolve a wikilink target to a file path."""
    if target.startswith("http") or target.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg")):
        return None

    has_extension = "." in Path(target).name
    if has_extension:
        exact = VAULT / target
        return exact

    candidates = [
        VAULT / f"{target}.md",
        VAULT / "论文笔记" / f"{target}.md",
        VAULT / "概念库" / f"{target}.md",
        VAULT / "模型库" / f"{target}.md",
        VAULT / "数据集" / f"{target}.md",
        VAULT / "任务库" / f"{target}.md",
        VAULT / "_MOC" / f"{target}.md",
    ]
    for c in candidates:
        if c.exists():
            return c
    return VAULT / f"{target}.md"

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

PAPER_COMMON = ["type", "title", "authors", "year", "tags", "status", "created", "updated"]
PAPER_DEEP_REPRO = PAPER_COMMON + ["tier", "kb_context_sources"]
PAPER_CARD = PAPER_COMMON + ["tier"]

CONCEPT_REQUIRED = ["type", "title", "aliases", "category", "tags", "key_papers",
                    "status", "lifecycle", "created", "updated"]
MODEL_REQUIRED = ["type", "title", "aliases", "org", "year", "tags", "key_concepts",
                  "tasks", "key_papers", "status", "lifecycle", "created", "updated"]
DATASET_REQUIRED = ["type", "title", "aliases", "domain", "tags",
                    "status", "lifecycle", "created", "updated"]
TASK_REQUIRED = ["type", "title", "aliases", "tags", "key_approaches", "key_models",
                 "benchmarks", "metrics", "status", "lifecycle", "created", "updated"]


def get_schema(fm: dict) -> tuple[list[str], str]:
    t = fm.get("type", "")
    if t == "paper":
        tier = fm.get("tier", "")
        if tier in ("deep", "repro"):
            return PAPER_DEEP_REPRO, f"paper/{tier}"
        elif tier in ("card", "enhanced-card"):
            return PAPER_CARD, f"paper/{tier}"
        else:
            return PAPER_COMMON, "paper/unknown-tier"
    elif t == "concept":
        return CONCEPT_REQUIRED, "concept"
    elif t == "model":
        return MODEL_REQUIRED, "model"
    elif t == "dataset":
        return DATASET_REQUIRED, "dataset"
    elif t == "task":
        return TASK_REQUIRED, "task"
    return [], f"unknown/{t}"

# ---------------------------------------------------------------------------
# Check implementations
# ---------------------------------------------------------------------------

errors: list[str] = []

def err(check: str, file: str, msg: str):
    errors.append(f"ERROR [{check}] {file}: {msg}")


def collect_md_files(*dirs: Path) -> list[Path]:
    files = []
    for d in dirs:
        if d.exists():
            files.extend(sorted(d.glob("*.md")))
    return files


# L1: Frontmatter field existence validation
def check_l1():
    content_dirs = [DIRS["notes"], DIRS["concepts"], DIRS["models"],
                    DIRS["datasets"], DIRS["tasks"]]
    for f in collect_md_files(*content_dirs):
        fm = parse_frontmatter(f)
        if fm is None:
            err("L1", f.relative_to(VAULT).as_posix(), "文件缺少 YAML frontmatter。请添加 --- 包围的 frontmatter 块")
            continue
        if "type" not in fm:
            err("L1", f.relative_to(VAULT).as_posix(), "frontmatter 缺少 'type' 字段。请添加 type: paper|concept|model|dataset|task")
            continue
        schema, label = get_schema(fm)
        if not schema:
            continue
        for field in schema:
            if field not in fm:
                err("L1", f.relative_to(VAULT).as_posix(),
                    f"frontmatter 缺少必填字段 '{field}' (schema: {label})。请在 frontmatter 中添加此字段")
            elif fm[field] == "" and field not in ("origin_paper", "merged_into", "deprecated_reason", "arxiv_id", "source", "venue"):
                err("L1", f.relative_to(VAULT).as_posix(),
                    f"frontmatter 字段 '{field}' 为空 (schema: {label})。请填入有效值")


def check_l1_single(path: Path):
    fm = parse_frontmatter(path)
    if fm is None:
        err("L1", path.relative_to(VAULT).as_posix(), "文件缺少 YAML frontmatter")
        return
    if "type" not in fm:
        err("L1", path.relative_to(VAULT).as_posix(), "frontmatter 缺少 'type' 字段")
        return
    schema, label = get_schema(fm)
    for field in schema:
        if field not in fm:
            err("L1", path.relative_to(VAULT).as_posix(),
                f"frontmatter 缺少必填字段 '{field}' (schema: {label})。请在 frontmatter 中添加此字段")
        elif fm[field] == "" and field not in ("origin_paper", "merged_into", "deprecated_reason", "arxiv_id", "source", "venue"):
            err("L1", path.relative_to(VAULT).as_posix(),
                f"frontmatter 字段 '{field}' 为空 (schema: {label})。请填入有效值")


# L2: Frontmatter year vs MOC section year consistency
def check_l2():
    note_years: dict[str, int] = {}
    for f in collect_md_files(DIRS["notes"]):
        fm = parse_frontmatter(f)
        if fm and "year" in fm:
            name = f.stem
            try:
                note_years[name] = int(fm["year"])
            except (ValueError, TypeError):
                pass

    moc_dir = DIRS["moc"]
    if not moc_dir.exists():
        return
    for moc_file in sorted(moc_dir.glob("*.md")):
        content = read_content(moc_file)
        current_year = None
        for line_num, line in enumerate(content.split("\n"), 1):
            year_match = re.match(r'^###\s+(\d{4})\b', line)
            if year_match:
                current_year = int(year_match.group(1))
                continue
            if current_year is None:
                continue
            for wl_match in WIKILINK_RE.finditer(line):
                raw = wl_match.group(1)
                target = raw.split("|")[0].strip()
                if target.startswith("论文笔记/"):
                    note_name = target[len("论文笔记/"):]
                else:
                    note_name = target
                if note_name in note_years and note_years[note_name] != current_year:
                    err("L2", moc_file.relative_to(VAULT).as_posix(),
                        f"[[{target}]] 在 '### {current_year}' section 下,但 frontmatter year={note_years[note_name]}。"
                        f"请移至 '### {note_years[note_name]}' section")


# L3: Duplicate wikilinks within same MOC
def check_l3():
    moc_dir = DIRS["moc"]
    if not moc_dir.exists():
        return
    for moc_file in sorted(moc_dir.glob("*.md")):
        content = read_content(moc_file)
        seen: dict[str, list[int]] = defaultdict(list)
        for line_num, line in enumerate(content.split("\n"), 1):
            for m in WIKILINK_RE.finditer(line):
                raw = m.group(1)
                target = raw.split("|")[0].strip().lower()
                seen[target].append(line_num)
        for target, lines in seen.items():
            if len(lines) > 1:
                line_list = ", ".join(str(l) for l in lines)
                err("L3", moc_file.relative_to(VAULT).as_posix(),
                    f"[[{target}]] 出现 {len(lines)} 次 (行 {line_list})。请保留一处,删除重复")


# L4: Concept page title/aliases overlap detection
def check_l4():
    concept_dir = DIRS["concepts"]
    if not concept_dir.exists():
        return
    pages: list[tuple[Path, str, list[str]]] = []
    for f in sorted(concept_dir.glob("*.md")):
        fm = parse_frontmatter(f)
        if fm:
            title = str(fm.get("title", f.stem)).lower().strip()
            aliases_raw = fm.get("aliases", [])
            if isinstance(aliases_raw, list):
                aliases = [str(a).lower().strip() for a in aliases_raw]
            else:
                aliases = [str(aliases_raw).lower().strip()]
            pages.append((f, title, aliases))

    for i, (f1, t1, a1) in enumerate(pages):
        names1 = {t1} | set(a1)
        for j in range(i + 1, len(pages)):
            f2, t2, a2 = pages[j]
            names2 = {t2} | set(a2)
            overlap = names1 & names2
            overlap.discard("")
            if overlap:
                err("L4", f1.relative_to(VAULT).as_posix(),
                    f"与 {f2.relative_to(VAULT).as_posix()} 存在名称重叠: {overlap}。"
                    f"请考虑合并,把一方内容迁移到另一方,并执行 [lifecycle/merge]")


# L5: Deep/repro note MOC coverage
def check_l5():
    moc_dir = DIRS["moc"]
    if not moc_dir.exists():
        return
    moc_targets: set[str] = set()
    for moc_file in sorted(moc_dir.glob("*.md")):
        content = read_content(moc_file)
        for m in WIKILINK_RE.finditer(content):
            raw = m.group(1)
            target = raw.split("|")[0].strip()
            if target.startswith("论文笔记/"):
                moc_targets.add(target[len("论文笔记/"):])
            else:
                moc_targets.add(target)

    for f in collect_md_files(DIRS["notes"]):
        fm = parse_frontmatter(f)
        if fm and fm.get("tier") in ("deep", "repro"):
            name = f.stem
            if name not in moc_targets:
                err("L5", f.relative_to(VAULT).as_posix(),
                    f"(tier={fm['tier']}) 未被任何 MOC 收录。"
                    f"请检查其 tags 并在相关 MOC 的对应年份 section 中添加条目")


# L6: Review callout existence
def check_l6():
    for f in collect_md_files(DIRS["notes"]):
        fm = parse_frontmatter(f)
        if fm and fm.get("tier") in ("deep", "repro"):
            content = read_content(f)
            if "[!review]" not in content:
                err("L6", f.relative_to(VAULT).as_posix(),
                    f"(tier={fm['tier']}) 缺少审阅 callout。"
                    f"请运行审阅或手动添加 '> [!review]' 区块")


def check_l6_single(path: Path):
    fm = parse_frontmatter(path)
    if fm and fm.get("tier") in ("deep", "repro"):
        content = read_content(path)
        if "[!review]" not in content:
            err("L6", path.relative_to(VAULT).as_posix(),
                f"(tier={fm['tier']}) 缺少审阅 callout。"
                f"请运行审阅或手动添加 '> [!review]' 区块")


# L7: source field PDF existence
def check_l7():
    for f in collect_md_files(DIRS["notes"]):
        fm = parse_frontmatter(f)
        if fm:
            source = fm.get("source", "")
            if isinstance(source, str) and source.startswith("Sources/") and source.endswith(".pdf"):
                pdf_path = VAULT / source
                if not pdf_path.exists():
                    err("L7", f.relative_to(VAULT).as_posix(),
                        f"source 指向 '{source}' 但文件不存在。"
                        f"请下载 PDF 到 {source} 或更新 source 字段")


def check_l7_single(path: Path):
    fm = parse_frontmatter(path)
    if fm:
        source = fm.get("source", "")
        if isinstance(source, str) and source.startswith("Sources/") and source.endswith(".pdf"):
            pdf_path = VAULT / source
            if not pdf_path.exists():
                err("L7", path.relative_to(VAULT).as_posix(),
                    f"source 指向 '{source}' 但文件不存在。"
                    f"请下载 PDF 到 {source} 或更新 source 字段")


# L8: Wikilink target file existence
def check_l8():
    content_dirs = [DIRS["notes"], DIRS["concepts"], DIRS["models"],
                    DIRS["datasets"], DIRS["tasks"], DIRS["moc"]]
    for f in collect_md_files(*content_dirs):
        content = read_content(f)
        for target, line_num in extract_wikilinks(content):
            if target.startswith("http") or target.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf")):
                continue
            resolved = resolve_wikilink(target)
            if resolved and not resolved.exists():
                err("L8", f.relative_to(VAULT).as_posix(),
                    f"(行 {line_num}) wikilink [[{target}]] 指向的文件不存在。"
                    f"请创建对应页面或修正 wikilink")


def check_l8_single(path: Path):
    content = read_content(path)
    for target, line_num in extract_wikilinks(content):
        if target.startswith("http") or target.endswith((".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf")):
            continue
        resolved = resolve_wikilink(target)
        if resolved and not resolved.exists():
            err("L8", path.relative_to(VAULT).as_posix(),
                f"(行 {line_num}) wikilink [[{target}]] 指向的文件不存在。"
                f"请创建对应页面或修正 wikilink")


# L9: CLAUDE.md vault stats accuracy
FIX_MODE = False

def compute_vault_stats() -> dict:
    stats = {}
    tiers = defaultdict(int)
    for f in collect_md_files(DIRS["notes"]):
        fm = parse_frontmatter(f)
        if fm:
            tiers[fm.get("tier", "unknown")] += 1
    stats["total_notes"] = sum(tiers.values())
    stats["tiers"] = dict(tiers)

    entity_counts = {}
    confirmed = 0
    pending = 0
    for label, d in [("概念", DIRS["concepts"]), ("模型", DIRS["models"]),
                      ("任务", DIRS["tasks"]), ("数据集", DIRS["datasets"])]:
        count = 0
        for f in collect_md_files(d):
            fm = parse_frontmatter(f)
            if fm:
                count += 1
                s = fm.get("status", "")
                if s == "confirmed":
                    confirmed += 1
                elif s == "pending-review":
                    pending += 1
        entity_counts[label] = count
    stats["entities"] = entity_counts
    stats["entity_total"] = sum(entity_counts.values())
    stats["confirmed"] = confirmed
    stats["pending"] = pending

    reviewed_notes = 0
    for f in collect_md_files(DIRS["notes"]):
        fm = parse_frontmatter(f)
        if fm and fm.get("status") == "reviewed":
            reviewed_notes += 1
    stats["reviewed_notes"] = reviewed_notes

    moc_files = list(DIRS["moc"].glob("*.md")) if DIRS["moc"].exists() else []
    stats["moc_count"] = len(moc_files)
    stats["moc_names"] = sorted(f.stem for f in moc_files)

    deep_repro = []
    for f in collect_md_files(DIRS["notes"]):
        fm = parse_frontmatter(f)
        if fm and fm.get("tier") in ("deep", "repro"):
            deep_repro.append(f)
    stats["deep_repro_count"] = len(deep_repro)

    moc_targets: set[str] = set()
    for mf in moc_files:
        content = read_content(mf)
        for m in WIKILINK_RE.finditer(content):
            t = m.group(1).split("|")[0].strip()
            if t.startswith("论文笔记/"):
                moc_targets.add(t[len("论文笔记/"):])
            else:
                moc_targets.add(t)
    covered = sum(1 for f in deep_repro if f.stem in moc_targets)
    stats["moc_coverage"] = f"{covered}/{len(deep_repro)}"

    with_review = 0
    for f in deep_repro:
        content = read_content(f)
        if "[!review]" in content:
            with_review += 1
    stats["review_coverage"] = f"{with_review}/{len(deep_repro)}"

    review_dir = VAULT / "_review"
    review_count = len(list(review_dir.glob("*-review.yml"))) if review_dir.exists() else 0
    stats["review_reports"] = review_count

    return stats


def format_vault_section(stats: dict) -> str:
    t = stats["tiers"]
    tier_parts = []
    for tier in ["deep", "repro", "enhanced-card", "card"]:
        if t.get(tier, 0) > 0:
            tier_parts.append(f"{t[tier]} {tier}")
    tier_str = " + ".join(tier_parts)

    e = stats["entities"]
    entity_parts = [f"{v} {k}" for k, v in e.items() if v > 0]
    entity_str = " + ".join(entity_parts)

    today = date.today().strftime("%Y-%m-%d")

    lines = [
        f"## 当前 vault 状态 ({today})",
        "",
        f"- 论文笔记: {stats['total_notes']} 篇 ({tier_str})",
        f"- 实体页: {stats['entity_total']} 个 ({entity_str}), 其中 {stats['confirmed']} confirmed / {stats['pending']} pending-review",
        f"- 可信层: {stats['confirmed']} confirmed 实体 + {stats['reviewed_notes']} reviewed 笔记",
        f"- MOC: {stats['moc_count']} 个",
        f"- MOC 覆盖: {stats['moc_coverage']} deep/repro",
        f"- 审阅覆盖: {stats['review_coverage']} deep/repro 有 review callout",
        f"- 审阅报告: {stats['review_reports']} 个 (_review/*.yml)",
    ]
    return "\n".join(lines)


def check_l9():
    claude_md = VAULT / "CLAUDE.md"
    if not claude_md.exists():
        err("L9", "CLAUDE.md", "文件不存在")
        return

    stats = compute_vault_stats()
    new_section = format_vault_section(stats)

    content = read_content(claude_md)
    m = re.search(r'^## 当前 vault 状态.*', content, re.MULTILINE)
    if not m:
        err("L9", "CLAUDE.md", "找不到 '## 当前 vault 状态' section。请添加此 section")
        return

    section_start = m.start()
    next_section = re.search(r'\n## [^\n]', content[section_start + 1:])
    if next_section:
        section_end = section_start + 1 + next_section.start()
    else:
        section_end = len(content)

    old_section = content[section_start:section_end].rstrip()

    new_lines = new_section.split("\n")
    mismatches = []
    for line in new_lines:
        if line.startswith("- "):
            key = line.split(":")[0].strip("- ")
            if key in old_section:
                old_line = [l for l in old_section.split("\n") if key in l]
                if old_line and old_line[0].strip() != line.strip():
                    mismatches.append(key)

    if not mismatches:
        return

    if FIX_MODE:
        remaining = content[section_end:].lstrip("\n")
        updated = content[:section_start] + new_section + "\n\n" + remaining
        claude_md.write_text(updated, encoding="utf-8")
        print(f"FIXED [L9] CLAUDE.md: vault 状态已更新 ({', '.join(mismatches)})")
    else:
        err("L9", "CLAUDE.md",
            f"vault 状态数字过时 ({', '.join(mismatches)})。"
            f"运行 python3 scripts/lint.py --check L9 --fix 自动更新")


# ---------------------------------------------------------------------------
# L10-L16: Extended checks (formerly manual)
# ---------------------------------------------------------------------------

warnings: list[str] = []

def warn(check: str, file: str, msg: str):
    warnings.append(f"WARNING [{check}] {file}: {msg}")


# L10: Orphan pages — entity pages with zero inlinks
def check_l10():
    entity_dirs = [DIRS["concepts"], DIRS["models"], DIRS["datasets"], DIRS["tasks"]]
    entity_files = collect_md_files(*entity_dirs)
    if not entity_files:
        return

    entity_names: dict[str, Path] = {}
    for f in entity_files:
        stem = f.stem
        entity_names[stem.lower()] = f

    all_md_dirs = [DIRS["notes"], DIRS["concepts"], DIRS["models"],
                   DIRS["datasets"], DIRS["tasks"], DIRS["moc"]]
    all_files = collect_md_files(*all_md_dirs)

    inlink_count: dict[str, int] = defaultdict(int)

    for f in all_files:
        content = read_content(f)
        links = extract_wikilinks(content)
        for target, _ in links:
            base = target.split("/")[-1].lower()
            if base in entity_names:
                inlink_count[base] += 1

    for name_lower, path in entity_names.items():
        fm = parse_frontmatter(path)
        if fm and fm.get("lifecycle") in ("deprecated", "merged"):
            continue
        if inlink_count[name_lower] == 0:
            err("L10", path.relative_to(VAULT).as_posix(),
                "实体页无入链(孤儿页) — 考虑合并到相关页或从 MOC/论文笔记中引用")


# L11: Stale concepts — active concept pages not updated in 90+ days
def check_l11():
    concept_files = collect_md_files(DIRS["concepts"])
    today = date.today()
    for f in concept_files:
        fm = parse_frontmatter(f)
        if not fm:
            continue
        if fm.get("lifecycle") in ("deprecated", "merged"):
            continue
        updated_str = fm.get("updated", "")
        if not updated_str or not isinstance(updated_str, str):
            continue
        try:
            parts = updated_str.replace("/", "-").split("-")
            updated_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            continue
        days_stale = (today - updated_date).days
        if days_stale > 90:
            err("L11", f.relative_to(VAULT).as_posix(),
                f"概念页超过 {days_stale} 天未更新(阈值 90 天) — 考虑用新论文刷新或标记 deprecated")


# L12: Review backlog — pending-review >= 10 or draft deep/repro >= 5
def check_l12():
    entity_dirs = [DIRS["concepts"], DIRS["models"], DIRS["datasets"], DIRS["tasks"]]
    pending_count = 0
    for f in collect_md_files(*entity_dirs):
        fm = parse_frontmatter(f)
        if fm and fm.get("status") == "pending-review":
            pending_count += 1

    draft_deep_count = 0
    for f in collect_md_files(DIRS["notes"]):
        fm = parse_frontmatter(f)
        if fm and fm.get("status") == "draft" and fm.get("tier") in ("deep", "repro"):
            draft_deep_count += 1

    if pending_count >= 10:
        err("L12", "概念库+模型库+数据集+任务库",
            f"审阅积压: {pending_count} 个实体页 pending-review(阈值 10) — 触发批量审阅或用户手动确认")
    if draft_deep_count >= 5:
        err("L12", "论文笔记",
            f"审阅积压: {draft_deep_count} 个 draft deep/repro 笔记(阈值 5) — 需审阅确认")


# L13: Trust-layer progress — confirmed / total active entities (info only)
def check_l13():
    entity_dirs = [DIRS["concepts"], DIRS["models"], DIRS["datasets"], DIRS["tasks"]]
    total_active = 0
    confirmed = 0
    for f in collect_md_files(*entity_dirs):
        fm = parse_frontmatter(f)
        if not fm:
            continue
        if fm.get("lifecycle") in ("deprecated", "merged"):
            continue
        total_active += 1
        if fm.get("status") == "confirmed":
            confirmed += 1
    if total_active > 0:
        ratio = confirmed / total_active * 100
        print(f"INFO [L13] 可信层进度: {confirmed}/{total_active} ({ratio:.1f}%) active 实体页为 confirmed")


# L14: Log completeness — paper note count vs log.md ingest entries
def check_l14():
    note_count = len(collect_md_files(DIRS["notes"]))
    log_path = VAULT / "log.md"
    if not log_path.exists():
        warn("L14", "log.md", "log.md 不存在 — 无法验证 log 完整性")
        return
    log_content = read_content(log_path)
    ingest_count = len(re.findall(r'\[ingest/', log_content))
    diff = abs(note_count - ingest_count)
    if diff > 5:
        warn("L14", "log.md",
             f"论文笔记数({note_count}) 与 log ingest 条目数({ingest_count})差异 {diff}(阈值 5) — 检查是否有遗漏的 log 条目")


# L15: MOC capacity — paper references > 40 in a single MOC
def check_l15():
    moc_files = collect_md_files(DIRS["moc"])
    for f in moc_files:
        content = read_content(f)
        paper_refs = re.findall(r'\[\[论文笔记/', content)
        count = len(paper_refs)
        if count > 40:
            warn("L15", f.relative_to(VAULT).as_posix(),
                 f"MOC 内论文引用数 {count}(阈值 40,上限 50) — 考虑拆分 sub-MOC")


# L16: Stale pending — pending-review entities not updated in 30+ days
def check_l16():
    entity_dirs = [DIRS["concepts"], DIRS["models"], DIRS["datasets"], DIRS["tasks"]]
    today = date.today()
    stale_pages = []
    for f in collect_md_files(*entity_dirs):
        fm = parse_frontmatter(f)
        if not fm:
            continue
        if fm.get("status") != "pending-review":
            continue
        if fm.get("lifecycle") in ("deprecated", "merged"):
            continue
        updated_str = fm.get("updated", "")
        if not updated_str or not isinstance(updated_str, str):
            continue
        try:
            parts = updated_str.replace("/", "-").split("-")
            updated_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
        except (ValueError, IndexError):
            continue
        days_stale = (today - updated_date).days
        if days_stale > 30:
            stale_pages.append(f.relative_to(VAULT).as_posix())

    if stale_pages:
        warn("L16", f"{len(stale_pages)} 页",
             f"{len(stale_pages)} 个实体页 pending-review 超 30 天未更新 — 建议触发批量实体页审阅")


# L17: Undigested learning_signals in review reports
_LS_ACTIONABLE_KEYS = {
    "new_issue_type", "rule_gap", "checklist_upgrade_suggestion",
    "system_upgrade_suggestion", "checks_passed_but_principle_failed",
}
_LS_SKIP_VALUES = {"", "none", "无", "false", "无新类型", "[]"}


def _parse_learning_signals(text: str) -> dict[str, str]:
    """Extract learning_signals block from a review YAML (no yaml dep)."""
    lines = text.split("\n")
    result: dict[str, str] = {}
    in_block = False
    for line in lines:
        stripped = line.rstrip()
        if re.match(r'^learning_signals\s*:', stripped):
            rest = stripped.split(":", 1)[1].strip()
            if rest in ("[]", ""):
                in_block = rest == ""
            else:
                break
            continue
        if in_block:
            if stripped == "" or (not stripped.startswith(" ") and stripped != ""):
                break
            m = re.match(r'^\s+([\w_-]+)\s*:\s*"?(.*?)"?\s*$', stripped)
            if m:
                result[m.group(1)] = m.group(2)
    return result


def check_l17():
    review_dir = VAULT / "_review"
    if not review_dir.exists():
        return

    suggestions: list[tuple[str, str, str]] = []
    for f in sorted(review_dir.glob("*.yml")):
        text = read_content(f)
        ls = _parse_learning_signals(text)
        if not ls:
            continue
        if ls.get("digested", "").lower() in ("true", "rejected"):
            continue
        for key in _LS_ACTIONABLE_KEYS:
            val = ls.get(key, "")
            if val.lower().strip('"') in _LS_SKIP_VALUES:
                continue
            suggestions.append((f.name, key, val))

    if suggestions:
        print(f"INFO [L17] 未消化 learning_signals 汇总:")
        for fname, key, val in suggestions:
            print(f"  [{fname}] {key}: {val}")
        print(f"共 {len(suggestions)} 条未消化建议。请评估后在审阅报告中标记 digested: true/rejected")


# L18: key_papers count exceeds limit
# 概念页无硬上限 (quality by entity-review)；模型/数据集/任务页上限 12
def check_l18():
    concept_dir = DIRS["concepts"]
    other_dirs = [DIRS["models"], DIRS["datasets"], DIRS["tasks"]]
    # 模型/数据集/任务页: 硬上限 12
    for f in collect_md_files(*other_dirs):
        fm = parse_frontmatter(f)
        if not fm:
            continue
        if fm.get("lifecycle") in ("deprecated", "merged"):
            continue
        kp = fm.get("key_papers", [])
        if isinstance(kp, list) and len(kp) > 12:
            err("L18", f.relative_to(VAULT).as_posix(),
                f"key_papers 有 {len(kp)} 条(上限 12) — "
                f"请精简为奠基/代表/转折级文献,多余的移到正文'相关工作'段或 MOC")
    # 概念页: 仅打印 info (不计入 error)
    for f in collect_md_files(concept_dir):
        fm = parse_frontmatter(f)
        if not fm:
            continue
        if fm.get("lifecycle") in ("deprecated", "merged"):
            continue
        kp = fm.get("key_papers", [])
        if isinstance(kp, list) and len(kp) > 30:
            print(f"INFO [L18] 概念页 {f.relative_to(VAULT).as_posix()}: key_papers {len(kp)} 条 — 建议 entity-review 时评估信噪比")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_CHECKS = {
    "L1": check_l1, "L2": check_l2, "L3": check_l3, "L4": check_l4,
    "L5": check_l5, "L6": check_l6, "L7": check_l7, "L8": check_l8,
    "L9": check_l9, "L10": check_l10, "L11": check_l11, "L12": check_l12,
    "L13": check_l13, "L14": check_l14, "L15": check_l15, "L16": check_l16,
    "L17": check_l17, "L18": check_l18,
}

PER_INGEST_CHECKS = ["L1", "L6", "L7", "L8"]


def run_full(selected: list[str] | None = None):
    checks = selected or list(ALL_CHECKS.keys())
    for name in checks:
        if name in ALL_CHECKS:
            ALL_CHECKS[name]()


def run_per_ingest(filepath: str):
    path = Path(filepath).resolve()
    if not path.exists():
        err("--", filepath, "文件不存在")
        return
    check_l1_single(path)
    check_l6_single(path)
    check_l7_single(path)
    check_l8_single(path)


def main():
    parser = argparse.ArgumentParser(description="PaperWiki vault linter")
    parser.add_argument("--per-ingest", metavar="FILE", help="Run per-ingest checks on a single file")
    parser.add_argument("--check", metavar="L1,L2,...", help="Run specific checks only (comma-separated)")
    parser.add_argument("--fix", action="store_true", help="Auto-fix fixable issues (currently: L9 vault stats)")
    args = parser.parse_args()

    global FIX_MODE
    FIX_MODE = args.fix

    if args.per_ingest:
        run_per_ingest(args.per_ingest)
    elif args.check:
        selected = [c.strip().upper() for c in args.check.split(",")]
        run_full(selected)
    else:
        run_full()

    if warnings:
        for w in warnings:
            print(w)

    if errors:
        for e in errors:
            print(e)
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s).")
        sys.exit(1)
    else:
        if warnings:
            print(f"\nNo errors. {len(warnings)} warning(s).")
        else:
            print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
