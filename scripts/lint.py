#!/usr/bin/env python3
"""PaperWiki vault linter — 8 mechanizable checks.

Usage:
    python3 scripts/lint.py                       # full check (L1-L8)
    python3 scripts/lint.py --per-ingest FILE      # per-ingest subset (L1,L6,L7,L8)
    python3 scripts/lint.py --check L1,L3,L5       # specific checks only

Exit code: 0 = all pass, 1 = errors found.
Each error line: ERROR [Lx] file: agent-readable fix instruction.
"""

import argparse
import os
import re
import sys
from collections import defaultdict
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


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_CHECKS = {
    "L1": check_l1, "L2": check_l2, "L3": check_l3, "L4": check_l4,
    "L5": check_l5, "L6": check_l6, "L7": check_l7, "L8": check_l8,
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
    args = parser.parse_args()

    if args.per_ingest:
        run_per_ingest(args.per_ingest)
    elif args.check:
        selected = [c.strip().upper() for c in args.check.split(",")]
        run_full(selected)
    else:
        run_full()

    if errors:
        for e in errors:
            print(e)
        print(f"\n{len(errors)} error(s) found.")
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
