#!/usr/bin/env python3
r"""Extract key figures/tables for paper notes.

Priority follows the `evil-read-arxiv` idea:

1. Prefer original figures from arXiv source packages by parsing figure
   environments and their ``\includegraphics`` files.
2. Fall back to caption-aware PDF cropping only when original source figures
   are unavailable.
3. Tables are usually LaTeX, so caption-aware PDF cropping is still used for
   Table 1 and Table 2.
"""

from __future__ import annotations

import argparse
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

import fitz
import requests


ROOT = Path(__file__).resolve().parents[2]
NOTES_DIR = ROOT / "01_literature" / "paper_notes"
IMAGES_ROOT = NOTES_DIR / "images"


TARGETS = [
    ("figure1", ["Figure 1", "Fig. 1", "FIGURE 1", "FIG. 1"]),
    ("figure2", ["Figure 2", "Fig. 2", "FIGURE 2", "FIG. 2"]),
    ("table1", ["Table 1", "TABLE 1"]),
    ("table2", ["Table 2", "TABLE 2"]),
]

IMAGE_EXTS = [".pdf", ".png", ".jpg", ".jpeg", ".eps"]


def read_meta(note_path: Path) -> dict[str, str]:
    text = note_path.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("#").strip()
    meta = {"title": title}
    for key in ["PDF", "Paper"]:
        m = re.search(rf"^- {key}:\s*(.+)$", text, re.M)
        if m:
            meta[key.lower()] = m.group(1).strip()
    return meta


def download_pdf(url: str, path: Path) -> bool:
    try:
        resp = requests.get(url, timeout=90, allow_redirects=True)
        if resp.status_code != 200 or not resp.content:
            return False
        path.write_bytes(resp.content)
        return True
    except requests.RequestException:
        return False


def arxiv_id_from_url(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d+)", url)
    return m.group(1) if m else None


def strip_comments(tex: str) -> str:
    lines = []
    for line in tex.splitlines():
        out = []
        escaped = False
        for ch in line:
            if ch == "%" and not escaped:
                break
            out.append(ch)
            escaped = ch == "\\" and not escaped
            if ch != "\\":
                escaped = False
        lines.append("".join(out))
    return "\n".join(lines)


def extract_braced_after(text: str, start: int) -> str | None:
    left = text.find("{", start)
    if left < 0:
        return None
    depth = 0
    for i in range(left, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[left + 1:i]
    return None


def includegraphics_paths(env: str) -> list[str]:
    paths = []
    pos = 0
    while True:
        idx = env.find(r"\includegraphics", pos)
        if idx < 0:
            break
        body = extract_braced_after(env, idx)
        if body:
            paths.append(body.strip())
        pos = idx + len(r"\includegraphics")
    return paths


def find_graphic_file(source_dir: Path, graphic: str) -> Path | None:
    g = graphic.strip().strip("{}")
    if not g:
        return None
    candidate = source_dir / g
    if candidate.suffix and candidate.exists():
        return candidate
    if not candidate.suffix:
        for ext in IMAGE_EXTS:
            p = source_dir / f"{g}{ext}"
            if p.exists():
                return p
    name = Path(g).name
    for p in source_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name == name or (not Path(name).suffix and p.stem == name and p.suffix.lower() in IMAGE_EXTS):
            return p
    return None


def render_or_copy_source_image(src: Path, dst: Path) -> bool:
    suffix = src.suffix.lower()
    try:
        if suffix == ".pdf":
            doc = fitz.open(src)
            pix = doc[0].get_pixmap(dpi=220)
            pix.save(dst)
            doc.close()
            return True
        if suffix == ".eps":
            return False
        shutil.copy2(src, dst)
        return True
    except Exception:
        return False


def download_arxiv_source(arxiv_id: str, target: Path) -> bool:
    try:
        resp = requests.get(f"https://arxiv.org/e-print/{arxiv_id}", timeout=90, allow_redirects=True)
        if resp.status_code != 200 or not resp.content:
            return False
        target.write_bytes(resp.content)
        return True
    except requests.RequestException:
        return False


def extract_arxiv_source_figures(arxiv_id: str, out_dir: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as td:
        temp = Path(td)
        archive = temp / f"{arxiv_id}.tar"
        source_dir = temp / "src"
        source_dir.mkdir()
        if not download_arxiv_source(arxiv_id, archive):
            return records
        try:
            with tarfile.open(archive, "r:*") as tar:
                members = []
                for member in tar.getmembers():
                    if member.name.startswith("/") or ".." in member.name:
                        continue
                    if member.issym() or member.islnk():
                        continue
                    members.append(member)
                tar.extractall(source_dir, members=members)
        except tarfile.TarError:
            return records

        figure_envs = []
        for tex_path in sorted(source_dir.rglob("*.tex")):
            tex = strip_comments(tex_path.read_text(encoding="utf-8", errors="ignore"))
            for m in re.finditer(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", tex, re.S):
                env = m.group(1)
                caption = extract_braced_after(env, env.find(r"\caption")) if r"\caption" in env else ""
                graphics = includegraphics_paths(env)
                if graphics:
                    figure_envs.append((tex_path, caption or "", graphics))

        for fig_idx, (tex_path, caption, graphics) in enumerate(figure_envs[:2], start=1):
            graphic = graphics[0]
            src = find_graphic_file(source_dir, graphic)
            if not src:
                continue
            dst = out_dir / f"figure{fig_idx}_source_{src.stem}.png"
            if render_or_copy_source_image(src, dst):
                records.append({
                    "file": dst.name,
                    "kind": f"figure{fig_idx}",
                    "page": "source",
                    "patterns": f"arXiv source: {src.relative_to(source_dir)}",
                    "caption": re.sub(r"\s+", " ", caption).strip()[:240],
                    "priority": "arxiv-source",
                })
    return records


def caption_rect(page: fitz.Page, patterns: list[str]) -> fitz.Rect | None:
    hits: list[fitz.Rect] = []
    for pattern in patterns:
        hits.extend(page.search_for(pattern))
    if hits:
        hits.sort(key=lambda r: (r.y0, r.x0))
        return hits[0]

    # Fallback through text blocks for line breaks such as "Figure\n1".
    pat = re.compile(r"(Figure|Fig\.?|TABLE|Table)\s*\.?\s*([12])", re.I)
    for block in page.get_text("blocks"):
        text = block[4].replace("\n", " ")
        for p in patterns:
            target_num = "1" if "1" in p else "2"
            m = pat.search(text)
            if m and m.group(2) == target_num and p.lower().split()[0].replace(".", "") in m.group(1).lower():
                return fitz.Rect(block[:4])
    return None


def crop_for_caption(page: fitz.Page, rect: fitz.Rect, kind: str) -> fitz.Rect:
    page_rect = page.rect
    margin_x = 24

    if kind.startswith("figure"):
        # Most NLP papers place figure captions below the figure.
        y0 = max(page_rect.y0, rect.y0 - 360)
        y1 = min(page_rect.y1, rect.y1 + 90)
    else:
        # Tables often place captions above or immediately before the table.
        y0 = max(page_rect.y0, rect.y0 - 60)
        y1 = min(page_rect.y1, rect.y1 + 360)

    # If the caption is near the top, the object may be below the caption.
    if rect.y0 < page_rect.height * 0.22:
        y0 = max(page_rect.y0, rect.y0 - 40)
        y1 = min(page_rect.y1, rect.y1 + 420)

    return fitz.Rect(page_rect.x0 + margin_x, y0, page_rect.x1 - margin_x, y1)


def extract_pdf_key_artifacts(pdf_path: Path, out_dir: Path, skip: set[str]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return records

    try:
        found: set[str] = set(skip)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            for kind, patterns in TARGETS:
                if kind in found:
                    continue
                rect = caption_rect(page, patterns)
                if not rect:
                    continue
                clip = crop_for_caption(page, rect, kind)
                pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=clip)
                fname = f"{kind}_pdfcrop_page{page_idx + 1:02d}.png"
                pix.save(out_dir / fname)
                records.append({
                    "file": fname,
                    "kind": kind,
                    "page": str(page_idx + 1),
                    "patterns": "/".join(patterns),
                    "caption": "",
                    "priority": "pdf-caption-crop",
                })
                found.add(kind)
    finally:
        doc.close()
    return records


def extract_key_artifacts(note_path: Path, force: bool) -> list[dict[str, str]]:
    meta = read_meta(note_path)
    pdf_url = meta.get("pdf", "")
    if not pdf_url or pdf_url.startswith("未找到"):
        return []

    out_dir = IMAGES_ROOT / note_path.stem / "key_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for old in out_dir.glob("*.png"):
            old.unlink()

    records: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as td:
        arxiv_id = arxiv_id_from_url(meta.get("paper", "")) or arxiv_id_from_url(pdf_url)
        if arxiv_id:
            records.extend(extract_arxiv_source_figures(arxiv_id, out_dir))

        pdf_path = Path(td) / f"{note_path.stem}.pdf"
        if not download_pdf(pdf_url, pdf_path):
            write_index(out_dir, note_path, meta, records)
            return records
        found_kinds = {item["kind"] for item in records}
        records.extend(extract_pdf_key_artifacts(pdf_path, out_dir, found_kinds))
        write_index(out_dir, note_path, meta, records)
    print(f"{note_path.name}: {len(records)} key artifacts")
    return records


def write_index(out_dir: Path, note_path: Path, meta: dict[str, str], records: list[dict[str, str]]) -> None:
    lines = [
        f"# 关键图表索引：{meta['title']}",
        "",
        f"- Note: `{note_path.name}`",
        f"- PDF: {meta.get('pdf', '')}",
        f"- Total key artifacts: {len(records)}",
        "",
    ]
    for item in records:
        path = out_dir / item["file"]
        size_kb = path.stat().st_size / 1024
        lines.extend([
            f"## {item['kind']} - page {item['page']}",
            "",
            f"- File: `{item['file']}`",
            f"- Priority: {item.get('priority', 'unknown')}",
            f"- Source / matched caption: {item['patterns']}",
            f"- Caption text: {item.get('caption') or '(empty)'}",
            f"- Size: {size_kb:.1f} KB",
            "",
            f"![{item['file']}]({item['file']})",
            "",
        ])
    (out_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    notes = [args.note] if args.note else sorted(p for p in NOTES_DIR.glob("*.md") if p.name != "TEMPLATE.md")
    for note in notes:
        extract_key_artifacts(note, args.force)


if __name__ == "__main__":
    main()
