#!/usr/bin/env python3
"""Extract key figures/tables by locating captions in paper PDFs.

This is a second-pass extractor for reading notes. The first-pass extractor
collects whatever images are available; this script focuses on meaningful
paper artifacts such as Figure 1, Figure 2, Table 1, and Table 2.
"""

from __future__ import annotations

import argparse
import re
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


def union_rect(rects: list[fitz.Rect]) -> fitz.Rect:
    rect = fitz.Rect(rects[0])
    for r in rects[1:]:
        rect |= r
    return rect


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
        pdf_path = Path(td) / f"{note_path.stem}.pdf"
        if not download_pdf(pdf_url, pdf_path):
            return records
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return records

        try:
            found: set[str] = set()
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
                    fname = f"{kind}_page{page_idx + 1:02d}.png"
                    pix.save(out_dir / fname)
                    records.append({
                        "file": fname,
                        "kind": kind,
                        "page": str(page_idx + 1),
                        "patterns": "/".join(patterns),
                    })
                    found.add(kind)
            write_index(out_dir, note_path, meta, records)
        finally:
            doc.close()
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
            f"- Matched caption: {item['patterns']}",
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
