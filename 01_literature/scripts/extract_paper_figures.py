#!/usr/bin/env python3
"""Extract or render figures for literature notes.

The script is adapted for this repository from the workflow idea in
`juliye2025/evil-read-arxiv`: prefer arXiv source images, then PDF embedded
images, and finally rendered PDF pages as a robust fallback.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import fitz
import requests


ROOT = Path(__file__).resolve().parents[2]
NOTES_DIR = ROOT / "01_literature" / "paper_notes"
IMAGES_ROOT = NOTES_DIR / "images"


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def parse_note(note_path: Path) -> dict[str, str]:
    text = note_path.read_text(encoding="utf-8")
    title = text.splitlines()[0].lstrip("#").strip()
    result = {"title": title}
    for key in ["Paper", "PDF", "Code", "Dataset"]:
        m = re.search(rf"^- {key}:\s*(.+)$", text, re.M)
        if m:
            result[key.lower()] = m.group(1).strip()
    return result


def arxiv_id_from_url(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d+)", url)
    return m.group(1) if m else None


def download(url: str, path: Path) -> bool:
    try:
        resp = requests.get(url, timeout=90, allow_redirects=True)
        if resp.status_code != 200 or not resp.content:
            return False
        path.write_bytes(resp.content)
        return True
    except requests.RequestException:
        return False


def extract_arxiv_source(arxiv_id: str, out_dir: Path, max_images: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory() as td:
        temp = Path(td)
        tar_path = temp / f"{arxiv_id}.tar.gz"
        if not download(f"https://arxiv.org/e-print/{arxiv_id}", tar_path):
            return records
        try:
            with tarfile.open(tar_path, "r:*") as tar:
                members = []
                for member in tar.getmembers():
                    if member.name.startswith("/") or ".." in member.name:
                        continue
                    if member.issym() or member.islnk():
                        continue
                    members.append(member)
                tar.extractall(temp, members=members)
        except tarfile.TarError:
            return records

        candidates = []
        for p in temp.rglob("*"):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext not in {".png", ".jpg", ".jpeg", ".pdf"}:
                continue
            low = p.name.lower()
            if any(x in low for x in ["logo", "icon", "license"]):
                continue
            candidates.append(p)

        for i, src in enumerate(candidates[:max_images], start=1):
            if src.suffix.lower() == ".pdf":
                try:
                    doc = fitz.open(src)
                    page = doc[0]
                    pix = page.get_pixmap(dpi=180)
                    name = f"source_{i:02d}_{safe_name(src.stem)}.png"
                    dst = out_dir / name
                    pix.save(dst)
                    doc.close()
                except Exception:
                    continue
            else:
                name = f"source_{i:02d}_{safe_name(src.name)}"
                dst = out_dir / name
                shutil.copy2(src, dst)
            records.append({"file": dst.name, "source": "arxiv-source", "note": src.name})
    return records


def extract_pdf_images(pdf_path: Path, out_dir: Path, max_images: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return records
    seen = set()
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            for img_idx, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                if xref in seen:
                    continue
                seen.add(xref)
                try:
                    base = doc.extract_image(xref)
                except Exception:
                    continue
                width = base.get("width", 0)
                height = base.get("height", 0)
                data = base.get("image", b"")
                if width < 240 or height < 160 or len(data) < 8_000:
                    continue
                ext = base.get("ext", "png")
                name = f"pdf_p{page_idx + 1:02d}_{img_idx:02d}.{ext}"
                (out_dir / name).write_bytes(data)
                records.append({"file": name, "source": "pdf-extraction", "note": f"page {page_idx + 1}"})
                if len(records) >= max_images:
                    return records
    finally:
        doc.close()
    return records


def render_pages(pdf_path: Path, out_dir: Path, max_pages: int) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return records
    try:
        for page_idx in range(min(max_pages, len(doc))):
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=140)
            name = f"page_{page_idx + 1:02d}.png"
            pix.save(out_dir / name)
            records.append({"file": name, "source": "pdf-page-render", "note": f"page {page_idx + 1}"})
    finally:
        doc.close()
    return records


def pdf_filename(note_stem: str, pdf_url: str) -> str:
    parsed = urlparse(pdf_url)
    suffix = Path(parsed.path).suffix or ".pdf"
    return f"{note_stem}{suffix if suffix == '.pdf' else '.pdf'}"


def process_note(note_path: Path, max_images: int, max_pages: int, force: bool, always_render_pages: bool) -> None:
    meta = parse_note(note_path)
    pdf_url = meta.get("pdf")
    if not pdf_url or pdf_url.startswith("未找到"):
        return
    note_stem = note_path.stem
    out_dir = IMAGES_ROOT / note_stem
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, str]] = []
    arxiv_id = arxiv_id_from_url(pdf_url)
    if arxiv_id:
        records.extend(extract_arxiv_source(arxiv_id, out_dir, max_images))

    pdf_path = out_dir / pdf_filename(note_stem, pdf_url)
    if not pdf_path.exists():
        download(pdf_url, pdf_path)

    if pdf_path.exists() and len(records) < max_images:
        records.extend(extract_pdf_images(pdf_path, out_dir, max_images - len(records)))
    if pdf_path.exists() and not records:
        records.extend(render_pages(pdf_path, out_dir, max_pages))
    elif pdf_path.exists() and always_render_pages:
        existing = {item["file"] for item in records}
        page_records = [item for item in render_pages(pdf_path, out_dir, max_pages) if item["file"] not in existing]
        records.extend(page_records)

    # Keep PDFs out of the repo; only extracted images and index are needed.
    if pdf_path.exists():
        pdf_path.unlink()

    index = out_dir / "index.md"
    lines = [
        f"# 图片索引：{meta['title']}",
        "",
        f"- Note: `{note_path.name}`",
        f"- PDF: {pdf_url}",
        f"- Total images: {len(records)}",
        "",
    ]
    for item in records:
        size_kb = (out_dir / item["file"]).stat().st_size / 1024
        lines.extend([
            f"## {item['file']}",
            "",
            f"- Source: {item['source']}",
            f"- Note: {item['note']}",
            f"- Size: {size_kb:.1f} KB",
            "",
            f"![{item['file']}]({item['file']})",
            "",
        ])
    index.write_text("\n".join(lines), encoding="utf-8")
    print(f"{note_path.name}: {len(records)} images -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--note", type=Path, help="single note path")
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--always-render-pages", action="store_true")
    args = parser.parse_args()

    notes = [args.note] if args.note else sorted(p for p in NOTES_DIR.glob("*.md") if p.name != "TEMPLATE.md")
    for note in notes:
        process_note(note, args.max_images, args.max_pages, args.force, args.always_render_pages)


if __name__ == "__main__":
    main()
