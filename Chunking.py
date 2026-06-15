"""
Chunking.py
===========
PDF loading, table-aware chunking, and chunk persistence.

Functions
---------
load_and_chunk : open each PDF once, extract tables as atomic chunks +
                 sliding-window text chunks → List[Chunk]
save_chunks    : write per-company .txt files for inspection
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pdfplumber

from Config import CHUNK_DIR, CHUNK_OVERLAP, CHUNK_SIZE
from DataClasses import Chunk


def _chunk_single_page(
    page,
    page_num:   int,
    company:    str,
    source:     str,
    chunk_size: int = CHUNK_SIZE,
    overlap:    int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Extract one pdfplumber page into Chunk objects.

    Step 1 — tables: each table is one atomic chunk (chunk_type='table').
              Never split across the sliding window — a split row is
              unrecoverable for numerical extraction.
    Step 2 — text: full page text with table cell content removed to
              avoid double-indexing, then split with a sliding window.
    """
    chunks: List[Chunk] = []

    # ── Step 1: tables → one chunk each ───────────────────────────────────
    tables = page.extract_tables() or []
    for table in tables:
        if not table:
            continue
        rows = []
        for row in table:
            rows.append("\t".join([cell.strip() if cell else "" for cell in row]))
        table_text = "\n".join(rows)
        chunks.append(Chunk(
            text=table_text,
            page=page_num,
            company=company,
            source=source,
            chunk_type="table",
        ))

    # ── Step 2: remaining text → sliding window ────────────────────────────
    full_text = page.extract_text() or ""

    # Remove table cell content from the text to avoid double-indexing
    for table in tables:
        for row in table:
            for cell in row:
                if cell:
                    full_text = full_text.replace(cell.strip(), "", 1)

    full_text = " ".join(full_text.split())   # normalise whitespace

    if len(full_text) > 50:
        start = 0
        while start < len(full_text):
            end = start + chunk_size
            chunks.append(Chunk(
                text=full_text[start:end],
                page=page_num,
                company=company,
                source=source,
                chunk_type="text",
            ))
            start += chunk_size - overlap

    return chunks


def load_and_chunk(
    pdf_files:  Dict[str, str],
    pdf_dir:    Path,
    chunk_size: int = CHUNK_SIZE,
    overlap:    int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """
    Open each PDF once with pdfplumber, extract and chunk every page.

    Tables and text are both extracted while the file handle is open —
    this is required because pdfplumber's extract_tables() needs the live
    page object, not a pre-extracted text string.

    Pages with no extractable content (scanned/image-only) are skipped.

    Returns a flat list of Chunk objects across all companies.
    """
    all_chunks: List[Chunk] = []

    for company, filename in pdf_files.items():
        path = pdf_dir / filename
        if not path.exists():
            print(f"  WARNING: {filename} not found — skipping.")
            continue

        with pdfplumber.open(path) as pdf:
            n_pages     = len(pdf.pages)
            n_extracted = 0
            n_chunks    = 0

            for i, page in enumerate(pdf.pages):
                page_num = i + 1   # 1-based, matches pdfplumber convention

                # Skip entirely blank pages
                if not (page.extract_text() or "").strip() and not page.extract_tables():
                    continue

                page_chunks = _chunk_single_page(
                    page, page_num, company, filename, chunk_size, overlap
                )
                if page_chunks:
                    all_chunks.extend(page_chunks)
                    n_extracted += 1
                    n_chunks    += len(page_chunks)

            table_chunks = sum(1 for c in all_chunks
                               if c.company == company and c.chunk_type == "table")
            print(
                f"  {company}: {n_extracted}/{n_pages} pages → "
                f"{n_chunks} chunks ({table_chunks} table, {n_chunks - table_chunks} text)"
                f"  [{filename}]"
            )

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks: List[Chunk], chunk_dir: Path = CHUNK_DIR) -> None:
    """
    Write one .txt file per company to chunk_dir for inspection / debugging.

    Format per block:
        --- chunk_id=<N> | page=<P> | type=<table|text> | source=<filename> ---
        <chunk text>
        (blank line)
    """
    chunk_dir.mkdir(exist_ok=True)
    buckets: Dict[str, List[Chunk]] = defaultdict(list)
    for c in chunks:
        buckets[c.company].append(c)

    for company, clist in buckets.items():
        safe_name = company.lower().replace(" ", "_").replace("/", "_")
        out_path  = chunk_dir / f"{safe_name}_chunks.txt"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Chunks for: {company}\n")
            fh.write(f"# Total chunks: {len(clist)}\n\n")
            for c in clist:
                fh.write(
                    f"--- chunk_id={c.chunk_id} | page={c.page} "
                    f"| type={c.chunk_type} | source={c.source} ---\n"
                )
                fh.write(c.text)
                fh.write("\n\n")
        print(f"  Saved {len(clist):>5} chunks  →  {out_path}")

    print(f"\nAll chunk files written to ./{chunk_dir}/")