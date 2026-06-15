"""
Chunking.py
===========
PDF loading and table-aware chunking.

Functions
---------
load_and_chunk  : open each PDF once → tables as atomic chunks + sliding-window text
save_chunks     : write per-company .txt files to chunks/ for inspection
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pdfplumber

from Config import CHUNK_DIR, CHUNK_OVERLAP, CHUNK_SIZE
from DataClasses import Chunk


def _chunk_single_page(
    page, page_num: int, company: str, source: str,
    chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """Extract one pdfplumber page into Chunk objects.

    Tables → one atomic chunk each (splitting mid-row loses numbers).
    Remaining prose → sliding-window chunks.
    """
    chunks: List[Chunk] = []

    # Step 1: tables as atomic chunks
    tables = page.extract_tables() or []
    for table in tables:
        if not table:
            continue
        rows = ["\t".join([cell.strip() if cell else "" for cell in row]) for row in table]
        chunks.append(Chunk(text="\n".join(rows), page=page_num,
                            company=company, source=source, chunk_type="table"))

    # Step 2: remaining prose → sliding window
    full_text = page.extract_text() or ""
    for table in tables:           # remove table cells to avoid double-indexing
        for row in table:
            for cell in row:
                if cell:
                    full_text = full_text.replace(cell.strip(), "", 1)
    full_text = " ".join(full_text.split())

    if len(full_text) > 50:
        start = 0
        while start < len(full_text):
            chunks.append(Chunk(text=full_text[start:start + chunk_size], page=page_num,
                                company=company, source=source, chunk_type="text"))
            start += chunk_size - overlap

    return chunks


def load_and_chunk(
    pdf_files: Dict[str, str], pdf_dir: Path,
    chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    """Open each PDF once and extract all chunks.

    Loading and chunking are merged into a single pass because
    pdfplumber's extract_tables() requires the live page object.
    Sequential chunk_ids are assigned at the end across the full corpus.
    """
    all_chunks: List[Chunk] = []

    for company, filename in pdf_files.items():
        path = pdf_dir / filename
        if not path.exists():
            print(f"  WARNING: {filename} not found — skipping.")
            continue

        with pdfplumber.open(path) as pdf:
            n_pages = len(pdf.pages)
            n_extracted = n_chunks = 0

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                if not (page.extract_text() or "").strip() and not page.extract_tables():
                    continue
                page_chunks = _chunk_single_page(page, page_num, company, filename, chunk_size, overlap)
                if page_chunks:
                    all_chunks.extend(page_chunks)
                    n_extracted += 1
                    n_chunks    += len(page_chunks)

            table_count = sum(1 for c in all_chunks if c.company == company and c.chunk_type == "table")
            print(f"  {company}: {n_extracted}/{n_pages} pages → "
                  f"{n_chunks} chunks ({table_count} table, {n_chunks - table_count} text)  [{filename}]")

    for i, c in enumerate(all_chunks):
        c.chunk_id = i

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks


def save_chunks(chunks: List[Chunk], chunk_dir: Path = CHUNK_DIR) -> None:
    """Write one .txt file per company to chunk_dir for inspection."""
    chunk_dir.mkdir(exist_ok=True)
    buckets: Dict[str, List[Chunk]] = defaultdict(list)
    for c in chunks:
        buckets[c.company].append(c)

    for company, clist in buckets.items():
        safe_name = company.lower().replace(" ", "_").replace("/", "_")
        out_path  = chunk_dir / f"{safe_name}_chunks.txt"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(f"# Chunks for: {company}  (total: {len(clist)})\n\n")
            for c in clist:
                fh.write(f"--- id={c.chunk_id} | p.{c.page} | {c.chunk_type} | {c.source} ---\n")
                fh.write(c.text + "\n\n")
        print(f"  Saved {len(clist):>5} chunks  →  {out_path}")

    print(f"\nAll chunk files written to ./{chunk_dir}/")
