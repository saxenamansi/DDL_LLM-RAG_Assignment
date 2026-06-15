"""
DataClasses.py
===============
Dataclasses shared across the RAG pipeline.
Kept separate so every other module can import them
without pulling in heavy dependencies.
"""

from dataclasses import dataclass


@dataclass
class PageRecord:
    """Raw text from a single PDF page with provenance metadata."""
    company: str
    source:  str   # filename
    page:    int   # 1-based page number
    text:    str

@dataclass
class Chunk:
    """Fixed-size text window with full provenance."""
    chunk_id: int
    company:  str
    source:   str
    page:     int
    text:     str
    chunk_type: str = "text"   # "text" or "table"
