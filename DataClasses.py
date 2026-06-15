"""
DataClasses.py
==============
Shared dataclasses for the ESG RAG pipeline.
"""

from dataclasses import dataclass


@dataclass
class PageRecord:
    """Raw text from a single PDF page with provenance metadata."""
    company: str
    source:  str
    page:    int   # 1-based
    text:    str


@dataclass
class Chunk:
    """
    Text chunk with provenance. chunk_type is 'table' or 'text'.
    chunk_id is assigned sequentially by load_and_chunk; default 0 is a placeholder.
    """
    company:    str
    source:     str
    page:       int
    text:       str
    chunk_id:   int = 0
    chunk_type: str = "text"
