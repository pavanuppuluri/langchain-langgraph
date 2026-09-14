"""Small domain models used by the indexer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PageSummary:
    document_id: str
    title: str
    version: int
    source_url: str


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    chunk_hash: str
    chunk_sequence: int
    chunk_text: str
