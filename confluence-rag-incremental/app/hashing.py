"""Deterministic hashes are the key to incremental ingestion."""

import hashlib


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def chunk_id(document_id: str, chunk_hash: str, occurrence: int) -> str:
    """Include occurrence so identical chunks in one page do not collide."""
    raw = f"{document_id}:{chunk_hash}:{occurrence}"
    return sha256_text(raw)
