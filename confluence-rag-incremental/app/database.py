"""Relational metadata store.

Postgres holds the synchronization state. PGVector holds searchable vectors.
Keeping this metadata separately makes incremental decisions explicit.
"""

from __future__ import annotations

from typing import Iterable

import psycopg

from app.models import ChunkRecord


def raw_psycopg_url(sqlalchemy_url: str) -> str:
    return sqlalchemy_url.replace("postgresql+psycopg://", "postgresql://", 1)


class MetadataStore:
    def __init__(self, postgres_url: str):
        self.url = raw_psycopg_url(postgres_url)

    def _connect(self):
        return psycopg.connect(self.url)

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_index (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    document_hash TEXT NOT NULL,
                    confluence_version INTEGER NOT NULL,
                    chunking_version TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    indexed_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_index (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES document_index(document_id) ON DELETE CASCADE,
                    chunk_hash TEXT NOT NULL,
                    chunk_sequence INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding_model TEXT NOT NULL
                )
                """
            )

    def get_document(self, document_id: str):
        with self._connect() as conn:
            return conn.execute(
                "SELECT document_hash, confluence_version, chunking_version, embedding_model "
                "FROM document_index WHERE document_id = %s",
                (document_id,),
            ).fetchone()

    def get_chunks(self, document_id: str) -> list[tuple[str, str]]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT chunk_id, chunk_hash FROM chunk_index WHERE document_id = %s",
                (document_id,),
            ).fetchall()

    def upsert_document(
        self,
        document_id: str,
        title: str,
        source_url: str,
        document_hash: str,
        version: int,
        chunking_version: str,
        embedding_model: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO document_index
                    (document_id, title, source_url, document_hash, confluence_version,
                     chunking_version, embedding_model)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (document_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    source_url = EXCLUDED.source_url,
                    document_hash = EXCLUDED.document_hash,
                    confluence_version = EXCLUDED.confluence_version,
                    chunking_version = EXCLUDED.chunking_version,
                    embedding_model = EXCLUDED.embedding_model,
                    indexed_at = CURRENT_TIMESTAMP
                """,
                (document_id, title, source_url, document_hash, version, chunking_version, embedding_model),
            )

    def save_chunks(self, chunks: Iterable[ChunkRecord], embedding_model: str) -> None:
        with self._connect() as conn:
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT INTO chunk_index
                        (chunk_id, document_id, chunk_hash, chunk_sequence, chunk_text, embedding_model)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        chunk_hash = EXCLUDED.chunk_hash,
                        chunk_sequence = EXCLUDED.chunk_sequence,
                        chunk_text = EXCLUDED.chunk_text,
                        embedding_model = EXCLUDED.embedding_model
                    """,
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.chunk_hash,
                        chunk.chunk_sequence,
                        chunk.chunk_text,
                        embedding_model,
                    ),
                )

    def delete_chunks(self, chunk_ids: Iterable[str]) -> None:
        ids = list(chunk_ids)
        if not ids:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM chunk_index WHERE chunk_id = ANY(%s)", (ids,))

    def delete_document(self, document_id: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id FROM chunk_index WHERE document_id = %s",
                (document_id,),
            ).fetchall()
            conn.execute("DELETE FROM document_index WHERE document_id = %s", (document_id,))
            return [row[0] for row in rows]

    def indexed_document_ids(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT document_id FROM document_index").fetchall()
            return {row[0] for row in rows}
