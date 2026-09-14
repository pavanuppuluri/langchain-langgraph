"""Incremental Confluence -> chunks -> embeddings synchronization."""

from __future__ import annotations

from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.confluence import ConfluenceClient, storage_html_to_text
from app.database import MetadataStore
from app.hashing import chunk_id, sha256_text
from app.models import ChunkRecord


class IncrementalIndexer:
    def __init__(self, settings: Settings, confluence: ConfluenceClient, metadata: MetadataStore, vector_store):
        self.settings = settings
        self.confluence = confluence
        self.metadata = metadata
        self.vector_store = vector_store
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def sync(self) -> dict[str, int]:
        self.metadata.initialize()
        stats = Counter()
        seen_ids: set[str] = set()

        for page in self.confluence.iter_page_summaries():
            seen_ids.add(page.document_id)
            existing = self.metadata.get_document(page.document_id)

            # Fast path: version is unchanged, so there is no reason to download the body.
            if existing and existing[1] == page.version and existing[2] == self.settings.chunking_version:
                stats["skipped"] += 1
                continue

            self._sync_page(page, stats)

        # Full-space reconciliation handles pages deleted/removed from the configured space.
        for deleted_id in self.metadata.indexed_document_ids() - seen_ids:
            old_chunk_ids = self.metadata.delete_document(deleted_id)
            if old_chunk_ids:
                self.vector_store.delete(ids=old_chunk_ids)
            stats["deleted_documents"] += 1
            stats["deleted_chunks"] += len(old_chunk_ids)

        return dict(stats)

    def _sync_page(self, page, stats: Counter) -> None:
        html = self.confluence.get_page_storage_html(page.document_id)
        text = storage_html_to_text(html)
        document_hash = sha256_text(text)
        existing = self.metadata.get_document(page.document_id)

        # Version changed but normalized content did not. Avoid embedding again.
        if (
            existing
            and existing[0] == document_hash
            and existing[2] == self.settings.chunking_version
            and existing[3] == self.settings.embedding_model
        ):
            self.metadata.upsert_document(
                page.document_id,
                page.title,
                page.source_url,
                document_hash,
                page.version,
                self.settings.chunking_version,
                self.settings.embedding_model,
            )
            stats["metadata_only"] += 1
            return

        raw_chunks = self.splitter.split_text(text)
        old_chunks = self.metadata.get_chunks(page.document_id)
        old_by_hash = {}
        for old_chunk_id, old_hash in old_chunks:
            old_by_hash.setdefault(old_hash, []).append(old_chunk_id)

        new_records: list[ChunkRecord] = []
        new_documents: list[Document] = []
        new_ids: list[str] = []
        seen_occurrences: dict[str, int] = {}
        reused_old_ids: set[str] = set()

        for sequence, chunk_text in enumerate(raw_chunks):
            digest = sha256_text(chunk_text)
            occurrence = seen_occurrences.get(digest, 0)
            seen_occurrences[digest] = occurrence + 1

            candidates = old_by_hash.get(digest, [])
            if occurrence < len(candidates):
                reused_old_ids.add(candidates[occurrence])
                continue

            cid = chunk_id(page.document_id, digest, occurrence)
            record = ChunkRecord(cid, page.document_id, digest, sequence, chunk_text)
            new_records.append(record)
            new_ids.append(cid)
            new_documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "document_id": page.document_id,
                        "title": page.title,
                        "source": page.source_url,
                        "space": self.settings.confluence_space_key,
                        "version": page.version,
                        "chunk_id": cid,
                        "chunk_hash": digest,
                        "chunking_version": self.settings.chunking_version,
                        "embedding_model": self.settings.embedding_model,
                        "chunk_sequence": sequence,
                    },
                )
            )

        old_ids = {cid for cid, _ in old_chunks}
        ids_to_delete = list(old_ids - reused_old_ids - set(new_ids))

        # Add new vectors first. If embedding fails, the old searchable state remains intact.
        if new_documents:
            self.vector_store.add_documents(new_documents, ids=new_ids)
            self.metadata.save_chunks(new_records, self.settings.embedding_model)
            stats["embedded_chunks"] += len(new_documents)

        if ids_to_delete:
            self.vector_store.delete(ids=ids_to_delete)
            self.metadata.delete_chunks(ids_to_delete)
            stats["deleted_chunks"] += len(ids_to_delete)

        self.metadata.upsert_document(
            page.document_id,
            page.title,
            page.source_url,
            document_hash,
            page.version,
            self.settings.chunking_version,
            self.settings.embedding_model,
        )
        stats["updated_documents"] += 1
