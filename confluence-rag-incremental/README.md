# Incremental Confluence RAG with LangChain + PostgreSQL/PGVector

A production-oriented learning repository showing how to build a **Confluence knowledge search using RAG without re-embedding the entire knowledge base whenever documentation changes**.

## The problem

Imagine a Confluence space with 10,000 pages.

- Day 1: index all 10,000 pages.
- Day 2: one paragraph on one page changes.
- A naive RAG pipeline downloads all 10,000 pages, chunks them again, calls the embedding API again, and replaces the vector store.

That is wasteful.

This repository treats the vector database as a **derived search index** and Confluence as the **source of truth**.

The synchronization flow is:

```text
Confluence
   |
   | list page metadata: id + version
   v
Change detector
   |
   | unchanged? --------------------> SKIP
   |
   | changed/new
   v
Fetch only changed page body
   |
   v
Normalize HTML -> text
   |
   v
Document SHA-256 hash
   |
   v
Split into chunks
   |
   v
Chunk SHA-256 hashes
   |
   +---- unchanged chunk ----> REUSE existing vector
   |
   +---- new chunk ----------> EMBED + store
   |
   +---- removed chunk ------> DELETE vector
   v
PGVector
   |
   v
Retriever -> Chat model -> Answer + source URL
```

## Why this design?

### 1. Page version is the cheap first filter

The indexer first calls Confluence for page metadata. It asks for `id`, `title`, and `version`, but not the page body.

If Confluence says page `12345` is still version `17` and our metadata table already says version `17`, there is nothing to process.

That means we do **not** download the page and we do **not** create embeddings.

### 2. Document hash is a safety check

A page can have a new Confluence version but normalize to the same text. We calculate:

```python
document_hash = sha256_text(text)
```

The hash gives us content identity independent of the Confluence version number.

### 3. Chunk hashes prevent unnecessary embeddings

Suppose a page contains 100 chunks and only chunk 37 changes.

The indexer calculates a hash for every chunk:

```python
chunk_hash = sha256_text(chunk_text)
```

Existing chunk hashes are reused. Only the new chunk is embedded.

### 4. Duplicate chunks need an occurrence number

Two chunks can contain exactly the same text. A pure `document_id + chunk_hash` ID would collide.

This repository therefore uses:

```python
chunk_id(document_id, chunk_hash, occurrence)
```

The occurrence is only an identity mechanism; it is not used as the primary change detector.

### 5. Metadata and vectors have different jobs

`document_index` and `chunk_index` are relational synchronization state.

PGVector is the semantic search index.

That separation makes it much easier to answer questions such as:

- What Confluence version is indexed?
- Which chunks belong to this page?
- Which chunk hashes changed?
- Which vector IDs must be deleted?

## Repository structure

```text
confluence-rag-incremental/
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── confluence.py
│   ├── database.py
│   ├── hashing.py
│   ├── indexer.py
│   ├── main.py
│   ├── models.py
│   ├── search.py
│   └── vector_store.py
├── tests/
│   └── test_hashing.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Code walkthrough

### `app/confluence.py`

The important optimization is that `iter_page_summaries()` does **not** request `body.storage`.

```python
"expand": "version"
```

The indexer therefore gets cheap metadata first.

Only after a page is known to be new or changed does it call:

```python
get_page_storage_html(document_id)
```

This is the difference between **checking whether work is needed** and **doing the expensive work**.

Confluence Cloud supports pagination, so the repository loops over `start`/`limit` rather than assuming one response contains the whole space.

### `app/database.py`

The two tables form our synchronization ledger.

`document_index` stores page-level state:

```text
document_id
confluence_version
document_hash
chunking_version
embedding_model
indexed_at
```

`chunk_index` stores chunk-level state:

```text
chunk_id
document_id
chunk_hash
chunk_sequence
chunk_text
embedding_model
```

Why store `chunking_version` and `embedding_model`?

Because changing either can invalidate previously generated vectors. For example, moving from `text-embedding-3-small` to another embedding model should trigger re-indexing rather than silently mixing incompatible vector representations.

### `app/indexer.py`

This is the heart of the repository.

The first fast path is:

```python
if existing and existing[1] == page.version and existing[2] == settings.chunking_version:
    stats["skipped"] += 1
    continue
```

This means an unchanged page is skipped **before its body is downloaded**.

Next, the body is fetched and normalized. If its content hash is unchanged, the index metadata is updated but embeddings are not recreated.

Then the page is chunked.

For each chunk:

1. calculate `chunk_hash`
2. check whether an old chunk with that hash exists
3. reuse it if possible
4. otherwise create a new LangChain `Document`
5. add only new documents to PGVector

The key call is:

```python
vector_store.add_documents(new_documents, ids=new_ids)
```

Because `new_documents` contains only newly required chunks, the embedding API is not called for unchanged chunks.

### `app/vector_store.py`

The repository uses LangChain's dedicated `langchain-postgres` PGVector integration.

```python
PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=postgres_url,
    use_jsonb=True,
)
```

The metadata stored with every vector includes the Confluence page ID, title, URL, version, chunk hash, and chunking/embedding versions.

### `app/search.py`

The search path is deliberately simple:

```text
question
   -> vector similarity search
   -> top K Confluence chunks
   -> context prompt
   -> chat model
   -> answer
```

The prompt tells the model to use only retrieved Confluence context and to say when the answer cannot be found.

For production, this is where you can add reranking, score thresholds, MMR, ACL filtering, citations, and conversation-aware retrieval.

## What happens when one paragraph changes in a 10,000-page KB?

Assume:

- 10,000 Confluence pages
- 100 chunks per page on average
- 1,000,000 indexed chunks
- one paragraph changes on page `P123`

On the next sync:

1. Confluence metadata for all pages is scanned.
2. 9,999 pages have the same version and are skipped without body downloads.
3. `P123` has a newer version, so its body is fetched.
4. Its 100 chunks are recalculated.
5. 99 chunk hashes match existing hashes.
6. 1 chunk hash is new.
7. Only that chunk is embedded.
8. Removed chunks, if any, are deleted from PGVector.

Conceptually:

```text
Old page P123
  chunk-1  same
  chunk-2  same
  ...
  chunk-37 OLD
  ...
  chunk-100 same

New page P123
  chunk-1  same       -> reuse
  chunk-2  same       -> reuse
  ...
  chunk-37 NEW        -> embed
  ...
  chunk-100 same      -> reuse
```

### Important chunking caveat

If a change is large enough to shift chunk boundaries, more than one chunk can change. That is expected. The optimization is still much better than blindly re-embedding every page in the knowledge base.

For very structured Confluence pages, an even better production strategy is to parse by headings/sections first and then chunk each section. That makes changes more local and preserves semantic context.

## Setup

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd confluence-rag-incremental
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Fill in:

```text
CONFLUENCE_BASE_URL
CONFLUENCE_EMAIL
CONFLUENCE_API_TOKEN
CONFLUENCE_SPACE_KEY
OPENAI_API_KEY
```

For Confluence Cloud, the API supports direct REST authentication with Basic Auth using the user identity and API token. Use a dedicated least-privilege service identity in production.

### 4. Start PostgreSQL + pgvector

```bash
docker compose up -d
```

Check:

```bash
docker ps
```

### 5. Run the initial index

```bash
python -m app.main index
```

Example output:

```text
Indexing complete:
  embedded_chunks: 1248
  updated_documents: 42
```

### 6. Ask a question

```bash
python -m app.main search "How does the Product Foundation platform handle SKU limits?"
```

### 7. Run indexing again

```bash
python -m app.main index
```

If nothing changed, you should see something similar to:

```text
Indexing complete:
  skipped: 42
```

## Test

```bash
pytest -q
```

## Production improvements

This repository is intentionally small enough to understand. Before using it as an enterprise production service, add:

### Event-driven updates

Instead of periodically scanning every page, use Confluence webhooks/events to enqueue changed page IDs. The worker then fetches only those pages.

### ACL-aware retrieval

This is critical for enterprise RAG. A vector store containing restricted Confluence content must not return a chunk to a user who could not access the original page.

Typical approaches include indexing ACL metadata and applying authorization filters before retrieval, or using a retrieval service that validates Confluence permissions.

### Better Confluence parsing

The sample converts storage HTML to text. Production parsing should preserve:

- headings
- tables
- code blocks
- links
- lists
- attachments
- macros

### Retry and rate limiting

Confluence and embedding APIs can throttle requests. Add exponential backoff, bounded retries, and a dead-letter queue for permanently failed pages.

### Observability

Track:

```text
pages_seen
pages_skipped
pages_changed
chunks_reused
chunks_embedded
chunks_deleted
embedding_failures
confluence_failures
index_duration
```

### Atomicity

The sample adds new vectors before deleting obsolete vectors so a failed embedding operation does not immediately destroy the previous searchable state. A production implementation can go further with versioned indexes or a staging collection followed by an atomic promotion.

### Embedding cache

For large systems, cache embeddings by:

```text
SHA-256(chunk_text)
+ embedding_model
+ embedding_model_version
```

Then the same text appearing in multiple pages can potentially reuse the same embedding.

## Architecture interview answer

If asked in an interview:

> "I would not treat RAG ingestion as a batch job. I would treat it as a synchronization pipeline. Confluence remains the source of truth and the vector store is a derived search index. First I compare the source page version with the indexed version. If it has not changed, I skip the page completely. If it changed, I fetch only that page, calculate a document hash, split it into chunks, and compare chunk hashes with the previous state. Unchanged chunks reuse their existing vectors, new chunks are embedded, and deleted chunks are removed. I also version the chunking strategy and embedding model so configuration changes can trigger controlled re-indexing. For enterprise Confluence, I would additionally make retrieval ACL-aware so the RAG system cannot leak restricted pages." 

## References

- Confluence Cloud REST API: https://developer.atlassian.com/cloud/confluence/rest/v1/
- LangChain PGVector integration: https://docs.langchain.com/oss/python/integrations/vectorstores/pgvector
- LangChain OpenAI embeddings: https://docs.langchain.com/oss/python/integrations/embeddings/openai
