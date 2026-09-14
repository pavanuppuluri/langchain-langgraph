"""CLI entry point.

Examples:
  python -m app.main index
  python -m app.main search "How does Product Foundation handle SKU limits?"
"""

import argparse

from app.config import settings
from app.confluence import ConfluenceClient
from app.database import MetadataStore
from app.indexer import IncrementalIndexer
from app.search import answer_question
from app.vector_store import build_vector_store


def build_components():
    confluence = ConfluenceClient(
        settings.confluence_base_url,
        settings.confluence_email,
        settings.confluence_api_token,
        settings.confluence_space_key,
    )
    metadata = MetadataStore(settings.postgres_url)
    vector_store = build_vector_store(
        settings.postgres_url,
        settings.pgvector_collection,
        settings.embedding_model,
    )
    return confluence, metadata, vector_store


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index")
    search_parser = sub.add_parser("search")
    search_parser.add_argument("question")
    args = parser.parse_args()

    confluence, metadata, vector_store = build_components()

    if args.command == "index":
        stats = IncrementalIndexer(settings, confluence, metadata, vector_store).sync()
        print("Indexing complete:")
        for key, value in sorted(stats.items()):
            print(f"  {key}: {value}")
    else:
        print(answer_question(args.question, vector_store, settings.chat_model, settings.top_k))


if __name__ == "__main__":
    main()
