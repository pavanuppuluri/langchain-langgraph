"""PGVector setup isolated from the synchronization logic."""

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector


def build_vector_store(postgres_url: str, collection_name: str, embedding_model: str):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=postgres_url,
        use_jsonb=True,
    )
