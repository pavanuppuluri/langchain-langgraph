"""RAG query path: retrieve Confluence chunks, then ask the chat model."""

from langchain_openai import ChatOpenAI


def answer_question(question: str, vector_store, model: str, top_k: int) -> str:
    docs = vector_store.similarity_search(question, k=top_k)

    if not docs:
        return "I could not find relevant information in the Confluence knowledge base."

    context_blocks = []
    for doc in docs:
        context_blocks.append(
            f"SOURCE: {doc.metadata.get('title')}\n"
            f"URL: {doc.metadata.get('source')}\n"
            f"CONTENT:\n{doc.page_content}"
        )

    context = "\n\n---\n\n".join(context_blocks)
    prompt = f"""
You are an enterprise knowledge assistant.
Answer the user's question using ONLY the Confluence context below.
If the answer is not supported by the context, say that you could not find it in Confluence.
Do not invent facts.
When useful, mention the source page title and URL.

QUESTION:
{question}

CONFLUENCE CONTEXT:
{context}
"""

    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke(prompt)
    return response.content
