## Basic RAG Pipeline

<img width="762" height="611" alt="image" src="https://github.com/user-attachments/assets/136ea171-6b03-4e70-a668-73ee7d6e4f04" />


### Ingestion
- Loading the medium blog (TextLoader)
- Splitting the blog into smaller chunks (TextSplitter)
- Embed the chunks and get vectors (OpenAIEmbeddings)
- Store the embeddings in Pinecone vectorstore (PineconeVectorStore)

#### Embedding model
- Embedding model is simply a black box that takes input as text and output vectors in an embedding vector space

<img width="1376" height="605" alt="image" src="https://github.com/user-attachments/assets/6f569bdf-02a5-4adb-a513-9800fcee79b0" />
