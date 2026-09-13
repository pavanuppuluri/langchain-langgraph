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

##### Where is Pinecone fits in?
- We talked about embeddings that we create from the text, the vectors
- We need to store those vectors somewhere, we want the ability to search in vector space for the closest vectors of the current one
<br><br>
All of this is being handled by vector databases, Pinecone in this case

<img width="970" height="688" alt="image" src="https://github.com/user-attachments/assets/75682383-d743-46f5-b5aa-fa564c487194" />

<br><br>
##### In the context of RAG ingestion, what role do OpenAIEmbeddings serve?
- They convert text chunks into high-dimensional vector representations
- OpenAIEmbeddings transform text chunks into dense numerical vectors that capture semantic meaning. These embeddings enable similarity search in the vector store, allowing retrieval of relevant document chunks based on query similarity.

##### What happens when you call PineconeVectorStore.from_documents(docs, embeddings, index_name="rag-index")?
- It creates embeddings for documents and stores both vectors and metadata in Pinecone
- This method processes each document by generating embeddings using the provided embedding model, then stores both the vector representations and document metadata (including original text content) in the specified Pinecone index for later retrieval.

