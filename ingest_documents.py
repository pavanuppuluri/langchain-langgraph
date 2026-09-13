loader = UnstructuredLoader(
    file_path="/Users/ed.../GithubProjects/langchain-course/medium.log1.txt",
    chunking_strategy="basic",
    max_characters=100000
)

document = loader.load()

print("splitting...")
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0
)

texts = text_splitter.split_documents(document)

print(f"created {len(texts)} chunks")

embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

print("ingesting...")

PineconeVectorStore.from_documents(
    texts,
    embeddings,
    index_name=os.environ["INDEX_NAME"]
)


/*
INDEX_NAME is the name of your Pinecone vector index.

In this code:
  the program is essentially saying:

"Take my document chunks, convert them into embeddings, and store those embeddings inside the Pinecone index 
 whose name is stored in INDEX_NAME."

Where does INDEX_NAME come from?

Usually from your .env file:

OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
INDEX_NAME=my-rag-index
*/
