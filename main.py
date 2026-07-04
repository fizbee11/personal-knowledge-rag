from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from qdrant_client.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

m_loader_kwargs = {"autodetect_encoding": True}

loader = DirectoryLoader(
    path="./my_docs",
    glob="**/*.md",
    loader_cls=UnstructuredMarkdownLoader,
    loader_kwargs=m_loader_kwargs,
)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")


embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    validate_model_on_init=False,
    base_url="http://zephyrus-1.tailac30a.ts.net:11434",
)

# Qdrant client
client = QdrantClient(url="http://localhost:6333")

vector_size = len(embeddings.embed_query("sample text"))

if not client.collection_exists("test"):
    client.create_collection(
        collection_name="test",
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
vector_store = QdrantVectorStore(
    client=client,
    collection_name="test",
    embedding=embeddings,
)


document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])
