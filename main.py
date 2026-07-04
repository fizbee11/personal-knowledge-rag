from langchain_core.documents import Document
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
