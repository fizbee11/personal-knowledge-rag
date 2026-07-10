"""
Incremental indexer for an markdown files into Qdrant.

It tracks a content hash per file in a local manifest (vault_manifest.json).
On each run it only re-embeds files that are new or changed, and removes
chunks for files that were deleted from the vault. Unchanged files are
skipped entirely.
"""

import hashlib
import json
import uuid
from configparser import ConfigParser
from pathlib import Path

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core import vectorstores
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)
import frontmatter
import re
from datetime import datetime

from llama_index.core.node_parser import MarkdownNodeParser, SimpleNodeParser
from llama_index.core import SimpleDirectoryReader
from llama_index.core.text_splitter import TokenTextSplitter


# ---- Config ----------------------------------------------------------
def load_config() -> ConfigParser:
    config = ConfigParser()
    config.read("config.cfg")
    return config


config = load_config()

# Markdown paths from config (comma-separated)
MD_PATHS = [
    Path(p.strip()).expanduser()
    for p in config.get("markdown", "paths", fallback="./my_docs").split(",")
]
MANIFEST_PATH = Path(
    config.get("indexing", "manifest_path", fallback="./md_manifest.json")
)
COLLECTION_NAME = config.get("indexing", "collection_name", fallback="my_docs")
QDRANT_URL = config.get("qdrant", "url", fallback="http://localhost:6333")
OLLAMA_BASE_URL = config.get("ollama", "base_url", fallback="http://localhost:11434")
EMBEDDING_MODEL = config.get("ollama", "embedding_model", fallback="qwen3-embedding")
CHUNK_SIZE = config.getint("indexing", "chunk_size", fallback=1000)
CHUNK_OVERLAP = config.getint("indexing", "chunk_overlap", fallback=200)
EXCLUDE_DIRS = set(
    d.strip()
    for d in config.get("indexing", "exclude_dirs", fallback=".obsidian").split(",")
)
# -----------------------------------------------------------------------


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def deterministic_id(source: str, chunk_index: int) -> str:
    """Stable UUID derived from file path + chunk index, so re-embedding
    the same chunk overwrites the existing point instead of duplicating it."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}::{chunk_index}"))


def delete_chunks_for_source(client: QdrantClient, source: str) -> None:
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]
        ),
    )


def get_markdown_files() -> list[Path]:
    """Recursively find all markdown files in configured paths."""
    all_files = []
    for md_path in MD_PATHS:
        if md_path.exists():
            files = list(md_path.expanduser().rglob("*.md"))
            # print(files)
            all_files.extend(f for f in files if not EXCLUDE_DIRS.intersection(f.parts))
    return all_files


def index_vault() -> None:
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        validate_model_on_init=False,
        base_url=OLLAMA_BASE_URL,
    )
    client = QdrantClient(url=QDRANT_URL)

    if not client.collection_exists(COLLECTION_NAME):
        vector_size = len(embeddings.embed_query("sample text"))
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    manifest = load_manifest()
    current_files: dict[str, str] = {}
    changed_or_new: list[Path] = []

    for f in get_markdown_files():
        source = str(f)
        h = file_hash(f)
        current_files[source] = h
        if manifest.get(source) != h:
            changed_or_new.append(f)

    deleted_sources = set(manifest.keys()) - set(current_files.keys())

    if not changed_or_new and not deleted_sources:
        print("No changes detected. Vault index is up to date.")
        return

    for source in deleted_sources:
        print(f"Removing deleted file from index: {source}")
        delete_chunks_for_source(client, source)
        # Load only changed files to index

    # for n in nodes:
    for f in changed_or_new:
        # documents object for llama_index
        document = SimpleDirectoryReader(
            input_files=[f], required_exts=[".md"]
        ).load_data()

        nodes = MarkdownNodeParser(
            include_metadata=True, include_prev_next_rel=True
        ).get_nodes_from_documents(document)

        splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=100)
        chunked_nodes = splitter(
            nodes
        )  # instead of splitter.get_nodes_from_documents(nodes)

        # removing nodes with only whitespaces(empty)
        chunked_nodes = [node for node in chunked_nodes if node.text.strip()]
        texts = [node.text for node in chunked_nodes]
        metadatas = [node.metadata for node in chunked_nodes]

        ids = [deterministic_id(str(f), i) for i in range(len(chunked_nodes))]
        print(
            f"Text: {texts}, ID: {ids}, Metadats: {metadatas}, Length of Chunked Nodes: {len(chunked_nodes)}, Length of splits: {len(texts)}"
        )
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        # vector_store.add_documents(documents=text, ids=ids)

    save_manifest(current_files)

    print("Indexing complete.")


def extract_metadata(filepath: str, chunk_text: str, heading_path: list[str]):
    post = frontmatter.load(filepath)
    stat = Path(filepath).stat()

    return {
        "source_path": filepath,
        "title": post.metadata.get("title", Path(filepath).stem),
        "tags": post.metadata.get("tags", []),
        #        "wikilinks": extract_wikilinks(post.content),  # [[Note Name]] -> list
        "char_count": len(chunk_text),
    }


if __name__ == "__main__":
    index_vault()
