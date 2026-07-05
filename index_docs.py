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

# ---- Config ----------------------------------------------------------
def load_config() -> ConfigParser:
    config = ConfigParser()
    config.read("config.cfg")
    return config

config = load_config()

# Markdown paths from config (comma-separated)
MD_PATHS = [
    Path(p.strip()) for p in config.get("markdown", "paths", fallback="./my_docs").split(",")
]
MANIFEST_PATH = Path(config.get("indexing", "manifest_path", fallback="./md_manifest.json"))
COLLECTION_NAME = config.get("indexing", "collection_name", fallback="my_docs")
QDRANT_URL = config.get("qdrant", "url", fallback="http://localhost:6333")
OLLAMA_BASE_URL = config.get("ollama", "base_url", fallback="http://localhost:11434")
EMBEDDING_MODEL = config.get("ollama", "embedding_model", fallback="nomic-embed-text")
CHUNK_SIZE = config.getint("indexing", "chunk_size", fallback=1000)
CHUNK_OVERLAP = config.getint("indexing", "chunk_overlap", fallback=200)
EXCLUDE_DIRS = set(
    d.strip() for d in config.get("indexing", "exclude_dirs", fallback=".obsidian").split(",")
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
            files = list(md_path.rglob("*.md"))
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

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
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

    for f in changed_or_new:
        source = str(f)
        print(f"Indexing (new/changed): {source}")

        # Clear any existing chunks for this file first, in case the file
        # shrank and now produces fewer chunks than before.
        delete_chunks_for_source(client, source)

        loader = UnstructuredMarkdownLoader(str(f), autodetect_encoding=True)
        docs = loader.load()
        splits = text_splitter.split_documents(docs)

        if not splits:
            print("  -> no content extracted, skipping")
            continue

        ids = [deterministic_id(source, i) for i in range(len(splits))]
        vector_store.add_documents(documents=splits, ids=ids)
        print(f"  -> {len(splits)} chunks")

    save_manifest(current_files)

    print("Indexing complete.")


if __name__ == "__main__":
    index_vault()
