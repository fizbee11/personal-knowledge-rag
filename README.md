# Personal Knowledge RAG

A Python-based Retrieval Augmented Generation (RAG) system that indexes your Markdown/Obsidian documents into a vector store and provides an LLM chatbot interface via FastAPI.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend / API Client                      │
│                (Open WebUI, curl, Postman, etc.)              │
└─────────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────────┐
│                  FastAPI Server (/v1/chat/completions)        │
│  ┌──────────────┬──────────────────┬──────────────────────┐ │
│  │   LLM        │   Vector Store   │   Agent + Tools      │ │
│  │ (ChatOpenAI) │  (Qdrant +       │ (LangChain agents)   │ │
│  │              │   Embeddings)    │                      │ │
│  └──────────────┴──────────────────┴──────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────────┐
│              Local Vector Database (Qdrant)                   │
│         http://localhost:6333 (or your Qdrant instance)       │
└─────────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────────┐
│               Local LLM Server (Ollama / LM Studio)          │
│         <YOUR_LLM_SERVER_URL:1234/v1>                        │
│           http://localhost:11434 (embeddings only)             │
└─────────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────────┐
│                  Markdown Document Sources                    │
│  ┌──────────────────┬─────────────────────────────────────┐  │
│  │ ./my_docs        │ <YOUR_SECONDARY_DOCS_PATH>         │  │
│  └──────────────────┴─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Components

| Component | Description | Config Section |
|-----------|-------------|----------------|
| **FastAPI** | REST API server for chat completions | - |
| **LangChain** | Orchestration, agents, tools | - |
| **OpenAI Embeddings** | Text vectorization via LM Studio/Ollama | `ollama` |
| **Qdrant** | Vector database for semantic search | `qdrant` |
| **Ollama/LM Studio** | LLM inference engine | `ollama`, `llm-studio` |

## External Dependencies

### Required Services

1. **Qdrant** - Vector database
   - Self-hosted: Run via Docker or binary
   - URL: `http://localhost:6333` (default)

2. **Ollama** OR **LM Studio** - LLM hosting
    - Ollama for embeddings, LM Studio for chat (or either)
    - Default embedding model: `<YOUR_EMBEDDING_MODEL>`
    - Default chat model: `<YOUR_CHAT_MODEL>`

### Python Dependencies

Install via **Poetry** (recommended):

```bash
# Install dependencies from pyproject.toml
poetry install

# OR with pip/venv:
python -m venv venv
source venv/bin/activate  # Linux/macOS
source venv/Scripts/activate  # Windows
pip install -r pyproject.toml
```

## Setup Instructions

### 1. Configure Your Environment

Edit `config.cfg` with your settings:

```ini
[markdown]
paths = <YOUR_MARKDOWN_PATHS>
exclude_dirs = .obsidian

[indexing]
manifest_path = ./md_manifest.json
collection_name = <YOUR_COLLECTION_NAME>
chunk_size = 1000
chunk_overlap = 200

[qdrant]
url = <YOUR_QDRANT_URL>

[ollama]
base_url = <YOUR_OLLAMA_URL>
embedding_model = <YOUR_EMBEDDING_MODEL>

[llm-studio]
base_url = <YOUR_LLM_STUDIO_URL/v1>
llm_model = <YOUR_CHAT_MODEL>
api_key = <YOUR_API_KEY>  # Required field (ignored by LM Studio)
```

### 2. Create Document Directory

Create the default documentation directory if it doesn't exist:

```bash
mkdir -p my_docs
```

Copy your Markdown/Obsidian files into `my_docs/` or configure custom paths in `config.cfg`.

### 3. Start Required Services

```bash
# Qdrant (via Docker example)
docker run -d -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/var/lib/qdrant \
  qdrant/qdrant

# Ollama - pull embeddings model
ollama pull qwen3-embedding

# LM Studio or Ollama server running on port 1234/v1
```

### 4. Index Your Documents

Index all Markdown files into the vector store:

```bash
just index
# Or manually:
python index_docs.py
```

This will:
- Scan configured markdown paths
- Embed text chunks using Ollama embeddings model
- Store vectors in Qdrant collection `my_docs`
- Track file state in `md_manifest.json` for incremental updates

### 5. Start the Chatbot Server

```bash
just run
# Or manually:
python main.py
```

The server starts at `http://0.0.0.0:8000/v1/chat/completions`

### 6. Test the API

**Streaming chat:**
```bash
curl -N "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "langchain-rag-bot", "messages": [{"role": "user", "content": "What is RAG?"}], "stream": true}'
```

**Non-streaming chat:**
```bash
curl "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model": "langchain-rag-bot", "messages": [{"role": "user", "content": "What is RAG?"}]}'
```

Connect your API client (Open WebUI, LangChain, etc.) to `http://localhost:8000/v1/chat/completions`.

## Usage Examples

### Open WebUI Integration

1. Set API base URL: `http://localhost:8000`
2. Model ID: `langchain-rag-bot` (auto-registered at `/v1/models`)
3. Use your knowledge documents as context in chat

### Python API Client

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="any-value")

response = client.chat.completions.create(
    model="langchain-rag-bot",
    messages=[{"role": "user", "content": "What did I write about my project?"}]
)
print(response.choices[0].message.content)
```

## How It Works

### Indexing Pipeline

1. **Scan**: Finds all `.md` files in configured paths (excluding `*.obsidian`)
2. **Hash**: Computes SHA256 hash of each file
3. **Compare**: Checks against `md_manifest.json` to detect changes
4. **Chunk**: Splits changed documents into tokensized chunks (512/100)
5. **Embed**: Generates vector embeddings via Ollama/LM Studio
6. **Store**: Adds vectors to Qdrant with deterministic IDs (prevents duplicates)
7. **Cleanup**: Removes vectors for deleted files from previous index

### Chat Flow

1. User sends query to `/v1/chat/completions`
2. Agent calls `retrieve_context()` tool to search vector store
3. Retrieved context + user query → LLM response
4. Response streamed or returned as complete message

## Configuration Reference

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `markdown` | `paths` | `<YOUR_MARKDOWN_PATHS>` | Comma-separated markdown paths |
| | `exclude_dirs` | `.obsidian` | Directories to skip during indexing |
| `indexing` | `manifest_path` | `./md_manifest.json` | Path to index state manifest |
| | `collection_name` | `my_docs` | Qdrant collection name |
| | `chunk_size` | `1000` | Characters per chunk (raw, not tokenized) |
| | `chunk_overlap` | `200` | Overlap between chunks |
| `qdrant` | `url` | `http://localhost:6333` | Qdrant API endpoint |
| `ollama` | `base_url` | `<YOUR_OLLAMA_URL>` | Embeddings model server |
| | `embedding_model` | `<YOUR_EMBEDDING_MODEL>` | Model name for embeddings |
| `llm-studio` | `base_url` | `<YOUR_LLM_STUDIO_URL/v1>` | LLM API endpoint |
| | `llm_model` | `<YOUR_CHAT_MODEL>` | Chat model identifier |
| | `api_key` | `<YOUR_API_KEY>` | Required field (value ignored) |

## Important Notes

### Incremental Indexing
The system tracks file hashes in `md_manifest.json`. Only changed/new files are re-indexed. Never edit this file manually—use `just index` or `python index_docs.py`.

### Vector Store Persistence
Qdrant stores data on disk by default. The collection persists across restarts unless you delete it via Qdrant UI or API.

### Streaming Protocol
The API follows OpenAI's SSE format:
- Each chunk: `data: {"choices":[{"delta":{"content":"..."}}]}\n\n`
- End sentinel: `data: [DONE]\n\n`

### Agent Behavior
When documents don't contain an answer, the agent uses its internal knowledge but notes that documents lacked info.

### Tokenization Note
Embeddings use raw string input (not tokenized) with tiktoken disabled for compatibility with Ollama/LM Studio embeddings APIs.

## Troubleshooting

### "Collection doesn't exist" error
Run `python index_docs.py` to create the collection and index initial documents.

### Streaming hangs
Ensure Qdrant is running and accessible at the configured URL. Check network connectivity if using remote services.

### Empty responses
Verify:
1. LLM server is responding (`curl http://.../v1/models`)
2. Embeddings model loaded (`ollama list` or check LM Studio)
3. Collection has data (`qdrant-cli get collection <name> count`)

## License

MIT License - see LICENSE file for details.
