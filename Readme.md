# 🤖 Local Wikipedia RAG Assistant

A fully local, ChatGPT-style question-answering system about famous people and places, built with Wikipedia data, local embeddings, ChromaDB, and Ollama. No external LLM APIs are used — everything runs on your machine.

## Demo link

> 📹 [https://www.loom.com/share/ecf0342b2d83423a9790e84e82caacfd]

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Local Model (Ollama)](#running-the-local-model-ollama)
- [Data Ingestion](#data-ingestion)
- [Starting the Application](#starting-the-application)
- [Example Queries](#example-queries)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)

---

## Overview

This system ingests Wikipedia pages for **20 famous people** and **20 famous places** (40 entities total), chunks and embeds them locally, stores them in a ChromaDB vector database, and answers natural language questions using a local LLM via Ollama.

**Key properties:**
- ✅ Fully local — no OpenAI, Anthropic, or any external LLM API
- ✅ Semantic search using `all-MiniLM-L6-v2` (sentence-transformers)
- ✅ Retrieval-Augmented Generation (RAG) with metadata-filtered retrieval
- ✅ LLM-based query classification (person / place / mixed)
- ✅ Chat history memory for multi-turn conversations
- ✅ Follow-up question condensation (pronoun resolution across turns)
- ✅ Streaming responses — answer appears token-by-token in real time
- ✅ In-memory response cache — repeated queries return instantly (⚡ Cache Hit)
- ✅ Per-query latency metrics — search time and generation time shown in the UI
- ✅ Streamlit chat interface with collapsible source chunk viewer

---

## Architecture

```
User Query
    │
    ▼
[Response Cache Check]  ← returns instantly if query was seen before
    │
    ▼
[Query Condensation]    ← rewrites follow-up questions into standalone queries
    │
    ▼
[Query Classification]  → "person" | "place" | "mixed"
    │
    ▼
[ChromaDB Retrieval]    ← metadata-filtered vector search (Option B)
    │
    ▼
[Ollama LLM Generation] ← context-grounded answer, streamed token-by-token
    │
    ▼
Streamlit Chat UI       ← displays answer, query type, latency, source chunks
```

**Storage stack:**
| Layer | Technology |
|---|---|
| Raw documents + chunks | SQLite (`data/rag_database.db`) |
| Vector embeddings | ChromaDB (`data/chroma_db/`) |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers, local) |
| Language model | Mistral via Ollama (local) |
| Response cache | In-memory Python dict (session-scoped) |
| UI | Streamlit |

---

## Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/) installed and running
- ~4 GB free disk space (for model weights + database)
- Internet connection only for the first run (Wikipedia fetch + model download)

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd local-wikipedia-rag
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first time you run the embedding model, `sentence-transformers` will download `all-MiniLM-L6-v2` (~30 MB) automatically.

---

## Running the Local Model (Ollama)

### 1. Install Ollama

Download from [https://ollama.com/download](https://ollama.com/download) and follow the instructions for your OS.

### 2. Pull the language model

```bash
ollama pull mistral
```

> You can alternatively use `llama3.2:3b` or `phi3`. If you change the model, update `LLM_MODEL` in `rag_engine.py`.

### 3. Start the Ollama server

```bash
ollama serve
```

Leave this running in a separate terminal. Ollama listens on `http://localhost:11434` by default.

---

## Data Ingestion

Run the two ingestion scripts in order. **This only needs to be done once.**

### Step 1 — Fetch and chunk Wikipedia data

```bash
python3 ingest.py
```

This script:
- Fetches Wikipedia pages for 20 people and 20 places
- Splits each article into overlapping chunks (512 tokens, 128 token overlap)
- Stores raw documents and chunks in `data/rag_database.db`

Expected output: `40 documents`, `~800–1200 chunks`

> ⚠️ This step makes ~40 HTTP requests to Wikipedia. A 2-second rate-limit delay is built in — it will take **2–3 minutes** to complete.

### Step 2 — Generate embeddings and populate vector store

```bash
python3 embed_and_store.py
```

This script:
- Reads all chunks from SQLite
- Generates 384-dimensional embeddings locally using `all-MiniLM-L6-v2`
- Stores vectors in ChromaDB at `data/chroma_db/`
- Runs a quick retrieval test at the end

---

## Starting the Application

```bash
streamlit run app.py
```

Open your browser to [http://localhost:8501](http://localhost:8501).

The sidebar shows system info and a **Clear Chat History** button. The main area is a chat interface.

---

## Example Queries

### People
```
Who was Albert Einstein and what is he known for?
What did Marie Curie discover?
Why is Nikola Tesla famous?
Compare Lionel Messi and Cristiano Ronaldo
What is Frida Kahlo known for?
```

### Places
```
Where is the Eiffel Tower located?
Why is the Great Wall of China important?
What is Machu Picchu?
What was the Colosseum used for?
Where is Mount Everest?
```

### Mixed / Comparison
```
Which famous place is located in Turkey?
Which person is associated with electricity?
Compare Albert Einstein and Nikola Tesla
Compare the Eiffel Tower and the Statue of Liberty
```

### Expected failure cases
```
Who is the president of Mars?        → "I don't know"
Tell me about John Doe               → "I don't know"
```

---

## Project Structure

```
local-wikipedia-rag/
├── app.py                  # Streamlit chat interface (streaming, cache, latency UI)
├── rag_engine.py           # RAG pipeline (classify → retrieve → generate → cache)
├── ingest.py               # Wikipedia fetch + chunk + SQLite storage
├── embed_and_store.py      # Embedding generation + ChromaDB storage
├── debug_chroma.py         # Utility: inspect ChromaDB retrieval results
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── documentation/
│   ├── product_prd.md      # Product Requirements Document
│   └── recommendation.md  # Production deployment recommendation
└── data/                   # Auto-created by scripts
    ├── rag_database.db     # SQLite: raw docs + chunks
    └── chroma_db/          # ChromaDB: vector index
```

---

## Design Decisions

**Option B — Single collection with metadata filtering:**
One ChromaDB collection stores all entities. Each chunk carries `entity_type` (`person`/`place`) and `entity_name` metadata. At query time, the query is classified and the correct metadata filter is applied. This keeps the codebase simple while still enabling type-aware retrieval. See `product_prd.md` for the full rationale.

**Native chunking:**
Text splitting uses only Python's built-in `re` module and `str.split()` — no NLTK, LangChain, or LlamaIndex. Chunks are ~512 tokens with a 128-token overlap at sentence boundaries.

**LLM-based query classification:**
`classify_query()` in `rag_engine.py` calls the local LLM with a zero-temperature system prompt to route queries to the correct retrieval path. Comparison queries (e.g., "Compare X and Y") are always routed to `mixed`, which retrieves from both person and place namespaces.

**Chat history memory and query condensation:**
`condense_query()` rewrites follow-up questions into standalone queries before retrieval, so pronouns and implicit references (e.g., "What else did he discover?") resolve correctly across turns. Only the last 3 turns (6 messages) of history are sent to keep prompts concise.

**Streaming responses:**
LLM output is streamed token-by-token using Ollama's stream API. The UI renders each token as it arrives, giving the appearance of a live typing effect. This significantly reduces perceived latency compared to waiting for the full response.

**In-memory response cache:**
`rag_engine.py` maintains a session-scoped Python dict (`_response_cache`) keyed by the lowercased query string. Both the raw user query and the condensed standalone query are stored as keys so that follow-up rephrasing of the same question also benefits from the cache. Cache hits are shown in the UI as ⚡ Cache Hit with 0.00s latency.

**Per-query latency metrics:**
Each response displays two timing measurements: retrieval time (embedding + ChromaDB search) and generation time (LLM). This makes it easy to identify where latency is coming from and evaluate the impact of caching.

---
