# Product Requirements Document (PRD)
## Local Wikipedia RAG Assistant

**Version:** 1.0  
**Course:** BLG483E — Homework 3  
**Date:** 2025

---

## 1. Product Overview

The Local Wikipedia RAG Assistant is a fully offline, ChatGPT-style question-answering system that answers factual questions about famous people and famous places. The system uses Retrieval-Augmented Generation (RAG): it retrieves relevant text from a locally stored Wikipedia knowledge base and feeds it as context to a local language model to generate grounded answers.

The product operates entirely on the user's machine. No external LLM API, cloud service, or internet connection is required after the initial data ingestion.

---

## 2. Problem Statement

General-purpose language models hallucinate facts about real people and places because their weights encode approximate, possibly outdated information. Providing a grounded, factual QA experience requires anchoring generation to verified source documents. At the same time, privacy-conscious users and institutions often cannot or will not send queries to cloud APIs.

This product solves both problems: it grounds answers in real Wikipedia content and runs 100% locally.

---

## 3. Goals and Non-Goals

### Goals
- Ingest Wikipedia articles for at least 20 famous people and 20 famous places
- Split documents into manageable chunks without external NLP libraries
- Generate local embeddings and store them in a persistent vector database
- Classify user queries and retrieve relevant chunks via metadata-filtered semantic search
- Generate grounded answers using a local language model
- Provide a user-friendly chat interface with source chunk transparency
- Support multi-turn conversations with chat history memory

### Non-Goals
- Real-time Wikipedia updates (the knowledge base is a snapshot)
- Support for arbitrary Wikipedia topics beyond the 40 ingested entities
- Streaming token generation in the UI
- Multi-user or server deployment (this is a single-user local tool)
- External API calls of any kind during query time

---

## 4. User Stories

| As a... | I want to... | So that... |
|---|---|---|
| Student | Ask about a famous scientist | I can get a quick factual summary |
| User | Compare two athletes | I can see factual differences side by side |
| User | Ask follow-up questions | The system understands what "he" or "she" refers to |
| User | See which source chunks were used | I can verify the answer myself |
| User | Clear the conversation | I can start fresh without restarting the app |
| Developer | Run the entire system locally | No API keys or cloud accounts are needed |

---

## 5. System Architecture

### 5.1 Pipeline Overview

```
[Wikipedia API]
      │
      ▼
[ingest.py]  ──────────────────────── SQLite (raw_documents + chunks)
                                              │
                                              ▼
[embed_and_store.py] ─────────────── ChromaDB (vector index)
                                              │
                                        ┌─────┘
                                        ▼
[app.py] → [rag_engine.py]
   │            ├── condense_query()       (chat history rewrite)
   │            ├── classify_query()       (LLM-based routing)
   │            ├── retrieve_context()     (ChromaDB + metadata filter)
   │            └── generate_answer()      (Ollama LLM)
   │
   ▼
Streamlit Chat UI
```

### 5.2 Component Responsibilities

| Component | File | Responsibility |
|---|---|---|
| Ingestion | `ingest.py` | Fetch Wikipedia, chunk text, store in SQLite |
| Embedding | `embed_and_store.py` | Generate embeddings, populate ChromaDB |
| RAG Engine | `rag_engine.py` | Classify, retrieve, generate |
| UI | `app.py` | Chat interface, session state, source display |

---

## 6. Data Model

### SQLite — `raw_documents` table
| Column | Type | Description |
|---|---|---|
| id | TEXT (UUID) | Primary key |
| entity_name | TEXT | E.g. "Albert Einstein" |
| entity_type | TEXT | "person" or "place" |
| source_url | TEXT | Wikipedia URL |
| content | TEXT | Full article text |
| word_count | INTEGER | Approximate word count |
| ingestion_date | TIMESTAMP | When the record was created |

### SQLite — `chunks` table
| Column | Type | Description |
|---|---|---|
| id | TEXT (UUID) | Primary key |
| document_id | TEXT | Foreign key to raw_documents |
| chunk_index | INTEGER | Position within document |
| entity_name | TEXT | Denormalized for fast access |
| entity_type | TEXT | "person" or "place" |
| content | TEXT | Chunk text |
| token_count | INTEGER | Approximate token count |

### ChromaDB — `wikipedia_entities` collection
| Field | Description |
|---|---|
| id | `{entity_name}_{chunk_index}_{uuid8}` |
| embedding | 384-dimensional float vector |
| document | Chunk text |
| metadata | `entity_type`, `entity_name`, `chunk_index`, `token_count`, `document_id` |

---

## 7. Chunking Strategy

**Method:** Sentence-boundary chunking with token overlap  
**Implementation:** Native Python `re.split()` and `str.split()` only — no NLTK, LangChain, or LlamaIndex  
**Chunk size:** 512 tokens (approximate, word-based)  
**Overlap:** 128 tokens

**Rationale:**  
Fixed-size word-based chunks provide predictable context window usage. Sentence-boundary splitting prevents mid-sentence cuts, which degrade embedding quality. Overlap ensures that facts spanning adjacent sentences are not lost at chunk boundaries. The 512/128 configuration was chosen to comfortably fit within the embedding model's 256-token optimal window while preserving enough context for meaningful retrieval.

---

## 8. Vector Store Design — Option B

The system uses **Option B: a single ChromaDB collection with metadata filtering**.

### Choice: Option B over Option A

| Criterion | Option A (two collections) | Option B (one collection + metadata) |
|---|---|---|
| Codebase complexity | Higher — two client handles, two query paths | Lower — one collection, filter via `where=` |
| Mixed/comparison queries | Hard — requires merging two result sets | Native — query once, filter by type or skip filter |
| Scalability | Adding entity types requires new collections | Adding a type only requires a new metadata value |
| Retrieval consistency | Scores are not cross-comparable across collections | All scores come from one cosine space |

**Decision:** Option B provides cleaner code, better support for comparison queries, and easier extensibility. The metadata overhead is negligible.

---

## 9. Query Classification

User queries are classified into one of three categories:

| Category | Trigger | Retrieval action |
|---|---|---|
| `person` | Question about a person | Filter: `entity_type = "person"` |
| `place` | Question about a location | Filter: `entity_type = "place"` |
| `mixed` | Comparison or ambiguous | No filter; balanced retrieval across entities |

Classification is performed by the local LLM (`classify_query()` in `rag_engine.py`) using a zero-temperature system prompt with explicit few-shot examples. Comparisons (`"Compare X and Y"`, `"vs"`) are always routed to `mixed`.

For mixed queries, up to 5 chunks per entity are selected from a pool of 300 candidates to ensure balanced representation across both subjects of a comparison.

---

## 10. Chat History Memory

Multi-turn conversations are supported via two mechanisms:

1. **Query condensation:** `condense_query()` rewrites follow-up questions (e.g., "What else did he discover?") into standalone queries (e.g., "What else did Albert Einstein discover?") before embedding and retrieval.

2. **Prompt injection:** The last 6 messages (3 turns) of conversation history are injected into the generation prompt as read-only reference, allowing the LLM to resolve pronouns without treating history as a source of facts.

---

## 11. Anti-Hallucination Measures

The generation prompt enforces:
- Answer only from `[SOURCE: X]` blocks in context
- Facts from one source block must not be applied to a different entity
- If the answer is not in the context: respond with exactly `"I don't know"`
- Temperature set to 0.3 for near-deterministic output

---

## 12. Entities Ingested

### People (20)
Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare, Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo, Taylor Swift, Frida Kahlo, Steve Jobs, Oprah Winfrey, Martin Luther King Jr., Cleopatra, Isaac Newton, Stephen Hawking, Elon Musk, Serena Williams, Pablo Picasso, Jane Goodall

### Places (20)
Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu, Colosseum, Hagia Sophia, Statue of Liberty, Pyramids of Giza, Mount Everest, Big Ben, Angkor Wat, Petra, Christ the Redeemer, Stonehenge, Venice, Acropolis, Niagara Falls, Sydney Opera House, Kremlin

---

## 13. Technology Stack

| Layer | Technology | Justification |
|---|---|---|
| Language | Python 3.10+ | Wide library support, readable syntax |
| Wikipedia fetch | `wikipedia` library | Simple API; handles disambiguation |
| Text chunking | `re`, `str.split()` | Native Python; satisfies no-library constraint |
| Embedding model | `all-MiniLM-L6-v2` (sentence-transformers) | ~30 MB, 384-dim, fast CPU inference |
| Vector DB | ChromaDB (persistent, SQLite backend) | Fully local, metadata filtering, cosine similarity |
| Raw data store | SQLite | Zero-dependency, lightweight, relational |
| Language model | Mistral via Ollama | Strong instruction following, runs on consumer hardware |
| UI | Streamlit | Rapid prototyping; native chat components |

---

## 14. Known Limitations

- **Fixed knowledge base:** Answers reflect Wikipedia content at ingestion time. Articles are not updated automatically.
- **40-entity scope:** The system only has information about the ingested entities. Queries about other people or places will correctly return "I don't know".
- **Disambiguation sensitivity:** Wikipedia disambiguation pages may occasionally resolve to the wrong article. A 3-attempt retry with fallback search is implemented to mitigate this.
- **LLM quality ceiling:** Answer quality is bounded by the local model (Mistral 7B). The model may still occasionally hallucinate despite strict prompting.
- **No streaming:** Responses are returned as a complete block. Latency may be noticeable on slower hardware.
- **Single-user:** There is no authentication, session isolation, or concurrent user support.