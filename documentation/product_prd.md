# Product Requirements Document (PRD)
## Local Wikipedia RAG Assistant

---

## 1. Product Overview

The Local Wikipedia RAG Assistant is a fully local, ChatGPT-style question-answering system about famous people and places. It ingests Wikipedia articles, stores them in a local vector database, and uses a local language model to generate grounded answers — entirely without external APIs or network calls at inference time.

**Core Value Proposition:**
- Privacy-first: all data, embeddings, and LLM inference stay on the user's machine
- No API costs or rate limits
- Transparent retrieval with source chunk visibility
- Reproducible and auditable knowledge base

---

## 2. Goals and Non-Goals

### Goals
- Answer natural language questions about 40 famous people and places
- Retrieve relevant context using semantic (vector) search
- Generate answers grounded strictly in retrieved Wikipedia context
- Support multi-turn conversations with chat history memory
- Provide a clean Streamlit chat UI with source chunk viewer

### Non-Goals
- General-purpose Q&A beyond the ingested knowledge base
- Real-time or live Wikipedia updates after ingestion
- Multi-user or cloud deployment (single-user localhost only)
- Languages other than English
- Structured data queries (tables, statistics)

---

## 3. User Stories

| ID | As a user, I want to… | So that… |
|----|----------------------|----------|
| U1 | Ask who a famous person is | I can learn about them quickly |
| U2 | Ask where a famous place is located | I get a grounded, factual answer |
| U3 | Compare two people | I understand similarities and differences |
| U4 | Compare two places | I understand what sets them apart |
| U5 | Ask a mixed query spanning both categories | The system handles cross-category questions |
| U6 | See the source chunks behind an answer | I can verify where the information came from |
| U7 | Clear chat history | I can start a fresh conversation |
| U8 | Ask a follow-up question using pronouns | The system resolves references correctly |
| U9 | Ask about something not in the database | The system says "I don't know" instead of hallucinating |

---

## 4. Functional Requirements

### 4.1 Data Ingestion
- **FR-1:** The system SHALL ingest Wikipedia pages for at least 20 people and 20 places (40 entities minimum).
- **FR-2:** The system SHALL include all entities specified in the project brief (e.g., Albert Einstein, Eiffel Tower).
- **FR-3:** The system SHALL split documents into overlapping chunks using only Python built-in modules (`re`, `str.split()`). Third-party chunking libraries (LangChain, NLTK, LlamaIndex) are not permitted.
- **FR-4:** Chunk size SHALL be approximately 512 tokens with a 128-token overlap to preserve cross-sentence context.
- **FR-5:** Raw documents and chunks SHALL be persisted in a local SQLite database with separate `raw_documents` and `chunks` tables.
- **FR-6:** Ingestion SHALL include rate limiting (2 second delay per Wikipedia request) and retry logic (up to 3 attempts) to handle API errors gracefully.

### 4.2 Embedding and Vector Storage
- **FR-7:** Embeddings SHALL be generated locally using `sentence-transformers` (model: `all-MiniLM-L6-v2`). No external embedding API is permitted.
- **FR-8:** Embeddings SHALL be stored in a persistent ChromaDB vector database using cosine similarity.
- **FR-9:** Each stored chunk SHALL carry metadata: `entity_name`, `entity_type` (person/place), `chunk_index`, `token_count`, `document_id`.
- **FR-10:** The system SHALL use Option B: a single ChromaDB collection with metadata-based filtering (not separate collections per entity type).

### 4.3 Retrieval
- **FR-11:** Given a user query, the system SHALL classify it as "person", "place", or "mixed" using the local LLM at zero temperature.
- **FR-12:** The system SHALL apply `entity_type` metadata filtering in ChromaDB based on the classification result.
- **FR-13:** For "mixed" or comparison queries, the system SHALL retrieve chunks from both person and place namespaces and merge them into a single context.
- **FR-14:** The system SHALL retrieve the top-K most relevant chunks (default K=8).

### 4.4 Chat History and Query Condensation
- **FR-15:** The system SHALL maintain conversation history across turns within a session.
- **FR-16:** For follow-up questions, the system SHALL use `condense_query()` to rewrite the question into a standalone query before retrieval, so that pronouns and references resolve correctly.

### 4.5 Generation
- **FR-17:** The system SHALL use a local LLM via Ollama (Mistral) for answer generation.
- **FR-18:** Generated answers SHALL be grounded in retrieved context. The LLM SHALL be instructed not to use external knowledge.
- **FR-19:** If the answer cannot be found in retrieved context, the system SHALL respond with "I don't know."
- **FR-20:** The system SHALL stream the LLM response token-by-token to the UI.

### 4.6 Chat Interface
- **FR-21:** The system SHALL provide a Streamlit-based chat UI.
- **FR-22:** The UI SHALL display chat history across the session.
- **FR-23:** The UI SHALL allow users to clear chat history.
- **FR-24:** The UI SHALL display retrieved source chunks in a collapsible expander.
- **FR-25:** The UI SHALL show query classification type and latency metrics (search time, generation time) below each response.
- **FR-26:** The system SHALL cache responses in memory; identical queries within a session SHALL return instantly without re-running the pipeline.

---

## 5. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | The system SHALL run entirely on localhost with no external API calls at inference time |
| NFR-2 | Ingestion of 40 entities SHALL complete within 5 minutes (rate-limited at 2s/request) |
| NFR-3 | Retrieval (embedding + vector search) SHALL complete in under 2 seconds on a modern laptop CPU |
| NFR-4 | The system SHALL handle Wikipedia disambiguation errors and page-not-found errors without crashing |
| NFR-5 | The Chroma database SHALL persist across application restarts |
| NFR-6 | The system SHALL run on Python 3.10 or higher on macOS, Linux, or Windows |

---

## 6. Vector Store Design Decision: Option B

The system implements **Option B: a single ChromaDB collection with metadata filtering**, as opposed to Option A (two separate collections).

**Rationale:**

| Factor | Option A (Two Collections) | Option B (Single Collection + Metadata) ✅ |
|--------|---------------------------|------------------------------------------|
| Codebase complexity | Higher — two clients, two indices | Lower — one client, one index |
| Mixed / comparison queries | Requires two separate queries and manual merging | Single query with no filter, or two filtered sub-queries in one function |
| Maintenance | Double the index management | Single index to manage, back up, and rebuild |
| Query routing flexibility | Must pick a collection before querying | Metadata filter applied at query time — more flexible |
| Extensibility | Adding a third category requires a new collection | Adding a new category only requires a new metadata value |

**Conclusion:** Option B results in simpler code, better support for mixed queries, and easier extensibility. The slight added complexity of metadata filtering is negligible given ChromaDB's native support for `where` clauses.

---

## 7. Data Model

### SQLite: `raw_documents`
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT (UUID) | Primary key |
| entity_name | TEXT | Human-readable name (e.g., "Albert Einstein") |
| entity_type | TEXT | "person" or "place" |
| source_url | TEXT | Wikipedia URL |
| content | TEXT | Full article text |
| word_count | INTEGER | Word count |
| ingestion_date | TIMESTAMP | When the record was created |

### SQLite: `chunks`
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT (UUID) | Primary key |
| document_id | TEXT | Foreign key to raw_documents |
| chunk_index | INTEGER | Chunk position within the document |
| entity_name | TEXT | Denormalized for fast lookup |
| entity_type | TEXT | Denormalized for fast lookup |
| content | TEXT | Chunk text |
| token_count | INTEGER | Approximate token count |
| created_at | TIMESTAMP | When the chunk was created |

### ChromaDB Collection: `wikipedia_entities`
| Field | Description |
|-------|-------------|
| id | `{entity_name}_{chunk_index}_{uuid8}` |
| embedding | 384-dimensional float vector (all-MiniLM-L6-v2) |
| document | Chunk text |
| metadata.entity_type | "person" or "place" |
| metadata.entity_name | Entity name |
| metadata.chunk_index | Chunk position |
| metadata.token_count | Token count |
| metadata.document_id | Link to SQLite raw_documents |

---

## 8. System Architecture

```
User Query (Streamlit UI)
        │
        ▼
[condense_query()]         ← rewrites follow-up questions using chat history
        │
        ▼
[classify_query()]         → "person" | "place" | "mixed"   (Mistral, temp=0)
        │
        ▼
[retrieve_context()]       ← ChromaDB cosine search + entity_type metadata filter
        │
        ▼
[generate_answer_stream()] ← Mistral via Ollama, context-grounded, streamed
        │
        ▼
Streamlit Chat UI          ← displays streamed tokens, source chunks, latency
```

---

## 9. Entities Ingested

### People (20)
Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare, Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo, Taylor Swift, Frida Kahlo, Steve Jobs, Oprah Winfrey, Martin Luther King Jr., Cleopatra, Isaac Newton, Stephen Hawking, Elon Musk, Serena Williams, Pablo Picasso, Jane Goodall

### Places (20)
Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon, Machu Picchu, Colosseum, Hagia Sophia, Statue of Liberty, Pyramids of Giza, Mount Everest, Big Ben, Angkor Wat, Petra, Christ the Redeemer, Stonehenge, Venice, Acropolis, Niagara Falls, Sydney Opera House, Kremlin

---

## 10. Known Limitations

- The knowledge base is static — it reflects Wikipedia at ingestion time and does not update automatically.
- Chunking is word-based (approximate tokens), not true BPE-token-based, so chunk sizes may vary slightly.
- Query classification uses an LLM call, which adds ~0.5–2 seconds of latency per query.
- The "I don't know" guardrail relies on LLM instruction-following; it is not a hard-coded filter and may occasionally fail on edge cases.
- The system handles 40 entities. Scaling to thousands of entities would require re-evaluation of the chunking, retrieval, and indexing strategy.