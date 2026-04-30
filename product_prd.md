# Product PRD: Local Wikipedia RAG Assistant

## Executive Summary

This document outlines the architecture and design of a **Local Wikipedia RAG (Retrieval-Augmented Generation) Assistant** — a ChatGPT-style system that runs entirely on localhost. The system combines information retrieval with a local language model to answer questions about famous people and places without relying on external APIs.

---

## 1. Project Overview

### Vision
Build a fully self-contained AI assistant that ingests Wikipedia data, processes it locally, and generates contextually accurate answers using a local language model.

### Scope
- Ingest data on **20+ famous people** and **20+ famous places**
- Process documents into optimized chunks
- Generate embeddings locally
- Store vectors with metadata in a local vector database
- Route queries based on entity type
- Generate answers grounded in retrieved context
- Provide a user-friendly Streamlit interface

### Technical Stack
- **Language:** Python 3.9+
- **LLM:** Ollama with Llama 3.2 3b or Mistral
- **Embeddings:** Nomic Embed Text via Ollama or Sentence Transformers
- **Vector Store:** Chroma (SQLite backend)
- **Database:** SQLite for metadata
- **UI:** Streamlit
- **No External APIs:** All processing is local

---

## 2. Architecture Overview

### High-Level Data Flow

```
Wikipedia Data
     ↓
[Ingestion Module]
     ↓
Raw Text Documents
     ↓
[Chunking Module]
     ↓
Text Chunks with Metadata (type: person/place)
     ↓
[Embedding Module]
     ↓
Vectors + Metadata
     ↓
[Chroma Vector Store] (SQLite)
     ↓
User Query → [Retrieval Module] → [Generation Module] → Answer
     ↓
[Streamlit UI]
```

---

## 3. Core Components

### 3.1 Ingestion Module

**Objective:** Collect Wikipedia data for specified entities.

#### Data Sources
- Wikipedia API (via `wikipedia` library)
- **Coverage:** 20+ people (celebrities, scientists, historical figures) and 20+ places (landmarks, natural wonders)

#### Implementation Strategy
- **Batch Processing:** Fetch data for multiple entities
- **Error Handling:** Skip entities with fetch failures; log for review
- **Metadata Tracking:** Store source URL, entity type, and ingestion timestamp
- **Storage Format:** Raw text stored with metadata in SQLite before chunking

#### Minimum Entities Tested
**People:**
- Albert Einstein, Marie Curie, Leonardo da Vinci, William Shakespeare
- Ada Lovelace, Nikola Tesla, Lionel Messi, Cristiano Ronaldo
- Taylor Swift, Frida Kahlo

**Places:**
- Eiffel Tower, Great Wall of China, Taj Mahal, Grand Canyon
- Machu Picchu, Colosseum, Hagia Sophia, Statue of Liberty
- Pyramids of Giza, Mount Everest

---

### 3.2 Chunking Module

**Objective:** Split documents into manageable chunks for embedding and retrieval.

#### Strategy: Fixed-Size Chunks with Overlap

**Rationale:**
- **Simplicity:** Native Python implementation using only built-in modules (re, string operations)
- **Consistency:** Predictable chunk sizes for uniform embeddings
- **Context Preservation:** Overlap ensures continuity across chunk boundaries
- **Scalability:** Works efficiently with large documents
- **No External Dependencies:** Strict adherence to native functionality requirement

#### Configuration
- **Chunk Size:** 512 tokens (~380-400 words)
  - Balances context richness with embedding model capacity
  - Suitable for Chroma and Nomic Embed Text
- **Overlap:** 128 tokens (~95-100 words)
  - Ensures semantic continuity
  - Allows retrieval of context across chunk boundaries
- **Separator:** Sentence-level splitting using Python's built-in `re.split()` with regex pattern for periods, exclamation marks, and question marks

#### Implementation Details

**Tools Used:**
- `re` (Python built-in regex module) for sentence boundary detection
- `str.split()` (Python built-in) for tokenization
- Native list operations for chunk management

**Deliberately Avoided:**
- ❌ NLTK (Natural Language Toolkit) - full-featured NLP library
- ❌ LangChain - high-level framework handling chunking automatically
- ❌ LlamaIndex - RAG framework with built-in chunking
- ❌ spaCy - production NLP library with tokenization
- ❌ Other external text processing libraries

This approach ensures transparency, maintainability, and direct control over the chunking process.

#### Implementation
```python
import re

def chunk_text(text, chunk_size=512, overlap=128):
    """
    Split text into overlapping chunks at sentence boundaries.
    Uses only Python built-in modules (re and string operations).
    
    Args:
        text: Input document text
        chunk_size: Target chunk size in tokens (approximate)
        overlap: Number of tokens to overlap between chunks
    
    Yields:
        Chunks of text with context preservation through overlap
    """
    # Sentence splitting using regex (built-in re module)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    tokens = []
    chunk_start = 0
    
    for sentence in sentences:
        # Tokenization using native str.split() method
        sentence_tokens = sentence.split()
        
        # Check if adding this sentence exceeds chunk size
        if len(tokens) + len(sentence_tokens) > chunk_size:
            if tokens:
                # Yield current chunk
                yield ' '.join(tokens[chunk_start:])
                # Move start position back by overlap amount
                chunk_start = max(0, len(tokens) - overlap)
            # Add current sentence to token pool
            tokens.extend(sentence_tokens)
        else:
            # Add sentence to current chunk
            tokens.extend(sentence_tokens)
    
    # Yield final chunk if there are remaining tokens
    if tokens:
        yield ' '.join(tokens[chunk_start:])
```

---

### 3.3 Embedding and Vector Store

**Objective:** Generate embeddings locally and store with metadata.

#### Design Choice: Option B (Single Vector Store with Metadata)

**Selected:** Option B - Single vector store with entity type metadata

**Rationale:**
1. **Unified Retrieval:** Single-index approach simplifies query logic
2. **Metadata Filtering:** Type-aware retrieval without switching stores
3. **Scalability:** Easier to expand beyond person/place categories
4. **Consistency:** Uniform embedding space for cross-type comparisons
5. **Maintainability:** Simpler codebase and fewer synchronization concerns

**Alternative Rejected:** Option A (Two separate stores)
- Would require parallel queries and result merging
- Reduces flexibility for comparison queries
- Increases operational complexity

#### Embedding Model

**Primary Choice:** Nomic Embed Text via Ollama
- Optimized for local deployment
- ~4GB memory footprint
- High-quality semantic embeddings
- Free and open-source

**Fallback:** Sentence Transformers (`all-MiniLM-L6-v2`)
- Lightweight alternative (~30MB)
- If Ollama unavailable

#### Vector Store: Chroma

**Configuration:**
- **Backend:** SQLite (persistent local storage)
- **Collection:** Single collection `wikipedia_entities`
- **Metadata Fields:**
  - `entity_type`: "person" or "place"
  - `entity_name`: Name of the entity
  - `source_url`: Wikipedia URL
  - `chunk_index`: Position in original document
  - `ingestion_date`: When data was ingested

**Storage Path:** `./data/chroma_db`

#### Metadata Schema
```json
{
  "entity_type": "person | place",
  "entity_name": "Albert Einstein",
  "source_url": "https://en.wikipedia.org/wiki/Albert_Einstein",
  "chunk_index": 0,
  "ingestion_date": "2024-01-15"
}
```

---

### 3.4 Retrieval Module

**Objective:** Route queries intelligently and retrieve relevant context.

#### Query Understanding: Rule-Based Routing

**Approach:** Keyword and heuristic analysis (no ML)

**Implementation:**
1. **Entity Type Classification**
   - **Person Keywords:** "who", "biography", "born", "famous for", "scientist", "artist", "athlete"
   - **Place Keywords:** "where", "location", "visit", "landmark", "country", "city", "structure"
   - **Default:** Both types (for comparison queries)

2. **Query Patterns**
   - Comparison queries: "compare X and Y" → retrieve both entities
   - Mixed queries: "person + place" → retrieve both with type filtering
   - Ambiguous queries: → retrieve from both types, rank by relevance

#### Retrieval Logic

```python
def retrieve_context(query, top_k=5):
    """
    Retrieve relevant chunks with smart type routing.
    """
    # Step 1: Classify query intent
    query_type = classify_query(query)  # "person", "place", or "mixed"
    
    # Step 2: Search vector store
    if query_type == "person":
        results = chroma.query(
            query_texts=[query],
            where={"entity_type": {"$eq": "person"}},
            n_results=top_k
        )
    elif query_type == "place":
        results = chroma.query(
            query_texts=[query],
            where={"entity_type": {"$eq": "place"}},
            n_results=top_k
        )
    else:  # mixed
        results = chroma.query(
            query_texts=[query],
            n_results=top_k * 2
        )
        # Rerank to balance person/place results
        results = balance_results(results, top_k)
    
    # Step 3: Format context
    context = format_retrieved_chunks(results)
    return context, results
```

#### Retrieval Parameters
- **Top-K:** 5 chunks (default)
- **Similarity Threshold:** Optional filtering for low-confidence matches
- **Reranking:** Optional second-stage reranking using Cross-Encoder (future enhancement)

---

### 3.5 Generation Module

**Objective:** Generate answers grounded in retrieved context while minimizing hallucination.

#### Answer Generation Strategy

**Local LLM:** Ollama with Llama 3.2 3b or Mistral

#### Hallucination Prevention

1. **Explicit Context Grounding**
   - Use system prompt to enforce answer grounding
   - Reference specific chunks in explanation

2. **Confidence Markers**
   - Mark uncertain information
   - Return "I don't know" for out-of-context queries

3. **Token Limit:** 200-300 tokens
   - Encourages concise, focused answers
   - Reduces model drift and hallucination

4. **System Prompt**
```
You are a helpful assistant answering questions about famous people and places. 
Answer ONLY based on the provided context. 
If the answer is not in the context, respond with "I don't know".
Be concise and cite specific facts from the context.
Do not make up information.
```

#### Generation Function
```python
def generate_answer(query, context, source_chunks):
    """
    Generate answer using local LLM with context.
    """
    prompt = format_prompt(query, context)
    
    response = ollama.generate(
        model="llama3.2:3b",  # or "mistral"
        prompt=prompt,
        num_predict=250,  # token limit
        temperature=0.3,  # Low for factuality
        top_p=0.9
    )
    
    answer = response['response']
    
    # Optionally append source information
    sources = format_sources(source_chunks)
    
    return answer, sources
```

#### Quality Parameters
- **Temperature:** 0.3 (low for factual consistency)
- **Top-P:** 0.9 (nucleus sampling)
- **Max Tokens:** 250 (concise answers)

---

## 4. Data Flow

### 4.1 Ingestion Flow

```
1. Load entity list (people.txt, places.txt)
   ↓
2. Fetch Wikipedia data for each entity
   ↓
3. Store raw text + metadata in SQLite
   ↓
4. Log ingestion results (successes, failures)
```

### 4.2 Chunking Flow

```
1. Retrieve raw text from SQLite
   ↓
2. Split into chunks (512 tokens, 128 overlap)
   ↓
3. Add chunk-level metadata
   ↓
4. Store chunks in SQLite (chunks table)
```

### 4.3 Embedding and Storage Flow

```
1. Load chunks from SQLite
   ↓
2. Generate embeddings using Nomic Embed Text
   ↓
3. Store in Chroma with metadata
   ↓
4. Verify index completeness
```

### 4.4 Query-to-Answer Flow

```
User Query (Streamlit)
   ↓
1. Classify query type (person/place/mixed)
   ↓
2. Retrieve relevant chunks from Chroma
   ↓
3. Format context
   ↓
4. Generate answer with Ollama
   ↓
5. Display answer + sources
```

---

## 5. Database Schema

### SQLite Tables

#### `raw_documents`
```sql
CREATE TABLE raw_documents (
    id TEXT PRIMARY KEY,
    entity_name TEXT,
    entity_type TEXT,  -- 'person' or 'place'
    source_url TEXT,
    content TEXT,
    word_count INTEGER,
    ingestion_date TIMESTAMP
);
```

#### `chunks`
```sql
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT,
    chunk_index INTEGER,
    content TEXT,
    token_count INTEGER,
    created_at TIMESTAMP,
    FOREIGN KEY(document_id) REFERENCES raw_documents(id)
);
```

---

## 6. User Interface

### Streamlit Chat Interface

**Features:**
1. **Query Input:** Text box for user questions
2. **Answer Display:** Formatted response with typing effect
3. **Context View:** Toggle to show retrieved chunks
4. **Entity Type:** Display inferred entity type
5. **Source Links:** Optional Wikipedia source links
6. **Chat History:** Persistent session memory
7. **Reset Button:** Clear conversation and vector store

**UX Flow:**
```
Welcome Screen
   ↓
User enters query
   ↓
System shows:
   - Answer
   - Confidence indicator
   - Retrieved chunks (optional)
   - Source URLs
   ↓
User can continue chatting or reset
```

---

## 7. Error Handling and Edge Cases

### 7.1 Ingestion Failures
- **Action:** Log failed entities, continue with others
- **Recovery:** Manual re-ingestion available

### 7.2 Out-of-Context Queries
- **Behavior:** Return "I don't know" with suggestion to refine query
- **Example:** "Who is the president of Mars?" → "I don't know. This question is about Mars, which is not in my knowledge base."

### 7.3 Ambiguous Queries
- **Behavior:** Retrieve from both categories, rank by relevance
- **Example:** "Compare Einstein and the Eiffel Tower" → Attempt comparison with cross-category results

### 7.4 Embedding Model Unavailability
- **Fallback:** Switch to Sentence Transformers
- **Degradation:** Slightly lower embedding quality, acceptable tradeoff

---

## 8. Performance Considerations

### Memory Usage
- **Ollama LLM:** ~3.5GB for Llama 3.2 3b
- **Embedding Model:** ~4GB for Nomic or ~30MB for Sentence Transformers
- **Vector Store (Chroma):** ~500MB for ~1000 chunks
- **Total:** ~8GB typical system requirement

### Latency Targets
- **Ingestion:** ~5-10 seconds per entity
- **Embedding:** ~2-5ms per chunk (batch)
- **Retrieval:** <100ms
- **Generation:** ~3-8 seconds per answer
- **Total Query-to-Answer:** ~5-15 seconds

### Optimization Strategies
1. **Batch Embedding:** Process multiple chunks simultaneously
2. **Caching:** Cache frequently asked questions
3. **Approximate Search:** Use Chroma's ANN (Approximate Nearest Neighbor) indexing
4. **GPU Acceleration:** Optional GPU support for Ollama

---

## 9. Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Vector Store Design** | Option B (Single store + metadata) | Unified retrieval, metadata filtering, scalability |
| **Chunking Strategy** | Fixed-size with overlap | Simplicity, consistency, context preservation |
| **Embedding Model** | Nomic Embed Text | Local, high-quality, optimized for retrieval |
| **Routing Strategy** | Rule-based keywords | No ML overhead, transparent, maintainable |
| **Hallucination Prevention** | System prompt + context grounding | Effective without fine-tuning |
| **LLM Temperature** | 0.3 | Factuality over creativity |

---

## 10. Future Enhancements

1. **Advanced Retrieval**
   - Hybrid search (keyword + semantic)
   - Query expansion using local LLM
   - Cross-Encoder reranking

2. **Improved Generation**
   - Few-shot prompting with examples
   - Streaming response for better UX
   - Citation highlighting with source tracking

3. **Scalability**
   - Support for larger document collections
   - Incremental embedding updates
   - Distributed embedding generation

4. **Analysis**
   - Query success metrics
   - Retrieval quality evaluation
   - Answer hallucination detection

5. **Multi-Modal**
   - Image embedding for place queries
   - Portrait recognition for people

---

## 11. Success Criteria

✅ System ingests 20+ people and 20+ places from Wikipedia  
✅ Chunking produces coherent, overlapping chunks  
✅ Embeddings stored with metadata in single Chroma index  
✅ Rule-based routing correctly classifies queries  
✅ Generated answers grounded in retrieved context  
✅ "I don't know" returned for out-of-context queries  
✅ Streamlit UI provides intuitive chat experience  
✅ System runs entirely on localhost with no external APIs  
✅ Demo video demonstrates ingestion and Q&A  
✅ Code is organized, documented, and runnable per README  

---

## 12. Deployment Checklist

- [ ] Python 3.9+ environment set up
- [ ] Ollama installed with models downloaded
- [ ] Chroma and dependencies installed
- [ ] Wikipedia data ingested (20+ people, 20+ places)
- [ ] Embeddings generated and stored
- [ ] Streamlit app tested locally
- [ ] README with clear instructions
- [ ] requirements.txt with all dependencies
- [ ] recommendation.md for production deployment
- [ ] Demo video recorded and linked
- [ ] GitHub repository created and structured
- [ ] Code passes basic functionality tests

---

## Conclusion

This architecture provides a **complete, locally-runnable RAG system** that combines:
- **Intelligent data ingestion** from Wikipedia
- **Smart chunking** with context preservation
- **Unified vector storage** with semantic search
- **Rule-based query routing** for transparency
- **Grounded generation** to minimize hallucination
- **User-friendly interface** for interaction

By leveraging local models and native Python functionality, the system is fully self-contained, transparent, and efficient enough for real-time interactive use.
