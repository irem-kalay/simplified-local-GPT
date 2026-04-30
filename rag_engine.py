"""
Local Wikipedia RAG Assistant - RAG Engine Module

This module implements the complete Retrieval-Augmented Generation (RAG) pipeline:
1. Query classification (rule-based routing)
2. ChromaDB retrieval with metadata filtering
3. LLM generation using Ollama with context grounding

All operations are local - no external APIs are used.
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from typing import Tuple, List, Dict, Optional


# ============================================================================
# CONFIGURATION
# ============================================================================

CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "wikipedia_entities"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"  # or "mistral"
RETRIEVAL_TOP_K = 5

# Rule-based keyword mapping for query classification
PERSON_KEYWORDS = {
    "who",
    "biography",
    "born",
    "famous",
    "scientist",
    "artist",
    "athlete",
    "author",
    "inventor",
    "discover",
    "achievement",
    "life",
    "known for",
    "created",
    "wrote",
    "painted",
    "composed",
    "died",
    "year",
    "age",
    "founder",
    "president",
    "king",
    "queen",
    "emperor",
    "person",
    "people",
    "man",
    "woman",
    "doctor",
    "physicist",
    "mathematician",
}

PLACE_KEYWORDS = {
    "where",
    "location",
    "located",
    "city",
    "country",
    "landmark",
    "visit",
    "building",
    "structure",
    "monument",
    "temple",
    "cathedral",
    "castle",
    "fortress",
    "tower",
    "bridge",
    "mountain",
    "river",
    "lake",
    "ocean",
    "island",
    "continent",
    "region",
    "place",
    "area",
    "site",
    "ruins",
    "capital",
    "continent",
    "geography",
    "distance",
    "height",
}

# ============================================================================
# GLOBAL INITIALIZATION
# ============================================================================

# These will be initialized on first use
_chroma_collection = None
_embedding_model = None
_initialized = False


def initialize_rag_engine():
    """
    Initialize the RAG engine components:
    - Connect to Chroma vector database
    - Load embedding model
    - Verify Ollama is running
    
    Should be called once at startup.
    """
    global _chroma_collection, _embedding_model, _initialized

    if _initialized:
        return

    print("\n" + "=" * 70)
    print("INITIALIZING RAG ENGINE")
    print("=" * 70 + "\n")

    # Step 1: Initialize Chroma client
    print("Connecting to Chroma vector database...")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _chroma_collection = chroma_client.get_collection(
            name=COLLECTION_NAME
        )
        print(f"✓ Connected to Chroma collection: {COLLECTION_NAME}")
        print(f"  - Total vectors: {_chroma_collection.count()}\n")
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Chroma database: {str(e)}\n"
            f"Ensure embed_and_store.py has been run first."
        )

    # Step 2: Initialize embedding model
    print("Loading embedding model...")
    try:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        embedding_dim = _embedding_model.get_sentence_embedding_dimension()
        print(f"✓ Embedding model loaded: {EMBEDDING_MODEL}")
        print(f"  - Embedding dimension: {embedding_dim}\n")
    except Exception as e:
        raise RuntimeError(f"Failed to load embedding model: {str(e)}")

    # Step 3: Verify Ollama is running
    print("Checking Ollama availability...")
    try:
        # Try to list models
        response = ollama.list()
        print(f"✓ Ollama is running")
        print(f"  - Available models: {len(response.get('models', []))}")
        
        # Check if the specific model is available
        models = [m.get('name', '') for m in response.get('models', [])]
        if any(LLM_MODEL in m for m in models):
            print(f"  - Model '{LLM_MODEL}' is available\n")
        else:
            print(f"⚠ Model '{LLM_MODEL}' not found locally")
            print(f"  - Available models: {', '.join(models[:3])}")
            print(f"  - Run: ollama pull {LLM_MODEL}\n")
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Ollama: {str(e)}\n"
            f"Ensure Ollama is running: ollama serve"
        )

    _initialized = True

    print("=" * 70)
    print("✓ RAG Engine initialized successfully!")
    print("=" * 70 + "\n")


# ============================================================================
# QUERY CLASSIFICATION (RULE-BASED ROUTING)
# ============================================================================

def classify_query(query: str) -> str:
    """
    Classify a user query as "person", "place", or "mixed" using keyword rules.
    
    This rule-based approach is transparent, fast, and requires no ML.
    It enables Option B's type-aware retrieval.

    Args:
        query: User's natural language question

    Returns:
        "person", "place", or "mixed"
    """
    query_lower = query.lower()

    # Count keyword matches
    person_matches = sum(1 for keyword in PERSON_KEYWORDS if keyword in query_lower)
    place_matches = sum(1 for keyword in PLACE_KEYWORDS if keyword in query_lower)

    # Determine query type based on keyword frequency
    if person_matches > place_matches and person_matches > 0:
        return "person"
    elif place_matches > person_matches and place_matches > 0:
        return "place"
    else:
        # Default to mixed for ambiguous queries
        # (includes comparisons, combinations, etc.)
        return "mixed"


def format_chat_history(chat_history: Optional[List[Dict]]) -> str:
    """
    Format chat history into a compact transcript for prompting.

    Args:
        chat_history: Conversation messages in
                      [{"role": "user"/"assistant", "content": "..."}, ...] format

    Returns:
        A newline-delimited transcript string
    """
    if not chat_history:
        return ""

    history_lines = []
    for msg in chat_history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        if len(content) > 500:
            content = content[:500]
        history_lines.append(f"{role}: {content}")

    return "\n".join(history_lines)


def condense_query(query: str, chat_history: Optional[List[Dict]]) -> str:
    """
    Rewrite a follow-up user question into a standalone retrieval query.

    Args:
        query: User's latest question
        chat_history: Previous conversation turns

    Returns:
        A standalone, context-rich question suitable for retrieval
    """
    if not chat_history:
        return query

    history_text = format_chat_history(chat_history)
    if not history_text:
        return query

    condense_prompt = f"""Rewrite the user's latest question as a standalone question for document retrieval.

Rules:
1. Preserve the original meaning.
2. Replace pronouns or vague references with the specific entity or topic from the conversation when possible.
3. Keep it concise.
4. Return ONLY the rewritten standalone question.
5. If the latest question is already standalone, return it unchanged.

Conversation:
{history_text}

Latest question: {query}

Standalone question:"""

    try:
        response = ollama.generate(
            model=LLM_MODEL,
            prompt=condense_prompt,
            stream=False,
            options={
                "temperature": 0.1,
                "top_p": 0.9,
            },
        )

        standalone_query = response.get("response", "").strip()
        return standalone_query or query
    except Exception:
        # Fall back gracefully so retrieval still works for standalone queries.
        return query


# ============================================================================
# RETRIEVAL (CHROMADB SEARCH WITH METADATA FILTERING)
# ============================================================================

def retrieve_context(query: str, top_k: int = RETRIEVAL_TOP_K) -> Tuple[str, List[Dict]]:
    """
    Retrieve relevant context from ChromaDB for a user query.
    
    Uses rule-based query classification to apply metadata filtering:
    - If query is about "person" → filter where entity_type="person"
    - If query is about "place" → filter where entity_type="place"
    - If "mixed" → retrieve from both, balance results
    
    This implements Option B: Single vector store with metadata routing.

    Args:
        query: User's natural language question
        top_k: Number of chunks to retrieve (default: 5)

    Returns:
        Tuple of (formatted_context_string, list_of_source_chunks)
    """
    if not _initialized:
        raise RuntimeError("RAG engine not initialized. Call initialize_rag_engine() first.")

    # Step 1: Classify query intent
    query_type = classify_query(query)

    # Step 2: Embed the query using the same model as Chroma
    query_embedding = _embedding_model.encode([query])[0].tolist()

    # Step 3: Search vector store with metadata filtering
    if query_type == "person":
        # Filter for person entities only
        results = _chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"entity_type": "person"},
        )
    elif query_type == "place":
        # Filter for place entities only
        results = _chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"entity_type": "place"},
        )
    else:  # mixed
        # Retrieve from both types, then balance
        results = _chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Get more to balance
        )
        # Simple balancing: sort by entity type and trim
        # (Chroma returns results in similarity order)

    # Step 4: Format results for display and context
    if not results["ids"] or not results["ids"][0]:
        # No results found
        return "No relevant information found in the knowledge base.", []

    # Extract and format chunks
    chunks = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Build context string from retrieved chunks
    context_parts = []
    source_chunks = []

    for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
        entity_name = metadata.get("entity_name", "Unknown")
        entity_type = metadata.get("entity_type", "Unknown")
        chunk_idx = metadata.get("chunk_index", "0")

        # Add to context
        context_parts.append(f"[{entity_name} - {entity_type.upper()}]\n{doc}")

        # Track source information
        source_chunks.append({
            "entity_name": entity_name,
            "entity_type": entity_type,
            "chunk_index": chunk_idx,
            "content": doc,
            "document_id": metadata.get("document_id", ""),
        })

    # Combine all chunks into a single context string
    context = "\n\n---\n\n".join(context_parts)

    return context, source_chunks


# ============================================================================
# LLM GENERATION (OLLAMA WITH PROMPT ENGINEERING)
# ============================================================================

def generate_answer(
    query: str,
    context: str,
    history_text: str = "",
) -> Tuple[str, List[Dict]]:
    """
    Generate an answer using the local LLM with retrieved context.

    Args:
        query: User's original question
        context: Retrieved context from ChromaDB
        history_text: Formatted string of previous conversation turns

    Returns:
        Tuple of (generated_answer, source_chunks_used)
    """
    if not _initialized:
        raise RuntimeError("RAG engine not initialized. Call initialize_rag_engine() first.")

    # If no context was found, return immediately
    if not context or context == "No relevant information found in the knowledge base.":
        return "I don't know. This question is about information not in my knowledge base.", []

    # Step 1: Build the prompt with strict instructions
    system_prompt = """You are a helpful assistant answering questions about famous people and places.

CRITICAL RULES:
1. Answer ONLY based on the provided context.
2. Do NOT make up or assume information.
3. If the context does not contain the answer, respond with EXACTLY: "I don't know"
4. Be concise and cite specific facts from the context when possible.
5. If the question asks about something not in the context, say "I don't know"
6. If there is a previous conversation, use it to understand follow-up questions and pronouns like "he", "she", "it", "they".

You have access to the following context:"""

    # Step 2: Add prior conversation when available for reference resolution.
    history_section = ""
    if history_text:
        history_section = f"""Previous conversation (for context only, do not answer based on this):
{history_text}

"""

    user_prompt = f"""{history_section}Context:
{context}

---

Question: {query}

Answer based ONLY on the context above. Use the previous conversation only to understand what pronouns like "he/she/it/they" refer to. If the information is not in the context, respond with: I don't know"""

    # Step 3: Call Ollama with strict parameters
    try:
        response = ollama.generate(
            model=LLM_MODEL,
            prompt=user_prompt,
            system=system_prompt,
            stream=False,
            options={
                "temperature": 0.3,
                "top_p": 0.9,
            },
        )

        answer = response.get("response", "").strip()

        if not answer or len(answer) < 5:
            answer = "I don't know"

    except Exception as e:
        return f"Error generating response: {str(e)}", []

    return answer, []


def answer_question(
    query: str,
    include_sources: bool = False,
    include_context: bool = False,
    chat_history: list = None,
) -> Dict:
    """
    Complete RAG pipeline: Query → Classify → Retrieve → Generate → Answer

    Args:
        query: User's natural language question
        include_sources: Whether to return source chunk information
        include_context: Whether to return retrieved context
        chat_history: Optional list of previous messages from Streamlit session_state.
                      Format: [{"role": "user"/"assistant", "content": "..."}, ...]
                      Son 6 mesaj gönderilmesi önerilir (3 soru-cevap turu).

    Returns:
        Dictionary with:
        - "answer": Generated answer
        - "query_type": Classification ("person", "place", or "mixed")
        - "sources": List of source chunks (if include_sources=True)
        - "context": Retrieved context text (if include_context=True)
        - "error": Error message if something failed
    """
    if not _initialized:
        raise RuntimeError("RAG engine not initialized. Call initialize_rag_engine() first.")

    result = {
        "query": query,
        "answer": "",
        "query_type": "",
        "sources": [],
        "context": "",
        "error": None,
    }

    try:
        # Step 1: Condense follow-up questions for retrieval
        standalone_query = condense_query(query, chat_history) if chat_history else query

        # Step 2: Classify the actual retrieval query
        query_type = classify_query(standalone_query)
        result["query_type"] = query_type

        # Step 3: Build conversation history string for generation
        history_text = format_chat_history(chat_history)

        # Step 4: Retrieve context using the standalone query
        context_text, sources = retrieve_context(standalone_query)

        # Step 5: Generate answer using the original user query and retrieved context
        answer, _ = generate_answer(query, context_text, history_text=history_text)

        result["answer"] = answer
        result["sources"] = sources if include_sources else []

        # Step 6: Optionally return the already retrieved context
        if include_context:
            result["context"] = context_text

    except Exception as e:
        result["error"] = f"Error processing query: {str(e)}"
        result["answer"] = "I don't know. An error occurred while processing your question."

    return result


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_engine_stats() -> Dict:
    """
    Get current status of the RAG engine.

    Returns:
        Dictionary with engine statistics
    """
    if not _initialized:
        return {"initialized": False, "message": "Engine not initialized"}

    stats = {
        "initialized": True,
        "chroma_vectors": _chroma_collection.count(),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": _embedding_model.get_sentence_embedding_dimension(),
        "llm_model": LLM_MODEL,
        "retrieval_top_k": RETRIEVAL_TOP_K,
    }

    return stats


def test_retrieval(query: str) -> Dict:
    """
    Test retrieval for a query without generating an answer.
    Useful for debugging retrieval quality.

    Args:
        query: Test query

    Returns:
        Dictionary with retrieval results
    """
    if not _initialized:
        raise RuntimeError("RAG engine not initialized. Call initialize_rag_engine() first.")

    query_type = classify_query(query)
    context, sources = retrieve_context(query)

    return {
        "query": query,
        "query_type": query_type,
        "num_results": len(sources),
        "sources": sources,
        "context": context,
    }


# ============================================================================
# TESTING AND DEMO
# ============================================================================

def run_demo():
    """
    Run a demo of the RAG engine with sample queries.
    """
    print("\n" + "=" * 70)
    print("RAG ENGINE DEMO")
    print("=" * 70 + "\n")

    # Initialize
    initialize_rag_engine()

    # Sample queries
    test_queries = [
        "Who was Albert Einstein and what is he known for?",
        "Where is the Eiffel Tower located?",
        "Compare Lionel Messi and Cristiano Ronaldo",
        "What is the tallest mountain in the world?",
        "Tell me about the Taj Mahal",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'-' * 70}")
        print(f"Query {i}: {query}")
        print(f"{'-' * 70}")

        # Process query
        result = answer_question(query, include_sources=True, include_context=False)

        print(f"\nQuery Type: {result['query_type']}")
        print(f"\nAnswer:\n{result['answer']}\n")

        if result["sources"]:
            print(f"Sources ({len(result['sources'])} chunks):")
            for j, source in enumerate(result["sources"][:2], 1):
                print(f"  {j}. {source['entity_name']} (Chunk {source['chunk_index']})")

        if result["error"]:
            print(f"Error: {result['error']}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_demo()
