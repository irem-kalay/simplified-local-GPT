"""
Local Wikipedia RAG Assistant - Embedding and Vector Store Script

This script reads chunked Wikipedia data from SQLite, generates embeddings using
a local sentence-transformers model, and stores them in a local Chroma vector database.

Key Features:
- Uses sentence-transformers for local embedding generation (no external APIs)
- Persistent Chroma vector store with SQLite backend
- Option B implementation: Single collection with entity_type and entity_name metadata
- Batch processing for memory efficiency
- Full support for semantic search with metadata filtering
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import sqlite3
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import uuid
from typing import List, Dict, Tuple


# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = "data/rag_database.db"
CHROMA_DB_PATH = "data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight, ~30MB
COLLECTION_NAME = "wikipedia_entities"
BATCH_SIZE = 100  # Number of chunks to process per batch


# ============================================================================
# DATABASE CONNECTION
# ============================================================================

def connect_to_database(db_path: str) -> sqlite3.Connection:
    """
    Connect to SQLite database.

    Args:
        db_path: Path to the database file

    Returns:
        SQLite connection object
    """
    try:
        conn = sqlite3.connect(db_path)
        # Enable row factory for easier dictionary access
        conn.row_factory = sqlite3.Row
        print(f"✓ Connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        print(f"✗ Database connection failed: {str(e)}")
        raise


# ============================================================================
# DATA READING
# ============================================================================

def fetch_all_chunks(conn: sqlite3.Connection) -> List[Dict]:
    """
    Fetch all chunks from the SQLite database.

    Args:
        conn: SQLite connection

    Returns:
        List of chunk dictionaries with all metadata
    """
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT 
            id,
            document_id,
            chunk_index,
            entity_name,
            entity_type,
            content,
            token_count
        FROM chunks
        ORDER BY entity_name, chunk_index
        """
    )

    chunks = [dict(row) for row in cursor.fetchall()]
    print(f"✓ Fetched {len(chunks)} chunks from database")
    return chunks


# ============================================================================
# EMBEDDING MODEL INITIALIZATION
# ============================================================================

def initialize_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Initialize a local embedding model using sentence-transformers.

    Uses all-MiniLM-L6-v2 by default - lightweight and efficient:
    - Model size: ~30MB
    - Embedding dimension: 384
    - Inference speed: Fast on CPU
    - Quality: Good for semantic search

    Args:
        model_name: Name of the sentence-transformers model

    Returns:
        Initialized SentenceTransformer model
    """
    print(f"\nInitializing embedding model: {model_name}")
    print(
        "⏳ First run will download the model (~30MB)...\n"
    )

    model = SentenceTransformer(model_name)

    print(f"✓ Embedding model loaded successfully")
    print(f"  - Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


# ============================================================================
# CHROMA INITIALIZATION
# ============================================================================

def initialize_chroma_db(db_path: str, collection_name: str) -> Tuple:
    """
    Initialize Chroma vector database with persistent storage.

    Uses SQLite as the backend for persistent storage. Creates a new
    collection for Option B design: single unified index with metadata.

    Args:
        db_path: Path where Chroma database will be stored
        collection_name: Name of the collection to create

    Returns:
        Tuple of (Chroma client, collection object)
    """
    # Create directory if it doesn't exist
    Path(db_path).mkdir(parents=True, exist_ok=True)

    print(f"\nInitializing Chroma vector database")
    print(f"  - Storage path: {db_path}")
    print(f"  - Backend: SQLite (persistent)")

    # Initialize persistent Chroma client
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Delete collection if it already exists (for clean slate)
    try:
        chroma_client.delete_collection(name=collection_name)
        print(f"  - Cleared existing collection: {collection_name}")
    except Exception:
        pass  # Collection doesn't exist yet

    # Create new collection with metadata support
    # Metadata is used for Option B: filtering by entity_type and entity_name
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # Cosine similarity for semantic search
    )

    print(f"✓ Chroma collection created: {collection_name}")
    return chroma_client, collection


# ============================================================================
# BATCH EMBEDDING AND STORAGE
# ============================================================================

def embed_and_store_batch(
    batch: List[Dict], model: SentenceTransformer, collection
) -> int:
    """
    Generate embeddings for a batch of chunks and insert into Chroma.

    Args:
        batch: List of chunk dictionaries
        model: SentenceTransformer model for embedding
        collection: Chroma collection to insert into

    Returns:
        Number of chunks successfully stored
    """
    if not batch:
        return 0

    # Step 1: Extract chunk texts for embedding
    chunk_texts = [chunk["content"] for chunk in batch]

    # Step 2: Generate embeddings using sentence-transformers
    # This runs locally - no API calls
    embeddings = model.encode(chunk_texts, show_progress_bar=False)

    # Step 3: Prepare data for Chroma insertion
    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(batch):
        chunk_id = f"{chunk['entity_name']}_{chunk['chunk_index']}_{uuid.uuid4().hex[:8]}"
        ids.append(chunk_id)
        documents.append(chunk["content"])

        # Metadata for Option B: Type-aware retrieval
        metadata = {
            "entity_type": chunk["entity_type"],  # "person" or "place"
            "entity_name": chunk["entity_name"],
            "chunk_index": str(chunk["chunk_index"]),
            "token_count": str(chunk["token_count"]),
            "document_id": chunk["document_id"],
        }
        metadatas.append(metadata)

    # Step 4: Insert into Chroma collection
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=documents,
        metadatas=metadatas,
    )

    return len(batch)


def embed_and_store_all_chunks(
    chunks: List[Dict],
    model: SentenceTransformer,
    collection,
    batch_size: int = BATCH_SIZE,
) -> Tuple[int, int]:
    """
    Process all chunks in batches: embed and store in Chroma.

    Args:
        chunks: List of all chunk dictionaries
        model: SentenceTransformer model
        collection: Chroma collection
        batch_size: Number of chunks per batch

    Returns:
        Tuple of (total_chunks_stored, total_tokens_processed)
    """
    print(f"\n" + "=" * 70)
    print("EMBEDDING AND STORING CHUNKS")
    print("=" * 70 + "\n")

    total_stored = 0
    total_tokens = 0

    # Process in batches
    num_batches = (len(chunks) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(chunks))

        batch = chunks[start_idx:end_idx]
        batch_tokens = sum(chunk["token_count"] for chunk in batch)

        # Embed and store batch
        stored_count = embed_and_store_batch(batch, model, collection)

        total_stored += stored_count
        total_tokens += batch_tokens

        # Progress indicator
        print(
            f"✓ Batch {batch_idx + 1}/{num_batches}: {stored_count:3d} chunks embedded ({batch_tokens:5d} tokens)"
        )

    return total_stored, total_tokens


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_chroma_storage(collection) -> Dict:
    """
    Verify that data was successfully stored in Chroma.

    Args:
        collection: Chroma collection to verify

    Returns:
        Dictionary with storage statistics
    """
    stats = collection.count()

    print(f"\n" + "=" * 70)
    print("CHROMA VERIFICATION")
    print("=" * 70)
    print(f"Total vectors stored: {stats}")

    # Sample a few entries to verify metadata
    try:
        sample = collection.get(limit=3)
        if sample["ids"]:
            print(f"\nSample entries:")
            for i, (doc_id, metadata) in enumerate(
                zip(sample["ids"], sample["metadatas"]), 1
            ):
                print(
                    f"  {i}. {metadata['entity_name']} (Type: {metadata['entity_type']}) - Chunk {metadata['chunk_index']}"
                )
    except Exception as e:
        print(f"⚠ Could not sample entries: {str(e)}")

    return {"total_vectors": stats}


# ============================================================================
# QUERY TESTING
# ============================================================================

def test_retrieval(collection, model: SentenceTransformer):
    """
    Test retrieval with a sample query to verify the system works end-to-end.

    Args:
        collection: Chroma collection
        model: SentenceTransformer model
    """
    print(f"\n" + "=" * 70)
    print("RETRIEVAL TEST")
    print("=" * 70 + "\n")

    # Test queries
    test_queries = [
        ("Who was Albert Einstein?", "person"),
        ("Where is the Eiffel Tower?", "place"),
        ("Compare Messi and Ronaldo", "person"),
    ]

    for query, expected_type in test_queries:
        print(f"Query: {query}")
        print(f"Expected type: {expected_type}")

        # Embed the query
        query_embedding = model.encode([query])[0].tolist()

        # Search with type filter
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            where={"entity_type": expected_type},
        )

        if results["ids"]:
            print(f"✓ Retrieved {len(results['ids'][0])} results:")
            for i, (doc, metadata) in enumerate(
                zip(results["documents"][0], results["metadatas"][0]), 1
            ):
                entity_name = metadata["entity_name"]
                chunk_idx = metadata["chunk_index"]
                print(
                    f"  {i}. {entity_name} (Chunk {chunk_idx}): {doc[:80]}..."
                )
        else:
            print("✗ No results found")

        print()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main pipeline: Connect → Read → Embed → Store → Verify
    """
    print("\n" + "=" * 70)
    print("LOCAL WIKIPEDIA RAG - EMBEDDING AND VECTOR STORAGE")
    print("=" * 70)

    try:
        # Step 1: Connect to SQLite database
        conn = connect_to_database(DB_PATH)

        # Step 2: Fetch all chunks
        chunks = fetch_all_chunks(conn)
        conn.close()

        if not chunks:
            print("\n✗ No chunks found in database. Run ingest.py first.")
            return

        # Step 3: Initialize embedding model (local, no APIs)
        model = initialize_embedding_model(EMBEDDING_MODEL)

        # Step 4: Initialize Chroma vector database
        chroma_client, collection = initialize_chroma_db(CHROMA_DB_PATH, COLLECTION_NAME)

        # Step 5: Embed and store all chunks in batches
        total_stored, total_tokens = embed_and_store_all_chunks(
            chunks, model, collection, batch_size=BATCH_SIZE
        )

        # Step 6: Verify storage
        verify_chroma_storage(collection)

        # Step 7: Test retrieval with sample queries
        test_retrieval(collection, model)

        # Step 8: Print final summary
        print("=" * 70)
        print("EMBEDDING AND STORAGE SUMMARY")
        print("=" * 70)
        print(f"Total chunks embedded:    {total_stored}")
        print(f"Total tokens processed:   {total_tokens}")
        print(f"Avg tokens per chunk:     {total_tokens // total_stored if total_stored > 0 else 0}")
        print(f"Chroma database location: {CHROMA_DB_PATH}")
        print(f"Embedding model:          {EMBEDDING_MODEL}")
        print(f"Vector dimension:         {model.get_sentence_embedding_dimension()}")
        print("=" * 70 + "\n")

        print("✓ Embedding and storage complete!")
        print("✓ Ready for query retrieval and generation.\n")

    except Exception as e:
        print(f"\n✗ Pipeline failed: {str(e)}")
        raise


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
