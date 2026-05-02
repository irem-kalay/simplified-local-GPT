"""
Local Wikipedia RAG Assistant - Data Ingestion Script

This script ingests Wikipedia data for famous people and places, chunks the content
using native Python functionality, and stores everything in a local SQLite database.

Key Features:
- Uses wikipedia library for data fetching
- Native chunking with re module (no NLTK/LangChain/LlamaIndex)
- SQLite database for persistent storage
- Graceful error handling for disambiguation and missing pages
- Metadata tracking for Option B vector store routing
"""

import sqlite3
import wikipedia
import re
import uuid
import time
from datetime import datetime
from pathlib import Path
from typing import Generator, Tuple, List, Dict, Optional

# added User Agent to let know Wikipedia that I am not a bot
wikipedia.set_user_agent("LocalRAGProject_ITU/1.0 (kalayi22@itu.edu.tr)")


# ============================================================================
# CONFIGURATION
# ============================================================================

DB_PATH = "data/rag_database.db"
CHUNK_SIZE = 512  # Approximate token count per chunk
OVERLAP = 128  # Token overlap between chunks

# Entities to ingest: 20 people + 20 places (minimum) + 20 additional (reach 40 total)

PEOPLE = [
    # Minimum 20 people from PRD
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    # Additional 10 to reach 40 total
    "Steve Jobs",
    "Oprah Winfrey",
    "Martin Luther King Jr.",
    "Cleopatra",
    "Isaac Newton",
    "Stephen Hawking",
    "Elon Musk",
    "Serena Williams",
    "Pablo Picasso",
    "Jane Goodall",
]

PLACES = [
    # Minimum 20 places from PRD
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Pyramids of Giza",
    "Mount Everest",
    # Additional 10 to reach 40 total
    "Big Ben",
    "Angkor Wat",
    "Petra",
    "Sirmione",
    "Stonehenge",
    "Venice",
    "Acropolis",
    "Niagara Falls",
    "Sydney Opera House",
    "Kremlin",
]


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def initialize_database():
    """
    Initialize SQLite database with raw_documents and chunks tables.
    Creates database directory if it doesn't exist.
    """
    # Create data directory if it doesn't exist
    Path("data").mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Drop existing tables if they exist (for clean slate)
    cursor.execute("DROP TABLE IF EXISTS chunks")
    cursor.execute("DROP TABLE IF EXISTS raw_documents")

    # Create raw_documents table
    cursor.execute(
        """
        CREATE TABLE raw_documents (
            id TEXT PRIMARY KEY,
            entity_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            source_url TEXT NOT NULL,
            content TEXT NOT NULL,
            word_count INTEGER,
            ingestion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Create chunks table
    cursor.execute(
        """
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            entity_name TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            content TEXT NOT NULL,
            token_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(document_id) REFERENCES raw_documents(id)
        )
        """
    )

    conn.commit()
    conn.close()

    print(f"✓ Database initialized at {DB_PATH}")


# ============================================================================
# WIKIPEDIA FETCHING
# ============================================================================

def fetch_wikipedia_content(entity_name: str, entity_type: str) -> Optional[Dict]:
    """
    Fetch Wikipedia content for a given entity with robust error handling.

    Features:
    - Rate limiting (2-second delay between requests)
    - Retry logic (up to 3 attempts)
    - Fallback using wikipedia.search() if page lookup fails
    - Graceful handling of disambiguation and page not found errors

    Args:
        entity_name: Name of the entity (person or place)
        entity_type: Type of entity ('person' or 'place')

    Returns:
        Dictionary with content and metadata, or None if fetch failed
    """
    max_retries = 3
    retry_delay = 1.0  # seconds between retries

    for attempt in range(max_retries):
        try:
            # Rate limiting: Add delay between Wikipedia API requests
            time.sleep(2.0)

            # Try to fetch the page with auto_suggest disabled
            page = wikipedia.page(entity_name, auto_suggest=False)
            content = page.content
            url = page.url

            word_count = len(content.split())

            print(f"✓ Fetched {entity_type}: {entity_name} ({word_count} words)")

            return {
                "entity_name": entity_name,
                "entity_type": entity_type,
                "content": content,
                "url": url,
                "word_count": word_count,
            }

        except wikipedia.exceptions.DisambiguationError as e:
            # Disambiguation occurred: try the first suggestion
            if e.options:
                suggested_name = e.options[0]
                print(
                    f"⚠ Disambiguation for '{entity_name}': Trying '{suggested_name}'"
                )
                entity_name = suggested_name
                continue  # Retry with suggested name
            else:
                print(f"✗ Disambiguation error for {entity_name}: No suggestions available")
                return None

        except wikipedia.exceptions.PageError:
            # Page not found: fallback to search
            print(f"⚠ Page not found: '{entity_name}'. Attempting search fallback...")
            search_results = wikipedia.search(entity_name, results=1)

            if search_results:
                suggested_name = search_results[0]
                print(f"⚠ Fallback: Trying '{suggested_name}'")
                entity_name = suggested_name
                continue  # Retry with search result
            else:
                print(f"✗ No search results found for {entity_name}")
                return None

        except Exception as e:
            # Generic exception (includes JSONDecodeError from rate limiting)
            if attempt < max_retries - 1:
                print(
                    f"⚠ Error fetching {entity_name} (attempt {attempt + 1}/{max_retries}): {type(e).__name__}"
                )
                time.sleep(retry_delay)  # Wait before retrying
                continue
            else:
                print(f"✗ Failed to fetch {entity_name} after {max_retries} attempts: {str(e)}")
                return None

    return None


def fetch_all_entities(
    people: List[str], places: List[str]
) -> Tuple[List[Dict], List[Dict]]:
    """
    Fetch Wikipedia content for all entities.

    Returns:
        Tuple of (people_data, places_data)
    """
    print("\n" + "=" * 70)
    print("FETCHING WIKIPEDIA DATA")
    print("=" * 70 + "\n")

    people_data = []
    places_data = []

    print("Fetching People:")
    print("-" * 70)
    for person in people:
        data = fetch_wikipedia_content(person, "person")
        if data:
            people_data.append(data)

    print(f"\n✓ Successfully fetched {len(people_data)}/{len(people)} people\n")

    print("Fetching Places:")
    print("-" * 70)
    for place in places:
        data = fetch_wikipedia_content(place, "place")
        if data:
            places_data.append(data)

    print(f"\n✓ Successfully fetched {len(places_data)}/{len(places)} places\n")

    return people_data, places_data


# ============================================================================
# NATIVE CHUNKING (USING BUILT-IN re AND STRING OPERATIONS ONLY)
# ============================================================================

def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP
) -> Generator[Tuple[str, int], None, None]:
    """
    Split text into overlapping chunks at sentence boundaries.

    Uses ONLY Python built-in modules:
    - re (regex module) for sentence splitting
    - str.split() for tokenization
    - Native list operations for chunk management

    DELIBERATELY AVOIDS external libraries like NLTK, LangChain, LlamaIndex.

    Args:
        text: Input document text
        chunk_size: Target chunk size in tokens (approximate)
        overlap: Number of tokens to overlap between chunks

    Yields:
        Tuple of (chunk_text, token_count) for each chunk
    """
    # Step 1: Split text into sentences using regex (built-in re module)
    # This regex splits on sentence boundaries: period, exclamation, question mark
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return

    tokens = []
    chunk_start = 0

    # Step 2: Iteratively build chunks
    for sentence in sentences:
        # Tokenize sentence using native str.split() method
        sentence_tokens = sentence.split()

        # Check if adding this sentence exceeds chunk_size
        if len(tokens) + len(sentence_tokens) > chunk_size:
            # Yield current chunk if we have tokens
            if tokens:
                chunk_text_str = " ".join(tokens[chunk_start:])
                chunk_token_count = len(tokens) - chunk_start
                yield chunk_text_str, chunk_token_count

                # Move chunk_start back by overlap amount for next chunk
                chunk_start = max(0, len(tokens) - overlap)

            # Add current sentence to token pool
            tokens.extend(sentence_tokens)
        else:
            # Add sentence to current chunk
            tokens.extend(sentence_tokens)

    # Step 3: Yield final chunk if there are remaining tokens
    if tokens:
        chunk_text_str = " ".join(tokens[chunk_start:])
        chunk_token_count = len(tokens) - chunk_start
        yield chunk_text_str, chunk_token_count


# ============================================================================
# DATABASE STORAGE
# ============================================================================

def store_raw_document(
    entity_name: str,
    entity_type: str,
    content: str,
    url: str,
    word_count: int,
    conn: sqlite3.Connection,
) -> str:
    """
    Store raw document in database and return its ID.

    Args:
        entity_name: Name of the entity
        entity_type: Type of entity ('person' or 'place')
        content: Full text content
        url: Source URL
        word_count: Word count of content
        conn: SQLite connection

    Returns:
        Document ID (UUID)
    """
    doc_id = str(uuid.uuid4())
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO raw_documents (id, entity_name, entity_type, source_url, content, word_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (doc_id, entity_name, entity_type, url, content, word_count),
    )

    conn.commit()
    return doc_id


def store_chunks(
    document_id: str,
    entity_name: str,
    entity_type: str,
    chunks: List[Tuple[str, int]],
    conn: sqlite3.Connection,
) -> int:
    """
    Store chunks in database.

    Args:
        document_id: ID of the raw document
        entity_name: Name of the entity
        entity_type: Type of entity
        chunks: List of (chunk_text, token_count) tuples
        conn: SQLite connection

    Returns:
        Number of chunks stored
    """
    cursor = conn.cursor()
    chunk_count = 0

    for chunk_index, (chunk_text, token_count) in enumerate(chunks):
        chunk_id = str(uuid.uuid4())

        cursor.execute(
            """
            INSERT INTO chunks (id, document_id, chunk_index, entity_name, entity_type, content, token_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk_id,
                document_id,
                chunk_index,
                entity_name,
                entity_type,
                chunk_text,
                token_count,
            ),
        )

        chunk_count += 1

    conn.commit()
    return chunk_count


# ============================================================================
# INGESTION PIPELINE
# ============================================================================

def process_and_store_entity(
    entity_data: Dict, conn: sqlite3.Connection
) -> Tuple[int, int]:
    """
    Process a single entity: store raw doc, chunk, and store chunks.

    Args:
        entity_data: Dictionary with entity information
        conn: SQLite connection

    Returns:
        Tuple of (chunks_created, total_tokens)
    """
    # Step 1: Store raw document
    doc_id = store_raw_document(
        entity_data["entity_name"],
        entity_data["entity_type"],
        entity_data["content"],
        entity_data["url"],
        entity_data["word_count"],
        conn,
    )

    # Step 2: Chunk the text using native Python
    chunks = list(chunk_text(entity_data["content"]))

    # Step 3: Store chunks
    chunks_count = store_chunks(
        doc_id, entity_data["entity_name"], entity_data["entity_type"], chunks, conn
    )

    # Calculate total tokens
    total_tokens = sum(token_count for _, token_count in chunks)

    return chunks_count, total_tokens


def ingest_all_data():
    """
    Main ingestion pipeline: fetch, chunk, and store all entities.
    """
    print("\n" + "=" * 70)
    print("LOCAL WIKIPEDIA RAG - DATA INGESTION")
    print("=" * 70)

    # Step 1: Initialize database
    initialize_database()

    # Step 2: Fetch Wikipedia data
    people_data, places_data = fetch_all_entities(PEOPLE, PLACES)

    all_data = people_data + places_data

    if not all_data:
        print("\n✗ No data fetched. Exiting.")
        return

    # Step 3: Process and store all entities
    print("\n" + "=" * 70)
    print("CHUNKING AND STORING DATA")
    print("=" * 70 + "\n")

    conn = sqlite3.connect(DB_PATH)

    total_documents = 0
    total_chunks = 0
    total_tokens = 0

    for entity_data in all_data:
        chunks_count, chunk_tokens = process_and_store_entity(entity_data, conn)
        total_documents += 1
        total_chunks += chunks_count
        total_tokens += chunk_tokens

        print(
            f"✓ {entity_data['entity_name']:<30} → {chunks_count:3d} chunks ({chunk_tokens:5d} tokens)"
        )

    conn.close()

    # Step 4: Print summary
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"Documents ingested:     {total_documents}")
    print(f"Total chunks created:   {total_chunks}")
    print(f"Total tokens:           {total_tokens}")
    print(f"Avg tokens per chunk:   {total_tokens // total_chunks if total_chunks > 0 else 0}")
    print(f"Database location:      {DB_PATH}")
    print("=" * 70 + "\n")

    print("✓ Ingestion complete! Ready for embedding and retrieval.\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    ingest_all_data()