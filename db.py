"""Database operations for document ingestion and retrieval.

This module provides the core database layer for storing documents, chunks,
embeddings, and metadata. It handles PostgreSQL operations with pgvector
support for semantic search.
"""

import json
import os
import uuid
from typing import Any, Optional

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


def _database_target() -> str:
    """Return a safe, password-free description of the configured DB target."""
    url = os.getenv("PSYCOPG_URL")
    if url:
        try:
            from urllib.parse import urlsplit

            parsed = urlsplit(url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 5432
            database = parsed.path.lstrip("/") or "unknown"
            return f"{host}:{port}/{database}"
        except Exception:
            return "PSYCOPG_URL configured"

    return (
        f"{os.getenv('DB_HOST', 'localhost')}:"
        f"{os.getenv('DB_PORT', '5432')}/"
        f"{os.getenv('DB_NAME', 'smart_banking')}"
    )


def get_db_connection() -> psycopg.Connection:
    """Create and return a PostgreSQL database connection.

    Uses PSYCOPG_URL from .env, for example:

        PSYCOPG_URL=postgresql://postgres:<password>@localhost:5433/banking

    PSYCOPG_URL is preferred because it contains the complete connection
    information, including the non-default PostgreSQL port and password.

    Falls back to the DB_* environment variables if PSYCOPG_URL is not set.
    """
    load_dotenv()

    psycopg_url = os.getenv("PSYCOPG_URL")

    if psycopg_url:
        conn = psycopg.connect(psycopg_url)
    else:
        conn = psycopg.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("DB_NAME", "smart_banking"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )

    # Register pgvector type support
    register_vector(conn)
    return conn


def upsert_document(filename: str, filepath: str) -> str:
    """Register or update a document in the database.

    Inserts a new document record or updates an existing one with the same
    filename. This ensures that re-ingesting the same document reuses the
    same doc_id, allowing old chunks to be cleaned up if needed.

    Args:
        filename: The display name of the document (e.g., "report.pdf").
        filepath: The full file system path to the document.

    Returns:
        The document UUID (doc_id) for use in chunk ingestion.

    Raises:
        psycopg.Error: If the database operation fails.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Check if document already exists
            cur.execute(
                "SELECT doc_id FROM documents WHERE filename = %s",
                (filename,),
            )
            result = cur.fetchone()

            if result:
                # Update existing document
                doc_id = result[0]
                cur.execute(
                    """
                    UPDATE documents
                    SET filepath = %s, updated_at = NOW()
                    WHERE doc_id = %s
                    """,
                    (filepath, doc_id),
                )
                print(f"[db] Updated existing document: {doc_id}")
            else:
                # Create new document
                doc_id = str(uuid.uuid4())
                cur.execute(
                    """
                    INSERT INTO documents (doc_id, filename, filepath)
                    VALUES (%s, %s, %s)
                    """,
                    (doc_id, filename, filepath),
                )
                print(f"[db] Created new document: {doc_id}")

            conn.commit()
            return doc_id
    finally:
        conn.close()


def store_chunks(chunks: list[dict], doc_id: str) -> int:
    """Store parsed document chunks with embeddings in the database.

    Each chunk is embedded using a sentence transformer model, then stored
    in the `multimodal_chunks` table with its embedding vector, metadata,
    and optional image bytes (for image chunks).

    Args:
        chunks: List of dicts, each with:
          {
            "content": str,
            "content_type": "text" | "table" | "image",
            "metadata": {
              "page_number": int,
              "section": str,
              "bbox": [x0, y0, x1, y1],
              "image_base64": str (only for images)
            }
          }
        doc_id: The document UUID to associate chunks with.

    Returns:
        The number of chunks successfully stored.

    Raises:
        psycopg.Error: If the database operation fails.
    """
    if not chunks:
        print("[db] No chunks to store")
        return 0

    conn = get_db_connection()
    count = 0
    try:
        with conn.cursor() as cur:
            for chunk in chunks:
                chunk_id = str(uuid.uuid4())
                content = chunk.get("content", "")
                content_type = chunk.get("content_type", "text")
                metadata = chunk.get("metadata", {})

                # Embed the content
                embedding = _embed_text(content)

                # Handle image data if present
                image_bytes = None
                if content_type == "image" and "image_base64" in metadata:
                    import base64

                    image_bytes = base64.b64decode(metadata["image_base64"])

                # Store metadata as JSON
                metadata_json = json.dumps(metadata)

                # Insert chunk into database
                cur.execute(
                    """
                    INSERT INTO multimodal_chunks
                    (chunk_id, doc_id, content, content_type, embedding, metadata, image_bytes)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        chunk_id,
                        doc_id,
                        content,
                        content_type,
                        embedding,
                        metadata_json,
                        image_bytes,
                    ),
                )
                count += 1

            conn.commit()
            print(f"[db] Stored {count} chunks for doc_id={doc_id}")
            return count
    finally:
        conn.close()


def _embed_text(text: str) -> list[float]:
    """Generate embedding vector for a text string.

    Uses a sentence transformer model to create a dense vector representation
    of the text. This embedding is stored alongside the chunk for semantic
    search and retrieval.

    Args:
        text: The text to embed (up to ~8000 tokens).

    Returns:
        A list of floats representing the embedding vector (dimension ~384).

    Note:
        This function lazy-loads the model on first call to avoid startup
        overhead. The model is cached globally for subsequent calls.
    """
    global _embedding_model
    if "_embedding_model" not in globals():
        from sentence_transformers import SentenceTransformer

        print("[db] Loading embedding model (one-time initialization)...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    embedding = _embedding_model.encode(text, convert_to_tensor=False)
    return embedding.tolist()


def get_chunk_by_id(chunk_id: str) -> Optional[dict]:
    """Retrieve a chunk by its ID.

    Args:
        chunk_id: The UUID of the chunk to retrieve.

    Returns:
        A dict with chunk data, or None if not found.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, doc_id, content, content_type, metadata, image_bytes
                FROM multimodal_chunks
                WHERE chunk_id = %s
                """,
                (chunk_id,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "chunk_id": row[0],
                    "doc_id": row[1],
                    "content": row[2],
                    "content_type": row[3],
                    "metadata": json.loads(row[4]),
                    "image_bytes": row[5],
                }
            return None
    finally:
        conn.close()


def search_chunks(
    query: str, doc_id: Optional[str] = None, limit: int = 5
) -> list[dict]:
    """Search chunks using semantic similarity.

    Embeds the query and performs a vector similarity search in pgvector
    to find the most relevant chunks.

    Args:
        query: The search query text.
        doc_id: Optional document ID to filter results to a specific document.
        limit: Maximum number of results to return (default 5).

    Returns:
        List of dicts with chunk data, ordered by similarity (highest first).
    """
    conn = get_db_connection()
    try:
        query_embedding = _embed_text(query)

        # pgvector expects the parameter to be a pgvector `vector`,
        # not a PostgreSQL double precision[] array.
        query_embedding_text = (
            "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
        )

        with conn.cursor() as cur:
            if doc_id:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, content, content_type, metadata,
                           embedding <-> %s::vector AS distance
                    FROM multimodal_chunks
                    WHERE doc_id = %s AND content_type IN ('text', 'table')
                    ORDER BY distance ASC
                    LIMIT %s
                    """,
                    (query_embedding_text, doc_id, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT chunk_id, doc_id, content, content_type, metadata,
                           embedding <-> %s::vector AS distance
                    FROM multimodal_chunks
                    WHERE content_type IN ('text', 'table')
                    ORDER BY distance ASC
                    LIMIT %s
                    """,
                    (query_embedding_text, limit),
                )

            results = []
            for row in cur.fetchall():
                results.append(
                    {
                        "chunk_id": row[0],
                        "doc_id": row[1],
                        "content": row[2],
                        "content_type": row[3],
                        "metadata": (
                            row[4] if isinstance(row[4], dict) else json.loads(row[4])
                        ),
                        "score": 1 - row[5],
                    }
                )
            return results
    finally:
        conn.close()
