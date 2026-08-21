import pathlib
from typing import Optional

from dotenv import load_dotenv

from src.core.db import store_chunks, upsert_document
from src.ingestion.docling_parser import parse_document

load_dotenv()


_TEXT_CHUNK_SIZE = 1500
_TEXT_CHUNK_OVERLAP = 300


def _split_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []

    start = 0
    step = chunk_size - overlap

    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += step

    return chunks


def ingest_pdf(
    file_path: str,
    document_name: Optional[str] = None,
    document_id: Optional[str] = None,
) -> dict:
    """
    Ingest a PDF using the Docling multimodal pipeline.

    Text is chunked.
    Tables and images remain atomic.
    """

    resolved = pathlib.Path(file_path).resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"PDF not found: {resolved}")

    # ---------------------------------------------------------
    # 1. Document metadata
    # ---------------------------------------------------------

    final_document_name = (
        pathlib.Path(document_name).name if document_name else resolved.name
    )

    print(f"[ingestion] file={resolved}")

    print(f"[ingestion] document_name={final_document_name}")

    # ---------------------------------------------------------
    # 2. Register document
    # ---------------------------------------------------------

    # Your database currently generates/reuses the document UUID.
    # We retain document_id in the API for compatibility with main.py.

    doc_id = upsert_document(
        final_document_name,
        str(resolved),
    )

    print(f"[ingestion] doc_id={doc_id}")

    # ---------------------------------------------------------
    # 3. Parse using Docling
    # ---------------------------------------------------------

    print(f"[ingestion] Parsing with Docling: {resolved}")

    parsed_elements = parse_document(str(resolved))

    print(f"[ingestion] Docling produced " f"{len(parsed_elements)} elements")

    if not parsed_elements:
        raise ValueError("Docling produced zero elements.")

    # ---------------------------------------------------------
    # 4. Chunk text
    # ---------------------------------------------------------

    chunks: list[dict] = []

    for elem in parsed_elements:

        content_type = elem.get("content_type")

        content = elem.get("content", "")

        # -------------------------------
        # TEXT
        # -------------------------------

        if content_type == "text" and len(content) > _TEXT_CHUNK_SIZE:

            sub_chunks = _split_text(
                content,
                _TEXT_CHUNK_SIZE,
                _TEXT_CHUNK_OVERLAP,
            )

            for sub in sub_chunks:

                chunks.append(
                    {
                        "content": sub,
                        "content_type": "text",
                        "metadata": elem.get(
                            "metadata",
                            {},
                        ),
                    }
                )

        # -------------------------------
        # TEXT that is already short
        # TABLE
        # IMAGE
        # -------------------------------

        else:

            chunks.append(elem)

    print(f"[ingestion] {len(chunks)} chunks ready")

    if not chunks:
        raise ValueError("PDF produced zero chunks.")

    # ---------------------------------------------------------
    # 5. Store multimodal chunks
    # ---------------------------------------------------------

    count = store_chunks(
        chunks,
        doc_id,
    )

    print(f"[ingestion] Stored {count} chunks " f"→ multimodal_chunks")

    return {
        "status": "success",
        "filename": final_document_name,
        "doc_id": str(doc_id),
        "chunks_ingested": count,
    }


# -------------------------------------------------------------
# Command-line execution
# -------------------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) >= 2:

        pdf_path = pathlib.Path(sys.argv[1])

    else:

        pdf_path = pathlib.Path("data/RIL-Media-Release-RIL-Q2-FY2024-25-mini.pdf")

    if not pdf_path.exists():

        raise FileNotFoundError(f"PDF not found at: " f"{pdf_path.resolve()}")

    result = ingest_pdf(str(pdf_path))

    print(f"\nIngestion complete: {result}")
