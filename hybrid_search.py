from typing import Dict, List, Tuple

from langchain_core.documents import Document

from src.retrieval.vector_search import vector_search
from src.retrieval.fts_search import fts_search


# ============================================================
# Normalize text
# ============================================================

def normalize_text(text: str) -> str:
    """
    Normalize document text.

    Used for:
    - duplicate detection
    - RRF document matching
    """

    if not text:
        return ""

    return " ".join(text.lower().split())


# ============================================================
# Convert retrieval result to Document
# ============================================================

def extract_document(item):
    """
    Safely extract Document from different retrieval formats.

    Supported formats:

        Document

        (Document, score)

        (Document, score, extra)

    This prevents errors such as:

        ValueError: too many values to unpack
    """

    if isinstance(item, Document):
        return item

    if isinstance(item, tuple):
        if len(item) == 0:
            return None

        if isinstance(item[0], Document):
            doc = item[0]

            # Store retrieval score if available
            if len(item) >= 2:
                score = item[1]

                if isinstance(score, (int, float)):
                    doc.metadata["retrieval_score"] = float(score)

            return doc

    return None


# ============================================================
# Prepare retrieval results
# ============================================================

def prepare_documents(results, score_key: str) -> List[Document]:
    """
    Convert vector/FTS results into a clean List[Document].

    score_key:
        vector_score
        fts_score
    """

    documents = []

    for item in results:

        doc = extract_document(item)

        if doc is None:
            continue

        # ----------------------------------------------------
        # Preserve score using the correct metadata field
        # ----------------------------------------------------

        if "retrieval_score" in doc.metadata:

            doc.metadata[score_key] = doc.metadata[
                "retrieval_score"
            ]

            del doc.metadata["retrieval_score"]

        documents.append(doc)

    return documents


# ============================================================
# Deduplicate documents
# ============================================================

def deduplicate_documents(
    documents: List[Tuple[Document, float]]
) -> List[Tuple[Document, float]]:
    """
    Remove duplicate documents.

    Duplicate key:

        page + normalized content

    The document with the highest score is retained.
    """

    unique: Dict[str, Tuple[Document, float]] = {}

    for doc, score in documents:

        page = doc.metadata.get("page", "")

        content = normalize_text(
            doc.page_content
        )

        key = f"{page}|{content}"

        if key not in unique:

            unique[key] = (
                doc,
                score,
            )

        elif score > unique[key][1]:

            unique[key] = (
                doc,
                score,
            )

    return list(unique.values())


# ============================================================
# Reciprocal Rank Fusion
# ============================================================

def reciprocal_rank_fusion(
    vector_docs: List[Document],
    fts_docs: List[Document],
    k: int = 60,
) -> List[Document]:
    """
    Combine vector and FTS results using
    Reciprocal Rank Fusion.

    RRF formula:

        score = 1 / (k + rank)

    If a document appears in both retrieval systems,
    its scores are added together.
    """

    scores: Dict[str, float] = {}

    documents: Dict[str, Document] = {}

    # --------------------------------------------------------
    # VECTOR RESULTS
    # --------------------------------------------------------

    for rank, doc in enumerate(
        vector_docs,
        start=1,
    ):

        content_key = normalize_text(
            doc.page_content
        )

        scores.setdefault(
            content_key,
            0.0,
        )

        documents[content_key] = doc

        scores[content_key] += (
            1 / (k + rank)
        )

        doc.metadata["retriever"] = "vector"

    # --------------------------------------------------------
    # FTS RESULTS
    # --------------------------------------------------------

    for rank, doc in enumerate(
        fts_docs,
        start=1,
    ):

        content_key = normalize_text(
            doc.page_content
        )

        scores.setdefault(
            content_key,
            0.0,
        )

        # ----------------------------------------------------
        # If vector already found same document,
        # retain vector metadata and mark both.
        # ----------------------------------------------------

        if content_key in documents:

            existing_doc = documents[
                content_key
            ]

            existing_doc.metadata[
                "retriever"
            ] = "vector+fts"

            documents[
                content_key
            ] = existing_doc

        else:

            documents[
                content_key
            ] = doc

            doc.metadata[
                "retriever"
            ] = "fts"

        scores[content_key] += (
            1 / (k + rank)
        )

    # --------------------------------------------------------
    # SORT BY RRF SCORE
    # --------------------------------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []

    for content_key, score in ranked:

        doc = documents[
            content_key
        ]

        doc.metadata[
            "rrf_score"
        ] = round(score, 6)

        results.append(doc)

    return results


# ============================================================
# Table/question detection
# ============================================================

def is_table_query(query: str) -> bool:
    """
    Detect questions that are likely asking
    for structured/table information.
    """

    query_lower = query.lower()

    table_keywords = [
        "interest rate",
        "interest rates",
        "rate",
        "rates",
        "ltv",
        "maximum ltv",
        "loan amount",
        "processing fee",
        "fees",
        "charges",
        "eligibility",
        "minimum income",
        "credit score",
        "cibil",
        "percentage",
        "%",
    ]

    return any(
        keyword in query_lower
        for keyword in table_keywords
    )


# ============================================================
# Diversity filtering
# ============================================================

def diversify_results(
    documents: List[Document],
    max_per_page: int = 3,
) -> List[Document]:
    """
    Reduce repeated results from the same page.

    Tables receive priority because banking
    information such as rates/LTV is often
    stored in tables.
    """

    results = []

    page_counts: Dict[str, int] = {}

    seen_content = set()

    # --------------------------------------------------------
    # Determine whether table documents exist
    # --------------------------------------------------------

    has_tables = any(
        doc.metadata.get("chunk_type") == "table"
        for doc in documents
    )

    # --------------------------------------------------------
    # Sort:
    #
    # 1. Tables first if available
    # 2. Higher RRF score
    # --------------------------------------------------------

    ordered = sorted(
        documents,
        key=lambda doc: (
            0
            if (
                has_tables
                and doc.metadata.get(
                    "chunk_type"
                )
                == "table"
            )
            else 1,
            -float(
                doc.metadata.get(
                    "rrf_score",
                    0,
                )
            ),
        ),
    )

    # --------------------------------------------------------
    # Apply diversity
    # --------------------------------------------------------

    for doc in ordered:

        content_key = normalize_text(
            doc.page_content
        )

        # Exact content duplicate
        if content_key in seen_content:
            continue

        page = str(
            doc.metadata.get(
                "page",
                "unknown",
            )
        )

        count = page_counts.get(
            page,
            0,
        )

        if count >= max_per_page:
            continue

        seen_content.add(
            content_key
        )

        page_counts[page] = (
            count + 1
        )

        results.append(doc)

    return results


# ============================================================
# Hybrid Search
# ============================================================

def hybrid_search(
    query: str,
    top_k: int = 5,
) -> List[Document]:
    """
    Hybrid retrieval pipeline:

        Query
          |
          +---- Vector Search
          |
          +---- FTS Search
          |
          v
        Normalize
          |
          v
        RRF
          |
          v
        Deduplication
          |
          v
        Table Priority
          |
          v
        Page Diversity
          |
          v
        Top-K Documents

    Returns:

        List[Document]
    """

    print("=" * 80)

    print(
        f"HYBRID SEARCH: {query}"
    )

    print("=" * 80)

    # ========================================================
    # Retrieve more candidates
    # ========================================================

    retrieval_k = max(
        top_k * 5,
        10,
    )

    # ========================================================
    # VECTOR SEARCH
    # ========================================================

    try:

        vector_results = vector_search(
            query=query,
            top_k=retrieval_k,
        )

    except Exception as e:

        print(
            f"Vector search failed: {e}"
        )

        vector_results = []

    # ========================================================
    # FTS SEARCH
    # ========================================================

    try:

        fts_results = fts_search(
            query=query,
            top_k=retrieval_k,
        )

    except Exception as e:

        print(
            f"FTS search failed: {e}"
        )

        fts_results = []

    # ========================================================
    # Prepare Documents
    # ========================================================

    vector_docs = prepare_documents(
        vector_results,
        "vector_score",
    )

    fts_docs = prepare_documents(
        fts_results,
        "fts_score",
    )

    print(
        f"Vector documents: {len(vector_docs)}"
    )

    print(
        f"FTS documents: {len(fts_docs)}"
    )

    # ========================================================
    # If both searches failed
    # ========================================================

    if not vector_docs and not fts_docs:

        print(
            "No documents retrieved."
        )

        return []

    # ========================================================
    # RRF
    # ========================================================

    fused_documents = (
        reciprocal_rank_fusion(
            vector_docs=vector_docs,
            fts_docs=fts_docs,
        )
    )

    print(
        f"After RRF: {len(fused_documents)}"
    )

    # ========================================================
    # Table-aware ordering
    # ========================================================

    if is_table_query(query):

        print(
            "Table-oriented query detected."
        )

        # Put table chunks before ordinary text
        fused_documents = sorted(
            fused_documents,
            key=lambda doc: (
                0
                if doc.metadata.get(
                    "chunk_type"
                )
                == "table"
                else 1,
                -float(
                    doc.metadata.get(
                        "rrf_score",
                        0,
                    )
                ),
            ),
        )

    # ========================================================
    # Diversity filtering
    # ========================================================

    diversified_documents = (
        diversify_results(
            fused_documents,
            max_per_page=3,
        )
    )

    print(
        f"After diversity filtering: "
        f"{len(diversified_documents)}"
    )

    # ========================================================
    # Final Top-K
    # ========================================================

    final_documents = (
        diversified_documents[:top_k]
    )

    print(
        f"Final documents: "
        f"{len(final_documents)}"
    )

    # ========================================================
    # Debug output
    # ========================================================

    for index, doc in enumerate(
        final_documents,
        start=1,
    ):

        print("-" * 80)

        print(
            f"Document {index}"
        )

        print(
            f"Page: "
            f"{doc.metadata.get('page')}"
        )

        print(
            f"Chunk Type: "
            f"{doc.metadata.get('chunk_type')}"
        )

        print(
            f"Retriever: "
            f"{doc.metadata.get('retriever')}"
        )

        print(
            f"RRF Score: "
            f"{doc.metadata.get('rrf_score')}"
        )

        print(
            f"Vector Score: "
            f"{doc.metadata.get('vector_score')}"
        )

        print(
            f"FTS Score: "
            f"{doc.metadata.get('fts_score')}"
        )

        print(
            f"Image Path: "
            f"{doc.metadata.get('image_path')}"
        )

        print()

        print(
            doc.page_content[:1500]
        )

    print("=" * 80)

    return final_documents


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    questions = [

        "What are the home loan interest rates?",

        "What is the maximum LTV for a home loan?",
    ]

    for question in questions:

        print("\n\n")

        print("=" * 80)

        print(
            f"QUESTION: {question}"
        )

        print("=" * 80)

        docs = hybrid_search(
            query=question,
            top_k=5,
        )

        print(
            f"\nRetrieved documents: "
            f"{len(docs)}"
        )

        for index, doc in enumerate(
            docs,
            start=1,
        ):

            print("\n")

            print(
                f"Document {index}"
            )

            print(
                f"Page: "
                f"{doc.metadata.get('page')}"
            )

            print(
                f"Chunk Type: "
                f"{doc.metadata.get('chunk_type')}"
            )

            print(
                f"Retriever: "
                f"{doc.metadata.get('retriever')}"
            )

            print(
                f"RRF Score: "
                f"{doc.metadata.get('rrf_score')}"
            )

            print(
                f"Image Path: "
                f"{doc.metadata.get('image_path')}"
            )

            print(
                "\nContent:"
            )

            print(
                doc.page_content
            )