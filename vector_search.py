from src.config import VECTOR_STORE


def vector_search(
    query: str,
    top_k: int = 20,
):
    """
    Vector Similarity Search

    Returns:
        List[Document]
    """

    results = VECTOR_STORE.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    docs = []

    for doc, score in results:
        doc.metadata["vector_score"] = float(score)
        docs.append(doc)

    return docs