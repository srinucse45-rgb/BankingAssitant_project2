import psycopg

from langchain_core.documents import Document
from src.config import COLLECTION_NAME, PSYCOPG_URL


def fts_search(
    query: str,
    collection_name: str = COLLECTION_NAME,
    top_k: int = 20,
):
    sql = """
    SELECT
        e.document,
        e.cmetadata,

        ts_rank(
            to_tsvector('english', e.document),
            plainto_tsquery('english', %(query)s)
        ) AS rank

    FROM langchain_pg_embedding e

    JOIN langchain_pg_collection c
        ON c.uuid = e.collection_id

    WHERE
        c.name = %(collection)s
        AND
        to_tsvector('english', e.document)
        @@ plainto_tsquery('english', %(query)s)

    ORDER BY rank DESC

    LIMIT %(k)s;
    """

    docs = []

    print("Connecting to:", PSYCOPG_URL)

    with psycopg.connect(PSYCOPG_URL) as conn:
        with conn.cursor() as cur:

            cur.execute(
                sql,
                {
                    "query": query,
                    "collection": collection_name,
                    "k": top_k,
                },
            )

            rows = cur.fetchall()

            for row in rows:
                docs.append(
    Document(
        page_content=row[0],
        metadata={
            **row[1],
            "fts_score": row[2],
        },
    )
)

    return docs