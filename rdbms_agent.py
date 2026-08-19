import os
import re
from typing import Any, Dict, List, Tuple

import psycopg
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from src.config import LLM

load_dotenv()


ALLOWED_TABLES = {
    "accounts",
    "transactions",
    "loan_accounts",
    "fixed_deposits",
    "credit_cards",
    "card_transactions",
}

BLOCKED_SQL_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "CREATE", "GRANT", "REVOKE", "COMMENT", "VACUUM", "ANALYZE",
    "CALL", "DO", "EXECUTE", "MERGE", "COPY", "SET", "RESET",
}


def get_rdbms_connection_string() -> str:
    connection_string = os.getenv("PG_RDBMS_CONNECTION_STRING")
    if not connection_string:
        raise ValueError(
            "PG_RDBMS_CONNECTION_STRING is not configured in .env"
        )
    return connection_string


def get_bank_schema() -> str:
    """Read the live schema for only the banking tables exposed to the agent."""
    query = """
        SELECT
            c.table_name,
            c.column_name,
            c.data_type,
            c.is_nullable
        FROM information_schema.columns c
        WHERE c.table_schema = 'public'
          AND c.table_name = ANY(%s)
        ORDER BY c.table_name, c.ordinal_position
    """

    rows: List[Tuple[Any, ...]]
    with psycopg.connect(get_rdbms_connection_string()) as conn:
        with conn.cursor() as cur:
            cur.execute(query, [list(ALLOWED_TABLES)])
            rows = cur.fetchall()

    if not rows:
        raise ValueError(
            "No allowed banking tables were found in the RDBMS database."
        )

    grouped: Dict[str, List[str]] = {}
    for table, column, data_type, nullable in rows:
        grouped.setdefault(table, []).append(
            f"{column} {data_type}{' NULL' if nullable == 'YES' else ' NOT NULL'}"
        )

    return "\n".join(
        f"TABLE {table}:\n  " + "\n  ".join(columns)
        for table, columns in grouped.items()
    )


def clean_generated_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()


def validate_sql(sql: str) -> None:
    """Application-level guardrail. PostgreSQL read-only permissions remain the DB boundary."""
    sql = clean_generated_sql(sql)
    if not sql:
        raise ValueError("The SQL generator returned an empty query.")

    # Only one statement is permitted.
    statements = [part.strip() for part in sql.split(";") if part.strip()]
    if len(statements) != 1:
        raise ValueError("Only one SQL statement is permitted.")

    normalized = re.sub(r"\s+", " ", sql).strip()
    upper = normalized.upper()

    if not (upper.startswith("SELECT ") or upper.startswith("SELECT\n") or upper == "SELECT"):
        raise ValueError("Only SELECT read-only SQL is permitted.")

    for keyword in BLOCKED_SQL_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", upper):
            raise ValueError(f"Blocked SQL operation: {keyword}")

    # Require every referenced base table to be one of the banking tables.
    # This is intentionally conservative; unknown table references are rejected.
    references = re.findall(
        r"\b(?:FROM|JOIN)\s+(?:public\.)?([a-zA-Z_][a-zA-Z0-9_]*)",
        normalized,
        flags=re.IGNORECASE,
    )
    unknown = {name.lower() for name in references} - ALLOWED_TABLES
    if unknown:
        raise ValueError(
            "SQL references tables outside the banking allow-list: "
            + ", ".join(sorted(unknown))
        )

    # Prevent comments from being used to hide extra SQL.
    if "--" in sql or "/*" in sql or "*/" in sql:
        raise ValueError("SQL comments are not permitted.")


def requires_identifier(question: str) -> bool:
    """
    Prevent customer data extraction without a valid business identifier.

    Customer-specific database queries require an identifier such as:
    - account number
    - customer id
    - loan number
    - card number
    - transaction id
    """

    text = question.lower()

    customer_data_terms = [
        "account",
        "balance",
        "transaction",
        "statement",
        "customer",
        "card",
        "loan",
        "deposit",
        "profile",
        "detail",
        "details",
        "information",
        "data",
        "record",
        "records",
    ]

    identifier_terms = [
        "account number",
        "account id",
        "customer id",
        "loan number",
        "transaction id",
        "card number",
    ]

    asks_customer_data = any(
        term in text
        for term in customer_data_terms
    )

    has_identifier = any(
        term in text
        for term in identifier_terms
    )

    has_number = any(
        ch.isdigit()
        for ch in question
    )

    return (
        asks_customer_data
        and not (has_identifier or has_number)
    )


def generate_sql(question: str, schema: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the SQL generation component of a banking assistant.
Generate exactly one safe PostgreSQL SELECT query that answers the user's question.

The database contains ONLY these agent-approved tables:
- accounts
- transactions
- loan_accounts
- fixed_deposits
- credit_cards
- card_transactions

Rules:
1. Return ONLY SQL. No explanation, markdown, or code fences.
2. Only SELECT is allowed.
3. Never generate INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE,
   GRANT, REVOKE, or any other write/admin statement.
4. Use only the columns and tables in the supplied schema.
5. For non-aggregate detail queries, add LIMIT 50.
6. Prefer explicit column names rather than SELECT *.
7. Use the relationships shown by the column names. For example,
   transactions.account_id joins accounts.account_id.
8. For balances, use the latest relevant balance_after when the question asks
   for an account's current/latest balance, unless another balance field is available.
9. For transaction history, order newest first unless the user asks otherwise.
10. Do not invent account IDs, transaction IDs, loan IDs, or card IDs.
11. If the question cannot be answered from the supplied schema, return:
    SELECT 'INSUFFICIENT_DATA' AS reason;

LIVE DATABASE SCHEMA:
{schema}
""",
            ),
            ("human", "Question:\n{question}"),
        ]
    )

    response = (prompt | LLM).invoke({
        "schema": schema,
        "question": question,
    })
    return clean_generated_sql(response.content)


def execute_sql(sql: str) -> Tuple[List[str], List[Tuple[Any, ...]]]:
    validate_sql(sql)

    with psycopg.connect(get_rdbms_connection_string(), options="-c default_transaction_read_only=on") as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [desc.name for desc in cur.description] if cur.description else []
            rows = cur.fetchall() if cur.description else []

    return columns, rows


def format_result(columns: List[str], rows: List[Tuple[Any, ...]]) -> str:
    if not rows:
        return "No rows were returned."

    lines = [" | ".join(columns)]
    lines.append(" | ".join("---" for _ in columns))
    for row in rows[:50]:
        lines.append(" | ".join(str(value) if value is not None else "NULL" for value in row))
    return "\n".join(lines)


def generate_answer(question: str, sql: str, result: str, history: List[Dict[str, str]]) -> str:
    history_text = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in history[-6:]
    ) or "No previous conversation."

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are the data-answering component of a banking assistant.
Answer the user's question using ONLY the SQL result supplied below.

Rules:
- Do not invent or infer data that is not present in the result.
- Be concise and clear.
- Format money and lists clearly when appropriate.
- Never expose database credentials.
- Do not discuss SQL unless the user asks for it.
- If no rows were returned, clearly say that no matching information was found.
""",
            ),
            (
                "human",
                "Question: {question}\n\nRecent conversation:\n{history}\n\nSQL:\n{sql}\n\nSQL result:\n{result}",
            ),
        ]
    )

    response = (prompt | LLM).invoke({
        "question": question,
        "history": history_text,
        "sql": sql,
        "result": result,
    })
    return response.content.strip()


def run_rdbms_agent(
    question: str,
    history: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Run the banking NL-to-SQL agent against the read-only bank_data database."""
    history = history or []

    if requires_identifier(question):
        return {
            "answer": "Please provide the account number, loan number, card number, or transaction id so I can retrieve the details.",
            "sources": [],
            "sql_query_executed": "",
            "sql_result": "",
        }

    schema = get_bank_schema()
    sql = generate_sql(question, schema)
    validate_sql(sql)
    columns, rows = execute_sql(sql)
    result_text = format_result(columns, rows)
    answer = generate_answer(question, sql, result_text, history)

    return {
        "answer": answer,
        "sources": [],
        "sql_query_executed": sql,
        "sql_result": result_text,
    }
