import json
import re
from typing import Any

import psycopg

from src.config import LLM, PSYCOPG_URL

# ============================================================
# Allowed tables
# ============================================================

ALLOWED_TABLES = {
    "accounts",
    "transactions",
    "loan_accounts",
    "fixed_deposits",
    "credit_cards",
    "card_transactions",
}


# ============================================================
# Block dangerous SQL
# ============================================================

FORBIDDEN = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|"
    r"GRANT|REVOKE|COPY|CALL|DO|EXECUTE"
    r")\b",
    re.I,
)


# ============================================================
# Schema based sample queries
# ============================================================

# ============================================================
# Deterministic customer-data queries
# ============================================================
# The six banking tables in bank_data/public are:
#   accounts
#   transactions
#   loan_accounts
#   fixed_deposits
#   credit_cards
#   card_transactions
#
# Customer access is always validated through public.accounts first.
# Product/policy questions should be routed to RAG by rag_graph.py.
# ============================================================


def _last_n_match(question: str) -> int | None:
    match = re.search(
        r"\b(?:last|latest)\s+(\d+)\s+(?:transactions?|records?)\b",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    n = int(match.group(1))
    if n < 1 or n > 100:
        raise ValueError("The requested number of records must be between 1 and 100.")
    return n


def _account_details_sql(account_id: str) -> tuple[str, list[Any]]:
    return (
        """
        SELECT
            account_id,
            customer_name,
            account_type,
            branch_code,
            mobile,
            email,
            kyc_status,
            ifsc_code,
            created_at
        FROM public.accounts
        WHERE account_id = CAST(%s AS VARCHAR(20))
        """,
        [str(account_id)],
    )


def _transactions_sql(
    account_id: str, limit: int | None = None
) -> tuple[str, list[Any]]:
    sql = """
        SELECT
            txn_id,
            account_id,
            txn_date,
            txn_type,
            amount,
            balance_after,
            description,
            merchant_name,
            category,
            channel
        FROM public.transactions
        WHERE account_id = CAST(%s AS VARCHAR(20))
        ORDER BY txn_date DESC, txn_id DESC
    """
    params: list[Any] = [str(account_id)]
    if limit is not None:
        sql += "\nLIMIT %s"
        params.append(limit)
    return sql, params


def _loan_sql(account_id: str) -> tuple[str, list[Any]]:
    return (
        """
        SELECT
            loan_id,
            account_id,
            loan_type,
            outstanding,
            emi_amount,
            next_emi_date,
            interest_rate,
            status
        FROM public.loan_accounts
        WHERE account_id = CAST(%s AS VARCHAR(20))
        ORDER BY next_emi_date ASC NULLS LAST
        """,
        [str(account_id)],
    )


def _fd_sql(account_id: str) -> tuple[str, list[Any]]:
    return (
        """
        SELECT
            fd_id,
            account_id,
            principal,
            interest_rate,
            maturity_date,
            maturity_amount,
            interest_payout,
            status
        FROM public.fixed_deposits
        WHERE account_id = CAST(%s AS VARCHAR(20))
        ORDER BY maturity_date ASC NULLS LAST
        """,
        [str(account_id)],
    )


def _credit_card_sql(account_id: str) -> tuple[str, list[Any]]:
    return (
        """
        SELECT
            card_id,
            account_id,
            card_type,
            credit_limit,
            available_limit,
            status
        FROM public.credit_cards
        WHERE account_id = CAST(%s AS VARCHAR(20))
        """,
        [str(account_id)],
    )


def _card_transactions_sql(
    account_id: str, limit: int | None = None
) -> tuple[str, list[Any]]:
    sql = """
        SELECT
            ct.txn_id,
            c.account_id,
            ct.card_id,
            ct.txn_date,
            ct.amount,
            ct.merchant_name,
            ct.currency,
            ct.category,
            ct.is_international
        FROM public.card_transactions ct
        JOIN public.credit_cards c
          ON c.card_id = ct.card_id
        WHERE c.account_id = CAST(%s AS VARCHAR(20))
        ORDER BY ct.txn_date DESC, ct.txn_id DESC
    """
    params: list[Any] = [str(account_id)]
    if limit is not None:
        sql += "\nLIMIT %s"
        params.append(limit)
    return sql, params


def generate_customer_sql(
    question: str, account_id: str
) -> tuple[str, list[Any]] | None:
    """
    Deterministically handle common customer-data requests across ALL six
    banking tables. Returning None lets the LLM handle less common requests,
    but the final account guard still applies.
    """
    q = question.lower()
    n = _last_n_match(question)

    if "card transaction" in q or "card transactions" in q:
        return _card_transactions_sql(account_id, n)

    if "credit card" in q or "credit cards" in q:
        return _credit_card_sql(account_id)

    if "fixed deposit" in q or re.search(r"\bfd\b", q):
        return _fd_sql(account_id)

    if "loan" in q or "emi" in q or "outstanding" in q:
        return _loan_sql(account_id)

    if "transaction" in q or "statement" in q or "transaction history" in q:
        # If the user says "latest transaction" without a number, return one row.
        return _transactions_sql(account_id, n if n is not None else 1)

    if (
        "account details" in q
        or "account information" in q
        or "account profile" in q
        or "kyc status" in q
        or q.strip() in {"account", "my account"}
    ):
        return _account_details_sql(account_id)

    return None


# ============================================================
# Extract LLM response
# ============================================================


# ============================================================


def _content(response: Any) -> str:

    value = getattr(
        response,
        "content",
        response,
    )

    return value if isinstance(value, str) else str(value)


def qualify_table_references(sql: str) -> str:
    """
    Deterministically qualify every approved banking table with public.
    This prevents an LLM-generated `FROM accounts` from reaching PostgreSQL.
    """
    for table in sorted(ALLOWED_TABLES, key=len, reverse=True):
        sql = re.sub(
            rf"(\b(?:FROM|JOIN)\s+)(?!public\.){re.escape(table)}\b",
            rf"\1public.{table}",
            sql,
            flags=re.IGNORECASE,
        )
    return sql


# ============================================================
# SQL validation
# ============================================================


def _safe_sql(sql: str):

    # Final deterministic normalization: always use public.<table>.
    cleaned = qualify_table_references(sql.strip())

    if not cleaned:
        raise ValueError("Empty SQL generated")

    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not allowed")

    if not re.match(
        r"^\s*SELECT\b",
        cleaned,
        re.I,
    ):
        raise ValueError("Only SELECT queries allowed")

    if FORBIDDEN.search(cleaned):
        raise ValueError("Unsafe SQL keyword detected")

    # Accept both unqualified and public-qualified table references.
    # Example:
    #   FROM accounts            -> accounts
    #   FROM public.accounts     -> accounts
    # The previous regex captured "public" as the table name, which caused
    # "Unauthorized tables: ['public']" even for valid public.accounts.
    table_refs = re.findall(
        r"\b(?:FROM|JOIN)\s+(?:(?:public)\.)?([a-zA-Z_][a-zA-Z0-9_]*)",
        cleaned,
        re.I,
    )

    tables = {t.lower() for t in table_refs}
    unknown = tables - ALLOWED_TABLES

    if unknown:
        raise ValueError(f"Unauthorized tables: {unknown}")

    # LIMIT may be a literal (LIMIT 6) or a psycopg placeholder
    # (LIMIT %s). Do not append a second LIMIT when a placeholder is used.
    limit_match = re.search(
        r"\bLIMIT\s+(%s|\d+)",
        cleaned,
        re.I,
    )

    if not limit_match:
        cleaned += " LIMIT 100"

    elif limit_match.group(1) != "%s" and int(limit_match.group(1)) > 100:
        raise ValueError("Maximum 100 rows allowed")

    return cleaned


# ============================================================
# Customer/account access controls
# ============================================================

CUSTOMER_DATA_TERMS = {
    "my account",
    "account balance",
    "account details",
    "account information",
    "account profile",
    "transactions",
    "transaction",
    "statement",
    "loan",
    "emi",
    "outstanding",
    "fixed deposit",
    "fixed deposits",
    "fd",
    "credit card",
    "credit cards",
    "card transaction",
    "card transactions",
    "kyc status",
}

GLOBAL_ACCOUNT_PHRASES = {
    "any account",
    "any accounts",
    "all accounts",
    "any customer",
    "all customers",
}


def extract_account_id(question: str) -> str | None:
    """Extract a numeric account id explicitly supplied by the user."""
    patterns = [
        r"\baccount\s*(?:number|no\.?|id)?\s*[:#=]?\s*(\d{4,20})\b",
        r"\baccount\s*#\s*(\d{4,20})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if match:
            return match.group(1)

    return None


def is_customer_data_question(question: str) -> bool:
    text = question.lower()
    return any(term in text for term in CUSTOMER_DATA_TERMS)


def account_exists(account_id: str) -> bool:
    """Return True only when the supplied account exists in the bank DB."""
    with psycopg.connect(PSYCOPG_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM public.accounts WHERE account_id = CAST(%s AS VARCHAR(20)) LIMIT 1",
                (account_id,),
            )
            return cur.fetchone() is not None


def validate_customer_access(question: str) -> tuple[bool, str, str | None]:
    """
    Enforce account-based access before any customer-data SQL is executed.

    Product/policy questions are handled by RAG and do not call this function.
    Customer-data questions require an explicit, existing account number.
    Requests such as 'any account' are rejected rather than querying across
    customers.
    """
    text = question.lower().strip()

    if any(phrase in text for phrase in GLOBAL_ACCOUNT_PHRASES):
        return (
            False,
            "Please provide a valid account number. I can only retrieve customer-specific banking information for a specified account.",
            None,
        )

    if not is_customer_data_question(question):
        return True, "", None

    account_id = extract_account_id(question)

    if not account_id:
        return (
            False,
            "Please provide a valid account number so I can retrieve your banking information.",
            None,
        )

    if not account_exists(account_id):
        return (
            False,
            f"I couldn't find a valid account with number {account_id}. Please verify the account number and try again.",
            account_id,
        )

    return True, "", account_id


def enforce_account_filter(sql: str, account_id: str | None) -> None:
    """Ensure customer-data SQL is scoped to the validated account."""
    if not account_id:
        return

    normalized = re.sub(r"\s+", " ", sql).lower()

    if "account_id" not in normalized:
        raise ValueError("Customer query must be scoped to the validated account.")

    # The generated query must use a parameter for the account id; do not allow
    # a different hard-coded account number to be substituted.
    if "%s" not in sql:
        raise ValueError("Customer query must use a parameterized account filter.")


# ============================================================
# Generate SQL
# ============================================================


def generate_sql(question: str, account_id: str) -> tuple[str, list[Any]]:
    # Deterministic handling for all six customer-facing banking tables.
    deterministic = generate_customer_sql(question, account_id)
    if deterministic:
        return deterministic

    # For less common customer questions, let the LLM generate SQL, but provide
    # the complete schema and force the validated account to be a parameter.
    prompt = f"""
You are a PostgreSQL SQL generator for a banking assistant.

Return ONLY JSON:
{{
  "sql": "SELECT query",
  "params": []
}}

The database is bank_data, schema public.

Approved tables and columns:

public.accounts:
- account_id VARCHAR(20)
- customer_name
- account_type
- branch_code
- mobile
- email
- kyc_status
- ifsc_code
- created_at

public.transactions:
- txn_id
- account_id VARCHAR(20)
- txn_date
- txn_type
- amount
- balance_after
- description
- merchant_name
- category
- channel

public.loan_accounts:
- loan_id
- account_id VARCHAR(20)
- loan_type
- outstanding
- emi_amount
- next_emi_date
- interest_rate
- status

public.fixed_deposits:
- fd_id
- account_id VARCHAR(20)
- principal
- interest_rate
- maturity_date
- maturity_amount
- interest_payout
- status

public.credit_cards:
- card_id
- account_id VARCHAR(20)
- card_type
- credit_limit
- available_limit
- status

public.card_transactions:
- txn_id
- card_id
- txn_date
- amount
- merchant_name
- currency
- category
- is_international

Rules:
- Only SELECT.
- Always use public.<table>.
- Use only the six approved tables.
- Use %s placeholders.
- The validated customer account is: {account_id}
- Every customer-data query MUST be scoped to that account.
- For card_transactions, join public.credit_cards to public.card_transactions
  and filter public.credit_cards.account_id using the validated account.
- Never query all accounts.
- Never invent an account id.
- Never expose PAN or full mobile numbers.
- Maximum 100 rows.
- For "last N transactions", order by txn_date DESC, txn_id DESC and LIMIT N.

Question:
{question}
"""

    response = LLM.invoke(prompt)
    raw = _content(response).strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.I | re.S).strip()

    data = json.loads(raw)
    sql = data["sql"]
    params = data.get("params", [])

    if not isinstance(params, list):
        raise ValueError("SQL params must be a list.")

    # For a validated customer query, the account must be passed as the first
    # customer-scope parameter unless the query uses a different explicit
    # parameter layout. The final guard below verifies the account value.
    if "account_id" in sql.lower() and str(account_id) not in [str(p) for p in params]:
        # Do not silently invent/replace arbitrary parameters. Reject it.
        raise ValueError("Generated SQL did not use the validated account number.")

    return sql, params


def _deterministic_transaction_request(question: str, account_id: str):
    """
    Handle transaction requests deterministically instead of asking the LLM
    to decide whether 'show transactions' means one row or the full history.
    """
    q = question.lower().strip()

    if "transaction" not in q and "statement" not in q:
        return None

    # "latest transaction" / "last transaction" means exactly one row.
    if re.search(r"\b(?:latest|last)\s+transaction\b", q):
        return (
            """
            SELECT
                txn_id,
                account_id,
                txn_date,
                txn_type,
                amount,
                balance_after,
                description,
                merchant_name,
                category,
                channel
            FROM public.transactions
            WHERE account_id = CAST(%s AS VARCHAR(20))
            ORDER BY txn_date DESC, txn_id DESC
            LIMIT 1
            """,
            [str(account_id)],
            "latest",
        )

    # "last N transactions" means N rows.
    m = re.search(r"\b(?:last|latest)\s+(\d+)\s+transactions?\b", q)
    if m:
        n = min(max(int(m.group(1)), 1), 100)
        return (
            """
            SELECT
                txn_id,
                account_id,
                txn_date,
                txn_type,
                amount,
                balance_after,
                description,
                merchant_name,
                category,
                channel
            FROM public.transactions
            WHERE account_id = CAST(%s AS VARCHAR(20))
            ORDER BY txn_date DESC, txn_id DESC
            LIMIT %s
            """,
            [str(account_id), n],
            "list",
        )

    # "show transactions", "all transactions", "transaction history", etc.
    # means the available transaction history, capped at 100 rows.
    return (
        """
        SELECT
            txn_id,
            account_id,
            txn_date,
            txn_type,
            amount,
            balance_after,
            description,
            merchant_name,
            category,
            channel
        FROM public.transactions
        WHERE account_id = CAST(%s AS VARCHAR(20))
        ORDER BY txn_date DESC, txn_id DESC
        LIMIT 100
        """,
        [str(account_id)],
        "list",
    )


def execute_sql(question: str):

    # Customer-specific database access is allowed only for a valid account.
    allowed, message, account_id = validate_customer_access(question)

    if not allowed:
        return {
            "error": message,
            "sql": "",
            "params": [],
            "rows": [],
        }

    deterministic = _deterministic_transaction_request(question, account_id)
    if deterministic:
        sql, params, request_type = deterministic
    else:
        sql, params = generate_sql(question, account_id)
        request_type = "generic"

    # Final execution-time guardrail.
    sql = _safe_sql(sql)

    # Enforce the row cap for parameterized LIMIT values as well.
    # This prevents LIMIT %s from bypassing the 100-row safety cap.
    parameterized_limit = re.search(r"\bLIMIT\s+%s\b", sql, re.I)
    if parameterized_limit:
        placeholder_index = sql[: parameterized_limit.start()].count("%s")
        if placeholder_index >= len(params):
            raise ValueError("LIMIT placeholder has no matching parameter.")
        limit_value = params[placeholder_index]
        try:
            limit_value = int(limit_value)
        except (TypeError, ValueError):
            raise ValueError("LIMIT parameter must be an integer.")
        if limit_value < 1 or limit_value > 100:
            raise ValueError("Maximum 100 rows allowed.")

    # If an account was supplied, make sure the generated query is scoped to
    # that account and normalize the account parameter to STRING because
    # public.accounts.account_id and public.transactions.account_id are VARCHAR.
    if account_id:
        # account_id is VARCHAR(20) in the banking schema. Always send the
        # validated account identifier as a string, even if an LLM returned
        # it as an integer.
        normalized_params = []
        found_account_param = False
        for value in params:
            if str(value) == str(account_id):
                normalized_params.append(str(account_id))
                found_account_param = True
            else:
                normalized_params.append(value)
        params = normalized_params

        if "account_id" not in sql.lower():
            raise ValueError("Generated SQL is not scoped to the validated account.")
        if "%s" not in sql:
            raise ValueError("Customer query must use a parameterized account filter.")
        if account_id not in [str(value) for value in params]:
            raise ValueError("Generated SQL did not use the validated account number.")

    with psycopg.connect(PSYCOPG_URL) as conn:

        with conn.cursor() as cur:

            cur.execute(
                sql,
                params,
            )

            columns = [d.name for d in cur.description]

            rows = cur.fetchmany(100)

    result = []

    for row in rows:

        item = {}

        for key, value in zip(
            columns,
            row,
        ):

            lower = key.lower()

            if "mobile" in lower:

                if value:
                    item[key] = str(value)[:3] + "****" + str(value)[-3:]

            elif "pan" in lower:

                item[key] = "****"

            elif hasattr(
                value,
                "isoformat",
            ):

                item[key] = value.isoformat()

            else:

                item[key] = value

        result.append(item)

    return {
        "sql": sql,
        "params": params,
        "rows": result,
        "request_type": request_type,
    }
