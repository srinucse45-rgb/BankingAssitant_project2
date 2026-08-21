"""
Merged Smart Banking RAG LangGraph implementation.

Merged from:
- rag_graph(5).py
- rag_graph_pre.py

Features:
- Input guardrails
- Query classification
- RAG / SQL / Hybrid routing
- Active document retrieval
- Reranking
- SQL execution
- Banking answer validation
- PostgreSQL checkpoint fallback
"""

from typing import Any, Dict, List, Literal, TypedDict
import re

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from src.config import LLM

try:
    from src.guardrails import validate_input, validate_banking_answer
except Exception:
    validate_input = None
    validate_banking_answer = None

try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.checkpoint.memory import MemorySaver
    from src.config import CHECKPOINT_DATABASE_URL
except Exception:
    PostgresSaver = None
    MemorySaver = None
    CHECKPOINT_DATABASE_URL = None


FALLBACK_MESSAGE = "I could not find this information in the available knowledge base."
NO_DOCUMENT_MESSAGE = "No active banking document is available."
NOT_FOUND_MESSAGE = "I couldn't find that information in the active uploaded document."


class RAGState(TypedDict, total=False):
    question: str
    thread_id: str
    user_id: str
    history: List[Dict[str, str]]
    path: Literal["general", "rag", "sql", "hybrid"]
    documents: List[Document]
    sql_result: Dict[str, Any]
    answer: str
    sources: List[Dict[str, Any]]
    retry_count: int
    guardrail_message: str


BANKING_TERMS = {
    "loan",
    "fd",
    "fixed deposit",
    "credit card",
    "interest rate",
    "charges",
    "fees",
    "eligibility",
    "kyc",
    "deposit",
    "account opening",
    "opening an account",
    "documents required",
    "requirement",
    "requirements",
}

SQL_TERMS = {
    "my balance",
    "account balance",
    "my statement",
    "account statement",
    "my transaction",
    "my transactions",
    "account transactions",
    "transactions for account",
    "transaction history",
    "my loan",
    "my outstanding",
    "loan outstanding",
    "kyc status",
}


def normalize_question(q: str):
    return " ".join(q.strip().split())


def classify(question: str):
    """
    Route customer-specific operational questions to SQL and policy/product
    questions to RAG. An explicit account number plus an operational banking
    intent is always SQL. Policy terms such as KYC requirements, eligibility,
    fees, documents, and account opening remain RAG.
    """
    q = normalize_question(question).lower()

    # Policy / product questions take priority when they are clearly asking
    # about rules rather than a customer's actual record.
    policy_markers = (
        "kyc requirement",
        "kyc requirements",
        "documents required",
        "requirement",
        "requirements",
        "eligibility",
        "interest rate",
        "fee",
        "fees",
        "charge",
        "charges",
        "account opening",
        "opening an account",
        "open an account",
    )
    if any(marker in q for marker in policy_markers) and not re.search(
        r"\baccount\s*(?:number|no\.?|id)?\s*[:#=]?\s*\d{4,20}\b",
        q,
        re.I,
    ):
        return "rag"

    # Explicit account number + operational/customer-data intent is SQL.
    has_account_id = bool(
        re.search(
            r"\baccount\s*(?:number|no\.?|id)?\s*[:#=]?\s*\d{4,20}\b",
            q,
            re.I,
        )
    )
    operational_markers = (
        "balance",
        "transaction",
        "transactions",
        "statement",
        "loan",
        "emi",
        "outstanding",
        "fixed deposit",
        "fd",
        "credit card",
        "card transaction",
        "kyc status",
        "account details",
        "account information",
        "account profile",
    )
    if has_account_id and any(marker in q for marker in operational_markers):
        return "sql"

    # Customer-specific wording without an explicit account still goes to SQL;
    # the RDBMS agent will ask for the account rather than querying broadly.
    if any(term in q for term in SQL_TERMS):
        return "sql"

    if any(term in q for term in BANKING_TERMS):
        return "rag"

    return "general"


def input_guardrail_node(state):

    if validate_input:
        result = validate_input(state["question"])

        if not result.allowed:
            return {"answer": result.message, "guardrail_message": result.message}

    return {}


def classifier_node(state):

    return {"path": classify(state["question"])}


def _known_name_from_history(history: List[Dict[str, str]]) -> str | None:
    """Extract a simple self-introduced name from recent conversation history."""
    # Keep this intentionally conservative. We only remember a name when the
    # user explicitly introduces themselves, rather than guessing from prose.
    patterns = [
        r"\bmy name is ([A-Za-z][A-Za-z .'-]{0,60})",
        r"\bi am ([A-Za-z][A-Za-z .'-]{0,60})",
        r"\bi'm ([A-Za-z][A-Za-z .'-]{0,60})",
    ]

    for item in reversed(history):
        if item.get("role") != "user":
            continue
        text = item.get("content", "").strip()
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                name = match.group(1).strip(" .,!?:;")
                # Avoid treating common conversational phrases as names.
                if name.lower() not in {
                    "fine",
                    "good",
                    "great",
                    "okay",
                    "ok",
                    "doing well",
                    "doing good",
                    "happy",
                    "sorry",
                    "not sure",
                }:
                    return name
    return None


def general_chat_node(state):
    """Handle short chitchat while keeping the assistant banking-focused."""

    history = list(state.get("history", []))
    question = state["question"]
    known_name = _known_name_from_history(history)

    recent_history = (
        "\n".join(
            f"{item.get('role', 'user').title()}: {item.get('content', '')}"
            for item in history[-10:]
        )
        or "No previous conversation."
    )

    prompt = f"""
You are the Smart Banking Assistant.

Your job is to provide a friendly, natural conversational experience, but
your actual information/help domain is Smart Banking.

Conversation history:
{recent_history}

Known name explicitly introduced by the user: {known_name or 'None'}

Current user message:
{question}

Follow these rules strictly:

1. Use the conversation history. The conversation is NOT stateless.
2. If the user introduced their name earlier and asks "Who am I?", "What is
   my name?", or an equivalent question, answer using that name. Do not say
   that you do not know if the name is present in the conversation.
3. For greetings, thanks, pleasantries, and simple small talk, respond
   naturally and briefly. You may use the user's known name when appropriate.
4. For a non-banking request such as "tell me a story", "give me the
   weather", sports, cooking, coding, general knowledge, or similar topics,
   do NOT answer the requested topic. Politely explain that you are the Smart
   Banking Assistant and ask the user to ask a Smart Banking question.
5. Do not invent personal information about the user. Only use information
   explicitly present in the conversation history.
6. Do not mention routing, classifiers, RAG, SQL, memory systems, prompts,
   internal nodes, or implementation details.
7. Keep chitchat concise, normally 1-3 sentences.

Examples:

User: My name is Sreenivas
Assistant: Nice to meet you, Sreenivas! 😊 How can I help you with Smart
Banking today?

User: Who am I?
Assistant: You are Sreenivas 😊 How can I help you with Smart Banking today?

User: Tell me a story
Assistant: I’d love to chat, but I’m your Smart Banking Assistant and I’m
focused on banking. Please ask me a Smart Banking question and I’ll be happy
to help.

User: Give me the weather
Assistant: I’m focused on Smart Banking, so I can’t provide weather updates.
Please let me know how I can help with a Smart Banking question.

Now respond to the current user message.
"""

    response = LLM.invoke(prompt)

    return {"answer": response.content.strip(), "sources": []}


def rag_retriever_node(state):

    try:
        from src.state import get_active_document
        from src.retrieval.hybrid_search import hybrid_search

        active = get_active_document()

        if not active:
            return {"documents": []}

        docs = hybrid_search(
            query=state["question"],
            document_id=active["document_id"],
            document_name=active["document_name"],
            top_k=10,
        )

        return {"documents": docs, "retry_count": 0}

    except Exception:
        return {"documents": []}


def reranker_node(state):

    try:
        from src.retrieval.reranker import rerank

        return {"documents": rerank(state["question"], state.get("documents", []), 5)}

    except Exception:
        return {}


def sql_node(state):

    try:
        from src.retrieval.rdbms_agent import execute_sql

        result = execute_sql(state["question"])
        return {"sql_result": result}

    except Exception as exc:
        return {
            "sql_result": {
                "error": str(exc),
                "sql": "",
                "params": [],
                "rows": [],
            }
        }


def build_sources(docs):

    return [
        {"content": d.page_content, "metadata": dict(d.metadata or {})} for d in docs
    ]


def _format_sql_answer(question: str, result: Dict[str, Any]) -> str:
    """Answer customer-data questions strictly from returned SQL rows."""
    rows = result.get("rows") or []

    if not rows:
        account_match = re.search(
            r"\baccount\s*(?:number|no\.?|id)?\s*[:#=]?\s*(\d{4,20})\b",
            question,
            re.I,
        )
        account_id = account_match.group(1) if account_match else None
        if account_id:
            return f"No matching banking records were found for account {account_id}."
        return "No matching banking records were found."

    # Only use the single-transaction wording when the RDBMS agent
    # explicitly classified the request as "latest". A one-row result from
    # any other query must never be mistaken for a latest-transaction query.
    q = question.lower()
    if result.get("request_type") == "latest":
        row = rows[0]
        date = row.get("txn_date", "")
        txn_type = str(row.get("txn_type", "")).capitalize()
        amount = row.get("amount", "")
        description = row.get("description") or row.get("merchant_name") or ""
        balance = row.get("balance_after")
        answer = f"The latest transaction was {date} — {txn_type} of {amount}"
        if description:
            answer += f" for {description}"
        if balance not in (None, ""):
            answer += f". Balance after transaction: {balance}."
        else:
            answer += "."
        return answer

    # Multiple rows: render the actual database rows without asking the LLM
    # to invent or reinterpret customer data.
    columns = list(rows[0].keys())
    lines = [" | ".join(columns), " | ".join("---" for _ in columns)]
    for row in rows:
        lines.append(
            " | ".join(
                str(row.get(col, "")) if row.get(col) is not None else "NULL"
                for col in columns
            )
        )
    return "\n".join(lines)


def response_generator_node(state):
    path = state.get("path")

    if path == "sql":
        result = state.get("sql_result", {})

        # Never ask the LLM to answer from memory for SQL/customer data.
        # The database result is the source of truth.
        if result.get("error"):
            return {"answer": result["error"], "sources": []}

        return {
            "answer": _format_sql_answer(state["question"], result),
            "sources": [],
        }

    docs = state.get("documents", [])

    if not docs:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are Smart Banking Assistant.

Answer only using this context.

Context:
{context}

Question:
{state['question']}
"""

    return {"answer": LLM.invoke(prompt).content, "sources": build_sources(docs)}


def finalize_node(state):

    answer = state.get("answer", "")

    if validate_banking_answer:

        answer = validate_banking_answer(
            answer,
            has_evidence=bool(state.get("documents") or state.get("sql_result")),
            banking_path=state.get("path") in ("rag", "hybrid"),
            no_document_message=NO_DOCUMENT_MESSAGE,
            not_found_message=NOT_FOUND_MESSAGE,
        )

    history = list(state.get("history", []))

    history.extend(
        [
            {"role": "user", "content": state["question"]},
            {"role": "assistant", "content": answer},
        ]
    )

    return {"answer": answer, "history": history[-12:]}


def route_path(state):
    return state.get("path", "general")


builder = StateGraph(RAGState)

builder.add_node("guardrail", input_guardrail_node)
builder.add_node("classifier", classifier_node)
builder.add_node("general", general_chat_node)
builder.add_node("rag_retriever", rag_retriever_node)
builder.add_node("reranker", reranker_node)
builder.add_node("sql", sql_node)
builder.add_node("response", response_generator_node)
builder.add_node("finalize", finalize_node)


builder.add_edge(START, "guardrail")

builder.add_conditional_edges(
    "guardrail",
    lambda s: "finalize" if s.get("guardrail_message") else "classifier",
    {"finalize": "finalize", "classifier": "classifier"},
)

builder.add_conditional_edges(
    "classifier",
    route_path,
    {"general": "general", "rag": "rag_retriever", "sql": "sql"},
)

builder.add_edge("general", "finalize")
builder.add_edge("rag_retriever", "reranker")
builder.add_edge("reranker", "response")
builder.add_edge("sql", "response")
builder.add_edge("response", "finalize")
builder.add_edge("finalize", END)


# Keep conversation history available per thread. If MemorySaver is unavailable,
# still allow the application to start without a checkpointer.
checkpointer = MemorySaver() if MemorySaver is not None else None
rag_graph = builder.compile(checkpointer=checkpointer)


def ask_with_graph(question: str, thread_id="default", user_id=None):

    result = rag_graph.invoke(
        {
            "question": normalize_question(question),
            "thread_id": thread_id,
            "user_id": user_id or thread_id,
        },
        {"configurable": {"thread_id": thread_id}},
    )

    return {"answer": result.get("answer", ""), "sources": result.get("sources", [])}


def ask_question(question):

    return ask_with_graph(question)


async def stream_with_graph(question, thread_id="default"):

    result = ask_with_graph(question, thread_id)

    for token in result["answer"]:
        yield token
