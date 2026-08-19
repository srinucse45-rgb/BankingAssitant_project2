from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.config import LLM
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.rdbms_agent import run_rdbms_agent


# ============================================================
# CONFIGURATION
# ============================================================

FALLBACK_MESSAGE = (
    "I could not find this information in the banking "
    "knowledge base."
)

MAX_CONTEXT_DOCUMENTS = 5


# ============================================================
# GRAPH STATE
# ============================================================

class RAGState(TypedDict, total=False):

    question: str

    user_name: str

    intent: Literal[
        "conversation",
        "banking",
    ]

    route: Literal[
        "VECTOR_DB",
        "RDBMS",
    ]

    sql_query_executed: str

    documents: List[Document]

    answer: str

    sources: List[Dict[str, Any]]

    history: List[Dict[str, str]]


# ============================================================
# CONVERSATION MESSAGES
# ============================================================

CONVERSATION_MESSAGES = {
    "hi",
    "hello",
    "hey",
    "hi there",
    "hello there",
    "good morning",
    "good afternoon",
    "good evening",
    "thanks",
    "thank you",
    "thankyou",
    "thanks a lot",
    "thank you so much",
    "bye",
    "goodbye",
    "see you",
    "see you later",
}


# ============================================================
# HELPER - NORMALIZE QUESTION
# ============================================================

def normalize_question(
    question: str,
) -> str:

    return " ".join(
        question.strip().split()
    )


# ============================================================
# HELPER - EXTRACT USER NAME
# ============================================================

def extract_user_name(
    question: str,
) -> str:

    text = question.strip()

    lower_text = text.lower()

    patterns = (
        "my name is ",
        "i am ",
        "i'm ",
    )

    for pattern in patterns:

        if lower_text.startswith(pattern):

            name = text[
                len(pattern):
            ].strip()

            if name:

                return name.rstrip(
                    ".,!?"
                )

    return ""


# ============================================================
# HELPER - DETECT INTENT
# ============================================================

def detect_intent(
    question: str,
) -> Literal[
    "conversation",
    "banking",
]:

    text = question.lower().strip()

    # --------------------------------------------------------
    # Simple conversation
    # --------------------------------------------------------

    if text in CONVERSATION_MESSAGES:

        return "conversation"

    # --------------------------------------------------------
    # General conversation
    # --------------------------------------------------------

    if any(
        phrase in text
        for phrase in (
            "who are you",
            "what can you do",
            "how can you help",
        )
    ):

        return "conversation"

    # --------------------------------------------------------
    # Name statement
    # --------------------------------------------------------

    if extract_user_name(question):

        return "conversation"

    return "banking"


# ============================================================
# HELPER - BUILD SOURCES
# ============================================================

def build_sources(
    documents: List[Document],
) -> List[Dict[str, Any]]:

    return [
        {
            "content": doc.page_content,
            "metadata": dict(
                doc.metadata
            ),
        }
        for doc in documents
    ]


# ============================================================
# HELPER - CONVERSATION RESPONSE
# ============================================================

def conversation_response(
    text: str,
    user_name: str,
) -> str:

    name = (
        f" {user_name}"
        if user_name
        else ""
    )

    name_comma = (
        f", {user_name}"
        if user_name
        else ""
    )

    # --------------------------------------------------------
    # Greetings
    # --------------------------------------------------------

    if text in {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
    }:

        return (
            f"Hi{name}! 👋 "
            "Good morning! "
            "How can I help you today?"
        )

    # --------------------------------------------------------
    # Good morning
    # --------------------------------------------------------

    if text == "good morning":

        return (
            f"Good morning{name}! ☀️ "
            "How can I help you today?"
        )

    # --------------------------------------------------------
    # Good afternoon
    # --------------------------------------------------------

    if text == "good afternoon":

        return (
            f"Good afternoon{name}! ☀️ "
            "How can I help you today?"
        )

    # --------------------------------------------------------
    # Good evening
    # --------------------------------------------------------

    if text == "good evening":

        return (
            f"Good evening{name}! 🌆 "
            "How can I help you today?"
        )

    # --------------------------------------------------------
    # Thanks
    # --------------------------------------------------------

    if text in {
        "thanks",
        "thank you",
        "thankyou",
        "thanks a lot",
        "thank you so much",
    }:

        return (
            f"You're welcome{name_comma}! 😊 "
            "Let me know if you need help with "
            "NorthStar Bank products or policies."
        )

    # --------------------------------------------------------
    # Goodbye
    # --------------------------------------------------------

    if text in {
        "bye",
        "goodbye",
        "see you",
        "see you later",
    }:

        return (
            f"Goodbye{name_comma}! 👋 "
            "Have a great day!"
        )

    # --------------------------------------------------------
    # Default conversation
    # --------------------------------------------------------

    return (
        f"Hi{name}! 👋 "
        "How can I help you with "
        "NorthStar Bank products, loans, "
        "rates, eligibility, or charges?"
    )


# ============================================================
# NODE 1 - PREPARE QUERY
# ============================================================

def retrieve_node(
    state: RAGState,
) -> RAGState:

    question = normalize_question(
        state.get("question", "")
    )

    intent = detect_intent(question)
    current_name = extract_user_name(question)
    previous_name = state.get("user_name", "")
    user_name = current_name or previous_name

    return {
        "question": question,
        "intent": intent,
        "user_name": user_name,
        "documents": [],
    }


# ============================================================
# NODE 2 - ROUTER
# ============================================================

class RouteDecision:
    def __init__(self, route: str, reason: str):
        self.route = route
        self.reason = reason


def router_node(state: RAGState) -> RAGState:
    question = state.get("question", "")

    prompt = f"""
You are a routing component for a Smart Banking Assistant.
Classify the user's banking question into exactly one route.

VECTOR_DB:
Use this route when the question asks about banking policies, rules,
procedures, eligibility, fees/charges policy, product features, or other
information that should be answered from the document knowledge base.

RDBMS:
Use this route when the question requires data from structured banking tables:
accounts, transactions, loan_accounts, fixed_deposits, credit_cards,
card_transactions.

IMPORTANT SECURITY RULE:
Do not assume a generic request means retrieve all data.
The RDBMS agent must ask for a business identifier before returning
account-specific records. Identifiers include account number, loan number,
card number, or transaction id.

Examples:

User: Tell me the accounts
Route: RDBMS (agent should request account number)

User: Show account details for account 12345
Route: RDBMS

Return exactly one line in this format:
ROUTE=<VECTOR_DB or RDBMS>
REASON=<short reason>

Question: {question}
"""

    response = LLM.invoke(prompt).content.strip()
    match = __import__("re").search(r"ROUTE\s*=\s*(VECTOR_DB|RDBMS)", response, __import__("re").IGNORECASE)

    if not match:
        # Conservative default: document retrieval.
        route = "VECTOR_DB"
        reason = "Router response was not parseable; defaulting to document retrieval."
    else:
        route = match.group(1).upper()
        reason_match = __import__("re").search(r"REASON\s*=\s*(.*)", response, __import__("re").IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else "Classified by banking query type."

    print(f"[ROUTER] route={route} reason={reason}")
    return {"route": route}


# ============================================================
# NODE 3 - VECTOR/FTS HYBRID RETRIEVAL
# ============================================================

def vector_retrieve_node(state: RAGState) -> RAGState:
    question = state.get("question", "")

    print("\n" + "=" * 70)
    print(f"HYBRID SEARCH: {question}")
    print("=" * 70)

    # Existing hybrid_search is preserved: Vector + PostgreSQL FTS + RRF.
    documents = hybrid_search(
        query=question,
        top_k=MAX_CONTEXT_DOCUMENTS,
    )

    print(f"Retrieved documents: {len(documents)}")
    return {"documents": documents}


# ============================================================
# NODE 4 - RDBMS AGENT
# ============================================================

def rdbms_node(state: RAGState) -> RAGState:
    question = state.get("question", "")
    history = state.get("history", [])

    print("\n" + "=" * 70)
    print(f"RDBMS AGENT: {question}")
    print("=" * 70)

    result = run_rdbms_agent(
        question=question,
        history=history,
    )

    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "sql_query_executed": result.get("sql_query_executed", ""),
    }


# ============================================================
# NODE 2 - GENERATE
# ============================================================

def generate_node(
    state: RAGState,
) -> RAGState:

    question = state.get(
        "question",
        "",
    )

    intent = state.get(
        "intent",
        "banking",
    )

    user_name = state.get(
        "user_name",
        "",
    )

    documents = state.get(
        "documents",
        [],
    )

    # ========================================================
    # CONVERSATION
    # ========================================================

    if intent == "conversation":

        new_name = extract_user_name(
            question
        )

        # ----------------------------------------------------
        # User gives name
        # ----------------------------------------------------

        if new_name:

            return {
                "answer": (
                    f"Nice to meet you, "
                    f"{new_name}! 👋 "
                    "How can I help you today?"
                ),
                "sources": [],
                "user_name": new_name,
            }

        # ----------------------------------------------------
        # Normal conversation
        # ----------------------------------------------------

        return {
            "answer": conversation_response(
                question.lower().strip(),
                user_name,
            ),
            "sources": [],
        }

    # ========================================================
    # NO RETRIEVAL RESULTS
    # ========================================================

    if not documents:

        return {
            "answer": FALLBACK_MESSAGE,
            "sources": [],
        }

    # ========================================================
    # BUILD BANKING CONTEXT
    # ========================================================

    context = "\n\n".join(

        f"""
SOURCE {index}

Document:
{doc.metadata.get(
    "document_name",
    "Unknown document",
)}

Page:
{doc.metadata.get(
    "page",
    "N/A",
)}

Chunk Type:
{doc.metadata.get(
    "chunk_type",
    "unknown",
)}

Retriever:
{doc.metadata.get(
    "retriever",
    "unknown",
)}

RRF Score:
{doc.metadata.get(
    "rrf_score",
    0,
)}

Content:
{doc.page_content}
"""

        for index, doc in enumerate(
            documents,
            start=1,
        )
    )

    # ========================================================
    # CONVERSATION HISTORY
    # ========================================================

    history = state.get(
        "history",
        [],
    )

    recent_history = history[-6:]

    history_text = "\n".join(

        f"{item['role']}: "
        f"{item['content']}"

        for item in recent_history
    )

    if not history_text:

        history_text = (
            "No previous conversation."
        )

    # ========================================================
    # BANKING PROMPT
    # ========================================================

    prompt = f"""
You are the Smart Banking Assistant
for NorthStar Bank.

You are a friendly and conversational
banking assistant.

Answer the user's banking question using
ONLY the provided banking knowledge-base
context.

IMPORTANT RULES:

1. Do not invent banking information.

2. Do not use outside banking knowledge.

3. If the answer exists in the context,
   answer it clearly.

4. If a table contains the answer,
   preserve the values accurately.

5. If multiple customer categories exist,
   clearly identify which value belongs
   to each category.

6. Be concise and easy to understand.

7. Use the user's name naturally when
   appropriate.

8. If the answer is not available in
   the provided context, say exactly:

{FALLBACK_MESSAGE}

User name:
{user_name or "Not provided"}

Recent conversation:
{history_text}

User question:
{question}

Banking knowledge-base context:
{context}

Answer:
"""

    # ========================================================
    # LLM
    # ========================================================

    response = LLM.invoke(
        prompt
    )

    return {
        "answer": response.content,
        "sources": build_sources(
            documents
        ),
    }


# ============================================================
# NODE 3 - FINALIZE + MEMORY
# ============================================================

def finalize_node(
    state: RAGState,
) -> RAGState:

    history = list(
        state.get(
            "history",
            [],
        )
    )

    history.extend(
        [
            {
                "role": "user",
                "content": state.get(
                    "question",
                    "",
                ),
            },
            {
                "role": "assistant",
                "content": state.get(
                    "answer",
                    "",
                ),
            },
        ]
    )

    # Keep only recent messages.
    history = history[-10:]

    return {
        "question": state.get(
            "question",
            "",
        ),
        "answer": state.get(
            "answer",
            "",
        ),
        "sources": state.get(
            "sources",
            [],
        ),
        "user_name": state.get(
            "user_name",
            "",
        ),
        "history": history,
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def route_after_prepare(state: RAGState) -> str:
    if state.get("intent") == "conversation":
        return "CONVERSATION"
    return "ROUTER"


def route_after_router(state: RAGState) -> str:
    return state.get("route", "VECTOR_DB")


builder = StateGraph(RAGState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("router", router_node)
builder.add_node("vector_retrieve", vector_retrieve_node)
builder.add_node("rdbms_agent", rdbms_node)
builder.add_node("generate", generate_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "retrieve")

builder.add_conditional_edges(
    "retrieve",
    route_after_prepare,
    {
        "CONVERSATION": "generate",
        "ROUTER": "router",
    },
)

builder.add_conditional_edges(
    "router",
    route_after_router,
    {
        "VECTOR_DB": "vector_retrieve",
        "RDBMS": "rdbms_agent",
    },
)

builder.add_edge("vector_retrieve", "generate")
builder.add_edge("generate", "finalize")
builder.add_edge("rdbms_agent", "finalize")
builder.add_edge("finalize", END)

# ============================================================
# CHECKPOINTER
# ============================================================

checkpointer = InMemorySaver()

rag_graph = builder.compile(
    checkpointer=checkpointer,
)


# ============================================================
# PUBLIC API
# ============================================================

def ask_with_graph(
    question: str,
    thread_id: str = "default",
) -> Dict[str, Any]:

    question = normalize_question(
        question
    )

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    result = rag_graph.invoke(

        {
            "question": question,
        },

        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
    )

    return {
        "answer": result.get(
            "answer",
            "No answer generated.",
        ),
        "sources": result.get(
            "sources",
            [],
        ),
        "sql_query_executed": result.get(
            "sql_query_executed",
            "",
        ),
    }


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def ask_question(
    question: str,
) -> Dict[str, Any]:

    return ask_with_graph(
        question=question,
        thread_id="default",
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    thread_id = "local-test"

    questions = [
        "Hi",
        "My name is Sreevas",
        "What is the maximum LTV for a home loan?",
        "Thank you",
        "Hi",
    ]

    for question in questions:

        print(
            "\n" + "=" * 70
        )

        print(
            f"QUESTION: {question}"
        )

        print(
            "=" * 70
        )

        result = ask_with_graph(
            question=question,
            thread_id=thread_id,
        )

        print("\nANSWER:")

        print(
            result["answer"]
        )

        print(
            f"\nSOURCES: "
            f"{len(result['sources'])}"
        )