import time
import uuid

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = (
    "http://127.0.0.1:8000"
)

ASK_URL = (
    f"{API_BASE_URL}/ask"
)

UPLOAD_URL = (
    f"{API_BASE_URL}/upload"
)

HEALTH_URL = (
    f"{API_BASE_URL}/health"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Banking Assistant",
    page_icon="🏦",
    layout="wide",
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# IMPORTANT:
# One thread_id per Streamlit conversation.

if "thread_id" not in st.session_state:

    st.session_state.thread_id = (
        str(uuid.uuid4())
    )


# ============================================================
# SOURCE DISPLAY
# ============================================================

def display_sources(
    sources,
) -> None:

    if not sources:
        return

    with st.expander(
        f"📄 Sources ({len(sources)})"
    ):

        for index, source in enumerate(
            sources,
            start=1,
        ):

            metadata = source.get(
                "metadata",
                {},
            )

            st.markdown(
                f"""
**Source {index}**

**Document:** {
    metadata.get(
        "document_name",
        "Unknown document",
    )
}

**Page:** {
    metadata.get(
        "page",
        "N/A",
    )
}

**Chunk Type:** {
    metadata.get(
        "chunk_type",
        "unknown",
    )
}

**Retriever:** {
    metadata.get(
        "retriever",
        "unknown",
    )
}
"""
            )

            scores = (
                (
                    "vector_score",
                    "Vector Score",
                ),
                (
                    "fts_score",
                    "FTS Score",
                ),
                (
                    "rrf_score",
                    "RRF Score",
                ),
            )

            for key, label in scores:

                value = metadata.get(
                    key
                )

                if value is not None:

                    try:

                        st.write(
                            f"{label}: "
                            f"{float(value):.6f}"
                        )

                    except (
                        TypeError,
                        ValueError,
                    ):

                        st.write(
                            f"{label}: "
                            f"{value}"
                        )

            content = source.get(
                "content",
                "",
            )

            if content:

                st.text(
                    content
                )

            st.divider()


# ============================================================
# API ERROR
# ============================================================

def get_api_error(
    response,
) -> str:

    message = (
        f"FastAPI returned "
        f"HTTP {response.status_code}"
    )

    try:

        data = response.json()

        message += (
            f"\n\n"
            f"{data.get('detail', data)}"
        )

    except Exception:

        message += (
            f"\n\n"
            f"{response.text}"
        )

    return message


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "# 🏦 Smart Banking Assistant"
)

st.caption(
    "Ask questions about NorthStar Bank "
    "products, loans, rates, eligibility, "
    "charges and policies."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📚 Knowledge Base"
    )

    uploaded_file = (
        st.file_uploader(
            "Choose a PDF",
            type=["pdf"],
        )
    )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    if st.button(
        "📤 Upload & Ingest",
        use_container_width=True,
    ):

        if uploaded_file is None:

            st.warning(
                "Please select a PDF "
                "before uploading."
            )

        else:

            try:

                with st.spinner(
                    "Uploading PDF..."
                ):

                    response = (
                        requests.post(
                            UPLOAD_URL,

                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    "application/pdf",
                                )
                            },

                            timeout=60,
                        )
                    )

                if response.status_code != 200:

                    st.error(
                        get_api_error(
                            response
                        )
                    )

                else:

                    result = (
                        response.json()
                    )

                    job_id = (
                        result.get(
                            "job_id"
                        )
                    )

                    st.success(
                        "PDF uploaded successfully."
                    )

                    st.write(
                        f"**File:** "
                        f"`{result.get('filename')}`"
                    )

                    st.write(
                        f"**Size:** "
                        f"{result.get('size_bytes', 0):,} "
                        f"bytes"
                    )

                    if job_id:

                        status_placeholder = (
                            st.empty()
                        )

                        progress_bar = (
                            st.progress(0)
                        )

                        max_attempts = 300

                        for attempt in range(
                            max_attempts
                        ):

                            try:

                                status_response = (
                                    requests.get(
                                        f"{UPLOAD_URL}/status/{job_id}",
                                        timeout=10,
                                    )
                                )

                                if (
                                    status_response.status_code
                                    != 200
                                ):

                                    status_placeholder.error(
                                        "Unable to retrieve "
                                        "ingestion status."
                                    )

                                    break

                                data = (
                                    status_response.json()
                                )

                                status = data.get(
                                    "status",
                                    "unknown",
                                )

                                # ----------------------------
                                # QUEUED
                                # ----------------------------

                                if status == "queued":

                                    status_placeholder.info(
                                        "⏳ Waiting for "
                                        "ingestion to start..."
                                    )

                                # ----------------------------
                                # PROCESSING
                                # ----------------------------

                                elif status == "processing":

                                    status_placeholder.info(
                                        "🔄 PDF ingestion "
                                        "is running..."
                                    )

                                # ----------------------------
                                # SUCCESS
                                # ----------------------------

                                elif status == "success":

                                    progress_bar.progress(
                                        100
                                    )

                                    status_placeholder.success(
                                        "✅ PDF ingestion "
                                        "completed successfully."
                                    )

                                    st.write(
                                        f"**Chunks created:** "
                                        f"{data.get('chunks', 0)}"
                                    )

                                    break

                                # ----------------------------
                                # FAILED
                                # ----------------------------

                                elif status == "failed":

                                    status_placeholder.error(
                                        "❌ PDF ingestion failed."
                                    )

                                    st.error(
                                        data.get(
                                            "message",
                                            "Unknown "
                                            "ingestion error.",
                                        )
                                    )

                                    break

                                progress = min(
                                    int(
                                        (
                                            (
                                                attempt + 1
                                            )
                                            / max_attempts
                                        )
                                        * 100
                                    ),
                                    95,
                                )

                                progress_bar.progress(
                                    progress
                                )

                                time.sleep(
                                    1
                                )

                            except (
                                requests.RequestException
                            ) as exc:

                                status_placeholder.error(
                                    "Error checking "
                                    f"ingestion status: "
                                    f"{exc}"
                                )

                                break

                        else:

                            status_placeholder.warning(
                                "Ingestion is taking "
                                "longer than expected."
                            )

            except (
                requests.RequestException
            ) as exc:

                st.error(
                    "Could not connect "
                    f"to FastAPI.\n\n{exc}"
                )

    st.divider()

    # ========================================================
    # API
    # ========================================================

    st.header(
        "⚙️ API"
    )

    st.code(
        API_BASE_URL
    )

    if st.button(
        "🩺 Check API Health",
        use_container_width=True,
    ):

        try:

            response = (
                requests.get(
                    HEALTH_URL,
                    timeout=10,
                )
            )

            if response.status_code == 200:

                st.success(
                    "API is healthy."
                )

            else:

                st.error(
                    f"API returned "
                    f"HTTP "
                    f"{response.status_code}"
                )

        except (
            requests.RequestException
        ) as exc:

            st.error(
                "Cannot connect "
                f"to FastAPI:\n\n{exc}"
            )

    st.divider()

    # ========================================================
    # THREAD
    # ========================================================

    st.caption(
        f"Thread ID:\n"
        f"{st.session_state.thread_id}"
    )

    # ========================================================
    # CLEAR CHAT
    # ========================================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        # IMPORTANT:
        # Create a new LangGraph thread.

        st.session_state.thread_id = (
            str(uuid.uuid4())
        )

        st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in (
    st.session_state.messages
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"]
            == "assistant"
        ):

            display_sources(
                message.get(
                    "sources",
                    [],
                )
            )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a banking question..."
)


# ============================================================
# ASK QUESTION
# ============================================================

if question:

    question = question.strip()

    if not question:

        st.stop()

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # --------------------------------------------------------
    # Assistant
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        response_placeholder = (
            st.empty()
        )

        try:

            with st.spinner(
                "Searching the banking "
                "knowledge base..."
            ):

                response = (
                    requests.post(

                        ASK_URL,

                        json={
                            "question": question,

                            # IMPORTANT:
                            # Send thread_id,
                            # not conversation_id.

                            "thread_id": (
                                st.session_state
                                .thread_id
                            ),
                        },

                        timeout=120,
                    )
                )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if response.status_code == 200:

                result = (
                    response.json()
                )

                answer = result.get(
                    "answer",
                    "No answer returned.",
                )

                sources = result.get(
                    "sources",
                    [],
                )

                # Keep server-confirmed thread ID.

                returned_thread_id = (
                    result.get(
                        "thread_id"
                    )
                )

                if returned_thread_id:

                    st.session_state.thread_id = (
                        returned_thread_id
                    )

                response_placeholder.markdown(
                    answer
                )

                display_sources(
                    sources
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            # ------------------------------------------------
            # API ERROR
            # ------------------------------------------------

            else:

                error_message = (
                    get_api_error(
                        response
                    )
                )

                response_placeholder.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )

        # ----------------------------------------------------
        # CONNECTION ERROR
        # ----------------------------------------------------

        except requests.ConnectionError:

            error_message = (
                "❌ Cannot connect to FastAPI.\n\n"
                "Run:\n\n"
                "`uv run uvicorn "
                "src.api.main:app --reload`"
            )

            response_placeholder.error(
                error_message
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )

        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        except requests.Timeout:

            error_message = (
                "⏱️ The request timed out "
                "while waiting for the "
                "RAG response."
            )

            response_placeholder.error(
                error_message
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )

        # ----------------------------------------------------
        # REQUEST ERROR
        # ----------------------------------------------------

        except requests.RequestException as exc:

            error_message = (
                f"❌ Request failed:\n\n"
                f"{exc}"
            )

            response_placeholder.error(
                error_message
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )

        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        except Exception as exc:

            error_message = (
                f"❌ Unexpected error:\n\n"
                f"{exc}"
            )

            response_placeholder.error(
                error_message
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )