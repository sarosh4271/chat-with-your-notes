"""Streamlit interface for Chat with Your Notes."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

from note_chat.ingestion import (
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_PDF_PAGES,
    MAX_TOTAL_BYTES,
    fingerprint_files,
    load_uploaded_files,
    split_documents,
)
from note_chat.rag import MAX_QUESTION_CHARS, ask, build_rag_session, source_label

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Chat with Your Notes", page_icon="📚", layout="centered")
st.title("📚 Chat with Your Notes")
st.caption("Upload PDFs or text notes, then ask questions about those files.")

MAX_QUESTIONS_PER_SESSION = 20
MAX_INDEXES_PER_SESSION = 3
MAX_HISTORY_MESSAGES = 12


def reset_chat() -> None:
    st.session_state.messages = []


def langchain_history() -> list[HumanMessage | AIMessage]:
    history: list[HumanMessage | AIMessage] = []
    for message in st.session_state.messages[-MAX_HISTORY_MESSAGES:]:
        message_type = HumanMessage if message["role"] == "user" else AIMessage
        history.append(message_type(content=message["content"]))
    return history


for key, default in {
    "rag_session": None,
    "indexed_fingerprint": None,
    "messages": [],
    "questions_asked": 0,
    "indexes_built": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

with st.sidebar:
    st.header("Knowledge base")
    uploaded_files = st.file_uploader(
        "Choose documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help=(
            f"Up to {MAX_FILES} files, {MAX_FILE_BYTES // (1024 * 1024)} MB each, "
            f"{MAX_TOTAL_BYTES // (1024 * 1024)} MB combined."
        ),
    )
    api_key = st.text_input(
        "Your OpenAI API key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Used for this session and never written to disk or application logs.",
    )
    st.caption(
        "Bring your own key. On a hosted copy, the key passes through that app's server, "
        "so only use deployments you trust. For maximum privacy, clone and run locally."
    )
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5-mini")
    chunk_size = st.slider("Chunk size", 400, 2_000, 1_000, 100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 200, 50)

    index_clicked = st.button(
        "Index documents",
        type="primary",
        use_container_width=True,
        disabled=(
            not uploaded_files
            or not api_key
            or st.session_state.indexes_built >= MAX_INDEXES_PER_SESSION
        ),
    )

    if st.button("Clear chat", use_container_width=True):
        reset_chat()

    st.divider()
    st.caption(
        "Chroma runs in memory on this server. Document text is sent to OpenAI to create "
        "embeddings, and retrieved excerpts are sent to the chat model."
    )
    with st.expander("Usage limits"):
        st.markdown(
            f"- {MAX_FILES} files per index\n"
            f"- {MAX_FILE_BYTES // (1024 * 1024)} MB per file; "
            f"{MAX_TOTAL_BYTES // (1024 * 1024)} MB combined\n"
            f"- {MAX_PDF_PAGES} pages per PDF\n"
            f"- {MAX_QUESTION_CHARS:,} characters per question\n"
            f"- {MAX_INDEXES_PER_SESSION} indexes per session\n"
            f"- {MAX_QUESTIONS_PER_SESSION} questions per session"
        )

if uploaded_files:
    current_fingerprint = fingerprint_files(uploaded_files)
    if (
        st.session_state.indexed_fingerprint
        and current_fingerprint != st.session_state.indexed_fingerprint
    ):
        st.warning("Your selected files changed. Click **Index documents** to use the new set.")

if index_clicked:
    if chunk_overlap >= chunk_size:
        st.error("Chunk overlap must be smaller than chunk size.")
    else:
        try:
            with st.status("Building your knowledge base…", expanded=True) as status:
                st.write("Reading documents")
                documents = load_uploaded_files(uploaded_files)
                st.write("Splitting text into chunks")
                chunks = split_documents(documents, chunk_size, chunk_overlap)
                st.write(f"Embedding {len(chunks)} chunks in Chroma")
                st.session_state.rag_session = build_rag_session(chunks, api_key, model)
                st.session_state.indexed_fingerprint = fingerprint_files(uploaded_files)
                st.session_state.indexes_built += 1
                reset_chat()
                status.update(label="Knowledge base ready", state="complete", expanded=False)
            st.success(
                f"Indexed {len(uploaded_files)} file(s) into {len(chunks)} searchable chunks."
            )
        except ValueError as error:
            st.session_state.rag_session = None
            st.error(f"Could not index the documents: {error}")
        except Exception:
            logger.exception("Document indexing failed")
            st.session_state.rag_session = None
            st.error("Indexing failed. Check the file format and API configuration, then retry.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved sources"):
                for source in message["sources"]:
                    st.markdown(f"- **{source['label']}** — {source['preview']}")

question = st.chat_input(
    "Ask something about your notes…",
    max_chars=MAX_QUESTION_CHARS,
    disabled=(
        st.session_state.rag_session is None
        or st.session_state.questions_asked >= MAX_QUESTIONS_PER_SESSION
    ),
)

if question:
    history = langchain_history()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"), st.spinner("Searching your notes…"):
        try:
            result = ask(st.session_state.rag_session, question, history)
            answer = result["answer"]
            unique_sources: dict[str, dict[str, str]] = {}
            for document in result.get("context", []):
                label = source_label(document)
                preview = " ".join(document.page_content.split())[:180]
                unique_sources[label] = {"label": label, "preview": f"{preview}…"}
            sources = list(unique_sources.values())

            st.markdown(answer)
            if sources:
                with st.expander("Retrieved sources"):
                    for source in sources:
                        st.markdown(f"- **{source['label']}** — {source['preview']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            st.session_state.questions_asked += 1
        except ValueError as error:
            st.error(f"Could not answer the question: {error}")
        except Exception:
            logger.exception("Question answering failed")
            st.error("The answer request failed. Please retry in a moment.")

if st.session_state.rag_session is None:
    st.info("Add your API key, upload at least one document, and click **Index documents**.")
elif st.session_state.questions_asked >= MAX_QUESTIONS_PER_SESSION:
    st.warning("This session has reached its question limit. Start a new session to continue.")

if st.session_state.indexes_built >= MAX_INDEXES_PER_SESSION:
    st.sidebar.warning("This session has reached its document-indexing limit.")
