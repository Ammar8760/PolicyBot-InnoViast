"""
PolicyBot — Streamlit UI.

To run:
    streamlit run app.py

Before running for the first time, you must index the knowledge base:
    python -m src.ingest
"""


from __future__ import annotations
import sys
import os
import time

try:
    import streamlit as st
except ModuleNotFoundError as exc:
    raise RuntimeError(
        "Streamlit is required to run this app. Install it with `pip install streamlit`."
    ) from exc

from src import config, generator, retriever, vector_store

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------ state ---
if "messages" not in st.session_state:
    st.session_state.messages = []


def get_stats() -> dict:
    try:
        info = vector_store.stats()
        info.setdefault("backend", config.VECTOR_BACKEND)
        return info
    except Exception as exc:
        return {
            "chunks": 0,
            "sources": [],
            "backend": config.VECTOR_BACKEND,
            "error": str(exc),
        }


# ---------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.title("📘 " + config.APP_TITLE)
    st.caption(config.APP_SUBTITLE)
    st.divider()

    stats = get_stats()

    st.subheader("Knowledge Base")
    st.caption(f"Vector store: **{stats.get('backend', config.VECTOR_BACKEND)}**")

    if stats.get("error"):
        st.error(f"Error opening index: {stats['error']}")
    elif stats["chunks"] == 0:
        st.warning(
            "Index is empty.\n\n"
            "Run this in your terminal:\n\n"
            "```\npython -m src.ingest\n```"
        )
    else:
        col_a, col_b = st.columns(2)
        col_a.metric("Chunks", stats["chunks"])
        col_b.metric("Documents", len(stats["sources"]))
        with st.expander("Which documents are indexed?"):
            for source in stats["sources"]:
                st.write(f"• `{source}`")

    st.divider()

    st.subheader("Retrieval Settings")
    top_k = st.slider(
        "Top-K chunks",
        min_value=1,
        max_value=8,
        value=config.TOP_K,
        help="Number of chunks sent to the LLM for each query. "
        "Higher = more context but higher cost and potential noise.",
    )
    threshold = st.slider(
        "Relevance threshold",
        min_value=0.0,
        max_value=1.0,
        value=config.SIMILARITY_THRESHOLD,
        step=0.05,
        help="Chunks scoring below this value are rejected and trigger a fallback "
        "message. Higher = stricter (safer), lower = lenient (more answers).",
    )

    st.divider()

    st.subheader("Document Upload")
    uploaded = st.file_uploader(
        "Add a new document",
        type=["md", "txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded and st.button("Add to index", use_container_width=True):
        config.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        for file in uploaded:
            (config.UPLOADS_DIR / file.name).write_bytes(file.getvalue())

        with st.spinner("Chunking + embedding in progress..."):
            from src.ingest import ingest

            ingest(keep_existing=False)
        st.success(f"{len(uploaded)} file(s) indexed successfully")
        st.rerun()

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ------------------------------------------------------------------- main ---
st.title("Nexora HR & Policy Assistant")
st.caption(
    "This assistant provides answers strictly from official company policy documents. "
    "Sources are displayed with every response, and it will not fabricate answers for "
    "topics not found in the documents."
)

if not config.get_api_key():
    st.error(
        "**GOOGLE_API_KEY not found.**\n\n"
        "- Local: Rename `.env.example` to `.env` and add your key\n"
        "- Streamlit Cloud: Add your key in **Settings → Secrets**\n\n"
        "Get a free key here: https://aistudio.google.com/apikey"
    )
    st.stop()

# Suggested questions — for first-time users
if not st.session_state.messages and stats.get("chunks", 0) > 0:
    st.write("**Try asking these questions:**")
    cols = st.columns(3)
    samples = [
        "How much annual leave is provided, and what is the carry-forward policy?",
        "What are the eligibility requirements and core hours for remote work?",
        "Who needs to approve a travel expense of PKR 40,000?",
    ]
    for col, sample in zip(cols, samples):
        if col.button(sample, use_container_width=True):
            st.session_state.pending = sample
            st.rerun()


def render_sources(answer: generator.Answer) -> None:
    """
    Source display — Quality Bar requirement:
        "Source references should be visible when an answer comes from documents."

    We display not just the filename, but also the actual chunk text,
    allowing users to independently verify that the response is accurate.
    """
    hits = answer.sources
    if not hits:
        return

    with st.expander(f"📎 Sources ({len(hits)})", expanded=False):
        for i, hit in enumerate(hits, start=1):
            left, right = st.columns([4, 1])
            left.markdown(f"**{i}. `{hit.source}`** · chunk {hit.chunk_index}")
            right.markdown(f"`{hit.score:.3f}`")
            st.text(hit.text[:900] + ("..." if len(hit.text) > 900 else ""))
            if i < len(hits):
                st.divider()


# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("is_fallback"):
            st.warning(message["content"])
        else:
            st.markdown(message["content"])

        if message.get("answer") is not None:
            render_sources(message["answer"])
            meta = message.get("meta")
            if meta:
                st.caption(meta)


# New user query — from input box or sample question button
question = st.chat_input("Ask a question about the policies...")
if "pending" in st.session_state:
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            started = time.perf_counter()
            try:
                result = retriever.retrieve(question, top_k=top_k, threshold=threshold)
                answer = generator.generate(result, available_topics=stats["sources"])
            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.stop()
            elapsed = time.perf_counter() - started

        if answer.is_fallback:
            st.warning(answer.text)
        else:
            st.markdown(answer.text)

        render_sources(answer)

        meta = (
            f"⏱️ {elapsed:.1f}s · "
            f"{len(answer.sources)}/{len(result.all_hits)} chunks above threshold "
            f"({threshold:.2f}) · "
            f"best score {result.best_score:.3f} · model `{config.CHAT_MODEL}`"
        )
        st.caption(meta)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer.text,
            "is_fallback": answer.is_fallback,
            "answer": answer,
            "meta": meta,
        }
    )