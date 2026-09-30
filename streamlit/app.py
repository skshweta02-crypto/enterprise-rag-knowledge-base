import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Enterprise RAG Knowledge Base",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Enterprise RAG Knowledge Base & Orchestration Service")
st.markdown("Upload enterprise documents (.pdf, .docx, .txt, .md) and query their contents with AI-grounded retrieval.")

# --- Sidebar: System Status & Document Management ---
with st.sidebar:
    st.header("⚙️ Backend Service")
    try:
        health_res = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if health_res.status_code == 200:
            data = health_res.json()
            st.success(f"Status: {data['status'].upper()} ({data['environment']})")
        else:
            st.error("Backend unhealthy")
    except Exception:
        st.error("Cannot connect to FastAPI backend at " + API_BASE_URL)

    st.divider()
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader(
        "Upload Enterprise Document",
        type=["pdf", "docx", "txt", "md"],
        help="Supported formats: PDF, DOCX, TXT, MD",
    )

    if uploaded_file is not None:
        if st.button("Upload & Ingest Document", use_container_width=True):
            with st.spinner("Uploading document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    up_res = requests.post(f"{API_BASE_URL}/documents/upload", files=files, timeout=10)

                    if up_res.status_code == 200:
                        up_data = up_res.json()
                        st.info(f"File uploaded successfully ({up_data['file_size_bytes']} bytes)")

                        with st.spinner("Chunking & Embedding into Vector Database..."):
                            ingest_payload = {
                                "saved_path": up_data["saved_path"],
                                "filename": up_data["filename"],
                                "document_id": up_data["document_id"],
                            }
                            ingest_res = requests.post(
                                f"{API_BASE_URL}/documents/ingest",
                                json=ingest_payload,
                                timeout=60,
                            )
                            if ingest_res.status_code == 200:
                                ingest_data = ingest_res.json()
                                st.success(
                                    f"Successfully ingested '{ingest_data['filename']}'!\n\n"
                                    f"Total Chunks Created: **{ingest_data['total_chunks']}**"
                                )
                            else:
                                st.error(f"Ingestion failed: {ingest_res.text}")
                    else:
                        st.error(f"Upload failed: {up_res.text}")
                except Exception as e:
                    st.error(f"Error during ingestion: {str(e)}")

    st.divider()
    st.header("📂 Knowledge Base Index")
    try:
        docs_res = requests.get(f"{API_BASE_URL}/documents", timeout=5)
        if docs_res.status_code == 200:
            docs_data = docs_res.json()
            st.metric("Total Indexed Files", docs_data.get("total_documents", 0))
            for doc in docs_data.get("documents", []):
                st.caption(f"• **{doc['filename']}** ({doc['file_size_bytes']} bytes)")
    except Exception:
        st.warning("Could not fetch document index list")


# --- Main Area: RAG Question Answering ---
st.subheader("🔍 Ask Questions")

question = st.text_area(
    "Enter your question regarding the uploaded knowledge base:",
    placeholder="e.g., What is the company password policy?",
    height=100,
)

col1, col2 = st.columns([1, 4])
with col1:
    top_k = st.slider("Top K Chunks", min_value=1, max_value=10, value=4)
with col2:
    st.write("")
    st.write("")
    ask_button = st.button("Ask Question", type="primary", use_container_width=False)

if ask_button:
    if not question.strip():
        st.warning("Please enter a question before submitting.")
    else:
        with st.spinner("Running semantic search and generating answer..."):
            try:
                query_payload = {"question": question, "top_k": top_k}
                res = requests.post(f"{API_BASE_URL}/query", json=query_payload, timeout=60)

                if res.status_code == 200:
                    data = res.json()

                    st.markdown("### 🤖 Answer")
                    st.info(data["answer"])

                    # Performance Analysis Metrics
                    st.markdown("### ⚡ Query Analysis & Performance")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Latency", f"{data['latency_ms']:.1f} ms")
                    m2.metric("Retrieval Latency", f"{data['retrieval_latency_ms']:.1f} ms")
                    m3.metric("LLM Latency", f"{data['llm_latency_ms']:.1f} ms")
                    m4.metric("Retrieved Chunks", len(data["retrieved_chunks"]))

                    # Sources Display
                    st.markdown("### 📌 Sources & Grounding")
                    if data["sources"]:
                        for source in data["sources"]:
                            st.markdown(f"- 📄 **{source}**")
                    else:
                        st.caption("No sources cited (context insufficient)")

                    # Retrieved Chunks Details
                    with st.expander("🔍 Inspect Retrieved Context Chunks"):
                        for idx, chunk in enumerate(data["retrieved_chunks"], 1):
                            st.markdown(
                                f"**Chunk {idx}** | Source: `{chunk['source_display']}` | Distance Score: `{chunk['score']:.4f}`"
                            )
                            st.code(chunk["content"], language="text")
                else:
                    st.error(f"Query error: {res.text}")
            except Exception as e:
                st.error(f"Failed to execute query: {str(e)}")
