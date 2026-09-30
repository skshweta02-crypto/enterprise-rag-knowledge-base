# Enterprise RAG Knowledge Base & Orchestration Service

A production-ready Retrieval-Augmented Generation (RAG) backend and frontend service that allows users to upload enterprise documents (.pdf, .docx, .txt, .md) and ask natural-language questions grounded strictly in document contents.

---

## 🏗️ Architecture Overview

```text
Documents (.pdf, .docx, .txt, .md)
         ↓
  Document Loader & Text Cleaner
         ↓
 Recursive Character Chunking
         ↓
  HuggingFace Embedding Generation (all-MiniLM-L6-v2)
         ↓
  FAISS Vector Store (Disk Persisted)
         ↓
  User Query → Similarity Search (Top-K Chunks)
         ↓
  Strict Grounding Prompt + LLM (Gemini / OpenAI / Mock)
         ↓
  FastAPI REST API
         ↓
  Streamlit Web Dashboard
```

---

## ✨ Features

- **Multi-Format Ingestion**: Supports `.pdf`, `.docx`, `.txt`, and `.md` file formats with file extension validation.
- **Context-Aware Chunking**: Configurable `chunk_size` and `chunk_overlap` using recursive character splitting.
- **Dense Vector Search**: Powered by FAISS local vector store with similarity score calculation and local disk persistence.
- **Strict Answer Grounding**: Prompt engineering ensures answers are generated strictly using retrieved document context with fallback when context is insufficient.
- **Source Citation**: Precise document and page-level source references (`employee_handbook.pdf, page 14`).
- **RESTful API & Interactive UI**: Clean FastAPI endpoints with interactive Streamlit dashboard.
- **Latency & Performance Tracking**: Empirical breakdown of total latency, retrieval latency, and LLM inference time.
- **Dockerized**: Full Docker & Docker Compose setup for containerized deployment.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.11+, FastAPI, Uvicorn, Pydantic, Pydantic-Settings
- **RAG & Vector Store**: LangChain, FAISS (`faiss-cpu`), Sentence-Transformers (`all-MiniLM-L6-v2`)
- **LLM Integrations**: Google Gemini, OpenAI, and Mock LLM for offline execution
- **Frontend**: Streamlit
- **Testing**: pytest, HTTPX / TestClient
- **Containerization**: Docker, Docker Compose

---

## 📁 Project Structure

```text
Enterprise RAG Knowledge Base & Orchestration Service/
├── app/
│   ├── main.py                  # FastAPI application entrypoint
│   ├── api/
│   │   ├── health.py            # Health check endpoint
│   │   ├── documents.py         # Document upload, ingestion, listing APIs
│   │   └── query.py             # RAG query execution API
│   ├── core/
│   │   ├── config.py            # Pydantic Settings configuration
│   │   └── logging.py           # Structured logging setup
│   ├── loaders/
│   │   ├── document_loader.py   # Multi-format document parsing
│   │   ├── text_cleaner.py      # Control character & space normalization
│   │   └── chunker.py           # Recursive text chunking
│   ├── models/
│   │   ├── document.py          # Document & Chunk Pydantic schemas
│   │   └── rag.py               # RAG response & citation models
│   ├── rag/
│   │   ├── embeddings.py        # HuggingFace embedding manager
│   │   ├── vector_store.py      # FAISS vector store manager
│   │   ├── retriever.py         # RAG retrieval & context formatter
│   │   ├── llm.py               # LLM provider wrapper (Gemini/OpenAI/Mock)
│   │   ├── prompts.py           # Grounding prompt template
│   │   └── pipeline.py          # End-to-end RAG pipeline
│   └── services/
│       └── document_service.py  # Ingestion & document management service
├── streamlit/
│   └── app.py                   # Streamlit web application
├── data/
│   ├── documents/               # Uploaded raw documents storage
│   └── vectorstore/             # FAISS index persistence directory
├── tests/
│   ├── test_health.py           # Health endpoint unit test
│   ├── test_document_processing.py # Ingestion & chunking unit tests
│   ├── test_vector_store.py     # FAISS vector search tests
│   ├── test_retrieval.py        # Context retrieval tests
│   ├── test_rag_pipeline.py     # Pipeline & LLM generation tests
│   ├── test_api.py              # FastAPI integration tests
│   └── test_performance.py      # Latency & throughput benchmark
├── Dockerfile                   # Service Docker build file
├── docker-compose.yml           # Multi-container orchestration
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
└── README.md                    # Project documentation
```

---

## 🚀 Running Locally

### 1. Environment Setup

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Parameters inside `.env`:
```ini
APP_NAME="Enterprise RAG Knowledge Base"
APP_ENV=development
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

LLM_PROVIDER=google
LLM_API_KEY=your_api_key_here
LLM_MODEL=gemini-2.5-flash

EMBEDDING_PROVIDER=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2

VECTOR_DB_PATH=data/vectorstore
DOCUMENTS_PATH=data/documents
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=4
```

### 3. Run FastAPI Backend

```bash
uvicorn app.main:app --reload --port 8000
```
Swagger UI documentation is available at `http://localhost:8000/docs`.

### 4. Run Streamlit UI

In a new terminal:
```bash
streamlit run streamlit/app.py
```
Access the web dashboard at `http://localhost:8501`.

---

## 🐳 Running with Docker

Run backend and frontend services using Docker Compose:

```bash
docker compose up --build
```

- Backend API: `http://localhost:8000`
- Streamlit UI: `http://localhost:8501`

---

## 🧪 Testing

Run the full pytest suite:

```bash
pytest -s -v
```

All 19 test cases verify API endpoints, document parsing, FAISS vector indexing, grounding fallbacks, and performance metrics.

---

## 📊 Measured Performance

Empirical metrics measured on Windows test environment:
- **Vector Retrieval Latency**: `~7.15 ms`
- **Document Ingestion Speed**: `~195.64 ms` per document (10 chunks)
- **Full RAG Pipeline Query Latency (Mock Mode)**: `~8.19 ms`
