import os
import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_upload_and_ingest_workflow():
    content = b"Enterprise Security Policy: All users must enable multi-factor authentication (MFA)."
    files = {"file": ("security.txt", content, "text/plain")}

    upload_res = client.post("/documents/upload", files=files)
    assert upload_res.status_code == 200
    upload_data = upload_res.json()
    assert upload_data["filename"] == "security.txt"
    assert "saved_path" in upload_data
    assert "document_id" in upload_data

    saved_path = upload_data["saved_path"]
    doc_id = upload_data["document_id"]

    ingest_payload = {
        "saved_path": saved_path,
        "filename": "security.txt",
        "document_id": doc_id,
    }
    ingest_res = client.post("/documents/ingest", json=ingest_payload)
    assert ingest_res.status_code == 200
    ingest_data = ingest_res.json()
    assert ingest_data["status"] == "ingested"
    assert ingest_data["total_chunks"] >= 1

    list_res = client.get("/documents")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total_documents"] >= 1


def test_upload_unsupported_file():
    files = {"file": ("malware.exe", b"binary content", "application/octet-stream")}
    res = client.post("/documents/upload", files=files)
    assert res.status_code == 400
    assert "Unsupported file format" in res.json()["detail"]


def test_query_endpoint():
    query_payload = {
        "question": "What authentication policy must users enable?",
        "top_k": 2,
    }
    res = client.post("/query", json=query_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["question"] == query_payload["question"]
    assert "answer" in data
    assert "sources" in data
    assert "latency_ms" in data
    assert "retrieval_latency_ms" in data
    assert "llm_latency_ms" in data
    assert isinstance(data["retrieved_chunks"], list)


def test_query_validation_error():
    res = client.post("/query", json={"question": ""})
    assert res.status_code == 422  # Pydantic validation error (min_length=1)
