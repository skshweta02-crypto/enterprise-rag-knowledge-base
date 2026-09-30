from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from app.services.document_service import DocumentService

router = APIRouter(prefix="/documents", tags=["Documents"])


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    file_size_bytes: int
    saved_path: str
    message: str


class DocumentIngestRequest(BaseModel):
    saved_path: str
    filename: str
    document_id: str


class DocumentIngestResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    total_chunks: int
    status: str


class DocumentListResponse(BaseModel):
    total_documents: int
    documents: List[dict]


def get_document_service() -> DocumentService:
    return DocumentService()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
):
    doc_id, saved_path, file_size = await service.save_uploaded_file(file)
    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename or "uploaded_doc",
        file_size_bytes=file_size,
        saved_path=saved_path,
        message="Document uploaded successfully",
    )


@router.post("/ingest", response_model=DocumentIngestResponse)
def ingest_document(
    request: DocumentIngestRequest,
    service: DocumentService = Depends(get_document_service),
):
    processed = service.ingest_file(
        file_path=request.saved_path,
        original_filename=request.filename,
        document_id=request.document_id,
    )
    return DocumentIngestResponse(
        document_id=processed.document_id,
        filename=processed.filename,
        file_type=processed.file_type,
        total_chunks=processed.total_chunks,
        status="ingested",
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(service: DocumentService = Depends(get_document_service)):
    docs = service.list_documents()
    return DocumentListResponse(
        total_documents=len(docs),
        documents=docs,
    )
