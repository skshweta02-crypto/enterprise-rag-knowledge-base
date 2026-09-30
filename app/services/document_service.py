import os
import shutil
import uuid
from typing import List, Dict, Any, Optional, Tuple
from fastapi import UploadFile, HTTPException
from app.core.config import settings
from app.core.logging import logger
from app.loaders.document_loader import is_supported_file
from app.loaders.chunker import chunk_document
from app.rag.vector_store import VectorStoreManager
from app.models.document import ProcessedDocument


class DocumentService:
    def __init__(self, vector_store_manager: Optional[VectorStoreManager] = None):
        self.upload_dir = settings.DOCUMENTS_PATH
        os.makedirs(self.upload_dir, exist_ok=True)
        self.vector_store_manager = vector_store_manager or VectorStoreManager()

    async def save_uploaded_file(self, file: UploadFile) -> Tuple[str, str, int]:
        filename = file.filename or "uploaded_doc"
        if not is_supported_file(filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported extensions: .pdf, .docx, .txt, .md",
            )

        doc_id = str(uuid.uuid4())
        ext = os.path.splitext(filename)[1].lower()
        saved_filename = f"{doc_id}_{filename}"
        saved_path = os.path.join(self.upload_dir, saved_filename)

        try:
            with open(saved_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_size = os.path.getsize(saved_path)
            logger.info(f"Saved uploaded file '{filename}' to '{saved_path}' ({file_size} bytes)")
            return doc_id, saved_path, file_size
        except Exception as e:
            logger.error(f"Failed to save upload file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save uploaded document")

    def ingest_file(self, file_path: str, original_filename: str, document_id: str) -> ProcessedDocument:
        try:
            processed_doc = chunk_document(
                file_path=file_path,
                filename=original_filename,
                document_id=document_id,
            )
            self.vector_store_manager.add_chunks(processed_doc.chunks)
            logger.info(f"Ingested document '{original_filename}' with {processed_doc.total_chunks} chunks")
            return processed_doc
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    def list_documents(self) -> List[Dict[str, Any]]:
        documents = []
        if not os.path.exists(self.upload_dir):
            return []

        for f in os.listdir(self.upload_dir):
            if f == ".gitkeep":
                continue
            full_path = os.path.join(self.upload_dir, f)
            if os.path.isfile(full_path):
                file_size = os.path.getsize(full_path)
                documents.append(
                    {
                        "filename": f,
                        "file_size_bytes": file_size,
                    }
                )
        return documents
