import os
import uuid
from typing import List, Dict, Any, Tuple, Optional
from pypdf import PdfReader
from docx import Document as DocxDocument
from app.loaders.text_cleaner import clean_text
from app.core.logging import logger

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def is_supported_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def extract_pages_from_pdf(file_path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(file_path)
    pages = []
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        cleaned = clean_text(text)
        if cleaned:
            pages.append((idx + 1, cleaned))
    return pages


def extract_text_from_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text:
            full_text.append(para.text)
    return clean_text("\n".join(full_text))


def extract_text_from_txt_or_md(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    return clean_text(content)


def load_document(file_path: str, filename: Optional[str] = None) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    actual_filename = filename or os.path.basename(file_path)
    ext = os.path.splitext(actual_filename)[1].lower()

    if not is_supported_file(actual_filename):
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: {SUPPORTED_EXTENSIONS}")

    logger.info(f"Extracting text from document '{actual_filename}' ({ext})")

    # Returns list of page/section dicts with content and metadata
    raw_sections = []

    if ext == ".pdf":
        pdf_pages = extract_pages_from_pdf(file_path)
        for page_num, text in pdf_pages:
            raw_sections.append({
                "content": text,
                "page_number": page_num
            })
    elif ext == ".docx":
        text = extract_text_from_docx(file_path)
        if text:
            raw_sections.append({"content": text, "page_number": None})
    elif ext in (".txt", ".md"):
        text = extract_text_from_txt_or_md(file_path)
        if text:
            raw_sections.append({"content": text, "page_number": None})

    return raw_sections
