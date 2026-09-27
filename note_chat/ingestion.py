"""Load uploaded notes and split them into retrieval-sized chunks."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILES = 5
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
MAX_PDF_PAGES = 120
MAX_DOCUMENT_CHARS = 600_000
MAX_CHUNKS = 750


class UploadedFile(Protocol):
    """Small protocol shared by Streamlit uploads and test doubles."""

    name: str

    def getvalue(self) -> bytes: ...


def fingerprint_files(files: Iterable[UploadedFile]) -> str:
    """Return a stable fingerprint so the UI can detect changed uploads."""
    digest = hashlib.sha256()
    for uploaded_file in sorted(files, key=lambda item: item.name):
        content = uploaded_file.getvalue()
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _load_pdf(filename: str, content: bytes) -> list[Document]:
    """Load a PDF through PyPDFLoader without retaining the uploaded file."""
    suffix = Path(filename).suffix.lower()
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(content)
            temporary_path = temporary_file.name

        page_count = len(PdfReader(temporary_path).pages)
        if page_count > MAX_PDF_PAGES:
            raise ValueError(
                f"{filename} has {page_count} pages; the public demo limit is "
                f"{MAX_PDF_PAGES}."
            )

        documents = PyPDFLoader(temporary_path).load()
        for document in documents:
            document.metadata["source"] = filename
            page = document.metadata.get("page")
            document.metadata["citation"] = (
                f"{filename}, page {page + 1}" if isinstance(page, int) else filename
            )
        return documents
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _load_text(filename: str, content: bytes) -> list[Document]:
    text = content.decode("utf-8-sig", errors="replace").strip()
    if not text:
        return []
    return [
        Document(
            page_content=text,
            metadata={"source": filename, "page": 0, "citation": filename},
        )
    ]


def load_uploaded_files(files: Iterable[UploadedFile]) -> list[Document]:
    """Turn PDF, Markdown, and text uploads into LangChain documents."""
    files = list(files)
    if len(files) > MAX_FILES:
        raise ValueError(f"Upload at most {MAX_FILES} files at a time.")

    total_bytes = sum(len(uploaded_file.getvalue()) for uploaded_file in files)
    if total_bytes > MAX_TOTAL_BYTES:
        raise ValueError("The combined upload is larger than 25 MB.")

    documents: list[Document] = []
    for uploaded_file in files:
        content = uploaded_file.getvalue()
        extension = Path(uploaded_file.name).suffix.lower()

        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {uploaded_file.name}")
        if len(content) > MAX_FILE_BYTES:
            raise ValueError(f"{uploaded_file.name} is larger than 10 MB.")

        if extension == ".pdf":
            documents.extend(_load_pdf(uploaded_file.name, content))
        else:
            documents.extend(_load_text(uploaded_file.name, content))

    readable_documents = [document for document in documents if document.page_content.strip()]
    if not readable_documents:
        raise ValueError(
            "No readable text was found. This may be a scanned or image-only PDF; "
            "run OCR on it before uploading."
        )

    character_count = sum(len(document.page_content) for document in readable_documents)
    if character_count > MAX_DOCUMENT_CHARS:
        raise ValueError(
            "The extracted text is too large for this public demo "
            f"({character_count:,} characters; limit {MAX_DOCUMENT_CHARS:,})."
        )
    return readable_documents


def split_documents(
    documents: list[Document], chunk_size: int = 1_000, chunk_overlap: int = 200
) -> list[Document]:
    """Split while retaining filename, page, and character-offset metadata."""
    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No searchable text chunks could be created.")
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(
            f"These documents create {len(chunks)} chunks; the public demo limit is "
            f"{MAX_CHUNKS}. Upload a smaller document set or increase chunk size."
        )
    return chunks
