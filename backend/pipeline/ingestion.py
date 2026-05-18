"""
pipeline/ingestion.py — PDF Loading and Chunking
=================================================
Stage 1 of the RAG pipeline.

Strategy: RecursiveCharacterTextSplitter
- chunk_size=512, overlap=64
- Tries to split on [paragraph, sentence, word] boundaries in that order
- Preserves semantic coherence while keeping chunks small enough for accurate retrieval
- Each chunk carries metadata: source_file, page_number, chunk_index
"""

import os
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def load_and_chunk_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a PDF and split into chunks with rich metadata.

    Returns:
        List of dicts with keys: content, metadata (source_file, page, chunk_index)
    """
    path = Path(file_path)
    loader = PyPDFLoader(str(path))
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(pages)

    results = []
    for idx, chunk in enumerate(chunks):
        results.append({
            "content": chunk.page_content.strip(),
            "metadata": {
                "source_file": path.name,
                "source_path": str(path),
                "page": chunk.metadata.get("page", 0) + 1,  # 1-indexed
                "chunk_index": idx,
                "total_chunks": len(chunks),
            },
        })

    print(f"[Ingestion] {path.name}: {len(pages)} pages -> {len(results)} chunks")
    return results


def ingest_files(file_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Ingest multiple PDF files, returning all chunks with file-level metadata.
    """
    all_chunks = []
    for fp in file_paths:
        try:
            chunks = load_and_chunk_pdf(fp)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[Ingestion] ERROR processing {fp}: {e}")
    print(f"[Ingestion] Total chunks from {len(file_paths)} file(s): {len(all_chunks)}")
    return all_chunks
