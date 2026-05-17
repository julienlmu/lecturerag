"""
Ingest a PDF file into Qdrant.
Usage: python ingest.py pdfs/lecture1.pdf
"""
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

COLLECTION_NAME = "lecture_chunks"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" # 384 dimensions
VECTOR_SIZE = 384


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """
    Read a PDF and return a list of (page_number, text) tuples.
    Page numbers start at 1 (human-readable)."""
    

    pages = []
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        pages.append((i + 1, text))
    return pages


def chunk_pages(pages: list[tuple[int, str]]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    
    """
    Split each page into chunks. Return a list of dicts:
    [{"page": 1, "text": "..."}, {"page": 1, "text": "..."}, ...]
    """
    for page_num, text in pages:
        page_chunks = splitter.split_text(text)
        for chunk_text in page_chunks:
            chunks.append({"page": page_num, "text": chunk_text})
    return chunks
       

def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    """
    Embed all chunk texts. Return a list of embedding vectors.
    """

    embedder = TextEmbedding(model_name=EMBED_MODEL)
    texts = []
    for chunk in chunks:
        texts.append(chunk["text"])
    vectors = [vec.tolist() for vec in embedder.embed(texts)]
    return vectors
    
    


def ensure_collection(client: QdrantClient):
    """Create the Qdrant collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}'")


def upload_to_qdrant(client: QdrantClient, chunks: list[dict], vectors: list[list[float]], filename: str):
    """Upload chunks + vectors to Qdrant with metadata."""
    points = []
    for chunk, vector in zip(chunks, vectors):
        point= PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "filename": filename,
                "page": chunk["page"],
                "text": chunk["text"]
            }
        )
        points.append(point)
    client.upsert(collection_name=COLLECTION_NAME, points=points)


def main(pdf_path: str):
    filename = Path(pdf_path).name
    print(f"Ingesting {filename}...")

    pages = extract_pages(pdf_path)
    print(f"  Extracted {len(pages)} pages with text")

    chunks = chunk_pages(pages)
    print(f"  Created {len(chunks)} chunks")

    vectors = embed_chunks(chunks)
    print(f"  Embedded {len(vectors)} chunks")

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
    ensure_collection(client)
    upload_to_qdrant(client, chunks, vectors, filename)
    print(f"Done! Uploaded {len(chunks)} chunks to Qdrant.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest.py path/to/file.pdf")
        sys.exit(1)
    main(sys.argv[1])