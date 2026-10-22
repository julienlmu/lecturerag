"""
Query the indexed lecture notes.
Usage: python query.py "What is the TCP three-way handshake?"
"""
import os
import re
import sys
from functools import lru_cache
from io import BytesIO

import requests
from pypdf import PdfReader
from dotenv import load_dotenv

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from groq import Groq

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="fastembed")
warnings.filterwarnings("ignore", message=".*mean pooling.*")  # known, checked: ingest and query use the same version

load_dotenv()

COLLECTION_NAME = "lecture_chunks"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5  # how many chunks go into the prompt
RERANK_MODEL = "jinaai/jina-reranker-v2-base-multilingual"  # German + English
RERANK_CANDIDATES = 20  # how many chunks the vector search hands to the reranker
# Below this reranker score for the best chunk, we answer "not found" without calling the LLM.
# Chosen on eval/testset.csv: answerable questions scored >= -0.66, off-topic ones mostly below -1.0.
MIN_RERANK_SCORE = -1.0
NOT_FOUND_MESSAGE = "I couldn't find this in your notes."

# Small-to-big: we search on small chunks, but give the LLM the whole page each chunk came from,
# so definitions split across two chunks still arrive complete.
MAX_PAGE_CHARS = 3000  # cap per page (long script pages), keeps tokens per question predictable

# LLM that writes the answer (Groq free tier: 30 req/min, 1K req/day, 8K tokens/min, 200K tokens/day).
LLM_MODEL = "openai/gpt-oss-120b"
LLM_REASONING_EFFORT = "low"  # gpt-oss is a reasoning model; summarising 5 chunks needs little thinking

_embedder: TextEmbedding | None = None
_reranker: TextCrossEncoder | None = None


def get_embedder() -> TextEmbedding:
    """Load the embedding model once per process instead of on every question."""
    global _embedder
    if _embedder is None:
        _embedder = TextEmbedding(model_name=EMBED_MODEL)
    return _embedder


def embed_query(question: str) -> list[float]:
    """Embed a single question. Returns one vector."""
    vectors = list(get_embedder().embed([question]))
    return vectors[0].tolist()


def search_chunks(
    client: QdrantClient,
    query_vector: list[float],
    course: str | None = None,
    limit: int = TOP_K,
) -> list[dict]:
    """
    Search Qdrant for the most similar chunks.
    If course is given, only return chunks from that course.
    Returns a list of dicts with keys: course, filename, page, text, score.
    """
    query_filter = None
    if course:
        query_filter = Filter(
            must=[FieldCondition(key="course", match=MatchValue(value=course))]
        )

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
    )
    
    sources = []
    for point in result.points:
        sources.append({
            "course": point.payload["course"],
            "filename": point.payload["filename"],
            "page": point.payload["page"],
            "text": point.payload["text"],
            "pdf_url": point.payload.get("pdf_url", ""),
            "score": point.score,
        })
    return sources


def get_reranker() -> TextCrossEncoder:
    """Load the reranker once per process (it is ~1 GB)."""
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(model_name=RERANK_MODEL)
    return _reranker


def rerank(question: str, sources: list[dict], top_k: int = TOP_K) -> list[dict]:
    """
    Re-score candidates with a cross-encoder, which reads question and chunk together.
    Slower than vector search, so it only runs on the top RERANK_CANDIDATES chunks.
    Adds a "rerank_score" key and returns the top_k best.
    """
    if not sources:
        return sources
    scores = list(get_reranker().rerank(question, [s["text"] for s in sources]))
    for src, score in zip(sources, scores):
        src["rerank_score"] = float(score)
    return sorted(sources, key=lambda s: s["rerank_score"], reverse=True)[:top_k]


def unique_pages(ranked: list[dict], max_pages: int = TOP_K) -> list[dict]:
    """
    Keep only the best-ranked chunk per (file, page), in rank order, up to max_pages.
    Two chunks from the same page would otherwise become two sources with duplicate content.
    """
    seen, pages = set(), []
    for src in ranked:
        key = (src["filename"], src["page"])
        if key not in seen:
            seen.add(key)
            pages.append(src)
        if len(pages) == max_pages:
            break
    return pages


@lru_cache(maxsize=32)
def _load_pdf(url: str) -> PdfReader | None:
    """Download a PDF once per process and keep the parsed reader in memory."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return PdfReader(BytesIO(response.content))
    except Exception as e:
        print(f"Could not load PDF {url}: {type(e).__name__}: {e}")
        return None


def _page_window(page_text: str, chunk_text: str, max_chars: int = MAX_PAGE_CHARS) -> str:
    """Return the whole page, or a max_chars window centred on the chunk if the page is long."""
    if len(page_text) <= max_chars:
        return page_text
    idx = page_text.find(chunk_text[:100])
    if idx == -1:
        return page_text[:max_chars]
    start = max(0, min(idx + len(chunk_text) // 2 - max_chars // 2, len(page_text) - max_chars))
    return page_text[start:start + max_chars]


def attach_page_texts(sources: list[dict]) -> list[dict]:
    """
    Add "page_text" to every source: the full text of its PDF page (same pypdf extraction as ingest).
    Falls back to the chunk text if the PDF cannot be loaded.
    """
    for src in sources:
        reader = _load_pdf(src["pdf_url"]) if src.get("pdf_url") else None
        text = ""
        if reader is not None and 0 < src["page"] <= len(reader.pages):
            text = reader.pages[src["page"] - 1].extract_text() or ""
        src["page_text"] = _page_window(text, src["text"]) if text.strip() else src["text"]
    return sources


def retrieve(client: QdrantClient, question: str, course: str | None = None) -> list[dict]:
    """
    Vector search (top RERANK_CANDIDATES chunks) -> rerank all of them -> best chunk per page, top TOP_K pages.
    Returns sources sorted by rerank score; page texts are attached separately, only if we answer.
    """
    query_vector = embed_query(question)
    candidates = search_chunks(client, query_vector, course=course, limit=RERANK_CANDIDATES)
    ranked = rerank(question, candidates, top_k=len(candidates))
    return unique_pages(ranked)


def is_answerable(sources: list[dict], min_score: float = MIN_RERANK_SCORE) -> bool:
    """True if the best reranked chunk is relevant enough to answer from."""
    return bool(sources) and sources[0].get("rerank_score", float("inf")) >= min_score


def build_prompt(question: str, sources: list[dict]) -> str:
    """Build a prompt that includes the question and numbered sources."""
    
    if not sources:
        return f"""You are a helpful tutor. The user asked a question, but no relevant content was found in their lecture notes.

Respond with exactly: "I couldn't find this in your notes."
Do not provide any other information.

Question: {question}"""
    
    instructions = """You are a helpful tutor answering questions about university lecture notes.

STRICT RULES:
- Answer using ONLY the sources below.
- Cite every claim inline as [1], [2], etc., matching the source numbers.
- If the sources don't contain enough information to fully answer the question, say so explicitly and only include what IS supported.
- Do NOT use your general knowledge to fill in gaps.
- If no sources are relevant, say "I couldn't find this in your notes."
- Answer in the same language as the question.

Sources:
"""
    
    source_lines = []
    for i, src in enumerate(sources, 1):
        text = src.get("page_text", src["text"])
        source_lines.append(f"[{i}] (page {src['page']}, {src['filename']}): {text}")
    sources_block = "\n".join(source_lines)
    
    return instructions + sources_block + f"\n\nQuestion: {question}"


def clean_answer(text: str) -> str:
    """Normalise model-specific formatting so the UI renders it correctly."""
    text = re.sub(r"【(\d+)[^】]*】", r"[\1]", text)              # 【1】 / 【1†L3】 -> [1]
    text = re.sub(r"\\\((.+?)\\\)", r"$\1$", text)                # \( x \) -> $x$
    text = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", text, flags=re.S)  # \[ x \] -> $$x$$
    return text


def generate(prompt: str) -> str:
    """Send the prompt to the LLM and return the answer text. The only place that talks to the LLM provider."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # deterministic answers, better for evaluation
        reasoning_effort=LLM_REASONING_EFFORT,
    )
    return clean_answer(response.choices[0].message.content or "")


def answer_question(question: str, course: str | None = None) -> tuple[str, list[dict]]:
    """
    Full pipeline: retrieve and rerank, threshold, expand to full pages, build prompt, call LLM.
    Returns (answer_text, sources_used).
    """
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
    sources = retrieve(qdrant, question, course=course)

    if not is_answerable(sources):
        return NOT_FOUND_MESSAGE, []

    attach_page_texts(sources)
    prompt = build_prompt(question, sources)
    return generate(prompt), sources


def main(question: str, course: str | None = None):
    if course:
        print(f"Q: {question}  (filter: course={course})\n")
    else:
        print(f"Q: {question}\n")
    
    answer, sources = answer_question(question, course=course)
    print("A:", answer)
    print("\nSources:")
    for i, src in enumerate(sources, 1):
        preview = src["text"][:120].replace("\n", " ")
        print(f"  [{i}] [{src['course']}] Page {src['page']} of {src['filename']} (score: {src['score']:.3f})")
        print(f"      {preview}...")


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print('Usage: python query.py "your question here" [COURSE]')
        print('Example: python query.py "Was ist TCP?" Rechnernetze')
        sys.exit(1)
    
    question = sys.argv[1]
    course = sys.argv[2] if len(sys.argv) == 3 else None
    main(question, course)