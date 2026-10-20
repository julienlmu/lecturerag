"""
Query the indexed lecture notes.
Usage: python query.py "What is the TCP three-way handshake?"
"""
import os
import sys
from dotenv import load_dotenv

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from google import genai

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="fastembed")

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

_reranker: TextCrossEncoder | None = None


def embed_query(question: str) -> list[float]:
    """Embed a single question. Returns one vector."""
    embedder= TextEmbedding(model_name=EMBED_MODEL)
    vectors = list(embedder.embed([question]))
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
        source_lines.append(f"[{i}] (page {src['page']}, {src['filename']}): {src['text']}")
    sources_block = "\n".join(source_lines)
    
    return instructions + sources_block + f"\n\nQuestion: {question}"


def answer_question(question: str, course: str | None = None) -> tuple[str, list[dict]]:
    """
    Full pipeline: embed question, search, build prompt, call LLM, return answer.
    Returns (answer_text, sources_used).
    """
    query_vector = embed_query(question)

    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
    candidates = search_chunks(qdrant, query_vector, course=course, limit=RERANK_CANDIDATES)
    sources = rerank(question, candidates)

    if not is_answerable(sources):
        return NOT_FOUND_MESSAGE, []

    prompt = build_prompt(question, sources)

    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text, sources


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