"""
Query the indexed lecture notes.
Usage: python query.py "What is the TCP three-way handshake?"
"""
import os
import sys
from dotenv import load_dotenv

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from google import genai

load_dotenv()

COLLECTION_NAME = "lecture_chunks"
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 5  # how many chunks to retrieve


def embed_query(question: str) -> list[float]:
    """Embed a single question. Returns one vector."""
    embedder= TextEmbedding(model_name=EMBED_MODEL)
    vectors = list(embedder.embed([question]))
    return vectors[0].tolist()


def search_chunks(client: QdrantClient, query_vector: list[float]) -> list[dict]:
    """
    Search Qdrant for the most similar chunks.
    Returns a list of dicts with keys: filename, page, text, score.
    """
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=TOP_K,
    )
    sources = []
    for point in result.points: 
        sources.append({
            "filename":point.payload["filename"],
            "page":point.payload["page"],
            "text":point.payload["text"],
            "score":point.score,  
        })
    return sources


def build_prompt(question: str, sources: list[dict]) -> str:
    """Build a prompt that includes the question and numbered sources."""
    prompt = """You are a helpful tutor answering questions about university lecture notes. Answer using ONLY the sources below. Cite sources inline as [1], [2], etc. If the answer is not in the sources, say "I couldn't find this in your notes." Be concise and accurate. Answer in the same language as the question.

Sources:
"""
    
    source_lines = []
    for i, src in enumerate(sources, 1):
        source_lines.append(f"[{i}] (page {src['page']}, {src['filename']}): {src['text']}")
    sources_block = "\n".join(source_lines)
    
    return prompt + sources_block + f"\n\nQuestion: {question}"


def answer_question(question: str) -> tuple[str, list[dict]]:
    """
    Full pipeline: embed question, search, build prompt, call LLM, return answer.
    Returns (answer_text, sources_used).
    """
    # 1. Embed the question
    query_vector = embed_query(question)

    # 2. Search Qdrant
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
    sources = search_chunks(qdrant, query_vector)

    # 3. Build the prompt
    prompt = build_prompt(question, sources)

    # 4. Call Gemini
    gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = gemini.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text, sources


def main(question: str):
    print(f"Q: {question}\n")
    answer, sources = answer_question(question)
    print("A:", answer)
    print("\nSources:")
    for i, src in enumerate(sources, 1):
        preview = src["text"][:120].replace("\n", " ")
        print(f"  [{i}] Page {src['page']} of {src['filename']} (score: {src['score']:.3f})")
        print(f"      {preview}...")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)
    main(sys.argv[1])
   