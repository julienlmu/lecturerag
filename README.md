# LectureRAG

A retrieval-augmented Q&A app for university lecture notes. Ask a question in natural language and get an answer grounded in your own PDFs, with inline citations and the exact source page shown alongside each answer.

**Live demo:** https://lecturerag-eptovhywnagwcyaguvnkx3.streamlit.app

## What it does

- Ask questions about your lecture notes in natural language (German or English).
- Answers are generated only from the retrieved source material, with inline citations like `[1]`, `[2]`.
- When the notes do not contain an answer, the app says so instead of making one up.
- Filter queries by course, or search across all courses at once.
- Each cited source can be expanded to show the retrieved text and the actual PDF page it came from.

## How it works

The system has two pipelines: an offline ingestion pipeline and an online query pipeline.

**Ingestion** (`ingest.py`):
1. A PDF is read page by page with `pypdf`, preserving page numbers.
2. Each page's text is split into roughly 500-character chunks (50-character overlap) using `RecursiveCharacterTextSplitter`, which prefers to break at natural boundaries.
3. Each chunk is embedded locally with `paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional vectors, chosen for its German-language support since the lecture notes are mostly German).
4. Chunks are stored in Qdrant with metadata: course, filename, page, text, and a link to the source PDF in cloud storage.

**Query** (`query.py`, `app.py`):
1. The user's question is embedded with the same model.
2. The 20 most similar chunks are retrieved from Qdrant, optionally filtered to a specific course using an indexed payload field.
3. A multilingual cross-encoder (`jina-reranker-v2-base-multilingual`) reads the question and each candidate chunk together and re-sorts them. The top 5 are kept.
4. If even the best chunk scores below a minimum reranker score, the app answers "I couldn't find this in your notes." directly, without calling the LLM.
5. Otherwise the top 5 chunks are assembled into a prompt with strict instructions to answer only from the sources and cite them by number.
6. Google Gemini generates the answer, which is displayed with expandable source cards. Each card can show the original PDF page, fetched from cloud storage.

## Evaluation

Retrieval quality is measured with a hand-labeled test set (`eval/testset.csv`): 47 questions in German and English, 39 answerable across three courses and 8 that the notes cannot answer (off-topic and near-miss topics). For each answerable question, every page that contains the answer is labeled, since the same content often appears in slides, lecture script and formula sheet.

`eval/eval_retrieval.py` checks whether a correct page is among the retrieved chunks, without calling the LLM, so a run is cheap and deterministic.

| | Vector search | + Reranking |
|---|---|---|
| Hit@1 | 41.0% | 79.5% |
| Hit@5 (what the LLM sees) | 69.2% | **94.9%** |
| MRR@10 | 0.545 | **0.861** |

- The baseline's main failure was ordering: in most misses the correct page was retrieved, but on rank 6 to 10. Reranking the top 20 fixes most of these.
- The remaining misses are formula-heavy slides, where `pypdf` text extraction loses most of the content.
- The minimum reranker score (−1.0) refuses <!-- X --> of 8 unanswerable questions and none of the 39 answerable ones. Reranker scores separate the two groups far better than cosine similarity, which made a reliable threshold possible.

The test set is small (13 questions per course), so per-course and per-language numbers are indicative only.

```bash
python eval/eval_retrieval.py --tag baseline          # vector search only
python eval/eval_retrieval.py --rerank --tag rerank   # with reranking + threshold table
python eval/eval_retrieval.py --list-files            # indexed files per course
```

## Tech stack

- **Language:** Python 3.13
- **Embeddings:** [fastembed](https://github.com/qdrant/fastembed) with `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, runs locally, no API cost)
- **Reranking:** fastembed cross-encoder with `jinaai/jina-reranker-v2-base-multilingual` (runs locally)
- **Vector database:** [Qdrant Cloud](https://qdrant.tech/) with payload indexing for course filtering
- **LLM:** [Google Gemini 2.5 Flash](https://aistudio.google.com/)
- **PDF storage:** [Supabase Storage](https://supabase.com/) (public bucket, serves original PDFs to the viewer)
- **UI and deployment:** [Streamlit](https://streamlit.io/), deployed on Streamlit Community Cloud
- **PDF viewer:** [streamlit-pdf-viewer](https://github.com/lfoppiano/streamlit-pdf-viewer)

## Features

- Semantic retrieval with cross-encoder reranking across a multi-course corpus (93 PDFs from four courses).
- Multi-course filtering backed by Qdrant payload indexes.
- Grounded answers with inline citations and honest "not found in your notes" behavior, enforced by a reranker score threshold and the prompt.
- Retrieval evaluation with a labeled test set, run before and after every change.
- Inline PDF viewer: click a source to see the exact cited page rendered in the app.
- Graceful handling of LLM rate limits and transient errors.
- Minimal UI with example questions.

## Running locally

```bash
git clone https://github.com/julienlmu/lecturerag.git
cd lecturerag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then fill in your keys
```

Required environment variables (see `.env.example`):

```
GEMINI_API_KEY=          # from https://aistudio.google.com/apikey
QDRANT_URL=              # your Qdrant Cloud cluster URL
QDRANT_API_KEY=          # your Qdrant Cloud API key
SUPABASE_URL=            # your Supabase project URL
SUPABASE_SERVICE_KEY=    # your Supabase secret key (local use only)
```

Ingest some PDFs and run the app:

```bash
# Upload PDFs to Supabase (for the in-app viewer)
python upload_pdfs.py

# Ingest a PDF into the vector database (course name, then path)
python ingest.py Rechnernetze pdfs/Rechnernetze/Kapitel_3_Transport.pdf

# Launch the UI
streamlit run app.py
```

The first query downloads the embedding model (~0.25 GB) and the reranker (~1.1 GB) from Hugging Face. `fastembed` is pinned in `requirements.txt` so that ingestion and queries always use the same embedding behavior.

## Project structure

```
lecturerag/
├── app.py                  # Streamlit UI, query orchestration, PDF viewer
├── query.py                # embedding, retrieval, reranking, threshold, prompt building
├── ingest.py               # PDF -> chunks -> embeddings -> Qdrant
├── upload_pdfs.py          # bulk-upload local PDFs to Supabase Storage
├── eval/
│   ├── testset.csv         # labeled questions with correct file/page per answer
│   └── eval_retrieval.py   # Hit@k, MRR, breakdowns, threshold table
├── requirements.txt
├── .env.example
└── .streamlit/
    └── config.toml         # theme
```

## License

MIT — see [LICENSE](LICENSE).