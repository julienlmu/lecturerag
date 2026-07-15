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
2. The top 5 most similar chunks are retrieved from Qdrant, optionally filtered to a specific course using an indexed payload field.
3. The retrieved chunks are assembled into a prompt with strict instructions to answer only from the sources and cite them by number.
4. Google Gemini generates the answer, which is displayed with expandable source cards. Each card can show the original PDF page, fetched from cloud storage.

## Tech stack

- **Language:** Python 3.13
- **Embeddings:** [fastembed](https://github.com/qdrant/fastembed) with `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions, runs locally, no API cost)
- **Vector database:** [Qdrant Cloud](https://qdrant.tech/) with payload indexing for course filtering
- **LLM:** [Google Gemini 2.5 Flash](https://aistudio.google.com/)
- **PDF storage:** [Supabase Storage](https://supabase.com/) (public bucket, serves original PDFs to the viewer)
- **UI and deployment:** [Streamlit](https://streamlit.io/), deployed on Streamlit Community Cloud
- **PDF viewer:** [streamlit-pdf-viewer](https://github.com/lfoppiano/streamlit-pdf-viewer)

## Features

- Semantic retrieval across a multi-course corpus (currently ~2,900 chunks from four courses).
- Multi-course filtering backed by Qdrant payload indexes.
- Grounded answers with inline citations and honest "not found in your notes" behavior.
- Inline PDF viewer: click a source to see the exact cited page rendered in the app.
- Graceful handling of LLM rate limits and transient errors.
- Clean, minimal UI with example questions.

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

## Project structure

```
lecturerag/
├── app.py            # Streamlit UI, query orchestration, PDF viewer
├── query.py          # embedding, retrieval, prompt building
├── ingest.py         # PDF -> chunks -> embeddings -> Qdrant
├── upload_pdfs.py    # bulk-upload local PDFs to Supabase Storage
├── requirements.txt
├── .env.example
└── .streamlit/
    └── config.toml   # theme
```


## License

MIT — see [LICENSE](LICENSE).
