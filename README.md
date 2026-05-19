# LectureRAG

A Q&A app for asking questions about university lecture notes. Built with Python, Qdrant, and Google Gemini.

🔗 **Live demo:** https://lecturerag-eptovhywnagwcyaguvnkx3.streamlit.app

🚧 Work in progress — first real project to learn RAG (Retrieval-Augmented Generation) end-to-end.

## Status

- [x] PDF ingestion pipeline (chunking, embeddings, vector DB)
- [x] Query and answer generation
- [x] Streamlit UI
- [x] Deployment

## Tech stack

- Python 3.13
- [fastembed](https://github.com/qdrant/fastembed) with `paraphrase-multilingual-MiniLM-L12-v2` for embeddings (384 dims)
- [Qdrant Cloud](https://qdrant.tech/) for vector storage
- [Google Gemini](https://aistudio.google.com/) (planned, for answer generation)
- [Streamlit](https://streamlit.io/) (planned, for the UI)
