"""
LectureRAG - Streamlit UI
Run with: streamlit run app.py
"""
import os
from dotenv import load_dotenv
import streamlit as st
from qdrant_client import QdrantClient

from query import embed_query, search_chunks, build_prompt
from google import genai

load_dotenv()

# --- Page config ---
st.set_page_config(
    page_title="LectureRAG",
    page_icon="📚",
    layout="wide",
)

# --- Title and intro ---
st.title("📚 LectureRAG")
st.caption("Ask questions about your lecture notes and get answers with cited sources.")


# --- Helper to load list of courses from Qdrant ---
@st.cache_data(ttl=60)
def get_courses() -> list[str]:
    """Fetch the unique courses present in Qdrant."""
    import sys
    
    # Debug: show whether secrets are loaded
    url = os.environ.get("QDRANT_URL", "NOT SET")
    key = os.environ.get("QDRANT_API_KEY", "NOT SET")
    st.sidebar.caption(f"URL set: {url != 'NOT SET'} (length: {len(url)})")
    st.sidebar.caption(f"Key set: {key != 'NOT SET'} (length: {len(key)})")
    
    try:
        client = QdrantClient(url=url, api_key=key)
        courses = set()
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name="lecture_chunks",
                limit=200,
                offset=offset,
                with_payload=["course"],
                with_vectors=False,
            )
            for p in points:
                if p.payload and "course" in p.payload:
                    courses.add(p.payload["course"])
            if offset is None:
                break
        return sorted(courses)
    except Exception as e:
        st.error(f"Qdrant error: {type(e).__name__}: {str(e)[:500]}")
        st.stop()

# --- Sidebar: course filter ---
with st.sidebar:
    st.header("Filters")
    available_courses = get_courses()
    options = ["(all courses)"] + available_courses
    selected = st.selectbox("Course", options, index=0)
    course_filter = None if selected == "(all courses)" else selected
    
    st.divider()
    st.markdown("**About**")
    st.caption("Built with Python, Qdrant, fastembed, and Google Gemini.")


# --- Main: question input + answer ---
question = st.text_input(
    "Your question",
    placeholder="e.g. Was ist das TCP Drei-Wege-Handshake?",
)

if st.button("Ask", type="primary") and question:
    with st.spinner("Searching your notes..."):
        # 1. Embed the question
        query_vector = embed_query(question)

        # 2. Search Qdrant
        qdrant = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
        sources = search_chunks(qdrant, query_vector, course=course_filter)

        # 3. Build the prompt
        prompt = build_prompt(question, sources)

        # 4. Call Gemini
        gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        response = gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        answer = response.text

    # --- Display answer ---
    st.markdown("### Answer")
    st.write(answer)

    # --- Display sources ---
    if sources:
        st.markdown("### Sources")
        for i, src in enumerate(sources, 1):
            with st.expander(
                f"[{i}] {src['filename']} — page {src['page']} "
                f"({src['course']}, score {src['score']:.3f})"
            ):
                st.write(src["text"])