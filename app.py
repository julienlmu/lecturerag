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
    layout="wide",
)

# --- Title and intro ---
st.title("LectureRAG")
st.caption("Ask questions about your lecture notes and get answers with cited sources.")
st.caption(
    "First query takes ~30 seconds while the embedding model loads. "
    "Subsequent queries are fast."
)


# --- Helper to load list of courses from Qdrant ---
@st.cache_data(ttl=60)
def get_courses() -> list[str]:
    """Fetch the unique courses present in Qdrant."""
    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
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

EXAMPLE_QUESTIONS = [
    "Was ist das TCP Drei-Wege-Handshake?",
    "Was ist eine Zufallsvariable?",
    "Welche OSI-Schichten gibt es?",
]

# Initialize the input field's session state (only once)
if "question" not in st.session_state:
    st.session_state["question"] = ""


# Callback that fills the input when a button is clicked
def set_question(text: str):
    st.session_state["question"] = text


st.caption("Try one of these examples, or ask your own:")
example_cols = st.columns(3)
for col, example in zip(example_cols, EXAMPLE_QUESTIONS):
    col.button(
        example,
        use_container_width=True,
        on_click=set_question,
        args=(example,),
    )

question = st.text_input(
    "Your question",
    placeholder="e.g. Was ist das TCP Drei-Wege-Handshake?",
    key="question",
)

# --- Handle the Ask button click ---
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

        # 4. Call Gemini (with graceful error handling)
        gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        try:
            response = gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            answer = response.text
        except Exception as e:
            error_type = type(e).__name__
            if "ServerError" in error_type or "503" in str(e) or "500" in str(e):
                msg = (
                    "The language model is temporarily unavailable "
                    "(likely a rate limit on the free tier). Please wait ~30 seconds and try again."
                )
            elif "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                msg = (
                    "Rate limit reached on the free tier. "
                    "Please wait a minute before asking another question."
                )
            else:
                msg = f"Something went wrong calling the language model: {error_type}."
            st.warning(msg)
            st.stop()

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