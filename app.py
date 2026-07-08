"""
LectureRAG - Streamlit UI
Run with: streamlit run app.py
"""
import os
import requests
from streamlit_pdf_viewer import pdf_viewer
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

# --- Custom CSS for spacing and hero ---
st.markdown("""
<style>

/* Override focus ring on inputs to match accent color */
    .stTextInput > div > div > input:focus,
    .stTextInput > div[data-baseweb="input"]:focus-within {
        border-color: #7CB7FF !important;
        box-shadow: 0 0 0 1px #7CB7FF !important;
    }
    .stTextInput > div[data-baseweb="input"] {
        border-color: #2a2f3a !important;
    }

    /* Override focus ring on selectbox */
    .stSelectbox > div[data-baseweb="select"]:focus-within > div {
        border-color: #7CB7FF !important;
        box-shadow: 0 0 0 1px #7CB7FF !important;
    }
    .stSelectbox > div[data-baseweb="select"] > div {
        border-color: #2a2f3a !important;
    }
    }
    .stSelectbox > div[data-baseweb="select"] > div {
        border-color: #2a2f3a;
    }

    /* Tighten the top padding */
    .block-container {
        padding-top: 3rem;
        max-width: 920px;
    }

    /* Hero header */
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #9aa0a6;
        margin-bottom: 0.5rem;
        font-weight: 400;
    }
    .hero-meta {
        font-size: 0.85rem;
        color: #6c7280;
        margin-bottom: 2rem;
    }
    .hero-accent {
        width: 48px;
        height: 3px;
        background: linear-gradient(90deg, #7CB7FF 0%, #5B8BD9 100%);
        border-radius: 2px;
        margin: 0.5rem 0 1rem 0;
    }

    /* Section labels */
    .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9aa0a6;
        margin: 1.5rem 0 0.5rem 0;
    }

    /* Tighter sidebar */
    [data-testid="stSidebar"] .stMarkdown h2 {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9aa0a6;
        margin-bottom: 0.5rem;
    }
    [data-testid="stSidebar"] {
        padding-top: 2rem;
    }

    /* Example buttons */
    .stButton > button {
        background-color: transparent;
        border: 1px solid #2a2f3a;
        border-radius: 8px;
        color: #c0c4cc;
        font-weight: 400;
        font-size: 0.9rem;
        padding: 0.6rem 0.9rem;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        border-color: #7CB7FF;
        color: #ffffff;
        background-color: rgba(124, 183, 255, 0.05);
    }
    /* Primary (Ask) button override */
    .stButton > button[kind="primary"] {
        background-color: #7CB7FF;
        color: #0E1117;
        border: none;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #5B8BD9;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


# --- Hero header ---
st.markdown("""
<div class="hero-title">📚 LectureRAG</div>
<div class="hero-accent"></div>
<div class="hero-subtitle">Ask questions about your lecture notes — get answers grounded in cited sources.</div>
<div class="hero-meta">First query takes ~30s while the embedding model loads · Built with Python, Qdrant, fastembed, Gemini</div>
""", unsafe_allow_html=True)


# --- Helper to load list of courses from Qdrant ---
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pdf(url: str) -> bytes | None:
    """Download a PDF from a URL and return its bytes. Cached for 1 hour."""
    if not url:
        return None
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.content
    except Exception:
        return None

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


# --- Sidebar ---
with st.sidebar:
    st.markdown("## Filters")
    available_courses = get_courses()
    options = ["(all courses)"] + available_courses
    selected = st.selectbox(
        "Course",
        options,
        index=0,
        label_visibility="collapsed",
    )
    course_filter = None if selected == "(all courses)" else selected
    st.caption(f"{len(available_courses)} course(s) available")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("## About")
    st.caption(
        "A retrieval-augmented Q&A app for university lecture notes. "
        "Built with Python, Qdrant, fastembed, and Google Gemini."
    )
    st.markdown(
        "[View on GitHub →](https://github.com/julienlmu/lecturerag)",
        unsafe_allow_html=True,
    )


# --- Example questions ---
EXAMPLE_QUESTIONS = [
    "Was ist der TCP Drei-Wege-Handshake?",
    "Was ist eine Zufallsvariable?",
    "Welche OSI-Schichten gibt es?",
]

if "question" not in st.session_state:
    st.session_state["question"] = ""


def set_question(text: str):
    st.session_state["question"] = text


st.markdown('<div class="section-label">Examples</div>', unsafe_allow_html=True)
example_cols = st.columns(3)
for col, example in zip(example_cols, EXAMPLE_QUESTIONS):
    col.button(
        example,
        use_container_width=True,
        on_click=set_question,
        args=(example,),
    )


# --- Question input ---
st.markdown('<div class="section-label">Your question</div>', unsafe_allow_html=True)
question = st.text_input(
    "Your question",
    placeholder="Type a question about your lecture notes...",
    key="question",
    label_visibility="collapsed",
)

ask_clicked = st.button("Ask", type="primary", use_container_width=False)


# --- Handle the Ask button click ---
if ask_clicked and question:
    with st.spinner("Searching your notes..."):
        query_vector = embed_query(question)

        qdrant = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
        sources = search_chunks(qdrant, query_vector, course=course_filter)

        prompt = build_prompt(question, sources)

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

    st.markdown('<div class="section-label">Answer</div>', unsafe_allow_html=True)
    st.write(answer)

    if sources:
        st.markdown('<div class="section-label">Sources</div>', unsafe_allow_html=True)
        for i, src in enumerate(sources, 1):
            with st.expander(
                f"[{i}]  {src['filename']} · page {src['page']} · "
                f"{src['course']} · score {src['score']:.3f}"
            ):
                st.write(src["text"])

                # Show the actual PDF page
                pdf_url = src.get("pdf_url", "")
                if pdf_url:
                    with st.spinner("Loading page..."):
                        pdf_bytes = fetch_pdf(pdf_url)
                    if pdf_bytes:
                        pdf_viewer(
                            pdf_bytes,
                            pages_to_render=[src["page"]],
                            height=600,
                            key=f"pdf_{i}",
                        )
                    else:
                        st.caption("⚠️ Could not load the PDF for this source.")
                else:
                    st.caption("No PDF available for this source.")