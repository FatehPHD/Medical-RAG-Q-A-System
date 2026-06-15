
import os
import html
import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")


st.set_page_config(
    page_title="Medical RAG Q&A",
    page_icon="",
    layout="wide"
)


st.markdown(
    """
    <style>
        .stApp {
            background-color: #fbf7f0;
            color: #3b2418;
            font-family: Georgia, "Times New Roman", serif;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        .block-container {
            max-width: 1050px;
            padding-top: 3rem;
            padding-bottom: 4rem;
        }

        .top-nav {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #ead8c4;
            padding-bottom: 1rem;
            margin-bottom: 3rem;
        }

        .brand {
            font-size: 1.1rem;
            letter-spacing: 0.14em;
            font-weight: 700;
            color: #5d3a26;
        }

        .nav-links {
            font-size: 0.9rem;
            color: #8a5a3b;
            letter-spacing: 0.05em;
        }

        .hero {
            text-align: center;
            margin-bottom: 3rem;
        }

        .hero h1 {
            font-size: 3rem;
            font-weight: 400;
            letter-spacing: 0.08em;
            color: #3b2418;
            margin-bottom: 1rem;
        }

        .hero .highlight {
            font-weight: 700;
            color: #6f442b;
        }

        .hero p {
            color: #7c563e;
            font-size: 1.05rem;
            letter-spacing: 0.03em;
        }

        .rag-card {
            background: #fffaf3;
            border: 1px solid #e1c7aa;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 10px 28px rgba(93, 58, 38, 0.06);
            margin-bottom: 2rem;
        }

        .ask-card {
            background: #fffaf3;
            border: 1px solid #e1c7aa;
            border-radius: 10px;
            padding: 1.5rem 2rem;
            box-shadow: 0 10px 28px rgba(93, 58, 38, 0.06);
            margin-top: 2rem;
            margin-bottom: 1.5rem;
        }

        .section-label {
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: #8a5a3b;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }

        .message {
            border: 1px solid #ead8c4;
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin: 1rem 0;
            line-height: 1.65;
            font-size: 0.98rem;
        }

        .user-message {
            background-color: #f2e4d2;
            color: #3b2418;
        }

        .assistant-message {
            background-color: #fffdf8;
            color: #3b2418;
        }

        .message-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: #8a5a3b;
            margin-bottom: 0.5rem;
            font-weight: 700;
        }

        textarea {
            background-color: #fffaf3 !important;
            border: none !important;
            color: #3b2418 !important;
            caret-color: #3b2418 !important;
            font-family: Georgia, "Times New Roman", serif !important;
            font-size: 1rem !important;
        }

        textarea::placeholder {
            color: #9b765c !important;
            opacity: 1 !important;
        }

        div[data-baseweb="textarea"] {
            background-color: #fffaf3 !important;
            border: 1px solid #d9b995 !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }

        div[data-baseweb="textarea"]:focus-within {
            border: 1px solid #8a5a3b !important;
            box-shadow: 0 0 0 1px #8a5a3b !important;
        }

        div[data-testid="stFormSubmitButton"] button {
            background-color: #fffaf3 !important;
            color: #6f442b !important;
            border: 1px solid #d9b995 !important;
            border-radius: 6px !important;
            padding: 0.55rem 1.5rem !important;
            min-width: 90px !important;
            font-family: Georgia, "Times New Roman", serif !important;
            letter-spacing: 0.04em !important;
            opacity: 1 !important;
            box-shadow: none !important;
        }

        div[data-testid="stFormSubmitButton"] button p,
        div[data-testid="stFormSubmitButton"] button span,
        div[data-testid="stFormSubmitButton"] button div {
            color: #6f442b !important;
            -webkit-text-fill-color: #6f442b !important;
            opacity: 1 !important;
        }

        div[data-testid="stFormSubmitButton"] button:hover {
            background-color: #f2e4d2 !important;
            color: #3b2418 !important;
            border-color: #8a5a3b !important;
            box-shadow: none !important;
        }

        div[data-testid="stFormSubmitButton"] button:hover p,
        div[data-testid="stFormSubmitButton"] button:hover span,
        div[data-testid="stFormSubmitButton"] button:hover div {
            color: #3b2418 !important;
            -webkit-text-fill-color: #3b2418 !important;
        }

        .streamlit-expanderHeader {
            color: #5d3a26 !important;
            font-family: Georgia, "Times New Roman", serif !important;
        }

        a {
            color: #7a4b2f;
        }
    </style>
    """,
    unsafe_allow_html=True
)


def escape_text(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def render_message(role: str, content: str):
    safe_content = escape_text(content)

    if role == "user":
        label = "Question"
        css_class = "user-message"
    else:
        label = "Answer"
        css_class = "assistant-message"

    st.markdown(
        f"""
        <div class="message {css_class}">
            <div class="message-label">{label}</div>
            <div>{safe_content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def ask_api(question: str, chat_history):
    response = requests.post(
        API_URL,
        json={
            "question": question,
            "chat_history": chat_history
        },
        timeout=120
    )

    response.raise_for_status()
    return response.json()


if "messages" not in st.session_state:
    st.session_state.messages = []


st.markdown(
    """
    <div class="top-nav">
        <div class="brand">FA</div>
        <div class="nav-links">Medical RAG &nbsp; | &nbsp; LangGraph &nbsp; | &nbsp; ChromaDB</div>
    </div>

    <div class="hero">
        <h1>Medical <span class="highlight">RAG Q&A</span> System</h1>
        <p>Ask source-grounded questions across WHO, FDA, CDC, and NICE guideline documents.</p>
    </div>

    <div class="rag-card">
        <div class="section-label">Clinical Guideline Assistant</div>
        <p style="color:#6f442b; margin-bottom:0;">
            This tool retrieves relevant medical guideline passages, generates an answer, and shows the exact document sources used.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# Conversation history appears above the input
for message in st.session_state.messages:
    render_message(message["role"], message["content"])

    if message["role"] == "assistant" and message.get("sources"):
        with st.expander("Sources"):
            for source in message["sources"]:
                st.write(
                    f"[{source['citation_id']}] "
                    f"{source['source_file']} — page {source['page']} "
                    f"(score: {source['similarity_score']})"
                )


# Input box stays at the bottom of the page
st.markdown(
    """
    <div class="ask-card">
        <div class="section-label">Ask a Question</div>
        <p style="color:#6f442b; margin-bottom:0;">
            Enter a clinical guideline question below.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


with st.form("question_form", clear_on_submit=True):
    user_question = st.text_area(
        "Question",
        placeholder="Example: What are the recommended first-line drug classes for treating hypertension?",
        label_visibility="collapsed",
        height=90
    )

    submitted = st.form_submit_button("Ask", type="secondary")


if submitted and user_question.strip():
    previous_history = [
        {
            "role": message["role"],
            "content": message["content"]
        }
        for message in st.session_state.messages
    ]

    with st.spinner("Searching guideline documents..."):
        try:
            result = ask_api(user_question.strip(), previous_history)

            answer = result["answer"]
            sources = result["sources"]

            st.session_state.messages.append({
                "role": "user",
                "content": user_question.strip()
            })

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

            st.rerun()

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI server. Make sure the API is running.")

        except Exception as error:
            st.error(f"Error: {error}")
