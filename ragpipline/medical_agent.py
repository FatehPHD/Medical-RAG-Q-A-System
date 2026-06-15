
import os
from pathlib import Path
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
from langgraph.graph import StateGraph, START, END

from data_ingestion import VectorStore, EmbeddingManager
from data_retriever import DataRetriever


# =========================
# Load environment variables
# =========================

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Make sure .env is in the main project folder.")

client = OpenAI(api_key=api_key)


# =========================
# Test questions
# =========================

test_questions = [
    "What blood pressure threshold should trigger pharmacological treatment in adults?",
    "Can hypertension treatment be started before laboratory testing is completed?",
    "What are the recommended first-line drug classes for treating hypertension?",
    "How often should patients be reassessed after starting antihypertensive medication?",
    "How should blood pressure be measured accurately in adults?",
    "When should ambulatory blood pressure monitoring be used to confirm hypertension?",
    "What are the general objectives of diabetes management?",
    "Why is patient education important in diabetes management?",
    "What long-term complications can diabetes cause?",
    "What is metformin used for?",
    "How does metformin lower blood glucose?",
    "What are the main indications for lisinopril?",
    "What is the boxed warning for lisinopril?",
    "What is atorvastatin used to reduce the risk of?",
    "What serious muscle-related warning is associated with atorvastatin?",
    "What is amlodipine used to treat?",
    "What is the recommended adult starting dose of amlodipine?",
    "What is antibiotic stewardship?",
    "Why should diagnostic tests be used wisely before prescribing antibiotics?",
    "What does the WHO tuberculosis report say about global TB progress and challenges?"
]


# =========================
# LangGraph state
# =========================

class MedicalAgentState(TypedDict):
    question: str
    chat_history: List[Dict[str, str]]
    search_query: str
    retrieved_docs: List[Dict[str, Any]]
    context: str
    answer: str


# =========================
# Setup retriever once
# =========================

embedding_manager = EmbeddingManager()

vector_store = VectorStore(
    persist_directory=str(ROOT_DIR / "data" / "vector_store")
)

retriever = DataRetriever(vector_store, embedding_manager)


# =========================
# Helper functions
# =========================

def retrieve_medical_context(search_query: str) -> List[Dict[str, Any]]:
    """
    Custom RAG retrieval tool.
    Takes a search query, searches ChromaDB,
    and returns the most relevant chunks with metadata.
    """

    retrieved_docs = retriever.retrieve(
        query=search_query,
        top_k=5,
        score_threshold=0.50
    )

    return retrieved_docs


def format_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    context_parts = []

    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc["metadata"].get("source_file", "Unknown source")
        page = doc["metadata"].get("page", "Unknown page")
        score = doc["similarity_score"]
        content = doc["content"]

        context_parts.append(
            f"[{i}] Source: {source}, Page: {page}, Score: {score:.4f}\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)


def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    if not chat_history:
        return "No previous conversation."

    history_lines = []

    for message in chat_history[-6:]:
        role = message.get("role", "unknown")
        content = message.get("content", "")
        history_lines.append(f"{role.upper()}: {content}")

    return "\n".join(history_lines)


def build_search_query(question: str, chat_history: List[Dict[str, str]]) -> str:
    """
    Builds a stronger retrieval query.

    This is important for follow-up questions like:
    "What about for patients with diabetes?"

    Instead of searching only that vague sentence, we include recent chat history
    so ChromaDB understands the topic is still hypertension treatment.
    """

    if not chat_history:
        return question

    recent_history = []

    for message in chat_history[-4:]:
        role = message.get("role", "")
        content = message.get("content", "")
        recent_history.append(f"{role}: {content}")

    history_text = "\n".join(recent_history)

    search_query = f"""
Conversation context:
{history_text}

Current question:
{question}

Search for guideline information that answers the current question in the context of the conversation.
"""

    return search_query


def call_chatgpt(question: str, context: str, chat_history: List[Dict[str, str]]) -> str:
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    history = format_chat_history(chat_history)

    prompt = f"""
You are a medical guideline assistant.

Answer the user's current question using ONLY the provided retrieved context.
Use the conversation history only to understand follow-up questions.
Do not use outside medical knowledge.

If the answer is not in the context, say:
"I could not find this in the provided documents."

Use citations like [1], [2], [3] based on the context chunks.
Keep the answer clear, direct, and clinically cautious.
Do not give personal medical advice.
Do not tell the user to take or stop medication.

Conversation history:
{history}

Current question:
{question}

Retrieved context:
{context}
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return response.output_text


# =========================
# LangGraph nodes
# =========================

def retrieve_context_node(state: MedicalAgentState) -> MedicalAgentState:
    question = state["question"]
    chat_history = state.get("chat_history", [])

    search_query = build_search_query(question, chat_history)

    retrieved_docs = retrieve_medical_context(search_query)
    context = format_context(retrieved_docs)

    return {
        "question": question,
        "chat_history": chat_history,
        "search_query": search_query,
        "retrieved_docs": retrieved_docs,
        "context": context,
        "answer": ""
    }


def generate_answer_node(state: MedicalAgentState) -> MedicalAgentState:
    question = state["question"]
    chat_history = state.get("chat_history", [])
    search_query = state.get("search_query", question)
    retrieved_docs = state["retrieved_docs"]
    context = state["context"]

    if not retrieved_docs:
        answer = "I could not find this in the provided documents."
    else:
        answer = call_chatgpt(question, context, chat_history)

    return {
        "question": question,
        "chat_history": chat_history,
        "search_query": search_query,
        "retrieved_docs": retrieved_docs,
        "context": context,
        "answer": answer
    }


# =========================
# Build LangGraph agent
# =========================

def build_medical_agent():
    graph = StateGraph(MedicalAgentState)

    graph.add_node("retrieve_context", retrieve_context_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.add_edge(START, "retrieve_context")
    graph.add_edge("retrieve_context", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()


medical_agent = build_medical_agent()


# =========================
# Run agent
# =========================

def run_agent(question: str):
    result = medical_agent.invoke({
        "question": question,
        "chat_history": [],
        "search_query": "",
        "retrieved_docs": [],
        "context": "",
        "answer": ""
    })

    print("\nQUESTION:")
    print(result["question"])

    print("\nSEARCH QUERY:")
    print(result["search_query"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nSOURCES:")
    for i, doc in enumerate(result["retrieved_docs"], start=1):
        source = doc["metadata"].get("source_file", "Unknown source")
        page = doc["metadata"].get("page", "Unknown page")
        score = doc["similarity_score"]

        print(f"[{i}] {source}, page {page}, score {score:.4f}")


def main():
    mode = input("Type 1 for one question, 2 for test questions: ")

    if mode == "1":
        question = input("Ask a medical question: ")
        run_agent(question)

    elif mode == "2":
        for question in test_questions:
            print("\n" + "=" * 100)
            run_agent(question)

    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()
