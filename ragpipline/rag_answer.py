
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from data_ingestion import VectorStore, EmbeddingManager
from data_retriever import DataRetriever


# Load .env from the main project folder
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")


api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Make sure your .env file is in the main project folder.")

client = OpenAI(api_key=api_key)


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


def format_context(retrieved_docs):
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


def generate_answer(question, context):
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    prompt = f"""
You are a medical guideline assistant.

Answer the user's question using ONLY the provided context.
Do not use outside knowledge.
If the answer is not in the context, say: "I could not find this in the provided documents."

Use citations like [1], [2], [3] based on the context chunks.
Keep the answer clear, direct, and clinically cautious.
Do not give personal medical advice.

Question:
{question}

Context:
{context}
"""

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    return response.output_text


def answer_question(question, retriever):
    retrieved_docs = retriever.retrieve(
        query=question,
        top_k=5,
        score_threshold=0.50
    )

    if not retrieved_docs:
        print("\nQUESTION:")
        print(question)
        print("\nANSWER:")
        print("I could not find this in the provided documents.")
        return

    context = format_context(retrieved_docs)
    answer = generate_answer(question, context)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nSOURCES:")
    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc["metadata"].get("source_file", "Unknown source")
        page = doc["metadata"].get("page", "Unknown page")
        score = doc["similarity_score"]

        print(f"[{i}] {source}, page {page}, score {score:.4f}")


def main():
    embedding_manager = EmbeddingManager()

    vector_store = VectorStore(
        persist_directory=str(ROOT_DIR / "data" / "vector_store")
    )

    retriever = DataRetriever(vector_store, embedding_manager)

    mode = input("Type 1 for one question, 2 for test questions: ")

    if mode == "1":
        question = input("Ask a medical question: ")
        answer_question(question, retriever)

    elif mode == "2":
        for question in test_questions:
            print("\n" + "=" * 100)
            answer_question(question, retriever)

    else:
        print("Invalid option.")


if __name__ == "__main__":
    main()
