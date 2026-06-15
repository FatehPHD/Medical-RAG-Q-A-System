
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from medical_agent import medical_agent


app = FastAPI(
    title="Medical RAG Q&A API",
    description="FastAPI backend for a LangGraph medical RAG system using ChromaDB and OpenAI.",
    version="1.0.0"
)


class ChatMessage(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    chat_history: List[ChatMessage] = Field(default_factory=list)


class Source(BaseModel):
    citation_id: int
    source_file: str
    page: int | str
    similarity_score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]


@app.get("/")
def home():
    return {
        "message": "Medical RAG Q&A API is running.",
        "endpoint": "POST /ask"
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chat_history = [
        {
            "role": message.role,
            "content": message.content
        }
        for message in request.chat_history
    ]

    result = medical_agent.invoke({
        "question": question,
        "chat_history": chat_history,
        "retrieved_docs": [],
        "context": "",
        "answer": ""
    })

    sources = []

    for i, doc in enumerate(result["retrieved_docs"], start=1):
        metadata = doc.get("metadata", {})

        sources.append(
            Source(
                citation_id=i,
                source_file=metadata.get("source_file", "Unknown source"),
                page=metadata.get("page", "Unknown page"),
                similarity_score=round(doc.get("similarity_score", 0.0), 4)
            )
        )

    return AskResponse(
        question=result["question"],
        answer=result["answer"],
        sources=sources
    )

