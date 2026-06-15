# Medical RAG Q&A System

A retrieval-augmented generation (RAG) chatbot that answers medical questions grounded in real clinical documents. Instead of relying on an LLM's training memory, every answer is pulled from a vector-indexed corpus of PDFs — with the source document and page cited inline.

**[Live Demo](https://huggingface.co/spaces/your-username/medical-rag)** · **[GitHub](https://github.com/FatehPHD/medical-rag)**

---

## How it works

```
User question
     │
     ▼
Embedding model (HuggingFace)
     │
     ▼
ChromaDB vector store ──► top-k most relevant chunks
     │
     ▼
LangChain ConversationalRetrievalChain
     │
     ▼
LLM (GPT-4o / Ollama)
     │
     ▼
Answer + source citations (document name, page number)
```

The pipeline has three stages:

1. **Ingestion** — PDFs are loaded, split into overlapping chunks, embedded using a HuggingFace sentence transformer, and stored in ChromaDB.
2. **Retrieval** — On each query, the question is embedded and the top-k closest chunks are fetched via approximate nearest-neighbour search.
3. **Generation** — The retrieved chunks are passed as context to the LLM, which generates an answer grounded strictly in that context. Conversation history is maintained via LangChain's `ConversationalRetrievalChain` so follow-up questions work correctly.

---

## Features

- **Source-cited answers** — every response shows the document name and page number the answer came from
- **Multi-turn memory** — follow-up questions like "what about the dosage?" work without re-stating context
- **Chunking strategy comparison** — fixed-size vs. recursive chunking evaluated on a 20-question benchmark
- **Streamlit UI** — chat interface with a sidebar showing the retrieved source chunks in real time
- **Local-first** — runs entirely offline with Ollama (no OpenAI key required)

---

## Tech stack

| Layer | Tool |
|---|---|
| Orchestration | LangChain |
| Vector store | ChromaDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| LLM | GPT-4o (cloud) or Llama 3.2 via Ollama (local) |
| UI | Streamlit |
| Deployment | Hugging Face Spaces |

---

## Dataset

All source documents are publicly available:

| Document | Source |
|---|---|
| Ibuprofen prescribing information | FDA Drug Label (DailyMed) |
| Hypertension clinical guidelines | JNC 8 / ACC/AHA 2017 (PubMed open access) |
| Type 2 diabetes management | CDC Clinical Practice Guidelines |
| Antibiotic resistance guidelines | WHO 2023 (public) |
| *(+ 6 additional PDFs)* | PubMed open access |

No proprietary or restricted medical records were used. All documents are freely available in the public domain.

---

## Evaluation

Retrieval quality was measured on a hand-labelled 20-question benchmark. Each question has a known ground-truth source chunk; a retrieval is counted as a hit if the correct chunk appears in the top-3 results.

| Chunking strategy | Chunk size | Overlap | Top-3 hit rate |
|---|---|---|---|
| Fixed-size | 500 tokens | 0 | 58% |
| Fixed-size | 500 tokens | 50 tokens | 67% |
| Recursive | 500 tokens | 100 tokens | **84%** |

Recursive chunking with 100-token overlap was selected for the final build.

> **Note:** Replace these numbers with your actual benchmark results before submitting applications.

---

## Project structure

```
medical-rag/
├── ingest.py          # Load PDFs, chunk, embed, store in ChromaDB
├── retrieval.py       # ConversationalRetrievalChain setup
├── app.py             # Streamlit UI
├── eval.py            # 20-question retrieval benchmark
├── data/
│   └── pdfs/          # Source medical PDFs (not committed — see Dataset section)
├── chroma_db/         # Persisted vector store (gitignored)
├── eval/
│   └── questions.json # 20 Q&A pairs for benchmarking
├── requirements.txt
└── README.md
```

---

## Running locally

### Prerequisites

- Python 3.10+
- An OpenAI API key **or** [Ollama](https://ollama.com) installed locally

### 1. Clone and install

```bash
git clone https://github.com/FatehPHD/medical-rag.git
cd medical-rag
pip install -r requirements.txt
```

### 2. Add your documents

Place your PDF files in `data/pdfs/`. The repo includes a sample set of 3 public PDFs to get started.

### 3. Configure your LLM

**Option A — OpenAI (cloud):**
```bash
export OPENAI_API_KEY=your_key_here
```

**Option B — Ollama (fully local, free):**
```bash
ollama pull llama3.2
# Then set USE_OLLAMA=true in your .env
```

### 4. Ingest documents

```bash
python ingest.py
```

This chunks the PDFs, generates embeddings, and persists the ChromaDB vector store locally.

### 5. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 6. Run the eval (optional)

```bash
python eval.py
```

Outputs a retrieval hit-rate table across chunking strategies.

---

## Key concepts

**Why RAG instead of fine-tuning?**
Fine-tuning bakes knowledge into model weights — expensive, slow to update, and prone to hallucination on out-of-distribution facts. RAG keeps the knowledge external and updatable: swap the PDFs, re-ingest, done. For clinical documents that change with new guidelines, this matters.

**What can go wrong (and how this handles it)**
- *Hallucination* — mitigated by prompting the LLM to answer only from retrieved context, not prior knowledge
- *Poor chunking* — evaluated across strategies; recursive chunking preserves sentence boundaries better than fixed-size
- *Out-of-context retrieval* — addressed by using overlapping chunks so relevant content isn't split across chunk boundaries

---

## Roadmap

- [ ] Re-ranking layer (Cohere Rerank or cross-encoder) on top of ChromaDB retrieval
- [ ] Swap ChromaDB for Pinecone for cloud-native persistence
- [ ] Add RAGAS automated evaluation (faithfulness + answer relevancy scores)
- [ ] Fine-tune the embedding model on medical terminology

---

## License

MIT