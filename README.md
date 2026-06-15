
# Medical RAG Q&A System

A medical retrieval-augmented generation (RAG) system that answers clinical guideline questions using source-grounded context from WHO, FDA, CDC, and NICE documents.

Instead of relying only on an LLM's general knowledge, the system retrieves relevant passages from a ChromaDB vector store and generates answers with document-level and page-level citations.

**GitHub:** [Medical RAG Q&A System](https://github.com/FatehPHD/Medical-RAG-Q-A-System)

---

## Overview

This project ingests clinical guideline and drug-label PDFs, splits them into cleaned text chunks, embeds them using a HuggingFace sentence-transformer model, stores them in ChromaDB, and uses a LangGraph-based agent to retrieve relevant context before generating a final answer.

The system includes:

- PDF ingestion pipeline
- ChromaDB vector store
- HuggingFace embeddings
- LangGraph medical RAG agent
- OpenAI-powered answer generation
- FastAPI backend
- Streamlit chat interface
- Multi-turn conversational memory
- Retrieval evaluation benchmark

---

## How It Works

```txt
User Question
     |
     v
HuggingFace Embedding Model
     |
     v
ChromaDB Vector Store
     |
     v
Top-K Relevant PDF Chunks
     |
     v
LangGraph Medical Agent
     |
     v
OpenAI Response Generation
     |
     v
Answer + Source Citations


The pipeline has four main stages:

1. **Ingestion**
   PDFs are loaded with PyMuPDF, split into recursive overlapping chunks, cleaned using quality filters, embedded using `all-MiniLM-L6-v2`, and stored in ChromaDB.

2. **Retrieval**
   A user question is embedded and compared against stored document chunks using cosine similarity. Low-quality matches are filtered using a score threshold.

3. **Agent Reasoning**
   A LangGraph workflow builds a retrieval query, fetches relevant medical context, and passes it to the answer-generation step.

4. **Generation**
   The LLM answers using only the retrieved context and includes citations pointing to the source document and page number.

---

## Features

* **Source-grounded answers** with document and page citations
* **10 clinical PDFs ingested** from WHO, FDA, CDC, and NICE
* **1,659 cleaned chunks** stored in ChromaDB
* **384-dimensional HuggingFace embeddings**
* **Cosine similarity retrieval**
* **Score-threshold filtering** to suppress weak matches
* **LangGraph agent architecture**
* **FastAPI REST endpoint**
* **Streamlit chat UI**
* **Multi-turn conversational memory**
* **Retrieval benchmark with 20 clinical questions**

---

## Tech Stack

| Layer           | Tool                                     |
| --------------- | ---------------------------------------- |
| Language        | Python                                   |
| PDF Loading     | PyMuPDF                                  |
| Chunking        | LangChain RecursiveCharacterTextSplitter |
| Embeddings      | HuggingFace `all-MiniLM-L6-v2`           |
| Vector Store    | ChromaDB                                 |
| Agent Framework | LangGraph                                |
| LLM             | OpenAI                                   |
| Backend API     | FastAPI                                  |
| Frontend        | Streamlit                                |
| Evaluation      | Custom retrieval benchmark               |

---

## Documents Used

The system was tested on 10 public clinical documents:

| Document                            |
| ----------------------------------- |
| NICE Hypertension Guideline         |
| WHO Diabetes Guidelines             |
| WHO Hypertension 2021               |
| WHO Tuberculosis 2022               |
| WHO HIV Treatment Guideline         |
| FDA Metformin Label                 |
| FDA Lisinopril Label                |
| FDA Atorvastatin Label              |
| FDA Amlodipine Label                |
| CDC Antibiotic Prescribing Guidance |

No patient records, private medical data, or proprietary datasets were used.

---

## Evaluation

Retrieval quality was evaluated on a 20-question benchmark. Each question was assigned an expected source document. A result was counted as correct if the expected document appeared in the retrieved results.

| Metric                   | Result         |
| ------------------------ | -------------- |
| Top-1 Retrieval Accuracy | 19/20 = 95.0%  |
| Top-3 Retrieval Accuracy | 20/20 = 100.0% |
| Top-5 Retrieval Accuracy | 20/20 = 100.0% |

The final configuration used:

| Setting             | Value              |
| ------------------- | ------------------ |
| Embedding Model     | `all-MiniLM-L6-v2` |
| Embedding Dimension | 384                |
| Vector Store        | ChromaDB           |
| Similarity Metric   | Cosine Similarity  |
| Chunking Strategy   | Recursive Chunking |
| Chunk Size          | 1000 characters    |
| Chunk Overlap       | 200 characters     |
| Score Threshold     | 0.50               |

---

## Project Structure

```txt
Medical RAG Q&A System/
├── data/
│   ├── NICE_Hypertension_Guideline.pdf
│   ├── WHO_Diabetes_Guidelines.pdf
│   ├── WHO_Hypertension_2021.pdf
│   ├── WHO_Tuberculosis_2022.pdf
│   ├── WHO_HIV_Treatment.pdf
│   ├── FDA_Metformin_Label.pdf
│   ├── FDA_Lisinopril_Label.pdf
│   ├── FDA_Atorvastatin_Label.pdf
│   ├── FDA_Amlodipine_Label.pdf
│   ├── CDC_Antibiotic_Prescribing.pdf
│   └── vector_store/
│
├── ragpipeline/
│   ├── data_ingestion.py
│   ├── data_retriever.py
│   ├── rag_qa.py
│   ├── rag_answer.py
│   ├── medical_agent.py
│   ├── api.py
│   ├── app.py
│   └── evaluation.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── pyproject.toml
```

---

## Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/FatehPHD/Medical-RAG-Q-A-System.git
cd "Medical RAG Q&A System"
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Environment Variables

Create a `.env` file in the project root:

```txt
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 5. Ingest the PDFs

From inside the `ragpipeline` folder:

```bash
cd ragpipeline
python data_ingestion.py
```

This loads the PDFs, creates chunks, generates embeddings, and stores them in ChromaDB.

### 6. Run the FastAPI Backend

```bash
uvicorn api:app --reload
```

The API will run at:

```txt
http://127.0.0.1:8000
```

Interactive API docs:

```txt
http://127.0.0.1:8000/docs
```

### 7. Run the Streamlit App

Open another terminal, go to the `ragpipeline` folder, and run:

```bash
streamlit run app.py
```

The app will open at:

```txt
http://localhost:8501
```

### 8. Run the Evaluation Benchmark

```bash
python evaluation.py
```

---

## Example Questions

```txt
What are the recommended first-line drug classes for treating hypertension?
```

```txt
What about for patients with diabetes?
```

```txt
What is metformin used for?
```

```txt
What is antibiotic stewardship?
```

```txt
What is the boxed warning for lisinopril?
```

---

## Example Output

```txt
Question:
What are the recommended first-line drug classes for treating hypertension?

Answer:
The recommended first-line drug classes for treating hypertension in adults are:

1. Thiazide and thiazide-like agents
2. Angiotensin-converting enzyme inhibitors (ACEis) / angiotensin-receptor blockers (ARBs)
3. Long-acting dihydropyridine calcium channel blockers (CCBs)

Sources:
[1] WHO_Hypertension_2021.pdf — page 22
[2] WHO_Hypertension_2021.pdf — page 9
```

---

## Important Note

This project is for educational and portfolio purposes only. It is not a medical device and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

---

## Future Improvements

* Add reranking with a cross-encoder or Cohere Rerank
* Add automated RAG evaluation with RAGAS
* Deploy Streamlit frontend publicly
* Add user-uploaded PDF support
* Add source preview cards in the UI
* Add Docker support
* Add CI testing for ingestion and retrieval

---

## License

MIT

```
```
