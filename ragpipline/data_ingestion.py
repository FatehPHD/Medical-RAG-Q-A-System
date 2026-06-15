from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any, Tuple
import os


def process_all_pdfs(pdf_directory):
    all_documents = []

    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    print(f"Found {len(pdf_files)} PDF files to process")

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")

        try:
            loader = PyMuPDFLoader(str(pdf_file))
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = pdf_file.name
                doc.metadata["file_type"] = "pdf"

            all_documents.extend(documents)

            print(f"✓ Loaded {len(documents)} pages")

        except Exception as error:
            print(f"✗ Error loading {pdf_file.name}: {error}")

    print(f"\nTotal documents loaded: {len(all_documents)}")

    return all_documents

def is_bad_chunk(text: str) -> bool:
    text = text.strip()

    # Remove empty or tiny chunks
    if len(text) < 100:
        return True

    # Remove chunks with very few words
    words = text.split()
    if len(words) < 20:
        return True

    # Remove chunks that are mostly repeated title/header text
    unique_words = set(word.lower() for word in words)
    unique_ratio = len(unique_words) / len(words)

    if unique_ratio < 0.35:
        return True

    return False

def split_documents_to_chunks(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(docs)
    original_chunk_count = len(split_docs)
    split_docs = [doc for doc in split_docs if not is_bad_chunk(doc.page_content)]
    print(f"Split {len(docs)} documents into {original_chunk_count} chunks")
    print(f"After cleaning: {len(split_docs)} chunks")
    print(f"Removed: {original_chunk_count - len(split_docs)} bad chunks")

    print(f"Split {len(docs)} documents into {len(split_docs)} chunks")
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    return split_docs


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.model_name = model_name
        self._load_model()
    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_embedding_dimension()}")  
        except Exception as error:
            print(f"Error loading model {self.model_name}: {error}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Embedding model is not loaded.")
        try:
            print(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            print(f"Embeddings generated successfully.")
            print(f"Embedding shape: {embeddings.shape}")
            return embeddings
        except Exception as error:
            print(f"Error generating embeddings: {error}")
            raise




class VectorStore:
    
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "../data/vector_store"):
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embeddings for RAG",
                    "hnsw:space": "cosine"
                }
            )

            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise


    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store

        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # Document content
            documents_text.append(doc.page_content)

            # Embedding
            embeddings_list.append(embedding.tolist())

        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise


        
def main():
    docs = process_all_pdfs("../data")
    chunks= split_documents_to_chunks(docs)
    print(f"\nFinal number of chunks: {len(chunks)}")
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    # Get the text from each chunk
    texts = [chunk.page_content for chunk in chunks]

    # Generate embeddings for all chunks
    embeddings = embedding_manager.generate_embeddings(texts)

    # Store chunks + embeddings in the vector store
    vector_store.add_documents(chunks, embeddings)





if __name__ == "__main__":
    main()