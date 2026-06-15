from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from data_ingestion import VectorStore, EmbeddingManager



class DataRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            print("Raw distances:", results['distances'][0])

            # Process results
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error retrieving documents: {e}")
            raise



def main():
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    retriever = DataRetriever(vector_store, embedding_manager)

    queries = [
        "What blood pressure threshold should trigger pharmacological treatment in adults?",
        "Can treatment for hypertension be started before laboratory testing is completed?",
        "What are the recommended first-line drug classes for treating hypertension?",
        "How often should patients be reassessed after starting antihypertensive medication?",
        "A patient has blood pressure of 135/85 and diabetes. Should antihypertensive medication be considered?"
    ]

    for query in queries:
        print("\n" + "=" * 100)
        print(f"QUERY: {query}")
        print("=" * 100)

        results = retriever.retrieve(query)

        for r in results[:5]:
            print(f"\nRank {r['rank']} (score: {r['similarity_score']:.4f})")
            print(f"Source: {r['metadata'].get('source_file')}, page {r['metadata'].get('page')}")
            print(f"Content: {r['content'][:300]}...")


if __name__ == "__main__":
    main()