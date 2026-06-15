from data_ingestion import VectorStore, EmbeddingManager
from data_retriever import DataRetriever


test_questions = [
    {
        "question": "What blood pressure threshold should trigger pharmacological treatment in adults?",
        "expected_source": "WHO_Hypertension_2021.pdf"
    },
    {
        "question": "Can hypertension treatment be started before laboratory testing is completed?",
        "expected_source": "WHO_Hypertension_2021.pdf"
    },
    {
        "question": "What are the recommended first-line drug classes for treating hypertension?",
        "expected_source": "WHO_Hypertension_2021.pdf"
    },
    {
        "question": "How often should patients be reassessed after starting antihypertensive medication?",
        "expected_source": "WHO_Hypertension_2021.pdf"
    },
    {
        "question": "How should blood pressure be measured accurately in adults?",
        "expected_source": "NICE_Hypertension_Guideline.pdf"
    },
    {
        "question": "When should ambulatory blood pressure monitoring be used to confirm hypertension?",
        "expected_source": "NICE_Hypertension_Guideline.pdf"
    },
    {
        "question": "What are the general objectives of diabetes management?",
        "expected_source": "WHO_Diabetes_Guidelines.pdf"
    },
    {
        "question": "Why is patient education important in diabetes management?",
        "expected_source": "WHO_Diabetes_Guidelines.pdf"
    },
    {
        "question": "What long-term complications can diabetes cause?",
        "expected_source": "WHO_Diabetes_Guidelines.pdf"
    },
    {
        "question": "What is metformin used for?",
        "expected_source": "FDA_Metformin_Label.pdf"
    },
    {
        "question": "How does metformin lower blood glucose?",
        "expected_source": "FDA_Metformin_Label.pdf"
    },
    {
        "question": "What are the main indications for lisinopril?",
        "expected_source": "FDA_Lisinopril_Label.pdf"
    },
    {
        "question": "What is the boxed warning for lisinopril?",
        "expected_source": "FDA_Lisinopril_Label.pdf"
    },
    {
        "question": "What is atorvastatin used to reduce the risk of?",
        "expected_source": "FDA_Atorvastatin_Label.pdf"
    },
    {
        "question": "What serious muscle-related warning is associated with atorvastatin?",
        "expected_source": "FDA_Atorvastatin_Label.pdf"
    },
    {
        "question": "What is amlodipine used to treat?",
        "expected_source": "FDA_Amlodipine_Label.pdf"
    },
    {
        "question": "What is the recommended adult starting dose of amlodipine?",
        "expected_source": "FDA_Amlodipine_Label.pdf"
    },
    {
        "question": "What is antibiotic stewardship?",
        "expected_source": "CDC_Antibiotic_Prescribing.pdf"
    },
    {
        "question": "Why should diagnostic tests be used wisely before prescribing antibiotics?",
        "expected_source": "CDC_Antibiotic_Prescribing.pdf"
    },
    {
        "question": "What does the WHO tuberculosis report say about global TB progress and challenges?",
        "expected_source": "WHO_Tuberculosis_2022.pdf"
    }
]


def format_context(retrieved_docs):
    context_parts = []

    for doc in retrieved_docs:
        source = doc["metadata"].get("source_file", "Unknown source")
        page = doc["metadata"].get("page", "Unknown page")
        content = doc["content"]

        context_parts.append(
            f"[Source: {source}, Page: {page}]\n{content}"
        )

    return "\n\n---\n\n".join(context_parts)


def answer_question(question: str):
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    retriever = DataRetriever(vector_store, embedding_manager)

    retrieved_docs = retriever.retrieve(
        query=question,
        top_k=5,
        score_threshold=0.50
    )

    context = format_context(retrieved_docs)

    print("\nQUESTION:")
    print(question)

    print("\nRETRIEVED CONTEXT:")
    print(context)

    print("\nSOURCES:")
    for doc in retrieved_docs:
        print(
            f"- {doc['metadata'].get('source_file')} "
            f"page {doc['metadata'].get('page')} "
            f"score {doc['similarity_score']:.4f}"
        )


def main():
    embedding_manager = EmbeddingManager()
    vector_store = VectorStore()
    retriever = DataRetriever(vector_store, embedding_manager)

    for item in test_questions:
        question = item["question"]
        expected_source = item["expected_source"]

        print("\n" + "=" * 100)
        print(f"QUESTION: {question}")
        print(f"EXPECTED SOURCE: {expected_source}")
        print("=" * 100)

        retrieved_docs = retriever.retrieve(
            query=question,
            top_k=5,
            score_threshold=0.50
        )

        found_in_top_3 = False

        for doc in retrieved_docs[:3]:
            source = doc["metadata"].get("source_file")
            page = doc["metadata"].get("page")
            score = doc["similarity_score"]

            print(f"- {source}, page {page}, score {score:.4f}")

            if source == expected_source:
                found_in_top_3 = True

        if found_in_top_3:
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")

if __name__ == "__main__":
    main()