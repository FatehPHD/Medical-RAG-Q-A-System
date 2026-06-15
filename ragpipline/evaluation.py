from pathlib import Path

from data_ingestion import VectorStore, EmbeddingManager
from data_retriever import DataRetriever


ROOT_DIR = Path(__file__).resolve().parents[1]


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


def evaluate_retrieval():
    embedding_manager = EmbeddingManager()

    vector_store = VectorStore(
        persist_directory=str(ROOT_DIR / "data" / "vector_store")
    )

    retriever = DataRetriever(vector_store, embedding_manager)

    top_1_passes = 0
    top_3_passes = 0
    top_5_passes = 0

    total = len(test_questions)

    print("\nRunning retrieval benchmark...")
    print("=" * 100)

    for index, item in enumerate(test_questions, start=1):
        question = item["question"]
        expected_source = item["expected_source"]

        print(f"\n[{index}/{total}] QUESTION: {question}")
        print(f"Expected source: {expected_source}")

        results = retriever.retrieve(
            query=question,
            top_k=5,
            score_threshold=0.50
        )

        retrieved_sources = [
            result["metadata"].get("source_file")
            for result in results
        ]

        top_1_hit = expected_source in retrieved_sources[:1]
        top_3_hit = expected_source in retrieved_sources[:3]
        top_5_hit = expected_source in retrieved_sources[:5]

        if top_1_hit:
            top_1_passes += 1

        if top_3_hit:
            top_3_passes += 1

        if top_5_hit:
            top_5_passes += 1

        for rank, result in enumerate(results[:5], start=1):
            source = result["metadata"].get("source_file")
            page = result["metadata"].get("page")
            score = result["similarity_score"]

            print(f"  Rank {rank}: {source}, page {page}, score {score:.4f}")

        if top_3_hit:
            print("RESULT: PASS")
        else:
            print("RESULT: FAIL")

    print("\n" + "=" * 100)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 100)

    print(f"Top-1 retrieval accuracy: {top_1_passes}/{total} = {(top_1_passes / total) * 100:.1f}%")
    print(f"Top-3 retrieval accuracy: {top_3_passes}/{total} = {(top_3_passes / total) * 100:.1f}%")
    print(f"Top-5 retrieval accuracy: {top_5_passes}/{total} = {(top_5_passes / total) * 100:.1f}%")


if __name__ == "__main__":
    evaluate_retrieval()