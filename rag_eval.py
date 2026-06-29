from rag import ingest_documents, search

TEST_CASES = [
    {
        "question": "What are the three main types of machine learning?",
        "expected_source": "ml_basics.txt",
        "expected_terms": ["supervised", "unsupervised", "reinforcement"],
    },
    {
        "question": "What is overfitting and how can it be prevented?",
        "expected_source": "ml_basics.txt",
        "expected_terms": ["overfitting", "regularization", "cross-validation"],
    },
    {
        "question": "What Python libraries are commonly used for machine learning?",
        "expected_source": "ml_basics.txt",
        "expected_terms": ["scikit-learn", "tensorflow", "pytorch"],
    },
    {
        "question": "What are the three cloud service models?",
        "expected_source": "cloud_computing.txt",
        "expected_terms": ["iaas", "paas", "saas"],
    },
    {
        "question": "What is serverless computing?",
        "expected_source": "cloud_computing.txt",
        "expected_terms": ["serverless", "provisioning", "lambda"],
    },
    {
        "question": "Who are the major cloud providers?",
        "expected_source": "cloud_computing.txt",
        "expected_terms": ["aws", "azure", "google cloud"],
    },
]


def evaluate_rag():
    print("=" * 70)
    print("RAG RETRIEVAL EVALUATION")
    print("=" * 70)

    print("\nIndexing documents...")
    ingest_documents()

    source_matches = 0
    content_matches = 0

    for i, test in enumerate(TEST_CASES, start=1):
        question = test["question"]
        expected_source = test["expected_source"]
        expected_terms = test["expected_terms"]

        results = search(question, n_results=3)

        retrieved_docs = results["documents"][0]
        retrieved_metadata = results["metadatas"][0]

        retrieved_sources = [meta["source"] for meta in retrieved_metadata]
        combined_text = " ".join(retrieved_docs).lower()

        source_correct = expected_source in retrieved_sources
        terms_found = [
            term for term in expected_terms
            if term.lower() in combined_text
        ]
        content_correct = len(terms_found) > 0

        if source_correct:
            source_matches += 1

        if content_correct:
            content_matches += 1

        print(f"\nTest {i}: {question}")
        print(f"Expected source: {expected_source}")
        print(f"Retrieved sources: {retrieved_sources}")
        print(f"Source match: {'PASS' if source_correct else 'FAIL'}")
        print(f"Expected terms found: {terms_found}")
        print(f"Content match: {'PASS' if content_correct else 'FAIL'}")
        print(f"Top chunk preview: {retrieved_docs[0][:180]}...")

    total = len(TEST_CASES)
    source_accuracy = (source_matches / total) * 100
    content_accuracy = (content_matches / total) * 100

    print("\n" + "=" * 70)
    print("FINAL RAG EVALUATION RESULTS")
    print("=" * 70)
    print(f"Source retrieval accuracy: {source_matches}/{total} ({source_accuracy:.1f}%)")
    print(f"Retrieved content match: {content_matches}/{total} ({content_accuracy:.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_rag()