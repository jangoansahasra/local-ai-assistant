import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# Initialize embedding model (runs locally, ~90MB)
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB (local vector store)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

def load_text_file(filepath):
    with open(filepath, "r") as f:
        return f.read()

def load_pdf_file(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=50, overlap=10):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def ingest_documents(folder_path="./documents"):
    """Load all documents from a folder into the vector store."""
    files = os.listdir(folder_path)
    total_chunks = 0

    for filename in files:
        filepath = os.path.join(folder_path, filename)

        if filename.endswith(".txt"):
            text = load_text_file(filepath)
        elif filename.endswith(".pdf"):
            text = load_pdf_file(filepath)
        else:
            print(f"Skipping unsupported file: {filename}")
            continue

        chunks = chunk_text(text)
        print(f"Processing {filename}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            doc_id = f"{filename}_chunk_{i}"
            embedding = embedder.encode(chunk).tolist()

            collection.upsert(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": filename, "chunk_index": i}]
            )

        total_chunks += len(chunks)

    print(f"\nIngested {len(files)} files, {total_chunks} total chunks.")

def search(query, n_results=3):
    """Search the vector store for relevant chunks."""
    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results

def build_rag_prompt(query, search_results):
    """Build a prompt with retrieved context."""
    context_parts = []
    for i, doc in enumerate(search_results["documents"][0]):
        source = search_results["metadatas"][0][i]["source"]
        context_parts.append(f"[Source: {source}]\n{doc}")

    context = "\n\n".join(context_parts)

    prompt = f"""Use the following context to answer the question. If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""

    return prompt

if __name__ == "__main__":
    # Step 1: Ingest documents
    print("=" * 60)
    print("INGESTING DOCUMENTS")
    print("=" * 60)
    ingest_documents()

    # Step 2: Test a search
    print("\n" + "=" * 60)
    print("TESTING SEARCH")
    print("=" * 60)

    test_queries = [
        "What are the types of machine learning?",
        "What is overfitting and how do you prevent it?",
        "What Python libraries are used for machine learning?",
        "What are the three cloud service models?",
        "What is serverless computing?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = search(query)
        print(f"Top result: {results['documents'][0][0][:150]}...")
        print(f"Source: {results['metadatas'][0][0]['source']}")