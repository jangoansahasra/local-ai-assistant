import requests
from rag import embedder, collection, search, build_rag_prompt, ingest_documents

OLLAMA_URL = "http://localhost:11434/api/generate"

def ask(query, model="llama3.2:3b"):
    # Search for relevant context
    results = search(query)

    # Build prompt with context
    prompt = build_rag_prompt(query, results)

    # Send to Ollama
    response = requests.post(OLLAMA_URL, json={
        "model": model,
        "prompt": prompt,
        "temperature": 0.3,
        "stream": False
    })

    answer = response.json().get("response", "No response.")

    # Show sources
    sources = set()
    for meta in results["metadatas"][0]:
        sources.add(meta["source"])

    return answer, sources

if __name__ == "__main__":
    print("=" * 60)
    print("RAG CHAT — Ask questions about your documents")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break

        answer, sources = ask(query)
        print(f"\nAssistant: {answer}")
        print(f"\n📄 Sources: {', '.join(sources)}")