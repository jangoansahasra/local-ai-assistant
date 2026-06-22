import streamlit as st
import requests
import time
import os
import chromadb
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

OLLAMA_URL = "http://localhost:11434/api/generate"
MODELS = ["llama3.2:3b", "phi4-mini:latest", "mistral:7b"]

# --- RAG Setup ---
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(name="documents", metadata={"hnsw:space": "cosine"})

embedder = load_embedder()
collection = load_collection()

def chunk_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def ingest_file(filepath, filename):
    if filename.endswith(".txt"):
        with open(filepath, "r") as f:
            text = f.read()
    elif filename.endswith(".pdf"):
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    else:
        return 0

    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        doc_id = f"{filename}_chunk_{i}"
        embedding = embedder.encode(chunk).tolist()
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": filename, "chunk_index": i}]
        )
    return len(chunks)

def search_docs(query, n_results=3):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results

def build_rag_prompt(query, search_results):
    context_parts = []
    for i, doc in enumerate(search_results["documents"][0]):
        source = search_results["metadatas"][0][i]["source"]
        context_parts.append(f"[Source: {source}]\n{doc}")
    context = "\n\n".join(context_parts)
    return f"""Use the following context to answer the question. If the answer is not in the context, say "I don't have enough information in my documents to answer that."

Context:
{context}

Question: {query}

Answer:"""

# --- UI ---
st.set_page_config(page_title="Local AI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Local AI Assistant")
st.caption("Powered by Ollama — running 100% locally on your Mac")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    selected_model = st.selectbox("Model", MODELS)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    use_rag = st.toggle("📄 Use RAG (chat with documents)", value=False)

    st.divider()

    # Document upload
    st.header("📁 Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            save_path = os.path.join("./documents", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            chunks = ingest_file(save_path, uploaded_file.name)
            st.success(f"✅ {uploaded_file.name}: {chunks} chunks indexed")

    # Show indexed documents
    try:
        all_docs = collection.get()
        if all_docs["ids"]:
            sources = set(m["source"] for m in all_docs["metadatas"])
            st.caption(f"📊 {len(all_docs['ids'])} chunks from {len(sources)} files")
            for source in sorted(sources):
                st.caption(f"  • {source}")
    except:
        st.caption("No documents indexed yet.")

    st.divider()
    st.markdown("**Hardware:** Apple M4, 16GB RAM")
    st.markdown("**Runtime:** Ollama v0.21.0")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metrics" in msg:
            m = msg["metrics"]
            st.caption(
                f"📊 {m['tokens_generated']} tokens | "
                f"{m['tokens_per_second']} tok/s | "
                f"TTFT: {m['ttft']}s | "
                f"Latency: {m['latency']}s"
            )
        if "sources" in msg:
            st.caption(f"📄 Sources: {msg['sources']}")

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start_time = time.time()

            # Build prompt (with or without RAG)
            if use_rag:
                search_results = search_docs(prompt)
                final_prompt = build_rag_prompt(prompt, search_results)
                sources = set(m["source"] for m in search_results["metadatas"][0])
                sources_str = ", ".join(sources)
            else:
                final_prompt = prompt
                sources_str = None

            try:
                response = requests.post(OLLAMA_URL, json={
                    "model": selected_model,
                    "prompt": final_prompt,
                    "temperature": temperature,
                    "stream": False
                }, timeout=120)

                result = response.json()
                total_time = time.time() - start_time
                eval_count = result.get("eval_count", 0)
                eval_duration = result.get("eval_duration", 1)
                prompt_eval_duration = result.get("prompt_eval_duration", 0)

                tps = round(eval_count / (eval_duration / 1e9), 2) if eval_duration else 0
                ttft = round(prompt_eval_duration / 1e9, 4)
                latency = round(total_time, 4)

                answer = result.get("response", "No response received.")
                st.markdown(answer)
                st.caption(
                    f"📊 {eval_count} tokens | {tps} tok/s | "
                    f"TTFT: {ttft}s | Latency: {latency}s"
                )
                if sources_str:
                    st.caption(f"📄 Sources: {sources_str}")

                msg_data = {
                    "role": "assistant",
                    "content": answer,
                    "metrics": {
                        "tokens_generated": eval_count,
                        "tokens_per_second": tps,
                        "ttft": ttft,
                        "latency": latency
                    }
                }
                if sources_str:
                    msg_data["sources"] = sources_str

                st.session_state.messages.append(msg_data)

            except Exception as e:
                st.error(f"Error: {e}. Make sure Ollama is running.")