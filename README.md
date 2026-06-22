# 🤖 Local AI Assistant

A fully local AI assistant with RAG (Retrieval-Augmented Generation) that runs open-source language models entirely on your machine — no cloud, no API keys, no cost per token, and complete data privacy.

Built on Apple M4 (16GB RAM) using Ollama, FastAPI, ChromaDB, and Streamlit.

## Features

- **Multi-Model Support** — Switch between Llama 3.2 3B, Phi-4 Mini, and Mistral 7B
- **RAG Pipeline** — Upload PDF/TXT documents and chat with them using semantic search
- **Benchmarking Suite** — Measure TPS, TTFT, latency, and memory across models
- **Structured Output** — JSON schema enforcement with Pydantic validation and retry logic
- **Streamlit Web UI** — Chat interface with model switching, RAG toggle, and live metrics

## Performance Benchmarks

| Model | Memory | Tokens/sec | TTFT | Avg Latency |
|-------|--------|-----------|------|-------------|
| Llama 3.2 3B | 2,315 MB | 45.06 | 0.077s | 8.16s |
| Phi-4 Mini | 4,842 MB | 34.71 | 0.118s | 9.47s |
| Mistral 7B | 5,067 MB | 18.52 | 0.393s | 18.56s |

## Tech Stack

- **Runtime:** Ollama
- **Backend:** FastAPI
- **Validation:** Pydantic
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB:** ChromaDB
- **Frontend:** Streamlit
- **Language:** Python

## Quick Start

### Prerequisites
- macOS with Homebrew
- Python 3.10+
- 10GB+ free disk space

### Setup
```bash
# Install Ollama and pull models
brew install ollama
brew services start ollama
ollama pull llama3.2:3b
ollama pull phi4-mini
ollama pull mistral:7b

# Set up Python environment
cd local-ai-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Project Structure
├── app.py                  # Streamlit UI with RAG

├── main.py                 # FastAPI server

├── rag.py                  # RAG engine (ingestion + search)

├── rag_chat.py             # CLI RAG chat

├── benchmark.py            # Baseline benchmarking

├── structured.py           # JSON schema + Pydantic validation

├── reliability_test.py     # Temperature comparison

├── model_comparison.py     # 3-model benchmark

├── quantization_test.py    # Quantization test

├── generate_report.py      # Report generator

├── report.md               # Technical report

├── requirements.txt        # Dependencies

└── documents/              # Knowledge base (PDF/TXT)

## Key Findings

- Llama 3.2 3B achieves 45 tok/s with 105ms TTFT — real-time for end users
- Llama 3.2 3B is 2.4x faster than Mistral 7B while using 53% less memory
- Quantization (Q4_K_M) had minimal speed impact on the 3B model
- Temperature 0.7 achieved 80% JSON schema reliability vs 73.3% at Temperature 0
- RAG retrieval adds only ~50-100ms latency with 100% accuracy across test domains