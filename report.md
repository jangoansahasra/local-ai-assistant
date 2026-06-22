# Local AI Assistant — Technical Report

**Date:** April 21, 2026  
**Hardware:** Apple M4, 16GB Unified Memory, MacBook Air  
**Runtime:** Ollama v0.21.0  
**Framework:** FastAPI + Pydantic + Streamlit + ChromaDB

---

## Phase 1: Baseline Benchmarking

### Setup
- Installed Ollama via Homebrew on macOS
- Built a FastAPI wrapper around the Ollama API to programmatically interact with models
- Measured tokens per second (TPS), time to first token (TTFT), and total latency

### Llama 3.2 3B Baseline Results (10 prompts, averaged)
| Metric               | Value     |
|-----------------------|-----------|
| Avg Tokens Generated  | 331.1     |
| Avg Tokens/Second     | 45.14     |
| Avg TTFT              | 0.1051s   |
| Avg Total Latency     | 7.64s     |

### Key Takeaway
~45 tokens/second on the M4 is essentially real-time for a reader. The TTFT of ~105ms means responses feel instant. For many use cases, this eliminates the need for a cloud API entirely.

---

## Phase 2: Structured Output & Reliability

### JSON Schema Enforcement
- Defined a `MovieReview` schema using Pydantic (title, rating, pros, cons, summary)
- Model was prompted to return only valid JSON
- Built retry logic that re-prompts on failure and fails gracefully after 2 attempts

### Temperature Comparison (5 prompts x 3 runs each)
| Temperature | Reliability |
|-------------|-------------|
| 0.0         | 73.3% (11/15) |
| 0.7         | 80.0% (12/15) |

### Key Takeaway
Temperature 0.7 was slightly more reliable than 0.0 in this test. All failures were "no JSON" errors where the model added explanatory text. For production use, combining schema prompting + validation + retry is essential.

---

## Phase 3: Model Comparison

### Three-Model Benchmark (10 prompts each, Temperature 0.0)
| Model              | Disk  | Memory    | Avg TPS | Avg TTFT | Avg Latency |
|--------------------|-------|-----------|---------|----------|-------------|
| Llama 3.2 3B       | 2.0 GB| 2,315 MB  | 45.06   | 0.077s   | 8.16s       |
| Phi-4 Mini         | 2.5 GB| 4,842 MB  | 34.71   | 0.118s   | 9.47s       |
| Mistral 7B         | 4.4 GB| 5,067 MB  | 18.52   | 0.393s   | 18.56s      |

### Quantization Test (Llama 3.2 3B, 5 prompts)
| Variant              | Avg TPS | Avg Latency |
|----------------------|---------|-------------|
| Default              | 45.08   | 9.62s       |
| Quantized (Q4_K_M)  | 44.66   | 8.82s       |

### Key Takeaways
1. Llama 3.2 3B is the clear winner for the M4 MacBook Air — fastest TPS, lowest memory, lowest latency.
2. Mistral 7B is 2.4x slower and uses over double the memory.
3. Quantization had minimal impact on the 3B model since it already fits comfortably in memory.

---

## Phase 4: Retrieval-Augmented Generation (RAG)

### Architecture
- **Embedding Model:** all-MiniLM-L6-v2 (sentence-transformers) — runs locally, ~90MB
- **Vector Database:** ChromaDB with persistent storage and cosine similarity
- **Document Support:** PDF and TXT files
- **Chunking Strategy:** 50-word chunks with 10-word overlap for granular retrieval

### How It Works
1. **Ingestion:** Documents are loaded, split into overlapping chunks, and each chunk is converted into a 384-dimensional vector using the MiniLM embedding model.
2. **Storage:** Vectors are stored in ChromaDB with metadata (source filename, chunk index) for source attribution.
3. **Retrieval:** When a user asks a question, the query is embedded and the top 3 most similar chunks are retrieved using cosine similarity.
4. **Generation:** Retrieved chunks are injected into the prompt as context, and the LLM generates an answer grounded in the documents.
5. **Source Attribution:** Each response shows which documents were used to generate the answer.

### RAG Performance
- Correctly retrieves relevant chunks from the right source documents
- ML questions pull from ml_basics.txt, cloud questions pull from cloud_computing.txt
- When asked about topics not in the documents, the model correctly responds with "I don't have enough information"
- Retrieval adds minimal latency (~50-100ms) on top of generation time

### Key Takeaway
RAG transforms the assistant from a general-purpose chatbot into a knowledge-grounded system that can answer questions about specific documents. The entire pipeline — embedding, retrieval, and generation — runs 100% locally with no external API calls.

---

## Streamlit Web Interface

### Features
- **Model Selector:** Switch between Llama 3.2 3B, Phi-4 Mini, and Mistral 7B
- **Temperature Control:** Adjustable slider from 0.0 to 1.0
- **RAG Toggle:** Enable/disable document-grounded responses
- **Document Upload:** Drag-and-drop PDF and TXT files directly in the browser
- **Live Metrics:** Tokens generated, tokens/second, TTFT, and latency per response
- **Source Attribution:** Shows which documents were used when RAG is enabled
- **Chat History:** Maintains conversation context within a session

---

## Project Structure

local-ai-assistant/
├── main.py                 # FastAPI server wrapping Ollama
├── app.py                  # Streamlit web UI with RAG integration
├── rag.py                  # RAG engine (document ingestion + vector search)
├── rag_chat.py             # CLI-based RAG chat interface
├── benchmark.py            # Baseline benchmarking (10 prompts)
├── structured.py           # JSON schema + Pydantic validation
├── reliability_test.py     # Temperature 0 vs 0.7 comparison
├── model_comparison.py     # 3-model benchmark
├── quantization_test.py    # Quantization speed test
├── generate_report.py      # This report generator
├── report.md               # This report
├── requirements.txt        # Python dependencies
├── documents/              # Knowledge base (PDF and TXT files)
├── chroma_db/              # ChromaDB vector database (persistent)
└── venv/                   # Python virtual environment

---

## Recommendations

- **For speed-critical local tasks:** Llama 3.2 3B — unbeatable speed-to-quality ratio on Apple Silicon
- **For higher quality when latency is acceptable:** Phi-4 Mini — reasonable tradeoff with better reasoning
- **For production structured output:** Always use Pydantic validation + retry logic regardless of model
- **For document Q&A:** RAG with ChromaDB and sentence-transformers provides accurate, source-attributed answers with minimal overhead
- **Cloud vs Local tradeoff:** At 45 TPS with ~100ms TTFT, local inference on M4 is competitive with cloud APIs for single-user workloads, with the added benefits of zero cost per token, full privacy, and no rate limits

---

## How to Run the Project

### Prerequisites
- macOS with Homebrew installed
- Python 3.10+
- At least 10GB free disk space for models

### Step 1: Install Ollama and Pull Models
```bash
brew install ollama
brew services start ollama
ollama pull llama3.2:3b
ollama pull phi4-mini
ollama pull mistral:7b
```

### Step 2: Set Up the Python Environment
```bash
cd ~/local-ai-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Web UI
```bash
source venv/bin/activate
streamlit run app.py
```
This opens a browser at http://localhost:8501 with the full chat interface, model switching, and RAG support.

### Step 4: Run the FastAPI Server (Optional, for API Access)
```bash
source venv/bin/activate
python main.py
```
The API will be available at http://localhost:8000. Test with:
```bash
curl -X POST "http://localhost:8000/generate?prompt=Hello"
```

### Step 5: Run Benchmarks
```bash
python benchmark.py           # Baseline performance
python reliability_test.py    # Temperature comparison
python model_comparison.py    # 3-model comparison (start FastAPI first)
python quantization_test.py   # Quantization test (start FastAPI first)
```
Note: model_comparison.py and quantization_test.py require the FastAPI server to be running.

### Step 6: Add Documents for RAG
Place PDF or TXT files in the `documents/` folder, then run:
```bash
python rag.py
```
Or upload files directly through the Streamlit UI sidebar.

### Managing Models
```bash
ollama list                   # See installed models
ollama pull <model-name>      # Download a new model
ollama rm <model-name>        # Delete a model to free disk space
brew services stop ollama     # Stop Ollama when not in use
```


