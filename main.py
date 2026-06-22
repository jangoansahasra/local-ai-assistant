import time
import requests
from fastapi import FastAPI

app = FastAPI(title="Local AI Assistant")

OLLAMA_URL = "http://localhost:11434/api/generate"

@app.post("/generate")
def generate(prompt: str, model: str = "llama3.2:3b", temperature: float = 0.7):
    start_time = time.time()

    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    result = response.json()

    total_time = time.time() - start_time
    eval_count = result.get("eval_count", 0)
    eval_duration = result.get("eval_duration", 1)
    prompt_eval_duration = result.get("prompt_eval_duration", 0)

    tokens_per_second = eval_count / (eval_duration / 1e9) if eval_duration else 0
    ttft = prompt_eval_duration / 1e9

    return {
        "response": result.get("response", ""),
        "metrics": {
            "tokens_generated": eval_count,
            "tokens_per_second": round(tokens_per_second, 2),
            "time_to_first_token_sec": round(ttft, 4),
            "total_latency_sec": round(total_time, 4)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
