import requests
import json
import time
import subprocess

API_URL = "http://localhost:8000/generate"

MODELS = ["llama3.2:3b", "phi4-mini:latest", "mistral:7b"]

test_prompts = [
    "What is machine learning?",
    "Explain the difference between Python and JavaScript.",
    "What are the benefits of exercise?",
    "Summarize the history of the internet in 3 sentences.",
    "What is the capital of France and why is it famous?",
    "Explain gravity to a 10 year old.",
    "What are three common data structures?",
    "Why is the sky blue?",
    "What is an API and why is it useful?",
    "Describe the water cycle in simple terms.",
]

def get_memory_usage(model_name):
    """Get approximate memory usage by checking Ollama process."""
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True
        )
        total_mem = 0
        for line in result.stdout.split("\n"):
            if "ollama" in line.lower():
                parts = line.split()
                if len(parts) > 5:
                    total_mem += float(parts[5]) / 1024  # Convert KB to MB
        return round(total_mem, 1)
    except:
        return 0

def benchmark_model(model_name):
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model_name}")
    print(f"{'='*60}")

    # Warm up the model with a short prompt
    print("Warming up model...")
    try:
        requests.post(API_URL, params={
            "prompt": "Hi",
            "model": model_name,
            "temperature": 0.0
        }, timeout=120)
    except:
        print(f"ERROR: Could not load {model_name}. Skipping.")
        return None

    mem_usage = get_memory_usage(model_name)
    print(f"Memory usage (approx): {mem_usage} MB")

    results = []
    for i, prompt in enumerate(test_prompts):
        print(f"  [{i+1}/{len(test_prompts)}] {prompt[:45]}...")
        try:
            response = requests.post(API_URL, params={
                "prompt": prompt,
                "model": model_name,
                "temperature": 0.0
            }, timeout=120)
            data = response.json()
            metrics = data["metrics"]
            results.append(metrics)
            print(f"    Tokens: {metrics['tokens_generated']} | "
                  f"TPS: {metrics['tokens_per_second']} | "
                  f"TTFT: {metrics['time_to_first_token_sec']}s | "
                  f"Latency: {metrics['total_latency_sec']}s")
        except Exception as e:
            print(f"    ERROR: {e}")

    if results:
        avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
        avg_ttft = sum(r["time_to_first_token_sec"] for r in results) / len(results)
        avg_latency = sum(r["total_latency_sec"] for r in results) / len(results)
        avg_tokens = sum(r["tokens_generated"] for r in results) / len(results)

        summary = {
            "model": model_name,
            "memory_mb": mem_usage,
            "avg_tokens": round(avg_tokens, 1),
            "avg_tps": round(avg_tps, 2),
            "avg_ttft": round(avg_ttft, 4),
            "avg_latency": round(avg_latency, 4),
        }

        print(f"\n  SUMMARY for {model_name}:")
        print(f"    Memory        : {mem_usage} MB")
        print(f"    Avg Tokens    : {summary['avg_tokens']}")
        print(f"    Avg TPS       : {summary['avg_tps']}")
        print(f"    Avg TTFT      : {summary['avg_ttft']}s")
        print(f"    Avg Latency   : {summary['avg_latency']}s")

        return summary
    return None

if __name__ == "__main__":
    print("=" * 60)
    print("MODEL COMPARISON BENCHMARK")
    print(f"Hardware: Apple M4, 16GB RAM")
    print(f"Prompts: {len(test_prompts)}")
    print("=" * 60)

    all_results = []
    for model in MODELS:
        result = benchmark_model(model)
        if result:
            all_results.append(result)

    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"{'Model':<22} {'Memory':>8} {'Avg TPS':>10} {'Avg TTFT':>10} {'Avg Latency':>12}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model']:<22} {r['memory_mb']:>7} MB {r['avg_tps']:>9} {r['avg_ttft']:>9}s {r['avg_latency']:>11}s")
    print(f"{'='*70}")