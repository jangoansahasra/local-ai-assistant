import requests
import json
import time

API_URL = "http://localhost:8000/generate"

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
    "Describe the water cycle in simple terms."
]

def run_benchmark(model="llama3.2:3b", temperature=0.7):
    results = []
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model} | Temperature: {temperature}")
    print(f"{'='*60}\n")

    for i, prompt in enumerate(test_prompts):
        print(f"[{i+1}/{len(test_prompts)}] {prompt[:50]}...")
        
        try:
            response = requests.post(API_URL, params={
                "prompt": prompt,
                "model": model,
                "temperature": temperature
            })
            data = response.json()
            metrics = data["metrics"]
            results.append(metrics)
            print(f"  Tokens: {metrics['tokens_generated']} | "
                  f"TPS: {metrics['tokens_per_second']} | "
                  f"TTFT: {metrics['time_to_first_token_sec']}s | "
                  f"Latency: {metrics['total_latency_sec']}s")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Calculate averages
    if results:
        avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
        avg_ttft = sum(r["time_to_first_token_sec"] for r in results) / len(results)
        avg_latency = sum(r["total_latency_sec"] for r in results) / len(results)
        avg_tokens = sum(r["tokens_generated"] for r in results) / len(results)

        print(f"\n{'='*60}")
        print("AVERAGE RESULTS")
        print(f"{'='*60}")
        print(f"  Avg Tokens Generated : {avg_tokens:.1f}")
        print(f"  Avg Tokens/Second    : {avg_tps:.2f}")
        print(f"  Avg TTFT             : {avg_ttft:.4f}s")
        print(f"  Avg Total Latency    : {avg_latency:.4f}s")
        print(f"{'='*60}\n")

    return results

if __name__ == "__main__":
    run_benchmark()