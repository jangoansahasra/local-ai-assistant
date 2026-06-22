import requests

API_URL = "http://localhost:8000/generate"

test_prompts = [
    "What is machine learning?",
    "Explain the difference between Python and JavaScript.",
    "What are the benefits of exercise?",
    "Summarize the history of the internet in 3 sentences.",
    "What is the capital of France and why is it famous?",
]

MODELS = ["llama3.2:3b", "llama3.2:3b-instruct-q4_K_M"]  # Change tag if needed

for model in MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {model}")
    print(f"{'='*60}")

    results = []
    for i, prompt in enumerate(test_prompts):
        print(f"  [{i+1}/{len(test_prompts)}] {prompt[:45]}...")
        try:
            resp = requests.post(API_URL, params={
                "prompt": prompt, "model": model, "temperature": 0.0
            }, timeout=120)
            data = resp.json()
            m = data["metrics"]
            results.append(m)
            print(f"    TPS: {m['tokens_per_second']} | Latency: {m['total_latency_sec']}s")
            print(f"    Response preview: {data['response'][:100]}...")
        except Exception as e:
            print(f"    ERROR: {e}")

    if results:
        avg_tps = sum(r["tokens_per_second"] for r in results) / len(results)
        avg_lat = sum(r["total_latency_sec"] for r in results) / len(results)
        print(f"\n  AVG TPS: {avg_tps:.2f} | AVG Latency: {avg_lat:.2f}s")