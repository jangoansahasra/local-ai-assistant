import requests
import json
from pydantic import BaseModel, ValidationError
from typing import List

API_URL = "http://localhost:8000/generate"

class MovieReview(BaseModel):
    title: str
    rating: float
    pros: List[str]
    cons: List[str]
    summary: str

SCHEMA_PROMPT = """Respond ONLY with valid JSON matching this exact schema. No extra text before or after the JSON.
{
    "title": "string",
    "rating": number between 1 and 10,
    "pros": ["string", "string"],
    "cons": ["string", "string"],
    "summary": "string"
}

"""

test_prompts = [
    "Write a review of the movie Inception.",
    "Write a review of the movie The Matrix.",
    "Write a review of the movie Interstellar.",
    "Write a review of the movie Titanic.",
    "Write a review of the movie The Dark Knight.",
]

def test_reliability(temperature: float, runs_per_prompt: int = 3):
    total = len(test_prompts) * runs_per_prompt
    passed = 0
    failed = 0

    print(f"\nTemperature: {temperature} | Total tests: {total}")
    print("-" * 50)

    for prompt in test_prompts:
        for run in range(runs_per_prompt):
            full_prompt = SCHEMA_PROMPT + prompt

            response = requests.post(API_URL, params={
                "prompt": full_prompt,
                "model": "llama3.2:3b",
                "temperature": temperature
            })

            raw = response.json()["response"]

            try:
                cleaned = raw.strip()
                start = cleaned.find("{")
                end = cleaned.rfind("}") + 1
                if start != -1 and end > start:
                    parsed = json.loads(cleaned[start:end])
                    MovieReview(**parsed)
                    passed += 1
                    status = "PASS"
                else:
                    failed += 1
                    status = "FAIL (no JSON)"
            except (json.JSONDecodeError, ValidationError) as e:
                failed += 1
                status = f"FAIL ({type(e).__name__})"

            movie = prompt.split("movie ")[-1].rstrip(".")
            print(f"  [{movie}] Run {run+1}: {status}")

    rate = (passed / total) * 100
    print(f"\n{'='*50}")
    print(f"Temperature {temperature} Results:")
    print(f"  Passed: {passed}/{total} ({rate:.1f}%)")
    print(f"  Failed: {failed}/{total}")
    print(f"{'='*50}")
    return rate

if __name__ == "__main__":
    print("=" * 50)
    print("RELIABILITY TEST: Temperature 0 vs 0.7")
    print("=" * 50)

    rate_0 = test_reliability(temperature=0.0)
    rate_7 = test_reliability(temperature=0.7)

    print(f"\n{'='*50}")
    print("FINAL COMPARISON")
    print(f"{'='*50}")
    print(f"  Temperature 0.0 : {rate_0:.1f}% reliability")
    print(f"  Temperature 0.7 : {rate_7:.1f}% reliability")
    print(f"{'='*50}")