import requests
import json
from pydantic import BaseModel, ValidationError
from typing import List, Optional

API_URL = "http://localhost:8000/generate"

# Define the schema we want the model to follow
class MovieReview(BaseModel):
    title: str
    rating: float
    pros: List[str]
    cons: List[str]
    summary: str

def get_structured_response(prompt: str, model: str = "llama3.2:3b", temperature: float = 0.0, max_retries: int = 1):
    schema_instruction = """
Respond ONLY with valid JSON matching this exact schema. No extra text before or after the JSON.
{
    "title": "string",
    "rating": number between 1 and 10,
    "pros": ["string", "string"],
    "cons": ["string", "string"],
    "summary": "string"
}
"""
    full_prompt = f"{schema_instruction}\n\n{prompt}"
    
    for attempt in range(max_retries + 1):
        print(f"\nAttempt {attempt + 1}...")
        
        response = requests.post(API_URL, params={
            "prompt": full_prompt,
            "model": model,
            "temperature": temperature
        })
        
        raw = response.json()["response"]
        
        # Try to extract JSON from the response
        try:
            # Clean up common issues
            cleaned = raw.strip()
            # Find JSON boundaries
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start != -1 and end > start:
                json_str = cleaned[start:end]
                parsed = json.loads(json_str)
                result = MovieReview(**parsed)
                print("Validation PASSED!")
                return result
            else:
                print("No JSON found in response")
                print(f"Raw output: {cleaned[:200]}...")
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Raw output: {raw[:200]}...")
        except ValidationError as e:
            print(f"Pydantic validation error: {e}")
    
    print("\nAll attempts failed. Returning None.")
    return None


if __name__ == "__main__":
    # Test 1: Temperature 0 (deterministic)
    print("=" * 60)
    print("TEST 1: Temperature 0 (Deterministic)")
    print("=" * 60)
    result = get_structured_response(
        "Write a review of the movie Inception.",
        temperature=0.0
    )
    if result:
        print(f"\nTitle: {result.title}")
        print(f"Rating: {result.rating}")
        print(f"Pros: {result.pros}")
        print(f"Cons: {result.cons}")
        print(f"Summary: {result.summary}")

    # Test 2: Temperature 0.7 (creative)
    print("\n" + "=" * 60)
    print("TEST 2: Temperature 0.7 (Creative)")
    print("=" * 60)
    result2 = get_structured_response(
        "Write a review of the movie Inception.",
        temperature=0.7
    )
    if result2:
        print(f"\nTitle: {result2.title}")
        print(f"Rating: {result2.rating}")
        print(f"Pros: {result2.pros}")
        print(f"Cons: {result2.cons}")
        print(f"Summary: {result2.summary}")