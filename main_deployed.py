from dotenv import load_dotenv
load_dotenv()

import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware


HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/ml6team/gpt-2-medium-conditional-quote-generator"
print("Hugging Face Token:", os.environ.get("HF_TOKEN"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://thara-3201.github.io", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuoteRequest(BaseModel):
    topic: str
    num_quotes: int = 1

class QuoteResponse(BaseModel):
    topic: str
    quotes: List[str]

def query_huggingface(prompt: str):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 60}
    }
    print("Sending request to Hugging Face...")
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json()
        print("Response received!")
        return result[0]['generated_text']
    except Exception as e:
        print(f"Request failed: {e}")
        return "Quote generation failed."

@app.post("/generate", response_model=QuoteResponse)
async def generate_quotes(data: QuoteRequest):
    results = []
    for _ in range(data.num_quotes):
        prompt = f"Topics: {data.topic} | Related Quote:"
        output = query_huggingface(prompt)
        quote = output.split("Related Quote:")[-1].strip()
        results.append(quote)
    return {"topic": data.topic, "quotes": results}
