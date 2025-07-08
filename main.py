from dotenv import load_dotenv
load_dotenv()

import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

# Use a pipeline as a high-level helper
from transformers import pipeline

class PromptInput(BaseModel):
    prompt: str

app = FastAPI()
print("Available routes:")
for route in app.routes:
    print(route.path)

# Global placeholder
pipe = None

@app.post("/generate_local")
def generate_quotes(data: PromptInput):
    global pipe
    if pipe is None:
        pipe = pipeline("text-generation", model="tharapearlly/phi2-affirmations")
    output = pipe(data.prompt, max_new_tokens=60, do_sample=True, temperature=0.9)
    return {"output": output[0]["generated_text"]}


HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/tharapearlly/phi2-affirmations"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

print("Hugging Face Token:", os.environ.get("HF_TOKEN"))


@app.post("/generate")
def generate_affirmation(data: PromptInput):
    payload = {
        "inputs": data.prompt,
        "parameters": {
            "max_new_tokens": 40,
            "temperature": 0.7
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)

    # 👇 Add this to debug
    print("Status Code:", response.status_code)
    print("Response Text:", response.text)

    # Try to parse JSON safely
    try:
        return response.json()
    except Exception as e:
        return {"error": str(e), "raw": response.text}
