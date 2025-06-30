from fastapi import FastAPI
from pydantic import BaseModel
from typing import  List
from fastapi.middleware.cors import CORSMiddleware

from transformers import pipeline
from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

# Load model once on startup
quote_pipe = pipeline("text-generation", model="ml6team/gpt-2-medium-conditional-quote-generator")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://thara-3201.github.io"],
    allow_methods=["*"],
    allow_headers=["*"],
)




class QuoteRequest(BaseModel):
    topic: str
    num_quotes: int = 1

# Response schema (optional, for docs)
class QuoteResponse(BaseModel):
    topic: str
    quotes: List[str]

@app.post("/generate", response_model=QuoteResponse)
async def generate_quotes(data: QuoteRequest):
    results = []
    # Generate the quotes
    for i in range(data.num_quotes):
        prompt = f"Topics: {data.topic} | Related Quote:"
        output = quote_pipe(prompt, max_new_tokens=60, do_sample=True, temperature=0.9)[0]['generated_text']
        # Optional: Clean up the output
        quote = output.split("Related Quote:")[-1].strip()
        results.append(quote)
    return {"topic": data.topic, "quotes": results}


# Get user input
#topic = input("Enter the topic (e.g., success, life, motivation): ").strip()
#num_quotes = int(input("Enter the number of quotes to generate: "))
# print(f"\nGenerating {num_quotes} quote(s) about '{topic}'...\n")
# print(f"{i + 1}. {quote}\n")