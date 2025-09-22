import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import torch

# Reuse model utilities from test_model.py
from test_model import load_best_model, generate_text

app = FastAPI(title="Chinese Legal RAG - Live Test API")


class RunRequest(BaseModel):
    text: str
    max_length: int = 50


# Load model once at startup
print("Loading best trained model for API...")
model, vocab, model_type = load_best_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"API using {model_type.upper()} model with vocab size {len(vocab)}")


@app.post("/run")
def run(req: RunRequest):
    prompt = (req.text or "").strip()
    if not prompt:
        return {"error": "text is required"}

    generated = generate_text(model, vocab, prompt, max_length=req.max_length)
    if generated.startswith(prompt):
        generated = generated[len(prompt):]
    return {
        "prompt": prompt,
        "generated": generated,
        "model": model_type,
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)


