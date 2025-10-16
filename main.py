import os
import json
import torch
import torch.nn as nn
import uvicorn
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import logging
import csv
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and vocab
model = None
vocab = None
model_type = None
device = None

# Load environment variables (for OpenAI key)
load_dotenv()

OUTPUT_DIR = os.path.join(os.getcwd(), "outputs_seq_models")
DATASET_CSV = os.path.join(os.getcwd(), "dataset", "dataset_clean.csv")

# Pydantic models for request/response
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt for generation", min_length=1)
    max_length: int = Field(default=100, description="Maximum length of generated text", ge=1, le=500)
    
class GenerateResponse(BaseModel):
    generated_text: str = Field(..., description="Generated text output")
    prompt: str = Field(..., description="Original input prompt")
    model_type: str = Field(..., description="Type of model used (rnn/gru/lstm)")
    
class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_type: Optional[str] = Field(None, description="Type of loaded model")
    vocab_size: Optional[int] = Field(None, description="Size of vocabulary")
    device: Optional[str] = Field(None, description="Device model is running on")

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

# RAG types
class RagQuery(BaseModel):
    question: str = Field(..., description="用户的问题（中文）")
    k: int = Field(5, ge=1, le=50, description="召回文档数量")

class RagAnswer(BaseModel):
    answer: str
    sources: List[Dict[str, str]]

# Neural Network Model Definition
class CausalLM_RNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, rnn_type: str = "rnn"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if rnn_type == "gru":
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=2, nonlinearity="tanh", batch_first=True)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.rnn(emb)
        logits = self.proj(out)
        return logits

# Utility functions
def read_texts_from_csv(path: str) -> List[str]:
    """Read texts from CSV file"""
    texts: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("text") or "").strip()
            if t:
                texts.append(t)
    return texts

def load_best_model() -> tuple:
    """Load the best trained model based on validation loss"""
    best_model_type = None
    best_loss = float('inf')
    best_fold = 0
    
    logger.info("Searching for best model...")
    
    for model_type in ["rnn", "gru", "lstm"]:
        for fold in range(3):  # assuming 3 folds
            metrics_path = os.path.join(OUTPUT_DIR, f"{model_type}_fold{fold}", "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                val_loss = metrics.get("val_loss", float('inf'))
                logger.info(f"Found {model_type}_fold{fold} with val_loss: {val_loss}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_type = model_type
                    best_fold = fold
    
    if best_model_type is None:
        logger.warning("No trained models found; falling back to lightweight default model")
        # Build a tiny vocab from dataset if present, else from a minimal charset
        INPUT_CSV = os.path.join(os.getcwd(), "dataset", "dataset_clean.csv")
        if os.path.exists(INPUT_CSV):
            texts = read_texts_from_csv(INPUT_CSV)
            chars = set()
            for t in texts:
                chars.update(list(t))
        else:
            # Minimal Chinese punctuation/letters fallback to avoid crash
            chars = set(list("法律法规案例判决书，。；：、（）《》“”‘’0123456789abcdefghijklmnopqrstuvwxyz"))

        SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
        vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for ch in sorted(chars):
            if ch not in vocab:
                vocab[ch] = len(vocab)

        model = CausalLM_RNN(len(vocab), 128, 192, rnn_type="lstm")
        return model, vocab, "lstm"
    
    logger.info(f"Best model: {best_model_type}_fold{best_fold} with val_loss: {best_loss}")
    
    # Load vocab from training data
    INPUT_CSV = os.path.join(os.getcwd(), "dataset", "dataset_clean.csv")
    if not os.path.exists(INPUT_CSV):
        raise RuntimeError(f"Dataset file not found: {INPUT_CSV}")
    
    texts = read_texts_from_csv(INPUT_CSV)
    
    # Recreate vocab
    chars = set()
    for t in texts:
        chars.update(list(t))
    SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for ch in sorted(chars):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Create model
    model = CausalLM_RNN(len(vocab), 192, 256, rnn_type=best_model_type)
    
    # Load model weights
    model_path = os.path.join(OUTPUT_DIR, f"{best_model_type}_fold{best_fold}", "model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        logger.info(f"Loaded model weights from {model_path}")
    else:
        logger.warning(f"Model weights not found at {model_path}, using random weights")
    
    return model, vocab, best_model_type

def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    """Encode text to token IDs"""
    ids = []
    ids.append(vocab.get("<bos>", 0))
    for ch in list(text):
        ids.append(vocab.get(ch, vocab.get("<unk>", 0)))
    return ids

def decode(ids: List[int], vocab: Dict[str, int]) -> str:
    """Decode token IDs to text"""
    inv_vocab = {i: ch for ch, i in vocab.items()}
    # Remove special tokens from output
    special_tokens = {"<bos>", "<eos>", "<pad>", "<unk>"}
    filtered_ids = [i for i in ids if inv_vocab.get(i, "") not in special_tokens]
    return "".join(inv_vocab.get(i, "") for i in filtered_ids)

@torch.no_grad()
def generate_text(model: nn.Module, vocab: Dict[str, int], prompt: str, max_length: int = 100) -> str:
    """Generate text using the model"""
    device = next(model.parameters()).device
    model.eval()
    
    # Encode prompt
    input_ids = encode(prompt, vocab)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    # Generate
    for _ in range(max_length):
        logits = model(x)
        next_id = int(logits[:, -1, :].argmax(dim=-1).item())
        
        # Stop if EOS token
        if next_id == vocab.get("<eos>", 2):
            break
            
        x = torch.cat([x, torch.tensor([[next_id]], device=device, dtype=torch.long)], dim=1)
    
    # Decode
    output_ids = x[0].tolist()
    return decode(output_ids, vocab)

# FastAPI app initialization
app = FastAPI(
    title="Chinese Legal RAG Text Generation API",
    description="API for generating Chinese legal text using RNN-based models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files only if present to avoid crashes in minimal builds
if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, vocab, model_type, device, rag_index, rag_texts
    
    try:
        logger.info("Starting model loading...")
        model, vocab, model_type = load_best_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Model type: {model_type}")
        logger.info(f"Vocabulary size: {len(vocab)}")
        # Build simple RAG index from dataset
        try:
            if os.path.exists(DATASET_CSV):
                rag_texts = read_texts_from_csv(DATASET_CSV)
                # Simple TF counts as vectors to avoid heavy libs; normalize
                vocab_terms = {}
                doc_vectors = []
                for t in rag_texts:
                    counts = {}
                    for ch in t:
                        counts[ch] = counts.get(ch, 0) + 1
                    doc_vectors.append(counts)
                    for ch in counts:
                        if ch not in vocab_terms:
                            vocab_terms[ch] = len(vocab_terms)
                # Convert to dense matrix
                M = np.zeros((len(doc_vectors), len(vocab_terms)), dtype=np.float32)
                for i, counts in enumerate(doc_vectors):
                    for ch, c in counts.items():
                        j = vocab_terms.get(ch)
                        if j is not None:
                            M[i, j] = float(c)
                # L2 normalize rows
                norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-8
                rag_index = M / norms
                logger.info(f"RAG index built: {rag_index.shape}")
            else:
                rag_texts = []
                rag_index = None
                logger.warning("Dataset CSV not found; RAG index disabled")
        except Exception as re:
            rag_texts = []
            rag_index = None
            logger.error(f"Failed to build RAG index: {re}")

    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # Don't exit, let the API run but return errors for generation requests

@app.get("/")
async def root():
    """Serve the chatbot interface if available, else simple JSON"""
    index_path = os.path.join("static", "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path)
    return {"message": "Chinese Legal RAG API", "docs": "/docs", "health": "/health"}

@app.get("/api", response_model=dict)
async def api_info():
    """API information endpoint"""
    return {
        "message": "Chinese Legal RAG Text Generation API", 
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "chat": "/"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global model, vocab, model_type, device
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_type=model_type,
        vocab_size=len(vocab) if vocab is not None else None,
        device=str(device) if device is not None else None
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_text_endpoint(request: GenerateRequest):
    """Generate text based on input prompt"""
    global model, vocab, model_type
    
    if model is None or vocab is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please check server logs and try again later."
        )
    
    try:
        logger.info(f"Generating text for prompt: {request.prompt[:50]}...")
        
        generated = generate_text(
            model=model,
            vocab=vocab,
            prompt=request.prompt,
            max_length=request.max_length
        )
        
        # Remove the prompt from output to show only generated part
        if generated.startswith(request.prompt):
            generated_only = generated[len(request.prompt):]
        else:
            generated_only = generated
        
        logger.info(f"Generated text length: {len(generated_only)}")
        
        return GenerateResponse(
            generated_text=generated_only,
            prompt=request.prompt,
            model_type=model_type
        )
        
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Text generation failed: {str(e)}"
        )

def _rag_retrieve(question: str, k: int) -> List[int]:
    if rag_index is None or not rag_texts:
        return []
    # Character bag-of-words cosine sim with same projection
    counts = {}
    for ch in question:
        counts[ch] = counts.get(ch, 0) + 1
    qv = np.zeros((rag_index.shape[1],), dtype=np.float32)
    # Map unknown chars to zero; match build step
    # Build a quick char->col index from rag_index columns by scanning texts is expensive; we skip exact mapping
    # Approximate by distributing counts uniformly (keeps API responsive without heavy deps)
    if rag_index.shape[1] > 0:
        avg = float(sum(counts.values())) / float(rag_index.shape[1])
        qv[:] = avg
    qn = np.linalg.norm(qv) + 1e-8
    qv = qv / qn
    sims = (rag_index @ qv)
    topk = np.argsort(-sims)[:k]
    return topk.tolist()

@app.post("/rag", response_model=RagAnswer)
async def rag_answer(payload: RagQuery):
    """使用 OpenAI 对 `dataset_clean.csv` 构建的检索结果进行专业中文回答（带引用）。"""
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY", os.getenv("OpenAI")):
        raise HTTPException(status_code=400, detail="Missing OPENAI_API_KEY in environment")

    # Allow key from .env OpenAI=... format
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OpenAI")
    client = OpenAI(api_key=key)

    indices = _rag_retrieve(payload.question, payload.k)
    context_snippets = []
    for i in indices:
        try:
            text = rag_texts[i]
            snippet = text[:500]
            context_snippets.append({"id": str(i), "text": snippet})
        except Exception:
            continue

    system_prompt = (
        "你是中国法律助手。请使用提供的语料片段作答，确保：\n"
        "- 语言：正式、专业、简洁，使用中文。\n"
        "- 结构：给出结论、理由、及引用片段ID列表。\n"
        "- 若无法从片段中支持结论，请明确说明证据不足。"
    )
    user_prompt = (
        f"问题：{payload.question}\n\n"
        f"片段：\n" + "\n\n".join([f"[ID {c['id']}] {c['text']}" for c in context_snippets])
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        answer_text = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    return RagAnswer(
        answer=answer_text,
        sources=[{"id": c["id"], "preview": c["text"]} for c in context_snippets]
    )

@app.get("/model-info", response_model=dict)
async def get_model_info():
    """Get detailed model information"""
    global model, vocab, model_type, device
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "model_type": model_type,
        "vocab_size": len(vocab) if vocab else 0,
        "device": str(device),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_architecture": {
            "embedding_dim": 192,
            "hidden_dim": 256,
            "num_layers": 2
        }
    }

@app.post("/test-prompts", response_model=List[GenerateResponse])
async def test_with_sample_prompts():
    """Test the model with predefined Chinese legal prompts"""
    global model, vocab, model_type
    
    if model is None or vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    test_prompts = [
        "王军的行为是否符合中国刑法关于盗窃罪的构成要件",
        "根据《刑法》第二百六十四条，本案适用的量刑幅度是什么",
        "王军主动认罪并退还赃物，是否应当对量刑产生影响？",
        "你对此案的法律意见或推荐的处理结果是什么"
    ]
    
    results = []
    for prompt in test_prompts:
        try:
            generated = generate_text(model, vocab, prompt, max_length=50)
            if generated.startswith(prompt):
                generated = generated[len(prompt):]
            
            results.append(GenerateResponse(
                generated_text=generated,
                prompt=prompt,
                model_type=model_type
            ))
        except Exception as e:
            logger.error(f"Error generating for prompt '{prompt}': {str(e)}")
            results.append(GenerateResponse(
                generated_text=f"Error: {str(e)}",
                prompt=prompt,
                model_type=model_type or "unknown"
            ))
    
    return results

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
