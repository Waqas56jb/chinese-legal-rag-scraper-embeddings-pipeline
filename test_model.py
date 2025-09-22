import os
import json
import torch
import torch.nn as nn
from typing import Dict, List


OUTPUT_DIR = os.path.join(os.getcwd(), "outputs_seq_models")


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


def load_best_model() -> tuple:
    # Find best model based on validation loss
    best_model_type = None
    best_loss = float('inf')
    best_fold = 0
    
    for model_type in ["rnn", "gru", "lstm"]:
        for fold in range(3):  # assuming 3 folds
            metrics_path = os.path.join(OUTPUT_DIR, f"{model_type}_fold{fold}", "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                val_loss = metrics.get("val_loss", float('inf'))
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_model_type = model_type
                    best_fold = fold
    
    if best_model_type is None:
        raise SystemExit("No trained models found")
    
    # Load vocab from training script (recreate it)
    import csv
    
    def read_texts_from_csv(path: str) -> List[str]:
        texts: List[str] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = (row.get("text") or "").strip()
                if t:
                    texts.append(t)
        return texts
    
    INPUT_CSV = os.path.join(os.getcwd(), "dataset", "dataset_clean.csv")
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
    
    # Create model
    model = CausalLM_RNN(len(vocab), 192, 256, rnn_type=best_model_type)
    
    return model, vocab, best_model_type


def encode(text: str, vocab: Dict[str, int]) -> List[int]:
    ids = []
    ids.append(vocab.get("<bos>", 0))
    for ch in list(text):
        ids.append(vocab.get(ch, vocab.get("<unk>", 0)))
    return ids


def decode(ids: List[int], vocab: Dict[str, int]) -> str:
    inv_vocab = {i: ch for ch, i in vocab.items()}
    # Remove special tokens from output
    special_tokens = {"<bos>", "<eos>", "<pad>", "<unk>"}
    filtered_ids = [i for i in ids if inv_vocab.get(i, "") not in special_tokens]
    return "".join(inv_vocab.get(i, "") for i in filtered_ids)


@torch.no_grad()
def generate_text(model: nn.Module, vocab: Dict[str, int], prompt: str, max_length: int = 100) -> str:
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


def main():
    print("Loading best trained model...")
    model, vocab, model_type = load_best_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Loaded {model_type.upper()} model")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Test prompts in Chinese
    test_prompts = [
        # here is 5 differnt questions for testing
        "中华人民共和国",
        "市场监管总局",
        "法律法规",
        "商务发展",
        "国家政策"
    ]
    
    print("\n=== Text Generation Tests ===")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        generated = generate_text(model, vocab, prompt, max_length=50)
        # Remove the prompt from output to show only generated part
        if generated.startswith(prompt):
            generated = generated[len(prompt):]
        print(f"Generated: {generated}")
        print("-" * 50)


if __name__ == "__main__":
    main()
