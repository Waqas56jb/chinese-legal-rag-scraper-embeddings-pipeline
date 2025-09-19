import os
import csv
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns


INPUT_CSV = os.path.join(os.getcwd(), "dataset", "dataset_clean.csv")
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs_seq_models")

# Hyperparameters via env
KFOLDS = int(os.environ.get("KFOLDS", "3"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
LR = float(os.environ.get("LR", "1e-3"))
EMBED_DIM = int(os.environ.get("EMBED_DIM", "192"))
HIDDEN_DIM = int(os.environ.get("HIDDEN_DIM", "256"))
NLAYERS = int(os.environ.get("NLAYERS", "2"))
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def read_texts_from_csv(path: str) -> List[str]:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "text" not in reader.fieldnames:
            raise SystemExit("dataset_clean.csv must contain 'text' column")
        for row in reader:
            t = (row.get("text") or "").strip()
            if t:
                texts.append(t)
    return texts


def char_vocab(texts: List[str]) -> Dict[str, int]:
    chars = set()
    for t in texts:
        chars.update(list(t))
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for ch in sorted(chars):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int], add_bos: bool, add_eos: bool) -> List[int]:
    ids = []
    if add_bos:
        ids.append(vocab.get("<bos>", 0))
    for ch in list(text):
        ids.append(vocab.get(ch, vocab.get("<unk>", 0)))
    if add_eos:
        ids.append(vocab.get("<eos>", 2))
    return ids[:MAX_LEN]


class LMDataset(Dataset):
    def __init__(self, texts: List[str], vocab: Dict[str, int]):
        self.texts = texts
        self.vocab = vocab
        self.pad_id = vocab["<pad>"]

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        t = self.texts[idx]
        ids = encode(t, self.vocab, add_bos=True, add_eos=True)
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


def collate(batch, pad_id: int):
    xs, ys = zip(*batch)
    max_x = max(x.size(0) for x in xs)
    max_y = max(y.size(0) for y in ys)
    X = torch.full((len(xs), max_x), pad_id, dtype=torch.long)
    Y = torch.full((len(ys), max_y), pad_id, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, :x.size(0)] = x
        Y[i, :y.size(0)] = y
    return X, Y


class CausalLM_RNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, rnn_type: str = "rnn"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        if rnn_type == "gru":
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=NLAYERS, batch_first=True)
        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=NLAYERS, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_dim, hidden_dim, num_layers=NLAYERS, nonlinearity="tanh", batch_first=True)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.rnn(emb)
        logits = self.proj(out)
        return logits


def kfold_indices(n: int, k: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    folds = np.array_split(idx, k)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        splits.append((train_idx, val_idx))
    return splits


def train_eval(model_key: str, texts: List[str], vocab: Dict[str, int], device: torch.device) -> List[Dict[str, float]]:
    pad_id = vocab["<pad>"]
    results: List[Dict[str, float]] = []
    splits = kfold_indices(len(texts), KFOLDS)

    for fold_id, (tr_idx, va_idx) in enumerate(splits):
        fold_dir = os.path.join(OUTPUT_DIR, f"{model_key}_fold{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        tr_texts = [texts[i] for i in tr_idx]
        va_texts = [texts[i] for i in va_idx]

        train_ds = LMDataset(tr_texts, vocab)
        val_ds = LMDataset(va_texts, vocab)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate(b, pad_id))
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate(b, pad_id))

        model = CausalLM_RNN(len(vocab), EMBED_DIM, HIDDEN_DIM, rnn_type=model_key).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
        optim = torch.optim.AdamW(model.parameters(), lr=LR)

        train_losses: List[float] = []
        val_losses: List[float] = []

        for _ in range(EPOCHS):
            # Train
            model.train()
            total_loss = 0.0
            total_tokens = 0
            for X, Y in train_loader:
                X = X.to(device)
                Y = Y.to(device)
                optim.zero_grad()
                logits = model(X)
                loss = criterion(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
                loss.backward()
                optim.step()
                total_loss += float(loss.item()) * Y.numel()
                total_tokens += int(Y.numel())
            train_losses.append(total_loss / max(total_tokens, 1))

            # Validate
            model.eval()
            v_total_loss = 0.0
            v_tokens = 0
            with torch.no_grad():
                for X, Y in val_loader:
                    X = X.to(device)
                    Y = Y.to(device)
                    logits = model(X)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), Y.reshape(-1))
                    v_total_loss += float(loss.item()) * Y.numel()
                    v_tokens += int(Y.numel())
            val_losses.append(v_total_loss / max(v_tokens, 1))

        # Save curves
        plt.figure(figsize=(7,4))
        sns.lineplot(x=range(1, len(train_losses)+1), y=train_losses, label="train")
        sns.lineplot(x=range(1, len(val_losses)+1), y=val_losses, label="val")
        plt.title(f"{model_key.upper()} Loss (Fold {fold_id})")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, "loss_curve.png"), dpi=160, bbox_inches="tight")
        plt.close()

        final_val_loss = val_losses[-1]
        perplexity = float(np.exp(min(20, final_val_loss)))
        metrics = {"val_loss": final_val_loss, "perplexity": perplexity}
        with open(os.path.join(fold_dir, "metrics.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        results.append(metrics)

    return results


def compare_plots(all_scores: Dict[str, List[Dict[str, float]]]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    labels = list(all_scores.keys())
    # Mean metrics
    mean_val = [float(np.mean([s["val_loss"] for s in all_scores[m]])) for m in labels]
    mean_ppl = [float(np.mean([s["perplexity"] for s in all_scores[m]])) for m in labels]

    plt.figure(figsize=(7,4))
    sns.barplot(x=labels, y=mean_val)
    plt.title("Mean Validation Loss by Model")
    plt.ylabel("Val Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cmp_val_loss.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7,4))
    sns.barplot(x=labels, y=mean_ppl)
    plt.title("Mean Perplexity by Model")
    plt.ylabel("Perplexity")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "cmp_perplexity.png"), dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not os.path.isfile(INPUT_CSV):
        raise SystemExit("dataset/dataset_clean.csv not found")
    texts = read_texts_from_csv(INPUT_CSV)
    if len(texts) < 6:
        raise SystemExit("Not enough documents to run k-fold CV")

    vocab = char_vocab(texts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_scores: Dict[str, List[Dict[str, float]]] = {}
    for model_key in ["rnn", "gru", "lstm"]:
        all_scores[model_key] = train_eval(model_key, texts, vocab, device)

    # Save aggregate
    import json
    with open(os.path.join(OUTPUT_DIR, "kfold_all_scores.json"), "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False, indent=2)

    compare_plots(all_scores)
    print("Saved results and charts to outputs_seq_models/.")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    main()


