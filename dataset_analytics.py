import os
import csv
import re
from collections import Counter
from typing import List

import matplotlib.pyplot as plt
import seaborn as sns


INPUT_CSV = os.path.join(os.getcwd(), "dataset", "dataset_clean.csv")
OUT_DIR = os.path.join(os.getcwd(), "dataset", "analytics")


def read_texts(path: str) -> List[str]:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("text") or "").strip()
            if t:
                texts.append(t)
    return texts


def tokenize_for_counts(text: str) -> List[str]:
    # split by whitespace and punctuation, keep CJK chars as single tokens
    # latin words kept whole
    tokens = []
    # separate CJK
    for seg in re.split(r"([\u4E00-\u9FFF])", text):
        if not seg:
            continue
        if re.match(r"^[\u4E00-\u9FFF]$", seg):
            tokens.append(seg)
        else:
            parts = re.split(r"[^A-Za-z0-9_]+", seg)
            tokens.extend([p.lower() for p in parts if p])
    return tokens


def plot_and_save(figpath: str):
    plt.tight_layout()
    plt.savefig(figpath, dpi=160, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not os.path.isfile(INPUT_CSV):
        raise SystemExit("dataset_clean.csv not found")
    os.makedirs(OUT_DIR, exist_ok=True)

    texts = read_texts(INPUT_CSV)

    # 1) Document length (characters)
    char_lens = [len(t) for t in texts]
    plt.figure(figsize=(7,4))
    sns.histplot(char_lens, bins=40)
    plt.title("Document Length (characters)")
    plt.xlabel("Chars")
    plot_and_save(os.path.join(OUT_DIR, "doc_char_length_hist.png"))

    # 2) Estimated word length: split on whitespace; for CJK, fallback to char count
    def est_words(t: str) -> int:
        # count latin tokens + CJK chars as words
        return len(tokenize_for_counts(t))

    word_lens = [est_words(t) for t in texts]
    plt.figure(figsize=(7,4))
    sns.histplot(word_lens, bins=40, color="orange")
    plt.title("Document Length (token estimate)")
    plt.xlabel("Tokens (mixed)")
    plot_and_save(os.path.join(OUT_DIR, "doc_token_length_hist.png"))

    # 3) Top token frequencies (mixed Chinese char and latin)
    counter = Counter()
    for t in texts:
        counter.update(tokenize_for_counts(t))
    most_common = counter.most_common(30)
    labels = [w for w, _ in most_common]
    values = [c for _, c in most_common]
    plt.figure(figsize=(8,6))
    sns.barplot(x=values, y=labels)
    plt.title("Top 30 Tokens")
    plt.xlabel("Frequency")
    plot_and_save(os.path.join(OUT_DIR, "top_tokens.png"))

    # 4) Rare token long tail (rank-frequency plot)
    freqs = sorted(counter.values(), reverse=True)
    plt.figure(figsize=(7,4))
    sns.lineplot(x=list(range(1, len(freqs)+1)), y=freqs)
    plt.yscale('log')
    plt.title("Token Rank-Frequency (log scale)")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plot_and_save(os.path.join(OUT_DIR, "rank_frequency.png"))

    # 5) Average line length distribution
    avg_line_lens: List[float] = []
    for t in texts:
        lines = [ln for ln in t.split("\n") if ln.strip()]
        if not lines:
            avg_line_lens.append(0)
        else:
            avg_line_lens.append(sum(len(ln) for ln in lines) / len(lines))
    plt.figure(figsize=(7,4))
    sns.histplot(avg_line_lens, bins=40, color="green")
    plt.title("Average Line Length per Document")
    plt.xlabel("Chars/line")
    plot_and_save(os.path.join(OUT_DIR, "avg_line_length_hist.png"))

    # 6) Number of lines per document
    num_lines = [len([ln for ln in t.split("\n") if ln.strip()]) for t in texts]
    plt.figure(figsize=(7,4))
    sns.histplot(num_lines, bins=40, color="#556")
    plt.title("Non-empty Lines per Document")
    plt.xlabel("Lines")
    plot_and_save(os.path.join(OUT_DIR, "lines_per_doc_hist.png"))

    # 7) Chinese vs Latin token ratio per doc
    ratios: List[float] = []
    for t in texts:
        toks = tokenize_for_counts(t)
        if not toks:
            ratios.append(0.0)
            continue
        cjk = sum(1 for z in toks if re.match(r"^[\u4E00-\u9FFF]$", z))
        ratios.append(cjk / len(toks))
    plt.figure(figsize=(7,4))
    sns.histplot(ratios, bins=30, color="#a55")
    plt.title("Chinese Token Ratio per Document")
    plt.xlabel("Ratio")
    plot_and_save(os.path.join(OUT_DIR, "cjk_ratio_hist.png"))

    # 8) Boxplot of document token lengths
    plt.figure(figsize=(7,4))
    sns.boxplot(y=word_lens)
    plt.title("Token Length Distribution (Boxplot)")
    plot_and_save(os.path.join(OUT_DIR, "token_length_boxplot.png"))

    print(f"Saved analytics charts to {OUT_DIR}")


if __name__ == "__main__":
    main()


