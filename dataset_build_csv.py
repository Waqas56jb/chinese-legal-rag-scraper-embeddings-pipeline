import os
import csv
import re
from typing import List


SCRAPED_DIR = os.path.join(os.getcwd(), "scraped_json")
OUTPUT_DIR = os.path.join(os.getcwd(), "dataset")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "dataset.csv")


def list_txt_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    names = [n for n in os.listdir(directory) if n.lower().endswith(".txt")]
    names.sort()
    return [os.path.join(directory, n) for n in names]


def normalize_text(text: str) -> str:
    # Standardize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove potential header line like "Category | Name | URL" at the very beginning
    lines = text.split("\n")
    if lines and re.match(r"^[^\n|]+\s*\|\s*[^\n|]+\s*\|\s*[^\n|]+$", lines[0].strip()):
        # Drop header and any immediate blank line following
        lines = lines[1:]
        if lines and lines[0].strip() == "":
            lines = lines[1:]

    # Trim trailing spaces, collapse multiple blanks, keep paragraph structure
    cleaned: List[str] = []
    blank_streak = 0
    for ln in lines:
        s = ln.rstrip()
        if s == "":
            blank_streak += 1
        else:
            blank_streak = 0
        if blank_streak <= 1:
            cleaned.append(s)

    out = "\n".join(cleaned).strip()

    # Remove repeated consecutive duplicate lines
    deduped: List[str] = []
    prev = None
    for ln in out.split("\n"):
        if ln != prev:
            deduped.append(ln)
        prev = ln

    result = "\n".join(deduped).strip()
    # Optionally discard too-short documents
    return result


def main() -> None:
    files = list_txt_files(SCRAPED_DIR)
    if not files:
        raise SystemExit("No .txt files found in scraped_json. Run format_dataset.py first.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect rows with minimum length to ensure quality
    rows: List[str] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        text = normalize_text(raw)
        if len(text) >= 200:  # filter very-short pages
            rows.append(text)

    # Deduplicate whole documents
    seen = set()
    unique_rows: List[str] = []
    for t in rows:
        if t not in seen:
            unique_rows.append(t)
            seen.add(t)

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])  # header
        for t in unique_rows:
            writer.writerow([t])

    print(f"Wrote {len(unique_rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


