import os
import json
from typing import List

SCRAPED_DIR = os.path.join(os.getcwd(), "scraped_json")


def list_json_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".json")
    ]


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        return ""
    # Ensure we operate on real newlines (json.load already unescapes \n)
    text = value.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse more than 2 consecutive blank lines to exactly one
    # and trim trailing spaces on each line
    normalized_lines: List[str] = []
    blank_streak = 0
    for line in text.split("\n"):
        stripped_line = line.rstrip()
        if stripped_line == "":
            blank_streak += 1
        else:
            blank_streak = 0
        if blank_streak <= 1:
            normalized_lines.append(stripped_line)

    return "\n".join(normalized_lines).strip()


def process_file(json_path: str) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    content = data.get("Content", "")
    normalized = normalize_text(content)

    # Write companion .txt next to the json
    base, _ = os.path.splitext(json_path)
    txt_path = f"{base}.txt"
    with open(txt_path, "w", encoding="utf-8", newline="\n") as out:
        # Optional header for context in RAG
        header_parts = [
            data.get("Category") or "",
            data.get("Name") or "",
            data.get("URL") or "",
        ]
        header = " | ".join(part for part in header_parts if part)
        if header:
            out.write(header + "\n\n")
        out.write(normalized)

    return txt_path


def main() -> None:
    files = list_json_files(SCRAPED_DIR)
    if not files:
        print("No JSON files found in scraped_json.")
        return

    written = 0
    for path in sorted(files):
        try:
            txt_path = process_file(path)
            print(f"✔ Wrote: {txt_path}")
            written += 1
        except Exception as exc:
            print(f"[!] Failed {path}: {exc}")

    print(f"Done. Generated {written} text files.")


if __name__ == "__main__":
    main()


