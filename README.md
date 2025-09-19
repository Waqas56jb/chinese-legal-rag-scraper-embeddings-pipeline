### Chinese Legal RAG — Scraper & Formatting Pipeline

Scrape Chinese legal/government websites, normalize extracted text, and prepare high‑quality corpora for RAG systems.

This repo contains:
- `app.py`: Headless Selenium scraper for a curated set of official/legal Chinese sites
- `format_dataset.py`: Cleans and normalizes scraped content; writes structured `.txt` companions
- `scraped_json/`: Example outputs (`page_#.json` + `.txt`)

---

## Prerequisites
- Python 3.9+ (3.10/3.11 recommended)
- Google Chrome installed
- Windows, macOS, or Linux

The scraper uses `webdriver-manager` to automatically install a compatible ChromeDriver, so you typically do not need to manage `chromedriver.exe` yourself.

## Quickstart
1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```bash
pip install -r requirements.txt
```

3) Run the scraper (headless Chrome)

```bash
python app.py
```

4) Normalize and export text files for RAG

```bash
python format_dataset.py
```

Outputs are written to `scraped_json/`:
- JSON: `page_#.json` with metadata + raw cleaned text
- Text: `page_#.txt` with a small header for context and normalized body

## What gets scraped
`app.py` contains an in‑code list of authoritative sources (e.g., NPC, SPC, MOFCOM, SAMR, provincial/municipal portals, and select academic/community resources). Each page is fetched, scripts/styles are removed, and visible text is extracted.

### JSON schema (example)
```json
{
  "Category": "Government/Justice",
  "Name": "Supreme People’s Court",
  "URL": "http://www.court.gov.cn/",
  "Content": "... extracted visible text ..."
}
```

### Text output
For each JSON file, `format_dataset.py` writes a `.txt` neighbor that starts with a single header line:

```
Category | Name | URL

<normalized body>
```

Normalization includes:
- Converting line endings to `\n`
- Trimming trailing spaces
- Collapsing excessive blank lines to at most one

## Configuration tips
- To watch the browser while scraping, open `app.py` and remove or comment the `--headless` option in `setup_browser()`.
- Random delays (3–7s) between requests are applied to be polite to hosts; adjust in `app.py` if needed.
- To add/remove target sites, edit the `URLS` list in `app.py`.

## Building embeddings
This repo produces high‑quality text files suitable for embedding. Use your preferred embedding model and chunking strategy. Example (pseudo‑workflow):

```python
from pathlib import Path

texts = []
for p in Path("scraped_json").glob("*.txt"):
    texts.append(p.read_text(encoding="utf-8"))

# chunk -> embed -> index (e.g., FAISS / Milvus / Weaviate)
```

## Troubleshooting
- Selenium cannot start Chrome
  - Ensure Google Chrome is installed and up‑to‑date
  - Delete any stray `chromedriver.exe` in PATH that might conflict; `webdriver-manager` will fetch a match
- TLS/SSL or corporate proxy issues
  - Configure your system proxy or set `REQUESTS_CA_BUNDLE` where necessary
- Chinese characters display as garbled
  - Always open files with UTF‑8 encoding; this repo writes UTF‑8 explicitly
- Sites block automation or load slowly
  - Increase waits in `app.py`, add explicit waits, or reduce concurrency (this scraper is single‑page at a time by design)

## Legal and ethical use
Review and comply with each website’s Terms of Service and applicable laws. Use respectful crawl rates and cache where possible.

## Project structure
```
├─ app.py                 # Selenium-based scraper (headless by default)
├─ format_dataset.py      # Normalization and .txt export
├─ requirements.txt       # Python dependencies
├─ scraped_json/          # Example outputs (JSON + TXT)
└─ README.md
```

## License
MIT — see `LICENSE` (add one if missing).
