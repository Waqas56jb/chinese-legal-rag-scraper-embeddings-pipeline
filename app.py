import os
import time
import random
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

# ✅ All URLs with metadata (directly inside code)
URLS = [
    {"Category": "Government/Justice", "Name": "National Laws & Regs DB (NPC)", "URL": "https://flk.npc.gov.cn/"},
    {"Category": "Government/Justice", "Name": "National People’s Congress (main)", "URL": "http://www.npc.gov.cn/"},
    {"Category": "Government/Justice", "Name": "Supreme People’s Court", "URL": "http://www.court.gov.cn/"},
    {"Category": "Government/Justice", "Name": "China Court Network", "URL": "http://www.chinacourt.org/"},
    {"Category": "Government/Justice", "Name": "Chinese Government Portal", "URL": "http://www.gov.cn/"},
    {"Category": "Government/Justice", "Name": "Ministry of Justice", "URL": "http://www.moj.gov.cn/"},
    {"Category": "Government/Justice", "Name": "MOHRSS", "URL": "http://www.mohrss.gov.cn/"},
    {"Category": "Government/Justice", "Name": "MOFCOM", "URL": "http://www.mofcom.gov.cn/"},
    {"Category": "Government/Justice", "Name": "MOFCOM FTA Portal", "URL": "http://fta.mofcom.gov.cn/"},
    {"Category": "Government/Justice", "Name": "SAMR", "URL": "http://www.samr.gov.cn/"},
    {"Category": "Government/Justice", "Name": "State Taxation Admin", "URL": "http://www.chinatax.gov.cn/"},
    {"Category": "Government/Justice", "Name": "CBIRC", "URL": "http://www.cbirc.gov.cn/"},
    {"Category": "Government/Justice", "Name": "CSRC", "URL": "http://www.csrc.gov.cn/"},
    {"Category": "Government/Justice", "Name": "National Health Commission", "URL": "http://www.nhc.gov.cn/"},
    {"Category": "Government/Justice", "Name": "National Bureau of Statistics", "URL": "http://www.stats.gov.cn/"},
    {"Category": "Government/Justice", "Name": "Guangdong Provincial Gov", "URL": "http://www.gd.gov.cn/"},
    {"Category": "Government/Justice", "Name": "Shanghai Municipal Gov", "URL": "https://www.shanghai.gov.cn/"},
    {"Category": "Government/Justice", "Name": "Beijing HRSS", "URL": "http://rsj.beijing.gov.cn/"},
    {"Category": "Government/Justice", "Name": "Shanghai HRSS", "URL": "http://rsj.sh.gov.cn/"},
    {"Category": "International", "Name": "UN Treaty Collection", "URL": "https://treaties.un.org/"},
    {"Category": "Community/Portals", "Name": "OpenLaw.cn", "URL": "http://www.openlaw.cn/"},
    {"Category": "Templates", "Name": "Findlaw China (templates)", "URL": "https://china.findlaw.cn/fanben/"},
    {"Category": "Templates", "Name": "Lawtime Templates", "URL": "https://fanben.lawtime.cn/"},
    {"Category": "Academic", "Name": "Stanford China Guiding Cases", "URL": "https://law.stanford.edu/china-guiding-cases-project/"}
]

OUTPUT_DIR = "scraped_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_browser():
    options = Options()
    options.add_argument("--headless")  # comment if you want to see browser
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    # Use webdriver-manager to install and resolve a compatible ChromeDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def scrape_page(driver, url_entry, idx):
    try:
        url = url_entry["URL"]
        print(f"[+] Scraping {idx}: {url}")
        driver.get(url)

        # Random wait between 3-7 seconds
        time.sleep(random.uniform(3, 7))

        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Remove scripts/styles
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = soup.get_text(separator="\n")
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

        data = {
            "Category": url_entry["Category"],
            "Name": url_entry["Name"],
            "URL": url,
            "Content": text
        }

        # Save JSON file
        filename = os.path.join(OUTPUT_DIR, f"page_{idx}.json")
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"    ✔ Saved: {filename}")

    except Exception as e:
        print(f"[!] Failed {url_entry['URL']}: {e}")

def main():
    driver = setup_browser()
    for idx, entry in enumerate(URLS, start=1):
        scrape_page(driver, entry, idx)
    driver.quit()

if __name__ == "__main__":
    main()
