import os
import csv
import re
import unicodedata
from typing import List


INPUT_CSV = os.path.join(os.getcwd(), "dataset", "dataset.csv")
OUTPUT_CSV = os.path.join(os.getcwd(), "dataset", "dataset_clean.csv")


def to_halfwidth(s: str) -> str:
    # Normalize to NFKC (handles most full-width to half-width)
    s = unicodedata.normalize("NFKC", s)
    # Explicit mapping for spaces
    s = s.replace("\u3000", " ")
    return s


def normalize_punct(s: str) -> str:
    # Standardize common Chinese/English punctuation variants
    repl = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "―": "-",
        "…": "...",
        "，": ",",
        "。": ".",
        "；": ";",
        "：": ":",
        "！": "!",
        "？": "?",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "《": "<",
        "》": ">",
        "、": ",",
    }
    return s.translate(str.maketrans(repl))


def strip_control_chars(s: str) -> str:
    # Remove zero-width and control characters
    s = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", s)
    s = "".join(ch for ch in s if ch == "\n" or ch >= " " )
    return s


def fix_spacing_mixed(s: str) -> str:
    # Collapse excessive spaces
    s = re.sub(r"[\t\f\v\u00A0]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    # Remove spaces around CJK chars
    cjk = r"\u4E00-\u9FFF\u3400-\u4DBF\u3000-\u303F\uFF00-\uFFEF"
    s = re.sub(fr"([\u0000-\u007F])\s+([[{cjk}]])", r"\1\2", s)
    s = re.sub(fr"([[{cjk}]])\s+([\u0000-\u007F])", r"\1\2", s)
    return s


def remove_boilerplate_lines(text: str) -> str:
    lines = text.split("\n")
    junk_patterns = [
        r"版权所有|版权|隐私政策|网站地图|免责声明|常见问题|帮助中心|用户协议|关于我们",
        r"English|简体|繁體|返回首页|返回顶部|登录注册|用户空间|无障碍浏览|扫码|小程序",
        r"ICP备\d+号|京ICP备|沪ICP备|粤ICP备|公安备案|站点地图",
        r"分享到|官方微信|官方微博|关注我们",
        r"(上一页|下一页|首页|末页)[\s\d/]*$",
        r"^\d{1,2}$",  # bare small page numbers like 1 2 3
        r"^\d{2}-\d{2}$|^\d{4}-\d{2}-\d{2}$|\d{4}年\d{1,2}月\d{1,2}日",
        r"^相关链接$|^友情链接$|^更多$|^点击查看更多内容$",
        r"^\|+$|^[-–—]+$|^[•·]+$",
        r"^\s*(首页|机构|新闻|政务|服务|互动|专题|数据|图片|视频|公告|通知)(\s*\|\s*|\s+){2,}.*$",
        r"^(中国政府网|国务院部门|外交部|国防部|国家发展和改革委员会|教育部|科技部|工信部|公安部|司法部|财政部|人力资源和社会保障部|自然资源部|生态环境部|住建部|交通运输部|水利部|农业农村部|商务部|文化和旅游部|国家卫生健康委员会|退役军人事务部|应急管理部|中国人民银行|审计署)\s*$",
        r"^[A-Z]{2}-[\u4E00-\u9FFFA-Za-z\(\)（）]+$",  # AU-澳大利亚 style lists
        r"^(Français|Español|Deutsch)\s*$",
        r"^来源[:：].*$",
        r"^\d{2}-\d{2}$",  # date chips like 09-15
        r"^\d{4}年$|^\d{4}$",
    ]
    kept: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            kept.append("")
            continue
        if any(re.search(p, s) for p in junk_patterns):
            continue
        kept.append(s)
    # Collapse multiple blanks
    out: List[str] = []
    blank = 0
    for ln in kept:
        if ln == "":
            blank += 1
        else:
            blank = 0
        if blank <= 1:
            out.append(ln)
    # Dedup consecutive lines
    dedup: List[str] = []
    prev = None
    for ln in out:
        if ln != prev:
            dedup.append(ln)
        prev = ln
    return "\n".join(dedup).strip()


def keep_informative(text: str) -> str:
    kept: List[str] = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            kept.append("")
            continue
        # Heuristics: keep lines with enough CJK/alpha and not mostly digits/symbols
        cjk_count = len(re.findall(r"[\u4E00-\u9FFF]", s))
        alpha_count = len(re.findall(r"[A-Za-z]", s))
        digit_count = len(re.findall(r"\d", s))
        sym_count = len(re.findall(r"[^\w\u4E00-\u9FFF\s]", s))
        total = len(s)
        if total == 0:
            continue
        informative_ratio = (cjk_count + alpha_count) / total
        digits_ratio = digit_count / total
        # drop very short or low informative lines, or lines mostly numbers/symbols
        if len(s) < 6:
            continue
        if informative_ratio < 0.35:
            continue
        if digits_ratio > 0.6:
            continue
        kept.append(s)
    # Collapse blanks again
    out: List[str] = []
    blank = 0
    for s in kept:
        if s == "":
            blank += 1
        else:
            blank = 0
        if blank <= 1:
            out.append(s)
    return "\n".join(out).strip()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_control_chars(text)
    text = to_halfwidth(text)
    text = normalize_punct(text)
    text = fix_spacing_mixed(text)
    text = remove_boilerplate_lines(text)
    text = keep_informative(text)
    # Trim repeated punctuation (keep at most 3)
    text = re.sub(r"([!?.。！？])\1{3,}", r"\1\1\1", text)
    # Drop documents too short after cleaning
    return text


def main() -> None:
    if not os.path.isfile(INPUT_CSV):
        raise SystemExit("dataset/dataset.csv not found. Build it first.")

    rows_in = 0
    rows_out = 0
    unique = set()
    cleaned_rows: List[str] = []

    with open(INPUT_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "text" not in reader.fieldnames:
            raise SystemExit("dataset.csv must have a 'text' column")
        for row in reader:
            rows_in += 1
            raw = row.get("text") or ""
            txt = clean_text(raw)
            if len(txt) < 200:
                continue
            if txt in unique:
                continue
            unique.add(txt)
            cleaned_rows.append(txt)
            rows_out += 1

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="\n") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        for t in cleaned_rows:
            writer.writerow([t])

    print(f"Read {rows_in} rows, wrote {rows_out} cleaned rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


