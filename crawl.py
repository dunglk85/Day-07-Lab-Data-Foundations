"""
crawl.py — Thu thập ~5 tài liệu về một chủ đề và lưu thành file .md

Cài dependencies:
    pip install requests beautifulsoup4 markdownify

Chạy:
    python crawl.py

Kết quả: thư mục data/<TOPIC_SLUG>/ chứa các file .md
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# ------------------------------------------------------------------ #
# CONFIG — chỉnh tại đây                                              #
# ------------------------------------------------------------------ #

TOPIC = "vector database retrieval augmented generation"   # chủ đề tìm kiếm
TOPIC_SLUG = "phapluat"                               # tên thư mục output
MAX_DOCS = 5                                               # số tài liệu muốn crawl
OUTPUT_DIR = Path("data") / TOPIC_SLUG
REQUEST_DELAY = 1.5                                        # giây giữa các request

# Danh sách URL cụ thể (ưu tiên dùng nếu có)
# Để rỗng [] để tự động search qua DuckDuckGo
SEED_URLS: list[str] = [
    "https://vnexpress.net/phap-luat",
    "https://vietnamnet.vn/phap-luat",
    "https://tuoitre.vn/phap-luat.htm"
]

# ------------------------------------------------------------------ #
# Helpers                                                             #
# ------------------------------------------------------------------ #

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ResearchBot/1.0; "
        "+https://example.com/bot)"
    )
}


def search_duckduckgo(query: str, max_results: int = 10) -> list[str]:
    """Tìm kiếm DuckDuckGo HTML và trả về danh sách URL."""
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query, "kl": "us-en"}
    resp = requests.post(url, data=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []
    for a in soup.select("a.result__url"):
        href = a.get("href", "")
        if href.startswith("http"):
            urls.append(href)
        if len(urls) >= max_results:
            break
    return urls


def fetch_page(url: str) -> BeautifulSoup | None:
    """Tải trang và trả về BeautifulSoup, None nếu lỗi."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  [skip] {url} — {e}")
        return None


def extract_main_content(soup: BeautifulSoup, url: str) -> str | None:
    """
    Trích xuất nội dung chính của trang.
    Thử các selector phổ biến theo thứ tự ưu tiên.
    """
    # Xóa các phần không cần thiết
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    # Thử selector phổ biến
    candidates = [
        soup.find("article"),
        soup.find("main"),
        soup.find(attrs={"role": "main"}),
        soup.find(class_=re.compile(r"(content|post|article|entry|body)", re.I)),
        soup.find("div", id=re.compile(r"(content|main|article)", re.I)),
        soup.find("body"),
    ]

    for element in candidates:
        if element:
            text = element.get_text(separator=" ", strip=True)
            if len(text) > 300:   # đủ dài mới lấy
                return md(str(element), heading_style="ATX", strip=["a", "img"])

    return None


def slugify(text: str) -> str:
    """Chuyển tiêu đề thành tên file an toàn."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "_", text)
    return text[:60]


def get_title(soup: BeautifulSoup, url: str) -> str:
    """Lấy tiêu đề trang."""
    if soup.find("h1"):
        return soup.find("h1").get_text(strip=True)
    if soup.title:
        return soup.title.string.strip()
    return urlparse(url).netloc


def clean_markdown(text: str) -> str:
    """Dọn dẹp markdown: bỏ dòng trống thừa, bỏ dòng chỉ có dấu."""
    lines = text.splitlines()
    cleaned = []
    prev_blank = False
    for line in lines:
        stripped = line.strip()
        is_blank = stripped == ""
        is_junk = re.fullmatch(r"[*\-_=|#`~]{1,3}", stripped) is not None

        if is_junk:
            continue
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned).strip()


def save_as_markdown(title: str, url: str, content: str, output_dir: Path, index: int) -> Path:
    """Lưu nội dung thành file .md với frontmatter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{index:02d}_{slugify(title)}.md"
    filepath = output_dir / filename

    frontmatter = f"""---
title: "{title}"
source: "{url}"
topic: "{TOPIC}"
---

# {title}

> Source: {url}

"""
    filepath.write_text(frontmatter + clean_markdown(content), encoding="utf-8")
    return filepath


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main() -> None:
    print("=" * 60)
    print(f"  Crawl: {TOPIC}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Lấy danh sách URL
    urls = list(SEED_URLS)
    if len(urls) < MAX_DOCS:
        print(f"\n[search] DuckDuckGo: {TOPIC!r} ...")
        try:
            found = search_duckduckgo(TOPIC, max_results=MAX_DOCS * 3)
            # Loại bỏ duplicate và domain không mong muốn
            skip_domains = {"reddit.com", "youtube.com", "twitter.com", "facebook.com"}
            for u in found:
                domain = urlparse(u).netloc
                if not any(s in domain for s in skip_domains):
                    if u not in urls:
                        urls.append(u)
                if len(urls) >= MAX_DOCS * 2:
                    break
            print(f"  Tìm được {len(found)} URL, sau lọc còn {len(urls)}")
        except Exception as e:
            print(f"  [warn] Search thất bại: {e}")
            if not urls:
                print("  Không có URL nào để crawl. Hãy điền SEED_URLS thủ công.")
                return

    # Crawl từng URL
    saved = 0
    attempted = 0

    for url in urls:
        if saved >= MAX_DOCS:
            break
        attempted += 1
        print(f"\n[{saved+1}/{MAX_DOCS}] {url}")

        soup = fetch_page(url)
        if soup is None:
            continue

        title = get_title(soup, url)
        content = extract_main_content(soup, url)

        if not content or len(content.strip()) < 200:
            print(f"  [skip] Nội dung quá ngắn hoặc không trích xuất được")
            continue

        filepath = save_as_markdown(title, url, content, OUTPUT_DIR, saved + 1)
        char_count = len(content)
        print(f"  [ok] {filepath.name} — {char_count} ký tự")
        saved += 1

        time.sleep(REQUEST_DELAY)

    print(f"\n{'='*60}")
    print(f"  Hoàn thành: {saved}/{MAX_DOCS} tài liệu → {OUTPUT_DIR}/")
    print(f"  Đã thử: {attempted} URL")
    if saved < MAX_DOCS:
        print(f"  [gợi ý] Thêm URL vào SEED_URLS nếu crawl tự động thiếu tài liệu")
    print("=" * 60)


if __name__ == "__main__":
    main()
