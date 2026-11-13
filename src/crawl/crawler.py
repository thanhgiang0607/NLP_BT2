"""Crawl bài báo tiếng Việt theo chủ đề từ VnExpress, Tuổi Trẻ, Thanh Niên, Dân trí.

Kết quả: data/raw/<chu_de>.json, mỗi phần tử có các cột
    id, title, content, url, source, date, true_label

Cách chạy:
    python -m src.crawl.crawler                    # mặc định 20 bài/chủ đề
    python -m src.crawl.crawler --per-topic 25
    python -m src.crawl.crawler --topics the_thao suc_khoe

Phụ trách: Thanh Giang
"""
import argparse
import json
import re
import time
import unicodedata
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from src import config
from src.crawl.sources import (
    ACCEPTED_SECTIONS,
    ARTICLE_URL_PATTERNS,
    CATEGORY_PATHS,
    SELECTORS,
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "vi-VN,vi;q=0.9",
}
DELAY = 0.5 
TIMEOUT = 20

session = requests.Session()
session.headers.update(HEADERS)


def norm(text: str) -> str:
    """Chuẩn hóa Unicode (NFC), gộp khoảng trắng."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text or "")).strip()


def section_key(text: str) -> str:
    """Chuẩn hóa tên chuyên mục để so khớp ('Sức khoẻ' == 'Sức khỏe')."""
    t = norm(text).lower()
    return t.replace("khoẻ", "khỏe").replace("hoá", "hóa")


def fetch(url: str) -> BeautifulSoup | None:
    try:
        r = session.get(url, timeout=TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  ! lỗi tải {url}: {e}")
        return None
    finally:
        time.sleep(DELAY)
    r.encoding = "utf-8"
    return BeautifulSoup(r.text, "lxml")


def list_article_urls(source: str, category_url: str) -> list[str]:
    """Lấy link bài viết trên trang chuyên mục (giữ thứ tự, bỏ trùng)."""
    soup = fetch(category_url)
    if soup is None:
        return []
    pattern = re.compile(ARTICLE_URL_PATTERNS[source])
    urls = []
    for a in soup.find_all("a", href=True):
        url = urljoin(category_url, a["href"]).split("?")[0].split("#")[0]
        if pattern.match(url) and url not in urls:
            urls.append(url)
    return urls


def parse_json_ld(soup: BeautifulSoup) -> tuple[list[str], str]:
    """Trả về (breadcrumb, datePublished) từ JSON-LD."""
    breadcrumb, date = [], ""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "", strict=False)
        except json.JSONDecodeError:
            continue
        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("@type"))
            if "BreadcrumbList" in kind:
                for el in item.get("itemListElement", []):
                    el = el[0] if isinstance(el, list) and el else el
                    if isinstance(el, dict):
                        name = el.get("name") or (el.get("item") or {}).get("name", "")
                        breadcrumb.append(norm(name))
            if kind in ("NewsArticle", "Article") and not date:
                date = item.get("datePublished", "")
    return breadcrumb, date


def extract_body(container) -> str:
    """Ghép các đoạn <p> trong thân bài, bỏ chú thích ảnh và dòng tên tác giả cuối bài."""
    for bad in container.select("figure, figcaption, table, script, style, .box-tinlienquanv2, .VCSortableInPreviewMode"):
        bad.decompose()
    paras = [norm(p.get_text(" ")) for p in container.find_all("p")]
    paras = list(dict.fromkeys(p for p in paras if p))  # bỏ đoạn rỗng/lặp (bài ảnh)
    # Dòng cuối thường là tên tác giả / "Theo ..." (rất ngắn)
    while paras and len(paras[-1].split()) <= 5:
        paras.pop()
    return "\n".join(paras)


def parse_article(url: str, source: str, topic: str) -> tuple[dict | None, str]:
    """Trả về (bài viết, lý do bỏ qua)."""
    soup = fetch(url)
    if soup is None:
        return None, "lỗi tải"

    breadcrumb, date = parse_json_ld(soup)
    sections = [section_key(b) for b in breadcrumb if section_key(b) != "trang chủ"]
    if not sections or sections[0] not in ACCEPTED_SECTIONS[topic]:
        return None, f"sai chuyên mục ({breadcrumb[:3]})"

    sel = SELECTORS[source]
    title_el = soup.select_one(sel["title"])
    body_el = soup.select_one(sel["body"])
    if title_el is None or body_el is None:
        return None, "không có thân bài (video/ảnh/magazine)"

    sapo_el = soup.select_one(sel["sapo"])
    sapo = norm(sapo_el.get_text(" ")) if sapo_el else ""
    if sapo_el is not None and body_el in sapo_el.parents:
        sapo_el.extract()  # VnExpress: sapo nằm trong thân bài -> tránh lặp
    sapo = re.sub(r"^\((Dân trí|Dân Trí)\)\s*-\s*", "", sapo)
    body = extract_body(body_el)
    content = "\n".join(x for x in (sapo, body) if x)

    if len(content.split()) < config.MIN_WORDS:
        return None, f"quá ngắn ({len(content.split())} từ)"

    if not date:
        meta = soup.find("meta", property="article:published_time")
        date = meta["content"] if meta else ""

    return {
        "title": norm(title_el.get_text(" ")),
        "content": content,
        "url": url,
        "source": source,
        "date": date,
        "true_label": config.TOPICS[topic],
    }, ""


def crawl_topic(topic: str, per_topic: int) -> list[dict]:
    """Crawl 1 chủ đề, chia đều số bài cho các nguồn; nguồn nào thiếu thì nguồn khác bù."""
    sources = list(CATEGORY_PATHS[topic])
    candidates = {s: list_article_urls(s, CATEGORY_PATHS[topic][s]) for s in sources}
    for s in sources:
        print(f"  {s}: {len(candidates[s])} link")

    quota = -(-per_topic // len(sources))  # chia lên
    articles, seen_titles = [], set()

    def take(source: str, limit: int):
        while candidates[source] and limit > 0 and len(articles) < per_topic:
            url = candidates[source].pop(0)
            art, reason = parse_article(url, source, topic)
            if art is None:
                print(f"    - bỏ {url.rsplit('/', 1)[-1][:60]}: {reason}")
                continue
            key = art["title"].lower()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            articles.append(art)
            limit -= 1

    for s in sources:  # vòng 1: mỗi nguồn tối đa `quota` bài
        take(s, quota)
    for s in sources:  # vòng 2: bù nếu còn thiếu
        take(s, per_topic)

    for i, art in enumerate(articles, 1):
        art["id"] = f"{topic}_{i:03d}"
    return [{c: a[c] for c in config.COLUMNS} for a in articles]


def main():
    parser = argparse.ArgumentParser(description="Crawl tin tức tiếng Việt theo chủ đề")
    parser.add_argument("--per-topic", type=int, default=config.ARTICLES_PER_TOPIC)
    parser.add_argument("--topics", nargs="*", default=list(config.TOPICS))
    args = parser.parse_args()

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    for topic in args.topics:
        print(f"\n=== {topic} ({config.TOPICS[topic]}) ===")
        articles = crawl_topic(topic, args.per_topic)
        out = config.RAW_DIR / f"{topic}.json"
        out.write_text(json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[topic] = len(articles)
        print(f"  => {len(articles)} bài -> {out.relative_to(config.ROOT)}")

    print("\nTổng kết:", summary, "| tổng:", sum(summary.values()))
    short = [t for t, n in summary.items() if n < 10]
    if short:
        print("CẢNH BÁO: chủ đề dưới 10 bài:", short)


if __name__ == "__main__":
    main()
