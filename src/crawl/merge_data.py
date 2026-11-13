"""Gộp data/raw/*.json -> data/processed/articles.csv.

Các bước:
    1. Đọc toàn bộ file JSON trong data/raw/, kiểm tra đủ cột chuẩn (config.COLUMNS).
    2. Chuẩn hóa Unicode (NFC) + khoảng trắng cho title/content.
    3. Lọc: thiếu tiêu đề/nội dung, bài < MIN_WORDS từ (gồm bài chỉ có video/ảnh),
       trùng URL, trùng tiêu đề, gần trùng nội dung (Jaccard >= NEAR_DUP_THRESHOLD).
    4. Đánh lại id toàn cục (doc_0001, ...) để id không lộ nhãn chủ đề.
    5. Ghi CSV UTF-8 và in thống kê, cảnh báo chủ đề dưới 10 bài.

Cách chạy:
    python -m src.crawl.merge_data

Phụ trách: Thanh Giang
"""
import json
import re
import unicodedata

import pandas as pd

from src import config

NEAR_DUP_THRESHOLD = 0.8  # tỉ lệ trùng tập từ giữa 2 bài để coi là gần trùng
MIN_PER_TOPIC = 10


def norm(text) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    lines = (re.sub(r"[ \t ]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def word_set(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def load_raw() -> pd.DataFrame:
    files = sorted(config.RAW_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"Không có file JSON trong {config.RAW_DIR}. Chạy crawler trước.")
    frames = []
    for f in files:
        records = json.loads(f.read_text(encoding="utf-8"))
        df = pd.DataFrame(records)
        missing = set(config.COLUMNS) - set(df.columns)
        if missing:
            raise SystemExit(f"{f.name} thiếu cột: {sorted(missing)}")
        df["raw_file"] = f.name
        frames.append(df)
        print(f"  đọc {f.name}: {len(df)} bài")
    return pd.concat(frames, ignore_index=True)


def drop_near_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Bỏ bài có tập từ trùng >= ngưỡng với một bài đã giữ trước đó."""
    kept_idx, kept_sets = [], []
    for idx, content in df["content"].items():
        ws = word_set(content)
        dup = any(len(ws & ks) / max(len(ws | ks), 1) >= NEAR_DUP_THRESHOLD for ks in kept_sets)
        if dup:
            print(f"    - gần trùng: {df.at[idx, 'title'][:70]}")
            continue
        kept_idx.append(idx)
        kept_sets.append(ws)
    return df.loc[kept_idx]


def merge() -> pd.DataFrame:
    df = load_raw()
    n0 = len(df)
    log = {}

    for col in ("title", "content"):
        df[col] = df[col].map(norm)
    df["url"] = df["url"].astype(str).str.strip()

    def step(name, mask):
        nonlocal df
        log[name] = int((~mask).sum())
        df = df[mask]

    step("thiếu tiêu đề/nội dung", (df["title"] != "") & (df["content"] != ""))
    step(f"dưới {config.MIN_WORDS} từ", df["content"].str.split().str.len() >= config.MIN_WORDS)
    step("trùng URL", ~df["url"].duplicated())
    step("trùng tiêu đề", ~df["title"].str.lower().duplicated())
    before = len(df)
    df = drop_near_duplicates(df)
    log["gần trùng nội dung"] = before - len(df)

    df = df.sort_values(["true_label", "source", "date"]).reset_index(drop=True)
    df["id"] = [f"doc_{i:04d}" for i in range(1, len(df) + 1)]
    df = df[config.COLUMNS]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.ARTICLES_CSV, index=False, encoding="utf-8")

    print(f"\nĐầu vào: {n0} bài")
    for name, n in log.items():
        print(f"  bỏ {n:>3} bài: {name}")
    print(f"Còn lại: {len(df)} bài -> {config.ARTICLES_CSV.relative_to(config.ROOT)}\n")
    print(pd.crosstab(df["true_label"], df["source"], margins=True, margins_name="Tổng"))

    counts = df["true_label"].value_counts()
    short = counts[counts < MIN_PER_TOPIC]
    if len(counts) < 5 or len(short):
        print(f"\nCẢNH BÁO: cần >= 5 chủ đề, mỗi chủ đề >= {MIN_PER_TOPIC} bài. Thiếu: {short.to_dict()}")
    return df


if __name__ == "__main__":
    merge()
