"""Tiền xử lý toàn bộ dữ liệu: articles.csv -> articles_clean.csv (thêm cột `tokens`).

Cách chạy:  python -m src.preprocessing.build_clean
"""
import argparse

import pandas as pd

from src import config
from src.preprocessing.clean import keep_words, strip_markup
from src.preprocessing.stopwords import load_stopwords, remove_stopwords
from src.preprocessing.tokenizer import tokenize


def preprocess_text(text: str, stopwords: set[str], backend: str = "pyvi") -> list[str]:
    """Văn bản thô -> danh sách token sạch (làm sạch, tách từ, hạ chữ thường, bỏ stopwords)."""
    # Tách từ khi còn dấu câu, số và chữ hoa (để pyvi nhận ra ranh giới cụm và tên riêng),
    # rồi mới lọc dấu câu, số và hạ chữ thường
    tokens = keep_words(tokenize(strip_markup(text), backend))
    return remove_stopwords(tokens, stopwords)


def build_clean(backend: str = "pyvi") -> pd.DataFrame:
    df = pd.read_csv(config.ARTICLES_CSV, encoding="utf-8")
    stopwords = load_stopwords()
    print(f"Đọc {len(df)} bài; {len(stopwords)} stopwords; tách từ bằng {backend}")

    # Ghép tiêu đề + nội dung (tiêu đề chứa từ khóa chủ đề rất mạnh)
    raw = (df["title"].fillna("") + ". " + df["content"].fillna("")).tolist()
    df["tokens"] = [" ".join(preprocess_text(t, stopwords, backend)) for t in raw]

    empty = (df["tokens"].str.len() == 0).sum()
    if empty:
        print(f"CẢNH BÁO: {empty} bài rỗng sau tiền xử lý")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df[config.COLUMNS + ["tokens"]].to_csv(config.ARTICLES_CLEAN_CSV, index=False, encoding="utf-8")
    n = df["tokens"].str.split().str.len()
    print(f"Ghi {config.ARTICLES_CLEAN_CSV.relative_to(config.ROOT)}: "
          f"token/bài TB {n.mean():.0f} (min {n.min()}, max {n.max()})")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["pyvi", "underthesea"], default="pyvi")
    build_clean(ap.parse_args().backend)