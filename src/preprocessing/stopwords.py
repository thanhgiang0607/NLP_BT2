"""Load và lọc stopwords (data/stopwords/vietnamese_stopwords.txt)."""
from pathlib import Path

from src import config


def load_stopwords(path: Path | str = config.STOPWORDS_PATH) -> set[str]:
    """Đọc file stopwords thành một tập hợp (set). Bỏ dòng trống và dòng chú thích bắt đầu bằng #."""
    words = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip().lower()
        if line and not line.startswith("#"):
            words.add(line)
    return words


def remove_stopwords(tokens: list[str], stopwords: set[str], min_len: int = 2) -> list[str]:
    """Bỏ stopwords và các token quá ngắn (dưới min_len ký tự, không tính dấu gạch dưới)."""
    return [t for t in tokens if t not in stopwords and len(t.replace("_", "")) >= min_len]