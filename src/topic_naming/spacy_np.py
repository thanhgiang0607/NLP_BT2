"""Phân tích bổ trợ bằng spaCy (spacy.blank('vi')) + pyvi đếm từ ghép/cụm danh từ mỗi cụm.

Lưu ý:
    spaCy hiện chưa có model tiếng Việt chính thức (không có bộ gán nhãn từ loại POS/NP sẵn),
    vì vậy dùng `spacy.blank("vi")` tách token trên văn bản đã được `pyvi` nối từ ghép bằng `_`
    (ví dụ "bóng_đá", "đội_tuyển"), sau đó đếm tần suất từ ghép để đối chiếu bổ trợ với
    top từ khóa TF-IDF ở bước sinh tên chủ đề.

Phụ trách: Xuân Hoàng
"""
from collections import Counter

import spacy

from src.preprocessing.tokenizer import tokenize


def top_compound_nouns(docs, labels, n: int = 10) -> dict[int, list[str]]:
    """Dùng spaCy tách token và đếm top-n từ ghép (có '_') phổ biến nhất mỗi cụm."""
    nlp = spacy.blank("vi")
    out: dict[int, list[str]] = {}
    for cid in sorted(set(labels)):
        cnt: Counter[str] = Counter()
        for doc, lab in zip(docs, labels):
            if lab != cid:
                continue
            text = str(doc or "")
            # Dữ liệu articles_clean.csv đã có sẵn '_'; chỉ gọi pyvi khi truyền văn bản thô
            if text and "_" not in text:
                text = " ".join(tokenize(text)).lower()
            # Chỉ giữ token có dấu '_' (từ ghép/cụm danh từ tiếng Việt) để loại bớt từ đơn rời rạc
            cnt.update(tok.text for tok in nlp(text) if "_" in tok.text)
        out[int(cid)] = [w for w, _ in cnt.most_common(n)]
    return out
