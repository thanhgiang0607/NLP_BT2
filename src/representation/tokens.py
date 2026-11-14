"""Hàm tách token dùng chung cho CountVectorizer / TfidfVectorizer."""


def split_tokens(doc: str) -> list[str]:
    """Cột `tokens` đã tách từ sẵn, các token cách nhau bằng khoảng trắng nên chỉ cần split."""
    return doc.split()