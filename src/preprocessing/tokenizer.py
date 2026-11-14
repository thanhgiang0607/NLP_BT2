"""Tách từ tiếng Việt bằng pyvi / underthesea (bóng đá -> bóng_đá)."""
import warnings


def tokenize(text: str, backend: str = "pyvi") -> list[str]:
    """Tách từ, trả về danh sách token. Từ ghép được nối bằng dấu gạch dưới, ví dụ 'bóng_đá'."""
    if not text:
        return []
    if backend == "pyvi":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")        # tắt cảnh báo vô hại của pyvi
            from pyvi import ViTokenizer
        return ViTokenizer.tokenize(text).split()  # pyvi trả về chuỗi, .split() biến thành danh sách
    if backend == "underthesea":
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").split()
    raise ValueError(f"backend phải là 'pyvi' hoặc 'underthesea', nhận được: {backend!r}")


def tokenize_corpus(texts, backend: str = "pyvi") -> list[list[str]]:
    """Tách từ cho cả một danh sách văn bản."""
    return [tokenize(t, backend) for t in texts]