"""Test nhỏ cho tiền xử lý (chạy: python -m pytest tests -q  hoặc python -m tests.test_preprocessing)."""
from src.preprocessing.clean import clean_text
from src.preprocessing.stopwords import load_stopwords, remove_stopwords
from src.preprocessing.build_clean import preprocess_text


def test_clean_removes_noise():
    raw = "<p>Giá vàng tăng 1.200 đồng, xem https://vnexpress.net/abc hoặc mail a@b.com!</p>"
    out = clean_text(raw)
    assert out == "giá vàng tăng đồng xem hoặc mail"


def test_clean_keeps_case_when_asked():
    assert clean_text("Việt Nam thắng!", lowercase=False) == "Việt Nam thắng"


def test_clean_non_string():
    assert clean_text(None) == ""


def test_stopwords_loaded_and_filtered():
    sw = load_stopwords()
    assert "cho_biết" in sw and "và" in sw
    assert remove_stopwords(["và", "bóng_đá", "a", "hay"], sw) == ["bóng_đá"]


def test_proper_noun_kept_as_one_token():
    sw = load_stopwords()
    tokens = preprocess_text("Đội tuyển Việt Nam thắng trận bóng đá.", sw)
    assert "việt_nam" in tokens and "bóng_đá" in tokens


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK", name)
