"""Test nhỏ cho tiền xử lý. Chạy: python -m pytest tests -q   (hoặc python -m tests.test_preprocessing)"""
from src.preprocessing.build_clean import preprocess_text
from src.preprocessing.clean import keep_words, strip_markup
from src.preprocessing.stopwords import load_stopwords, remove_stopwords


def test_strip_markup_removes_html_url_email():
    raw = "<p>Xem https://vnexpress.net/abc hoặc a@b.com</p>"
    assert strip_markup(raw) == "Xem hoặc"


def test_strip_markup_keeps_case_digits_punctuation():
    # Giai đoạn 1 chưa được động đến dấu câu, số, chữ hoa (để pyvi tách từ đúng)
    assert strip_markup("Việt Nam thắng 3-0!") == "Việt Nam thắng 3-0!"


def test_strip_markup_non_string():
    assert strip_markup(None) == ""


def test_keep_words_drops_punctuation_and_numbers():
    tokens = ["Việt_Nam", "thắng", "3", "-", "0", "!", "U17", "AI."]
    assert keep_words(tokens) == ["việt_nam", "thắng", "ai"]


def test_stopwords_loaded_and_filtered():
    sw = load_stopwords()
    assert "và" in sw and "diễn" in sw
    assert remove_stopwords(["và", "bóng_đá", "a", "hay"], sw) == ["bóng_đá"]


def test_proper_noun_kept_as_one_token():
    tokens = preprocess_text("Đội tuyển Việt Nam thắng trận bóng đá.", load_stopwords())
    assert "việt_nam" in tokens and "bóng_đá" in tokens


def test_comma_does_not_glue_words():
    # Trước đây bỏ dấu phẩy trước khi tách từ làm pyvi nối "Hồng Quân" với "Phó Cục trưởng"
    tokens = preprocess_text("Bà Hồng Quân, Phó Cục trưởng Cục Quản lý thị trường.", load_stopwords())
    assert "hồng_quân" in tokens
    assert not any("quân" in t and "phó" in t for t in tokens)


def test_number_does_not_glue_words():
    # Trước đây bỏ số trước khi tách từ biến "U17 Việt Nam" thành "u_việt_nam"
    tokens = preprocess_text("Chủ tịch FIFA nói về World Cup có U17 Việt Nam tham dự.", load_stopwords())
    assert "việt_nam" in tokens and "u_việt_nam" not in tokens


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
