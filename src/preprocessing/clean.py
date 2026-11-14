"""Làm sạch văn bản tiếng Việt, chia làm 2 giai đoạn quanh bước tách từ (tokenizer.py).

Thứ tự đúng của cả quy trình:
    strip_markup(text)  ->  tokenize(...)  ->  keep_words(tokens)  ->  remove_stopwords(...)

Lý do tách từ TRƯỚC khi lọc dấu câu và số:
    - Dấu câu là ranh giới giữa các cụm từ. Nếu bỏ dấu chấm, dấu phẩy trước thì pyvi nối nhầm các
      từ ở hai bên, ví dụ "Hồng Quân, Phó Cục trưởng" thành "hồng_quân_phó_cục_trưởng".
    - Số cũng vậy: bỏ số trước thì "U17 Việt Nam" thành "u việt nam" rồi bị nối thành "u_việt_nam".
    - Giữ nguyên chữ hoa khi tách từ để pyvi nhận ra tên riêng ("Việt Nam" -> "Việt_Nam").
      Hạ chữ thường chỉ làm sau khi tách từ.
"""
import re
import unicodedata

_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY = re.compile(r"&[a-z]+;|&#\d+;")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_SPACES = re.compile(r"[ \t\r\f\v]+")
# Token hợp lệ: chỉ gồm chữ cái, các âm tiết của từ ghép nối nhau bằng "_" (không có số, dấu câu)
_WORD = re.compile(r"[^\W\d_]+(?:_[^\W\d_]+)*")


def strip_markup(text) -> str:
    # Giai đoạn 1 (TRƯỚC khi tách từ): chỉ bỏ "rác" kỹ thuật, giữ nguyên dấu câu, số và chữ hoa.
    if not isinstance(text, str):                 # ô trống (NaN) thì trả về chuỗi rỗng
        return ""
    text = unicodedata.normalize("NFC", text)     # thống nhất cách mã hóa dấu tiếng Việt
    text = _HTML_TAG.sub(" ", text)
    text = _HTML_ENTITY.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = text.replace("\n", " . ")              # hết đoạn thì coi như hết câu, để pyvi không nối hai đoạn
    return _SPACES.sub(" ", text).strip()


def keep_words(tokens: list[str]) -> list[str]:
    # Giai đoạn 2 (SAU khi tách từ): bỏ token là dấu câu, số, ký hiệu; hạ chữ thường các token còn lại.
    # pyvi dính dấu chấm cuối câu vào chữ viết tắt in hoa ("AI.", "FIFA."), nên cắt dấu chấm cuối trước.
    stripped = (t.rstrip(".") for t in tokens)
    return [t.lower() for t in stripped if _WORD.fullmatch(t)]