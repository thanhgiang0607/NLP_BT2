"""Làm sạch văn bản: bỏ HTML, URL, số, ký tự đặc biệt, lowercase.

Phụ trách: Thanh Hằng

Thứ tự xử lý (quan trọng, đừng đảo):
    1. Chuẩn hóa Unicode NFC (tránh "ả" bị tách thành 2 ký tự khác nhau).
    2. Bỏ thẻ HTML, bỏ URL / email.
    3. Chữ thường (tuỳ chọn, xem lưu ý bên dưới).
    4. Bỏ số (kể cả số có dấu . , như 1.200, 3,5).
    5. Bỏ ký tự đặc biệt / dấu câu, chỉ giữ chữ cái (có dấu tiếng Việt) và khoảng trắng.
    6. Gộp khoảng trắng.

Bước làm sạch chạy TRƯỚC khi tách từ, để tách từ không bị nhiễu bởi dấu câu và số.

LƯU Ý: pyvi/underthesea dùng chữ hoa để nhận ra tên riêng ("Việt Nam" -> "Việt_Nam").
Nếu hạ chữ thường trước khi tách từ thì "việt nam" bị tách thành 2 từ rời. Vì vậy pipeline gọi
clean_text(..., lowercase=False), tách từ xong mới hạ chữ thường từng token.
"""
"""Làm sạch văn bản: bỏ HTML, URL, số, ký tự đặc biệt, lowercase."""
import re
import unicodedata

# Mỗi dòng dưới đây là một "mẫu" (regex) để tìm một loại nhiễu trong văn bản
_HTML_TAG = re.compile(r"<[^>]+>")                      # thẻ HTML như <p>, </div>
_HTML_ENTITY = re.compile(r"&[a-z]+;|&#\d+;")           # &nbsp; &amp; ...
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_NUMBER = re.compile(r"\d+")                            # mọi chuỗi chữ số
# Giữ lại chuỗi chữ cái (kể cả có dấu tiếng Việt) và khoảng trắng; bỏ mọi thứ còn lại
_NON_LETTER = re.compile(r"[^\W\d_]+|\s+", re.UNICODE)
_SPACES = re.compile(r"\s+")


def clean_text(text, lowercase: bool = True) -> str:
    """Làm sạch một văn bản. lowercase=False để giữ chữ hoa (cần cho bước tách từ)."""
    if not isinstance(text, str):                       # gặp ô trống thì trả về chuỗi rỗng
        return ""
    text = unicodedata.normalize("NFC", text)           # thống nhất cách mã hóa dấu tiếng Việt
    text = _HTML_TAG.sub(" ", text)
    text = _HTML_ENTITY.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    if lowercase:
        text = text.lower()
    text = _NUMBER.sub(" ", text)
    # findall lấy ra các cụm chữ cái, dấu câu và ký hiệu bị loại, rồi nối lại bằng khoảng trắng
    text = " ".join(_NON_LETTER.findall(text))
    return _SPACES.sub(" ", text).strip()               # gộp nhiều khoảng trắng thành một


def clean_corpus(texts, lowercase: bool = True) -> list[str]:
    """Làm sạch cả một danh sách văn bản."""
    return [clean_text(t, lowercase) for t in texts]