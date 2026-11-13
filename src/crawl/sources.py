"""Cấu hình nguồn crawl: URL chuyên mục, mẫu link bài, selector nội dung của từng báo.

Phụ trách: Thanh Giang
"""

# Mỗi chủ đề -> {nguồn: đường dẫn chuyên mục}
CATEGORY_PATHS = {
    "thoi_su": {
        "vnexpress": "https://vnexpress.net/thoi-su",
        "tuoitre": "https://tuoitre.vn/thoi-su.htm",
        "thanhnien": "https://thanhnien.vn/thoi-su.htm",
        "dantri": "https://dantri.com.vn/xa-hoi.htm",
    },
    "kinh_te": {
        "vnexpress": "https://vnexpress.net/kinh-doanh",
        "tuoitre": "https://tuoitre.vn/kinh-doanh.htm",
        "thanhnien": "https://thanhnien.vn/kinh-te.htm",
        "dantri": "https://dantri.com.vn/kinh-doanh.htm",
    },
    "the_thao": {
        "vnexpress": "https://vnexpress.net/the-thao",
        "tuoitre": "https://tuoitre.vn/the-thao.htm",
        "thanhnien": "https://thanhnien.vn/the-thao.htm",
        "dantri": "https://dantri.com.vn/the-thao.htm",
    },
    "giai_tri": {
        "vnexpress": "https://vnexpress.net/giai-tri",
        "tuoitre": "https://tuoitre.vn/giai-tri.htm",
        "thanhnien": "https://thanhnien.vn/giai-tri.htm",
        "dantri": "https://dantri.com.vn/giai-tri.htm",
    },
    "cong_nghe": {
        "vnexpress": "https://vnexpress.net/so-hoa",
        "tuoitre": "https://tuoitre.vn/cong-nghe.htm",
        "thanhnien": "https://thanhnien.vn/cong-nghe.htm",
        "dantri": "https://dantri.com.vn/cong-nghe.htm",
    },
    "suc_khoe": {
        "vnexpress": "https://vnexpress.net/suc-khoe",
        "tuoitre": "https://tuoitre.vn/suc-khoe.htm",
        "thanhnien": "https://thanhnien.vn/suc-khoe.htm",
        "dantri": "https://dantri.com.vn/suc-khoe.htm",
    },
}

# Tên chuyên mục cấp 1 trên breadcrumb được chấp nhận cho từng chủ đề
# (dùng để loại link "đọc nhiều"/"liên quan" thuộc chuyên mục khác lọt vào trang danh sách)
ACCEPTED_SECTIONS = {
    "thoi_su": {"thời sự", "xã hội"},
    "kinh_te": {"kinh doanh", "kinh tế"},
    "the_thao": {"thể thao"},
    "giai_tri": {"giải trí"},
    "cong_nghe": {"số hóa", "công nghệ", "khoa học công nghệ"},
    "suc_khoe": {"sức khỏe"},
}

# Regex cho đường dẫn bài viết (áp dụng lên URL tuyệt đối)
ARTICLE_URL_PATTERNS = {
    "vnexpress": r"^https://vnexpress\.net/[a-z0-9-]+-\d{6,}\.html$",
    "tuoitre": r"^https://tuoitre\.vn/[a-z0-9-]+-\d{12,}\.htm$",
    "thanhnien": r"^https://thanhnien\.vn/[a-z0-9-]+-\d{12,}\.htm$",
    "dantri": r"^https://dantri\.com\.vn/[a-z0-9-]+(/[a-z0-9-]+)?/[a-z0-9-]+-\d{12,}\.htm$",
}

# Selector trên trang bài viết
SELECTORS = {
    "vnexpress": {"title": "h1.title-detail", "sapo": "p.description", "body": "article.fck_detail"},
    "tuoitre": {"title": "h1.detail-title", "sapo": "h2.detail-sapo", "body": "div.detail-content"},
    "thanhnien": {"title": "h1.detail-title", "sapo": "h2.detail-sapo", "body": "div.detail-content"},
    "dantri": {"title": "h1", "sapo": "[data-slot=sapo]", "body": "#articleContent [data-slot=content]"},
}
