"""Cấu hình dùng chung cho cả nhóm: đường dẫn, seed, chủ đề, tham số mặc định."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Đường dẫn
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
STOPWORDS_PATH = DATA_DIR / "stopwords" / "vietnamese_stopwords.txt"
ARTICLES_CSV = PROCESSED_DIR / "articles.csv"
ARTICLES_CLEAN_CSV = PROCESSED_DIR / "articles_clean.csv"

MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
TOPIC_NAMES_DIR = OUTPUTS_DIR / "topic_names"

# Tái lập kết quả
RANDOM_STATE = 42

# Chủ đề: tên file trong data/raw -> nhãn hiển thị (true_label)
TOPICS = {
    "thoi_su": "Thời sự",
    "kinh_te": "Kinh tế",
    "the_thao": "Thể thao",
    "giai_tri": "Giải trí",
    "cong_nghe": "Công nghệ",
    "suc_khoe": "Sức khỏe",
}

# Cột chuẩn của file dữ liệu
COLUMNS = ["id", "title", "content", "url", "source", "date", "true_label"]

# Thu thập dữ liệu
ARTICLES_PER_TOPIC = 20
MIN_WORDS = 150

# Biểu diễn văn bản
MIN_DF = 2
MAX_DF = 0.8
NGRAM_RANGE = (1, 2)

# LSA
LSA_COMPONENTS = 80

# Word2Vec (tách riêng khỏi LSA: đổi số chiều LSA không làm đổi Word2Vec)
W2V_SIZE = 80

# Phân cụm
K_RANGE = range(2, 13)
