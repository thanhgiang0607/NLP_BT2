"""Biểu đồ EDA, scatter 2D (t-SNE/UMAP), wordcloud theo cụm, confusion matrix."""
from collections import Counter
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")  # vẽ và lưu ảnh được cả khi không có cửa sổ hiển thị
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from PIL import ImageDraw
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

from src import config

# Bảng màu 12 màu, phân biệt được cả khi in đen trắng hoặc người bị mù màu
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
           "#56B4E9", "#F0E442", "#999999", "#882255", "#44AA99",
           "#117733", "#AA4499"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",   # font có đủ dấu tiếng Việt
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def _save(fig, name):
    """Lưu hình vào outputs/figures/<name>.png (nếu có name) rồi trả lại hình."""
    if name:
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(config.FIGURES_DIR / f"{name}.png")
    return fig


def _colors(labels):
    """Gán mỗi nhãn một màu cố định (sắp theo tên) để các hình dùng chung màu."""
    uniq = sorted(set(labels), key=str)
    return {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(uniq)}


# ---------------------------------------------------------------- EDA
def plot_topic_counts(labels, name="eda_topic_counts"):
    """Số bài theo chủ đề (cột ngang, có ghi số)."""
    counts = pd.Series(labels).value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.barh(counts.index, counts.values, color=PALETTE[0])
    for y, v in enumerate(counts.values):
        ax.text(v + 0.2, y, str(v), va="center", fontsize=9)
    ax.set_xlabel("Số bài")
    ax.set_title("Số bài theo chủ đề")
    return _save(fig, name)


def plot_length_distribution(lengths, labels, name="eda_length_tokens", xlabel="Số từ mỗi bài"):
    """Biểu đồ cột độ dài bài (toàn bộ) và boxplot độ dài theo chủ đề."""
    df = pd.DataFrame({"len": lengths, "label": labels})
    order = df.groupby("label")["len"].median().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(df["len"], bins=25, color=PALETTE[0], edgecolor="white")
    axes[0].axvline(df["len"].median(), color=PALETTE[3], ls="--",
                    label=f"Trung vị = {df['len'].median():.0f}")
    axes[0].set(xlabel=xlabel, ylabel="Số bài", title="Phân bố độ dài bài")
    axes[0].legend(frameon=False)
    sns.boxplot(data=df, y="label", x="len", order=order, color=PALETTE[0], ax=axes[1], fliersize=3)
    axes[1].set(xlabel=xlabel, ylabel="", title="Độ dài bài theo chủ đề")
    fig.tight_layout()
    return _save(fig, name)


def plot_top_words(docs, labels, n=10, name="eda_top_words"):
    """Top n từ phổ biến nhất của từng chủ đề."""
    labs = sorted(set(labels), key=str)
    cols = 3
    rows = int(np.ceil(len(labs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    colors = _colors(labels)
    docs, labels = list(docs), np.asarray(labels)
    for ax, lab in zip(axes, labs):
        cnt = Counter(w for d, l in zip(docs, labels) if l == lab for w in d.split())
        words, vals = zip(*cnt.most_common(n)[::-1])
        ax.barh([w.replace("_", " ") for w in words], vals, color=colors[lab])
        ax.set_title(str(lab), fontsize=11)
    for ax in axes[len(labs):]:
        ax.axis("off")
    fig.suptitle(f"Top {n} từ theo chủ đề (sau tiền xử lý)", y=1.01)
    fig.tight_layout()
    return _save(fig, name)


# ---------------------------------------------------------------- Scatter 2D
def reduce_2d(X, method="tsne", random_state=config.RANDOM_STATE):
    """Giảm chiều xuống 2D để vẽ. Ma trận nhiều cột được rút xuống 50 chiều bằng SVD trước."""
    n = X.shape[0]
    if X.shape[1] > 50:
        X = TruncatedSVD(n_components=min(50, n - 1, X.shape[1] - 1),
                         random_state=random_state).fit_transform(X)
    X = np.asarray(X)
    if method == "tsne":
        perplexity = max(2, min(30, (n - 1) / 3))  # perplexity phải nhỏ hơn số mẫu
        return TSNE(n_components=2, metric="cosine", perplexity=perplexity, init="random",
                    random_state=random_state).fit_transform(X)
    if method == "umap":
        import umap
        return umap.UMAP(n_components=2, metric="cosine", n_neighbors=min(15, n - 1),
                         random_state=random_state).fit_transform(X)
    raise ValueError("method phải là 'tsne' hoặc 'umap'")


def plot_scatter_2d(X, labels, method="tsne", title=None, name=None, coords=None,
                    legend_title="Nhãn"):
    """Scatter 2D, mỗi bài là một chấm, tô màu theo `labels` (cụm hoặc chủ đề thật)."""
    if coords is None:
        coords = reduce_2d(X, method)
    labels = np.asarray(labels)
    colors = _colors(labels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for lab in sorted(set(labels), key=str):
        m = labels == lab
        ax.scatter(coords[m, 0], coords[m, 1], s=42, color=colors[lab], edgecolor="white",
                   linewidth=0.6, label=f"{lab} ({m.sum()})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.set_title(title or f"Phân bố bài viết ({method.upper()})")
    ax.legend(title=legend_title, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return _save(fig, name)


# ---------------------------------------------------------------- Wordcloud
# Font tiếng Việt đi kèm repo (Be Vietnam Pro). Không tìm thấy file thì dùng DejaVu Sans.
FONT_DIR = config.ROOT / "assets" / "fonts"
_VI_FONTS = ["BeVietnamPro-Regular.ttf", "NotoSans-Regular.ttf"]


def _font_path():
    """Đường dẫn font cho wordcloud: ưu tiên font tiếng Việt trong assets/fonts, không có thì DejaVu."""
    for fname in _VI_FONTS:
        path = FONT_DIR / fname
        if path.exists():
            return str(path)
    return font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))


class _BoxDraw(ImageDraw.ImageDraw):
    """Khi wordcloud xếp chỗ (vẽ lên ảnh xám 'L'), tô kín cả khung chữ thay vì chỉ nét chữ.

    Mặc định wordcloud chỉ coi điểm mực là đã chiếm chỗ, nên nhét từ nhỏ vào khe giữa chữ và
    dấu tiếng Việt (vd. "vi phạm" nằm sát dấu nặng của "bộ" -> đọc thành "bô" + "vi phạm•").
    Ảnh màu cuối cùng (mode RGB) vẫn vẽ chữ bình thường.
    """

    def text(self, xy, text, fill=None, font=None, *args, **kwargs):
        if self.im.mode == "L":
            self.rectangle(self.textbbox(xy, text, font=font), fill=fill)
        else:
            super().text(xy, text, fill, font, *args, **kwargs)


@contextmanager
def _box_layout():
    """Tạm thay ImageDraw bên trong wordcloud bằng _BoxDraw (kiểm tra lại nếu nâng cấp wordcloud)."""
    import wordcloud.wordcloud as wcmod

    class _Shim:
        Draw = staticmethod(lambda im, mode=None: _BoxDraw(im, mode))

    orig, wcmod.ImageDraw = wcmod.ImageDraw, _Shim
    try:
        yield
    finally:
        wcmod.ImageDraw = orig


def make_wordcloud(freqs: dict, color="#0072B2"):
    """Tạo 1 wordcloud từ dict {từ: tần suất}. Từ càng quan trọng thì chữ càng to."""
    from wordcloud import WordCloud
    freqs = {k.replace("_", " "): v for k, v in freqs.items()}  # hiển thị "bóng đá" thay vì "bóng_đá"
    wc = WordCloud(width=700, height=420, background_color="white", font_path=_font_path(),
                   # ít từ hơn + bỏ chữ quá nhỏ (min_font_size) + chênh lệch cỡ chữ vừa phải
                   # (relative_scaling) để hình dày, chữ nào cũng đọc được
                   max_words=35, min_font_size=16, margin=3, prefer_horizontal=0.95,
                   relative_scaling=0.25,
                   color_func=lambda *a, **k: color, random_state=config.RANDOM_STATE)
    with _box_layout():
        return wc.generate_from_frequencies(freqs)


def group_frequencies(docs, labels, weights=None):
    """Đếm tần suất từ theo từng nhóm. `weights` (ví dụ idf) làm từ đặc trưng nổi hơn từ phổ biến."""
    freq = {}
    docs, labels = list(docs), np.asarray(labels)
    for lab in sorted(set(labels), key=str):
        cnt = Counter(w for d, l in zip(docs, labels) if l == lab for w in d.split())
        if weights:
            cnt = {w: c * weights.get(w, 1.0) for w, c in cnt.items()}
        freq[lab] = dict(cnt)
    return freq


def plot_wordcloud_grid(docs, labels, name=None, weights=None, cols=3, title=None):
    """Lưới wordcloud, mỗi ô là một cụm hoặc một chủ đề."""
    freq = group_frequencies(docs, labels, weights)
    labs = list(freq)
    colors = _colors(labels)
    rows = int(np.ceil(len(labs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, lab in zip(axes, labs):
        ax.imshow(make_wordcloud(freq[lab], colors[lab]), interpolation="bilinear")
        ax.set_title(str(lab), fontsize=12)
        ax.axis("off")
    for ax in axes[len(labs):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return _save(fig, name)


def idf_weights(vectorizer) -> dict:
    """Lấy trọng số idf của từng từ (chỉ từ đơn) từ TfidfVectorizer đã học."""
    return {t: w for t, w in zip(vectorizer.get_feature_names_out(), vectorizer.idf_)
            if " " not in t}


# ---------------------------------------------------------------- Confusion matrix
def contingency(true_labels, cluster_labels) -> pd.DataFrame:
    """Bảng chéo: mỗi hàng là một cụm, mỗi cột là một chủ đề thật, ô là số bài."""
    return pd.crosstab(pd.Series(cluster_labels, name="Cụm"),
                       pd.Series(true_labels, name="Chủ đề thật"))


def plot_confusion_matrix(true_labels, cluster_labels, name=None, title=None, normalize=False):
    """Heatmap so cụm với chủ đề thật. Cụm tốt thì mỗi hàng chỉ có 1 ô đậm."""
    ct = contingency(true_labels, cluster_labels)
    # Sắp xếp các cụm theo chủ đề chiếm nhiều nhất để dễ đọc
    order_cols = list(ct.columns)
    dominant = ct.idxmax(axis=1)
    ct = ct.loc[sorted(ct.index, key=lambda c: (order_cols.index(dominant[c]), -ct.loc[c].max()))]
    data = ct.div(ct.sum(axis=1), axis=0) if normalize else ct
    fig, ax = plt.subplots(figsize=(1.1 * len(ct.columns) + 2, 0.55 * len(ct) + 2.2))
    sns.heatmap(data, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", linewidths=0.5,
                linecolor="white", cbar_kws={"label": "Tỉ lệ" if normalize else "Số bài"}, ax=ax)
    ax.set_title(title or "Cụm so với chủ đề thật")
    ax.set_xlabel("Chủ đề thật")
    ax.set_ylabel("Cụm")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, name)


def purity(true_labels, cluster_labels) -> float:
    """Purity = số bài thuộc chủ đề chiếm nhiều nhất của mỗi cụm, cộng lại, chia tổng số bài."""
    ct = contingency(true_labels, cluster_labels)
    return float(ct.max(axis=1).sum() / ct.values.sum())