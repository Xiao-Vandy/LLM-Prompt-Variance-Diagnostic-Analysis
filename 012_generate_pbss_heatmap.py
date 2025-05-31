import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os
from scipy.stats import zscore

# === SBERT sentence embedding ===
# model_sbert = SentenceTransformer("all-MiniLM-L6-v2")

# model_sbert = SentenceTransformer("paraphrase-MiniLM-L12-v2")
model_sbert = SentenceTransformer("all-mpnet-base-v2")

# model_name="MiniLM-L6-v2"

# model_name="MiniLM-L12-v2"
#
model_name="mpnet-base-v2"

# filefolder="pbss_MiniLM-L6"
# filefolder="pbss_MiniLM-L12"
filefolder="pbss_mpnet"

# ⛔注意这里要改三个位置

# === Compute PBSS matrix ===
def compute_pbss_matrix(vectors):
    n = len(vectors)
    pbss = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                score = 1 - cosine_similarity([vectors[i]], [vectors[j]])[0][0]
                pbss[i][j] = score
    return pbss

# === Heatmap plotting ===
def plot_heatmap_raw(pbss, labels, title, heatmap_name,vmax_global=0.8,):
    n = len(labels)
    base_size = 0.5  # 每个cell分配的图像宽度（英寸）
    figsize = (max(8, n * base_size), max(6, n * base_size))  # 最小限制防止太小

    plt.figure(figsize=figsize)
    sns.heatmap(pbss, annot=True, fmt=".2f",
                xticklabels=labels, yticklabels=labels,
                cmap="coolwarm", vmin=0, vmax=vmax_global)

    plt.title(title)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{filefolder}/{model_name}_{heatmap_name}_raw.pdf", bbox_inches="tight")
    # plt.show()
    plt.clf()  # 清空当前 figure
    plt.close()  # 关闭当前 plot（释放内存）


def plot_heatmap_zscore_global(pbss, labels, title, heatmap_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import zscore
    import numpy as np

    flat_pbss = np.array(pbss)
    non_diag = flat_pbss[~np.eye(flat_pbss.shape[0], dtype=bool)]
    z_vals = zscore(non_diag)

    # 回填到 n x n 结构
    z_pbss = np.zeros_like(flat_pbss)
    k = 0
    for i in range(flat_pbss.shape[0]):
        for j in range(flat_pbss.shape[1]):
            if i != j:
                z_pbss[i][j] = z_vals[k]
                k += 1

    n = len(labels)
    base_size = 0.5
    figsize = (max(8, n * base_size), max(6, n * base_size))

    plt.figure(figsize=figsize)

    vmax = np.abs(z_pbss).max()
    sns.heatmap(z_pbss, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-vmax, vmax=vmax,
                xticklabels=labels, yticklabels=labels)

    plt.title(f"{title} (Global Z-Score)")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    # ⛔ DON'T SHOW FIRST
    plt.savefig(f"{filefolder}/{model_name}_{heatmap_name}_zscore_global.pdf", bbox_inches="tight")
    # Optional: show only during debugging
    # plt.show()
    plt.clf()
    plt.close()


def plot_heatmap_zscore_per_row(pbss, labels, title, heatmap_name):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import zscore
    import numpy as np

    pbss = np.array(pbss)
    z_pbss = np.zeros_like(pbss)

    for i in range(pbss.shape[0]):
        row = pbss[i, :]
        row_no_diag = np.delete(row, i)  # remove self-similarity
        z_row = zscore(row_no_diag)

        idx = 0
        for j in range(pbss.shape[1]):
            if i == j:
                continue
            z_pbss[i][j] = z_row[idx]
            idx += 1

    n = len(labels)
    base_size = 0.5
    figsize = (max(8, n * base_size), max(6, n * base_size))

    plt.figure(figsize=figsize)

    vmax = np.abs(z_pbss).max()
    sns.heatmap(z_pbss, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-vmax, vmax=vmax,
                xticklabels=labels, yticklabels=labels)

    plt.title(f"{title} (Row-wise Z-Score)")
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{filefolder}/{model_name}_{heatmap_name}_zscore_per_row.pdf", bbox_inches="tight")
    # plt.show()
    plt.clf()
    plt.close()



# === Main summary script ===
def generate_pbss_summary(csv_path):
    df = pd.read_csv(csv_path)
    # 每个 origin → 对应一个 15 prompt 组（3 variant × 5 dimension）
    # 每个 model 和 temperature 给出 15 个输出
    # 你拿这 15 个输出生成 PBSS → 就是一个 15 × 15 的 cosine similarity matrix
    # 看某个 origin 的 prompt group 内部是否行为一致
    # 看哪个 dimension 的 prompt variants 在模型眼里完全不同（= drift）
    # 看高温 vs 低温下模型的 行为稳定性差异

    # grouped = df.groupby(["origin", "level", "model", "temperature"])
    grouped = df.groupby(["origin", "model", "temperature"])

    for (origin, model, temperature), group in grouped:
        group = group.sort_values("variant_id")
        texts = group["output"].fillna("").tolist()
        labels = [f"V{i + 1}" for i in range(len(texts))]

        vectors = model_sbert.encode(texts, normalize_embeddings=True)
        print(f"文本数量: {len(texts)} | 向量数量: {len(vectors)}")
        pbss = compute_pbss_matrix(vectors)

        title = f"PBSS Heatmap - Origin: {origin} | Model: {model} | Temp: {temperature}"
        heatmap_name = f"PBSS_{origin}_{model}_{temperature}"
        plot_heatmap_raw(pbss, labels, title,heatmap_name)
        plot_heatmap_zscore_global(pbss, labels, title, heatmap_name)
        plot_heatmap_zscore_per_row(pbss, labels, title, heatmap_name)

# === Run example ===
generate_pbss_summary("outputs/results_gpt2-large-0.2-1.3_fixed.csv")
generate_pbss_summary("outputs/results_gpt-neo-1.3B_fixed.csv")
generate_pbss_summary("outputs/results_gpt-3.5-turbo-0.2-1.3_fixed.csv")
generate_pbss_summary("outputs/results_meta-llama_Llama-2-7b-chat-hf_20250525_121538.csv")
generate_pbss_summary("outputs/results_Mistral-7B-Instruct-v0.2_fixed.csv")