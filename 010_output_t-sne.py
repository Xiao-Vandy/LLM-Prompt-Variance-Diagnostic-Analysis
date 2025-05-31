import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
# os.makedirs("nlp1", exist_ok=True)
# === Load & concat CSVs ===
# files = [
#     "outputs3/results_gpt2-large.csv",
#     "outputs3/results_gpt-neo-1.3B.csv",
#     "outputs3/results_gpt-3.5-turbo.csv",
#     "outputs3/results_Llama-2-7b-chat-hf.csv",
#     "outputs3/results_Mistral-7B-Instruct-v0.2.csv",
# ]
files = [
    "outputs_1/results_gpt2-large-0.2-1.3_fixed.csv",
    "outputs_1/results_gpt-3.5-turbo-0.2-1.3_fixed.csv",
    "outputs_1/results_gpt-neo-1.3B_fixed.csv",
    "outputs_1/results_meta-llama_Llama-2-7b-chat-hf_20250525_121538.csv",
    "outputs_1/results_Mistral-7B-Instruct-v0.2_fixed.csv",
    # "outputs_1/results_HuggingFaceTB_SmolLM-360M_20250528_105121.csv",
    # "outputs_1/results_microsoft_phi-2_20250527_184219.csv",
    # "outputs_1/results_openchat-3.5-1210_20250527_185953.csv",
    # "outputs_1/results_stabilityai_StableBeluga-7B_20250527_210924.csv",
    # "outputs_1/results_mythomax-I2-13B-Q6_20250527_180819.csv",
]
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# 什么时候选它而不是 MiniLM？
# 更稳定语义判断，语义漂移打得更清晰	✅ MPNet
# 想要更快速度、轻量调试	MiniLM
# 追求学术兼容性（论文里大家都在用）	SBERT / MPNet 都行
# 用来做 final 语义指标（PBSS）	MPNet 稳一点没毛病


# === Embedding + PCA ===
model_sbert = SentenceTransformer("all-mpnet-base-v2")
embeddings = model_sbert.encode(df["output"].fillna("").tolist(), normalize_embeddings=True)

pca = PCA(n_components=3)
pca_embed = pca.fit_transform(embeddings)

# === Add output_len for stylistic signal ===
df["output_len"] = df["output"].apply(lambda x: len(str(x).split()))

# model_name_map = {
#     'gpt2-large': 'gpt2-large',
#     'EleutherAI_gpt-neo-1.3B': 'gpt-neo-1.3B',
#     'gpt-3.5-turbo': 'gpt-3.5-turbo',
#     'meta-llama_Llama-2-7b-chat-hf': 'Llama-2-7b-chat-hf',
#     'Mistral-7B-Instruct-v0.2': 'Mistral-7B-instruct-v0.2'
# }

# model_name_map = {
#     'HuggingFaceTB_SmolLM-360M': 'Hugging-Face_SmolLM-360M',
#     'microsoft_phi-2': 'Microsoft_phi-2',
#     'stabilityai_StableBeluga-7B': 'StableBeluga-7B',
#     'openchat-3.5-1210': 'Openchat-3.5-1210',
#     'mythomax-I2-13B-Q6': 'Mythomax-I2-13B-Q6'
# }

model_name_map = {
    'gpt2-large': 'gpt2-large',
    'EleutherAI_gpt-neo-1.3B': 'gpt-neo-1.3B',
    'gpt-3.5-turbo': 'gpt-3.5-turbo',
    'meta-llama_Llama-2-7b-chat-hf': 'Llama-2-7b-chat-hf',
    'Mistral-7B-Instruct-v0.2': 'Mistral-7B-instruct-v0.2',
    'HuggingFaceTB_SmolLM-360M': 'Hugging-Face_SmolLM-360M',
    'microsoft_phi-2': 'Microsoft_phi-2',
    'stabilityai_StableBeluga-7B': 'StableBeluga-7B',
    'openchat-3.5-1210': 'Openchat-3.5-1210',
    'mythomax-I2-13B-Q6': 'Mythomax-I2-13B-Q6'
}
df["model"] = df["model"].map(model_name_map)

# # 全局语义人格图谱：T-SNE 彩图
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_result = tsne.fit_transform(embeddings)

df["tsne_x"] = tsne_result[:, 0]
df["tsne_y"] = tsne_result[:, 1]

plt.figure(figsize=(10, 6))  # 更宽，给 legend 留位置
sns.scatterplot(data=df, x="tsne_x", y="tsne_y", hue="model", alpha=0.7)
plt.title("Model Semantic Behavior Map (T-SNE)")
plt.legend(loc="upper right", frameon=True, fontsize='small')
plt.tight_layout()
plt.savefig(f"figure/T-SNE1_10p_3.pdf", bbox_inches="tight")
plt.show()
plt.clf()  # 清空当前 figure
plt.close()  # 关闭当前 plot（释放内存）


plt.figure(figsize=(10, 6))  # 更宽，给 legend 留位置
sns.scatterplot(data=df, x="tsne_x", y="tsne_y", hue="origin",alpha=0.7)
plt.title("Model Semantic Behavior Map (T-SNE)")
plt.legend(loc="upper right", frameon=True, fontsize='small')
plt.tight_layout()
plt.savefig(f"figure/T-SNE2_10p_3.pdf", bbox_inches="tight")
plt.show()
plt.clf()  # 清空当前 figure
plt.close()  # 关闭当前 plot（释放内存）

from collections import Counter
for model, g in df.groupby("model"):
    model_repeats = Counter(g["output"])
    for text, count in model_repeats.items():
        if count > 3:
            prompt_ids = g[g["output"] == text]["prompt_id"].unique()
            joined_ids = ", ".join(prompt_ids[:5])  # truncate if needed
            print(f"[{model}] 复读：'{text[:30]}' x {count} | prompt_ids: {joined_ids}")
            repeat_count = sum(1 for c in model_repeats.values() if c > 3)
            print(f"{model} 有 {repeat_count} 个复读输出")