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
model_sbert = SentenceTransformer("paraphrase-MiniLM-L12-v2")
# model_sbert = SentenceTransformer("all-mpnet-base-v2")

# model_name="all-MiniLM-L6-v2"
model_name="paraphrase-MiniLM-L12-v2"
# model_name="all-mpnet-base-v2"

# filefolder="pbss_MiniLM-L6"
filefolder="pbss_MiniLM-L12"
# filefolder="pbss_mpnet"



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


# === Main summary script ===
def generate_pbss_summary(csv_path):
    df = pd.read_csv(csv_path)

    grouped = df.groupby(["origin", "model", "temperature"])

    summary = []

    for (origin, model, temperature), group in grouped:
        group = group.sort_values("variant_id")
        texts = group["output"].fillna("").tolist()
        labels = [f"V{i + 1}" for i in range(len(texts))]

        vectors = model_sbert.encode(texts, normalize_embeddings=True)
        print(f"文本数量: {len(texts)} | 向量数量: {len(vectors)}")
        pbss = compute_pbss_matrix(vectors)
        avg_pbss = np.mean([pbss[i][j] for i in range(len(pbss)) for j in range(len(pbss)) if i != j])

        summary.append({
            "origin": origin,
            "model": model,
            "temperature": temperature,
            "avg_pbss": avg_pbss
        })


    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(f"{filefolder}/pbss_summary_{model_name}.csv", mode='a', index=False, header=False)
    print(f"✅ PBSS summary saved")

# === Run example ===
generate_pbss_summary("outputs/results_gpt2-large_20250529_162843.csv")
generate_pbss_summary("outputs/results_EleutherAI_gpt-neo-1.3B_20250529_221657.csv")
generate_pbss_summary("outputs/results_gpt-3.5-turbo_20250528_220102.csv")
generate_pbss_summary("outputs/results_meta-llama_Llama-2-7b-chat-hf_20250529_192722.csv")
generate_pbss_summary("outputs/results_mistralai_Mistral-7B-Instruct-v0.2_20250530_003759.csv")
#
generate_pbss_summary("outputs/results_HuggingFaceTB_SmolLM-360M_20250529_174202.csv")
generate_pbss_summary("outputs/results_microsoft_phi-2_20250529_131221.csv")
generate_pbss_summary("outputs/results_mythomax-I2-13B-Q6_20250529_065639.csv")
generate_pbss_summary("outputs/results_openchat-3.5-1210_20250528_230208.csv")