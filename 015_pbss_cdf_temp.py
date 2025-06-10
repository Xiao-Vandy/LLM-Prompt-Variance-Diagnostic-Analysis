import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os


folderpath="pbss_MiniLM-L6"
# folderpath="pbss_MiniLM-L12"
# folderpath="pbss_mpnet"


csvname="pbss_summary_all-MiniLM-L6-v2_5.csv"
# csvname="pbss_summary_paraphrase-MiniLM-L12-v2.csv"
# csvname="pbss_summary_all-mpnet-base-v2.csv"



columns = ["origin", "model", "temperature", "avg_pbss"]
df = pd.read_csv(f"{folderpath}/{csvname}", names=columns)
model_name_map = {
    'gpt2-large': 'gpt2-large',
    'EleutherAI_gpt-neo-1.3B': 'gpt-neo-1.3B',
    'gpt-3.5-turbo': 'gpt-3.5-turbo',
    'meta-llama_Llama-2-7b-chat-hf': 'Llama-2-7b-chat-hf',
    'Mistral-7B-Instruct-v0.2': 'Mistral-7B-instruct-v0.2',
    # 'HuggingFaceTB_SmolLM-360M': 'Hugging-Face_SmolLM-360M',
    # 'microsoft_phi-2': 'Microsoft_phi-2',
    # 'stabilityai_StableBeluga-7B': 'StableBeluga-7B',
    # 'openchat-3.5-1210': 'Openchat-3.5-1210',
    # 'mythomax-I2-13B-Q6': 'Mythomax-I2-13B-Q6'
}
df["model"] = df["model"].map(model_name_map)

mean_df = df.groupby("model")["avg_pbss"].mean().reset_index(name="pbss_mean")
max_df = df.groupby("model")["avg_pbss"].max().reset_index(name="pbss_max")
summary = pd.merge(mean_df, max_df, on="model")
print(summary.sort_values("pbss_mean"))


import matplotlib.pyplot as plt
import seaborn as sns

for temp in sorted(df["temperature"].unique()):
    plt.figure(figsize=(10, 6))
    temp_df = df[df["temperature"] == temp]

    for model in temp_df["model"].unique():
        values = temp_df[temp_df["model"] == model]["avg_pbss"].sort_values()
        sns.ecdfplot(values, label=model)

    plt.title(f"PBSS Cumulative Distribution (Temperature={temp})")
    plt.xlabel("PBSS")
    plt.ylabel("Cumulative %")
    plt.legend()
    plt.tight_layout()
    temp_str = str(temp).replace(".", "")
    plt.savefig(f"{folderpath}/pbss_cdf_temp{temp_str}_5.pdf", bbox_inches="tight")
    plt.show()
    plt.clf()
    plt.close()


