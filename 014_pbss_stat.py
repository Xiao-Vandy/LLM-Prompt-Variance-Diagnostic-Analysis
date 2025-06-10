import pandas as pd
from scipy.stats import kruskal

file_map = {
    "MiniLM-L6": "pbss_MiniLM-L6/pbss_summary_all-MiniLM-L6-v2.csv",
    "MiniLM-L12": "pbss_MiniLM-L12/pbss_summary_paraphrase-MiniLM-L12-v2.csv",
    "MPNet": "pbss_mpnet/pbss_summary_all-mpnet-base-v2.csv"
}

all_dfs = []

for encoder_label, filepath in file_map.items():
    df = pd.read_csv(filepath, names=["origin", "model", "temperature", "avg_pbss"])
    df["encoder"] = encoder_label  # 添加 encoder 名称标签
    all_dfs.append(df)


merged_df = pd.concat(all_dfs, ignore_index=True)

print("=== Individual Kruskal-Wallis Tests by Encoder ===")
for encoder_label in merged_df["encoder"].unique():
    sub_df = merged_df[merged_df["encoder"] == encoder_label]
    models = sub_df["model"].unique()
    pbss_groups = [sub_df[sub_df["model"] == m]["avg_pbss"].values for m in models]

    stat, p = kruskal(*pbss_groups)
    print(f"{encoder_label}: H = {stat:.4f}, p = {p:.4e}")


print("\n=== Combined Encoder Kruskal-Wallis Test ===")
models = merged_df["model"].unique()
pbss_groups = [merged_df[merged_df["model"] == m]["avg_pbss"].values for m in models]


for temp in [0.2, 1.3]:
    print(f"\n=== Kruskal-Wallis Tests at Temperature {temp} ===")

    temp_df = merged_df[merged_df["temperature"] == temp]

    for encoder_label in temp_df["encoder"].unique():
        sub_df = temp_df[temp_df["encoder"] == encoder_label]
        models = sub_df["model"].unique()
        pbss_groups = [sub_df[sub_df["model"] == m]["avg_pbss"].values for m in models]

        stat, p = kruskal(*pbss_groups)
        print(f"{encoder_label}: H = {stat:.4f}, p = {p:.4e}")

    print(f"\n--- Combined across encoders for Temp {temp} ---")
    models = temp_df["model"].unique()
    pbss_groups = [temp_df[temp_df["model"] == m]["avg_pbss"].values for m in models]

    stat, p = kruskal(*pbss_groups)
    print(f"Combined: H = {stat:.4f}, p = {p:.4e}")


stat, p = kruskal(*pbss_groups)
print(f"Combined: H = {stat:.4f}, p = {p:.4e}")

merged_df.to_csv("pbss_combined_all_encoders.csv", index=False)
print("Merged PBSS file saved as 'pbss_combined_all_encoders_3p.csv'")


summary_stats = merged_df.groupby("model")["avg_pbss"].describe()
print("\n=== Descriptive Statistics by Model ===")
print(summary_stats)
