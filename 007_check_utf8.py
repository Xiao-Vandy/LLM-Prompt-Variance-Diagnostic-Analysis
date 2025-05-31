import csv
# This is initially designed for checking csv index.
with open("outputs/results_gpt2-large.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        print(f"--- Row {i+1} ---")
        print("Origin: ", row["origin"])
        print("Prompt: ", row["prompt"][:100])
        print("Output: ", row["output"][:300])
        print()
