import openai
import os
import json
import csv
import time
import requests
from datetime import datetime

MODEL_NAME = "gpt-3.5-turbo"
openai.api_key="Your Key"

with open("prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_path = f"outputs/results_{MODEL_NAME}_{timestamp}.csv"
os.makedirs("outputs", exist_ok=True)


if not os.path.exists(output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["origin", "level", "variant_id", "prompt_id", "model", "temperature", "prompt", "output"])

# variant_id	直接表示 1-18 哪一条，对照 prompt 顺序
# prompt_id	唯一标识（防止多个 origin 混淆）
# level	分析强度分组（1-3）
# model	分析同一 prompt 多模型之间的输出差异

# === 批量跑 ===
for origin, prompt_list in prompts.items():
    for i, prompt in enumerate(prompt_list):
        level = (i // 5) + 1
        for temperature in [0.2, 1.3]:

            payload = {
                "model": "local-model",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": 200
            }
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=200,
                )
                output_text = response["choices"][0]["message"]["content"].strip()
                if output_text.startswith("[ERROR:"):
                    continue
            except Exception as e:
                output_text = f"[ERROR: {e}]"
            with open(output_path, "a", newline="", encoding="utf-8") as f:
                variant_id = i + 1
                prompt_id = f"{origin}_{variant_id}"
                writer = csv.writer(f)
                writer.writerow([
                    origin,
                    level,
                    variant_id,
                    prompt_id,
                    MODEL_NAME,
                    temperature,
                    prompt,
                    output_text
                ])
            print(f"[✓] {origin}-{i+1} | Temp={temperature} | Model={MODEL_NAME}")
            print("→ Output:", output_text[:15], "\n")
            time.sleep(1)