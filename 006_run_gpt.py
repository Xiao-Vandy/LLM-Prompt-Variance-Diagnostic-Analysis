import openai
import os
import json
import csv
import time
import requests
from datetime import datetime
# === 模型名固定为 gpt-3.5-turbo ===
MODEL_NAME = "gpt-3.5-turbo"
openai.api_key="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"

with open("prompts.json", "r", encoding="utf-8") as f:
    prompts = json.load(f)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# === 输出 CSV 初始化 ===
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
            # 发送请求到本地 UI API
            payload = {
                "model": "local-model",  # 可以写死
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
                    continue  # 或者 output_text = ""
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
            print("→ Output:", output_text[:15], "\n")  # 最多只显示前 300 字
            time.sleep(1)  # 防止出 bug 可适当加延时