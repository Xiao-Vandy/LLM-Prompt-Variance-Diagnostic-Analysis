import os
import json
import csv
import time
import requests
from datetime import datetime
# python server.py --api --listen --model EleutherAI_gpt-neo-1.3B --loader transformers
# python server.py --api --listen --model mistralai_Mistral-7B-Instruct-v0.2 --loader transformers
# python server.py --api --listen --model mythomax-I2-13B --loader llama.cpp
# We use the LLM backend: https://github.com/oobabooga/text-generation-webui

API_URL = "http://127.0.0.1:5000/v1/chat/completions"
MODEL_NAME = "mistralai_Mistral-7B-Instruct-v0.2"
# gpt2-large  meta-llama_Llama-2-7b-chat-hf   mistralai_Mistral-7B-Instruct-v0.2      EleutherAI_gpt-neo-1.3B
# meta-llama_Llama-2-13b-chat-hf microsoft_phi-2 openchat-3.5-1210 meta-llama_Llama-3.1-8B-Instruct



# python server.py --api --listen --model gpt2-large --loader transformers
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
                response = requests.post(API_URL, json=payload)
                response_json = response.json()
                output_text = response_json["choices"][0]["message"]["content"].strip()
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
