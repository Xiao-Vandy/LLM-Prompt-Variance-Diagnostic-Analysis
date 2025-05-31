from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

model = SentenceTransformer('all-mpnet-base-v2')


# ✅ Step 2: 原始 prompt（你的original）
origin_prompt = "Ask your supervisor for time off, formally."

prompt_num=10
# ✅ Step 3: prompt 变体（直接复制 18 条进去）
prompts = [
    "How would you formally ask a supervisor for time off?",
    "Can you write a professional message requesting leave from work?",
    "Okay but like, how do I say ‘I won’t be here Friday’ without sounding like I hate my job?",

    "Write a respectful request to take time off for personal reasons.",
    "What’s a formal way to ask for a day off next week?",
    "If you need time off, how do you phrase it to your manager politely?",

    "Explain how you’d request leave without disrupting your team’s work.",
    "Describe how to ask for time off while sounding responsible.",
    "How do you write a short, formal email asking for a few days off?",

    "What’s a tactful way to say ‘I need to be away from work’ temporarily?",
    "You're requesting scheduled leave—how do you present that formally?",
    "As an employee who needs time, how do you ask without sounding careless?",

    "hello i not come work thursday please allow?",
    "off day need me personal ok manager say yes?",
    "next week day gone reason important request leave"
]


group_labels = [
    "Stylistic Shift", "Stylistic Shift", "Stylistic Shift",
    "Syntax Manip", "Syntax Manip", "Syntax Manip",
    "Surface Perturb", "Surface Perturb", "Surface Perturb",
    "Contextual", "Contextual", "Contextual",
    "Broken Prompt", "Broken Prompt", "Broken Prompt"
]


# === Embedding model
model = SentenceTransformer('all-mpnet-base-v2')
original_embedding = model.encode(origin_prompt, convert_to_tensor=True)
variant_embeddings = model.encode(prompts, convert_to_tensor=True)

# === Shallow syntax
def shallow_features(prompt):
    return np.array([
        len(prompt.split()),
        len(re.findall(r"\b(is|are|was|were|be|been|being|have|has|had|do|does|did|can|could|will|would|shall|should|may|might|must)\b", prompt.lower())),
        int(bool(re.search(r"\b(be|is|are|was|were|been|being)\s+\w+ed\b", prompt.lower()))),
        len(re.findall(r"\b(that|if|when|because|although|while)\b", prompt.lower())),
        int("?" in prompt),
        int(bool(re.search(r"\b(I|you|we|me|my|your|our)\b", prompt)))
    ])

origin_feat = shallow_features(origin_prompt)

# === Prepare data
colors = {
    "Original Prompt": "black",
    "Stylistic Shift": "blue",
    "Syntax Manip": "green",
    "Surface Perturb": "purple",
    "Contextual": "brown",
    "Broken Prompt": "red"
}

points = []
group_counts = defaultdict(int)


# Add original prompt manually
points.append(("Original Prompt", 1.0, 0.0, "Origin"))

# Add variants
for i, prompt_text in enumerate(prompts):
    sim = util.cos_sim(original_embedding, variant_embeddings[i]).item()
    syntax_dist = np.linalg.norm(shallow_features(prompt_text) - origin_feat)
    label = group_labels[i]
    group_counts[label] += 1
    points.append((label, sim, syntax_dist, str(group_counts[label])))

# === Plotting
plt.figure(figsize=(10, 6))
legend_handles = {}

# Plot all points with number + similarity
for label, x, y, idx in points:
    color = colors[label]
    plt.scatter(x, y, color=color, alpha=0.7, s=80)

    if label == "Original Prompt":
        plt.text(x, y, "●", color='white', fontsize=9, ha='center', va='center', weight='bold')
    elif label == "Broken Prompt":
        continue  # 🧨 Don’t label broken prompts with index
    else:
        plt.text(x, y, idx, color='white', fontsize=8, ha='center', va='center', weight='bold')
        plt.text(x, y - 0.015, f"sim={float(x):.3f}", color='gray', fontsize=6, ha='center', va='top')

# Legend (unique)
for label in colors:
    legend_handles[label] = plt.scatter([], [], color=colors[label], label=label)

plt.legend(title="Prompt Group", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xlabel("SBERT Semantic Similarity to Original Prompt")
plt.ylabel("Shallow Syntax Feature Distance")
plt.title("Prompt Variation Map with Semantic + Syntactic Drift")
plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig(f"figure/sanity_check_case{prompt_num}.pdf", bbox_inches="tight")
plt.clf()  # 清空当前 figure
plt.close()  # 关闭当前 plot（释放内存）

print("\n=== 🔍 Prompt Debug Printout (Semantic + Syntax Drift) ===\n")
print(f"[00] Original Prompt (SBERT = 1.0000)\n→ {origin_prompt}\n")

group_points = defaultdict(list)
for group, entries in group_points.items():
    if group == "Broken Prompt":
        print(f"[{group}]")
        print("  🚨 Skipped trend analysis (stress test group)\n")
        continue  # Skip trend checks

for i, (label, x, y, idx) in enumerate(points[1:], start=1):
    prompt_text = prompts[i - 1]
    syntax_feats = shallow_features(prompt_text)
    syntax_diff = np.linalg.norm(syntax_feats - origin_feat)

    print(f"{x:.4f}")

