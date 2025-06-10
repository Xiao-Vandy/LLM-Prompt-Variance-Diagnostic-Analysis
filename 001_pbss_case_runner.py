import openai
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import manhattan_distances
openai.api_key="Your Key"
nltk.download('punkt')



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import difflib
import matplotlib.pyplot as plt
import numpy as np
import sys

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Define hedging and assertiveness terms
HEDGES = ['might', 'could', 'possibly', 'perhaps', 'often', 'typically']
ASSERTIVES = ['must', 'always', 'definitely', 'certainly', 'undeniably']

# === Step 1: Construct Prompt Variants ===
base_prompt = "Summarize this article for a high school student."
meta_prompt = f"""
You are evaluating the stylistic robustness of a language model's prompt interpretation.

Your task is to generate a series of prompt variants for the following instruction:

"{base_prompt}"

Each variant should retain the same underlying task, but alter at least one stylistic dimension, such as:
- surface form (wording, phrasing),
- level of formality (casual vs. professional),
- directness (commands vs. suggestions),
- or the implied social frame (e.g., peer-to-peer, user-to-assistant, expert-to-novice).

Do not simplify or abstract the prompt. Instead, rephrase it with the goal of exploring how subtle stylistic shifts may influence model responses.

For each prompt, also provide a brief comment (one sentence) identifying the stylistic feature you modified.

Begin with:
PROMPT 1:
"""

# === Step 2: Call GPT to generate prompt variants ===
def get_variants():
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": meta_prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content

# === Step 3: Parse GPT Output ===
def parse_variants(text):
    prompts = re.findall(r'PROMPT \d+:\s*(.*?)\n', text)
    return prompts[:3]  # Get first 3 variants

# === Step 4: Generate model responses ===
def get_outputs(prompts):
    outputs = []
    for prompt in prompts:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens = 800
        )
        outputs.append(response.choices[0].message.content)
    return outputs

# === Step 5: Extract style vector ===
def extract_vector(text):
    sents = nltk.sent_tokenize(text)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    avg_len = len(tokens) / max(len(sents), 1)

    hedge_count = sum(len(re.findall(rf'\b{w}\b', text.lower())) for w in HEDGES)
    assert_count = sum(len(re.findall(rf'\b{w}\b', text.lower())) for w in ASSERTIVES)

    hedge_ratio = hedge_count / max(len(tokens), 1)
    assert_ratio = assert_count / max(len(tokens), 1)

    self_ref = int("as an ai language model" in text.lower())
    structure_tokens = len(re.findall(r"(first,|second,|finally,|in conclusion|let's)", text.lower()))

    return [
        avg_len,
        hedge_ratio * 100,
        assert_ratio * 100,
        self_ref * 10,
        structure_tokens
    ]



# === Step 6: Compute PBSS matrix ===
def compute_pbss_matrix(vectors):
    n = len(vectors)
    pbss = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                score = 1 - cosine_similarity([vectors[i]], [vectors[j]])[0][0]
                pbss[i][j] = score
    return pbss


def plot_heatmap(pbss, labels):
    print("pbss:",pbss)
    pbsslog = np.log1p(pbss)
    sns.heatmap(pbsslog, annot=True, xticklabels=labels, yticklabels=labels,
                cmap="coolwarm", vmin=0)
    print(pbsslog)
    plt.title(f"🔥 PBSS Drift Heatmap")
    plt.show()

def plot_umap(vectors, labels):
    if len(vectors) < 2:
        print("Not enough samples for t-SNE projection.")
        return
    tsne = TSNE(n_components=2, perplexity=min(5, len(vectors)-1), random_state=42)
    proj = tsne.fit_transform(np.array(vectors))
    plt.figure()
    for i, label in enumerate(labels):
        plt.scatter(proj[i, 0], proj[i, 1], label=label)
        plt.text(proj[i, 0] + 0.1, proj[i, 1], label, fontsize=12)
    plt.title("🌀 t-SNE Projection of Style Vectors")
    plt.legend()
    plt.show()


def plot_radar(vectors, labels):
    import numpy as np
    angles = np.linspace(0, 2 * np.pi, len(vectors[0]), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

    for vec, label in zip(vectors, labels):
        data = vec + vec[:1]
        ax.plot(angles, data, label=label)
        ax.fill(angles, data, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['Length', 'Hedge', 'Assertive', 'Structure', 'Self-ref'])
    ax.legend()
    plt.title("Style Profile Comparison")
    plt.show()



def highlight_drift(a, b):
    matcher = difflib.SequenceMatcher(None, a.split(), b.split())
    output = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            output.extend(a.split()[i1:i2])
        elif opcode in ("replace", "delete", "insert"):
            changed = a.split()[i1:i2]
            output.extend([f"**{token}**" for token in changed])
    return " ".join(output)


def manual_output_input():
    print("🔧 Paste each GPT output when prompted.")
    print("Type `---END---` on a new line to finish one output.")
    print("Type `---DONE---` to stop adding more outputs.\n")

    outputs = []
    labels = []
    count = 0

    while True:
        count += 1
        print(f"\n🔹 Paste output {chr(64 + count)} (end with `---END---`):")
        lines = []
        while True:
            line = input()
            if line.strip() == "---END---":
                break
            if line.strip() == "---DONE---":
                count -= 1
                break
            lines.append(line)
        if not lines:
            break
        outputs.append(" ".join(lines))
        labels.append(chr(64 + count))
        if line.strip() == "---DONE---":
            break

    if len(outputs) < 2:
        print("🚨 At least two outputs are required.")
        return

    print(f"\n📊 Extracting style vectors from {len(outputs)} outputs...")
    vectors = [extract_vector(o) for o in outputs]
    for i, v in enumerate(vectors):
        print(f"Prompt {labels[i]} vector: {v}")

    print("\n📈 Computing PBSS scores...")
    pbss_matrix = compute_pbss_matrix(vectors)
    plot_heatmap(pbss_matrix, labels)

    print("\n🌐 Projecting to 2D space (t-SNE)...")
    plot_umap(vectors, labels)

    print("\n📝 Token-Level Differences:")
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            print(f"\n--- {labels[i]} vs {labels[j]} Diff ---")
            print(highlight_drift(outputs[i], outputs[j]))



# === Main ===
def main():
    mode = input("Type `1` to run with GPT, `2` to paste outputs manually: ").strip()
    if mode == "2":
        manual_output_input()
    else:
        print("🔧 Generating prompt variants...")
     # raw = get_variants()
    #     prompts = [
    #     "Should students be allowed to use AI tools for homework?",
    #     "You're a LGBTQ parent writing a reply to a strange terrible school group chat66. What would you say about kids using ChatGPT for homework?",
    #     "You're a 16-year-old ranting on Reddit about how school treats AI like it's black magic."
    #
    # ]
        prompts=[
            "Should students use AI tools to assist with their homework? Provide a balanced discussion of pros and cons.",
            "Can students really benefit from using AI for their homework? Discuss both sides of the argument.",
            "Examine whether it’s advisable for students to utilize AI tools in completing their assignments, considering potential advantages and disadvantages.",
            "I'd appreciate your thoughts on whether students should rely on AI tools for homework—what are the upsides and downsides?",
            "Analyze the pros and cons associated with students integrating AI tools into their homework routines. ",
            "Could you maybe outline some positives and negatives of students using AI tools for homework?",
            "Debate the implications of students turning to AI tools for homework assistance—considering carefully both the benefits and potential pitfalls.",
            "Hey, what's your take—should students be using AI for homework help? Pros and cons, please!",
            "Offer an impartial analysis: Is student usage of AI tools for homework justified? Outline clearly both arguments.",
            "If students are thinking about using AI for homework, what advantages and disadvantages should they be aware of?"

        ]
        print("🧠 Prompt variants:", prompts)

        print("🤖 Getting model outputs...")
        outputs = get_outputs(prompts)

        print("📊 Extracting style vectors...")
        vectors = [extract_vector(o) for o in outputs]
        for i, v in enumerate(vectors):
            print(f"Prompt {i+1} vector: {v}")

        print("📈 Computing PBSS scores...")
        pbss_matrix = compute_pbss_matrix(vectors)
        plot_heatmap(pbss_matrix, prompts)

if __name__ == "__main__":
    main()
