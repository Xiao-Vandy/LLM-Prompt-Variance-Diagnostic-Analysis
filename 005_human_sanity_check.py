import pandas as pd
import matplotlib.pyplot as plt

# This is designed for manually survey collection
data = {
    "Variant": [f"{i:02}" for i in range(1, 19)],
    "Mild":     [4, 2, 0, 5, 3, 2, 1, 2, 0, 3, 2, 1, 4, 3, 1, 0, 0, 1],
    "Moderate": [1, 3, 2, 0, 2, 3, 3, 2, 2, 2, 3, 2, 1, 2, 3, 1, 1, 2],
    "Severe":   [0, 0, 3, 0, 0, 0, 1, 1, 3, 0, 0, 2, 0, 0, 1, 4, 4, 3]
}
df = pd.DataFrame(data)


df.set_index("Variant").plot(kind="bar", stacked=True, figsize=(12, 6), colormap="coolwarm")
plt.title("Human Annotation Consistency on Prompt Style Severity")
plt.xlabel("Prompt Variant ID")
plt.ylabel("Number of Annotators")
plt.legend(title="Style Label")
plt.tight_layout()
plt.savefig("human_consistency_chart.pdf")
plt.show()
plt.clf()
plt.close()

