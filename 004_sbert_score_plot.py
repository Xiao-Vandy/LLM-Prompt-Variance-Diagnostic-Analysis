import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

# Task-wise SBERT similarity scores
tasks = {
    "Task 1": [0.7100, 0.6674, 0.5923, 0.8178, 0.6890, 0.5492, 0.5882, 0.5689, 0.5626, 0.6283, 0.5802, 0.5110],
    "Task 2": [0.8219, 0.6937, 0.6300, 0.9213, 0.8881, 0.8203, 0.8120, 0.7389, 0.4875, 0.7585, 0.6216, 0.6119],
    "Task 3": [0.8463, 0.7410, 0.6644, 0.8477, 0.7786, 0.6526, 0.8291, 0.7490, 0.6370, 0.7995, 0.7720, 0.6826],
    "Task 4": [0.8062, 0.7977, 0.5190, 0.7601, 0.7087, 0.6775, 0.7926, 0.7217, 0.6430, 0.7348, 0.7007, 0.5483],
    "Task 5": [0.7821, 0.7541, 0.6253, 0.7959, 0.7634, 0.6619, 0.8285, 0.6967, 0.6810, 0.7054, 0.7052, 0.7026],
    "Task 6": [0.9578, 0.8351, 0.7229, 0.8724, 0.8614, 0.8609, 0.8944, 0.7873, 0.7369, 0.9126, 0.8720, 0.5002],
    "Task 7": [0.7475, 0.6308, 0.5479, 0.7461, 0.6489, 0.6489, 0.6999, 0.6556, 0.6338, 0.5998, 0.5948, 0.5513],
    "Task 8": [0.7167, 0.6311, 0.5819, 0.8471, 0.7380, 0.6961, 0.8163, 0.6021, 0.5404, 0.7026, 0.6083, 0.5466],
    "Task 9": [0.8643, 0.6666, 0.6586, 0.9086, 0.7921, 0.7919, 0.8398, 0.7620, 0.7339, 0.7379, 0.7285, 0.6957],
    "Task 10": [0.6623, 0.5825, 0.5469, 0.7754, 0.6638, 0.6193, 0.8033, 0.6858, 0.6851, 0.6970, 0.6677, 0.6150]
}

# Prepare figure
plt.figure(figsize=(12, 6))

# Plot SBERT similarity distributions
for i, (task, scores) in enumerate(tasks.items(), 1):
    x = [i] * len(scores)
    plt.scatter(x, scores, label=task)

plt.xticks(ticks=np.arange(1, len(tasks)+1), labels=list(tasks.keys()), rotation=45)
plt.ylim(0.4, 1.0)
plt.ylabel("SBERT Similarity")
plt.title("SBERT Similarity Distribution Across 10 Tasks")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.legend(loc='lower right', fontsize='small', bbox_to_anchor=(1.15, 0.2))

plt.show()
plt.clf()  # 清空当前 figure
plt.close()  # 关闭当前 plot（释放内存）

