import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取 CSV 文件
file_path = "../api_annotation/annotations.csv"
df = pd.read_csv(file_path)

# 查看数据的前几行，确保数据加载正确
print(df.head())

# 选择我们需要的列
concepts = df['concept_text']
scores = df[['Safety', 'Belonging', 'Naturalness', 'Shared', 'Privacy', 'Convenience']]

# 可视化准备
# 创建一个新的 DataFrame，将 concept_text 与六个维度的评分进行合并
concepts_scores = pd.concat([concepts, scores], axis=1)

# 生成文本与评分的可视化图
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# 逐列绘制每个维度的分布
for idx, column in enumerate(scores.columns):
    # 使用 Seaborn 绘制每个维度的分布图
    sns.boxplot(x=column, y='concept_text', data=concepts_scores, ax=axes[idx], orient='h')
    axes[idx].set_title(f"Distribution of {column}")

plt.tight_layout()
plt.show()

# 额外：生成每个维度与概念文本的散点图
# 我们可以使用文字的平均位置来表示文本的关系
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, column in enumerate(scores.columns):
    axes[idx].scatter(concepts_scores[column], np.zeros_like(concepts_scores[column]), c=concepts_scores[column], cmap='viridis', alpha=0.7)
    axes[idx].set_title(f"Text vs {column}")
    axes[idx].set_xlabel(column)
    axes[idx].set_yticks([])

plt.tight_layout()
plt.show()
