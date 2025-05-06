import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 输入路径
IMAGES_DIR = Path("../crawl/converted_jpg")
OUTPUT_CSV = Path("../api_annotation/annotations.csv")

# 读取 CSV
df = pd.read_csv(OUTPUT_CSV)

# 六个维度英文名
dimension_labels = ['Safety', 'Belonging', 'Naturalness', 'Shared', 'Privacy', 'Convenience']

# 抽取4条样本
random_rows = df.sample(n=4)

# 创建 2x4 图（2行4列）
fig, axes = plt.subplots(2, 4, figsize=(20, 8))

# 六边形角度
angles = np.deg2rad(np.linspace(0, 360, 7)[:-1])  # 6维

# 遍历每个样本
for i, (index, row) in enumerate(random_rows.iterrows()):
    img_path = IMAGES_DIR / row['filename']
    img = Image.open(img_path)
    img.thumbnail((256, 256))

    # 六维评分 + 文本
    scores = row[["Safety", "Belonging", "Naturalness", "Shared", "Privacy", "Convenience"]].astype(float).values
    concept = row['concept_text']

    # ------- 第一行：六边形雷达图 -------
    ax1 = axes[0, i]
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # 主六边形
    hexagon = patches.RegularPolygon(
        (0.5, 0.5), numVertices=6, radius=0.4,
        orientation=np.pi/6, edgecolor='black', facecolor='none', lw=3
    )
    ax1.add_patch(hexagon)

    # 内部评分雷达图
    norm_scores = scores / 5 * 0.4
    x_vals = 0.5 + norm_scores * np.cos(angles)
    y_vals = 0.5 + norm_scores * np.sin(angles)
    ax1.plot(x_vals, y_vals, color='gray', lw=1)
    ax1.fill(x_vals, y_vals, color='gray', alpha=0.2)

    # 分数
    for j in range(6):
        x = 0.5 + 0.42 * np.cos(angles[j])
        y = 0.5 + 0.42 * np.sin(angles[j])
        ax1.text(x, y, f'{int(scores[j])}', ha='center', va='center', fontsize=10)

    # 标签
    for j, label in enumerate(dimension_labels):
        x = 0.5 + 0.52 * np.cos(angles[j])
        y = 0.5 + 0.52 * np.sin(angles[j])
        ax1.text(x, y, label, ha='center', va='center', fontsize=9)

    # 文件名（上方）
    ax1.text(0.5, 1.05, row['filename'], ha='center', va='center', fontsize=11)

    # ------- 第二行：图片 + concept -------
    ax2 = axes[1, i]
    ax2.imshow(img)
    ax2.axis('off')

    # concept_text（下方）
    ax2.text(0.5, -0.1, concept, ha='center', va='center', fontsize=10, wrap=True, transform=ax2.transAxes)

plt.tight_layout(h_pad=3)
plt.savefig("Six-dimensional graph.png")
plt.show()
