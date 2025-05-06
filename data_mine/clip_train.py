import torch
import clip
from PIL import Image
from pathlib import Path
import os
import numpy as np
from tqdm import tqdm

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# 输入数据路径和图像路径
IMAGES_DIR = Path("../crawl/converted_jpg")
OUTPUT_CSV = Path("../api_annotation/annotations.csv")

# 读取 CSV 文件
import csv

data = []
with open(OUTPUT_CSV, "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # 跳过表头
    for row in reader:
        data.append({
            "filename": row[0],
            "concept_text": row[-1]
        })

# 准备图像和文本
image_paths = [IMAGES_DIR / entry["filename"] for entry in data]
concept_texts = [entry["concept_text"] for entry in data]

# 提取图像和文本的特征
image_features = []
text_features = []

# 对所有图像和文本进行特征提取
for image_path, concept_text in tqdm(zip(image_paths, concept_texts), total=len(image_paths)):
    # 加载图像并进行预处理
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # 提取图像特征
    with torch.no_grad():
        image_feature = model.encode_image(image)
    image_features.append(image_feature.cpu().numpy())

    # 提取文本特征
    text = clip.tokenize([concept_text]).to(device)
    with torch.no_grad():
        text_feature = model.encode_text(text)
    text_features.append(text_feature.cpu().numpy())

# 将图像和文本的特征转换为 numpy 数组
image_features = np.vstack(image_features)
text_features = np.vstack(text_features)

# 计算图像与文本的余弦相似度
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(image_features, text_features)

# 输出每张图片和对应概念文本的相似度
for i, (image_path, concept_text) in enumerate(zip(image_paths, concept_texts)):
    print(f"图片: {image_path.name} -> 概念文本: {concept_text} 相似度: {similarity_matrix[i].max()}")

# 保存图像和文本特征
os.makedirs("clip_features", exist_ok=True)
np.save("clip_features/image_features.npy", image_features)
np.save("clip_features/text_features.npy", text_features)

# 保存索引对应文件
import json
index_data = [{"filename": p.name, "concept_text": t} for p, t in zip(image_paths, concept_texts)]
with open("clip_features/index_mapping.json", "w", encoding="utf-8") as f:
    json.dump(index_data, f, ensure_ascii=False, indent=2)

print("图像和文本特征已保存至 clip_features 文件夹。")