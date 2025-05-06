# ================= build_files.py =================
# 把 annotations.csv (六维情绪表)  → modules.csv + 2 个 json
# --------------------------------------------------

print(">>> build_files.py entered")          # 声呐：脚本已真正执行

import pandas as pd, json, numpy as np, os, sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

CSV_FILE = "annotations.csv"
EMO_COLS = ['Safety', 'Belonging', 'Naturalness',
            'Shared', 'Privacy', 'Convenience']

# ---------- 1. 读取 csv ----------
if not Path(CSV_FILE).exists():
    sys.exit(f"❌ 未找到 {CSV_FILE} ，请先生成 annotations.csv")
df = pd.read_csv(CSV_FILE)
if df.empty:
    sys.exit("❌ annotations.csv 为空，先检查上一步")

# ---------- 2. 文本嵌入 ----------
print("⏳ 生成句子向量 (all-MiniLM-L6-v2)")
model = SentenceTransformer('all-MiniLM-L6-v2')
emb   = model.encode(df['concept_text'].fillna("").tolist(), show_progress_bar=True)

# ---------- 3. 尝试 HDBSCAN 聚类 ----------
min_cluster = max(5, len(df)//10)
print(f"⏳ HDBSCAN(min_cluster_size={min_cluster}) …")
labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster).fit_predict(emb)

# 如果全噪声 (-1)，改用 KMeans 兜底
if (labels != -1).sum() == 0:
    k = max(1, min(10, len(df)//5))
    print(f"⚠️  HDBSCAN 全噪声 → 改用 KMeans(n_clusters={k})")
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(emb)

df['cluster'] = labels
print("聚类标签分布:\n", df['cluster'].value_counts())

records = []
for cid, grp in df.groupby('cluster'):
    # 4‑1 计算簇中心 & 代表样本
    center   = emb[grp.index].mean(0, keepdims=True)
    rep_idx  = grp.index[pairwise_distances_argmin(center, emb[grp.index])[0]]
    rep_row  = df.loc[rep_idx]

    # 4‑2 统计情绪均值
    emo_mean = grp[EMO_COLS].mean().round(3).tolist()

    # 4‑3 示例定价：Convenience × 1000
    price = round(float(rep_row['Convenience']) * 1000, 1)

    # 4‑4 生成记录
    records.append(dict(
        id=f"cluster{cid}",
        name = str(rep_row.get('concept_text', f"Concept {cid}")).strip()[:25],
        price=price,
        width=3.0, depth=3.0, height=2.8,          # 初始尺寸，可后续映射
        safe=emo_mean[0], belong=emo_mean[1], nature=emo_mean[2],
        share=emo_mean[3], privacy=emo_mean[4], convenient=emo_mean[5],
        img=rep_row['filename']
    ))

if not records:
    sys.exit("❌ 仍未生成任何记录，请检查聚类参数或数据量！")

# ---------- 5. 写出 modules.csv ----------
mods_df = pd.DataFrame(records)
mods_df.to_csv("modules.csv", index=False, encoding="utf-8-sig")
print(f"✅ modules.csv 写入 {len(records)} 行")

# ---------- 6. 写出 cluster_centroids.json ----------
with open("cluster_centroids.json", "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print("✅ cluster_centroids.json 写入完成")

# ---------- 7. 写出 mapping.json ----------
mapping = {
  "wall_thickness": ["safe", 0.10, 0.40],
  "window_ratio"  : ["belong", 0.20, 0.70],
  "green_patio"   : ["nature", 0, 1],
  "shared_table"  : ["share", 0, 1],
  "privacy_screen": ["privacy", 0, 1],
  "auto_door"     : ["convenient", 0, 1],
  "palette"       : ["#e57373","#64b5f6","#81c784","#ffd54f","#ba68c8","#4dd0e1"]
}
with open("mapping.json", "w", encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)
print("✅ mapping.json 写入完成")

print(">>> build_files.py finished")      # 声呐：脚本顺利结束
# ==================================================
