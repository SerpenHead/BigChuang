import os
import csv
import json
from datetime import datetime, timedelta

# 配置
IMAGE_DIR = 'data/raw/images'
LABEL_CSV = 'data/raw/labels.csv'
OUTPUT_INDEX = 'data/splits/index.json'
SEQ_LEN = 6  # 序列长度
CAMERAS = ['cam1', 'cam2', 'cam3', 'cam4']
TIME_FORMAT = '%Y%m%dT%H%M%S'

# 1. 读取标签 CSV，建立 timestamp -> label_path 映射
label_map = {}
with open(LABEL_CSV, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        ts, grid = row
        label_map[ts] = grid

# 2. 列举所有图像文件，按 timestamp 分组
img_files = os.listdir(IMAGE_DIR)
# 提取时间戳和摄像头信息
imgs = {}
for fname in img_files:
    name, _ = os.path.splitext(fname)
    ts, cam = name.split('_')
    imgs.setdefault(ts, {})[cam] = os.path.join('images', fname)

# 3. 生成索引样本
samples = []
all_ts = sorted(imgs.keys())
for i in range(SEQ_LEN - 1, len(all_ts)):
    seq_ts = all_ts[i - (SEQ_LEN - 1) : i + 1]
    last_ts = seq_ts[-1]
    # 确保每个时间点都有所有摄像头的图像
    if all(cam in imgs[ts] for ts in seq_ts for cam in CAMERAS):
        # 查找标签
        if last_ts in label_map:
            seq_paths = [[imgs[ts][cam] for cam in CAMERAS] for ts in seq_ts]
            samples.append({
                'seq_paths': seq_paths,
                'label_path': os.path.join('labels', label_map[last_ts]),
                'timestamp': last_ts
            })

# 4. 保存为 JSON
os.makedirs(os.path.dirname(OUTPUT_INDEX), exist_ok=True)
with open(OUTPUT_INDEX, 'w') as f:
    json.dump({'samples': samples}, f, indent=2)

print(f"生成索引文件，共 {len(samples)} 个样本，保存至 {OUTPUT_INDEX}")