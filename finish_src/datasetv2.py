import os
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageSequenceDataset(Dataset):
    def __init__(self, json_path, base_dir="F:\\dachuang_network"):
        self.base_dir = Path(base_dir).resolve()
        self.data_dir = self.base_dir / "data"
        self.npy_dir = self.data_dir / "interpolation_data"

        with open(json_path, "r") as f:
            self.data = json.load(f)

        # 读取所有序列
        self.sequences = []
        self.required_npy = set()

        for seq_key, time_steps in self.data.items():
            time_steps_sorted = sorted(time_steps.items())  # [(npy_name, [img1, img2, img3, img4]), ...]
            self.sequences.append(time_steps_sorted)
            for npy_name, _ in time_steps_sorted:
                self.required_npy.add(npy_name)

        # 缓存所有 npy 文件
        self.npy_cache = {}
        for npy_name in self.required_npy:
            npy_path = self.npy_dir / npy_name
            if not npy_path.exists():
                raise FileNotFoundError(f"Npy file not found: {npy_path}")
            self.npy_cache[npy_name] = torch.tensor(np.load(npy_path), dtype=torch.float32)

        # 图像增强变换
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(),  # 均值方差归一化
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.sequences)

    @staticmethod
    def camera_order(path):
        fname = os.path.basename(path).lower()
        if "cam1" in fname:
            return 0  # 东
        if "cam2" in fname:
            return 1  # 南
        if "cam3" in fname:
            return 2  # 西
        if "cam4" in fname:
            return 3  # 北
        return 99  # 未知

    def load_image(self, relative_path):
        """用 cv2 读取图片，并执行预处理"""
        relative_path = Path(relative_path.replace("..", "").lstrip("/\\"))
        full_path = self.base_dir / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Image not found: {full_path}")
        img = cv2.imread(str(full_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(image=img)["image"]

    def __getitem__(self, idx):
        try:
            seq = self.sequences[idx]
            images_tensor = []
            npy_tensor = []
            name = "abc"
            for npy_name, image_group in seq:
                npy_tensor.append(self.npy_cache[npy_name])  # 直接用缓存
                name = npy_name
                image_paths = sorted(image_group[0], key=self.camera_order)
                imgs = [self.load_image(p) for p in image_paths]
                images_tensor.append(torch.stack(imgs, dim=0))  # (4, C, H, W)

            return {
                "images": torch.stack(images_tensor, dim=0),  # (T, 4, C, H, W)
                 "npy": npy_tensor[-1],
                 "npy_name": name    
            }
        except Exception as e:
            print(f"[Dataset Error] idx={idx}: {e}")
            return None
