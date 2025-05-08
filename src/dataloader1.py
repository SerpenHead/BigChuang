import json
import torch
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

# A custom Dataset class must implement three functions:
#  __init__, __len__, and __getitem__.


class ImageSequenceDataset(Dataset):
    def __init__(self, json_path):

        with open(json_path, "r") as f:
            self.data = json.load(f)

        self.sequences = []
        """
        {
            "seq_0": {
                "npy_filename1": [[img1, img2, img3, img4]],
                "npy_filename2": [[img1, img2, img3, img4]],
                "npy_filename3": [[img1, img2, img3, img4]]
            },
            "seq_1": {
                ...
            }
        }
        """
        for seq_key, time_steps in self.data.items():
            time_steps_sorted = sorted(time_steps.items())
            self.sequences.append(time_steps_sorted)

        self.transform = transforms.Compose(
            [transforms.RandomResizedCrop((224, 224)), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.sequences)
    

    # 摄像头顺序。为了保证空间一致性
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

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        images_tensor = []
        npy_tensor = []

        for npy_path, image_lists in seq:
            # 加载 npy, O3 栅格浓度
            npy_data = np.load(npy_path)
            npy_tensor.append(torch.tensor(npy_data, dtype=torch.float32))

            # 读取4张图片
            image_paths = image_paths = sorted(image_lists[0], key=self.camera_order)
            imgs = []
            # 一个 seq 里有 3 个样本，一个样本里有 同一时间戳对应的 4 张图片。
            # 下面这个循环读取单个样本里的 4 张图片：
            for img_path in image_paths:
                img_path = os.path.normpath(img_path)  # 处理 Windows 路径
                # 读取图片。
                image = Image.open(img_path).convert("RGB")
                image = self.transform(image)
                imgs.append(image)
            # 拼成 (4, C, H, W)
            images_tensor.append(torch.stack(imgs, dim=0))

        # 最终 images_tensor shape: (3, 4, C, H, W)
        # 最终 npy_tensor shape: (3, D)
        return {
            "images": torch.stack(images_tensor, dim=0),
            "labels": torch.stack(npy_tensor, dim=0),
        }

    
