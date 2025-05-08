import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional

class MultiCameraDataLoader:
    def __init__(
        self,
        data_root: str,
        batch_size: int = 8,
        seq_length: int = 3,
        num_workers: int = 4,
        transform: Optional[transforms.Compose] = None
    ):
        """
        多相机数据加载器
        
        Args:
            data_root: 数据根目录
            batch_size: 批次大小
            seq_length: 序列长度
            num_workers: 数据加载的工作进程数
            transform: 数据转换
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.seq_length = seq_length
        
        # 默认的数据转换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
            
        # 加载数据集配置
        self.dataset_config = self._load_dataset_config()
        
        # 创建数据集
        self.dataset = MultiCameraDataset(
            data_root=data_root,
            dataset_config=self.dataset_config,
            seq_length=seq_length,
            transform=self.transform
        )
        
        # 创建数据加载器
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def _load_dataset_config(self) -> Dict:
        """加载数据集配置文件"""
        config_path = os.path.join(self.data_root, "dataset.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def get_dataloader(self) -> DataLoader:
        """获取数据加载器"""
        return self.dataloader

class MultiCameraDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_config: Dict,
        seq_length: int,
        transform: Optional[transforms.Compose] = None
    ):
        """
        多相机数据集
        
        Args:
            data_root: 数据根目录
            dataset_config: 数据集配置
            seq_length: 序列长度
            transform: 数据转换
        """
        self.data_root = data_root
        self.dataset_config = dataset_config
        self.seq_length = seq_length
        self.transform = transform
        
        # 预处理数据
        self.sequences = self._preprocess_sequences()
        
    def _preprocess_sequences(self) -> List[Dict]:
        """预处理序列数据"""
        processed_sequences = []
        
        for seq_name, seq_data in self.dataset_config.items():
            # 按时间戳排序
            timestamps = sorted(seq_data.keys())
            
            # 创建序列
            for i in range(len(timestamps) - self.seq_length + 1):
                sequence = {
                    'seq_name': seq_name,
                    'frames': []
                }
                
                # 获取连续的时间戳
                current_timestamps = timestamps[i:i + self.seq_length]
                
                for timestamp in current_timestamps:
                    frame_data = seq_data[timestamp][0]  # 获取第一组相机数据
                    sequence['frames'].append(frame_data)
                
                processed_sequences.append(sequence)
        
        return processed_sequences
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取数据样本"""
        sequence = self.sequences[idx]
        
        # 加载所有帧的图像
        images = []
        for frame_paths in sequence['frames']:
            frame_images = []
            for img_path in frame_paths:
                # 标准化路径并转换为绝对路径
                normalized_path = normalize_path(img_path)
                abs_path = os.path.join(self.data_root, normalized_path)
                
                # 确保文件存在
                if not os.path.exists(abs_path):
                    raise FileNotFoundError(f"找不到图像文件: {abs_path}")
                
                img = Image.open(abs_path).convert('RGB')
                
                if self.transform:
                    img = self.transform(img)
                
                frame_images.append(img)
            
            # 堆叠同一帧的多个相机图像
            frame_images = torch.stack(frame_images)
            images.append(frame_images)
        
        # 堆叠所有帧
        images = torch.stack(images)
        
        return {
            'images': images,  # shape: (seq_length, num_cameras, channels, height, width)
            'sequence': sequence['seq_name']
        }

class OzoneDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        seq_length: int = 3,
        image_size: Tuple[int, int] = (224, 224),
        transform: Optional[transforms.Compose] = None
    ):
        """
        臭氧浓度预测数据集
        
        Args:
            data_root: 数据根目录
            seq_length: 序列长度
            image_size: 图像大小
            transform: 数据转换
        """
        self.data_root = os.path.abspath(data_root)
        self.seq_length = seq_length
        self.image_size = image_size
        
        # 默认的数据转换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
            
        # 加载数据集配置
        self.dataset_config = self._load_dataset_config()
        
        # 预处理数据
        self.sequences = self._preprocess_sequences()
        
    def _load_dataset_config(self) -> Dict:
        """加载数据集配置文件"""
        config_path = os.path.join(self.data_root, "dataset.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _preprocess_sequences(self) -> List[Dict]:
        """预处理序列数据"""
        processed_sequences = []
        
        for seq_name, seq_data in self.dataset_config.items():
            # 按时间戳排序
            timestamps = sorted(seq_data.keys())
            
            # 创建序列
            for i in range(len(timestamps) - self.seq_length + 1):
                sequence = {
                    'seq_name': seq_name,
                    'frames': [],
                    'ozone_data': []
                }
                
                # 获取连续的时间戳
                current_timestamps = timestamps[i:i + self.seq_length]
                
                for timestamp in current_timestamps:
                    # 获取图像数据
                    frame_data = seq_data[timestamp][0]  # 获取第一组相机数据
                    sequence['frames'].append(frame_data)
                    
                    # 获取臭氧浓度数据
                    ozone_path = os.path.join(self.data_root, timestamp)
                    ozone_data = np.load(ozone_path)
                    sequence['ozone_data'].append(ozone_data)
                
                processed_sequences.append(sequence)
        
        return processed_sequences
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取数据样本"""
        sequence = self.sequences[idx]
        
        # 加载所有帧的图像
        images = []
        for frame_paths in sequence['frames']:
            frame_images = []
            for img_path in frame_paths:
                # 标准化路径并转换为绝对路径
                normalized_path = normalize_path(img_path)
                abs_path = os.path.join(self.data_root, normalized_path)
                
                # 确保文件存在
                if not os.path.exists(abs_path):
                    raise FileNotFoundError(f"找不到图像文件: {abs_path}")
                
                img = Image.open(abs_path).convert('RGB')
                
                if self.transform:
                    img = self.transform(img)
                
                frame_images.append(img)
            
            # 堆叠同一帧的多个相机图像
            frame_images = torch.stack(frame_images)
            images.append(frame_images)
        
        # 堆叠所有帧
        images = torch.stack(images)  # shape: (seq_len, num_cameras, channels, height, width)
        
        # 加载臭氧浓度数据
        ozone_data = []
        for ozone_path in sequence['ozone_data']:
            # 标准化路径并转换为绝对路径
            normalized_path = normalize_path(ozone_path)
            abs_path = os.path.join(self.data_root, normalized_path)
            
            # 确保文件存在
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"找不到臭氧数据文件: {abs_path}")
            
            ozone = torch.from_numpy(np.load(abs_path)).float()
            ozone_data.append(ozone)
        
        ozone_data = torch.stack(ozone_data)  # shape: (seq_len, height, width)
        
        return {
            'images': images,
            'ozone_data': ozone_data,
            'sequence': sequence['seq_name']
        }

class OzoneDataLoader:
    def __init__(
        self,
        data_root: str,
        batch_size: int = 8,
        seq_length: int = 3,
        image_size: Tuple[int, int] = (224, 224),
        num_workers: int = 4,
        transform: Optional[transforms.Compose] = None
    ):
        """
        臭氧浓度数据加载器
        
        Args:
            data_root: 数据根目录
            batch_size: 批次大小
            seq_length: 序列长度
            image_size: 图像大小
            num_workers: 数据加载的工作进程数
            transform: 数据转换
        """
        self.dataset = OzoneDataset(
            data_root=data_root,
            seq_length=seq_length,
            image_size=image_size,
            transform=transform
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_dataloader(self) -> DataLoader:
        """获取数据加载器"""
        return self.dataloader