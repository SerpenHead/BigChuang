import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import wandb  # 可选的实验跟踪工具

class OzoneTrainer:
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 config):
        """
        臭氧浓度预测模型训练器
        
        Args:
            model: OzonePredictor模型实例
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            config: 训练配置字典
        """
        self.config = config
        self.device = torch.device(config['device'])
        
        # 模型
        self.model = model.to(self.device)
        
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config['lr_patience'],
            verbose=True
        )
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 混合精度训练
        self.scaler = GradScaler()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # 设置日志
        self.setup_logging()
        
        # 可选：初始化wandb
        if config['use_wandb']:
            wandb.init(project="ozone_prediction", config=config)
            wandb.watch(self.model)
    
    def setup_logging(self):
        """设置日志"""
        log_dir = os.path.join(self.config['save_dir'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        save_path = os.path.join(self.config['save_dir'], 'latest_checkpoint.pth')
        torch.save(checkpoint, save_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config['save_dir'], 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['val_loss']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 获取数据
            images = batch['images'].to(self.device)  # [B, seq_len, 4, 3, 224, 224]
            targets = batch['target'].to(self.device)  # [B, 1, 40, 30]
            
            # 使用混合精度训练
            with autocast():
                # 前向传播
                predictions = self.model(images)
                loss = self.criterion(predictions, targets)
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # 累计损失
            total_loss += loss.item()
            
            # 打印进度
            if (batch_idx + 1) % self.config['print_freq'] == 0:
                self.logger.info(
                    f"Epoch [{self.current_epoch}][{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Time: {time.time() - start_time:.2f}s"
                )
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        
        for batch in self.val_loader:
            images = batch['images'].to(self.device)
            targets = batch['target'].to(self.device)
            
            predictions = self.model(images)
            loss = self.criterion(predictions, targets)
            
            total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """训练模型"""
        for epoch in range(self.current_epoch, self.config['epochs']):
            self.current_epoch = epoch
            
            # 训练一个epoch
            train_loss = self.train_epoch()
            
            # 验证
            val_loss = self.validate()
            
            # 更新学习率
            self.scheduler.step(val_loss)
            
            # 记录日志
            self.logger.info(
                f"Epoch {epoch} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val Loss: {val_loss:.4f}"
            )
            
            # 可选：记录到wandb
            if self.config['use_wandb']:
                wandb.log({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # 保存检查点
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # 早停
            if self.optimizer.param_groups[0]['lr'] < self.config['min_lr']:
                self.logger.info("Learning rate too small. Stopping training.")
                break

def main():
    # 训练配置
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 100,
        'batch_size': 32,
        'sequence_length': 6,
        'lr_patience': 5,
        'min_lr': 1e-6,
        'print_freq': 10,
        'save_dir': './checkpoints',
        'use_wandb': False  # 是否使用wandb记录实验
    }
    
    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 创建数据加载器
    train_loader = create_data_loader(
        data_dir='path/to/train/data',
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length']
    )
    
    val_loader = create_data_loader(
        data_dir='path/to/val/data',
        batch_size=config['batch_size'],
        sequence_length=config['sequence_length']
    )
    
    # 创建模型
    model = OzonePredictor(
        input_size=(224, 224),
        pretrained=True,
        shared_encoder=True
    )
    
    # 创建训练器
    trainer = OzoneTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # 可选：从检查点恢复训练
    # trainer.load_checkpoint('path/to/checkpoint.pth')
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()
