import torch
import torch.nn as nn
import torchvision.models as models

class CNNFeatureExtractor(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, out_dim=512):
        super().__init__()
        # 使用 torchvision 提供的轻量 CNN（可换成 resnet34、efficientnet 等）
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-2])  # 去掉 avgpool & fc
            self.out_dim = 512  # resnet18 的最终 feature map channel 数
        else:
            raise ValueError("Only resnet18 is supported for now")

        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 每张图特征压缩为 (out_dim,)
        self.flatten = nn.Flatten()

    def forward(self, x):
        """
        输入 x: (B, T, 4, C, H, W)
        输出:  (B, T, 4 * out_dim)
        """
        B, T, V, C, H, W = x.shape
        x = x.view(B * T * V, C, H, W)                # (B*T*4, C, H, W)
        x = self.feature_extractor(x)                # (B*T*4, out_dim, h, w)
        x = self.pool(x)                             # (B*T*4, out_dim, 1, 1)
        x = self.flatten(x)                          # (B*T*4, out_dim)
        x = x.view(B, T, V * self.out_dim)           # (B, T, 4*out_dim)
        return x
        