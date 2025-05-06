import torch
import torch.nn as nn
import torchvision.models as models

class SkyImageEncoder(nn.Module):
    def __init__(self, pretrained=True, shared_encoder=True):
        """
        天空图像编码器
        
        Args:
            pretrained: 是否使用预训练权重
            shared_encoder: 是否在4个视角间共享权重
        """
        super().__init__()
        # 加载预训练的ResNet18，去掉全连接层
        resnet = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # 去掉avg_pool和fc层
        self.shared_encoder = shared_encoder
        
        # 如果不共享权重，为每个视角创建独立的编码器
        if not shared_encoder:
            self.encoders = nn.ModuleList([
                nn.Sequential(*list(models.resnet18(pretrained=pretrained).children())[:-2])
                for _ in range(4)
            ])
        
        # 特征降维层
        self.reduction = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((16, 9))  # 将特征图压缩到16x9
        )
        
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, 4, 3, H, W)
        Returns:
            特征张量，形状为 (batch_size, seq_len, 256, 16, 9)
        """
        batch_size, seq_len = x.shape[:2]
        # 重塑输入以处理所有帧
        x = x.view(batch_size * seq_len * 4, 3, 224, 224)
        
        if self.shared_encoder:
            # 使用共享编码器
            features = self.encoder(x)  # (batch*seq*4, 512, H', W')
        else:
            # 分别处理每个视角
            features = []
            for i in range(4):
                view_x = x[i::4]  # 获取第i个视角的所有帧
                view_features = self.encoders[i](view_x)
                features.append(view_features)
            features = torch.cat(features, dim=0)
        
        # 降维
        features = self.reduction(features)  # (batch*seq*4, 256, 16, 9)
        
        # 重塑回序列形式，合并4个视角的特征
        features = features.view(batch_size, seq_len, 4, 256, 16, 9)
        features = features.mean(dim=2)  # 平均池化合并4个视角
        
        return features

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, padding):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,  # 4个门控
            kernel_size=kernel_size,
            padding=padding
        )
        
    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        
        # 分离门控值
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, dim=1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        c_next = (forgetgate * c) + (ingate * cellgate)
        h_next = outgate * torch.tanh(c_next)
        
        return h_next, c_next

class OzonePredictor(nn.Module):
    def __init__(self, input_size=(224, 224), pretrained=True, shared_encoder=True):
        """
        臭氧浓度预测模型
        
        Args:
            input_size: 输入图像大小
            pretrained: 是否使用预训练权重
            shared_encoder: 是否在4个视角间共享编码器权重
        """
        super().__init__()
        
        # 图像编码器
        self.encoder = SkyImageEncoder(pretrained=pretrained, shared_encoder=shared_encoder)
        
        # ConvLSTM参数
        self.hidden_channels = 64
        self.kernel_size = 3
        self.num_layers = 2
        
        # ConvLSTM层
        self.convlstm_cells = nn.ModuleList([
            ConvLSTMCell(
                input_channels=256 if i == 0 else self.hidden_channels,
                hidden_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size//2
            )
            for i in range(self.num_layers)
        ])
        
        # 解码器（类似U-Net的上采样路径）
        self.decoder = nn.Sequential(
            # 16x9 -> 20x15
            nn.ConvTranspose2d(self.hidden_channels, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 20x15 -> 40x30
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 最后的预测层
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # 假设标签已归一化到[0,1]
        )
        
    def _init_hidden(self, batch_size, spatial_size, device):
        height, width = spatial_size
        hidden = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
            c = torch.zeros(batch_size, self.hidden_channels, height, width, device=device)
            hidden.append((h, c))
        return hidden
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, 4, 3, H, W)
        Returns:
            预测的臭氧浓度图，形状为 (batch_size, 1, 40, 30)
        """
        batch_size, seq_len = x.shape[:2]
        device = x.device
        
        # 提取图像特征
        features = self.encoder(x)  # (batch_size, seq_len, 256, 16, 9)
        
        # 初始化ConvLSTM隐藏状态
        hidden = self._init_hidden(batch_size, (16, 9), device)
        
        # 逐帧处理序列
        for t in range(seq_len):
            current_input = features[:, t]  # (batch_size, 256, 16, 9)
            
            # 通过所有ConvLSTM层
            for layer_idx, cell in enumerate(self.convlstm_cells):
                h, c = hidden[layer_idx]
                h_next, c_next = cell(
                    current_input if layer_idx == 0 else h,
                    h, c
                )
                hidden[layer_idx] = (h_next, c_next)
                current_input = h_next
        
        # 使用最后一层的隐藏状态进行预测
        final_features = hidden[-1][0]  # 使用h而不是c
        
        # 解码得到最终预测
        prediction = self.decoder(final_features)
        
        return prediction

def main():
    # 测试模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OzonePredictor().to(device)
    
    # 创建示例输入
    batch_size = 2
    seq_len = 6
    x = torch.randn(batch_size, seq_len, 4, 3, 224, 224).to(device)
    
    # 前向传播
    output = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

if __name__ == "__main__":
    main()