import torch
import torch.nn as nn
from model.CNN_feature_extractor import CNNFeatureExtractor

class TemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,       # 输入特征维度，例如 2048
            hidden_size=hidden_dim,     # LSTM 隐状态维度
            batch_first=True,
            num_layers=num_layers,     # !todo 超参数可以调
            bidirectional=bidirectional
        )
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x):
        """
        输入 x: (B, T, input_dim)
        输出:  (B, output_dim) — 可用于分类/回归任务
        """
        out, _ = self.lstm(x)      # out: (B, T, H)
        # final_output = out[:, -1]  # 使用最后一个时间步输出
        # return final_output
        return out



class OzonePredictor(nn.Module):
    def __init__(self, cnn_out_dim=128, lstm_hidden_dim=64, output_dim=849):
        super().__init__()
        self.cnn = CNNFeatureExtractor(out_dim=cnn_out_dim)
        self.lstm = TemporalLSTM(cnn_out_dim * 4, lstm_hidden_dim)
        self.regressor = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x):  # x: (B, T, 4, C, H, W)
        feats = self.cnn(x)  # (B, T, 4*out_dim)
        # print("cnn feature shape:",feats.shape, flush=True) # torch.Size([4, 3, 2048])
        lstm_out = self.lstm(feats)  # (B, hidden_dim)
        # print("lstm_out shape:",lstm_out.shape, flush=True) # torch.Size([4, 3, 64])
        out = self.regressor(lstm_out)  # (B, output_dim)
        # print("out shape:",out.shape, flush=True) # torch.Size([4, 3, 2048])
        return out