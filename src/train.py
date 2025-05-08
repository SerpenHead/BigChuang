from model.CNN_feature_extractor import  CNNFeatureExtractor
from model.model1 import TemporalLSTM
from dataloader1 import ImageSequenceDataset

cnn = CNNFeatureExtractor(out_dim=512)              # 每张图抽 512 维特征，4 张图拼接后为 2048
lstm = TemporalLSTM(input_dim=2048, hidden_dim=256) # 输出为 256 向量
dataloader = ImageSequenceDataset

for batch in dataloader:
    imgs = batch['images']      # (B, T, 4, C, H, W)
    labels = batch['labels']    # (B, T, D)

    features = cnn(imgs)        # (B, T, 2048)
    seq_output = lstm(features) # (B, 256)

    # 接下去可以：
    # nn.Linear(256, num_classes) -> 分类
    # 或者 seq_output → decoder → 生成预测值