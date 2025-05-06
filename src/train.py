from trainner import OzoneTrainer
from model.model import OzonePredictor



# 创建模型和训练器
model = OzonePredictor()
trainer = OzoneTrainer(model, train_loader, val_loader, config)

# 开始训练
trainer.train()