from torchvision import transforms
import random

def get_train_transforms(image_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    返回训练集用的数据增强流水线
    包括随机裁剪、翻转、亮度/对比度抖动等
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_val_transforms(image_size=(224, 224), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    返回验证/测试集用的数据预处理流水线
    包括中心裁剪、缩放、标准化
    """
    return transforms.Compose([
        transforms.Resize(int(image_size[0] * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

class RandomCloudCover:
    """
    自定义模拟云遮挡的数据增强示例：随机擦除图像块
    可在 get_train_transforms 中插入该操作
    """
    def __init__(self, p=0.5, scale=(0.02, 0.15), ratio=(0.3, 3.3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        if random.random() > self.p:
            return img
        width, height = img.size
        area = height * width
        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))
            if w < width and h < height:
                x1 = random.randint(0, width - w)
                y1 = random.randint(0, height - h)
                img.paste((255, 255, 255), (x1, y1, x1 + w, y1 + h))
                return img
        return img