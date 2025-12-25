import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt

# 设置你的学号后两位
STUDENT_X = 8  # 学号倒数第二位
STUDENT_Y = 3  # 学号最后一位


# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 数据集类
class RSIDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        RSI-CB128数据集加载器
        Args:
            root_dir (string): 数据集根目录
            transform (callable, optional): 可选的图像变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.png', '.tif', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# 获取数据变换
def get_transforms(target_size=224):
    """
    根据学号后两位XY创建数据增强策略
    Returns:
        dict: 包含train_transform和test_transform的字典
    """
    # 测试集变换
    test_transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 训练集变换
    train_transforms = [
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    # 个性化增强：Y=3或7，随机垂直翻转
    if STUDENT_Y in [3, 7]:
        p_vflip = 0.25 + (STUDENT_X % 10) * 0.025  # 0.45
        train_transforms.insert(1, transforms.RandomVerticalFlip(p=p_vflip))
    return {
        'train': transforms.Compose(train_transforms),
        'test': test_transform
    }


# 获取数据加载器
def get_dataloaders(data_dir='RSI-CB128', batch_size=32):
    """
    创建训练和测试数据加载器
    Args:
        data_dir (string): 数据集目录
        batch_size (int): 批量大小
    Returns:
        dict: 包含train_loader, test_loader和class_names的字典
    """
    set_seed(42)
    target_image_size = 224  # 保证和模型一致
    transforms_dict = get_transforms(target_size=target_image_size)
    full_dataset = RSIDataset(
        root_dir=data_dir,
        transform=None
    )
    # 获取类别名称
    class_names = full_dataset.classes
    # 按类别收集样本索引
    class_indices = {}
    for idx, (_, label) in enumerate(full_dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    # 对每个类别进行80:20的划分（按自然顺序）
    train_indices = []
    test_indices = []
    for label, indices in class_indices.items():
        split = int(np.floor(0.8 * len(indices)))
        train_indices.extend(indices[:split])
        test_indices.extend(indices[split:])
    # 创建训练集和测试集
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    # 为训练集和测试集应用不同的变换
    train_dataset.dataset.transform = transforms_dict['train']
    test_dataset.dataset.transform = transforms_dict['test']
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'class_names': class_names,
        'train_dataset': train_dataset
    }

def visualize_augmentations(dataset, index=0, num_augmented=5):
    """
    可视化某张图像及其多个数据增强结果
    Args:
        dataset: 使用了 transform 的 Dataset
        index: 图像索引
        num_augmented: 显示几张增强图
    """
    img_path, label = dataset.dataset.samples[dataset.indices[index]]
    original_img = Image.open(img_path).convert('RGB')

    fig, axes = plt.subplots(2, num_augmented + 1, figsize=(3 * (num_augmented + 1), 6))
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title("img_before")
    axes[0, 0].axis('off')
    axes[1, 0].imshow(original_img)
    axes[1, 0].set_title("img_after")
    axes[1, 0].axis('off')

    for i in range(num_augmented):
        transformed_img = dataset.dataset.transform(original_img)
        # 将tensor转为图像
        img_np = transformed_img.permute(1, 2, 0).numpy()
        img_np = img_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]  # 去标准化
        img_np = np.clip(img_np, 0, 1)
        axes[0, i + 1].imshow(img_np)
        axes[0, i + 1].set_title(f"add{i+1}")
        axes[0, i + 1].axis('off')
        axes[1, i + 1].imshow(img_np)
        axes[1, i + 1].axis('off')

    plt.suptitle("fig1", fontsize=16)
    plt.tight_layout()
    plt.savefig("augmentation_visualization.png", dpi=300)


if __name__ == "__main__":
    # 简单测试数据加载是否正常
    data = get_dataloaders(batch_size=4)
    train_loader = data['train_loader']
    train_dataset = data['train_dataset']
    class_names = data['class_names']
    print(f"类别数量: {len(class_names)}")
    print(f"类别名称: {class_names}")
    # 获取并显示一个批次
    images, labels = next(iter(train_loader))
    print(f"图像张量形状: {images.shape}")
    print(f"标签: {labels}")
    visualize_augmentations(train_dataset, index=0, num_augmented=5)