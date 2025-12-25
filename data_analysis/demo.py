import argparse
import torch
from model import get_model
from data import get_dataloaders
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

def load_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img, img_tensor

def main():
    parser = argparse.ArgumentParser(description='RSI-CB128 单张图片分类演示')
    parser.add_argument('--model_type', type=str, default='custom_cnn', choices=['custom_cnn', 'vgg16'], help='模型类型')
    parser.add_argument('--weights_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--img_path', type=str, required=True, help='待预测图片路径')
    parser.add_argument('--data_dir', type=str, default='RSI-CB128', help='数据集目录（用于获取类别名）')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 获取类别名
    dataloaders = get_dataloaders(data_dir=args.data_dir, batch_size=1)
    class_names = dataloaders['class_names']
    num_classes = len(class_names)

    # 加载模型
    model = get_model(args.model_type, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()

    # 加载图片
    img, img_tensor = load_image(args.img_path)
    img_tensor = img_tensor.to(device)

    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        pred_class = class_names[pred_idx]
        confidence = probs[pred_idx]

    # 输出结果
    print(f"预测类别: {pred_class}")
    print(f"置信度: {confidence:.4f}")
    print("各类别置信度:")
    for i, cls in enumerate(class_names):
        print(f"{cls:20s}: {probs[i]:.4f}")

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"预测: {pred_class} (置信度: {confidence:.2f})")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()

