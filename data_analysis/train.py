import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders, set_seed
from model import get_model
import os

def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    avg_loss = running_loss / total
    acc = correct / total
    print(f"Train Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.4f}")
    return avg_loss, acc

# 评估
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    avg_loss = running_loss / total
    acc = correct / total
    print(f"Test: Loss={avg_loss:.4f}, Acc={acc:.4f}")
    return avg_loss, acc

def main():
    parser = argparse.ArgumentParser(description='RSI-CB128 场景分类训练')
    parser.add_argument('--model_type', type=str, default='custom_cnn', choices=['custom_cnn', 'vgg16'], help='模型类型')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--data_dir', type=str, default='RSI-CB128', help='数据集目录')
    parser.add_argument('--save_dir', type=str, default='.', help='模型保存目录')
    args = parser.parse_args()

    set_seed(42)  # 保证可复现
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据加载
    dataloaders = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)
    train_loader = dataloaders['train_loader']
    test_loader = dataloaders['test_loader']
    num_classes = len(dataloaders['class_names'])

    # 模型
    model = get_model(args.model_type, num_classes=num_classes).to(device)
    print(model)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, criterion, optimizer, device, epoch)
        _, test_acc = evaluate(model, test_loader, criterion, device)
        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"{args.model_type}_final.pth")
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存到: {save_path} (best acc: {best_acc:.4f})")

if __name__ == '__main__':
    main()

