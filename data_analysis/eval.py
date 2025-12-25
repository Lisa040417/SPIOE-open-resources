import argparse
import torch
import torch.nn as nn
from data import get_dataloaders
from model import get_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, f1_score, accuracy_score
import matplotlib.pyplot as plt
import os


def evaluate(model, test_loader, device, class_names):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    return all_labels, all_preds, all_probs


def plot_pr_roc_curves(y_true, y_probs, class_names, save_dir):
    n_classes = len(class_names)
    # One-hot encode labels
    y_true_1hot = np.eye(n_classes)[y_true]
    # PR曲线
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_1hot[:, i], y_probs[:, i])
        plt.plot(recall, precision, lw=2, label=f"{class_names[i]}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('P-R Curve (per class)')
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pr_curve.png'))
    plt.close()
    # ROC曲线
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_1hot[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (per class)')
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='RSI-CB128 场景分类评估')
    parser.add_argument('--model_type', type=str, default='custom_cnn', choices=['custom_cnn', 'vgg16'], help='模型类型')
    parser.add_argument('--weights_path', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批量大小')
    parser.add_argument('--data_dir', type=str, default='RSI-CB128', help='数据集目录')
    parser.add_argument('--save_dir', type=str, default='.', help='结果保存目录')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据加载
    dataloaders = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)
    test_loader = dataloaders['test_loader']
    class_names = dataloaders['class_names']
    num_classes = len(class_names)

    # 模型
    model = get_model(args.model_type, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.weights_path, map_location=device))
    model.eval()

    # 评估
    y_true, y_pred, y_probs = evaluate(model, test_loader, device, class_names)

    # 分类报告
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    print("\n分类报告:")
    for cls in class_names:
        print(f"{cls:20s} Precision: {report[cls]['precision']:.4f}  Recall: {report[cls]['recall']:.4f}")
    print(f"整体准确率: {accuracy_score(y_true, y_pred):.4f}")
    print(f"整体F1分数: {f1_score(y_true, y_pred, average='weighted'):.4f}")

    # 保存报告
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, f"{args.model_type}_eval_report.txt"), 'w', encoding='utf-8') as f:
        for cls in class_names:
            f.write(f"{cls:20s} Precision: {report[cls]['precision']:.4f}  Recall: {report[cls]['recall']:.4f}\n")
        f.write(f"整体准确率: {accuracy_score(y_true, y_pred):.4f}\n")
        f.write(f"整体F1分数: {f1_score(y_true, y_pred, average='weighted'):.4f}\n")

    # 绘制P-R曲线和ROC曲线
    plot_pr_roc_curves(y_true, y_probs, class_names, args.save_dir)
    print(f"P-R曲线和ROC曲线已保存到: {args.save_dir}")

if __name__ == '__main__':
    main()

