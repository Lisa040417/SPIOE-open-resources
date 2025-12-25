import torch
import torch.nn as nn
import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels * 2, kernel_size=1) if in_channels != out_channels * 2 else nn.Identity()

    def forward(self, x):
        out1 = self.conv3(x)
        out2 = self.conv5(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.bn(out)
        shortcut = self.shortcut(x)
        out += shortcut
        out = self.relu(out)
        return out

class CustomCNN(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.layer1 = ResidualBlock(32, 32)   # 输出64
        self.pool1 = nn.MaxPool2d(2, 2)
        self.layer2 = ResidualBlock(64, 64)   # 输出128
        self.pool2 = nn.MaxPool2d(2, 2)
        self.layer3 = ResidualBlock(128, 128) # 输出256
        self.pool3 = nn.MaxPool2d(2, 2)
        self.layer4 = ResidualBlock(256, 256) # 输出512
        self.pool4 = nn.AdaptiveAvgPool2d(1)  # 全局池化
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)
        x = self.pool3(x)
        x = self.layer4(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def get_vgg16(num_classes=20):
    model = models.vgg16(pretrained=False)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def get_model(model_type, num_classes=20):
    if model_type == 'custom_cnn':
        return CustomCNN(num_classes=num_classes)
    elif model_type == 'vgg16':
        return get_vgg16(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg16 = get_model('vgg16').to(device)
    print("VGG16模型结构:")
    print(vgg16)
    custom_cnn = get_model('custom_cnn').to(device)
    print("自定义CNN模型结构:")
    print(custom_cnn)
    x = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        y = vgg16(x)
        y_custom = custom_cnn(x)
    print(f"VGG16输出形状: {y.shape}")
    print(f"自定义CNN输出形状: {y_custom.shape}")