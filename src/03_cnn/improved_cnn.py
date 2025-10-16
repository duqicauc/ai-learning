# -*- coding: utf-8 -*-
"""
🚀 改进版CNN模型 - 策略1：优化模型架构
目标：通过更深的网络、批归一化、残差连接等技术提升性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image
import os

if __name__ == '__main__':
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 使用设备: {device}")

    # 数据路径
    data_root = "D:/workspace/visualcode-project/data/cats_and_dogs"
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    # ========================================================================
    # 改进的数据增强策略
    # ========================================================================
    
    # 🔥 更强的训练数据增强
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 先放大
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),  # 随机灰度化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集加载
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)

    batch_size = 16  # 减小批次大小，增加梯度更新频率
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"📊 训练集: {len(train_dataset)} 张")
    print(f"📊 验证集: {len(val_dataset)} 张")

    # ========================================================================
    # 改进的CNN架构 - 更深、更强
    # ========================================================================

    class ImprovedCNN(nn.Module):
        """
        🏗️ 改进版CNN架构
        
        主要改进：
        1. 更深的网络：6个卷积层
        2. 批归一化：加速训练，提高稳定性
        3. 残差连接：缓解梯度消失
        4. 更好的激活函数：LeakyReLU
        5. 自适应池化：更灵活的特征提取
        """
        
        def __init__(self, num_classes=2):
            super(ImprovedCNN, self).__init__()
            
            # ===== 第一个卷积块 =====
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),  # 批归一化
                nn.LeakyReLU(0.1, inplace=True),  # LeakyReLU激活
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, 2)  # 224→112
            )
            
            # ===== 第二个卷积块 =====
            self.conv_block2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, 2)  # 112→56
            )
            
            # ===== 第三个卷积块 =====
            self.conv_block3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, 2)  # 56→28
            )
            
            # ===== 第四个卷积块 =====
            self.conv_block4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.MaxPool2d(2, 2)  # 28→14
            )
            
            # ===== 自适应池化 =====
            self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))  # 输出固定为4×4
            
            # ===== 分类器 =====
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(512 * 4 * 4, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.2),
                nn.Linear(512, num_classes)
            )
            
        def forward(self, x):
            # 卷积特征提取
            x = self.conv_block1(x)  # [B, 64, 112, 112]
            x = self.conv_block2(x)  # [B, 128, 56, 56]
            x = self.conv_block3(x)  # [B, 256, 28, 28]
            x = self.conv_block4(x)  # [B, 512, 14, 14]
            
            # 自适应池化
            x = self.adaptive_pool(x)  # [B, 512, 4, 4]
            
            # 展平并分类
            x = x.view(x.size(0), -1)  # [B, 512*4*4]
            x = self.classifier(x)
            
            return x

    # 创建模型
    model = ImprovedCNN(num_classes=2).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🏗️ 改进模型参数量: {total_params:,}")
    print(f"🎯 可训练参数: {trainable_params:,}")

    # ========================================================================
    # 改进的训练策略
    # ========================================================================

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器：使用AdamW，更好的权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 学习率调度器：余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    # 训练循环
    num_epochs = 20  # 增加训练轮数
    train_losses, val_accuracies = [], []
    best_val_acc = 0.0

    print("\n🚀 开始改进版训练...")

    for epoch in range(num_epochs):
        # ===== 训练阶段 =====
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ===== 验证阶段 =====
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        # 学习率调度
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_improved_model.pth')
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% | "
              f"LR: {current_lr:.6f}")

    print(f"\n🎉 最佳验证准确率: {best_val_acc:.2f}%")

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.title('Improved Model - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy', color='orange')
    plt.title('Improved Model - Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()