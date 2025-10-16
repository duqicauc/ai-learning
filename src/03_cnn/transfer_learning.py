# -*- coding: utf-8 -*-
"""
🔄 迁移学习方案 - 策略2：使用预训练模型
目标：利用在ImageNet上预训练的模型，快速获得高性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
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
    # 针对预训练模型的数据预处理
    # ========================================================================
    
    # ImageNet预训练模型的标准预处理
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transform)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"📊 训练集: {len(train_dataset)} 张")
    print(f"📊 验证集: {len(val_dataset)} 张")

    # ========================================================================
    # 迁移学习模型选择
    # ========================================================================

    def create_transfer_model(model_name='resnet18', num_classes=2, freeze_features=True):
        """
        创建迁移学习模型
        
        Args:
            model_name: 预训练模型名称 ('resnet18', 'resnet50', 'efficientnet_b0')
            num_classes: 分类数量
            freeze_features: 是否冻结特征提取层
        """
        
        if model_name == 'resnet18':
            # ResNet18 - 轻量级，训练快
            model = models.resnet18(pretrained=True)
            num_features = model.fc.in_features
            
            if freeze_features:
                # 冻结所有特征层
                for param in model.parameters():
                    param.requires_grad = False
            
            # 替换最后的分类层
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
            
        elif model_name == 'resnet50':
            # ResNet50 - 更深，性能更好
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            
            if freeze_features:
                for param in model.parameters():
                    param.requires_grad = False
            
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, num_classes)
            )
            
        elif model_name == 'efficientnet_b0':
            # EfficientNet-B0 - 效率与性能平衡
            try:
                model = models.efficientnet_b0(pretrained=True)
                num_features = model.classifier[1].in_features
                
                if freeze_features:
                    for param in model.features.parameters():
                        param.requires_grad = False
                
                model.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(num_features, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(256, num_classes)
                )
            except:
                print("⚠️ EfficientNet不可用，使用ResNet18替代")
                return create_transfer_model('resnet18', num_classes, freeze_features)
        
        return model

    # ========================================================================
    # 多种迁移学习策略对比
    # ========================================================================

    def train_transfer_model(model, model_name, num_epochs=15):
        """训练迁移学习模型"""
        
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        
        # 不同的优化策略
        if 'frozen' in model_name:
            # 冻结特征层时，只训练分类器，可以用更大学习率
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
        else:
            # 微调整个网络时，用较小学习率
            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
        
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        train_losses, val_accuracies = [], []
        best_val_acc = 0.0
        
        print(f"\n🚀 开始训练 {model_name}...")
        
        for epoch in range(num_epochs):
            # 训练阶段
            model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
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
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_{model_name}.pth')
            
            scheduler.step()
            
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Loss: {avg_train_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%")
        
        return train_losses, val_accuracies, best_val_acc

    # ========================================================================
    # 实验不同的迁移学习策略
    # ========================================================================

    results = {}

    # 策略1: ResNet18 + 冻结特征层
    print("\n" + "="*60)
    print("🧊 策略1: ResNet18 + 冻结特征层 (快速训练)")
    print("="*60)
    
    model1 = create_transfer_model('resnet18', num_classes=2, freeze_features=True)
    losses1, accs1, best_acc1 = train_transfer_model(model1, 'resnet18_frozen', num_epochs=10)
    results['ResNet18_Frozen'] = {'losses': losses1, 'accs': accs1, 'best_acc': best_acc1}

    # 策略2: ResNet18 + 微调全网络
    print("\n" + "="*60)
    print("🔥 策略2: ResNet18 + 微调全网络 (更好性能)")
    print("="*60)
    
    model2 = create_transfer_model('resnet18', num_classes=2, freeze_features=False)
    losses2, accs2, best_acc2 = train_transfer_model(model2, 'resnet18_finetune', num_epochs=15)
    results['ResNet18_Finetune'] = {'losses': losses2, 'accs': accs2, 'best_acc': best_acc2}

    # 策略3: ResNet50 + 微调 (如果计算资源允许)
    print("\n" + "="*60)
    print("💪 策略3: ResNet50 + 微调 (最强性能)")
    print("="*60)
    
    model3 = create_transfer_model('resnet50', num_classes=2, freeze_features=False)
    losses3, accs3, best_acc3 = train_transfer_model(model3, 'resnet50_finetune', num_epochs=12)
    results['ResNet50_Finetune'] = {'losses': losses3, 'accs': accs3, 'best_acc': best_acc3}

    # ========================================================================
    # 结果对比可视化
    # ========================================================================

    plt.figure(figsize=(15, 5))

    # 训练损失对比
    plt.subplot(1, 3, 1)
    for name, data in results.items():
        plt.plot(data['losses'], label=name, linewidth=2)
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 验证准确率对比
    plt.subplot(1, 3, 2)
    for name, data in results.items():
        plt.plot(data['accs'], label=name, linewidth=2)
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 最佳准确率对比
    plt.subplot(1, 3, 3)
    names = list(results.keys())
    best_accs = [results[name]['best_acc'] for name in names]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = plt.bar(names, best_accs, color=colors, alpha=0.8)
    plt.title('Best Validation Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    # 在柱状图上显示数值
    for bar, acc in zip(bars, best_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

    # ========================================================================
    # 总结报告
    # ========================================================================

    print("\n" + "="*60)
    print("📊 迁移学习实验总结")
    print("="*60)
    
    for name, data in results.items():
        print(f"{name:20} | 最佳验证准确率: {data['best_acc']:.2f}%")
    
    best_model = max(results.items(), key=lambda x: x[1]['best_acc'])
    print(f"\n🏆 最佳模型: {best_model[0]} (准确率: {best_model[1]['best_acc']:.2f}%)")
    
    # 与原始模型对比
    original_acc = 65.71
    improvement = best_model[1]['best_acc'] - original_acc
    print(f"📈 相比原始模型提升: {improvement:.2f}个百分点")
    
    if improvement > 10:
        print("🎉 显著提升！迁移学习效果很好")
    elif improvement > 5:
        print("✅ 明显提升！迁移学习有效")
    else:
        print("⚠️ 提升有限，可能需要更多数据或其他策略")