#!/usr/bin/env python3
"""
本地快速验证脚本 - 水果分类模型
避免Unicode编码问题的简化版本
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import yaml
import time

def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"配置文件加载失败: {e}")
        return None

def create_data_loaders(data_path, config):
    """创建数据加载器"""
    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = datasets.ImageFolder(
        root=data_path / "train",
        transform=transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=data_path / "val", 
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    return train_loader, val_loader, len(train_dataset.classes)

def create_model(num_classes, config):
    """创建模型"""
    if config['model']['architecture'] == 'resnet18':
        model = models.resnet18(pretrained=config['model']['pretrained'])
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"不支持的模型架构: {config['model']['architecture']}")
    
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 5 == 0:  # 每5个batch打印一次
            print(f"  Batch {i+1}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={100.*correct/total:.2f}%")
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def main():
    """主函数"""
    print("=" * 60)
    print("水果分类模型 - 本地快速验证")
    print("=" * 60)
    
    # 项目根目录
    project_root = Path(__file__).parent.parent
    
    # 加载配置
    config_path = project_root / "configs" / "local.yaml"
    config = load_config(config_path)
    
    if config is None:
        print("配置文件加载失败，退出")
        return
    
    print(f"配置文件: {config_path}")
    print(f"数据集: {config['data']['dataset']}")
    print(f"设备: {config['training']['device']}")
    print(f"批次大小: {config['training']['batch_size']}")
    print(f"训练轮数: {config['training']['epochs']}")
    
    # 设置设备
    device = torch.device(config['training']['device'])
    print(f"使用设备: {device}")
    
    # 数据路径
    data_path = project_root / "data" / config['data']['dataset']
    if not data_path.exists():
        print(f"数据路径不存在: {data_path}")
        return
    
    print(f"数据路径: {data_path}")
    
    # 创建数据加载器
    try:
        train_loader, val_loader, num_classes = create_data_loaders(data_path, config)
        print(f"类别数量: {num_classes}")
        print(f"训练样本: {len(train_loader.dataset)}")
        print(f"验证样本: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 创建模型
    try:
        model = create_model(num_classes, config)
        model = model.to(device)
        print(f"模型架构: {config['model']['architecture']}")
        print(f"预训练: {config['model']['pretrained']}")
    except Exception as e:
        print(f"模型创建失败: {e}")
        return
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    print("\n" + "=" * 60)
    print("开始训练...")
    print("=" * 60)
    
    # 训练循环
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        print("-" * 40)
        
        # 训练
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_time = time.time() - start_time
        
        # 验证
        start_time = time.time()
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_time = time.time() - start_time
        
        print(f"\n训练结果:")
        print(f"  损失: {train_loss:.4f} | 准确率: {train_acc:.2f}% | 时间: {train_time:.1f}s")
        print(f"验证结果:")
        print(f"  损失: {val_loss:.4f} | 准确率: {val_acc:.2f}% | 时间: {val_time:.1f}s")
    
    print("\n" + "=" * 60)
    print("本地验证完成!")
    print("=" * 60)
    print(f"最终验证准确率: {val_acc:.2f}%")
    
    if val_acc > 10:  # 随机猜测应该是1%左右
        print("验证成功: 模型能够学习到有用的特征")
    else:
        print("验证警告: 模型性能接近随机猜测")

if __name__ == "__main__":
    main()