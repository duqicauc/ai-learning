#!/usr/bin/env python3
"""
CIFAR-10图像分类器
使用AutoDL预置数据集进行训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import yaml
import argparse
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

class SimpleCNN(nn.Module):
    """简单的CNN模型用于CIFAR-10分类"""
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 卷积 + 池化
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = self.pool(self.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # 展平
        x = x.view(-1, 128 * 4 * 4)
        
        # 全连接
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_data_loaders(config):
    """获取数据加载器"""
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载CIFAR-10数据集
    data_path = config['dataset']['path']
    
    try:
        # 尝试从指定路径加载
        trainset = torchvision.datasets.CIFAR10(
            root=data_path, train=True, download=False, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_path, train=False, download=False, transform=transform_test
        )
    except:
        # 如果失败，自动下载到当前目录
        print("从指定路径加载失败，自动下载CIFAR-10数据集...")
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
    
    # 创建数据加载器
    trainloader = DataLoader(
        trainset, 
        batch_size=config['dataset']['batch_size'], 
        shuffle=True, 
        num_workers=config['dataset']['num_workers']
    )
    
    testloader = DataLoader(
        testset, 
        batch_size=config['dataset']['batch_size'], 
        shuffle=False, 
        num_workers=config['dataset']['num_workers']
    )
    
    return trainloader, testloader

def train_model(model, trainloader, testloader, config, device):
    """训练模型"""
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # 创建输出目录
    os.makedirs(config['paths']['model_save_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # 训练循环
    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{config["training"]["epochs"]}], '
                      f'Step [{i+1}/{len(trainloader)}], '
                      f'Loss: {running_loss/100:.4f}, '
                      f'Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0
        
        # 验证
        if (epoch + 1) % 2 == 0:
            test_accuracy = evaluate_model(model, testloader, device)
            print(f'Epoch [{epoch+1}] Test Accuracy: {test_accuracy:.2f}%')
        
        # 保存模型
        if (epoch + 1) % config['training']['save_interval'] == 0:
            save_path = os.path.join(
                config['paths']['model_save_dir'], 
                f'cifar10_cnn_epoch_{epoch+1}.pth'
            )
            torch.save(model.state_dict(), save_path)
            print(f'模型已保存到: {save_path}')

def evaluate_model(model, testloader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN训练')
    parser.add_argument('--config', type=str, default='configs/cifar10_autodl.yaml',
                        help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设备检查
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    # 创建模型
    model = SimpleCNN(num_classes=config['model']['num_classes'])
    model = model.to(device)
    
    print(f'模型参数数量: {sum(p.numel() for p in model.parameters()):,}')
    
    # 获取数据加载器
    print('加载数据集...')
    trainloader, testloader = get_data_loaders(config)
    print(f'训练集大小: {len(trainloader.dataset)}')
    print(f'测试集大小: {len(testloader.dataset)}')
    
    # 训练模型
    print('开始训练...')
    train_model(model, trainloader, testloader, config, device)
    
    # 最终评估
    final_accuracy = evaluate_model(model, testloader, device)
    print(f'最终测试准确率: {final_accuracy:.2f}%')
    
    # 保存最终模型
    final_save_path = os.path.join(
        config['paths']['model_save_dir'], 
        'cifar10_cnn_final.pth'
    )
    torch.save(model.state_dict(), final_save_path)
    print(f'最终模型已保存到: {final_save_path}')

if __name__ == '__main__':
    main()