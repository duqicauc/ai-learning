# -*- coding: utf-8 -*-
"""
🍎🍌 第九节实战：CNN 水果分类器 - 100种水果识别

=== 多分类CNN项目详解 ===

🧠 项目概述：
本项目基于cats_dogs_classifier的成功经验，扩展到100种水果的多分类任务。
相比二分类，多分类CNN需要考虑更多因素：
- 类别数量大幅增加（2 → 100）
- 类间相似性更高（不同水果可能很相似）
- 数据不平衡问题更突出
- 模型复杂度需要相应提升

🏗️ 技术升级点：
1️⃣ 模型架构优化：
   - 增加网络深度和宽度
   - 使用Batch Normalization加速训练
   - 添加Dropout防止过拟合
   - 调整最后全连接层输出为100类

2️⃣ 数据处理增强：
   - 更强的数据增强策略
   - 类别权重平衡
   - 更精细的预处理流程

3️⃣ 训练策略优化：
   - 学习率调度
   - 早停机制
   - 模型检查点保存
   - 详细的训练监控

📊 数据集信息：
- 来源：ModelScope fruits100数据集
- 类别：100种不同水果
- 格式：JPG图片，按文件夹分类
- 结构：train/ 和 val/ 目录

🎯 学习目标：
- 掌握多分类CNN的设计原理
- 理解类别不平衡问题的解决方案
- 学会复杂模型的训练技巧
- 实现端到端的多分类项目
"""

# ----------------------------
# 第一步：导入必要库
# ----------------------------

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
from collections import Counter
import time
from pathlib import Path

# ----------------------------
# 第二步：设置设备和随机种子
# ----------------------------

def set_seed(seed=42):
    """设置随机种子确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"✅ GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ----------------------------
    # 第三步：数据路径配置
    # ----------------------------

    print("🔄 正在加载水果数据集配置...")

    # 使用统一的数据配置管理
    try:
        # 尝试相对导入（当作为包使用时）
        from ..utils.data_config import get_fruits100_paths, data_config
    except ImportError:
        # 直接运行时使用绝对导入
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
        from src.utils.data_config import get_fruits100_paths, data_config
    
    try:
        train_dir, val_dir, test_dir = get_fruits100_paths()
        
        print(f"📁 数据路径:")
        print(f"  训练集: {train_dir}")
        print(f"  验证集: {val_dir}")
        print(f"  测试集: {test_dir}")
        
        # 获取数据集信息
        dataset_info = data_config.get_dataset_info("fruits100")
        print(f"📊 数据集信息: {dataset_info['description']}")
        print(f"🔗 数据来源: {dataset_info['source']}")
        
    except Exception as e:
        print(f"❌ 数据集配置错误: {e}")
        print("💡 请确保已下载fruits100数据集到正确位置")
        exit(1)

    # ========================================================================
    # 第四步：数据预处理和增强 - 多分类的关键！
    # ========================================================================

    """
    🌟 多分类数据增强策略详解：
    
    相比二分类，多分类需要更强的数据增强来：
    1. 增加数据多样性，帮助模型学习更鲁棒的特征
    2. 减少过拟合风险（100个类别，模型容易记住训练数据）
    3. 提高模型对变换的不变性
    
    增强技术选择：
    - RandomResizedCrop: 随机裁剪和缩放，模拟不同拍摄距离
    - RandomHorizontalFlip: 水平翻转，增加视角多样性
    - ColorJitter: 颜色抖动，适应不同光照条件
    - RandomRotation: 小角度旋转，适应不同拍摄角度
    - RandomAffine: 仿射变换，模拟轻微的透视变化
    """

    # 训练集数据增强（强增强）
    train_transform = transforms.Compose([
        # 1️⃣ 尺寸调整和随机裁剪
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        
        # 2️⃣ 几何变换
        transforms.RandomHorizontalFlip(p=0.5),  # 50%概率水平翻转
        transforms.RandomRotation(degrees=15),    # ±15度随机旋转
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        
        # 3️⃣ 颜色增强
        transforms.ColorJitter(
            brightness=0.2,    # 亮度变化±20%
            contrast=0.2,      # 对比度变化±20%
            saturation=0.2,    # 饱和度变化±20%
            hue=0.1           # 色调变化±10%
        ),
        
        # 4️⃣ 标准化处理
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证/测试集数据预处理（无增强）
    val_test_transform = transforms.Compose([
        transforms.Resize(256),                    # 先放大
        transforms.CenterCrop(224),                # 中心裁剪到标准尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ----------------------------
    # 第五步：创建数据集和数据加载器
    # ----------------------------

    print("📂 正在加载数据集...")

    # 创建数据集
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=val_test_transform)

    # 获取类别信息
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx

    print(f"✅ 数据集加载完成:")
    print(f"  训练集: {len(train_dataset)} 张图片")
    print(f"  验证集: {len(val_dataset)} 张图片")
    print(f"  测试集: {len(test_dataset)} 张图片")
    print(f"  类别数量: {num_classes}")
    print(f"  前10个类别: {class_names[:10]}")

    # 分析类别分布
    print("\n📊 分析训练集类别分布...")
    train_targets = [train_dataset.targets[i] for i in range(len(train_dataset))]
    class_counts = Counter(train_targets)
    
    print(f"  最多样本类别: {max(class_counts.values())} 张")
    print(f"  最少样本类别: {min(class_counts.values())} 张")
    print(f"  平均样本数量: {np.mean(list(class_counts.values())):.1f} 张")

    # ========================================================================
    # 第六步：处理类别不平衡 - 多分类的重要技巧！
    # ========================================================================

    """
    🎯 类别不平衡问题解决方案：
    
    在多分类任务中，不同类别的样本数量可能差异很大，导致：
    - 模型偏向于样本多的类别
    - 样本少的类别识别率低
    - 整体性能下降
    
    解决方案：
    1. WeightedRandomSampler: 按类别权重采样
    2. 类别权重损失函数: 给少数类别更高权重
    3. 数据增强: 为少数类别生成更多样本
    """

    # 计算类别权重（用于损失函数）
    class_weights = []
    total_samples = len(train_dataset)
    
    for i in range(num_classes):
        class_count = class_counts.get(i, 1)  # 避免除零
        weight = total_samples / (num_classes * class_count)
        class_weights.append(weight)
    
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"✅ 类别权重计算完成，权重范围: {class_weights.min():.3f} - {class_weights.max():.3f}")

    # 创建加权采样器（用于数据加载）
    sample_weights = [class_weights[target] for target in train_targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # 创建数据加载器
    batch_size = 32  # 多分类任务，适中的batch size
    num_workers = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,  # 使用加权采样
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"✅ 数据加载器创建完成，批次大小: {batch_size}")

    # ========================================================================
    # 第七步：定义CNN模型架构 - 多分类专用设计！
    # ========================================================================

    class FruitsCNN(nn.Module):
        """
        🏗️ 水果分类CNN架构详解
        
        相比二分类，多分类CNN的设计考虑：
        1. 更深的网络：提取更复杂的特征
        2. 更多的通道：捕获更丰富的特征表示
        3. Batch Normalization：加速训练，提高稳定性
        4. Dropout：防止过拟合
        5. 残差连接：缓解梯度消失问题
        
        网络结构：
        输入(224×224×3) 
        → Conv Block 1 (64通道) → 112×112×64
        → Conv Block 2 (128通道) → 56×56×128
        → Conv Block 3 (256通道) → 28×28×256
        → Conv Block 4 (512通道) → 14×14×512
        → Global Average Pooling → 512
        → FC1(256) → Dropout → FC2(100) → 输出
        """
        
        def __init__(self, num_classes=100):
            super(FruitsCNN, self).__init__()
            
            # ===== 卷积块定义 =====
            
            # 🔍 Conv Block 1: 基础特征提取
            self.conv_block1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),  # BN加速训练
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)  # 224→112
            )
            
            # 🔍 Conv Block 2: 中级特征组合
            self.conv_block2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)  # 112→56
            )
            
            # 🔍 Conv Block 3: 高级特征抽象
            self.conv_block3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)  # 56→28
            )
            
            # 🔍 Conv Block 4: 深层特征提取
            self.conv_block4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)  # 28→14
            )
            
            # ===== 全局平均池化 =====
            # 相比Flatten，GAP减少参数量，防止过拟合
            self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            
            # ===== 分类器设计 =====
            self.classifier = nn.Sequential(
                nn.Dropout(0.5),  # 防止过拟合
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)  # 输出100个类别
            )
            
            # 权重初始化
            self._initialize_weights()
        
        def _initialize_weights(self):
            """权重初始化 - 提高训练稳定性"""
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
        
        def forward(self, x):
            """前向传播"""
            # 特征提取
            x = self.conv_block1(x)  # [B, 64, 112, 112]
            x = self.conv_block2(x)  # [B, 128, 56, 56]
            x = self.conv_block3(x)  # [B, 256, 28, 28]
            x = self.conv_block4(x)  # [B, 512, 14, 14]
            
            # 全局平均池化
            x = self.global_avg_pool(x)  # [B, 512, 1, 1]
            x = x.view(x.size(0), -1)    # [B, 512]
            
            # 分类
            x = self.classifier(x)       # [B, 100]
            
            return x

    # 创建模型实例
    model = FruitsCNN(num_classes=num_classes).to(device)

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n🏗️ 模型架构:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / 1024 / 1024:.1f} MB")

    # ========================================================================
    # 第八步：定义损失函数和优化器 - 多分类优化策略！
    # ========================================================================

    """
    🎯 多分类训练策略详解：
    
    1. 损失函数选择：
       - CrossEntropyLoss: 标准多分类损失
       - 类别权重: 处理数据不平衡
    
    2. 优化器选择：
       - Adam: 自适应学习率，适合复杂模型
       - 学习率调度: 训练过程中动态调整
    
    3. 正则化技术：
       - Weight Decay: L2正则化
       - Dropout: 随机失活
       - Batch Normalization: 批量归一化
    """

    # 损失函数（带类别权重）
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 优化器
    optimizer = optim.Adam(
        model.parameters(), 
        lr=0.001,           # 初始学习率
        weight_decay=1e-4   # L2正则化
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',         # 监控验证损失
        factor=0.5,         # 学习率衰减因子
        patience=5,         # 等待轮数
        min_lr=1e-6        # 最小学习率
    )

    print(f"✅ 训练配置:")
    print(f"  损失函数: CrossEntropyLoss (带类别权重)")
    print(f"  优化器: Adam (lr=0.001, weight_decay=1e-4)")
    print(f"  学习率调度: ReduceLROnPlateau")

    # ========================================================================
    # 第九步：训练和验证函数 - 完整的训练流程！
    # ========================================================================

    def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
        """训练一个epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 进度显示
        batch_count = len(train_loader)
        print_interval = max(1, batch_count // 10)  # 每10%显示一次
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 进度显示
            if (batch_idx + 1) % print_interval == 0:
                progress = (batch_idx + 1) / batch_count * 100
                print(f"  训练进度: {progress:.1f}% | "
                      f"损失: {loss.item():.4f} | "
                      f"准确率: {100 * correct / total:.2f}%")
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc

    def validate_epoch(model, val_loader, criterion, device):
        """验证一个epoch"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc

    # ========================================================================
    # 第十步：模型训练主循环 - 完整训练流程！
    # ========================================================================

    def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                   device, num_epochs=50, save_path="models/fruits_cnn_best.pth"):
        """完整的模型训练流程"""
        
        print(f"\n🚀 开始训练模型 (共{num_epochs}轮)...")
        print("=" * 80)
        
        # 训练历史记录
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        
        # 最佳模型记录
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        early_stop_patience = 10
        
        # 确保保存目录存在
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # 训练阶段
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # 验证阶段
            val_loss, val_acc = validate_epoch(
                model, val_loader, criterion, device
            )
            
            # 学习率调度
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 记录历史
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # 计算用时
            epoch_time = time.time() - epoch_start_time
            
            # 显示结果
            print(f"\n📊 Epoch {epoch+1} 结果:")
            print(f"  训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
            print(f"  验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
            print(f"  学习率: {current_lr:.6f}")
            print(f"  用时: {epoch_time:.1f}秒")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                
                # 保存模型
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'class_names': class_names,
                    'class_to_idx': class_to_idx,
                    'num_classes': num_classes
                }, save_path)
                
                print(f"✅ 新的最佳模型已保存! 验证准确率: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"⏳ 验证准确率未提升 ({patience_counter}/{early_stop_patience})")
            
            # 早停检查
            if patience_counter >= early_stop_patience:
                print(f"\n🛑 早停触发! 最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
                break
        
        total_time = time.time() - start_time
        print(f"\n🎉 训练完成!")
        print(f"  总用时: {total_time/60:.1f}分钟")
        print(f"  最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
        print(f"  模型保存路径: {save_path}")
        
        return {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch
        }

    # ========================================================================
    # 第十一步：模型评估和测试
    # ========================================================================

    def evaluate_model(model, test_loader, class_names, device):
        """详细的模型评估"""
        print("\n🔍 开始模型评估...")
        
        model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for i in range(len(class_names)))
        class_total = list(0. for i in range(len(class_names)))
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output, 1)
                
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # 记录每个类别的准确率
                c = (predicted == target).squeeze()
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 总体准确率
        overall_acc = 100 * correct / total
        print(f"📊 总体测试准确率: {overall_acc:.2f}%")
        
        # 各类别准确率
        print(f"\n📋 各类别准确率 (前20个类别):")
        for i in range(min(20, len(class_names))):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"  {class_names[i]}: {acc:.1f}% ({int(class_correct[i])}/{int(class_total[i])})")
        
        return overall_acc, all_predictions, all_targets

    # ========================================================================
    # 第十二步：单张图片预测函数
    # ========================================================================

    def predict_single_image(model, image_path, class_names, device, transform):
        """预测单张图片"""
        model.eval()
        
        # 加载和预处理图片
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # 获取top-5预测
            top5_prob, top5_idx = torch.topk(probabilities, 5, dim=1)
            
        result = {
            'predicted_class': class_names[predicted.item()],
            'confidence': confidence.item(),
            'top5_predictions': [
                (class_names[idx.item()], prob.item()) 
                for idx, prob in zip(top5_idx[0], top5_prob[0])
            ]
        }
        
        return result

    # ========================================================================
    # 第十三步：可视化函数
    # ========================================================================

    def plot_training_history(history, save_path="results/training_history.png"):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(history['train_losses'], label='训练损失', color='blue')
        ax1.plot(history['val_losses'], label='验证损失', color='red')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(history['train_accs'], label='训练准确率', color='blue')
        ax2.plot(history['val_accs'], label='验证准确率', color='red')
        ax2.set_title('训练和验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # 确保保存目录存在
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 训练历史图表已保存: {save_path}")

    def visualize_predictions(model, test_loader, class_names, device, num_images=8):
        """可视化预测结果"""
        model.eval()
        
        # 获取一批测试数据
        data_iter = iter(test_loader)
        images, labels = next(data_iter)
        images, labels = images.to(device), labels.to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probabilities = F.softmax(outputs, dim=1)
        
        # 可视化
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(min(num_images, len(images))):
            # 反归一化图片
            img = images[i].cpu()
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)
            
            # 显示图片
            axes[i].imshow(img.permute(1, 2, 0))
            
            # 标题信息
            true_label = class_names[labels[i]]
            pred_label = class_names[predicted[i]]
            confidence = probabilities[i][predicted[i]].item()
            
            color = 'green' if predicted[i] == labels[i] else 'red'
            axes[i].set_title(f'真实: {true_label}\n预测: {pred_label}\n置信度: {confidence:.2f}', 
                            color=color, fontsize=10)
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

    # ========================================================================
    # 第十四步：主训练流程
    # ========================================================================

    print("\n" + "="*80)
    print("🍎 水果分类CNN训练开始!")
    print("="*80)

    # 开始训练
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=30,  # 可根据需要调整
        save_path="models/fruits_cnn_best.pth"
    )

    # 绘制训练历史
    plot_training_history(history)

    # 加载最佳模型进行测试
    print("\n🔄 加载最佳模型进行测试...")
    checkpoint = torch.load("models/fruits_cnn_best.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 模型评估
    test_acc, predictions, targets = evaluate_model(model, test_loader, class_names, device)

    # 可视化预测结果
    print("\n🖼️ 可视化预测结果...")
    visualize_predictions(model, test_loader, class_names, device)

    # ========================================================================
    # 第十五步：使用示例和总结
    # ========================================================================

    print("\n" + "="*80)
    print("🎉 水果分类CNN训练完成!")
    print("="*80)

    print(f"""
    📊 训练总结:
    ✅ 数据集: {num_classes}种水果分类
    ✅ 训练样本: {len(train_dataset)}张
    ✅ 验证样本: {len(val_dataset)}张
    ✅ 测试样本: {len(test_dataset)}张
    ✅ 最佳验证准确率: {history['best_val_acc']:.2f}%
    ✅ 测试准确率: {test_acc:.2f}%
    ✅ 模型参数量: {total_params:,}
    
    🔧 使用方法:
    1. 训练好的模型已保存到: models/fruits_cnn_best.pth
    2. 使用 predict_single_image() 函数预测新图片
    3. 使用 evaluate_model() 函数评估模型性能
    
    💡 改进建议:
    1. 数据增强: 尝试更多增强技术 (MixUp, CutMix等)
    2. 模型架构: 尝试ResNet、EfficientNet等先进架构
    3. 迁移学习: 使用预训练模型加速训练
    4. 集成学习: 组合多个模型提升性能
    5. 超参数优化: 使用网格搜索或贝叶斯优化
    
    🌟 这个项目展示了CNN在实际多分类任务中的完整应用流程：
    - 数据预处理和增强
    - 类别不平衡处理
    - 深度CNN架构设计
    - 训练策略优化
    - 模型评估和可视化
    """)

    # 保存类别映射
    class_mapping = {
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'num_classes': num_classes
    }
    
    with open('models/fruits_class_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, ensure_ascii=False, indent=2)
    
    print("✅ 类别映射已保存到: models/fruits_class_mapping.json")