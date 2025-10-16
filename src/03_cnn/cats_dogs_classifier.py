# -*- coding: utf-8 -*-
"""
🐱🐶 第八节实战：从零训练 CNN 识别猫 vs 狗

=== CNN (卷积神经网络) 核心概念 ===

🧠 什么是CNN？
CNN (Convolutional Neural Network) 是专门处理图像数据的深度学习模型。
与传统全连接网络不同，CNN通过以下三个核心操作来理解图像：

1️⃣ 卷积 (Convolution)：
   - 使用卷积核(滤波器)在图像上滑动，提取局部特征
   - 例如：边缘检测、纹理识别、形状检测
   - 参数共享：同一个卷积核可以检测图像任意位置的相同特征

2️⃣ 池化 (Pooling)：
   - 降低特征图尺寸，减少计算量
   - 保留重要信息，增强模型对位置变化的鲁棒性
   - 常用：MaxPooling(取最大值)、AvgPooling(取平均值)

3️⃣ 全连接 (Fully Connected)：
   - 将提取的特征映射到最终分类结果
   - 类似传统神经网络的输出层

🏗️ 本项目CNN架构：
输入图像(224×224×3) 
→ Conv1(32个3×3卷积核) → ReLU → MaxPool(2×2) → (112×112×32)
→ Conv2(64个3×3卷积核) → ReLU → MaxPool(2×2) → (56×56×64)  
→ Conv3(128个3×3卷积核) → ReLU → MaxPool(2×2) → (28×28×128)
→ Flatten → FC1(512神经元) → Dropout(0.5) → FC2(2神经元) → 输出概率

📊 数据集结构：
data/cats_and_dogs/
    ├── train/          # 训练集 (275张)
    │   ├── cat/        # 猫的图片
    │   └── dog/        # 狗的图片
    ├── val/            # 验证集 (70张)
    │   ├── cat/
    │   └── dog/
    └── README.md

🎯 学习目标：
- 理解CNN的工作原理和各层作用
- 掌握数据增强技术防止过拟合
- 学会训练、验证、测试的完整流程
- 实现端到端的图像分类项目
"""

# ----------------------------
# 第一步：导入库
# ----------------------------

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models  # models 可选（本节用自定义CNN）
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from PIL import Image  # 用于加载单张新图片
# from modelscope.msdatasets import MsDataset
# from modelscope.utils.constant import DownloadMode

# ----------------------------
# 第二步：设置设备
# ----------------------------

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")

    # ----------------------------
    # 第二步：设置数据路径
    # ----------------------------

    print("🔄 正在加载真实的猫狗数据集...")

    # 使用统一的数据配置管理
    from ..utils.data_config import get_cats_dogs_paths
    train_dir, val_dir, test_dir = get_cats_dogs_paths()
    
    print(f"📁 数据路径:")
    print(f"  训练集: {train_dir}")
    print(f"  验证集: {val_dir}")
    print(f"  测试集: {test_dir}")

    # ========================================================================
    # 第五步：定义图像变换（Transforms）— CNN数据预处理的核心！
    # ========================================================================

    """
    🖼️ 为什么CNN需要数据预处理？
    
    1. 统一输入尺寸：CNN要求固定的输入维度 (batch_size, channels, height, width)
    2. 数值归一化：将像素值从[0,255]缩放到[0,1]或[-1,1]，加速收敛
    3. 数据增强：人工扩充训练数据，提高模型泛化能力
    4. 标准化：使用预训练模型的统计信息，便于迁移学习
    
    🌟 数据增强 (Data Augmentation) 的重要性：
    - 问题：深度学习需要大量数据，但收集标注数据成本高
    - 解决：通过变换生成"新"样本，让模型见过更多变化
    - 原理：真实世界的猫狗照片会有不同角度、光照、位置
    - 效果：显著减少过拟合，提升模型在新数据上的表现
    """

    # 🚀 训练集变换（含数据增强）
    # 目标：让模型学会识别各种变化下的猫狗特征
    train_transform = transforms.Compose([
        
        # 1️⃣ 随机水平翻转 (p=0.5表示50%概率翻转)
        # 原理：猫的镜像依然是猫，增加样本多样性
        # CNN优势：卷积操作具有平移不变性，但需要学习翻转不变性
        transforms.RandomHorizontalFlip(p=0.5),
        
        # 2️⃣ 随机旋转 (±10度范围内)
        # 原理：模拟拍照时的轻微倾斜，增强鲁棒性
        # 注意：角度不宜过大，避免破坏图像语义
        transforms.RandomRotation(degrees=10),
        
        # 3️⃣ 随机裁剪+缩放 (Random Resized Crop)
        # 原理：模拟不同拍摄距离和构图方式
        # scale=(0.8, 1.0)：裁剪面积为原图80%-100%
        # 最终resize到224×224：CNN需要固定输入尺寸
        transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        
        # 4️⃣ 转换为PyTorch张量 + 归一化到[0,1]
        # 原理：PIL图像(0-255) → Tensor(0.0-1.0)
        # 好处：小数值有利于梯度计算和权重更新
        transforms.ToTensor(),
        
        # 5️⃣ 标准化 (Normalization)
        # 使用ImageNet数据集的RGB通道均值和标准差
        # R通道: mean=0.485, std=0.229
        # G通道: mean=0.456, std=0.224  
        # B通道: mean=0.406, std=0.225
        # 公式：normalized = (pixel - mean) / std
        # 作用：将数据分布调整为均值0、标准差1，加速训练收敛
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 🎯 验证/测试集变换（不使用数据增强！）
    # 原则：评估时要保持数据的原始性，确保结果可重复
    # 目标：获得模型在"真实"未变换图像上的性能
    val_test_transform = transforms.Compose([
        
        # 1️⃣ 直接缩放到目标尺寸
        # 不使用随机裁剪，保持图像完整信息
        transforms.Resize((224, 224)),
        
        # 2️⃣ 转张量 + 归一化（与训练集保持一致）
        transforms.ToTensor(),
        
        # 3️⃣ 标准化（必须与训练时完全相同）
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ----------------------------
    # 第六步：创建真实数据集
    # ----------------------------

    # 使用 ImageFolder 加载真实图片数据
    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_test_transform)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=val_test_transform)  # 使用验证集作为测试集

    # 打印类别映射（非常重要！）
    print(f"\n✅ 类别标签映射: {train_dataset.class_to_idx}")
    # 输出示例: {'cats': 0, 'dogs': 1}

    # 打印数据量
    print(f"✅ 训练集: {len(train_dataset)} 张")
    print(f"✅ 验证集: {len(val_dataset)} 张")
    print(f"✅ 测试集: {len(test_dataset)} 张")

    # ----------------------------
    # 第七步：创建 DataLoader（批量加载器）
    # ----------------------------

    batch_size = 32  # 彩色图较大，batch_size 通常比 MNIST 小
    
    # Windows 系统需要设置 num_workers=0 避免多进程问题
    num_workers = 0 if os.name == 'nt' else 4
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"✅ 批次大小: {batch_size}")

    # ========================================================================
    # 第八步：定义CNN模型架构 — 深度学习的核心！
    # ========================================================================
    
    class CatDogCNN(nn.Module):
        """
        🏗️ 自定义CNN架构详解
        
        CNN的层次化特征提取原理：
        - 浅层：检测边缘、角点等低级特征
        - 中层：组合低级特征，形成纹理、形状等中级特征  
        - 深层：组合中级特征，形成物体部件等高级特征
        - 全连接：将高级特征映射到分类结果
        
        本模型设计思路：
        1. 逐层增加通道数(3→32→64→128)：提取更丰富的特征
        2. 逐层减小空间尺寸(224→112→56→28)：聚焦重要区域
        3. 最后用全连接层进行分类决策
        """
        
        def __init__(self):
            super(CatDogCNN, self).__init__()
            
            # ===== 卷积层设计 =====
            
            # 🔍 第一层卷积 (特征提取的起点)
            # Conv2d参数详解：
            # - in_channels=3: 输入RGB三通道
            # - out_channels=32: 输出32个特征图(32个不同的滤波器)
            # - kernel_size=3: 3×3卷积核，常用尺寸，能捕获局部特征
            # - padding=1: 边缘填充，保持输出尺寸不变
            # 作用：检测边缘、颜色变化等基础特征
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            
            # 🔍 第二层卷积 (特征组合)
            # 32→64通道：用更多滤波器捕获更复杂的特征组合
            # 作用：检测纹理、简单形状等中级特征
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            
            # 🔍 第三层卷积 (高级特征)
            # 64→128通道：提取更抽象的高级特征
            # 作用：检测物体部件(如猫耳朵、狗鼻子)等复杂特征
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            
            # ===== 池化层设计 =====
            
            # 🏊 最大池化层 (降维 + 不变性)
            # MaxPool2d(2, 2)：2×2窗口取最大值，步长为2
            # 作用：
            # 1. 降低计算量：特征图尺寸减半
            # 2. 增强不变性：对小幅位移不敏感
            # 3. 扩大感受野：后续卷积能"看到"更大区域
            self.pool = nn.MaxPool2d(2, 2)
            
            # ===== 全连接层设计 =====
            
            # 📐 尺寸计算过程：
            # 输入: 224×224×3
            # Conv1+Pool1: 224×224×32 → 112×112×32
            # Conv2+Pool2: 112×112×64 → 56×56×64  
            # Conv3+Pool3: 56×56×128 → 28×28×128
            # Flatten: 28×28×128 = 100,352个特征
            
            # 🧠 第一个全连接层 (特征整合)
            # 将空间特征图展平为一维向量，进行全局特征整合
            # 100,352 → 512：大幅降维，提取最重要的判别特征
            self.fc1 = nn.Linear(128 * 28 * 28, 512)
            
            # 🎯 输出层 (分类决策)
            # 512 → 2：映射到两个类别(猫、狗)的得分
            # 输出logits，经过softmax后得到概率分布
            self.fc2 = nn.Linear(512, 2)
            
            # ===== 正则化层 =====
            
            # 🛡️ Dropout层 (防止过拟合)
            # p=0.5：训练时随机将50%的神经元输出置零
            # 作用：防止模型过度依赖某些神经元，提高泛化能力
            # 注意：仅在训练时生效，推理时自动关闭
            self.dropout = nn.Dropout(0.5)
            
        def forward(self, x):
            """
            🚀 前向传播过程 (Forward Pass)
            
            这是CNN的"思考"过程：
            1. 特征提取：通过卷积层逐层提取特征
            2. 特征选择：通过池化层选择重要特征
            3. 特征整合：通过全连接层整合全局信息
            4. 分类决策：输出每个类别的置信度
            """
            
            # 🔄 第一个卷积块：基础特征提取
            # 输入: [batch_size, 3, 224, 224] (RGB图像)
            x = self.conv1(x)           # → [batch_size, 32, 224, 224] (32个特征图)
            x = torch.relu(x)           # → ReLU激活，引入非线性
            x = self.pool(x)            # → [batch_size, 32, 112, 112] (尺寸减半)
            
            # 🔄 第二个卷积块：中级特征提取
            x = self.conv2(x)           # → [batch_size, 64, 112, 112] (64个特征图)
            x = torch.relu(x)           # → ReLU激活
            x = self.pool(x)            # → [batch_size, 64, 56, 56] (尺寸减半)
            
            # 🔄 第三个卷积块：高级特征提取
            x = self.conv3(x)           # → [batch_size, 128, 56, 56] (128个特征图)
            x = torch.relu(x)           # → ReLU激活
            x = self.pool(x)            # → [batch_size, 128, 28, 28] (尺寸减半)
            
            # 🔄 特征展平：从2D特征图转为1D特征向量
            # view()函数重塑张量形状，-1表示自动计算该维度大小
            x = x.view(x.size(0), -1)  # → [batch_size, 128*28*28] = [batch_size, 100352]
            
            # 🔄 全连接层：全局特征整合与分类
            x = self.fc1(x)             # → [batch_size, 512] (特征压缩)
            x = torch.relu(x)           # → ReLU激活
            x = self.dropout(x)         # → Dropout正则化(仅训练时)
            x = self.fc2(x)             # → [batch_size, 2] (最终logits)
            
            # 📊 返回logits (未经softmax的原始得分)
            # 损失函数CrossEntropyLoss会自动处理softmax
            return x

    model = CatDogCNN().to(device)
    print("\n✅ 模型结构:")
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ 模型总参数量: {total_params:,} 个")
    
    # ========================================================================
    # 第九步：设置损失函数和优化器 — 深度学习的"学习机制"
    # ========================================================================
    
    # 🎯 损失函数 (Loss Function) — 衡量模型预测与真实标签的差距
    # CrossEntropyLoss详解：
    # 1. 适用于多分类问题(虽然我们只有2类，但仍然适用)
    # 2. 内部自动执行softmax + 负对数似然损失
    # 3. 数学原理：-log(P(正确类别))，预测越准确，损失越小
    # 4. 梯度特性：在错误预测时提供强梯度信号，加速学习
    criterion = nn.CrossEntropyLoss()
    
    # ⚡ 优化器 (Optimizer) — 根据梯度更新模型参数
    # Adam优化器详解：
    # 1. 自适应学习率：不同参数使用不同的学习率
    # 2. 动量机制：利用历史梯度信息，加速收敛
    # 3. 偏差修正：避免初期更新偏向零
    # 4. lr=0.001：学习率，控制参数更新的步长
    #    - 太大：可能跳过最优解，训练不稳定
    #    - 太小：收敛速度慢，可能陷入局部最优
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ========================================================================
    # 第十步：训练循环 — 深度学习的"学习过程"
    # ========================================================================
    
    print("\n🚀 开始训练...")
    
    num_epochs = 10  # 训练轮数：完整遍历训练集的次数
    train_losses, val_accuracies = [], []  # 记录训练过程，用于绘制学习曲线
    
    for epoch in range(num_epochs):
        print(f"\n📚 Epoch {epoch+1}/{num_epochs} 开始...")
        
        # ========== 训练阶段 (Training Phase) ==========
        # 🎓 模型进入训练模式
        # train()的作用：
        # 1. 启用Dropout：随机关闭部分神经元，防止过拟合
        # 2. 启用BatchNorm：使用当前批次统计量进行归一化
        # 3. 启用梯度计算：允许反向传播更新参数
        model.train()
        running_loss = 0.0
        
        # 🔄 遍历训练数据的每个批次
        for batch_idx, (images, labels) in enumerate(train_loader):
            # 📱 数据移动到GPU/CPU
            images, labels = images.to(device), labels.to(device)
            
            # ===== 前向传播 (Forward Pass) =====
            # 🚀 这是CNN的"思考"过程
            # 1. 输入图像通过卷积层提取特征
            # 2. 通过池化层降维和增强不变性  
            # 3. 通过全连接层进行分类决策
            # 4. 输出每个类别的置信度分数(logits)
            outputs = model(images)  # [batch_size, 2] 每个样本对应猫、狗的得分
            
            # 📊 计算损失 (Loss Computation)
            # 比较模型预测(outputs)与真实标签(labels)的差距
            # CrossEntropyLoss会自动：
            # 1. 对outputs应用softmax得到概率分布
            # 2. 计算负对数似然损失
            # 3. 对批次内所有样本求平均
            loss = criterion(outputs, labels)
            
            # ===== 反向传播 (Backward Pass) =====
            # 🔄 这是CNN的"学习"过程
            
            # 1️⃣ 清零梯度
            # 为什么需要？PyTorch默认累积梯度，不清零会导致梯度爆炸
            optimizer.zero_grad()
            
            # 2️⃣ 反向传播计算梯度
            # 🧮 自动微分：从损失函数开始，逐层计算每个参数的梯度
            # 链式法则：∂Loss/∂w = ∂Loss/∂output × ∂output/∂w
            # 这一步计算所有参数相对于损失的偏导数
            loss.backward()
            
            # 3️⃣ 参数更新
            # 🎯 根据梯度和学习率更新模型参数
            # Adam算法：w_new = w_old - lr × (梯度的自适应调整版本)
            optimizer.step()
            
            # 📈 累积损失用于监控训练进度
            running_loss += loss.item()  # .item()将tensor转为Python数值
        
        # 📊 计算平均训练损失
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # ========== 验证阶段 (Validation Phase) ==========
        # 🔍 模型进入评估模式
        # eval()的作用：
        # 1. 关闭Dropout：使用所有神经元，获得稳定预测
        # 2. 关闭BatchNorm训练：使用全局统计量
        # 3. 不影响梯度计算(但我们用no_grad()显式关闭)
        model.eval()
        correct, total = 0, 0
        
        # 🚫 关闭梯度计算，节省内存和计算
        # 验证时不需要梯度，因为不会更新参数
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # 🔮 模型预测(仅前向传播，无反向传播)
                outputs = model(images)  # [batch_size, 2]
                
                # 🎯 获取预测类别
                # torch.max(outputs, 1)返回(最大值, 最大值索引)
                # 我们只需要索引，即预测的类别
                _, predicted = torch.max(outputs, 1)
                
                # 📊 统计准确率
                total += labels.size(0)  # 累计样本数
                correct += (predicted == labels).sum().item()  # 累计正确预测数
        
        # 📈 计算验证准确率
        val_acc = 100 * correct / total
        val_accuracies.append(val_acc)
        
        # 📋 打印训练进度
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}%")
        
        # 💡 训练技巧说明：
        # - 训练损失下降 + 验证准确率上升 = 模型正在学习 ✅
        # - 训练损失下降 + 验证准确率不变/下降 = 过拟合 ⚠️
        # - 训练损失不变 = 学习率太小或模型容量不足 ⚠️

    # ========================================================================
    # 训练过程可视化 — 理解模型学习曲线
    # ========================================================================
    
    # 📊 绘制训练曲线 (Learning Curves)
    # 学习曲线的重要性：
    # 1. 诊断模型训练状态：是否收敛、是否过拟合
    # 2. 调参指导：学习率、模型容量、正则化强度
    # 3. 早停决策：避免过度训练
    plt.figure(figsize=(12, 4))
    
    # 📉 训练损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 💡 训练损失解读：
    # - 持续下降：模型正在学习 ✅
    # - 震荡下降：学习率可能偏高，但仍在学习
    # - 平稳不变：学习率太小或已收敛
    # - 上升：学习率太大或梯度爆炸 ⚠️
    
    # 📈 验证准确率曲线  
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy', color='orange', linewidth=2)
    plt.title('Validation Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 💡 验证准确率解读：
    # - 持续上升：模型泛化能力在提升 ✅
    # - 先升后降：过拟合，应该早停 ⚠️
    # - 震荡：数据不平衡或批次大小太小
    # - 平稳：模型容量不足或已达到数据上限
    
    plt.tight_layout()
    plt.show()
    
    # ========================================================================
    # 第十一步：最终测试 — 评估模型真实性能
    # ========================================================================
    
    print("\n🧪 开始最终测试...")
    
    # 🔒 测试集评估的重要原则：
    # 1. 只在训练完全结束后使用一次
    # 2. 测试集从未参与训练和验证，保证评估的公正性
    # 3. 测试准确率是模型在真实世界的预期性能
    
    model.eval()  # 确保模型处于评估模式
    correct, total = 0, 0
    
    # 🚫 关闭梯度计算，节省内存
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 🔮 模型推理
            outputs = model(images)  # [batch_size, 2] logits
            
            # 🎯 获取预测结果
            # torch.max()找到每个样本的最大logit对应的类别索引
            _, predicted = torch.max(outputs, 1)
            
            # 📊 统计正确预测
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 📈 计算最终测试准确率
    test_acc = 100 * correct / total
    print(f"\n🎉 最终测试准确率: {test_acc:.2f}%")
    
    # 💡 测试结果解读：
    # - 测试准确率 ≈ 验证准确率：模型泛化良好 ✅
    # - 测试准确率 < 验证准确率：轻微过拟合，但可接受
    # - 测试准确率 << 验证准确率：严重过拟合 ⚠️
    # - 测试准确率 > 验证准确率：验证集可能有问题

# ========================================================================
# 第十二步：单图预测函数 — 实际应用的接口
# ========================================================================

def predict_image(image_path, model, transform, device, class_names):
    """
    🔮 CNN单图预测函数 — 将训练好的模型应用到新图像
    
    这个函数展示了CNN在实际应用中的完整流程：
    1. 图像预处理：与训练时保持一致
    2. 模型推理：前向传播获得预测
    3. 结果解释：将logits转换为可理解的概率和类别
    
    参数说明：
    :param image_path: 待预测图像的文件路径
    :param model: 训练好的CNN模型
    :param transform: 图像预处理管道(必须与训练时一致!)
    :param device: 计算设备(CPU/GPU)
    :param class_names: 类别名称列表，如['cats', 'dogs']
    
    返回值：
    :return: (预测类别名, 置信度) 元组
    """
    
    # ===== 第一步：图像加载与预处理 =====
    
    # 📷 加载图像并确保格式正确
    # convert('RGB')的重要性：
    # 1. 统一格式：无论输入是灰度图、RGBA还是其他格式，都转为RGB
    # 2. 通道一致：确保输入是3通道，与模型期望一致
    # 3. 避免错误：防止因图像格式不匹配导致的运行时错误
    image = Image.open(image_path).convert('RGB')
    
    # 🔄 应用预处理变换
    # 关键原则：预处理必须与训练时完全一致！
    # 包括：尺寸调整、归一化、数据类型转换等
    input_tensor = transform(image)  # [3, 224, 224]
    
    # 📦 添加批次维度
    # unsqueeze(0)：[3, 224, 224] → [1, 3, 224, 224]
    # 为什么需要？CNN模型期望输入是批次格式，即使只有一张图
    input_tensor = input_tensor.unsqueeze(0)
    
    # 📱 移动到计算设备
    input_tensor = input_tensor.to(device)
    
    # ===== 第二步：模型推理 =====
    
    # 🔒 设置评估模式
    # 确保Dropout关闭、BatchNorm使用全局统计量
    model.eval()
    
    # 🚫 关闭梯度计算
    # 推理时不需要梯度，关闭可以：
    # 1. 节省内存：不存储中间梯度
    # 2. 加速计算：跳过梯度相关操作
    # 3. 避免意外：防止意外修改模型参数
    with torch.no_grad():
        # 🚀 前向传播
        output = model(input_tensor)  # [1, 2] 输出logits
        
        # ===== 第三步：结果解释 =====
        
        # 📊 Logits转概率
        # softmax函数：将任意实数转换为概率分布
        # 数学原理：P(class_i) = exp(logit_i) / Σ(exp(logit_j))
        # 特性：所有概率和为1，值域[0,1]
        probabilities = torch.softmax(output, dim=1)  # [1, 2]
        
        # 🎯 获取预测类别
        # argmax：找到概率最大的类别索引
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        # 📈 获取置信度
        # 置信度 = 预测类别的概率值
        # 高置信度(>0.9)：模型很确定
        # 中等置信度(0.6-0.9)：模型较确定
        # 低置信度(<0.6)：模型不确定，可能需要人工检查
        confidence = probabilities[0][predicted_class].item()
    
    # 🏷️ 返回可读的结果
    return class_names[predicted_class], confidence

# ========================================================================
# 使用示例 — 如何在实际项目中应用训练好的模型
# ========================================================================

# 💡 实际使用步骤：
# 1. 准备新图像路径
# 2. 确保类别名称顺序与训练时一致
# 3. 调用预测函数
# 4. 根据置信度判断结果可靠性

# 📝 使用示例代码：
# new_image_path = "./my_cat.jpg"  # 替换为你的图片路径
# class_names = ['cats', 'dogs']   # ⚠️ 顺序必须与dataset.class_to_idx一致！
# pred_class, conf = predict_image(new_image_path, model, val_test_transform, device, class_names)
# print(f"\n🔍 预测结果: {pred_class} (置信度: {conf:.2%})")
# 
# # 💡 置信度解读：
# # - 置信度 > 90%：高度可信 ✅
# # - 置信度 70-90%：较可信，可接受
# # - 置信度 50-70%：不确定，建议人工检查 ⚠️
# # - 置信度 < 50%：不可信，可能是模型未见过的类别 ❌