# -*- coding: utf-8 -*-
"""
第七节实战：用 CNN 分类 MNIST 手写数字
目标：从零搭建、训练、评估、可视化一个真实 CNN 模型
"""

# ----------------------------
# 第一步：导入所需库
# ----------------------------

import torch                     # PyTorch 核心库，提供张量和自动微分
import torch.nn as nn            # 神经网络模块（如 Conv2d, Linear, ReLU）
import torch.optim as optim      # 优化器（如 Adam, SGD）
from torchvision import datasets, transforms  # torchvision：计算机视觉专用工具
from torch.utils.data import DataLoader       # 数据加载器
import matplotlib.pyplot as plt               # 绘图库，用于可视化

# ----------------------------
# 第二步：设置计算设备（CPU 或 GPU）
# ----------------------------

# 检查是否有可用的 GPU（CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")
# 说明：若有 NVIDIA GPU 且安装了 CUDA，会自动使用 GPU 加速训练；
# 否则回退到 CPU（速度慢但能运行）

# ----------------------------
# 第三步：数据预处理与加载
# ----------------------------

# 定义图像变换（Transforms）：将原始图像转为模型能处理的张量
transform = transforms.Compose([
    # ToTensor() 会做两件事：
    # 1. 将 PIL 图像（0~255 的整数）转为 PyTorch 张量；
    # 2. 自动将像素值除以 255，归一化到 [0, 1] 范围（有利于训练稳定）
    transforms.ToTensor(),
])

# 下载并加载 MNIST 训练集
# root: 数据存储路径（自动创建 ./data 文件夹）
# train=True: 加载训练集（60,000 张）
# download=True: 如果本地没有，自动从网上下载
# transform=transform: 对每张图应用上面定义的变换
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 加载 MNIST 测试集（10,000 张）
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 打印数据集信息
print(f"✅ 训练集: {len(train_dataset)} 张图, 测试集: {len(test_dataset)} 张图")

# ----------------------------
# 第四步：创建 DataLoader（数据“快递打包员”）
# ----------------------------

batch_size = 64  # 每次训练送入模型的图像数量（批量大小）

# DataLoader 的作用：
# - 按 batch_size 打包数据（避免一次性加载全部数据）
# - shuffle=True：每个 epoch 开始前打乱训练数据顺序（防止模型记住顺序）
# - num_workers=0（默认）：单线程加载（教学环境安全）；实际可用多线程加速
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # 测试集无需打乱

print(f"✅ 批次大小: {batch_size}")
print(f"✅ 每个 epoch 有 {len(train_loader)} 个训练批次")

# ----------------------------
# 第五步：定义 CNN 模型（重点！结合第六节原理）
# ----------------------------

class MNIST_CNN(nn.Module):
    """
    自定义 CNN 模型，继承自 nn.Module（PyTorch 所有模型的基类）
    """
    def __init__(self):
        super(MNIST_CNN, self).__init__()  # 调用父类初始化
        
        # 第一个卷积块：
        # 输入通道数 = 1（因为 MNIST 是灰度图）
        # 输出通道数 = 32（即使用 32 个不同的卷积核 → 输出 32 张特征图）
        # kernel_size=3：每个卷积核是 3×3 大小（现代 CNN 主流选择）
        # padding=1：在图像边缘补一圈 0，使得输出尺寸 = 输入尺寸（same padding）
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # 第一个池化层：2×2 最大池化，步长=2 → 输出尺寸减半
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积块：
        # 输入通道数 = 32（上一层输出的通道数）
        # 输出通道数 = 64（使用 64 个新卷积核）
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 展平层：将多维特征图拉成一维向量，以便输入全连接层
        # 例如：[64, 7, 7] → [64*7*7] = [3136]
        self.flatten = nn.Flatten()
        
        # 全连接层（分类头）：
        # 输入维度 = 64 * 7 * 7 = 3136（由前面卷积+池化决定）
        # 输出维度 = 128（隐藏层神经元数）
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        
        # 最终输出层：10 个神经元，对应 0~9 十个数字类别
        # 注意：这里不加 Softmax！因为 CrossEntropyLoss 内部已包含
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        前向传播函数：定义数据如何流经网络
        x 的形状: [batch_size, 1, 28, 28]
        """
        # 第一个卷积块：Conv → ReLU → Pool
        x = self.conv1(x)          # [B,1,28,28] → [B,32,28,28] （因 padding=1）
        x = torch.relu(x)          # ReLU 激活：负值变 0，保留正特征
        x = self.pool1(x)          # [B,32,28,28] → [B,32,14,14] （尺寸减半）
        
        # 第二个卷积块
        x = self.conv2(x)          # [B,32,14,14] → [B,64,14,14]
        x = torch.relu(x)
        x = self.pool2(x)          # [B,64,14,14] → [B,64,7,7]
        
        # 展平 + 全连接分类
        x = self.flatten(x)        # [B,64,7,7] → [B, 64*7*7] = [B, 3136]
        x = self.fc1(x)            # [B,3136] → [B,128]
        x = torch.relu(x)          # 再加一层非线性
        x = self.fc2(x)            # [B,128] → [B,10]（10 个类别的 logits）
        
        return x  # 注意：输出是 logits，不是概率！

# 创建模型实例，并移至指定设备（CPU/GPU）
model = MNIST_CNN().to(device)
print("\n✅ 模型结构:")
print(model)

# 打印模型总参数量（验证是否合理）
total_params = sum(p.numel() for p in model.parameters())
print(f"\n✅ 模型总参数量: {total_params:,} 个")

# ----------------------------
# 第六步：设置损失函数与优化器
# ----------------------------

# 损失函数：交叉熵（适用于多分类）
# 注意：PyTorch 的 CrossEntropyLoss 内部已包含 LogSoftmax，
# 所以模型 forward 中**不要加 Softmax**！
criterion = nn.CrossEntropyLoss()

# 优化器：Adam（自适应学习率，通常比 SGD 更快收敛）
# model.parameters()：自动收集所有可学习参数（包括卷积核的 w 和 b！）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 学习率 = 0.001

# ----------------------------
# 第七步：训练循环（核心！）
# ----------------------------

print("\n🚀 开始训练...")

num_epochs = 5  # 训练 5 个完整周期（遍历全部训练数据 5 次）
train_losses = []  # 记录损失，用于后续绘图

for epoch in range(num_epochs):
    model.train()  # 设置为训练模式（影响 Dropout/BatchNorm，本模型无影响但好习惯）
    running_loss = 0.0  # 累计当前 epoch 的损失
    
    # enumerate: 同时获取批次索引和数据
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 将数据移至设备（GPU/CPU）
        images = images.to(device)   # 形状: [64, 1, 28, 28]
        labels = labels.to(device)   # 形状: [64]（整数标签 0~9）
        
        # === 前向传播 ===
        outputs = model(images)      # 输出形状: [64, 10]（logits）
        loss = criterion(outputs, labels)  # 计算损失
        
        # === 反向传播 ===
        optimizer.zero_grad()        # 清空上一步的梯度（非常重要！）
        loss.backward()              # 自动计算所有参数的梯度（包括卷积核！）
        optimizer.step()             # 用梯度更新所有参数（w 和 b）
        
        # 累计损失
        running_loss += loss.item()  # .item() 将单元素张量转为 Python 数
        
        # 每 100 个 batch 打印一次平均损失
        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / 100
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Step [{batch_idx+1}/{len(train_loader)}], "
                  f"Loss: {avg_loss:.4f}")
            train_losses.append(avg_loss)
            running_loss = 0.0  # 重置累计

# ----------------------------
# 第八步：测试准确率（评估模型泛化能力）
# ----------------------------

model.eval()  # 设置为评估模式（关闭 Dropout 等）
correct = 0   # 正确预测数
total = 0     # 总样本数

# torch.no_grad()：关闭梯度计算（节省内存，加速）
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)                # [B,10] logits
        _, predicted = torch.max(outputs, 1)   # 取每行最大值的索引（即预测类别）
        
        total += labels.size(0)                # 累计样本数
        correct += (predicted == labels).sum().item()  # 累计正确数

accuracy = 100 * correct / total
print(f"\n✅ 测试准确率: {accuracy:.2f}%")

# ----------------------------
# 第九步：可视化（验证“卷积核是学出来的”）
# ----------------------------

# 获取第一个卷积层的权重（即 32 个卷积核）
# .data：获取张量数据（不带梯度）
# .cpu()：移回 CPU（matplotlib 不能直接处理 GPU 张量）
conv1_weights = model.conv1.weight.data.cpu()  # 形状: [32, 1, 3, 3]

# 可视化前 16 个卷积核
plt.figure(figsize=(10, 6))
for i in range(16):
    plt.subplot(4, 4, i+1)
    # conv1_weights[i, 0] 是第 i 个核的 3×3 权重（单通道）
    plt.imshow(conv1_weights[i, 0], cmap='gray', vmin=-1, vmax=1)
    plt.title(f"Kernel {i+1}")
    plt.axis('off')
plt.suptitle('第一层卷积核（训练后，32个中的前16个）')
plt.tight_layout()
plt.show()

# 可视化某张测试图的特征图
sample_image, true_label = test_dataset[0]  # 取第一张测试图
print(f"\n🔍 可视化第 1 张测试图（真实标签: {true_label}）")

# 增加 batch 维度：[1,28,28] → [1,1,28,28]
sample_image = sample_image.unsqueeze(0).to(device)

# 获取第一层卷积后的输出（需临时前向，但不更新参数）
with torch.no_grad():
    conv1_output = torch.relu(model.conv1(sample_image))  # [1,32,28,28]

# 可视化前 16 个特征图
plt.figure(figsize=(10, 6))
for i in range(16):
    plt.subplot(4, 4, i+1)
    # conv1_output[0, i] 是第 i 个特征图（28×28）
    plt.imshow(conv1_output[0, i].cpu(), cmap='viridis')
    plt.title(f"Feature Map {i+1}")
    plt.axis('off')
plt.suptitle('第一层卷积后的特征图（对第一张测试图）')
plt.tight_layout()
plt.show()

print("\n🎉 实战完成！你已成功训练一个 CNN 并验证其工作原理。")