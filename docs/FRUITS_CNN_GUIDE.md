# 🍎 水果分类CNN项目指南

## 项目概述

本项目基于cats_and_dogs分类器的成功经验，扩展到100种水果的多分类任务。这是一个完整的深度学习项目，展示了从数据准备到模型部署的全流程。

## 🎯 项目特点

### 技术升级
- **多分类任务**: 从2类扩展到100类
- **深度网络**: 4层卷积块 + 全局平均池化
- **数据增强**: 强化的数据增强策略
- **类别平衡**: 处理数据不平衡问题
- **训练优化**: 学习率调度、早停、模型检查点

### 模型架构
```
输入(224×224×3) 
→ Conv Block 1 (64通道) → 112×112×64
→ Conv Block 2 (128通道) → 56×56×128
→ Conv Block 3 (256通道) → 28×28×256
→ Conv Block 4 (512通道) → 14×14×512
→ Global Average Pooling → 512
→ FC1(256) → Dropout → FC2(100) → 输出
```

## 📊 数据集信息

- **来源**: [ModelScope fruits100数据集](https://www.modelscope.cn/datasets/tany0699/fruits100) <mcreference link="https://www.modelscope.cn/datasets/tany0699/fruits100" index="0">0</mcreference>
- **类别**: 100种不同水果
- **格式**: JPG图片，按文件夹分类组织
- **结构**: `train/` 和 `val/` 目录，每个类别一个子文件夹

## 🚀 快速开始

### 1. 环境准备

确保已安装必要的依赖：
```bash
pip install -r requirements.txt
```

### 2. 数据准备

#### 方法一：手动下载
1. 访问 [ModelScope数据集页面](https://www.modelscope.cn/datasets/tany0699/fruits100)
2. 下载数据集到 `data/fruits100/` 目录
3. 确保目录结构如下：
```
data/fruits100/
├── train/
│   ├── apple/
│   ├── banana/
│   └── ... (其他水果类别)
└── val/
    ├── apple/
    ├── banana/
    └── ... (其他水果类别)
```

#### 方法二：使用下载脚本（开发中）
```bash
python scripts/download_datasets.py --dataset fruits100
```

### 3. 验证数据集

```bash
python scripts/validate_data.py --dataset fruits100
```

### 4. 开始训练

```bash
cd src/03_cnn
python fruits_classifier.py
```

## 🔧 核心功能详解

### 数据预处理
- **训练集增强**: 随机裁剪、翻转、旋转、颜色抖动
- **验证集预处理**: 标准化尺寸调整和归一化
- **类别权重**: 自动计算类别权重处理数据不平衡

### 模型特性
- **Batch Normalization**: 加速训练收敛
- **Dropout**: 防止过拟合
- **全局平均池化**: 减少参数量
- **权重初始化**: Kaiming初始化提高训练稳定性

### 训练策略
- **学习率调度**: ReduceLROnPlateau自适应调整
- **早停机制**: 防止过拟合
- **模型检查点**: 自动保存最佳模型
- **加权采样**: 平衡类别分布

## 📈 训练监控

训练过程中会显示：
- 实时训练进度和准确率
- 验证损失和准确率
- 学习率变化
- 最佳模型保存信息

训练完成后会生成：
- 训练历史图表
- 模型评估报告
- 预测结果可视化

## 🎯 使用示例

### 训练模型
```python
from src.cnn.fruits_classifier import *

# 模型会自动开始训练
# 训练完成后保存到 models/fruits_cnn_best.pth
```

### 预测单张图片
```python
# 加载训练好的模型
model = FruitsCNN(num_classes=100)
checkpoint = torch.load('models/fruits_cnn_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 预测新图片
result = predict_single_image(
    model=model,
    image_path='path/to/fruit_image.jpg',
    class_names=checkpoint['class_names'],
    device=device,
    transform=val_test_transform
)

print(f"预测类别: {result['predicted_class']}")
print(f"置信度: {result['confidence']:.2f}")
print("Top-5预测:")
for class_name, prob in result['top5_predictions']:
    print(f"  {class_name}: {prob:.3f}")
```

## 📊 性能指标

### 模型规模
- **参数量**: ~2.5M参数
- **模型大小**: ~10MB
- **推理速度**: ~50ms/张 (CPU)

### 预期性能
- **验证准确率**: 70-85% (取决于数据质量)
- **Top-5准确率**: 90%+ 
- **训练时间**: 2-4小时 (GPU) / 8-12小时 (CPU)

## 🔧 自定义配置

### 修改模型架构
在 `FruitsCNN` 类中调整：
- 卷积层通道数
- 网络深度
- 全连接层大小

### 调整训练参数
```python
# 学习率
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 批次大小
batch_size = 32

# 训练轮数
num_epochs = 30

# 数据增强强度
train_transform = transforms.Compose([...])
```

### 处理新数据集
1. 更新 `src/utils/data_config.py` 中的数据集配置
2. 修改 `num_classes` 参数
3. 调整模型最后一层输出维度

## 🚀 进阶优化

### 1. 迁移学习
使用预训练模型加速训练：
```python
import torchvision.models as models

# 使用预训练ResNet
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

### 2. 数据增强优化
- **MixUp**: 混合不同样本
- **CutMix**: 裁剪混合
- **AutoAugment**: 自动数据增强策略

### 3. 模型集成
组合多个模型提升性能：
```python
# 训练多个不同架构的模型
models = [model1, model2, model3]

# 集成预测
ensemble_pred = torch.mean(torch.stack([
    F.softmax(model(x), dim=1) for model in models
]), dim=0)
```

## 🐛 常见问题

### 1. 内存不足
- 减小 `batch_size`
- 使用 `pin_memory=False`
- 减少 `num_workers`

### 2. 训练过慢
- 使用GPU加速
- 增大 `batch_size`
- 减少数据增强操作

### 3. 过拟合
- 增强数据增强
- 提高Dropout比例
- 减少模型复杂度
- 使用早停机制

### 4. 欠拟合
- 增加模型复杂度
- 降低正则化强度
- 增加训练轮数
- 调整学习率

## 📚 扩展学习

### 相关论文
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### 推荐资源
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [CS231n课程](http://cs231n.stanford.edu/)
- [Deep Learning Book](https://www.deeplearningbook.org/)

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

### 改进方向
- 支持更多数据集
- 添加更多模型架构
- 优化训练策略
- 改进可视化功能
- 添加模型部署功能

---

**Happy Learning! 🎉**