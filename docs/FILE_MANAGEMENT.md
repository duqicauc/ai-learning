# 文件管理指南

## 目录结构说明

本项目采用标准化的AI学习项目目录结构，便于代码组织和管理。

### 核心目录

#### `src/` - 源代码目录
- **`01_basics/`** - 基础概念和入门代码
  - 张量操作、线性回归、分类等基础示例
- **`02_neural_networks/`** - 神经网络基础
  - 基础神经网络实现和训练
- **`03_cnn/`** - 卷积神经网络
  - CNN架构、图像分类、迁移学习等
- **`04_rnn/`** - 循环神经网络
  - RNN、LSTM、GRU等序列模型
- **`05_transformers/`** - Transformer架构
  - 注意力机制、Transformer实现
- **`06_advanced/`** - 高级主题
  - GPU训练、分布式训练、模型优化等
- **`utils/`** - 工具函数
  - 通用工具和辅助函数

#### `data/` - 数据目录
- 训练数据集和测试数据集
- 数据预处理脚本
- 数据集说明文档

#### `notebooks/` - Jupyter笔记本
- 实验性代码和数据分析
- 教学演示和可视化

#### `models/` - 模型文件
- 训练好的模型权重
- 模型配置文件
- 模型评估结果

#### `experiments/` - 实验记录
- 实验配置和结果
- 超参数调优记录
- 性能对比分析

#### `docs/` - 文档目录
- **`notes/`** - 学习笔记
- API文档和使用说明
- 项目文档

## 文件命名规范

### Python文件命名
- 使用小写字母和下划线：`basic_cnn.py`
- 避免使用驼峰命名：~~`BasicCNN.py`~~
- 文件名应该描述功能：`cats_dogs_classifier.py`

### 目录命名
- 使用小写字母和下划线
- 数字前缀表示学习顺序：`01_basics/`
- 简洁明了：`cnn/` 而不是 `convolutional_neural_networks/`

### 模型文件命名
```
models/
├── cnn/
│   ├── cats_dogs_v1.pth
│   ├── cats_dogs_v2_improved.pth
│   └── resnet50_transfer.pth
└── rnn/
    ├── text_classifier_lstm.pth
    └── sentiment_analysis_gru.pth
```

## 代码组织原则

### 1. 单一职责
每个文件应该专注于一个特定的功能或概念：
- `iris_classification.py` - 鸢尾花分类
- `linear_regression.py` - 线性回归
- `tensor_operations.py` - 张量操作

### 2. 渐进式学习
目录按学习难度递增：
```
01_basics → 02_neural_networks → 03_cnn → 04_rnn → 05_transformers → 06_advanced
```

### 3. 代码复用
- 通用功能放在 `utils/` 目录
- 避免代码重复
- 使用配置文件管理超参数

## 数据管理

### 数据存储
- 原始数据：`data/raw/`
- 处理后数据：`data/processed/`
- 数据集分割：`data/splits/`

### 数据版本控制
- 大文件使用Git LFS
- 数据集变更记录在 `data/CHANGELOG.md`
- 提供数据集下载脚本

## 实验管理

### 实验记录
```
experiments/
├── 2024-01-15_cnn_baseline/
│   ├── config.yaml
│   ├── results.json
│   ├── logs/
│   └── plots/
└── 2024-01-16_cnn_improved/
    ├── config.yaml
    ├── results.json
    ├── logs/
    └── plots/
```

### 配置管理
- 使用YAML文件存储配置
- 版本控制所有配置文件
- 记录实验环境信息

## 最佳实践

### 1. 文档化
- 每个模块添加docstring
- 重要函数添加类型注解
- 维护README文件

### 2. 测试
- 为核心功能编写单元测试
- 使用pytest框架
- 保持测试覆盖率

### 3. 版本控制
- 频繁提交，小步迭代
- 使用有意义的提交信息
- 创建分支进行实验

### 4. 环境管理
- 使用虚拟环境
- 固定依赖版本
- 提供环境配置文件

## 迁移指南

如果需要重新组织现有代码：

1. **备份现有代码**
2. **创建新目录结构**
3. **按功能分类移动文件**
4. **更新导入路径**
5. **测试代码运行**
6. **更新文档**

## 工具推荐

- **代码格式化**: Black, isort
- **代码检查**: flake8, mypy
- **文档生成**: Sphinx
- **实验跟踪**: MLflow, Weights & Biases
- **版本控制**: Git + Git LFS