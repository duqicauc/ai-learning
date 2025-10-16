# 神经网络回归预测实现

## 📋 项目概述

本项目实现了使用神经网络进行回归预测，并与传统线性回归方法进行对比分析。基于 `linear_regression.py` 的相同数据集，展示了深度学习在回归任务中的应用。

## 📁 文件结构

```
src/01_basics/
├── linear_regression.py           # 原始线性回归实现
├── neural_network_regression.py   # 神经网络回归实现 (主要文件)
├── regression_comparison.py       # 对比分析脚本
└── README_neural_regression.md    # 使用说明 (本文件)
```

## 🚀 快速开始

### 1. 运行神经网络回归

```bash
cd src/01_basics
python neural_network_regression.py
```

**输出内容:**
- 详细的训练过程
- 性能指标对比
- 可视化图表 (4个子图)
- 完整的实验总结

### 2. 运行对比分析

```bash
python regression_comparison.py
```

**输出内容:**
- 简洁的性能对比表格
- 并排可视化结果
- 方法选择建议

## 🏗️ 技术架构

### 神经网络模型 (RegressionMLP)

```
输入层 (1) → 隐藏层1 (64) → 隐藏层2 (32) → 隐藏层3 (16) → 输出层 (1)
              ↓ReLU        ↓ReLU         ↓ReLU
              ↓Dropout     ↓Dropout      ↓Dropout
```

**关键特性:**
- **激活函数**: ReLU
- **正则化**: Dropout (0.1)
- **权重初始化**: Xavier初始化
- **优化器**: Adam
- **学习率调度**: ReduceLROnPlateau

### 数据处理流程

1. **数据标准化**: 使用 StandardScaler
2. **张量转换**: NumPy → PyTorch Tensor
3. **训练**: 1000轮迭代
4. **反标准化**: 恢复原始尺度预测

## 📊 性能对比

| 指标 | 线性回归 | 神经网络 | 说明 |
|------|----------|----------|------|
| **MSE** | ~0.05 | ~0.03 | 均方误差，越小越好 |
| **R²** | ~0.99 | ~0.99 | 决定系数，越接近1越好 |
| **训练时间** | <0.01s | ~2-5s | 神经网络需要更多时间 |
| **参数数量** | 2个 | 2,000+个 | 神经网络参数更多 |

## 🎯 使用场景

### 选择线性回归的情况:
- ✅ 数据呈现明显线性关系
- ✅ 样本数量较少 (<100)
- ✅ 需要快速训练和预测
- ✅ 要求模型可解释性

### 选择神经网络的情况:
- ✅ 数据关系复杂或非线性
- ✅ 样本数量较大 (>1000)
- ✅ 有充足的计算资源
- ✅ 追求更高的预测精度

## 🔧 自定义配置

### 修改网络架构

```python
# 在 neural_network_regression.py 中修改
model = RegressionMLP(
    input_size=1, 
    hidden_sizes=[128, 64, 32],  # 自定义隐藏层
    output_size=1
)
```

### 调整训练参数

```python
# 训练轮次
num_epochs = 2000  # 增加训练轮次

# 学习率
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 降低学习率

# 批次大小 (如果数据量大)
batch_size = 32
```

### 添加更多数据

```python
# 扩展数据集
x_data = np.array([...])  # 添加更多x值
y_data = np.array([...])  # 添加对应的y值
```

## 📈 可视化说明

### 主要图表 (neural_network_regression.py)

1. **预测结果对比**: 真实数据点 + 两种方法的拟合曲线
2. **训练损失曲线**: 神经网络训练过程中损失的变化
3. **残差分析**: 预测误差的分布情况
4. **性能指标对比**: MSE、RMSE、R²的柱状图对比

### 简化图表 (regression_comparison.py)

1. **预测对比**: 并排显示两种方法的预测结果
2. **残差对比**: 两种方法的预测误差对比
3. **训练过程**: 神经网络的损失下降曲线

## 🛠️ 依赖环境

```bash
pip install torch numpy matplotlib scikit-learn
```

**版本要求:**
- Python >= 3.7
- PyTorch >= 1.8
- NumPy >= 1.19
- Matplotlib >= 3.3
- Scikit-learn >= 0.24

## 🔍 常见问题

### Q: 为什么神经网络效果没有明显提升？
A: 因为数据集较小且呈线性关系，神经网络的优势无法充分体现。尝试使用更复杂的非线性数据。

### Q: 如何提高神经网络性能？
A: 
- 增加训练数据量
- 调整网络架构 (更多层或神经元)
- 尝试不同的激活函数
- 调整学习率和优化器

### Q: 训练时间太长怎么办？
A: 
- 减少训练轮次 (num_epochs)
- 简化网络架构
- 使用GPU加速 (如果可用)

### Q: 如何处理过拟合？
A: 
- 增加Dropout比例
- 添加L2正则化
- 减少网络复杂度
- 使用早停机制

## 📚 扩展学习

### 进阶实验建议:

1. **非线性数据**: 尝试二次函数、三角函数等
2. **多特征回归**: 扩展到多维输入
3. **时间序列**: 使用RNN/LSTM进行序列预测
4. **集成方法**: 结合多个模型的预测结果

### 相关文件:
- `tensor_operations.py`: PyTorch基础操作
- `../04_rnn/basic_rnn.py`: 时间序列回归示例
- `../02_neural_networks/`: 更多神经网络示例

---

**作者**: AI Learning Project  
**更新时间**: 2024年  
**许可证**: MIT License