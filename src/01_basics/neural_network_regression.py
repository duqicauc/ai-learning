"""
神经网络回归预测实现
使用PyTorch构建多层感知机(MLP)来解决与linear_regression.py相同的回归问题

关键结论：对于简单的线性数据规律，适合用线性回归模型，神经网络适合处理复杂的数据分类或预测任务。

作者: AI Learning Project and 阿杜
功能: 
- 使用相同的数据集进行神经网络回归
- 对比线性回归和神经网络的效果
- 可视化训练过程和预测结果
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

# ============================
# 第一步：数据准备
# ============================

print("🔍 准备数据...")

# 使用与linear_regression.py相同的数据
x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(-1, 1)
y_data = np.array([3.1, 5.0, 7.2, 9.1, 11.0, 13.1, 15.0, 16.8, 19.2, 21.0])

print(f"数据形状: X={x_data.shape}, y={y_data.shape}")
print(f"X范围: [{x_data.min():.1f}, {x_data.max():.1f}]")
print(f"y范围: [{y_data.min():.1f}, {y_data.max():.1f}]")

# 数据标准化（神经网络训练的最佳实践）
scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_scaled = scaler_x.fit_transform(x_data)
y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()

print(f"标准化后 - X范围: [{x_scaled.min():.2f}, {x_scaled.max():.2f}]")
print(f"标准化后 - y范围: [{y_scaled.min():.2f}, {y_scaled.max():.2f}]")

# 转换为PyTorch张量
X_tensor = torch.FloatTensor(x_scaled)
y_tensor = torch.FloatTensor(y_scaled)

# ============================
# 第二步：定义神经网络模型
# ============================

class RegressionMLP(nn.Module):
    """
    多层感知机回归模型
    
    架构:
    - 输入层: 1个特征
    - 隐藏层1: 64个神经元 + ReLU激活
    - 隐藏层2: 32个神经元 + ReLU激活  
    - 隐藏层3: 16个神经元 + ReLU激活
    - 输出层: 1个神经元（回归输出）
    """
    
    def __init__(self, input_size=1, hidden_sizes=[64, 32, 16], output_size=1):
        super(RegressionMLP, self).__init__()
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        # 添加隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # 添加Dropout防止过拟合
            prev_size = hidden_size
        
        # 添加输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        # 组合所有层
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """使用Xavier初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)

# ============================
# 第三步：创建模型和训练配置
# ============================

print("\n🏗️ 创建神经网络模型...")

# 创建模型
model = RegressionMLP(input_size=1, hidden_sizes=[64, 32, 16], output_size=1)

# 打印模型结构
print("模型架构:")
print(model)

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 模型参数统计:")
print(f"总参数数: {total_params:,}")
print(f"可训练参数数: {trainable_params:,}")

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=100
    )

# ============================
# 第四步：训练神经网络
# ============================

print("\n🚀 开始训练神经网络...")

# 训练配置
num_epochs = 1000
train_losses = []
learning_rates = []

# 训练循环
model.train()
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_tensor)
    loss = criterion(outputs.squeeze(), y_tensor)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 梯度裁剪（防止梯度爆炸）
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 更新参数
    optimizer.step()
    
    # 学习率调度
    scheduler.step(loss)
    
    # 记录训练信息
    train_losses.append(loss.item())
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    # 打印训练进度
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {loss.item():.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

print(f"\n✅ 训练完成！最终损失: {train_losses[-1]:.6f}")

# ============================
# 第五步：模型评估和预测
# ============================

print("\n📊 评估模型性能...")

# 切换到评估模式
model.eval()

# 在原始数据上进行预测
with torch.no_grad():
    # 标准化输入
    x_test_scaled = scaler_x.transform(x_data)
    X_test_tensor = torch.FloatTensor(x_test_scaled)
    
    # 神经网络预测（标准化输出）
    y_pred_scaled = model(X_test_tensor).squeeze().numpy()
    
    # 反标准化得到原始尺度的预测
    y_pred_nn = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

# 计算评估指标
mse = mean_squared_error(y_data, y_pred_nn)
rmse = np.sqrt(mse)
r2 = r2_score(y_data, y_pred_nn)

print(f"神经网络性能指标:")
print(f"  均方误差 (MSE): {mse:.4f}")
print(f"  均方根误差 (RMSE): {rmse:.4f}")
print(f"  决定系数 (R²): {r2:.4f}")

# ============================
# 第六步：与线性回归对比
# ============================

print("\n🔍 与线性回归对比...")

# 使用sklearn的线性回归作为基准
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(x_data, y_data)
y_pred_lr = lr_model.predict(x_data)

# 线性回归性能
mse_lr = mean_squared_error(y_data, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
r2_lr = r2_score(y_data, y_pred_lr)

print(f"线性回归性能指标:")
print(f"  均方误差 (MSE): {mse_lr:.4f}")
print(f"  均方根误差 (RMSE): {rmse_lr:.4f}")
print(f"  决定系数 (R²): {r2_lr:.4f}")

print(f"\n📈 性能对比:")
print(f"  MSE改进: {((mse_lr - mse) / mse_lr * 100):+.2f}%")
print(f"  RMSE改进: {((rmse_lr - rmse) / rmse_lr * 100):+.2f}%")
print(f"  R²改进: {((r2 - r2_lr) / r2_lr * 100):+.2f}%")

# ============================
# 第七步：结果可视化
# ============================

print("\n🎨 生成可视化结果...")

# 创建更密集的测试点用于绘制平滑曲线
x_plot = np.linspace(0.5, 10.5, 100).reshape(-1, 1)
x_plot_scaled = scaler_x.transform(x_plot)
X_plot_tensor = torch.FloatTensor(x_plot_scaled)

with torch.no_grad():
    y_plot_scaled = model(X_plot_tensor).squeeze().numpy()
    y_plot_nn = scaler_y.inverse_transform(y_plot_scaled.reshape(-1, 1)).flatten()

y_plot_lr = lr_model.predict(x_plot)

# 创建综合可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('神经网络回归 vs 线性回归对比分析', fontsize=16, fontweight='bold')

# 1. 预测结果对比
ax1 = axes[0, 0]
ax1.scatter(x_data.flatten(), y_data, color='black', s=80, alpha=0.8, label='真实数据', zorder=5)
ax1.plot(x_plot.flatten(), y_plot_lr, color='red', linewidth=2, label='线性回归', alpha=0.8)
ax1.plot(x_plot.flatten(), y_plot_nn, color='blue', linewidth=2, label='神经网络', alpha=0.8)
ax1.set_xlabel('x (特征)')
ax1.set_ylabel('y (目标值)')
ax1.set_title('预测结果对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 训练损失曲线
ax2 = axes[0, 1]
ax2.plot(train_losses, color='blue', linewidth=1.5)
ax2.set_xlabel('训练轮次')
ax2.set_ylabel('损失值 (MSE)')
ax2.set_title('神经网络训练损失曲线')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# 3. 残差分析
ax3 = axes[1, 0]
residuals_lr = y_data - y_pred_lr
residuals_nn = y_data - y_pred_nn
ax3.scatter(y_pred_lr, residuals_lr, color='red', alpha=0.7, label='线性回归')
ax3.scatter(y_pred_nn, residuals_nn, color='blue', alpha=0.7, label='神经网络')
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax3.set_xlabel('预测值')
ax3.set_ylabel('残差 (真实值 - 预测值)')
ax3.set_title('残差分析')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 性能指标对比
ax4 = axes[1, 1]
metrics = ['MSE', 'RMSE', 'R²']
lr_values = [mse_lr, rmse_lr, r2_lr]
nn_values = [mse, rmse, r2]

x_pos = np.arange(len(metrics))
width = 0.35

bars1 = ax4.bar(x_pos - width/2, lr_values, width, label='线性回归', color='red', alpha=0.7)
bars2 = ax4.bar(x_pos + width/2, nn_values, width, label='神经网络', color='blue', alpha=0.7)

ax4.set_xlabel('评估指标')
ax4.set_ylabel('指标值')
ax4.set_title('性能指标对比')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(metrics)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 添加数值标签
for bar in bars1:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

for bar in bars2:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ============================
# 第八步：总结和建议
# ============================

print("\n" + "="*60)
print("📋 实验总结")
print("="*60)

print(f"""
🎯 实验目标: 使用神经网络解决线性回归问题

📊 数据集信息:
  - 样本数量: {len(x_data)}
  - 特征维度: 1
  - 数据范围: x∈[{x_data.min():.1f}, {x_data.max():.1f}], y∈[{y_data.min():.1f}, {y_data.max():.1f}]

🏗️ 神经网络架构:
  - 输入层: 1个神经元
  - 隐藏层: 64 → 32 → 16 神经元
  - 输出层: 1个神经元
  - 激活函数: ReLU
  - 正则化: Dropout(0.1)
  - 总参数: {total_params:,}

📈 性能对比:
  线性回归 - MSE: {mse_lr:.4f}, RMSE: {rmse_lr:.4f}, R²: {r2_lr:.4f}
  神经网络 - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}
  
💡 关键发现:
  1. 对于简单的线性关系，神经网络可以达到与线性回归相近的性能
  2. 神经网络具有更强的非线性拟合能力，适合复杂数据
  3. 需要更多的训练时间和计算资源
  4. 通过适当的正则化可以防止过拟合

🚀 改进建议:
  1. 增加数据量以充分发挥神经网络优势
  2. 尝试不同的网络架构和超参数
  3. 使用更复杂的非线性数据进行测试
  4. 考虑使用集成方法结合多个模型
""")

print("="*60)
print("✅ 实验完成！")