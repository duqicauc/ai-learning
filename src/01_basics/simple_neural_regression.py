"""
简化神经网络回归实现
使用纯NumPy实现多层感知机进行回归预测，避免PyTorch环境依赖

功能:
- 纯NumPy实现的神经网络
- 与linear_regression.py使用相同数据
- 可视化对比结果
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)

class SimpleNeuralNetwork:
    """简单的神经网络实现"""
    
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
        # 存储训练历史
        self.losses = []
    
    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU导数"""
        return (x > 0).astype(float)
    
    def forward(self, X):
        """前向传播"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2
    
    def backward(self, X, y, output, learning_rate=0.01):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层梯度
        dz2 = output - y.reshape(-1, 1)
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # 隐藏层梯度
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # 更新参数
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs=1000, learning_rate=0.01):
        """训练神经网络"""
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            
            # 计算损失
            loss = np.mean((output.flatten() - y) ** 2)
            self.losses.append(loss)
            
            # 反向传播
            self.backward(X, y, output, learning_rate)
            
            # 打印进度
            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """预测"""
        return self.forward(X).flatten()

def main():
    """主函数"""
    print("🚀 开始神经网络回归对比实验\n")
    
    # ============================
    # 数据准备
    # ============================
    
    print("📊 准备数据...")
    
    # 使用与linear_regression.py相同的数据
    x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(-1, 1)
    y_data = np.array([3.1, 5.0, 7.2, 9.1, 11.0, 13.1, 15.0, 16.8, 19.2, 21.0])
    
    print(f"数据形状: X={x_data.shape}, y={y_data.shape}")
    print(f"数据范围: X∈[{x_data.min():.1f}, {x_data.max():.1f}], y∈[{y_data.min():.1f}, {y_data.max():.1f}]")
    
    # 数据标准化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    x_scaled = scaler_x.fit_transform(x_data)
    y_scaled = scaler_y.fit_transform(y_data.reshape(-1, 1)).flatten()
    
    # ============================
    # 线性回归基准
    # ============================
    
    print("\n🔵 训练线性回归模型...")
    
    lr_model = LinearRegression()
    lr_model.fit(x_data, y_data)
    y_pred_lr = lr_model.predict(x_data)
    
    # 线性回归性能
    mse_lr = mean_squared_error(y_data, y_pred_lr)
    r2_lr = r2_score(y_data, y_pred_lr)
    
    print(f"线性回归结果: y = {lr_model.coef_[0]:.3f}x + {lr_model.intercept_:.3f}")
    print(f"线性回归性能: MSE={mse_lr:.4f}, R²={r2_lr:.4f}")
    
    # ============================
    # 神经网络训练
    # ============================
    
    print("\n🔴 训练神经网络模型...")
    
    # 创建神经网络
    nn_model = SimpleNeuralNetwork(input_size=1, hidden_size=32, output_size=1)
    
    # 训练
    nn_model.train(x_scaled, y_scaled, epochs=1000, learning_rate=0.1)
    
    # 预测
    y_pred_scaled = nn_model.predict(x_scaled)
    y_pred_nn = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 神经网络性能
    mse_nn = mean_squared_error(y_data, y_pred_nn)
    r2_nn = r2_score(y_data, y_pred_nn)
    
    print(f"神经网络性能: MSE={mse_nn:.4f}, R²={r2_nn:.4f}")
    
    # ============================
    # 性能对比
    # ============================
    
    print("\n" + "="*50)
    print("📈 性能对比结果")
    print("="*50)
    
    print(f"{'指标':<10} {'线性回归':<12} {'神经网络':<12} {'改进':<10}")
    print("-" * 50)
    
    mse_improvement = (mse_lr - mse_nn) / mse_lr * 100
    r2_improvement = (r2_nn - r2_lr) / r2_lr * 100
    
    print(f"{'MSE':<10} {mse_lr:<12.4f} {mse_nn:<12.4f} {mse_improvement:>+7.1f}%")
    print(f"{'R²':<10} {r2_lr:<12.4f} {r2_nn:<12.4f} {r2_improvement:>+7.1f}%")
    
    # ============================
    # 可视化结果
    # ============================
    
    print("\n🎨 生成可视化结果...")
    
    # 创建密集的测试点
    x_plot = np.linspace(0.5, 10.5, 100).reshape(-1, 1)
    
    # 线性回归预测
    y_plot_lr = lr_model.predict(x_plot)
    
    # 神经网络预测
    x_plot_scaled = scaler_x.transform(x_plot)
    y_plot_scaled = nn_model.predict(x_plot_scaled)
    y_plot_nn = scaler_y.inverse_transform(y_plot_scaled.reshape(-1, 1)).flatten()
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('神经网络回归 vs 线性回归对比', fontsize=16, fontweight='bold')
    
    # 1. 预测结果对比
    ax1 = axes[0, 0]
    ax1.scatter(x_data.flatten(), y_data, color='black', s=80, alpha=0.8, label='真实数据', zorder=5)
    ax1.plot(x_plot.flatten(), y_plot_lr, color='blue', linewidth=2, label='线性回归', alpha=0.8)
    ax1.plot(x_plot.flatten(), y_plot_nn, color='red', linewidth=2, label='神经网络', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('预测结果对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 训练损失曲线
    ax2 = axes[0, 1]
    ax2.plot(nn_model.losses, color='red', linewidth=1.5)
    ax2.set_xlabel('训练轮次')
    ax2.set_ylabel('损失值 (MSE)')
    ax2.set_title('神经网络训练过程')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    # 3. 残差分析
    ax3 = axes[1, 0]
    residuals_lr = y_data - y_pred_lr
    residuals_nn = y_data - y_pred_nn
    ax3.scatter(y_pred_lr, residuals_lr, color='blue', alpha=0.7, label='线性回归', s=60)
    ax3.scatter(y_pred_nn, residuals_nn, color='red', alpha=0.7, label='神经网络', s=60)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('预测值')
    ax3.set_ylabel('残差')
    ax3.set_title('残差分析')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 性能指标对比
    ax4 = axes[1, 1]
    metrics = ['MSE', 'R²']
    lr_values = [mse_lr, r2_lr]
    nn_values = [mse_nn, r2_nn]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax4.bar(x_pos - width/2, lr_values, width, label='线性回归', color='blue', alpha=0.7)
    bars2 = ax4.bar(x_pos + width/2, nn_values, width, label='神经网络', color='red', alpha=0.7)
    
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
    # 总结
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
  - 隐藏层: 32个神经元 (ReLU激活)
  - 输出层: 1个神经元
  - 实现方式: 纯NumPy

📈 性能对比:
  线性回归 - MSE: {mse_lr:.4f}, R²: {r2_lr:.4f}
  神经网络 - MSE: {mse_nn:.4f}, R²: {r2_nn:.4f}
  
💡 关键发现:
  1. 对于简单线性关系，两种方法性能相近
  2. 神经网络具有学习非线性关系的潜力
  3. 数据量较小时，线性回归可能更稳定
  4. 神经网络需要更多的训练时间

🚀 改进建议:
  1. 尝试更复杂的非线性数据
  2. 增加数据量以发挥神经网络优势
  3. 调整网络架构和超参数
  4. 使用正则化技术防止过拟合
""")
    
    print("="*60)
    print("✅ 实验完成！")

if __name__ == "__main__":
    main()