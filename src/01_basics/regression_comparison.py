"""
回归方法对比分析
快速比较线性回归和神经网络在相同数据集上的表现

功能:
- 并排运行两种方法
- 生成性能对比报告
- 可视化结果差异
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def prepare_data():
    """准备数据"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(-1, 1)
    y = np.array([3.1, 5.0, 7.2, 9.1, 11.0, 13.1, 15.0, 16.8, 19.2, 21.0])
    return x, y

def linear_regression_method(x, y):
    """线性回归方法"""
    print("🔵 运行线性回归...")
    start_time = time.time()
    
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    
    training_time = time.time() - start_time
    
    # 计算性能指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return {
        'model': model,
        'predictions': y_pred,
        'mse': mse,
        'r2': r2,
        'training_time': training_time,
        'params': f"w={model.coef_[0]:.3f}, b={model.intercept_:.3f}"
    }

class SimpleNN(nn.Module):
    """简化的神经网络"""
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def neural_network_method(x, y):
    """神经网络方法"""
    print("🔴 运行神经网络...")
    start_time = time.time()
    
    # 数据标准化
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    x_scaled = scaler_x.fit_transform(x)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    
    # 转换为张量
    X_tensor = torch.FloatTensor(x_scaled)
    y_tensor = torch.FloatTensor(y_scaled)
    
    # 创建模型
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 训练
    model.train()
    losses = []
    for epoch in range(500):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs.squeeze(), y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    # 预测
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).squeeze().numpy()
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    training_time = time.time() - start_time
    
    # 计算性能指标
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    
    return {
        'model': model,
        'predictions': y_pred,
        'mse': mse,
        'r2': r2,
        'training_time': training_time,
        'losses': losses,
        'params': f"{total_params} 个参数",
        'scalers': (scaler_x, scaler_y)
    }

def compare_methods():
    """比较两种方法"""
    print("🚀 开始回归方法对比分析\n")
    
    # 准备数据
    x, y = prepare_data()
    print(f"📊 数据集: {len(x)} 个样本, 特征范围 [{x.min():.1f}, {x.max():.1f}]")
    print(f"目标值范围 [{y.min():.1f}, {y.max():.1f}]\n")
    
    # 运行两种方法
    lr_results = linear_regression_method(x, y)
    nn_results = neural_network_method(x, y)
    
    # 打印对比结果
    print("\n" + "="*50)
    print("📈 性能对比结果")
    print("="*50)
    
    print(f"{'指标':<15} {'线性回归':<15} {'神经网络':<15} {'改进':<10}")
    print("-" * 60)
    
    # MSE对比
    mse_improvement = (lr_results['mse'] - nn_results['mse']) / lr_results['mse'] * 100
    print(f"{'MSE':<15} {lr_results['mse']:<15.4f} {nn_results['mse']:<15.4f} {mse_improvement:>+7.1f}%")
    
    # R²对比
    r2_improvement = (nn_results['r2'] - lr_results['r2']) / lr_results['r2'] * 100
    print(f"{'R²':<15} {lr_results['r2']:<15.4f} {nn_results['r2']:<15.4f} {r2_improvement:>+7.1f}%")
    
    # 训练时间对比
    time_ratio = nn_results['training_time'] / lr_results['training_time']
    print(f"{'训练时间(s)':<15} {lr_results['training_time']:<15.4f} {nn_results['training_time']:<15.4f} {time_ratio:>7.1f}x")
    
    print(f"{'模型参数':<15} {lr_results['params']:<15} {nn_results['params']:<15}")
    
    # 可视化对比
    visualize_comparison(x, y, lr_results, nn_results)
    
    return lr_results, nn_results

def visualize_comparison(x, y, lr_results, nn_results):
    """可视化对比结果"""
    
    # 创建密集的测试点
    x_plot = np.linspace(0.5, 10.5, 100).reshape(-1, 1)
    
    # 线性回归预测
    y_plot_lr = lr_results['model'].predict(x_plot)
    
    # 神经网络预测
    scaler_x, scaler_y = nn_results['scalers']
    x_plot_scaled = scaler_x.transform(x_plot)
    X_plot_tensor = torch.FloatTensor(x_plot_scaled)
    
    nn_results['model'].eval()
    with torch.no_grad():
        y_plot_scaled = nn_results['model'](X_plot_tensor).squeeze().numpy()
        y_plot_nn = scaler_y.inverse_transform(y_plot_scaled.reshape(-1, 1)).flatten()
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. 预测结果对比
    ax1 = axes[0]
    ax1.scatter(x.flatten(), y, color='black', s=100, alpha=0.8, label='真实数据', zorder=5)
    ax1.plot(x_plot.flatten(), y_plot_lr, color='blue', linewidth=2, label='线性回归', alpha=0.8)
    ax1.plot(x_plot.flatten(), y_plot_nn, color='red', linewidth=2, label='神经网络', alpha=0.8)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('预测结果对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 残差对比
    ax2 = axes[1]
    residuals_lr = y - lr_results['predictions']
    residuals_nn = y - nn_results['predictions']
    
    ax2.scatter(lr_results['predictions'], residuals_lr, color='blue', alpha=0.7, label='线性回归', s=80)
    ax2.scatter(nn_results['predictions'], residuals_nn, color='red', alpha=0.7, label='神经网络', s=80)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('预测值')
    ax2.set_ylabel('残差')
    ax2.set_title('残差分析')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 神经网络训练损失
    ax3 = axes[2]
    ax3.plot(nn_results['losses'], color='red', linewidth=1.5)
    ax3.set_xlabel('训练轮次')
    ax3.set_ylabel('损失值')
    ax3.set_title('神经网络训练过程')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    try:
        lr_results, nn_results = compare_methods()
        
        print("\n💡 总结:")
        if nn_results['mse'] < lr_results['mse']:
            print("✅ 神经网络在MSE指标上表现更好")
        else:
            print("✅ 线性回归在MSE指标上表现更好")
            
        if nn_results['training_time'] > lr_results['training_time'] * 2:
            print("⚠️  神经网络需要更多训练时间")
        else:
            print("✅ 两种方法训练时间相近")
            
        print("\n🎯 建议:")
        print("- 对于简单线性关系，线性回归更高效")
        print("- 对于复杂非线性关系，神经网络更有优势")
        print("- 数据量较小时，线性回归可能更稳定")
        print("- 数据量较大时，神经网络潜力更大")
        
    except Exception as e:
        print(f"❌ 运行出错: {e}")

if __name__ == "__main__":
    main()