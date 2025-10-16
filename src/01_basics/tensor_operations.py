import numpy as np
import matplotlib as plt
from sklearn.linear_model import LinearRegression
import torch

# 初始化数据
x_data = torch.tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0])
y_true = torch.tensor([3.1,5.0,7.2,9.1,11.0,13.1,15.0,16.8,19.2,21.0])

# 初始化参数
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# 设置学习率（控制参数增长的步长）
learning_rate = 0.01

# 训练循环（定义迭代次数）
for epoch in range(10000):
    # 前向计算
    y_pred = w * x_data + b
    # 计算损失（均方误差mse）
    loss = torch.mean((y_pred - y_true) ** 2)
    # 反向传播计算梯度，自动计算w.grad和b.grad
    loss.backward()
    # 更新参数（梯度下降）
    with torch.no_grad():  # 该代码块中的计算不进行梯度计算
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        # 清零梯度
        w.grad.zero_()
        b.grad.zero_()
        # 观察训练过程，每隔20轮训练打印一次结果
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: loss={loss.item():.4f}, w={w.item():.4f}, b={b.item():.4f}')

# 训练结束后打印最终的参数值
print(f'Final parameters: w={w.item():.4f}, b={b.item():.4f}')

# 预测结果
with torch.no_grad():
    y_pred = w * x_data + b
    print('Predicted values:', y_pred.numpy())

# 注意，不改变算法的情况下，如果要提高精度，要么增加训练轮次，要么增加数据
# 可视化结果
plt.pyplot.scatter(x_data.numpy(), y_true.numpy(), label='True Data')
plt.pyplot.plot(x_data.numpy(), y_pred.numpy(), color='red', label='Fitted Line')
plt.pyplot.xlabel('x') 