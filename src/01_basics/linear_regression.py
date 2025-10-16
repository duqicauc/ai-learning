import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. 生成数据（x和y）
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(-1, 1)
y = np.array([3.1, 5.0, 7.2, 9.1, 11.0, 13.1, 15.0, 16.8, 19.2, 21.0])

# 2. 训练一元线性回归模型
model = LinearRegression()  # 创建模型
model.fit(x, y)             # 拟合数据

# 3. 获取拟合的参数（斜率b和截距w）
b = model.coef_[0]    # 斜率
w = model.intercept_  # 截距
print(f"拟合结果: y = {b:.2f}x + {w:.2f}")

# 4. 预测值（用于绘制回归线）
y_pred = model.predict(x)

# 5. 绘制数据点和回归线
plt.scatter(x, y, color='blue', label='真实数据')  # 原始数据点
plt.plot(x, y_pred, color='red', label='回归线')   # 回归线
plt.xlabel('x (特征)')
plt.ylabel('y (标签)')
plt.title('一元线性回归拟合示例')
plt.legend()
plt.grid(True)
plt.show()