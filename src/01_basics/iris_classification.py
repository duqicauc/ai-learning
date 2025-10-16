import numpy as numpy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets
import torch

# 1. 加载数据集
iris = datasets.load_iris()
X = iris.data  # 每行4列，每列代表一个特征
y = iris.target # 每行1列，每列代表一个标签（0,1,2）

# 2. 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. 特征标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 5. 定义神经网络模型--最好定义为模型类
class IrisClassIfier(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=10, output_size=3):
        super(IrisClassIfier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.fc2 = torch.nn.Linear(hidden_size, output_size)  # 隐藏层到输出层

    def forward(self, x):
        x = self.fc1(x)          # 输入层
        x = torch.relu(x)        # 隐藏层激活函数
        x = self.fc2(x)          # 输出层
        return x
    
model = IrisClassIfier()

# 6. 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 7. 训练模型
num_epochs = 10000

train_losses = []

for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)  # 前向传播
    loss = criterion(outputs, y_train_tensor)  # 计算损失
    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

    train_losses.append(loss.item())

    if (epoch+1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 8. 测试模型
model.eval() #将模型设置为评估模式
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 9. 可视化训练过程中的损失变化
import matplotlib.pyplot as plt
plt.plot(range(num_epochs), train_losses)
plt.xlabel('Epoch')

# 可以改变的参数：轮次、学习率、隐藏层神经元个数
    # 1. 轮次：训练次数，训练次数越多，模型越准确，但是训练时间越长
    # 2. 学习率：学习率越小，训练速度越慢，但是训练效果越好
    # 3. 隐藏层神经元个数：隐藏层神经元个数越多，模型越准确，但是训练时间越长