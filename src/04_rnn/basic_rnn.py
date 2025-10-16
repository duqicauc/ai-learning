# -*- coding: utf-8 -*-
"""
第十一节实战：使用 LSTM 预测每日最低气温（时间序列回归）
数据来源：https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv
任务：用过去30天的气温，预测第31天的气温（连续值）
"""

# ----------------------------
# 第一步：导入所需库
# ----------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler  # 用于归一化
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import urllib.request
import os

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# ----------------------------
# 第二步：下载并加载数据
# ----------------------------

# 数据 URL
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
data_file = "daily-min-temperatures.csv"

# 如果本地没有，就下载
if not os.path.exists(data_file):
    print("📥 正在下载气温数据...")
    urllib.request.urlretrieve(url, data_file)
    print(f"✅ 数据已保存为 {data_file}")

# 读取 CSV 文件
# 文件格式：Date,Temp
# 示例：1981-01-01,20.7
df = pd.read_csv(data_file)
print(f"\n📊 原始数据形状: {df.shape}")
print(df.head())

# 转换日期列为 datetime 类型（便于后续处理）
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')  # 确保按时间排序

# 提取气温列（转为 float，注意原始数据可能有空格）
df['Temp'] = df['Temp'].astype(str).str.strip()
df['Temp'] = pd.to_numeric(df['Temp'], errors='coerce')  # 将无效值转为 NaN

# 删除缺失值
df = df.dropna()
print(f"✅ 清洗后数据量: {len(df)} 天")

# 可视化原始气温序列
plt.figure(figsize=(12, 4))
plt.plot(df['Date'], df['Temp'], color='steelblue')
plt.title('🌡️ 澳大利亚墨尔本每日最低气温 (1981–1990)')
plt.xlabel('日期')
plt.ylabel('气温 (°C)')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 第三步：数据预处理 — 归一化 + 构造滑动窗口
# ----------------------------

# 🌟 新概念：归一化（Normalization）
# 目的：将气温缩放到 [0, 1] 区间，加速训练、避免梯度爆炸
scaler = MinMaxScaler(feature_range=(0, 1))
temps_scaled = scaler.fit_transform(df['Temp'].values.reshape(-1, 1))  # 转为列向量
temps_scaled = temps_scaled.flatten()  # 转回一维数组

print(f"\n🔍 归一化后气温范围: [{temps_scaled.min():.3f}, {temps_scaled.max():.3f}]")

# 🌟 新概念：滑动窗口（Sliding Window）
# 思路：用连续 seq_length 天的数据作为输入，预测下一天
# 例如：[t1, t2, ..., t30] → t31
seq_length = 30  # 使用过去30天预测第31天

def create_sequences(data, seq_len):
    """
    从一维时间序列构造 (X, y) 样本
    :param data: 归一化后的一维数组，如 [0.1, 0.3, 0.5, ...]
    :param seq_len: 输入序列长度（如30）
    :return: X (样本数, seq_len, 1), y (样本数,)
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        # 输入：从 i 到 i+seq_len-1
        X.append(data[i:i + seq_len])
        # 输出：i+seq_len 时刻的值
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

# 构造全部样本
X, y = create_sequences(temps_scaled, seq_length)
print(f"\n🧩 构造的样本数: {X.shape[0]}")
print(f"输入形状 X: {X.shape} → (样本数, 时间步长, 特征数)")
print(f"输出形状 y: {y.shape} → (样本数,)")

# ----------------------------
# 第四步：划分训练集和测试集
# ----------------------------

# 按时间顺序划分（不能随机打乱！）
# 前 80% 为训练集，后 20% 为测试集
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n✅ 训练集: {X_train.shape[0]} 样本")
print(f"✅ 测试集: {X_test.shape[0]} 样本")

# 转为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)  # [N, 30] → [N, 30, 1]
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32)

print(f"\n🧠 张量形状确认:")
print(f"X_train: {X_train.shape} → (batch, seq_len, input_size=1)")
print(f"y_train: {y_train.shape} → (batch,)")

# ----------------------------
# 第五步：自定义 Dataset 类（可选，但推荐）
# ----------------------------

class TempDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建 DataLoader
batch_size = 64
train_dataset = TempDataset(X_train, y_train)
test_dataset = TempDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 时间序列不 shuffle！
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------
# 第六步：定义 LSTM 回归模型
# ----------------------------

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        """
        LSTM 回归模型
        :param input_size: 每个时间步的特征数（气温是1维）
        :param hidden_size: LSTM 隐藏单元数
        :param num_layers: LSTM 层数
        :param output_size: 输出维度（回归任务通常为1）
        """
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 🌟 核心：LSTM 层
        # input_size=1：因为每天只有1个气温值
        # batch_first=True：输入形状为 (batch, seq_len, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # 🌟 回归输出层：将最后一个时间步的隐藏状态映射到1个输出值
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状 (batch, seq_len, 1)
        :return: 预测值，形状 (batch, 1)
        """
        # 初始化隐藏状态和细胞状态（可选，PyTorch 会自动初始化为0）
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM 前向传播
        # output: (batch, seq_len, hidden_size)
        # (hn, cn): 最后时刻的隐藏状态和细胞状态
        output, (hn, cn) = self.lstm(x)  # hn 形状: (num_layers, batch, hidden_size)

        # 🌟 关键：我们只关心最后一个时间步的输出（即预测未来一天）
        # 取 output[:, -1, :] 或 hn[-1] 都可以（单层时等价）
        last_output = output[:, -1, :]  # (batch, hidden_size)

        # 通过全连接层得到最终预测
        out = self.fc(last_output)  # (batch, 1)

        return out.squeeze(-1)  # 去掉最后一维 → (batch,)

# 创建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRegressor(input_size=1, hidden_size=50, num_layers=1, output_size=1).to(device)
print(f"\n✅ 模型已创建，使用设备: {device}")
print(model)

# ----------------------------
# 第七步：设置损失函数与优化器
# ----------------------------

criterion = nn.MSELoss()  # 回归任务常用 MSE（均方误差）
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# 第八步：训练模型
# ----------------------------

print("\n🚀 开始训练...")

num_epochs = 50
train_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # 前向传播
        outputs = model(inputs)  # outputs: (batch,)
        loss = criterion(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

# 绘制训练损失曲线
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Training Loss')
plt.title('📉 训练损失曲线')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# 第九步：在测试集上评估
# ----------------------------

model.eval()
test_preds = []
test_targets = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        test_preds.extend(outputs.cpu().numpy())
        test_targets.extend(targets.cpu().numpy())

# 反归一化（将预测值和真实值转回原始气温单位）
test_preds = np.array(test_preds).reshape(-1, 1)
test_targets = np.array(test_targets).reshape(-1, 1)

test_preds_original = scaler.inverse_transform(test_preds).flatten()
test_targets_original = scaler.inverse_transform(test_targets).flatten()

# 计算回归指标
mae = mean_absolute_error(test_targets_original, test_preds_original)
rmse = math.sqrt(mean_squared_error(test_targets_original, test_preds_original))

print(f"\n🎉 测试集评估结果:")
print(f"平均绝对误差 (MAE): {mae:.2f} °C")
print(f"均方根误差 (RMSE): {rmse:.2f} °C")

# ----------------------------
# 第十步：可视化预测结果
# ----------------------------

plt.figure(figsize=(14, 6))
plt.plot(test_targets_original, label='真实气温', color='steelblue')
plt.plot(test_preds_original, label='LSTM 预测', color='orange', linestyle='--')
plt.title('🌡️ LSTM 气温预测结果（测试集）')
plt.xlabel('时间步（天）')
plt.ylabel('最低气温 (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 第十一：预测未来一天（演示）
# ----------------------------

def predict_next_day(model, last_sequence, scaler, device):
    """
    使用最近 seq_length 天的数据预测下一天气温
    :param model: 训练好的 LSTM 模型
    :param last_sequence: 最近 seq_length 天的归一化气温（一维数组）
    :param scaler: 归一化器
    :param device: CPU/GPU
    :return: 预测的原始气温值
    """
    # 转为张量并增加 batch 和 feature 维度
    input_tensor = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # [1, 30, 1]
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        pred = model(input_tensor)  # [1,]
        pred_scaled = pred.cpu().item()
        pred_original = scaler.inverse_transform([[pred_scaled]])[0, 0]
    return pred_original

# 示例：用测试集最后30天预测“未来一天”
last_30_days = temps_scaled[-seq_length:]  # 最后30天的归一化数据
future_pred = predict_next_day(model, last_30_days, scaler, device)
print(f"\n🔮 基于最近30天气温，预测下一天最低气温: {future_pred:.2f} °C")