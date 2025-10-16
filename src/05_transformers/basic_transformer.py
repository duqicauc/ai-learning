# -*- coding: utf-8 -*-
"""
第十二节实战：使用简化 Transformer 预测每日最低气温（时间序列回归）
任务：用过去30天的气温，预测第31天的气温（连续值）
对比第十一节的 LSTM，理解 Transformer 如何处理序列
"""

# ----------------------------
# 第一步：导入库（复用第十一节数据）
# ----------------------------

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import matplotlib.pyplot as plt
import urllib.request
import os

torch.manual_seed(42)
np.random.seed(42)

# ----------------------------
# 第二步：加载并预处理数据（同第十一节）
# ----------------------------

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
data_file = "daily-min-temperatures.csv"

if not os.path.exists(data_file):
    urllib.request.urlretrieve(url, data_file)

df = pd.read_csv(data_file)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Temp'] = pd.to_numeric(df['Temp'].astype(str).str.strip(), errors='coerce')
df = df.dropna()

# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
temps_scaled = scaler.fit_transform(df['Temp'].values.reshape(-1, 1)).flatten()

# 构造滑动窗口
seq_length = 30
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)

X, y = create_sequences(temps_scaled, seq_length)

# 划分训练/测试集
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 转为张量：注意 Transformer 输入是 (batch, seq_len, 1)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32)

# DataLoader
batch_size = 64
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=batch_size, shuffle=False)

# ----------------------------
# 第三步：实现位置编码（Positional Encoding）
# ----------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        为序列添加位置信息
        :param d_model: 模型维度（此处=1，但通常>=32）
        :param max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        # 偶数维用 sin，奇数维用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 所有行，偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 所有行，奇数列
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)  # 不作为模型参数，但随模型移动到GPU

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        返回 x + 位置编码（只取前 seq_len 个位置）
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

# ----------------------------
# 第四步：构建简化 Transformer Encoder 层
# ----------------------------

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim=1, d_model=32, nhead=4, num_layers=1, dim_feedforward=64, dropout=0.1):
        """
        简化版 Transformer 回归模型（仅 Encoder）
        :param input_dim: 输入特征维度（气温=1）
        :param d_model: Transformer 内部维度（必须能被 nhead 整除）
        :param nhead: 多头注意力的头数
        :param num_layers: Encoder 层数
        :param dim_feedforward: 前馈网络隐藏层维度
        """
        super(TransformerRegressor, self).__init__()
        
        # 🌟 步骤1：将输入从 input_dim 映射到 d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 🌟 步骤2：位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_length)
        
        # 🌟 步骤3：Transformer Encoder 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 输入形状: (batch, seq_len, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 🌟 步骤4：回归输出层（取最后一个时间步）
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim=1)
        return: (batch,)
        """
        # 1. 投影到 d_model 维度
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # 2. 添加位置编码
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)
        
        # 3. Transformer Encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # 4. 取最后一个时间步的输出用于回归
        x = x[:, -1, :]  # (batch, d_model)
        
        # 5. 回归输出
        out = self.regressor(x)  # (batch, 1)
        return out.squeeze(-1)  # (batch,)

# ----------------------------
# 第五步：创建模型、损失函数、优化器
# ----------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 注意：d_model=32（不能为1，因为 nhead=4 需整除）
model = TransformerRegressor(
    input_dim=1,
    d_model=32,
    nhead=4,
    num_layers=1,
    dim_feedforward=64,
    dropout=0.1
).to(device)

print(f"✅ 模型已创建，使用设备: {device}")
print(model)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# 第六步：训练模型
# ----------------------------

print("\n🚀 开始训练 Transformer...")

num_epochs = 50
train_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

# 绘制损失曲线
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Transformer Training Loss')
plt.title('📉 Transformer 训练损失')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# 第七步：测试与评估
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

# 反归一化
test_preds = np.array(test_preds).reshape(-1, 1)
test_targets = np.array(test_targets).reshape(-1, 1)
test_preds_original = scaler.inverse_transform(test_preds).flatten()
test_targets_original = scaler.inverse_transform(test_targets).flatten()

mae = mean_absolute_error(test_targets_original, test_preds_original)
rmse = math.sqrt(mean_squared_error(test_targets_original, test_preds_original))

print(f"\n🎉 Transformer 测试结果:")
print(f"MAE: {mae:.2f} °C")
print(f"RMSE: {rmse:.2f} °C")

# 可视化
plt.figure(figsize=(14, 6))
plt.plot(test_targets_original, label='真实气温', color='steelblue')
plt.plot(test_preds_original, label='Transformer 预测', color='red', linestyle='--')
plt.title('🌡️ Transformer 气温预测结果（测试集）')
plt.xlabel('时间步（天）')
plt.ylabel('最低气温 (°C)')
plt.legend()
plt.grid(True)
plt.show()