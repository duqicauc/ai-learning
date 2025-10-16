import torch
from torch.cuda.amp import autocast, GradScaler

# 1. 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 模型 & 数据迁移
model = MyModel().to(device)
train_loader = DataLoader(..., pin_memory=True)

# 3. 优化器 & AMP
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler()

# 4. 训练循环
model.train()
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 可选：定期清缓存（一般不需要）
        # if batch_idx % 100 == 0:
        #     torch.cuda.empty_cache()