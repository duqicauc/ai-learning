# 🚀 Trae + AutoDL 高效开发训练流程

## 📋 流程概述

这个工作流程让你在Trae IDE中进行代码开发和调试，然后无缝部署到AutoDL进行GPU训练，实现本地开发+云端训练的最佳实践。

## 🏗️ 整体架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Trae IDE      │    │   Git/GitHub    │    │   AutoDL 云端   │
│   本地开发       │───▶│   代码仓库       │───▶│   GPU训练       │
│   • 代码编写     │    │   • 版本控制     │    │   • 模型训练     │
│   • 调试测试     │    │   • 代码同步     │    │   • 结果保存     │
│   • 小规模验证   │    │   • 协作开发     │    │   • 模型下载     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔄 详细工作流程

### 阶段1: Trae本地开发 💻

#### 1.1 项目结构设计
```
ai-learning/
├── src/                    # 源代码
│   ├── models/            # 模型定义
│   ├── utils/             # 工具函数
│   └── training/          # 训练脚本
├── configs/               # 配置文件
│   ├── local.yaml        # 本地测试配置
│   └── autodl.yaml       # AutoDL训练配置
├── scripts/               # 部署脚本
│   ├── setup_autodl.sh   # AutoDL环境设置
│   ├── sync_code.sh      # 代码同步
│   └── start_training.sh # 启动训练
├── requirements/          # 依赖管理
│   ├── base.txt          # 基础依赖
│   ├── local.txt         # 本地开发依赖
│   └── autodl.txt        # AutoDL训练依赖
└── outputs/              # 输出目录
    ├── models/           # 训练好的模型
    ├── logs/             # 训练日志
    └── results/          # 实验结果
```

#### 1.2 本地开发最佳实践
- ✅ 使用小数据集进行快速验证
- ✅ 编写单元测试确保代码正确性
- ✅ 配置文件分离（本地vs云端）
- ✅ 模块化设计便于部署

### 阶段2: 代码同步策略 🔄

#### 2.1 Git工作流
```bash
# 1. 本地开发分支
git checkout -b feature/fruits-cnn-v2

# 2. 提交代码
git add .
git commit -m "feat: 优化水果分类CNN架构"

# 3. 推送到远程
git push origin feature/fruits-cnn-v2

# 4. 合并到主分支
git checkout main
git merge feature/fruits-cnn-v2
git push origin main
```

#### 2.2 自动化同步脚本
```bash
#!/bin/bash
# sync_to_autodl.sh
echo "🔄 开始同步代码到AutoDL..."

# 1. 推送最新代码
git add .
git commit -m "sync: 同步代码到AutoDL $(date)"
git push origin main

# 2. 在AutoDL上拉取
ssh autodl "cd /root/ai-learning && git pull origin main"

echo "✅ 代码同步完成!"
```

### 阶段3: AutoDL环境配置 ⚙️

#### 3.1 AutoDL实例选择建议
- **GPU**: RTX 3090 / A100 (根据预算)
- **内存**: 32GB+ 
- **存储**: 100GB+ SSD
- **镜像**: PyTorch 1.13+ / Python 3.8+

#### 3.2 环境初始化脚本
```bash
#!/bin/bash
# setup_autodl.sh

echo "🚀 初始化AutoDL训练环境..."

# 1. 更新系统
apt update && apt upgrade -y

# 2. 安装必要工具
apt install -y git htop tmux

# 3. 克隆代码仓库
cd /root
git clone https://github.com/your-username/ai-learning.git
cd ai-learning

# 4. 安装Python依赖
pip install -r requirements/autodl.txt

# 5. 创建必要目录
mkdir -p outputs/{models,logs,results}
mkdir -p data

echo "✅ AutoDL环境配置完成!"
```

### 阶段4: 训练执行策略 🎯

#### 4.1 配置文件管理
```yaml
# configs/autodl.yaml
training:
  batch_size: 64        # AutoDL上使用更大批次
  epochs: 100           # 完整训练轮数
  learning_rate: 0.001
  device: "cuda"        # 强制使用GPU
  
data:
  dataset_path: "/root/autodl-fs/data/fruits100"  # 文件存储挂载目录
  num_workers: 8        # 多进程数据加载
  
logging:
  log_dir: "/root/ai-learning/outputs/logs"
  save_interval: 10     # 每10轮保存一次
  
model:
  save_dir: "/root/ai-learning/outputs/models"
  checkpoint_interval: 5
```

#### 4.2 训练启动脚本
```bash
#!/bin/bash
# start_training.sh

echo "🍎 开始水果分类CNN训练..."

# 1. 激活虚拟环境
source /root/miniconda3/bin/activate

# 2. 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/ai-learning:$PYTHONPATH

# 3. 启动训练（使用tmux保持会话）
tmux new-session -d -s training "python src/03_cnn/fruits_classifier.py --config configs/autodl.yaml"

echo "✅ 训练已在tmux会话中启动!"
echo "💡 使用 'tmux attach -t training' 查看训练进度"
```

### 阶段5: 监控和管理 📊

#### 5.1 训练监控
```bash
# 查看GPU使用情况
nvidia-smi

# 查看训练进度
tmux attach -t training

# 查看日志
tail -f outputs/logs/training.log

# 查看系统资源
htop
```

#### 5.2 结果同步回本地
```bash
#!/bin/bash
# download_results.sh

echo "📥 下载训练结果..."

# 1. 下载模型文件
scp autodl:/root/ai-learning/outputs/models/*.pth ./models/

# 2. 下载日志文件
scp autodl:/root/ai-learning/outputs/logs/*.log ./logs/

# 3. 下载实验结果
scp -r autodl:/root/ai-learning/outputs/results/ ./

echo "✅ 结果下载完成!"
```

## 🛠️ 实用工具脚本

### 快速部署脚本
```bash
#!/bin/bash
# quick_deploy.sh

echo "⚡ 快速部署到AutoDL..."

# 1. 本地测试
echo "🧪 运行本地测试..."
python -m pytest tests/ -v

# 2. 同步代码
echo "🔄 同步代码..."
./scripts/sync_to_autodl.sh

# 3. 启动训练
echo "🚀 启动AutoDL训练..."
ssh autodl "cd /root/ai-learning && ./scripts/start_training.sh"

echo "✅ 部署完成! 训练已开始"
```

### 实时监控脚本
```bash
#!/bin/bash
# monitor_training.sh

echo "📊 监控AutoDL训练状态..."

while true; do
    clear
    echo "=== AutoDL训练监控 $(date) ==="
    
    # GPU状态
    ssh autodl "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
    
    # 训练进度
    ssh autodl "tail -n 5 /root/ai-learning/outputs/logs/training.log"
    
    sleep 30
done
```

## 📈 性能优化建议

### 1. 数据加载优化
- 使用多进程数据加载 (`num_workers=8`)
- 数据预处理缓存
- 使用SSD存储数据集

### 2. 训练优化
- 混合精度训练 (AMP)
- 梯度累积处理大批次
- 学习率预热和衰减

### 3. 资源管理
- 定期清理临时文件
- 监控磁盘空间使用
- 合理设置检查点保存频率

## 🔧 故障排除

### 常见问题解决
1. **SSH连接断开**: 使用tmux保持会话
2. **内存不足**: 减小batch_size或使用梯度累积
3. **磁盘空间不足**: 定期清理日志和临时文件
4. **代码同步失败**: 检查网络连接和Git配置

## 💡 最佳实践总结

1. **开发阶段**: 在Trae上快速迭代，小数据集验证
2. **测试阶段**: 本地单元测试，确保代码质量
3. **部署阶段**: 自动化脚本，一键部署到AutoDL
4. **训练阶段**: 云端GPU训练，本地监控进度
5. **结果阶段**: 自动下载结果，本地分析评估

这个流程让你能够充分利用Trae的开发体验和AutoDL的计算资源，实现高效的深度学习项目开发！