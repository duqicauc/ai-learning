#!/bin/bash
# AutoDL环境初始化脚本
# 用于在AutoDL实例上快速搭建训练环境

set -e  # 遇到错误立即退出

echo "🚀 开始初始化AutoDL训练环境..."
echo "=================================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 1. 系统更新
log_info "更新系统包..."
apt update && apt upgrade -y
log_success "系统更新完成"

# 2. 安装必要工具
log_info "安装系统工具..."
apt install -y \
    git \
    htop \
    tmux \
    tree \
    wget \
    curl \
    unzip \
    vim \
    screen \
    rsync
log_success "系统工具安装完成"

# 3. 配置Git（如果需要）
log_info "配置Git环境..."
if [ ! -f ~/.gitconfig ]; then
    log_warning "请手动配置Git用户信息:"
    echo "git config --global user.name 'Your Name'"
    echo "git config --global user.email 'your.email@example.com'"
fi

# 4. 创建项目目录结构
log_info "创建项目目录结构..."
cd /root
mkdir -p {data,models,logs,checkpoints,results}

# 5. 克隆代码仓库（需要替换为实际仓库地址）
log_info "克隆代码仓库..."
REPO_URL="https://github.com/your-username/ai-learning.git"  # 替换为实际仓库
if [ ! -d "ai-learning" ]; then
    git clone $REPO_URL
    log_success "代码仓库克隆完成"
else
    log_warning "代码仓库已存在，更新代码..."
    cd ai-learning
    git pull origin main
    cd ..
fi

# 6. 进入项目目录
cd /root/ai-learning

# 7. 创建Python虚拟环境
log_info "创建Python虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_success "虚拟环境创建完成"
fi

# 8. 激活虚拟环境并安装依赖
log_info "安装Python依赖..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements/autodl.txt
log_success "Python依赖安装完成"

# 9. 下载数据集（从AutoDL文件存储）
log_info "准备数据集..."
if [ ! -d "/root/data/fruits100" ]; then
    log_info "从AutoDL文件存储同步fruits100数据集..."
    # 直接从AutoDL文件存储加载数据，无需本地同步
ln -s /root/autodl-fs/fruits100/ /root/data/fruits100
    log_success "数据集同步完成"
else
    log_info "数据集已存在，跳过同步"
fi

# 10. 创建必要的输出目录
log_info "创建输出目录..."
mkdir -p outputs/{models,logs,results,checkpoints}
mkdir -p data

# 11. 配置tmux
log_info "配置tmux环境..."
cat > ~/.tmux.conf << 'EOF'
# tmux配置
set -g default-terminal "screen-256color"
set -g history-limit 10000
set -g mouse on

# 快捷键绑定
bind r source-file ~/.tmux.conf \; display "Config reloaded!"
bind | split-window -h
bind - split-window -v

# 状态栏配置
set -g status-bg colour235
set -g status-fg colour136
set -g status-left '#[fg=colour166]#S '
set -g status-right '#[fg=colour166]%Y-%m-%d %H:%M'
EOF
log_success "tmux配置完成"

# 12. 创建训练启动脚本
log_info "创建训练启动脚本..."
cat > start_training.sh << 'EOF'
#!/bin/bash
# 训练启动脚本

echo "🍎 开始水果分类CNN训练..."

# 激活虚拟环境
source /root/ai-learning/venv/bin/activate

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/ai-learning:$PYTHONPATH

# 检查GPU
nvidia-smi

# 启动训练
cd /root/ai-learning
python src/03_cnn/fruits_classifier.py --config configs/autodl.yaml

echo "✅ 训练完成!"
EOF

chmod +x start_training.sh
log_success "训练启动脚本创建完成"

# 13. 创建监控脚本
log_info "创建监控脚本..."
cat > monitor.sh << 'EOF'
#!/bin/bash
# 训练监控脚本

echo "📊 AutoDL训练监控面板"
echo "======================"

while true; do
    clear
    echo "=== 系统状态 $(date) ==="
    
    # GPU状态
    echo "🔥 GPU状态:"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
    
    echo ""
    echo "💾 内存使用:"
    free -h
    
    echo ""
    echo "💿 磁盘使用:"
    df -h /
    
    echo ""
    echo "📈 最新训练日志:"
    if [ -f "/root/ai-learning/outputs/logs/training.log" ]; then
        tail -n 5 /root/ai-learning/outputs/logs/training.log
    else
        echo "训练日志文件不存在"
    fi
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    sleep 10
done
EOF

chmod +x monitor.sh
log_success "监控脚本创建完成"

# 14. 设置自动备份
log_info "设置自动备份..."
cat > backup.sh << 'EOF'
#!/bin/bash
# 自动备份脚本

BACKUP_DIR="/root/backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# 备份模型
cp -r /root/ai-learning/outputs/models $BACKUP_DIR/
# 备份日志
cp -r /root/ai-learning/outputs/logs $BACKUP_DIR/
# 备份配置
cp -r /root/ai-learning/configs $BACKUP_DIR/

echo "✅ 备份完成: $BACKUP_DIR"
EOF

chmod +x backup.sh

# 添加到crontab（每小时备份一次）
(crontab -l 2>/dev/null; echo "0 * * * * /root/ai-learning/backup.sh") | crontab -
log_success "自动备份设置完成"

# 15. 显示环境信息
log_info "环境信息检查..."
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q True; then
    echo "GPU数量: $(python -c 'import torch; print(torch.cuda.device_count())')"
    echo "GPU名称: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# 16. 完成提示
echo ""
echo "=================================================="
log_success "🎉 AutoDL环境初始化完成!"
echo ""
echo "📋 接下来的步骤:"
echo "1. 下载数据集: python scripts/download_datasets.py"
echo "2. 启动训练: ./start_training.sh"
echo "3. 监控训练: ./monitor.sh"
echo "4. 查看日志: tail -f outputs/logs/training.log"
echo ""
echo "🔧 有用的命令:"
echo "- tmux new -s training    # 创建训练会话"
echo "- tmux attach -t training # 连接训练会话"
echo "- nvidia-smi             # 查看GPU状态"
echo "- htop                   # 查看系统资源"
echo ""
log_success "环境准备就绪，开始愉快的训练吧! 🚀"