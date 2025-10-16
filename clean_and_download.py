#!/usr/bin/env python3
import paramiko
import yaml
import os

# 加载配置
with open('configs/sync_config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

autodl_config = config['autodl']

# 连接AutoDL
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(
        hostname=autodl_config['host'],
        port=autodl_config['port'],
        username=autodl_config['username'],
        key_filename=os.path.expanduser(autodl_config['key_file'])
    )
    
    # 清理现有数据集并重新下载
    download_cmd = """
    cd /root/ai-learning && 
    rm -rf data/fruits100 && 
    mkdir -p data && 
    cd data && 
    echo "开始克隆数据集..." && 
    git clone https://www.modelscope.cn/datasets/tany0699/fruits100.git && 
    cd fruits100 && 
    echo "开始下载LFS文件..." && 
    git lfs pull && 
    echo "验证下载的图片..." && 
    find . -name "*.jpg" | wc -l && 
    echo "清理Git文件..." && 
    rm -rf .git && 
    echo "✅ 数据集下载完成"
    """
    
    print("执行数据集下载命令...")
    stdin, stdout, stderr = ssh.exec_command(download_cmd)
    
    # 实时输出
    while True:
        line = stdout.readline()
        if not line:
            break
        print(line.strip())
    
    error = stderr.read().decode('utf-8')
    if error:
        print(f"错误: {error}")

finally:
    ssh.close()