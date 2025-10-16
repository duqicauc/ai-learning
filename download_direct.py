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
    
    # 尝试直接下载数据集压缩包
    download_cmd = """
    cd /root/ai-learning && 
    rm -rf data/fruits100 && 
    mkdir -p data && 
    cd data && 
    echo "尝试下载数据集压缩包..." && 
    wget -O fruits100.zip "https://www.modelscope.cn/api/v1/datasets/tany0699/fruits100/repo?Revision=master&FilePath=" || 
    echo "wget失败，尝试curl..." && 
    curl -L -o fruits100.zip "https://www.modelscope.cn/api/v1/datasets/tany0699/fruits100/repo?Revision=master&FilePath=" || 
    echo "直接下载失败，使用git clone..." && 
    git clone https://www.modelscope.cn/datasets/tany0699/fruits100.git && 
    cd fruits100 && 
    echo "检查仓库内容..." && 
    ls -la && 
    echo "检查是否有LFS文件..." && 
    git lfs ls-files && 
    echo "尝试下载LFS文件..." && 
    git lfs pull || echo "LFS pull失败" && 
    echo "最终文件列表:" && 
    find . -type f | head -20
    """
    
    print("执行直接下载...")
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