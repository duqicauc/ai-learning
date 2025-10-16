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
    
    # 使用ModelScope SDK下载
    download_cmd = """
    cd /root/ai-learning && 
    rm -rf data/fruits100 && 
    pip3 install datasets && 
    pip3 install modelscope && 
    python3 -c "
import os
from modelscope.msdatasets import MsDataset

# 创建数据目录
os.makedirs('data', exist_ok=True)
os.chdir('data')

print('开始下载数据集...')
try:
    # 下载训练集
    train_dataset = MsDataset.load('tany0699/fruits100', subset_name='default', split='train')
    print(f'训练集样本数: {len(train_dataset)}')
    
    # 下载验证集
    val_dataset = MsDataset.load('tany0699/fruits100', subset_name='default', split='validation')
    print(f'验证集样本数: {len(val_dataset)}')
    
    # 检查第一个样本
    first_sample = next(iter(train_dataset))
    print(f'第一个样本: {first_sample}')
    
    print('✅ ModelScope SDK下载成功')
except Exception as e:
    print(f'ModelScope SDK下载失败: {e}')
"
    """
    
    print("执行ModelScope SDK下载...")
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