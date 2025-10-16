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
    
    # 检查CSV文件内容
    commands = [
        "cd /root/ai-learning/data/fruits100 && head -5 train.csv",
        "cd /root/ai-learning/data/fruits100 && head -5 classname.txt",
        "cd /root/ai-learning/data/fruits100 && cat README.md"
    ]
    
    for cmd in commands:
        print(f"\n执行命令: {cmd}")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        
        if output:
            print(f"输出: {output}")
        if error:
            print(f"错误: {error}")

finally:
    ssh.close()