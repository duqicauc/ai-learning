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
    
    # 检查数据集状态
    commands = [
        "cd /root/ai-learning && ls -la data/",
        "cd /root/ai-learning && ls -la data/fruits100/",
        "cd /root/ai-learning && find data/fruits100 -type f | head -10",
        "cd /root/ai-learning && du -sh data/fruits100",
        "cd /root/ai-learning/data/fruits100 && git lfs ls-files | head -5",
        "cd /root/ai-learning/data/fruits100 && ls -la | head -10"
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