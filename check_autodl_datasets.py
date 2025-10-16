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
    
    # 检查AutoDL预置数据集
    commands = [
        "ls -la /root/autodl-fs/",
        "ls -la /root/autodl-fs/data/ 2>/dev/null || echo '没有autodl-fs/data目录'",
        "ls -la /root/autodl-tmp/",
        "ls -la /root/autodl-pub/",
        "find /root -name '*fruit*' -type d 2>/dev/null",
        "find /root -name '*dataset*' -type d 2>/dev/null | head -10",
        "df -h | grep -E '(autodl|data)'",
        "echo '检查是否有预置的数据集目录...'",
        "ls -la /datasets/ 2>/dev/null || echo '没有/datasets目录'",
        "ls -la /data/ 2>/dev/null || echo '没有/data目录'"
    ]
    
    for cmd in commands:
        print(f"\n执行命令: {cmd}")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        
        if output:
            print(f"输出: {output}")
        if error and "No such file" not in error:
            print(f"错误: {error}")

finally:
    ssh.close()