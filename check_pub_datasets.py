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
    
    # 检查AutoDL公共数据集
    commands = [
        "ls -la /root/autodl-fs/ | grep -i fruit",
        "find /root/autodl-fs -name '*fruit*' -type d 2>/dev/null",
        "find /root/autodl-fs -name '*food*' -type d 2>/dev/null", 
        "find /root/autodl-fs -name '*classification*' -type d 2>/dev/null",
        "ls -la /root/autodl-fs/CIFAR* 2>/dev/null",
        "ls -la /autodl-pub/ | grep -i fruit",
        "find /autodl-pub -name '*fruit*' -type d 2>/dev/null",
        "find /autodl-pub -name '*food*' -type d 2>/dev/null",
        "find /autodl-pub -name '*classification*' -type d 2>/dev/null",
        "ls -la /autodl-pub/CIFAR* 2>/dev/null",
        "echo '检查ImageNet相关数据集...'",
        "ls -la /root/autodl-fs/ImageNet* 2>/dev/null",
        "ls -la /autodl-pub/ImageNet* 2>/dev/null"
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