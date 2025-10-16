#!/usr/bin/env python3
"""
检查AutoDL上的训练日志
"""

import yaml
import paramiko
import logging
import os

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """加载配置"""
    with open('configs/sync_config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_training_log():
    """检查训练日志"""
    config = load_config()
    autodl_config = config['autodl']
    
    # 建立SSH连接
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        key_path = os.path.expanduser(autodl_config['key_file'])
        ssh.connect(
            hostname=autodl_config['host'],
            port=autodl_config['port'],
            username=autodl_config['username'],
            key_filename=key_path
        )
        
        logger.info("SSH连接成功")
        remote_path = autodl_config['remote_path']
        
        # 检查训练日志
        commands = [
            f"cd {remote_path} && ls -la",
            f"cd {remote_path} && ls -la training.log",
            f"cd {remote_path} && cat training.log",
            f"cd {remote_path} && ls -la src/03_cnn/",
            f"cd {remote_path} && ls -la configs/",
            f"cd {remote_path} && python3 src/03_cnn/fruits_classifier.py --help",
        ]
        
        for cmd in commands:
            try:
                logger.info(f"执行命令: {cmd}")
                stdin, stdout, stderr = ssh.exec_command(cmd)
                output = stdout.read().decode('utf-8').strip()
                error = stderr.read().decode('utf-8').strip()
                
                if output:
                    logger.info(f"输出:\n{output}")
                if error:
                    logger.warning(f"错误:\n{error}")
                    
                logger.info("-" * 50)
                    
            except Exception as e:
                logger.error(f"执行命令 {cmd} 失败: {e}")
        
    except Exception as e:
        logger.error(f"SSH连接失败: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    check_training_log()