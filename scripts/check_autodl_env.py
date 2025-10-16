#!/usr/bin/env python3
"""
检查AutoDL环境的Python配置
"""

import yaml
import paramiko
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    """加载配置"""
    with open('configs/sync_config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_python_env():
    """检查Python环境"""
    config = load_config()
    autodl_config = config['autodl']
    
    # 建立SSH连接
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        import os
        key_path = os.path.expanduser(autodl_config['key_file'])
        ssh.connect(
            hostname=autodl_config['host'],
            port=autodl_config['port'],
            username=autodl_config['username'],
            key_filename=key_path
        )
        
        logger.info("SSH连接成功")
        
        # 检查各种Python命令
        commands = [
            "which python",
            "which python3", 
            "which python3.8",
            "which python3.9",
            "which python3.10",
            "which python3.11",
            "python --version",
            "python3 --version",
            "python3.8 --version",
            "python3.9 --version", 
            "python3.10 --version",
            "python3.11 --version",
            "which pip",
            "which pip3",
            "pip --version",
            "pip3 --version",
            "ls -la /usr/bin/python*",
            "ls -la /opt/conda/bin/python*",
            "echo $PATH"
        ]
        
        for cmd in commands:
            try:
                stdin, stdout, stderr = ssh.exec_command(cmd)
                output = stdout.read().decode('utf-8').strip()
                error = stderr.read().decode('utf-8').strip()
                
                if output:
                    logger.info(f"{cmd}: {output}")
                elif error:
                    logger.warning(f"{cmd}: {error}")
                else:
                    logger.warning(f"{cmd}: 无输出")
                    
            except Exception as e:
                logger.error(f"执行命令 {cmd} 失败: {e}")
        
    except Exception as e:
        logger.error(f"SSH连接失败: {e}")
    finally:
        ssh.close()

if __name__ == "__main__":
    check_python_env()