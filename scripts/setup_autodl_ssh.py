#!/usr/bin/env python3
"""
AutoDL SSH密钥配置脚本
用于在AutoDL服务器上配置GitHub SSH密钥
"""

import os
import sys
import yaml
import paramiko
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoDLSSHSetup:
    def __init__(self, config_file="configs/sync_config.yaml"):
        self.config = self.load_config(config_file)
        self.ssh_client = None
    
    def load_config(self, config_file):
        """加载配置文件"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return None
    
    def connect_ssh(self):
        """建立SSH连接"""
        try:
            # 修复私钥文件权限
            key_file = os.path.expanduser(self.config['autodl']['key_file'])
            if os.path.exists(key_file):
                os.chmod(key_file, 0o600)
                logger.info("已修复私钥文件权限")
            
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            self.ssh_client.connect(
                hostname=self.config['autodl']['host'],
                port=self.config['autodl']['port'],
                username=self.config['autodl']['username'],
                key_filename=key_file
            )
            logger.info("SSH连接建立成功")
            return True
        except Exception as e:
            logger.error(f"SSH连接失败: {e}")
            return False
    
    def execute_remote_command(self, command):
        """执行远程命令"""
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(command)
            output = stdout.read().decode('utf-8')
            error = stderr.read().decode('utf-8')
            
            if error:
                logger.warning(f"命令警告: {error}")
            
            return output, error
        except Exception as e:
            logger.error(f"执行远程命令失败: {e}")
            return "", str(e)
    
    def setup_ssh_key(self):
        """在AutoDL上设置SSH密钥"""
        try:
            logger.info("开始配置AutoDL SSH密钥...")
            
            # 检查是否已有SSH密钥
            output, error = self.execute_remote_command("ls -la ~/.ssh/")
            logger.info(f"当前SSH目录内容:\n{output}")
            
            # 生成SSH密钥（如果不存在）
            key_check_cmd = "test -f ~/.ssh/id_rsa && echo 'exists' || echo 'not_exists'"
            output, error = self.execute_remote_command(key_check_cmd)
            
            if 'not_exists' in output:
                logger.info("生成新的SSH密钥...")
                keygen_cmd = 'ssh-keygen -t rsa -b 4096 -C "autodl@github.com" -f ~/.ssh/id_rsa -N ""'
                output, error = self.execute_remote_command(keygen_cmd)
                
                if error and "already exists" not in error:
                    logger.error(f"SSH密钥生成失败: {error}")
                    return False
                
                logger.info("SSH密钥生成成功")
            else:
                logger.info("SSH密钥已存在")
            
            # 获取公钥内容
            output, error = self.execute_remote_command("cat ~/.ssh/id_rsa.pub")
            if output:
                logger.info("=== AutoDL SSH公钥 ===")
                logger.info(output.strip())
                logger.info("=== 请将上述公钥添加到GitHub SSH密钥中 ===")
                logger.info("GitHub设置地址: https://github.com/settings/ssh/new")
            else:
                logger.error("无法读取SSH公钥")
                return False
            
            # 配置SSH客户端
            ssh_config = """Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
"""
            
            config_cmd = f'echo "{ssh_config}" > ~/.ssh/config'
            self.execute_remote_command(config_cmd)
            
            # 设置正确的权限
            self.execute_remote_command("chmod 700 ~/.ssh")
            self.execute_remote_command("chmod 600 ~/.ssh/id_rsa")
            self.execute_remote_command("chmod 644 ~/.ssh/id_rsa.pub")
            self.execute_remote_command("chmod 644 ~/.ssh/config")
            
            logger.info("SSH配置完成")
            return True
            
        except Exception as e:
            logger.error(f"SSH密钥配置失败: {e}")
            return False
    
    def test_github_connection(self):
        """测试GitHub连接"""
        try:
            logger.info("测试GitHub SSH连接...")
            output, error = self.execute_remote_command("ssh -T git@github.com")
            
            if "successfully authenticated" in output or "successfully authenticated" in error:
                logger.info("✅ GitHub SSH连接成功")
                return True
            else:
                logger.warning(f"GitHub连接测试结果: {output} {error}")
                return False
                
        except Exception as e:
            logger.error(f"GitHub连接测试失败: {e}")
            return False
    
    def close(self):
        """关闭SSH连接"""
        if self.ssh_client:
            self.ssh_client.close()
            logger.info("SSH连接已关闭")

def main():
    setup = AutoDLSSHSetup()
    
    if not setup.connect_ssh():
        logger.error("无法连接到AutoDL")
        return
    
    try:
        # 设置SSH密钥
        if setup.setup_ssh_key():
            logger.info("SSH密钥配置完成")
            
            # 提示用户添加公钥到GitHub
            print("\n" + "="*50)
            print("重要提示:")
            print("1. 请复制上面显示的SSH公钥")
            print("2. 访问 https://github.com/settings/ssh/new")
            print("3. 将公钥添加到GitHub账户")
            print("4. 添加完成后，重新运行同步脚本")
            print("="*50)
            
        else:
            logger.error("SSH密钥配置失败")
    
    finally:
        setup.close()

if __name__ == "__main__":
    main()