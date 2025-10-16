#!/usr/bin/env python3
"""
代码同步脚本 - Trae到AutoDL
自动化代码同步、环境检查和训练启动流程
"""

import os
import sys
import subprocess
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import paramiko
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoDLSync:
    """AutoDL同步管理器"""
    
    def __init__(self, config_file="configs/sync_config.yaml"):
        self.config = self.load_config(config_file)
        self.ssh_client = None
        
    def load_config(self, config_file):
        """加载同步配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"配置文件 {config_file} 不存在，使用默认配置")
            return self.get_default_config()
    
    def get_default_config(self):
        """获取默认配置"""
        return {
            'autodl': {
                'host': 'your-autodl-instance.com',
                'port': 22,
                'username': 'root',
                'key_file': '~/.ssh/id_rsa',
                'remote_path': '/root/ai-learning'
            },
            'git': {
                'auto_commit': True,
                'commit_message_template': 'sync: 同步代码到AutoDL {timestamp}'
            },
            'sync': {
                'exclude_patterns': [
                    '*.pyc',
                    '__pycache__/',
                    '.git/',
                    'outputs/',
                    'data/',
                    '*.log',
                    '.vscode/',
                    '.idea/'
                ],
                'include_configs': True,
                'backup_before_sync': True
            }
        }
    
    def connect_ssh(self):
        # 检查私钥权限
        import os
        key_path = os.path.expanduser(self.config['autodl']['key_file'])
        if os.stat(key_path).st_mode & 0o777 != 0o600:
            os.chmod(key_path, 0o600)
            logger.info("已修复私钥文件权限")
        # 上传公钥到AutoDL
        pub_key = open(f"{key_path}.pub").read()
        # 移动到连接成功后执行
        # self.execute_remote_command(f"mkdir -p ~/.ssh && echo '{pub_key}' >> ~/.ssh/authorized_keys")
        """建立SSH连接"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            key_file = os.path.expanduser(self.config['autodl']['key_file'])
            self.ssh_client.connect(
                hostname=self.config['autodl']['host'],
                port=self.config['autodl']['port'],
                username=self.config['autodl']['username'],
                key_filename=key_file
            )
            logger.info("SSH连接建立成功")
            return True
        except Exception as e:
            logger.error(f"❌ SSH连接失败: {e}")
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
            logger.error(f"远程命令执行失败: {e}")
            return None, str(e)
    
    def git_operations(self):
        """Git操作"""
        if not self.config['git']['auto_commit']:
            logger.info("跳过自动Git提交")
            return True
        
        try:
            # 检查Git状态
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            
            if result.stdout.strip():
                # 有未提交的更改
                logger.info("发现未提交的更改，开始提交...")
                
                # 添加所有更改
                subprocess.run(['git', 'add', '.'], check=True)
                
                # 提交更改
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                commit_msg = self.config['git']['commit_message_template'].format(
                    timestamp=timestamp
                )
                subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
                
                # 推送到远程
                subprocess.run(['git', 'push', 'origin', 'main'], check=True)
                logger.info("Git操作完成")
            else:
                logger.info("没有需要提交的更改")
            
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"[ERROR] Git操作失败: {e}")
            return False
    
    def sync_code(self):
        # 确保远程目录存在
        self.execute_remote_command(f"mkdir -p {self.config['autodl']['remote_path']}")
        """同步代码到AutoDL"""
        try:
            remote_path = self.config['autodl']['remote_path']
            
            # 备份远程代码（如果启用）
            if self.config['sync']['backup_before_sync']:
                backup_cmd = f"cp -r {remote_path} {remote_path}_backup_$(date +%Y%m%d_%H%M%S)"
                self.execute_remote_command(backup_cmd)
                logger.info("远程代码备份完成")
            
            # 初始化Git仓库（如果不存在）
            self.execute_remote_command(f"cd {self.config['autodl']['remote_path']} && git init")
            # 拉取最新代码
            # 使用SCP同步本地代码到远程
            scp_cmd = f"scp -P {self.config['autodl']['port']} -i {self.config['autodl']['key_file']} -r ./* {self.config['autodl']['username']}@{self.config['autodl']['host']}:{self.config['autodl']['remote_path']}"
            subprocess.run(scp_cmd, shell=True, check=True)
            output, error = self.execute_remote_command(pull_cmd)
            
            if error and "fatal" in error.lower():
                logger.error(f"代码同步失败: {error}")
                return False
            
            logger.info("✅ 代码同步完成")
            return True
            
        except Exception as e:
            logger.error(f"代码同步异常: {e}")
            return False
    
    def check_environment(self):
        """检查AutoDL环境"""
        checks = [
            ("Python版本", "python --version"),
            ("PyTorch版本", "python -c 'import torch; print(torch.__version__)'"),
            ("CUDA可用性", "python -c 'import torch; print(torch.cuda.is_available())'"),
            ("GPU信息", "nvidia-smi --query-gpu=name --format=csv,noheader"),
            ("磁盘空间", "df -h /"),
            ("内存信息", "free -h")
        ]
        
        logger.info("🔍 检查AutoDL环境...")
        for check_name, command in checks:
            output, error = self.execute_remote_command(command)
            if output:
                logger.info(f"{check_name}: {output.strip()}")
            else:
                logger.warning(f"{check_name}: 检查失败 - {error}")
    
    def install_dependencies(self):
        """安装依赖"""
        try:
            remote_path = self.config['autodl']['remote_path']
            
            # 激活虚拟环境并安装依赖
            install_cmd = f"""
            cd {remote_path} && 
            source venv/bin/activate && 
            pip install -r requirements/autodl.txt
            """
            
            logger.info("📦 安装Python依赖...")
            output, error = self.execute_remote_command(install_cmd)
            
            if error and "error" in error.lower():
                logger.warning(f"依赖安装警告: {error}")
            
            logger.info("✅ 依赖安装完成")
            return True
            
        except Exception as e:
            logger.error(f"❌ 依赖安装失败: {e}")
            return False
    
    def start_training(self, config_name="autodl"):
        """启动训练"""
        try:
            remote_path = self.config['autodl']['remote_path']
            
            # 创建tmux会话并启动训练
            training_cmd = f"""
            cd {remote_path} && 
            tmux new-session -d -s training 
            'source venv/bin/activate && 
             export CUDA_VISIBLE_DEVICES=0 && 
             python src/03_cnn/fruits_classifier.py --config configs/{config_name}.yaml'
            """
            
            logger.info("启动训练...")
            output, error = self.execute_remote_command(training_cmd)
            
            if error:
                logger.warning(f"训练启动警告: {error}")
            
            logger.info("训练已在tmux会话中启动")
            logger.info("提示: 使用 'tmux attach -t training' 查看训练进度")
            return True
            
        except Exception as e:
            logger.error(f"❌ 训练启动失败: {e}")
            return False
    
    def monitor_training(self):
        """监控训练状态"""
        try:
            remote_path = self.config['autodl']['remote_path']
            
            # 检查tmux会话
            session_cmd = "tmux list-sessions | grep training"
            output, error = self.execute_remote_command(session_cmd)
            
            if output:
                logger.info("✅ 训练会话正在运行")
                
                # 获取最新日志
                log_cmd = f"tail -n 10 {remote_path}/outputs/logs/training.log"
                log_output, _ = self.execute_remote_command(log_cmd)
                
                if log_output:
                    logger.info("📈 最新训练日志:")
                    print(log_output)
                
                return True
            else:
                logger.warning("⚠️ 未找到训练会话")
                return False
                
        except Exception as e:
            logger.error(f"❌ 监控失败: {e}")
            return False
    
    def full_sync_and_train(self, config_name="autodl"):
        """完整的同步和训练流程"""
        logger.info("[INFO] 开始完整同步和训练流程...")
        
        # 1. Git操作
        if not self.git_operations():
            return False
        
        # 2. 建立SSH连接
        if not self.connect_ssh():
            return False
        
        # 3. 同步代码
        if not self.sync_code():
            return False
        
        # 4. 检查环境
        self.check_environment()
        
        # 5. 安装依赖
        if not self.install_dependencies():
            return False
        
        # 6. 启动训练
        if not self.start_training(config_name):
            return False
        
        logger.info("同步和训练启动完成!")
        return True
    
    def close(self):
        """关闭连接"""
        if self.ssh_client:
            self.ssh_client.close()
            logger.info("SSH连接已关闭")

def main():
    parser = argparse.ArgumentParser(description="AutoDL代码同步和训练工具")
    parser.add_argument('--config', default='configs/sync_config.yaml', 
                       help='同步配置文件路径')
    parser.add_argument('--training-config', default='autodl',
                       help='训练配置名称')
    parser.add_argument('--action', choices=['sync', 'train', 'monitor', 'full'],
                       default='full', help='执行的操作')
    
    args = parser.parse_args()
    
    # 创建同步器
    syncer = AutoDLSync(args.config)
    
    try:
        if args.action == 'sync':
            syncer.git_operations()
            syncer.connect_ssh()
            syncer.sync_code()
        elif args.action == 'train':
            syncer.connect_ssh()
            syncer.start_training(args.training_config)
        elif args.action == 'monitor':
            syncer.connect_ssh()
            syncer.monitor_training()
        elif args.action == 'full':
            syncer.full_sync_and_train(args.training_config)
    
    finally:
        syncer.close()

if __name__ == "__main__":
    main()