# API密钥安全使用指南

## ⚠️ 重要安全提醒

本项目已经修复了API密钥安全问题，所有硬编码的API密钥已被移除并改用环境变量。

## 🔒 当前安全状态

### 已修复的安全问题
1. **硬编码API密钥** - 所有聊天演示文件中的硬编码API密钥已被移除
2. **硬编码密码** - `scripts/sync_to_autodl.py`中的硬编码密码已被移除
3. **环境变量配置** - 所有敏感信息现在通过环境变量管理

### 受影响的文件（已修复）
- `src/07_chat/chatDemo.py`
- `src/07_chat/chatMultiDemo.py`
- `src/07_chat/chatReasoningDemo.py`
- `src/07_chat/chatStreamDemo.py`
- `src/07_chat/chatMultiAdvanceDemo.py`
- `tests/chat/test_multi_chat.py`
- `scripts/sync_to_autodl.py`

## 🛡️ 安全配置

### 1. 环境变量设置

创建 `.env` 文件（已在项目根目录提供模板）：

```bash
# SiliconFlow API Configuration
SILICONFLOW_API_KEY=your-actual-api-key-here

# AutoDL Server Configuration
AUTODL_PASSWORD=your-actual-password-here
AUTODL_HOST=your-server-host
AUTODL_PORT=your-server-port
AUTODL_USERNAME=your-username
AUTODL_REMOTE_PATH=your-remote-path
```

### 2. Windows环境变量设置

```powershell
# 临时设置（当前会话有效）
$env:SILICONFLOW_API_KEY="your-api-key"

# 永久设置（需要管理员权限）
[Environment]::SetEnvironmentVariable("SILICONFLOW_API_KEY", "your-api-key", "User")
```

### 3. 验证配置

运行任何聊天演示前，确保环境变量已正确设置：

```python
import os
print("API Key configured:", "✅" if os.getenv("SILICONFLOW_API_KEY") else "❌")
```

## 🚨 Git历史清理

### 问题说明
由于之前的提交包含硬编码API密钥，这些密钥可能仍然存在于Git历史中。

### 解决方案

#### 选项1：重写Git历史（推荐用于私有仓库）
```bash
# 使用git filter-branch移除敏感信息
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch src/07_chat/*.py tests/chat/*.py scripts/sync_to_autodl.py' \
  --prune-empty --tag-name-filter cat -- --all

# 强制推送（谨慎使用）
git push origin --force --all
```

#### 选项2：撤销API密钥（推荐）
1. **立即撤销所有暴露的API密钥**
2. **生成新的API密钥**
3. **使用新密钥配置环境变量**

#### 选项3：使用BFG Repo-Cleaner
```bash
# 下载BFG工具
# 替换敏感文件
java -jar bfg.jar --replace-text passwords.txt your-repo.git
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

## 📋 安全检查清单

- [ ] 所有硬编码API密钥已移除
- [ ] 环境变量已正确配置
- [ ] `.env`文件已添加到`.gitignore`
- [ ] 旧的API密钥已撤销
- [ ] 新的API密钥已生成并配置
- [ ] 团队成员已了解安全配置流程

## 🔄 持续安全实践

### 1. 代码审查
- 每次提交前检查是否包含敏感信息
- 使用自动化工具扫描敏感信息

### 2. 环境隔离
- 开发、测试、生产环境使用不同的API密钥
- 定期轮换API密钥

### 3. 监控和审计
- 监控API密钥使用情况
- 定期审计代码库中的敏感信息

## 📞 紧急响应

如果发现API密钥泄露：

1. **立即撤销泄露的密钥**
2. **生成新的密钥**
3. **更新所有使用该密钥的系统**
4. **检查是否有异常使用**
5. **通知相关团队成员**

## 📚 相关资源

- [GitHub: 移除敏感数据](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
- [Git Filter-Branch](https://git-scm.com/docs/git-filter-branch)

---

**记住：安全是一个持续的过程，而不是一次性的任务！**