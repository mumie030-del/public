# 如何将代码推送到 GitHub

## 步骤 1: 在 GitHub 上创建新仓库

1. 登录 [GitHub](https://github.com)
2. 点击右上角的 "+" → "New repository"
3. 填写仓库名称（例如：`kidney-segmentation`）
4. 选择 Public 或 Private
5. **不要**勾选 "Initialize this repository with a README"
6. 点击 "Create repository"

## 步骤 2: 连接到远程仓库并推送

在终端中执行以下命令（将 `YOUR_USERNAME` 和 `YOUR_REPO_NAME` 替换为你的实际值）：

```bash
cd /root

# 添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 或者使用 SSH（如果你配置了 SSH key）
# git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git

# 推送代码到 GitHub
git branch -M main
git push -u origin main
```

## 步骤 3: 如果遇到认证问题

### 方法 1: 使用 Personal Access Token (推荐)

1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. 点击 "Generate new token (classic)"
3. 勾选 `repo` 权限
4. 生成后复制 token
5. 推送时使用 token 作为密码：
   ```bash
   git push -u origin main
   # Username: 你的 GitHub 用户名
   # Password: 粘贴你的 token（不是密码）
   ```

### 方法 2: 使用 SSH Key

```bash
# 生成 SSH key（如果还没有）
ssh-keygen -t ed25519 -C "your_email@example.com"

# 查看公钥
cat ~/.ssh/id_ed25519.pub

# 复制公钥内容，添加到 GitHub → Settings → SSH and GPG keys
```

## 快速命令（一键推送）

如果你想快速推送，可以运行：

```bash
cd /root
# 替换下面的 URL 为你的仓库地址
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

## 后续更新

以后每次修改代码后，使用以下命令更新 GitHub：

```bash
cd /root
git add .
git commit -m "描述你的修改"
git push
```

