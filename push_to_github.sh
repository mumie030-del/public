#!/bin/bash
# 快速推送到 GitHub 的脚本

echo "=========================================="
echo "GitHub 代码推送助手"
echo "=========================================="
echo ""
read -p "请输入你的 GitHub 用户名: " GITHUB_USER
read -p "请输入你的仓库名称: " REPO_NAME
echo ""

# 添加远程仓库
git remote remove origin 2>/dev/null
git remote add origin https://github.com/${GITHUB_USER}/${REPO_NAME}.git

# 重命名分支为 main
git branch -M main

echo ""
echo "准备推送到: https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
echo ""
read -p "确认推送? (y/n): " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    git push -u origin main
    echo ""
    echo "✅ 推送完成！"
    echo "查看仓库: https://github.com/${GITHUB_USER}/${REPO_NAME}"
else
    echo "已取消推送"
fi
