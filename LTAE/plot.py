"""
绘制训练loss曲线并检测过拟合
支持从 checkpoint 文件或 JSON 文件读取训练历史，并支持绘制 Dice 和 IoU 曲线
"""
import json
import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse


def load_history_from_checkpoints(checkpoint_dir='checkpoints'):
    """
    从 checkpoint 文件中提取训练历史
    每个 checkpoint 文件包含: epoch, train_loss, val_loss, 以及可选的 val_dice, val_iou
    """
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth'))
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    if os.path.exists(best_model_path):
        checkpoint_files.append(best_model_path)
    
    if not checkpoint_files:
        raise FileNotFoundError(f"在 {checkpoint_dir} 中未找到 checkpoint 文件")
    
    history_data = []
    
    for ckpt_path in checkpoint_files:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            
            if 'epoch' in ckpt and 'train_loss' in ckpt and 'val_loss' in ckpt:
                # 尝试提取基础信息以及新增的指标信息 (使用 get 防止老模型报错)
                history_data.append({
                    'epoch': ckpt['epoch'],
                    'train_loss': ckpt['train_loss'],
                    'val_loss': ckpt['val_loss'],
                    'val_dice': ckpt.get('val_dice', None), # 获取 Dice
                    'val_iou': ckpt.get('val_iou', None)    # 获取 IoU
                })
            else:
                print(f"警告: {ckpt_path} 中缺少训练历史，跳过")
        except Exception as e:
            print(f"警告: 无法加载 {ckpt_path}: {e}")
            continue
    
    if not history_data:
        raise ValueError("没有找到包含训练历史的 checkpoint 文件")
    
    # 按 epoch 排序
    history_data.sort(key=lambda x: x['epoch'])
    
    # 转换为列表格式，自动过滤掉没有指标的历史记录
    history = {
        'epoch': [d['epoch'] for d in history_data],
        'train_loss': [d['train_loss'] for d in history_data],
        'val_loss': [d['val_loss'] for d in history_data],
        'val_dice': [d['val_dice'] for d in history_data if d['val_dice'] is not None],
        'val_iou': [d['val_iou'] for d in history_data if d['val_iou'] is not None]
    }
    
    print(f"从 {len(history_data)} 个 checkpoint 文件中提取训练历史")
    return history


def load_history(history_path):
    """加载训练历史（从 JSON 文件）"""
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Loss历史文件不存在: {history_path}")
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history


def detect_overfitting(history, window=5):
    """检测过拟合逻辑保持不变"""
    train_loss = np.array(history['train_loss'])
    val_loss = np.array(history['val_loss'])
    epochs = np.array(history['epoch'])
    
    if len(train_loss) < window:
        return None, "数据点不足，无法判断"
    
    train_trend = np.mean(np.diff(train_loss[-window:]))
    val_trend = np.mean(np.diff(val_loss[-window:]))
    gap = val_loss - train_loss
    gap_trend = np.mean(np.diff(gap[-window:]))
    
    is_overfitting = False
    warning_msg = ""
    
    if val_trend > 0 and train_trend < 0:
        is_overfitting = True
        warning_msg = f"⚠️ 检测到过拟合！训练loss下降({train_trend:.4f})，但验证loss上升({val_trend:.4f})"
    elif gap_trend > 0.01:
        is_overfitting = True
        warning_msg = f"⚠️ 检测到过拟合！训练和验证loss差距在增大({gap_trend:.4f})"
    elif val_trend > 0.001:
        warning_msg = f"⚠️ 警告：验证loss有上升趋势({val_trend:.4f})，可能出现过拟合"
    else:
        warning_msg = "✅ 训练正常，未检测到明显过拟合"
    
    best_epoch_idx = np.argmin(val_loss)
    best_epoch = epochs[best_epoch_idx]
    best_val_loss = val_loss[best_epoch_idx]
    
    return {
        'is_overfitting': is_overfitting,
        'warning': warning_msg,
        'best_epoch': int(best_epoch),
        'best_val_loss': float(best_val_loss),
        'train_trend': float(train_trend),
        'val_trend': float(val_trend),
        'gap_trend': float(gap_trend),
        'final_gap': float(gap[-1])
    }, None


def plot_curves(history, save_path='checkpoints/training_curves.png', show_plot=True):
    """绘制训练曲线 (新增 Dice 和 IoU 支持)"""
    epochs = history['epoch']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    val_dice = history.get('val_dice', [])
    val_iou = history.get('val_iou', [])
    
    # 检查是否有评估指标数据
    has_metrics = len(val_dice) > 0 or len(val_iou) > 0
    
    # 如果有指标数据，画 3 个子图，否则画 2 个
    if has_metrics:
        plt.figure(figsize=(18, 5))
        total_subplots = 3
    else:
        plt.figure(figsize=(12, 5))
        total_subplots = 2
        print("\n💡 提示：未在 checkpoint 中检测到 val_dice 或 val_iou 数据，仅绘制 Loss 曲线。")
    
    # 子图1: Loss曲线
    plt.subplot(1, total_subplots, 1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    best_idx = np.argmin(val_loss)
    best_epoch = epochs[best_idx]
    best_val = val_loss[best_idx]
    plt.plot(best_epoch, best_val, 'go', markersize=10)
    plt.annotate(f'Epoch {best_epoch}\nLoss: {best_val:.4f}', 
                 xy=(best_epoch, best_val), xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # 子图2: Loss差距（过拟合指标）
    plt.subplot(1, total_subplots, 2)
    gap = np.array(val_loss) - np.array(train_loss)
    plt.plot(epochs, gap, 'g-', label='Val Loss - Train Loss', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss Gap', fontsize=12)
    plt.title('Overfitting Indicator', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if len(gap) > 5:
        recent_gap = gap[-5:]
        if np.mean(np.diff(recent_gap)) > 0:
            plt.fill_between(epochs[-5:], gap[-5:], alpha=0.3, color='red', label='Increasing Gap')
    
    # 子图3: 评估指标曲线 (Dice & IoU)
    if has_metrics:
        plt.subplot(1, total_subplots, 3)
        if len(val_dice) > 0:
            # 取最后的对应长度 epochs 画图，防止列表长度不一致
            plt.plot(epochs[-len(val_dice):], val_dice, color='#2ca02c', linestyle='-', label='Val Dice', linewidth=2)
        if len(val_iou) > 0:
            plt.plot(epochs[-len(val_iou):], val_iou, color='#9467bd', linestyle='-', label='Val IoU', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Validation Metrics (Dice & IoU)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='绘制训练loss曲线并检测过拟合')
    parser.add_argument('--history_path', type=str, default=None,
                        help='训练历史JSON文件路径')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='checkpoint 文件目录')
    parser.add_argument('--save_path', type=str, default='checkpoints/training_curves.png',
                        help='保存图片的路径')
    parser.add_argument('--no_show', action='store_true',
                        help='不显示图片，只保存')
    
    args = parser.parse_args()
    
    try:
        if args.history_path and os.path.exists(args.history_path):
            history = load_history(args.history_path)
            print(f"成功从 JSON 文件加载训练历史")
        else:
            history = load_history_from_checkpoints(args.checkpoint_dir)
        print(f"总训练轮数: {len(history['epoch'])}")
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        return
    
    overfitting_info, error = detect_overfitting(history)
    if not error:
        print("\n" + "="*60)
        print("过拟合检测结果:")
        print("="*60)
        print(overfitting_info['warning'])
        print(f"最佳验证loss: {overfitting_info['best_val_loss']:.4f} (Epoch {overfitting_info['best_epoch']})")
        print("="*60)
    
    plot_curves(history, save_path=args.save_path, show_plot=not args.no_show)

if __name__ == '__main__':
    main()
