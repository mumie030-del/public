import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from datasets import Data3Dataset
from CBAM import UnetWithCBAM  # 请确保这里导入的是您跑出 0.24 的那个基础 U-Net
from CBAM import UnetWithCBAM
from B import UnetWithECA  # 修正：导入正确的类
from tqdm import tqdm
import albumentations as A
from scipy.ndimage import label


# ==================== 配置参数 ====================
DATA_DIR = '../data3'
CHECKPOINT_PATH = './checkpoints/best_model.pth'  # 指向您最好的那个模型权重
BATCH_SIZE = 1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_dice(pred_mask, true_mask, smooth=1e-5):
    """计算单个样本的二值化 Dice 分数"""
    pred_mask = pred_mask.flatten()
    true_mask = true_mask.flatten()
    intersection = (pred_mask * true_mask).sum()
    return (2. * intersection + smooth) / (pred_mask.sum() + true_mask.sum() + smooth)

def main():
    print(f"使用设备: {DEVICE}")
    
    # 1. 加载测试数据 (为了严谨，我们应该只在验证集上测试)
    # 这里我们简单加载全量数据，或者您可以固定之前的 random_split 种子
    dataset = Data3Dataset(data_root=DATA_DIR, target_size=(256, 256), num_channels=26, transform=None)
    
    # 为了演示，我们取最后 10 个样本作为测试集
    test_size = min(10, len(dataset))
    test_indices = list(range(len(dataset) - test_size, len(dataset)))
    test_dataset = Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"测试集大小: {len(test_dataset)}")

    # 2. 加载模型
    model = UnetWithECA(in_channels=26, out_channels=1, use_dropout=False).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型权重: {CHECKPOINT_PATH}")
        print(f"该模型保存时的验证集 Loss 为: {checkpoint.get('val_loss', '未知'):.4f}")
    else:
        raise FileNotFoundError(f"找不到模型权重文件: {CHECKPOINT_PATH}")

    model.eval()
    total_dice = 0.0

    # 3. 开始推理并可视化
    print("\n开始测试并生成对比图...")
    os.makedirs('test_results', exist_ok=True)

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # 前向传播
            outputs = model(images)
            
            # 将输出转为概率 (0~1)，并使用 0.5 作为阈值二值化
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.7).float()

            # 计算 Dice 分数
            print(f"[Debug] probs min={probs.min().item():.4f}, max={probs.max().item():.4f}, mean={probs.mean().item():.4f}")
            dice_score = calculate_dice(preds.cpu().numpy(), masks.cpu().numpy())
            total_dice += dice_score
            print(f"样本 {i+1}/{len(test_loader)} - Dice 分数: {dice_score:.4f}")

            # ==================== 可视化 ====================
            # 将 Tensor 转为 numpy 用于画图
            img_np = images[0].cpu().numpy()  # (26, H, W)
            true_mask_np = masks[0, 0].cpu().numpy() # (H, W)
            pred_mask_np = preds[0, 0].cpu().numpy() # (H, W)
            prob_mask_np = probs[0, 0].cpu().numpy() # (H, W) 概率图

            plt.figure(figsize=(16, 4))
            
            # 子图 1: 原图 (为了方便看，我们挑 26 帧里中间的第 13 帧展示)
            plt.subplot(1, 4, 1)
            plt.title("Input (Frame 13/26)")
            plt.imshow(img_np[13], cmap='gray')
            plt.axis('off')

            # 子图 2: 真实的 Ground Truth
            plt.subplot(1, 4, 2)
            plt.title("Ground Truth (Label)")
            plt.imshow(true_mask_np, cmap='gray')
            plt.axis('off')

            # 子图 3: 模型预测的概率热力图
            plt.subplot(1, 4, 3)
            plt.title("Predicted Probability")
            plt.imshow(prob_mask_np, cmap='magma') # 用热力图展示，越亮概率越高
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')

            # 子图 4: 最终的二值化预测结果
            plt.subplot(1, 4, 4)
            plt.title(f"Final Prediction (Dice: {dice_score:.2f})")
            plt.imshow(pred_mask_np, cmap='gray')
            plt.axis('off')

            plt.tight_layout()
            save_path = f'test_results/sample_{i+1}.png'
            plt.savefig(save_path)
            plt.close()

    avg_dice = total_dice / len(test_loader)
    print(f"\n测试完成！平均 Dice 分数: {avg_dice:.4f}")
    print("可视化结果已保存到 'test_results/' 文件夹中，请打开查看。")

if __name__ == '__main__':
    main()