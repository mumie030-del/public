import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import Data3Dataset
# 导入您自己写的 U-Net+LTAE 类
from module import UnetWithLTAE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_all_tacs(dataloader, model_weight_path):
    print("加载 U-Net 模型权重...")
    model = UnetWithLTAE(in_channels=26, out_channels=1).to(DEVICE)
    # 训练保存的是完整 checkpoint 字典，需要先取出其中的 model_state_dict
    checkpoint = torch.load(model_weight_path, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # 兼容直接保存 state_dict 的情况
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval() # 开启评估模式，冻结权重

    all_tacs = []
    all_labels = []

    print("开始批量提取 TAC 曲线...")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE) # shape: (B, 26, H, W)
            
            # 1. 预测掩码
            mask_logits = model(images)
            soft_mask = torch.sigmoid(mask_logits) # shape: (B, 1, H, W)
            
            # 2. 掩码过滤（抹除背景）
            masked_images = images * soft_mask 
            
            # 3. 空间池化，得到 26 维 TAC 曲线
            # 计算每帧图像在掩码区域内的平均亮度
            tac_curve = masked_images.mean(dim=(2, 3)) # shape: (B, 26)
            
            all_tacs.append(tac_curve.cpu().numpy())
            all_labels.append(labels.numpy())

    # 将提取出的黄金特征保存为 numpy 数组，供下一步直接读取！
    final_tacs = np.concatenate(all_tacs, axis=0)
    final_labels = np.concatenate(all_labels, axis=0)
    np.save('extracted_tacs.npy', final_tacs)
    np.save('extracted_labels.npy', final_labels)
    print(f"提取完成！共保存了 {final_tacs.shape[0]} 条 TAC 曲线。")

if __name__ == '__main__':
    # 1. 构建 DataLoader（与训练/测试保持一致的数据预处理）
    DATA_DIR = '../data3'
    dataset = Data3Dataset(data_root=DATA_DIR, target_size=(256, 256), num_channels=26)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)

    # 2. 使用 LTAE 训练得到的最佳权重
    MODEL_PATH = './checkpoints/best_model.pth'

    extract_all_tacs(dataloader, MODEL_PATH)