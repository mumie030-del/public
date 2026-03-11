##new

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import Data3Dataset
from module import UnetWithLTAE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_left_right_tacs(dataloader, model_weight_path):
    """
    分别提取左右肾的TAC曲线
    输出：100个样本（50个病人 × 2侧）
    """
    print("加载 U-Net 模型权重...")
    model = UnetWithLTAE(in_channels=26, out_channels=1).to(DEVICE)
    checkpoint = torch.load(model_weight_path, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    all_left_tacs = []
    all_right_tacs = []
    patient_ids = []

    print("开始批量提取左右肾 TAC 曲线...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(DEVICE)  # shape: (B, 26, H, W)
            mask_logits = model(images)
            soft_mask = torch.sigmoid(mask_logits)  # shape: (B, 1, H, W
            masked_images = images * soft_mask 
            
            # 3. 沿中线分离左右肾
            W_mid = masked_images.shape[3] // 2

            
            image_right_kidney = masked_images[:, :, :, :W_mid]
            image_left_kidney = masked_images[:, :, :, W_mid:]
            
            # --- 【修复区：医学常识纠正】 ---
            # 计算左右半区 Mask 的实际面积 (防止除零)
            area_right = soft_mask[:, :, :, :W_mid].sum(dim=(2, 3)) + 1e-8 # shape: (B, 1)
            area_left = soft_mask[:, :, :, W_mid:].sum(dim=(2, 3)) + 1e-8  # shape: (B, 1)
            
            # 提取真正的 ROI TAC 曲线：信号总和 / 有效面积
            tac_right = image_right_kidney.sum(dim=(2, 3)) / area_right    # shape: (B, 26)
            tac_left = image_left_kidney.sum(dim=(2, 3)) / area_left       # shape: (B, 26)
            # ---------------------------------
            
        
            # 5. 保存到列表
            all_right_tacs.append(tac_right.cpu().numpy())
            all_left_tacs.append(tac_left.cpu().numpy())
            
            # 记录病人ID（假设从P001开始）
            for i in range(images.shape[0]):
                patient_id = f"P{(batch_idx * images.shape[0] + i + 1):03d}"
                patient_ids.append(patient_id)
    
    # 6. 合并所有批次
    all_left_tacs = np.concatenate(all_left_tacs, axis=0)   # (50, 26)
    all_right_tacs = np.concatenate(all_right_tacs, axis=0) # (50, 26)
    
    # 7. 交错排列：P001左, P001右, P002左, P002右, ...
    all_tacs = []
    kidney_sides = []
    patient_ids_expanded = []
    
    for i in range(len(patient_ids)):
        # 左肾
        all_tacs.append(all_left_tacs[i])
        kidney_sides.append('left')
        patient_ids_expanded.append(patient_ids[i])
        
        # 右肾
        all_tacs.append(all_right_tacs[i])
        kidney_sides.append('right')
        patient_ids_expanded.append(patient_ids[i])
    
    all_tacs = np.array(all_tacs)  # (100, 26)
    
    # 8. 保存
    np.save('extracted_tacs_left_right.npy', all_tacs)
    print(f"✓ 提取完成！共保存了 {all_tacs.shape[0]} 条 TAC 曲线")
    print(f"  - 50个病人 × 2侧 = 100个样本")
    print(f"  - 保存到: extracted_tacs_left_right.npy")
    
    
    return all_tacs, patient_ids_expanded, kidney_sides


if __name__ == '__main__':
    # 1. 构建 DataLoader
    DATA_DIR = '../data3'
    dataset = Data3Dataset(data_root=DATA_DIR, target_size=(256, 256), num_channels=26)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)

    # 2. 使用训练好的模型权重
    MODEL_PATH = './checkpoints/best_model.pth'

    extract_left_right_tacs(dataloader, MODEL_PATH)

