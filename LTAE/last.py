##new
import os

import torch
import numpy as np
import pandas as pd

from deep import PharmacokineticLatentEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quantify_mixed_ratios():
    # 1. 加载模型和两个潜空间聚类中心
    model = PharmacokineticLatentEncoder(input_dim=26, latent_dim=64, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load("latent_encoder.pth", map_location=DEVICE))
    model.eval()

    # centers shape: [2, 64] -> center[0]是功能中心，center[1]是机械中心
    centers = torch.load("manifold_centers.pth", map_location=DEVICE)  # (2, 64)
    center_func = centers[0]
    center_mech = centers[1]

    # 2. 加载 TAC 特征并归一化（必须与训练时保持一致！）
    tacs = np.load('extracted_tacs_left_right.npy')  # 形状: (N, 26)

    # 归一化（与 deep.py 中的处理一致）
    tacs_min = tacs.min(axis=1, keepdims=True)
    tacs_max = tacs.max(axis=1, keepdims=True)
    tacs = (tacs - tacs_min) / (tacs_max - tacs_min + 1e-8)

    # 3. 从 clinical_labels_left_right.csv 读取病例级标签，并挑出混合型 (label == 2)
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(proj_root, "LTAE","clinical_labels_left_right.csv")
    df = pd.read_csv(csv_path)
    class_labels = df["label"].values.astype(np.int64)
    
    if len(class_labels) != tacs.shape[0]:
        raise ValueError(
            f"clinical_labels.csv 中样本数 {len(class_labels)} 与 extracted_tacs.npy "
            f"行数 {tacs.shape[0]} 不一致，请检查顺序或数量是否对应。"
        )

    mixed_idx = class_labels == 2
    mixed_tacs_np = tacs[mixed_idx]  # (N_mixed, 26)
    mixed_tacs = torch.tensor(mixed_tacs_np, dtype=torch.float32).to(DEVICE)
    
    # 获取混合型样本的病人ID和肾侧信息
    mixed_patient_ids = df[mixed_idx]["patient_id"].values
    mixed_kidney_sides = df[mixed_idx]["kidney_side"].values

    print(f"共有 {mixed_tacs.shape[0]} 个混合型梗阻病例需要量化评估。\n")

    with torch.no_grad():
        # 4. 将混合型病人送入网络，映射到潜空间
        _, mixed_latent_feats = model(mixed_tacs)  # shape: (Num_mixed, 64)

        # 5. 计算流形距离！
        for i, latent_feat in enumerate(mixed_latent_feats):
            # 计算到纯功能中心的欧式距离 (D_func)
            dist_func = torch.dist(latent_feat, center_func, p=2).item()
            # 计算到纯机械中心的欧式距离 (D_mech)
            dist_mech = torch.dist(latent_feat, center_mech, p=2).item()

            # 6. 换算为机械性梗阻占比 (核心公式)
            # 离功能越远，离机械越近，机械占比越高
            mech_ratio = dist_func / (dist_func + dist_mech)

            # 显示病人ID和肾侧信息
            print(f"病例 {mixed_patient_ids[i]} {mixed_kidney_sides[i]}:")
            print(f"  -> 距纯功能中心距离: {dist_func:.4f}")
            print(f"  -> 距纯机械中心距离: {dist_mech:.4f}")
            print(f"  => 最终诊断: 机械性梗阻占比为 {mech_ratio * 100:.1f}%\n")


if __name__ == "__main__":
    quantify_mixed_ratios()
