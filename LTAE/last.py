import torch
import numpy as np
# 导入 PharmacokineticLatentEncoder

def quantify_mixed_ratios():
    # 1. 加载模型和两个潜空间聚类中心
    model = PharmacokineticLatentEncoder(input_dim=26, latent_dim=64, num_classes=2).cuda()
    model.load_state_dict(torch.load('latent_encoder.pth'))
    model.eval()
    
    # centers shape: [2, 64] -> center[0]是功能中心，center[1]是机械中心
    centers = torch.load('manifold_centers.pth') 
    center_func = centers[0]
    center_mech = centers[1]

    # 2. 加载数据，这次只挑出“混合型/交叉型 (假设 Label 是一开始标的 2)”病人
    tacs = np.load('extracted_tacs.npy')
    labels = np.load('extracted_labels.npy')
    mixed_idx = (labels == 2)
    mixed_tacs = torch.tensor(tacs[mixed_idx], dtype=torch.float32).cuda()

    print(f"共有 {len(mixed_tacs)} 个混合型梗阻病例需要量化评估。\n")
    
    with torch.no_grad():
        # 3. 将混合型病人送入网络，映射到潜空间
        _, mixed_latent_feats = model(mixed_tacs) # shape: (Num_mixed, 64)
        
        # 4. 计算流形距离！
        for i, latent_feat in enumerate(mixed_latent_feats):
            # 计算到纯功能中心的欧式距离 (D_func)
            dist_func = torch.dist(latent_feat, center_func, p=2).item()
            # 计算到纯机械中心的欧式距离 (D_mech)
            dist_mech = torch.dist(latent_feat, center_mech, p=2).item()
            
            # 5. 换算为机械性梗阻占比 (核心公式)
            # 离功能越远，离机械越近，机械占比越高
            mech_ratio = dist_func / (dist_func + dist_mech)
            
            print(f"病例 {i+1}:")
            print(f"  -> 距纯功能中心距离: {dist_func:.4f}")
            print(f"  -> 距纯机械中心距离: {dist_mech:.4f}")
            print(f"  => 最终诊断: 机械性梗阻占比为 {mech_ratio * 100:.1f}%\n")

# 运行: quantify_mixed_ratios()