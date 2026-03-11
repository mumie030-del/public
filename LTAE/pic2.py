import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from deep import PharmacokineticLatentEncoder
from sklearn.decomposition import PCA
# ==========================================
# 顶刊级绘图全局设置 (Global Aesthetics)
# ==========================================
# 强制使用无衬线学术字体 (Arial/Helvetica风格)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
# 统一加粗线宽
plt.rcParams['axes.linewidth'] = 1.5

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def plot_latent_manifold():
    """
    可视化潜空间流形 (顶刊发表版)
    """
    print("1. 正在加载大模型与潜空间中心...")
    model = PharmacokineticLatentEncoder(input_dim=26, latent_dim=64, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load('latent_encoder.pth', map_location=DEVICE))
    model.eval()
    
    centers = torch.load('manifold_centers.pth', map_location=DEVICE)
    
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(proj_root, 'LTAE', "clinical_labels_left_right.csv")
    
    print("2. 正在加载病人 TAC 数据...")
    df = pd.read_csv(csv_path)
    labels = df["label"].values.astype(np.int64)
    tacs = np.load('extracted_tacs_left_right.npy')
    
    # Min-Max 归一化
    tacs_min = tacs.min(axis=1, keepdims=True)
    tacs_max = tacs.max(axis=1, keepdims=True)
    tacs = (tacs - tacs_min) / (tacs_max - tacs_min + 1e-8)

    print("3. 网络推理提取特征...")
    with torch.no_grad():
        tacs_tensor = torch.tensor(tacs, dtype=torch.float32).to(DEVICE)
        _, latent_feats = model(tacs_tensor)
        latent_feats = latent_feats.cpu().numpy()

    print("4. 拼接特征并进行 t-SNE 降维 (修复坐标系错位问题)...")
    centers_np = centers.cpu().numpy()
    # 【核心修复】：必须拼在一起降维！
    combined_feats = np.vstack([latent_feats, centers_np])
    
    pca = PCA(n_components=2,random_state=42)
    combined_2d = pca.fit_transform(combined_feats)
    
    # 拆分坐标
    latent_2d = combined_2d[:-2]
    centers_2d = combined_2d[-2:]

    print("5. 渲染顶刊级别流形散点图...")
    # 使用典型的单栏图尺寸 (8x6 英寸)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # ------------------------------------------
    # 顶刊高级配色方案 (Nature / Seaborn Muted 风格)
    # ------------------------------------------
    colors = {
        3: '#4C72B0',  # Normal (沉稳蓝)
        0: '#55A868',  # Functional (生命绿)
        2: '#F5964F',  # Mixed (警示橙)
        1: '#C44E52'   # Mechanical (危险红)
    }
    
    labels_eng = {
        3: 'Normal',
        0: 'Functional',
        2: 'Mixed',
        1: 'Mechanical'
    }
    
    # 1. 绘制背景网格 (zorder=0，放在最底层)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

    # 2. 绘制散点 (调整透明度和白边，增加高级感)
    # 按特定顺序绘制，让重要的梗阻类盖在正常类上面
    plot_order = [3, 0, 2, 1] 
    for class_id in plot_order:
        mask = (labels == class_id)
        if mask.sum() > 0:
            ax.scatter(
                latent_2d[mask, 0], latent_2d[mask, 1],
                c=colors[class_id], 
                label=f'{labels_eng[class_id]} (n={mask.sum()})',
                s=90,          # 点的面积适中
                alpha=0.85,    # 85%不透明度，让重叠区域有层次感
                edgecolors='white', # 白色描边是非常经典的现代学术图表设计
                linewidths=0.8,
                zorder=2       # 确保点在网格上方
            )
    
    # 3. 绘制两个核心锚点 (超大、强对比色、加粗黑边框)
    ax.scatter(centers_2d[0, 0], centers_2d[0, 1], 
                c='#55A868', marker='*', s=800, 
                edgecolors='black', linewidths=1.5,
                label='Functional Center', zorder=10) # zorder=10 永远在最上层
                
    ax.scatter(centers_2d[1, 0], centers_2d[1, 1], 
                c='#C44E52', marker='*', s=800,
                edgecolors='black', linewidths=1.5,
                label='Mechanical Center', zorder=10)
    
    # 4. 坐标轴极简美化 (去除多余边框)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 加粗底部和左侧边框
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    
    # 美化刻度线 (朝外，加粗)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6, direction='out')
    
    # 5. 纯英文坐标轴标签
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold', fontfamily='Arial')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold', fontfamily='Arial')
    
    # 注：顶刊的正文图表通常不加 title，标题都写在底部的 Figure Caption 里。
    # 这里保留一个非常克制的标题
    ax.set_title('Latent Space Manifold Visualization', fontsize=16, fontweight='bold', fontfamily='Arial', pad=15)
    
    # 6. 图例美化 (带轻微透明度的白底边框)
    legend = ax.legend(loc='best', fontsize=11, frameon=True, edgecolor='black', framealpha=0.9)
    legend.get_frame().set_linewidth(1.0)
    
    plt.tight_layout()
    
    # 7. 同时保存为高分辨率 PNG 和无损 PDF (论文排版神器)
    pdf_path = 'latent_manifold_Publication1.pdf'
    png_path = 'latent_manifold_Publication1.png'
    plt.savefig(pdf_path, dpi=600, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, dpi=600, format='png', bbox_inches='tight')
    
    print(f"✓ 顶刊级别图表已生成！")
    print(f"  - 矢量图(用于论文插入): {pdf_path}")
    print(f"  - 预览图(用于快速查看): {png_path}")
    
    plt.show()

if __name__ == '__main__':
    print(f"使用设备: {DEVICE}")
    plot_latent_manifold()
