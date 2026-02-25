import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CenterLoss(nn.Module):
    """中心损失：强迫纯机械和纯功能在 64 维潜空间中拉开距离并各自抱团"""
    def __init__(self, num_classes=2, feat_dim=64):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.centers=nn.Parameter(torch.randn(
            self.num_classes,
            self.feat_dim
        )).cuda(DEVICE)

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

def dice_loss(pred, target, smooth=1e-5):
    """基础的 Dice Loss (用于约束分割)"""
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)
    return 1.0 - ((2. * intersection + smooth) / (union + smooth)).mean()


# ==========================================
# 2. 潜空间与药代动力学诊断头
# ==========================================
class PharmacokineticLatentEncoder(nn.Module):
    """
    固定第 15 分钟(index 14)截断的潜空间编码器
    """
    def __init__(self, input_dim=26, latent_dim=64, num_classes=2):
        super(PharmacokineticLatentEncoder, self).__init__()
        self.injection_frame = 14  # 前14帧(0~13)为打药前，后12帧(14~25)为打药后
        
        # 打药前特征提取 (输入维度 14)
        self.pre_diuretic_net = nn.Sequential(
            nn.Linear(self.injection_frame, 16),
            nn.ReLU(inplace=True)
        )
        
        # 打药后特征提取 (输入维度 12)
        post_frames = input_dim - self.injection_frame
        self.post_diuretic_net = nn.Sequential(
            nn.Linear(post_frames, 16),
            nn.ReLU(inplace=True)
        )
        
        # 融合与潜空间投影
        self.latent_mapping = nn.Sequential(
            nn.Linear(16 + 16, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, latent_dim),
            nn.BatchNorm1d(latent_dim) # 输出 64 维潜空间特征
        )
        
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, tac_curve):
        pre_tac = tac_curve[:, :self.injection_frame]
        post_tac = tac_curve[:, self.injection_frame:]
        
        pre_feat = self.pre_diuretic_net(pre_tac)
        post_feat = self.post_diuretic_net(post_tac)
        
        combined_feat = torch.cat([pre_feat, post_feat], dim=1)
        latent_features = self.latent_mapping(combined_feat)
        logits = self.classifier(latent_features)
        
        return logits, latent_features

def train_anchors():
    # 1. 加载第一步提取的纯净特征
    tacs = np.load('extracted_tacs.npy')
    labels = np.load('extracted_labels.npy')
    
    # 2. 【核心过滤】：把混合型（比如 Label 2）剔除，我们只用极端病例来建立“两极锚点”
    pure_idx = (labels == 0) | (labels == 1)
    pure_tacs = torch.tensor(tacs[pure_idx], dtype=torch.float32).cuda()
    pure_labels = torch.tensor(labels[pure_idx], dtype=torch.long).cuda()

    # 3. 初始化网络与损失函数
    model = PharmacokineticLatentEncoder(input_dim=26, latent_dim=64, num_classes=2).cuda()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=2, feat_dim=64, use_gpu=True)
    
    # 双优化器
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_cent = optim.SGD(criterion_cent.parameters(), lr=0.5)

    model.train()
    print("开始构建潜空间两极流形...")
    for epoch in range(100): # TAC特征极容易训练，100个epoch足够
        optimizer_model.zero_grad()
        optimizer_cent.zero_grad()

        # 第 15 分钟截断提取已经在 model 的 forward 里自动完成了！
        logits, latent_feats = model(pure_tacs) 
        
        # 交叉熵分开类别，Center Loss 紧凑类内特征
        loss = criterion_cls(logits, pure_labels) + 0.1 * criterion_cent(latent_feats, pure_labels)
        
        loss.backward()
        optimizer_model.step()
        
        # 更新 CenterLoss 的中心
        for param in criterion_cent.parameters():
            param.grad.data *= (1. / 0.1)
        optimizer_cent.step()

    # 4. 【保存革命的果实】：网络权重 和 最重要的“潜空间中心坐标”
    torch.save(model.state_dict(), 'latent_encoder.pth')
    torch.save(criterion_cent.centers.data, 'manifold_centers.pth')
    print("潜空间锚点训练完毕，中心坐标已保存！")

# 运行: train_anchors()