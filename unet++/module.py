import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 注意力模块
    作用：自动学习 26 个时间通道（帧）的重要性权重
    """
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 确保降维后至少有 1 个通道
        mid_channel = max(1, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionResBlock(nn.Module):
    """
    终极版细胞模块：ResBlock + SE Attention + Spatial Dropout
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(AttentionResBlock, self).__init__()
        
        # 主干卷积
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 通道注意力
        self.se = SEBlock(out_channels)
        
        # 空间失活 (防过拟合神器)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

        # 残差捷径 (Shortcut)：如果输入输出通道不同，用 1x1 卷积对齐
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)      # 保留原始输入
        out = self.conv(x)          # 提取特征
        out = self.se(out)          # 评估通道(时间)重要性
        out = out + res             # 残差相加
        out = self.relu(out)        # 激活
        out = self.drop(out)        # 随机失活
        return out


class OptimizedNestedUNet(nn.Module):
    """
    优化版 UNet++ (结合 ResBlock 与 通道注意力)
    """
    def __init__(self, in_channels=26, out_channels=1, deep_supervision=True, dropout_rate=0.1):
        super(OptimizedNestedUNet, self).__init__()

        # 通道数配置 (可根据显存缩减，例如 [16, 32, 64, 128, 256])
        nb_filter = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # ==================== L0 层 ====================
        self.conv0_0 = AttentionResBlock(in_channels, nb_filter[0], dropout_rate)
        self.conv1_0 = AttentionResBlock(nb_filter[0], nb_filter[1], dropout_rate)
        self.conv2_0 = AttentionResBlock(nb_filter[1], nb_filter[2], dropout_rate)
        self.conv3_0 = AttentionResBlock(nb_filter[2], nb_filter[3], dropout_rate)
        self.conv4_0 = AttentionResBlock(nb_filter[3], nb_filter[4], dropout_rate)

        # ==================== L1 层 ====================
        self.conv0_1 = AttentionResBlock(nb_filter[0] + nb_filter[1], nb_filter[0], dropout_rate)
        self.conv1_1 = AttentionResBlock(nb_filter[1] + nb_filter[2], nb_filter[1], dropout_rate)
        self.conv2_1 = AttentionResBlock(nb_filter[2] + nb_filter[3], nb_filter[2], dropout_rate)
        self.conv3_1 = AttentionResBlock(nb_filter[3] + nb_filter[4], nb_filter[3], dropout_rate)

        # ==================== L2 层 ====================
        self.conv0_2 = AttentionResBlock(nb_filter[0]*2 + nb_filter[1], nb_filter[0], dropout_rate)
        self.conv1_2 = AttentionResBlock(nb_filter[1]*2 + nb_filter[2], nb_filter[1], dropout_rate)
        self.conv2_2 = AttentionResBlock(nb_filter[2]*2 + nb_filter[3], nb_filter[2], dropout_rate)

        # ==================== L3 层 ====================
        self.conv0_3 = AttentionResBlock(nb_filter[0]*3 + nb_filter[1], nb_filter[0], dropout_rate)
        self.conv1_3 = AttentionResBlock(nb_filter[1]*3 + nb_filter[2], nb_filter[1], dropout_rate)

        # ==================== L4 层 ====================
        self.conv0_4 = AttentionResBlock(nb_filter[0]*4 + nb_filter[1], nb_filter[0], dropout_rate)

        # ==================== 输出头 ====================
        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channels, kernel_size=1)

    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        if self.deep_supervision:
            return [self.final1(x0_1), self.final2(x0_2), self.final3(x0_3), self.final4(x0_4)]
        else:
            return self.final(x0_4)

if __name__ == '__main__':
    # 自测代码
    model = OptimizedNestedUNet(in_channels=26, out_channels=1, dropout_rate=0.1)
    x = torch.randn(2, 26, 256, 256)
    y = model(x)
    print(f"输入: {x.shape} -> 输出: {y.shape}")