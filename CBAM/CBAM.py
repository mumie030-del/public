import torch
import torch.nn as nn

# 1. 封装一个"双卷积块"，这是 U-Net 的基本单元
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(DoubleConv, self).__init__()
        layers = [
            # 第一次卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # 加上BN层，训练更稳定
            nn.ReLU(inplace=True),
        ]
        
        # 可选的 Dropout 层（防止过拟合）
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        
        layers.extend([
            # 第二次卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


# 2. U-Net 模型
class Unet(nn.Module):
    def __init__(self, in_channels=26, out_channels=1, use_dropout=False):
        super(Unet, self).__init__()
        
        # Dropout 概率（如果启用）
        dropout_rate = 0.2 if use_dropout else 0.0

        # --- 编码器 (下采样路径) ---
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = DoubleConv(64, 128, dropout=dropout_rate) 
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = DoubleConv(128, 256, dropout=dropout_rate)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = DoubleConv(256, 512, dropout=dropout_rate)
        self.pool4 = nn.MaxPool2d(2)

        # --- 瓶颈层 (最底层) ---
        self.bottleneck = DoubleConv(512, 1024, dropout=dropout_rate * 1.5)

        # --- 解码器 (上采样路径) ---
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512, dropout=dropout_rate)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256, dropout=dropout_rate) 

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128, dropout=dropout_rate)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64) 

        # --- 输出层 ---
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1) 

    def forward(self, x):
        # 1. 编码器
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # 2. 瓶颈层
        bottleneck = self.bottleneck(p4)
        
        # 3. 解码器（带跳跃连接）
        u1 = self.up1(bottleneck)
        cat1 = torch.cat([c4, u1], dim=1) 
        x = self.conv_up1(cat1)

        u2 = self.up2(x)
        cat2 = torch.cat([c3, u2], dim=1)
        x = self.conv_up2(cat2)

        u3 = self.up3(x)
        cat3 = torch.cat([c2, u3], dim=1)
        x = self.conv_up3(cat3)

        u4 = self.up4(x)
        cat4 = torch.cat([c1, u4], dim=1)
        x = self.conv_up4(cat4)

        # 4. 输出
        return self.out_conv(x)


# 3. CBAM 注意力机制组件
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享感知层
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 压缩通道，只看空间
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度做平均和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, channel, ratio=16):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(channel, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        identity = x                # 保存原始特征
        out = x * self.ca(x)        # 通道注意力
        out = out * self.sa(out)    # 空间注意力
        return identity + out       # 【关键修改】：残差相加！(原始特征 + 注意力特征)

# 4. U-Net + CBAM 融合模型
# 4. 修剪版 U-Net + CBAM 融合模型
class UnetWithCBAM(nn.Module):
    def __init__(self, in_channels=26, out_channels=1, use_dropout=False, cbam_ratio=16):
        super(UnetWithCBAM, self).__init__()
        
        dropout_rate = 0.2 if use_dropout else 0.0

        # --- 编码器 (下采样路径) - 纯净版，无注意力扰动 ---
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = DoubleConv(64, 128, dropout=dropout_rate)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = DoubleConv(128, 256, dropout=dropout_rate)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = DoubleConv(256, 512, dropout=dropout_rate)
        self.pool4 = nn.MaxPool2d(2)

        # --- 瓶颈层 + 全局 CBAM ---
        self.bottleneck = DoubleConv(512, 1024, dropout=dropout_rate * 1.5)
        # 全局唯一一个注意力模块，负责在高度抽象的特征上挑选关键帧和定位肾脏
        self.cbam_bottleneck = CBAM(1024, ratio=cbam_ratio)

        # --- 解码器 (上采样路径) ---
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512, dropout=dropout_rate)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256, dropout=dropout_rate)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128, dropout=dropout_rate)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up4 = DoubleConv(128, 64)

        # --- 输出层 ---
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 1. 编码器 (老老实实提取特征)
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)

        # 2. 瓶颈层 (集中进行注意力筛选)
        bottleneck = self.bottleneck(p4)
        bottleneck = self.cbam_bottleneck(bottleneck)
        
        # 3. 解码器（带跳跃连接）
        u1 = self.up1(bottleneck)
        cat1 = torch.cat([c4, u1], dim=1)
        x = self.conv_up1(cat1)

        u2 = self.up2(x)
        cat2 = torch.cat([c3, u2], dim=1)
        x = self.conv_up2(cat2)

        u3 = self.up3(x)
        cat3 = torch.cat([c2, u3], dim=1)
        x = self.conv_up3(cat3)

        u4 = self.up4(x)
        cat4 = torch.cat([c1, u4], dim=1)
        x = self.conv_up4(cat4)

        # 4. 输出
        return self.out_conv(x)


if __name__ == '__main__':
    # 测试模型
    print("="*50)
    print("测试 U-Net 和 U-Net+CBAM 模型")
    print("="*50)
    
    # 模拟输入数据：Batch=1, Channels=26, H=256, W=256
    input_data = torch.randn([1, 26, 256, 256])
    print(f"输入特征维度: {input_data.shape}")
    
    # 测试原始 U-Net
    print("\n--- 原始 U-Net ---")
    model = Unet(in_channels=26, out_channels=1, use_dropout=False)
    output = model(input_data)
    print(f"输出特征维度: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    # 测试 U-Net + CBAM
    print("\n--- U-Net + CBAM ---")
    model_cbam = UnetWithCBAM(in_channels=26, out_channels=1, use_dropout=False, cbam_ratio=16)
    output_cbam = model_cbam(input_data)
    print(f"输出特征维度: {output_cbam.shape}")
    total_params_cbam = sum(p.numel() for p in model_cbam.parameters())
    print(f"总参数量: {total_params_cbam:,}")
    print(f"增加参数量: {total_params_cbam - total_params:,}")
    
    print("\n测试通过！✓")
