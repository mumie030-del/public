import torch
import torch.nn as nn
import math

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


# 2. U-Net 模型 (纯净基础版)
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


# 3. ECA 注意力机制组件
class ECABlock(nn.Module):
    """
    ECA (Efficient Channel Attention) 模块
    使用 1D 卷积代替全连接层，保留通道间的时序/空间相邻关系，且参数极少。
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # 自动计算自适应的一维卷积核大小，根据通道数动态调整感受野
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 使用 1D 卷积处理通道维度，捕捉相邻通道（时间帧）的局部关系
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (Batch, Channels, H, W)
        y = self.avg_pool(x) # (Batch, Channels, 1, 1)
        
        # 维度变换以适应 Conv1d: (Batch, 1, Channels)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        
        y = self.sigmoid(y)
        # 残差乘法：给每个通道赋予不同的权重
        return x * y.expand_as(x) 


# 4. U-Net + ECA 融合模型 (修剪版，仅保留瓶颈层注意力)
class UnetWithECA(nn.Module):
    def __init__(self, in_channels=26, out_channels=1, use_dropout=False):
        super(UnetWithECA, self).__init__()
        
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

        # --- 瓶颈层 + 全局 ECA 注意力 ---
        self.bottleneck = DoubleConv(512, 1024, dropout=dropout_rate * 1.5)
        # 替换为 ECA 模块，1024 是传入的 channels
        self.eca_bottleneck = ECABlock(channels=1024)

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

        # 2. 瓶颈层 (集中进行 ECA 注意力筛选)
        bottleneck = self.bottleneck(p4)
        # 经过 ECA 模块
        bottleneck = self.eca_bottleneck(bottleneck)
        
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
    print("测试 U-Net 和 U-Net+ECA 模型")
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
    
    # 测试 U-Net + ECA
    print("\n--- U-Net + ECA ---")
    model_eca = UnetWithECA(in_channels=26, out_channels=1, use_dropout=False)
    output_eca = model_eca(input_data)
    print(f"输出特征维度: {output_eca.shape}")
    total_params_eca = sum(p.numel() for p in model_eca.parameters())
    print(f"总参数量: {total_params_eca:,}")
    print(f"增加参数量: {total_params_eca - total_params:,}")
    
    print("\n测试通过！✓")