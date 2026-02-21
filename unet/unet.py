import torch
import torch.nn as nn

# 1. 封装一个“双卷积块”，这是 U-Net 的基本单元
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            # 第一次卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), # 加上BN层，训练更稳定
            nn.ReLU(inplace=True),
            # 第二次卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


# 2. 基础的纯净版 U-Net 模型
class Unet(nn.Module):
    def __init__(self, in_channels=26, out_channels=1):
        super(Unet, self).__init__()

        # --- 编码器 (下采样路径) ---
        self.conv1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = DoubleConv(64, 128) 
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # --- 瓶颈层 (最底层) ---
        self.bottleneck = DoubleConv(512, 1024)
        
        # --- 解码器 (上采样路径) ---
        # Up 1
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(1024, 512)

        # Up 2
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(512, 256) 

        # Up 3
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(256, 128)

        # Up 4
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
        
        # 3. 解码器
        u1 = self.up1(bottleneck)
        cat1 = torch.cat([c4, u1], dim=1) 
        x = self.conv_up1(cat1)

        # Block 2
        u2 = self.up2(x)
        cat2 = torch.cat([c3, u2], dim=1)
        x = self.conv_up2(cat2)

        # Block 3
        u3 = self.up3(x)
        cat3 = torch.cat([c2, u3], dim=1)
        x = self.conv_up3(cat3)

        # Block 4
        u4 = self.up4(x)
        cat4 = torch.cat([c1, u4], dim=1)
        x = self.conv_up4(cat4)

        # 输出
        return self.out_conv(x)


if __name__ == '__main__':
    # 模拟输入你的数据：Batch=1, Channels=26, H=256, W=256
    input_data = torch.randn([1, 26, 256, 256])
    print(f"输入特征维度: {input_data.shape}")
    
    # 实例化最基础的 U-Net 模型
    model = Unet(in_channels=26, out_channels=1)
    
    # 跑一次前向传播
    output = model(input_data)
    
    print(f"输出特征维度: {output.shape}") 
    print("模型恢复为纯净版 U-Net。")