import torch
import torch.nn as nn

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


class ConvLSTMCell(nn.Module):
    """
    源自气象雷达预测的 ConvLSTM 单元
    结合了卷积的空间特征提取能力 和 LSTM 的时间记忆能力
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # 核心：用卷积替代传统 LSTM 中的全连接矩阵相乘
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim, # 对应 LSTM 的 4 个门
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # 将当前输入和上一时刻的隐藏状态在通道维度拼接
        combined = torch.cat([input_tensor, h_cur], dim=1)  
        combined_conv = self.conv(combined)
        
        # 分离出 LSTM 的 4 个门：输入门、遗忘门、输出门、细胞状态更新
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 更新记忆细胞和隐藏状态 (流体力学时间记忆传递)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class BottleneckConvLSTM(nn.Module):
    """
    专为 U-Net 瓶颈层设计的流体特征融合模块
    """
    def __init__(self, channels=1024):
        super(BottleneckConvLSTM, self).__init__()
        # 在瓶颈层，我们将特征视为长度为 4 的时间步 (一种巧妙的通道折叠降维法，防止过拟合)
        # 比如 1024 通道，我们可以折叠成 4 个 256 通道的"时间切片"
        self.time_steps = 4 
        self.split_channels = channels // self.time_steps
        
        self.convlstm = ConvLSTMCell(input_dim=self.split_channels, 
                                     hidden_dim=self.split_channels, 
                                     kernel_size=3)
        
        self.out_conv = nn.Conv2d(self.split_channels, channels, kernel_size=1)

    def forward(self, x):
        # x 形状在瓶颈层通常是 (B, 1024, 16, 16)
        B, C, H, W = x.shape
        
        # 巧妙切片：(B, 4, 256, 16, 16)，将极深层的通道转化为"伪时间步"
        # 模拟造影剂在极度抽象空间中的演变
        x_seq = x.view(B, self.time_steps, self.split_channels, H, W)
        
        # 初始化隐藏状态
        h_t = torch.zeros(B, self.split_channels, H, W).to(x.device)
        c_t = torch.zeros(B, self.split_channels, H, W).to(x.device)
        
        # 像看电影一样，一帧一帧过 ConvLSTM
        for t in range(self.time_steps):
            h_t, c_t = self.convlstm(x_seq[:, t, :, :, :], (h_t, c_t))
            
        # h_t 包含了浓缩后的所有流体时空记忆
        # 通过 1x1 卷积恢复回 1024 通道，无缝对接 U-Net 的解码器
        out = self.out_conv(h_t)
        
        # 残差连接，保证 U-Net 的下限
        return x + out


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
        # 先用卷积将通道从 512 提升到 1024，再在 1024 通道上做 ConvLSTM
        self.bottleneck = DoubleConv(512, 1024)
        self.bottleneck1 = BottleneckConvLSTM(channels=1024)
        
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

        # 2. 瓶颈层：先卷积到 1024 通道，再做 ConvLSTM
        bottleneck = self.bottleneck(p4)
        bottleneck = self.bottleneck1(bottleneck) 
        
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


# ====== 如何在您的 U-Net 中使用 ======
# 在 U-Net 的 __init__ 中：
# self.bottleneck_lstm = BottleneckConvLSTM(channels=1024)
#
# 在 U-Net 的 forward 中：
# bottleneck = self.bottleneck(p4)
# bottleneck = self.bottleneck_lstm(bottleneck) # 仅在此处插入
# u1 = self.up1(bottleneck)