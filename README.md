# 肾脏分割项目 - 多种模型架构对比

本项目实现了多种深度学习模型用于医学图像（肾脏）分割任务，包括基础 U-Net 以及多种改进版本。

## 📁 项目结构

```
.
├── unet/          # 基础 U-Net 模型
├── transformer/   # Transformer 改进版
├── CBAM/          # 带 CBAM 注意力机制的 U-Net
├── LTAE/          # 带 LTAE (Lightweight Temporal Attention) 的 U-Net
└── LSTM/          # 带 ConvLSTM 的 U-Net
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- PyTorch
- CUDA (推荐)

### 训练模型

```bash
cd unet  # 或其他模型目录
python3 train.py
```

### 测试模型

```bash
python3 test.py
```

### 绘制训练曲线

```bash
python3 plot.py --checkpoint_dir checkpoints
```

## 📊 模型说明

### 1. U-Net (基础版)
- 经典的 U-Net 架构
- 编码器-解码器结构
- 跳跃连接

### 2. Transformer
- 基于 Transformer 的改进
- 更好的长距离依赖建模

### 3. CBAM
- 结合通道注意力 (Channel Attention)
- 空间注意力 (Spatial Attention)
- 双重注意力机制

### 4. LTAE
- Lightweight Temporal Attention Encoder
- 针对多时间帧数据的注意力机制
- 轻量级设计，避免内存溢出

### 5. LSTM
- ConvLSTM 瓶颈层
- 时间序列特征建模
- 流体特征融合

## 📈 评估指标

- **Dice Score**: 分割重叠度
- **IoU**: Intersection over Union
- **Loss**: BCE + Dice Loss 组合

## 📝 数据格式

- 输入: 26 通道时间序列图像 (256×256)
- 输出: 单通道二值掩码 (256×256)

## 🔧 配置

主要配置参数在 `train.py` 中：
- `BATCH_SIZE`: 批次大小
- `LEARNING_RATE`: 学习率
- `NUM_EPOCHS`: 训练轮数
- `DATA_DIR`: 数据目录

## 📄 许可证

本项目仅供学习和研究使用。

