import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from datasets import Data3Dataset
# from module import NestedUNet
from tqdm import tqdm
import albumentations as A
from module import OptimizedNestedUNet


# ==================== 配置参数 ====================
DATA_DIR = '../data3'
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = './checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==================== 数据增强配置 ====================
train_transform = A.Compose([
    A.Rotate(limit=15, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ElasticTransform(p=0.3, alpha=1, sigma=50),
])

val_transform = None

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
        return 1 - dice.mean()

class BceDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, logits, targets):
        return 0.5 * self.bce(logits, targets) + 0.5 * self.dice(logits, targets)

criterion = BceDiceLoss()

if __name__ == '__main__':
    print(f"使用设备: {DEVICE}")
    
    # 1. 加载数据集
    full_dataset = Data3Dataset(data_root=DATA_DIR, target_size=(256, 256), num_channels=26)
    print(f"总数据集大小: {len(full_dataset)}")
    
    # 2. 划分训练集和验证集
    train_size = int(0.75 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size]
    )
    
    # 3. 创建数据集
    train_dataset = Data3Dataset(
        data_root=DATA_DIR, 
        target_size=(256, 256), 
        num_channels=26,
        transform=train_transform
    )
    val_dataset = Data3Dataset(
        data_root=DATA_DIR, 
        target_size=(256, 256), 
        num_channels=26,
        transform=val_transform
    )
    
    train_dataset = Subset(train_dataset, train_indices.indices)
    val_dataset = Subset(val_dataset, val_indices.indices)
    
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")
    
    # 4. 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 5. 初始化模型
    # NestedUNet 的构造参数为 (in_channels, out_channels, deep_supervision)
    model = OptimizedNestedUNet(in_channels=26, out_channels=1, deep_supervision=True, dropout_rate=0.05).to(DEVICE)
    print(f"模型: NestedUNet (UNet++)")
    print(f"输入通道数: 26, 输出通道数: 1")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")
    
    sample_img, sample_mask = next(iter(train_loader))
    print(f"图像形状: {sample_img.shape}, 掩码形状: {sample_mask.shape}")
    
    # 6. 优化器和调度器
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE,
        weight_decay=1e-5
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 7. 早停机制
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    # 8. 训练循环
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Train]')
        
        for batch_idx, (images, masks) in enumerate(train_bar):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(images)
            # 支持 deep supervision：如果模型返回多个尺度的预测，分别算 loss 后取平均
            if isinstance(outputs, list):
                loss = 0
                for out in outputs:
                    loss = loss + criterion(out, masks)
                loss = loss / len(outputs)
            else:
                loss = criterion(outputs, masks)
            
            if epoch % 10 == 0 and batch_idx == 0:
                # Debug 可视化：如果是 deep supervision，取最后一个输出做统计
                debug_out = outputs[-1] if isinstance(outputs, list) else outputs
                probs = torch.sigmoid(debug_out)
                print(f" [Debug] Max Prob: {probs.max().item():.4f}, Mean Prob: {probs.mean().item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS} [Val]')
            for images, masks in val_bar:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                outputs = model(images)
                if isinstance(outputs, list):
                    loss = 0
                    for out in outputs:
                        loss = loss + criterion(out, masks)
                    loss = loss / len(outputs)
                else:
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_bar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f'\nEpoch [{epoch+1}/{NUM_EPOCHS}]')
        print(f'  训练损失: {avg_train_loss:.4f}')
        print(f'  验证损失: {avg_val_loss:.4f}')
        print(f'  损失差距: {abs(avg_train_loss - avg_val_loss):.4f}')
        print(f'  学习率: {current_lr:.6f}')
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f'  ✓ 保存最佳模型到 {checkpoint_path}')
        else:
            patience_counter += 1
            print(f'  验证损失未改善 ({patience_counter}/{patience})')
            
            if patience_counter >= patience:
                print(f'\n早停触发！验证损失已经 {patience} 个 epoch 没有改善。')
                break
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f'  ✓ 保存检查点到 {checkpoint_path}')
        
        print('-' * 60)
    
    print('\n训练完成！')
    print(f'最佳验证损失: {best_val_loss:.4f}')
