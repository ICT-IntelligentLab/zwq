import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 显式导入
from src.dataset import EmotionSegmentDataset
from src.model import DynamicEmotionModel

# ================= 微调配置参数 (关键修改) =================
# 1. 减小 Batch Size 防止显存爆炸 (如果显存够大，可尝试 4)
BATCH_SIZE = 2          

# 2. 保持极低学习率 (微调黄金法则)
LR = 1e-5               

# 3. 增加轮数，微调需要耐心
EPOCHS = 30             

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVLM_PATH = "pretrained/wavlm.pt" 

# 4. 修改保存路径，避免覆盖之前的模型
SAVE_DIR = "models_finetune"     
# ========================================================

# CCC Loss 定义 (保持不变)
class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        x_mean = torch.mean(x, dim=0)
        y_mean = torch.mean(y, dim=0)
        x_var = torch.var(x, dim=0, unbiased=False)
        y_var = torch.var(y, dim=0, unbiased=False)
        cov = torch.mean((x - x_mean) * (y - y_mean), dim=0)
        numerator = 2 * cov
        denominator = x_var + y_var + (x_mean - y_mean) ** 2 + 1e-8
        ccc = numerator / denominator
        loss = 1.0 - torch.mean(ccc)
        return loss

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("🚀 启动全量微调 (Fine-tuning) 模式...")
    print(f"配置: Batch={BATCH_SIZE}, LR={LR}, Epochs={EPOCHS}")

    # 1. 加载数据
    print("正在加载全量数据集...")
    train_ds = EmotionSegmentDataset("data/train.csv", project_root=".")
    val_ds = EmotionSegmentDataset("data/val.csv", project_root=".")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"训练集: {len(train_ds)} | 验证集: {len(val_ds)}")

    # 2. 初始化模型
    print(f"正在加载模型 (Device: {DEVICE})...")
    model = DynamicEmotionModel(wavlm_path=WAVLM_PATH).to(DEVICE)
    
    # =======================================================
    # 🔴 核心修改：解冻 WavLM (Unfreeze)
    # =======================================================
    print("🔓 已解冻 WavLM 参数，开始进行端到端微调...")
    for param in model.wavlm.model.parameters():
        param.requires_grad = True  # 打开梯度开关
    # =======================================================

    # 3. 定义优化器
    # 注意：微调时，Weight Decay 非常重要，防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    mse_criterion = nn.MSELoss()
    ccc_criterion = CCCLoss()

    best_loss = float('inf')

    # 4. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # 这里的 tqdm 会变慢，因为反向传播要计算 WavLM 的 9000万个参数
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Finetune]")
        
        for wav, labels in loop:
            wav = wav.to(DEVICE)       
            labels = labels.to(DEVICE) 

            # 前向传播
            preds = model(wav)         
            
            # 混合 Loss
            loss_mse = mse_criterion(preds, labels)
            loss_ccc = ccc_criterion(preds, labels)
            loss = 0.5 * loss_mse + 0.5 * loss_ccc

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # 5. 验证循环
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for wav, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                wav = wav.to(DEVICE)
                labels = labels.to(DEVICE)
                
                preds = model(wav)
                
                l_mse = mse_criterion(preds, labels)
                l_ccc = ccc_criterion(preds, labels)
                loss = 0.5 * l_mse + 0.5 * l_ccc
                
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} 结束: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # 6. 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = os.path.join(SAVE_DIR, "best_model_finetuned.pth")
            torch.save(model.state_dict(), save_path)
            print(f"🎉 发现更优模型 (Loss: {best_loss:.4f})，已保存到: {save_path}")
        
        print("-" * 50)

if __name__ == "__main__":
    train()