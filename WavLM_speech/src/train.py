import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import EmotionSegmentDataset
from src.model import DynamicEmotionModel

# ================= 配置 =================
BATCH_SIZE = 8          # 冻结模式下可以大一点
LR = 1e-4               # 标准学习率
EPOCHS = 30             
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVLM_PATH = "pretrained/wavlm.pt"
SAVE_DIR = "models_fusion" # 改个名字，跟之前的区分开
# =======================================

class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x_mean, y_mean = torch.mean(x, dim=0), torch.mean(y, dim=0)
        x_var, y_var = torch.var(x, dim=0, unbiased=False), torch.var(y, dim=0, unbiased=False)
        cov = torch.mean((x - x_mean) * (y - y_mean), dim=0)
        ccc = (2 * cov) / (x_var + y_var + (x_mean - y_mean) ** 2 + 1e-8)
        return 1.0 - torch.mean(ccc)

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"🚀 启动多层融合训练 (Layer-wise Fusion)...")

    # 1. 数据
    train_ds = EmotionSegmentDataset("data/train.csv", project_root=".")
    val_ds = EmotionSegmentDataset("data/val.csv", project_root=".")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. 模型
    model = DynamicEmotionModel(wavlm_path=WAVLM_PATH).to(DEVICE)
    
    # 🔴 关键：冻结 WavLM 内部参数，但允许训练 layer_weights
    # model.parameters() 会包含 layer_weights，所以优化器会自动处理
    for name, param in model.named_parameters():
        if "wavlm" in name:
            param.requires_grad = False
        else:
            # 包括 transformer, regressor 和 layer_weights
            param.requires_grad = True
    
    print("✅ WavLM 主体已冻结，仅训练层权重适配器和下游网络。")

    # 3. 优化器
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-2)
    mse_criterion = nn.MSELoss()
    ccc_criterion = CCCLoss()

    best_loss = float('inf')

    # 4. 循环
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for wav, labels in loop:
            wav, labels = wav.to(DEVICE), labels.to(DEVICE)
            
            preds = model(wav)
            loss = 0.5 * mse_criterion(preds, labels) + 0.5 * ccc_criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for wav, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                wav, labels = wav.to(DEVICE), labels.to(DEVICE)
                preds = model(wav)
                l_mse = mse_criterion(preds, labels)
                l_ccc = ccc_criterion(preds, labels)
                val_loss += (0.5 * l_mse + 0.5 * l_ccc).item()

        avg_val = val_loss / len(val_loader)
        avg_train = train_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}: Train={avg_train:.4f} | Val={avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"🎉 保存最佳模型 (Loss: {best_loss:.4f})")
            
            # 打印一下当前的层权重，看看模型更喜欢哪一层
            weights = torch.nn.functional.softmax(model.layer_weights, dim=0).detach().cpu().numpy()
            print(f"🔍 当前层权重分布: {[f'{w:.2f}' for w in weights]}")

if __name__ == "__main__":
    train()