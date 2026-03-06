import torch
import numpy as np
from scipy.stats import pearsonr
import os
import sys

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
if project_root not in sys.path:
    sys.path.append(project_root)

from src.dataset import EmotionSegmentDataset
from src.model import DynamicEmotionModel
from torch.utils.data import DataLoader
from tqdm import tqdm

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVLM_PATH = "pretrained/wavlm.pt"
MODEL_PATH = "models_fusion/best_model.pth"

VAL_CSV = "data/val.csv"
# =======================================

def denormalize(val):
    return val * 4.0 + 1.0
def ccc_score(x, y):
    """计算一致性相关系数 CCC"""
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x)
    var_y = np.var(y)
    cov_xy = np.mean((x - mean_x) * (y - mean_y))
    return (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y)**2)

def evaluate():
    print("🚀 开始全量评估...", flush=True)
    
    # 1. 数据
    val_ds = EmotionSegmentDataset(VAL_CSV, project_root=".")
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    
    # 2. 模型
    model = DynamicEmotionModel(wavlm_path=WAVLM_PATH).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    # 3. 收集所有预测值和真实值
    all_preds = []
    all_labels = []
    
    print("正在推理验证集...")
    with torch.no_grad():
        for wav, labels in tqdm(val_loader):
            wav = wav.to(DEVICE)
            
            # 预测
            preds = model(wav).cpu().numpy()
            labels = labels.numpy()
            
            all_preds.append(preds)
            all_labels.append(labels)
            
    # 拼接
    all_preds = np.concatenate(all_preds, axis=0) # [N, 3]
    all_labels = np.concatenate(all_labels, axis=0) # [N, 3]
    
    # 反归一化
    all_preds = denormalize(all_preds)
    all_labels = denormalize(all_labels)
    
    # 4. 计算指标
    dims = ["Arousal (激活度)", "Valence (效价)", "Dominance (支配度)"]
    print("\n" + "="*50)
    print(f"{'维度':<20} | {'CCC (一致性)':<12} | {'PCC (相关性)':<12} | {'RMSE (误差)':<12}")
    print("-" * 60)
    
    avg_ccc = 0
    for i in range(3):
        true_vals = all_labels[:, i]
        pred_vals = all_preds[:, i]
        
        ccc = ccc_score(true_vals, pred_vals)
        pcc, _ = pearsonr(true_vals, pred_vals)
        rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
        
        avg_ccc += ccc
        print(f"{dims[i]:<20} | {ccc:.4f}       | {pcc:.4f}       | {rmse:.4f}")
        
    print("-" * 60)
    print(f"平均 CCC: {avg_ccc / 3:.4f}")
    print("="*50)

if __name__ == "__main__":
    evaluate()