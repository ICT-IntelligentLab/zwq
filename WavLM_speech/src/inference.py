import torch
import os
import random
from dataset import EmotionSegmentDataset
from model import DynamicEmotionModel

# ================= 配置 =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WAVLM_PATH = "pretrained/wavlm.pt"
MODEL_PATH = "models/best_model.pth"
# =======================================

def denormalize(val):
    """把 0-1 的预测值还原回 1-5"""
    return val * 4.0 + 1.0

def inference():
    # 1. 加载验证集 (找几个没见过的数据测测)
    val_ds = EmotionSegmentDataset("data/val.csv", project_root=".")
    
    # 2. 加载模型
    print(f"正在加载模型: {MODEL_PATH} ...")
    model = DynamicEmotionModel(wavlm_path=WAVLM_PATH).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    print("-" * 60)
    print(f"{'真实值 (A/V/D)':<20} | {'预测值 (A/V/D)':<20} | {'误差':<20}")
    print("-" * 60)

    # 3. 随机抽取 10 个样本进行测试
    indices = random.sample(range(len(val_ds)), 10)
    
    with torch.no_grad():
        for idx in indices:
            wav, label = val_ds[idx]
            wav = wav.unsqueeze(0).to(DEVICE) # [1, T]
            
            # 预测 (输出是 0-1)
            pred = model(wav).cpu().squeeze(0) # [3]
            
            # 反归一化 (变回 1-5)
            true_score = denormalize(label)
            pred_score = denormalize(pred)
            
            # 格式化输出
            t_str = f"{true_score[0]:.1f}, {true_score[1]:.1f}, {true_score[2]:.1f}"
            p_str = f"{pred_score[0]:.1f}, {pred_score[1]:.1f}, {pred_score[2]:.1f}"
            
            # 计算平均误差
            err = (true_score - pred_score).abs().mean().item()
            
            print(f"{t_str:<20} | {p_str:<20} | {err:.2f}")

    print("-" * 60)

if __name__ == "__main__":
    inference()