import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 路径修复
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from models.wavlm_wrapper import WavLMWrapper

class DynamicEmotionModel(nn.Module):
    def __init__(self, wavlm_path, hidden_dim=768, num_layers=1, output_dim=3):
        super().__init__()
        
        # 1. 加载 WavLM
        self.wavlm = WavLMWrapper(checkpoint_path=wavlm_path)
        
        # 🔴 核心修改：新增层权重参数
        # 针对 WavLM Base 的 13 层 (1 Embedding + 12 Transformer Layers)
        # 初始化为 0，经过 Softmax 后初始权重平均
        self.layer_weights = nn.Parameter(torch.zeros(13)) 

        # 2. Transformer Encoder (保持之前的优化配置)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=8, 
            dim_feedforward=1024, 
            dropout=0.4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. 回归头 (保持不变)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, output_dim)
        )

    def forward(self, wav):
        # 1. 提取所有层特征 [13, Batch, Time, 768]
        # 注意：WavLM 处于冻结状态，不计算梯度
        with torch.no_grad():
            all_layers = self.wavlm.extract_all_layers(wav)
        
        # 2. 加权融合 (Weighted Sum)
        # 计算 Softmax 权重，保证权重之和为 1
        # weights shape: [13]
        weights = F.softmax(self.layer_weights, dim=0)
        
        # 调整维度以便广播乘法: [13, 1, 1, 1]
        weights = weights.view(-1, 1, 1, 1)
        
        # 加权求和: sum([13, B, T, D] * [13, 1, 1, 1]) -> [B, T, D]
        # 这一步是可导的，layer_weights 会被训练更新
        x = (all_layers * weights).sum(dim=0)
        
        # 3. 时序建模
        x = self.transformer(x)
        
        # 4. 聚合 & 预测
        x = x.mean(dim=1)
        out = self.regressor(x)
        
        return out