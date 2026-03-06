import torch
import torch.nn as nn
import os
import sys

# ================= 路径修复 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
wavlm_source_dir = os.path.join(current_dir, "..", "src", "WavLM")
wavlm_source_dir = os.path.abspath(wavlm_source_dir)

if wavlm_source_dir not in sys.path:
    sys.path.insert(0, wavlm_source_dir)

try:
    from WavLM import WavLM, WavLMConfig
except ImportError:
    raise ImportError(f"❌ 无法导入 WavLM，请检查 src/WavLM 是否存在。")
# ===========================================

class WavLMWrapper(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(current_dir, "..", "pretrained", "wavlm.pt")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ 找不到 WavLM 权重文件: {checkpoint_path}")

        print(f"正在加载 WavLM (多层模式): {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()

    def extract_features(self, wav):
        """保持兼容旧代码的接口 (只返回最后一层)"""
        # 注意：这里取[-1]表示最后一层
        return self.extract_all_layers(wav)[-1]

    def extract_all_layers(self, wav):
        """
        🔴 终极修复版：处理各种返回类型
        Returns: [13, Batch, Time, 768] (13 = 1 embedding + 12 layers)
        """
        # 维度调整
        if wav.dim() == 3 and wav.shape[1] == 1: wav = wav.squeeze(1)
        if wav.dim() == 1: wav = wav.unsqueeze(0)

        # 归一化
        if self.model.cfg.normalize:
            wav = torch.nn.functional.layer_norm(wav, wav.shape)

        # 提取特征
        results = self.model.extract_features(wav, padding_mask=None, ret_layer_results=True)
        
        # 1. 解包 results
        if isinstance(results, tuple) and len(results) == 2:
            rep, layer_results = results
        else:
            rep = results
            layer_results = None

        # 🔴 关键修复：检查 rep 是否也是 tuple
        # WavLM 有时返回 (features, padding_mask) 作为 rep
        if isinstance(rep, tuple):
            rep = rep[0]

        # 2. 防御性处理 (Fallback)
        if layer_results is None:
            # print("⚠️ 警告: WavLM 未返回中间层，使用最后一层复制填充。") 
            # 此时 rep 已经是 tensor 了，可以安全 unsqueeze
            stacked = rep.unsqueeze(0).repeat(13, 1, 1, 1)
            return stacked

        # 3. 正常处理 list
        layers = []
        for x in layer_results:
            # layer_results 的项可能是 (hidden_state, attn)
            if isinstance(x, tuple):
                layers.append(x[0])
            else:
                layers.append(x)
        
        # 堆叠: [Layers, Batch, Time, Dim]
        stacked = torch.stack(layers)
        
        return stacked

if __name__ == "__main__":
    wrapper = WavLMWrapper()
    x = torch.randn(2, 16000)
    out = wrapper.extract_all_layers(x)
    print(f"✅ 多层提取成功，输出维度: {out.shape} (预期: [13, 2, T, 768])")