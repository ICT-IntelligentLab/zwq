import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class EmotionSegmentDataset(Dataset):
    def __init__(self, csv_path, project_root=".", sample_rate=16000, max_duration=6.0):
        self.df = pd.read_csv(csv_path)
        self.project_root = project_root
        self.sample_rate = sample_rate
        self.max_length = int(max_duration * sample_rate)

        # 自动识别音频路径列名
        possible_keys = ['wav_path', 'wav', 'path', 'filename', 'file_path']
        self.path_col = None
        for key in possible_keys:
            if key in self.df.columns:
                self.path_col = key
                break
        
        if self.path_col is None:
            raise KeyError(f"❌ CSV 中找不到音频路径列! 请确保列名是以下之一: {possible_keys}")
        
        self.resample_transforms = {} 

    def __len__(self):
        return len(self.df)

    def _get_resampler(self, orig_freq):
        if orig_freq not in self.resample_transforms:
            self.resample_transforms[orig_freq] = torchaudio.transforms.Resample(
                orig_freq=orig_freq, new_freq=self.sample_rate
            )
        return self.resample_transforms[orig_freq]

    def __getitem__(self, idx):
        # ===============================================================
        # 🔴 全局 Try-Except 保护：防止任何一条坏数据搞崩整个训练
        # ===============================================================
        try:
            row = self.df.iloc[idx]
            
            # 1. 路径处理
            wav_relative_path = row[self.path_col]
            full_wav_path = os.path.join(self.project_root, wav_relative_path)

            # 2. 加载音频
            if not os.path.exists(full_wav_path):
                raise FileNotFoundError(f"文件不存在: {full_wav_path}")

            waveform, sr = torchaudio.load(full_wav_path)

            # 3. 重采样
            if sr != self.sample_rate:
                resampler = self._get_resampler(sr)
                waveform = resampler(waveform)

            # 4. 转单声道
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 5. 切片
            start_sec = float(row['start'])
            end_sec = float(row['end'])
            
            # 检查 NaN
            if pd.isna(start_sec) or pd.isna(end_sec):
                raise ValueError("时间戳包含 NaN")

            start_frame = int(start_sec * self.sample_rate)
            end_frame = int(end_sec * self.sample_rate)

            if end_frame > waveform.shape[1]:
                end_frame = waveform.shape[1]
            
            if start_frame >= end_frame:
                # 如果切片无效，返回静音
                cropped_wave = torch.zeros(1, 16000)
            else:
                cropped_wave = waveform[:, start_frame:end_frame]

            # 6. 统一长度 (Pad/Truncate)
            current_len = cropped_wave.shape[1]
            if current_len > self.max_length:
                cropped_wave = cropped_wave[:, :self.max_length]
            elif current_len < self.max_length:
                pad_amount = self.max_length - current_len
                cropped_wave = torch.nn.functional.pad(cropped_wave, (0, pad_amount))

            # 7. 标签处理
            act = float(row['activation'])
            val = float(row['valence'])
            dom = float(row['dominance'])

            # 检查标签是否为 NaN
            if pd.isna(act) or pd.isna(val) or pd.isna(dom):
                raise ValueError(f"标签包含 NaN: {act}, {val}, {dom}")

            # 归一化 (1-5 -> 0-1)
            raw_labels = [act, val, dom]
            norm_labels = [(x - 1.0) / 4.0 for x in raw_labels]
            labels = torch.tensor(norm_labels, dtype=torch.float32)

            return cropped_wave, labels

        except Exception as e:
            # ===============================================================
            # 🔴 错误捕获区
            # ===============================================================
            # 打印出错的文件，方便你排查
            print(f"\n⚠️ 数据加载警告 [Index {idx}]: {e}")
            # print(f"出错文件: {row[self.path_col] if 'row' in locals() else 'Unknown'}")
            
            # 返回“假数据” (Dummy Data) 保证程序不崩溃
            # 返回 1秒的静音 + 标签[0.5, 0.5, 0.5] (代表中性情绪)
            dummy_wav = torch.zeros(1, self.max_length)
            dummy_label = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
            return dummy_wav, dummy_label