"""
Author: WeiqingZhu
Paper: GenAI-Enabled Dual-Stream Speech Semantic Communication under Dynamic Channels and Latency Constraints
Copyright: WeiqingZhu
Note: Please retain this attribution notice in any reuse of this code.
"""

import math

import torch
import torch.nn as nn


class PowerNormalization(nn.Module):
    """Normalize each transmitted feature vector to unit average power."""

    def forward(self, x):
        power = torch.mean(x**2, dim=-1, keepdim=True)
        return x / torch.sqrt(power + 1e-8)


class ChannelLayer(nn.Module):
    """Differentiable AWGN channel with float or tensor SNR input."""

    def __init__(self, channel_type="awgn"):
        super().__init__()
        self.channel_type = channel_type

    def forward(self, x, snr_db):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        if isinstance(snr_db, (float, int)):
            snr_db = torch.tensor(snr_db, device=x.device, dtype=x.dtype)
        elif isinstance(snr_db, torch.Tensor):
            snr_db = snr_db.to(device=x.device, dtype=x.dtype)
            if snr_db.dim() > 0:
                snr_db = snr_db.view(-1, 1, 1)

        snr_linear = torch.pow(10.0, snr_db / 10.0)
        noise_variance = 1.0 / (snr_linear + 1e-8)
        noise_std = torch.sqrt(noise_variance)
        noise = torch.randn_like(x) * noise_std
        return x + noise


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DeepSC_Adapter(nn.Module):
    """Transformer-based semantic communication adapter for layer-1 tokens."""

    def __init__(self, vocab_size=1024, d_model=256, nhead=4, num_layers=4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_tx = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.power_norm = PowerNormalization()
        self.channel = ChannelLayer()

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_rx = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.prediction_head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, snr_db, padding_mask=None):
        x = self.embedding(tokens)
        x = x * math.sqrt(x.size(-1))
        x = self.pos_encoder(x)

        tx_features = self.transformer_tx(x, src_key_padding_mask=padding_mask)
        tx_signal = self.power_norm(tx_features)
        rx_signal = self.channel(tx_signal, snr_db)
        rx_features = self.transformer_rx(rx_signal, src_key_padding_mask=padding_mask)
        return self.prediction_head(rx_features)
