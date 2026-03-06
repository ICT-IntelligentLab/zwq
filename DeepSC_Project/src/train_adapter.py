"""
Author: WeiqingZhu
Paper: GenAI-Enabled Dual-Stream Speech Semantic Communication under Dynamic Channels and Latency Constraints
Copyright: WeiqingZhu
Note: Please retain this attribution notice in any reuse of this code.
"""

import json
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config

try:
    from channel_model import DeepSC_Adapter
except ImportError:
    SRC_PATH = os.path.join(PROJECT_ROOT, "src")
    if SRC_PATH not in sys.path:
        sys.path.append(SRC_PATH)
    from channel_model import DeepSC_Adapter


BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
MAX_LEN = 750
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIXED_VAL_SNR = 0.0
SEED = 0

DATA_PATH = os.path.join(config.DATA_DIR, "train_unique_1k.npy")
SAVE_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, "deepsc_model_curriculum2.pth")
PLOT_SAVE_PATH = os.path.join(SAVE_DIR, "training_curves.png")
HISTORY_SAVE_PATH = os.path.join(SAVE_DIR, "training_history.json")
os.makedirs(SAVE_DIR, exist_ok=True)


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_snr_curriculum(epoch, total_epochs):
    if epoch < total_epochs * 0.2:
        return 15.0, 25.0, "Warm-up"
    if epoch < total_epochs * 0.5:
        return 5.0, 15.0, "Medium"
    if epoch < total_epochs * 0.8:
        return -5.0, 5.0, "Hard"
    return -5.0, 20.0, "Generalization"


class TokenDataset(Dataset):
    def __init__(self, npy_path):
        print(f"[LOAD] {npy_path}")
        data = np.load(npy_path, allow_pickle=True)
        self.data = [item for item in data if len(item) > 10]
        print(f"[OK] samples = {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        if len(tokens) > MAX_LEN:
            tokens = tokens[:MAX_LEN]

        tokens = torch.tensor(tokens, dtype=torch.long)
        if torch.any(tokens == 0):
            raise ValueError("Found token=0 in data; padding_value=0 breaks masking.")
        return tokens


def collate_fn(batch):
    padded_tokens = pad_sequence(batch, batch_first=True, padding_value=0)
    mask = padded_tokens == 0
    return padded_tokens, mask


def plot_history(history):
    print("[PLOT] Saving training curves.")
    epochs_range = range(1, EPOCHS + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(
        epochs_range,
        history["val_loss"],
        label=f"Val Loss (SNR={FIXED_VAL_SNR}dB)",
    )
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train ACC")
    plt.plot(
        epochs_range,
        history["val_acc"],
        label=f"Val ACC (SNR={FIXED_VAL_SNR}dB)",
    )
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_SAVE_PATH, dpi=300)
    print(f"[PLOT] Saved: {PLOT_SAVE_PATH}")


def train():
    full_dataset = TokenDataset(DATA_PATH)
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    generator = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    print(f"[SPLIT] train={train_size} | val={val_size}")

    model = DeepSC_Adapter(vocab_size=1024, d_model=256, nhead=4, num_layers=4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_val_acc = -1.0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        snr_min, snr_max, stage_name = get_snr_curriculum(epoch, EPOCHS)
        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1:03d}/{EPOCHS} [{stage_name}]")

        for tokens, mask in pbar:
            tokens = tokens.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            snr_db = (torch.rand(1).item() * (snr_max - snr_min)) + snr_min
            outputs = model(tokens, snr_db, padding_mask=mask)

            loss = criterion(outputs.view(-1, 1024), tokens.view(-1))
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item())
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=-1)
                valid_mask = ~mask
                train_correct += ((preds == tokens) & valid_mask).sum().item()
                train_total += valid_mask.sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.3f}", "snr": f"{snr_db:.1f}"})

        avg_train_loss = train_loss_sum / max(len(train_loader), 1)
        avg_train_acc = (train_correct / train_total) if train_total > 0 else 0.0

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for tokens, mask in val_loader:
                tokens = tokens.to(DEVICE, non_blocking=True)
                mask = mask.to(DEVICE, non_blocking=True)

                outputs = model(tokens, FIXED_VAL_SNR, padding_mask=mask)
                loss = criterion(outputs.view(-1, 1024), tokens.view(-1))
                val_loss_sum += float(loss.item())

                preds = torch.argmax(outputs, dim=-1)
                valid_mask = ~mask
                val_correct += ((preds == tokens) & valid_mask).sum().item()
                val_total += valid_mask.sum().item()

        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        avg_val_acc = (val_correct / val_total) if val_total > 0 else 0.0

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(avg_train_acc)
        history["val_acc"].append(avg_val_acc)

        save_msg = ""
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            save_msg = " [SAVE best val_acc]"

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Train loss={avg_train_loss:.4f} acc={avg_train_acc:.6f} | "
            f"Val({FIXED_VAL_SNR:.1f}dB) loss={avg_val_loss:.4f} acc={avg_val_acc:.6f}"
            f"{save_msg}"
        )

    print(f"\n[DONE] best val_acc = {best_val_acc:.6f}")
    plot_history(history)

    with open(HISTORY_SAVE_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)
    print(f"[DATA] Saved: {HISTORY_SAVE_PATH}")


if __name__ == "__main__":
    seed_everything(SEED)
    train()
