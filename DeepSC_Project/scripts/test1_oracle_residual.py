"""
Author: WeiqingZhu
Paper: GenAI-Enabled Dual-Stream Speech Semantic Communication under Dynamic Channels and Latency Constraints
Copyright: WeiqingZhu
Note: Please retain this attribution notice in any reuse of this code.

End-to-end upper-bound evaluation using oracle residual layers from VALL-E X.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from channel_model import DeepSC_Adapter
from vallex_wrapper import SemanticReceiver, SemanticSender


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def safe_wav(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -1.0, 1.0)


def write_wav(path, wav, sr=24000):
    sf.write(str(path), safe_wav(wav), int(sr), format="WAV", subtype="PCM_16")


def load_meta(meta_jsonl, num_utts):
    items = []
    with open(meta_jsonl, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items[:num_utts]


def unpack_full_codes(tx):
    if isinstance(tx, dict):
        full_codes = tx.get("full_codes")
    else:
        full_codes = tx[1] if len(tx) > 1 else None

    if full_codes is None:
        raise RuntimeError("sender.process(return_full_codes=True) did not return full_codes.")

    full_codes = np.asarray(full_codes)
    if full_codes.ndim == 3:
        full_codes = full_codes[0]
    if not (full_codes.ndim == 2 and full_codes.shape[1] == 8):
        raise RuntimeError(f"bad full_codes shape={full_codes.shape}")
    return full_codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_jsonl", required=True)
    parser.add_argument("--adapter_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_utts", type=int, default=10)
    parser.add_argument("--snr_list", type=str, default="-5,0,5,20")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "npz").mkdir(parents=True, exist_ok=True)
    snr_list = [float(item) for item in args.snr_list.split(",") if item.strip()]

    adapter = DeepSC_Adapter(vocab_size=1024, d_model=256, nhead=4, num_layers=4).to(DEVICE)
    adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=DEVICE))
    adapter.eval()
    print("[OK] adapter loaded:", args.adapter_ckpt)

    sender = SemanticSender()
    receiver = SemanticReceiver()
    print("[OK] sender/receiver ready")

    items = load_meta(args.meta_jsonl, args.num_utts)
    print("[OK] eval items =", len(items))

    for item in tqdm(items, desc="Utts"):
        utt_id = item["id"]
        prompt_wav = item["prompt_wav"]
        prompt_transcript = item["prompt_transcript"]

        tx = sender.process(
            text=prompt_transcript,
            prompt_path=str(prompt_wav),
            language="en",
            return_full_codes=True,
            prompt_transcript=prompt_transcript,
        )
        full_codes = unpack_full_codes(tx)

        ref_wav = receiver.full_codes_to_audio(full_codes)
        write_wav(out_dir / "audio" / f"{utt_id}_REF_oracle.wav", ref_wav, 24000)

        oracle_residual = full_codes[:, 1:8]
        tx_layer1 = full_codes[:, 0].astype(np.int64)
        tokens = torch.tensor(tx_layer1[None, :], dtype=torch.long, device=DEVICE)

        for snr in snr_list:
            with torch.no_grad():
                logits = adapter(tokens, snr_db=float(snr))
                rx_layer1 = torch.argmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.int64)

            rx_full = np.concatenate([rx_layer1[:, None], oracle_residual], axis=1)
            rx_wav = receiver.full_codes_to_audio(rx_full)
            write_wav(out_dir / "audio" / f"{utt_id}_SNR{snr:g}_oracleRes.wav", rx_wav, 24000)

            np.savez_compressed(
                out_dir / "npz" / f"{utt_id}_SNR{snr:g}.npz",
                tx_full=full_codes.astype(np.int16),
                rx_full=rx_full.astype(np.int16),
                tx_layer1=tx_layer1.astype(np.int16),
                rx_layer1=rx_layer1.astype(np.int16),
            )

    print("[DONE] out_dir =", out_dir)


if __name__ == "__main__":
    main()
