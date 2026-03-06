"""
Author: WeiqingZhu
Paper: GenAI-Enabled Dual-Stream Speech Semantic Communication under Dynamic Channels and Latency Constraints
Copyright: WeiqingZhu
Note: Please retain this attribution notice in any reuse of this code.

No-update baseline that always reuses the frozen acoustic cache.
"""

import argparse
import json
import os
import sys
import time
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
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


def write_wav(path: Path, wav, sr=24000):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), safe_wav(wav), int(sr), format="WAV", subtype="PCM_16")


def load_meta(meta_jsonl: str):
    items = []
    with open(meta_jsonl, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def spk_from_utt_id(utt_id: str) -> str:
    return utt_id.split("_")[0] if "_" in utt_id else utt_id


def parse_snr_list(snr_list_tokens):
    if snr_list_tokens is None:
        return []
    if isinstance(snr_list_tokens, str):
        return [float(item.strip()) for item in snr_list_tokens.split(",") if item.strip()]

    merged = []
    for token in snr_list_tokens:
        if token is None:
            continue
        token = str(token).strip()
        if not token:
            continue
        for item in token.split(","):
            item = item.strip()
            if item:
                merged.append(float(item))
    return merged


def unpack_full_codes(tx):
    if isinstance(tx, dict):
        full_codes = tx.get("full_codes", None)
    elif isinstance(tx, (tuple, list)):
        full_codes = tx[1] if len(tx) > 1 else None
    else:
        full_codes = None

    if full_codes is None:
        raise RuntimeError("Cannot get full_codes from sender.process(return_full_codes=True).")

    full_codes = np.asarray(full_codes)
    if full_codes.ndim == 3:
        full_codes = full_codes[0]
    if not (full_codes.ndim == 2 and full_codes.shape[1] == 8):
        raise RuntimeError(f"bad full_codes shape={full_codes.shape}, expect (T,8)")
    return full_codes


def payload_bytes_from_acoustic(tx_acoustic_t7: np.ndarray) -> int:
    frame_count = int(tx_acoustic_t7.shape[0])
    return int(frame_count * 7 * 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_jsonl", required=True)
    parser.add_argument("--adapter_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--snr_th_mode1", type=float, default=0.0)
    parser.add_argument("--snr_th_mode2", type=float, default=2.5)
    parser.add_argument(
        "--snr_list",
        nargs="+",
        default=["-5,-2,-1,0,1,2,2.5,3,5,10,15,20"],
        help='SNR list, e.g. --snr_list "-5,-2,0,5" or --snr_list -5 -2 0 5',
    )
    parser.add_argument("--tb_bytes", type=int, default=1024)
    parser.add_argument("--max_tx", type=int, default=4)
    parser.add_argument("--max_avg_tx_per_tb", type=float, default=1.5)
    parser.add_argument("--num_speakers", type=int, default=10)
    parser.add_argument("--utts_per_speaker", type=int, default=10)
    parser.add_argument("--min_utts_per_speaker", type=int, default=3)
    parser.add_argument("--max_total_utts", type=int, default=0, help="0 means no limit")
    parser.add_argument("--write_audio", action="store_true")
    parser.add_argument("--sr", type=int, default=24000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    if args.write_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    np.random.default_rng(args.seed)
    snr_list = parse_snr_list(args.snr_list)
    if not snr_list:
        raise ValueError("Empty snr_list after parsing. Check your CLI.")

    adapter = DeepSC_Adapter(vocab_size=1024, d_model=256, nhead=4, num_layers=4).to(DEVICE)
    adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=DEVICE))
    adapter.eval()
    sender = SemanticSender()
    receiver = SemanticReceiver()

    items_all = load_meta(args.meta_jsonl)
    by_spk = {}
    for item in items_all:
        utt_id = item["id"]
        spk = spk_from_utt_id(utt_id)
        by_spk.setdefault(spk, []).append(item)

    eligible_spks = [spk for spk, items in by_spk.items() if len(items) >= args.min_utts_per_speaker]
    eligible_spks.sort()
    if not eligible_spks:
        raise RuntimeError(
            f"No speakers with >= {args.min_utts_per_speaker} utts in meta. "
            "Your meta may be too small or too speaker-sparse for cache evaluation."
        )

    chosen_spks = eligible_spks[: args.num_speakers]
    selected_items = []
    for spk in chosen_spks:
        selected_items.extend(by_spk[spk][: args.utts_per_speaker])

    if args.max_total_utts and args.max_total_utts > 0:
        selected_items = selected_items[: args.max_total_utts]

    selected_items.sort(key=lambda item: (spk_from_utt_id(item["id"]), item["id"]))

    print(f"[INFO] DEVICE={DEVICE}")
    print(f"[INFO] meta={args.meta_jsonl}")
    print(f"[INFO] selected speakers={len(chosen_spks)} (min_utts_per_speaker={args.min_utts_per_speaker})")
    print(f"[INFO] total selected utts={len(selected_items)}")
    print(f"[INFO] snr_list={snr_list}")
    print(f"[INFO] thresholds: mode1={args.snr_th_mode1:.2f} dB, mode2={args.snr_th_mode2:.2f} dB")
    print(f"[INFO] HARQ: tb_bytes={args.tb_bytes}, max_tx={args.max_tx}, max_avg_tx_per_tb={args.max_avg_tx_per_tb}")
    print(f"[INFO] write_audio={args.write_audio} out_dir={out_dir}")

    acoustic_frozen_library = {}
    rows = []
    agg = {
        snr: {
            "n": 0,
            "upd": 0,
            "avg_tx_sum": 0.0,
            "tb_sum": 0,
            "adapter_lat_sum": 0.0,
            "vocoder_lat_sum": 0.0,
        }
        for snr in snr_list
    }

    for item in tqdm(selected_items, desc="System-Level Eval"):
        utt_id = item["id"]
        spk_id = spk_from_utt_id(utt_id)
        prompt_wav = item["prompt_wav"]
        transcript = item.get("prompt_transcript", "")

        t0 = time.perf_counter()
        tx = sender.process(
            text=transcript,
            prompt_path=str(prompt_wav),
            language="en",
            return_full_codes=True,
            prompt_transcript=transcript,
        )
        full_codes = unpack_full_codes(tx)
        latency_sender_sec = time.perf_counter() - t0

        tx_layer1 = full_codes[:, 0].astype(np.int64)
        tx_acoustic = full_codes[:, 1:8].astype(np.int64)

        if spk_id not in acoustic_frozen_library:
            acoustic_frozen_library[spk_id] = tx_acoustic.copy()
            handshake = True
        else:
            handshake = False

        payload_bytes = payload_bytes_from_acoustic(tx_acoustic)
        tokens = torch.tensor(tx_layer1[None, :], dtype=torch.long, device=DEVICE)

        oracle_wav_path = ""
        if args.write_audio:
            rx_full_clean = np.concatenate([tx_layer1[:, None], tx_acoustic], axis=1)
            t_oracle = time.perf_counter()
            wav_clean = receiver.full_codes_to_audio(rx_full_clean)
            latency_vocoder_oracle_sec = time.perf_counter() - t_oracle
            oracle_wav_path = str(audio_dir / f"{utt_id}_ORACLE.wav")
            write_wav(Path(oracle_wav_path), wav_clean, args.sr)
        else:
            latency_vocoder_oracle_sec = 0.0

        for snr in snr_list:
            t_adapter = time.perf_counter()
            with torch.no_grad():
                logits = adapter(tokens, snr_db=float(snr))
                rx_layer1 = torch.argmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.int64)
            latency_adapter_sec = time.perf_counter() - t_adapter

            update_success = False
            mode = 0
            tx_used = 0
            num_tbs = int(ceil(payload_bytes / args.tb_bytes))
            budget = int(ceil(num_tbs * args.max_avg_tx_per_tb))
            status_tag = "NoUpdate_Frozen"

            active_acoustic = acoustic_frozen_library[spk_id]

            t_sem = rx_layer1.shape[0]
            t_aco = active_acoustic.shape[0]
            if t_aco < t_sem:
                repeat_count = int(ceil(t_sem / t_aco))
                aligned_acoustic = np.tile(active_acoustic, (repeat_count, 1))[:t_sem, :]
            else:
                aligned_acoustic = active_acoustic[:t_sem, :]

            rx_full = np.concatenate([rx_layer1[:, None], aligned_acoustic], axis=1)

            degraded_wav_path = ""
            t_vocoder = time.perf_counter()
            if args.write_audio:
                wav_deg = receiver.full_codes_to_audio(rx_full)
                degraded_wav_path = str(audio_dir / f"{utt_id}_SNR{snr:g}_{status_tag}.wav")
                write_wav(Path(degraded_wav_path), wav_deg, args.sr)
            latency_vocoder_deg_sec = time.perf_counter() - t_vocoder

            avg_tx_per_tb = float(tx_used) / float(num_tbs) if num_tbs > 0 else 0.0
            rows.append(
                {
                    "utt_id": utt_id,
                    "spk_id": spk_id,
                    "snr_db": float(snr),
                    "mode": int(mode),
                    "status_tag": status_tag,
                    "update_success": bool(update_success),
                    "handshake_init_cache": bool(handshake),
                    "payload_bytes": int(payload_bytes),
                    "tb_bytes": int(args.tb_bytes),
                    "num_tbs": int(num_tbs),
                    "total_budget_tx": int(budget),
                    "total_tx_used": int(tx_used),
                    "avg_tx_per_tb": float(avg_tx_per_tb),
                    "prompt_wav": str(prompt_wav),
                    "transcript": transcript,
                    "oracle_wav": oracle_wav_path,
                    "degraded_wav": degraded_wav_path,
                    "latency_sender_sec": float(latency_sender_sec),
                    "latency_adapter_sec": float(latency_adapter_sec),
                    "latency_vocoder_oracle_sec": float(latency_vocoder_oracle_sec),
                    "latency_vocoder_deg_sec": float(latency_vocoder_deg_sec),
                }
            )

            agg[snr]["n"] += 1
            agg[snr]["upd"] += int(update_success)
            agg[snr]["avg_tx_sum"] += avg_tx_per_tb
            agg[snr]["tb_sum"] += num_tbs
            agg[snr]["adapter_lat_sum"] += latency_adapter_sec
            agg[snr]["vocoder_lat_sum"] += latency_vocoder_deg_sec

    df = pd.DataFrame(rows)
    csv_path = out_dir / "step3_generation_meta.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] wrote meta csv: {csv_path}")

    print("\n================= SUMMARY (per SNR) =================")
    for snr in snr_list:
        sample_count = agg[snr]["n"]
        if sample_count == 0:
            continue
        upd_rate = agg[snr]["upd"] / sample_count
        avg_tx = agg[snr]["avg_tx_sum"] / sample_count
        avg_ad_lat = agg[snr]["adapter_lat_sum"] / sample_count
        avg_voc_lat = agg[snr]["vocoder_lat_sum"] / sample_count
        print(
            f"SNR={snr:>6g} dB | samples={sample_count:>5d} | upd_rate={upd_rate:.4f} "
            f"| avg_tx/tb={avg_tx:.3f} | avg_adapter_lat={avg_ad_lat * 1000:.2f} ms "
            f"| avg_vocoder_lat={avg_voc_lat * 1000:.2f} ms"
        )
    print("=====================================================\n")

    if args.write_audio:
        print(f"[OK] wavs saved under: {audio_dir}")
    print("[DONE]")


if __name__ == "__main__":
    main()
