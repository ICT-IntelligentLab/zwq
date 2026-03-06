"""
Author: WeiqingZhu
Paper: GenAI-Enabled Dual-Stream Speech Semantic Communication under Dynamic Channels and Latency Constraints
Copyright: WeiqingZhu
Note: Please retain this attribution notice in any reuse of this code.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf


def try_stoi(ref, deg, sr):
    import torch
    import torchaudio
    from pystoi.stoi import stoi

    ref = np.asarray(ref, dtype=np.float32).reshape(-1)
    deg = np.asarray(deg, dtype=np.float32).reshape(-1)

    target_sr = 16000
    if sr != target_sr:
        ref_t = torch.from_numpy(ref).unsqueeze(0)
        deg_t = torch.from_numpy(deg).unsqueeze(0)
        ref_t = torchaudio.functional.resample(ref_t, sr, target_sr)
        deg_t = torchaudio.functional.resample(deg_t, sr, target_sr)
        ref = ref_t.squeeze(0).numpy()
        deg = deg_t.squeeze(0).numpy()
        sr = target_sr

    length = min(len(ref), len(deg))
    ref = ref[:length]
    deg = deg[:length]
    return float(stoi(ref, deg, sr, extended=False))


def load_wav(path):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    return wav, sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True, help="results/step1_oracle_residual")
    parser.add_argument("--snr_list", type=str, default="-5,0,5,20")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    npz_dir = out_dir / "npz"
    wav_dir = out_dir / "audio"
    if not npz_dir.is_dir():
        raise RuntimeError(f"missing {npz_dir}")
    if not wav_dir.is_dir():
        raise RuntimeError(f"missing {wav_dir}")

    snrs_float = sorted(float(item.strip()) for item in args.snr_list.split(",") if item.strip())
    all_npz = list(npz_dir.glob("*.npz"))
    if not all_npz:
        raise RuntimeError(f"npz_dir is empty: {npz_dir}")

    pattern = re.compile(r"^(.*)_SNR(-?\d+(?:\.\d+)?)$")
    by_snr = {snr: [] for snr in snrs_float}

    for path in all_npz:
        match = pattern.match(path.stem)
        if not match:
            continue
        utt_id = match.group(1)
        snr = float(match.group(2))
        if snr in by_snr:
            by_snr[snr].append((utt_id, path))

    print("==== Token ACC (tx_layer1 vs rx_layer1) + STOI (ref vs oracleRes) ====")
    print(f"out_dir = {out_dir}")
    print(f"snrs = {snrs_float}")
    print("------------------------------------------------------------")

    summary_data = {
        "snr_list": [],
        "acc_mean": [],
        "acc_std": [],
        "stoi_mean": [],
        "stoi_std": [],
        "num_utts": [],
    }

    for snr in snrs_float:
        pairs = sorted(by_snr.get(snr, []), key=lambda item: item[0])
        if not pairs:
            print(f"SNR={snr:g}: NO FILES")
            continue

        acc_list = []
        stoi_list = []
        bad_wav = 0

        for utt_id, npz_path in pairs:
            data = np.load(npz_path)
            tx = data["tx_layer1"].astype(np.int64).reshape(-1)
            rx = data["rx_layer1"].astype(np.int64).reshape(-1)
            length = min(len(tx), len(rx))
            if length == 0:
                continue

            acc = float((tx[:length] == rx[:length]).mean())
            acc_list.append(acc)

            ref_path = wav_dir / f"{utt_id}_REF_oracle.wav"
            deg_path = wav_dir / f"{utt_id}_SNR{snr:g}_oracleRes.wav"
            if (not ref_path.exists()) or (not deg_path.exists()):
                bad_wav += 1
                continue

            ref_wav, sr1 = load_wav(ref_path)
            deg_wav, sr2 = load_wav(deg_path)
            if sr1 != sr2:
                raise RuntimeError(f"SR mismatch: {ref_path} sr={sr1}, {deg_path} sr={sr2}")

            try:
                stoi_list.append(try_stoi(ref_wav, deg_wav, sr1))
            except Exception:
                bad_wav += 1

        acc_mean = float(np.mean(acc_list)) if acc_list else float("nan")
        acc_std = float(np.std(acc_list)) if acc_list else 0.0
        stoi_mean = float(np.mean(stoi_list)) if stoi_list else float("nan")
        stoi_std = float(np.std(stoi_list)) if stoi_list else 0.0
        n_utts = len(pairs)

        summary_data["snr_list"].append(snr)
        summary_data["acc_mean"].append(acc_mean)
        summary_data["acc_std"].append(acc_std)
        summary_data["stoi_mean"].append(stoi_mean)
        summary_data["stoi_std"].append(stoi_std)
        summary_data["num_utts"].append(n_utts)

        print(
            f"SNR={snr:>5g} dB | utts={n_utts:>4d} | "
            f"ACC: {acc_mean:.4f} +/- {acc_std:.4f} | "
            f"STOI: {stoi_mean:.4f} +/- {stoi_std:.4f} | wav_bad={bad_wav}"
        )

    json_out_path = out_dir / "metrics_summary.json"
    with open(json_out_path, "w", encoding="utf-8") as file:
        json.dump(summary_data, file, indent=4)

    print("------------------------------------------------------------")
    print(f"[DONE] Saved summary to: {json_out_path}")


if __name__ == "__main__":
    main()
