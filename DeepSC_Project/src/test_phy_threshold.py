"""
Author: WeiqingZhu
Paper: GenAI-Enabled Dual-Stream Speech Semantic Communication under Dynamic Channels and Latency Constraints
Copyright: WeiqingZhu
Note: Please retain this attribution notice in any reuse of this code.
"""

import argparse
import json
from math import ceil
from pathlib import Path

import numpy as np
from scipy.special import erfc


MCS_TABLE_51311 = {
    0: (2, 120 / 1024),
    1: (2, 157 / 1024),
    2: (2, 193 / 1024),
    3: (2, 251 / 1024),
    4: (2, 308 / 1024),
    5: (2, 379 / 1024),
    6: (2, 449 / 1024),
    7: (2, 526 / 1024),
    8: (2, 602 / 1024),
    9: (2, 679 / 1024),
    10: (4, 340 / 1024),
    11: (4, 378 / 1024),
    12: (4, 434 / 1024),
    13: (4, 490 / 1024),
    14: (4, 553 / 1024),
    15: (4, 616 / 1024),
    16: (4, 658 / 1024),
    17: (6, 438 / 1024),
    18: (6, 466 / 1024),
    19: (6, 517 / 1024),
    20: (6, 567 / 1024),
    21: (6, 616 / 1024),
    22: (6, 666 / 1024),
    23: (6, 719 / 1024),
    24: (6, 772 / 1024),
    25: (6, 822 / 1024),
    26: (6, 873 / 1024),
    27: (6, 910 / 1024),
    28: (6, 948 / 1024),
}


def qfunc(x: float) -> float:
    return 0.5 * erfc(x / np.sqrt(2.0))


def bler_normal_approx_awgn(snr_db: float, tb_bits: int, q_m: int, rate: float) -> float:
    """Finite blocklength normal approximation for a complex AWGN channel."""
    gamma = 10.0 ** (snr_db / 10.0)
    coded_rate = q_m * rate
    if coded_rate <= 0:
        return 1.0

    n = int(ceil(tb_bits / coded_rate))
    capacity = np.log2(1.0 + gamma)
    dispersion = (gamma * (gamma + 2.0) / ((1.0 + gamma) ** 2)) * (np.log2(np.e) ** 2)
    score = (np.sqrt(n) * (capacity - coded_rate)) / np.sqrt(max(dispersion, 1e-12))
    return float(np.clip(qfunc(score), 1e-8, 1.0))


def simulate_harq_cc(
    snr_db: float,
    tb_bits: int,
    q_m: int,
    rate: float,
    max_tx: int,
    rng: np.random.Generator,
):
    for tx_idx in range(1, max_tx + 1):
        eq_snr = snr_db + 10.0 * np.log10(tx_idx)
        bler = bler_normal_approx_awgn(eq_snr, tb_bits, q_m, rate)
        if rng.random() > bler:
            return True, tx_idx
    return False, max_tx


def build_cache_payload_size_from_step1(npz_dir: Path, utt_id: str, transcript: str) -> int:
    candidates = sorted(npz_dir.glob(f"{utt_id}_SNR0*.npz"), key=lambda path: len(path.name))
    if not candidates:
        raise FileNotFoundError(f"missing {utt_id}_SNR0*.npz under {npz_dir}")

    data = np.load(candidates[0])
    tx_full = data["tx_full"]
    acoustic_bytes = int(tx_full.shape[0]) * 8 * 2
    transcript_bytes = len((transcript or "").encode("utf-8"))
    return 4 + acoustic_bytes + 4 + transcript_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--snr_min", type=float, default=-5.0)
    parser.add_argument("--snr_max", type=float, default=20.0)
    parser.add_argument("--snr_step", type=float, default=0.5)
    parser.add_argument(
        "--snr_list",
        type=str,
        default="",
        help="If provided, overrides min/max/step. Example: -5,-4,-3,0,5,10",
    )
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--tb_bytes", type=int, default=1024)
    parser.add_argument("--max_tx", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mcs_idx1", type=int, default=6)
    parser.add_argument("--mcs_idx2", type=int, default=14)
    parser.add_argument("--csv_out", type=str, default="results_step2_phy_harq_threshold.csv")
    parser.add_argument(
        "--max_redundancy",
        type=float,
        default=1.5,
        help="Maximum latency or retransmission budget factor.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    npz_dir = out_dir / "npz"
    if not npz_dir.is_dir():
        raise RuntimeError(f"missing {npz_dir} (point --out_dir to step1 dir)")

    items = []
    with open(args.meta_jsonl, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    kept = []
    payload_sizes = []
    for item in items:
        utt_id = item["id"]
        transcript = item.get("prompt_transcript", "")
        try:
            payload_size = build_cache_payload_size_from_step1(npz_dir, utt_id, transcript)
            kept.append(item)
            payload_sizes.append(payload_size)
        except Exception:
            continue

    if not kept:
        raise RuntimeError("No usable utts: meta ids do not match step1 npz outputs.")

    payload_sizes = np.array(payload_sizes, dtype=np.int64)
    if args.mcs_idx1 not in MCS_TABLE_51311 or args.mcs_idx2 not in MCS_TABLE_51311:
        raise RuntimeError("Invalid MCS index.")

    q_m1, rate1 = MCS_TABLE_51311[args.mcs_idx1]
    q_m2, rate2 = MCS_TABLE_51311[args.mcs_idx2]
    tb_bits = int(args.tb_bytes) * 8
    rng = np.random.default_rng(args.seed)

    if args.snr_list:
        snr_axis = sorted(float(item.strip()) for item in args.snr_list.split(",") if item.strip())
    else:
        snr_axis = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)

    print(f"[INFO] step1 out_dir = {out_dir}")
    print(f"[INFO] utts = {len(kept)}, trials = {args.trials}, TB={args.tb_bytes}B, max_tx={args.max_tx}")
    print(f"[INFO] Mode1: mcs={args.mcs_idx1}, Qm={q_m1}, R={rate1:.4f}")
    print(f"[INFO] Mode2: mcs={args.mcs_idx2}, Qm={q_m2}, R={rate2:.4f}")
    print("-" * 110)
    print(f"{'SNR':>6} | {'M1_upd':>8} | {'M1_tx/TB':>8} | {'M1_Goodp':>8} | {'M2_upd':>8} | {'M2_tx/TB':>8} | {'M2_Goodp':>8} | {'Avg_TBs':>8}")
    print("-" * 110)

    rows = []
    th1_99 = None
    th1_999 = None
    th2_99 = None
    th2_999 = None

    for snr in snr_axis:
        tb_counts = []
        m1_succ_pkts = 0
        m1_total_tx_attempts = 0
        m1_total_tbs_tried = 0
        m1_delivered_bytes = 0
        m2_succ_pkts = 0
        m2_total_tx_attempts = 0
        m2_total_tbs_tried = 0
        m2_delivered_bytes = 0

        for _ in range(args.trials):
            payload_size = int(payload_sizes[rng.integers(0, len(payload_sizes))])
            num_tbs = int(ceil(payload_size / args.tb_bytes))
            tb_counts.append(num_tbs)
            max_total_tx_budget = int(ceil(num_tbs * args.max_redundancy))

            total_tx_used_m1 = 0
            ok_m1 = True
            for _tb in range(num_tbs):
                success, tx_used = simulate_harq_cc(snr, tb_bits, q_m1, rate1, args.max_tx, rng)
                total_tx_used_m1 += tx_used
                m1_total_tx_attempts += tx_used
                m1_total_tbs_tried += 1
                if (not success) or (total_tx_used_m1 > max_total_tx_budget):
                    ok_m1 = False
                    break
            if ok_m1:
                m1_succ_pkts += 1
                m1_delivered_bytes += payload_size

            total_tx_used_m2 = 0
            ok_m2 = True
            for _tb in range(num_tbs):
                success, tx_used = simulate_harq_cc(snr, tb_bits, q_m2, rate2, args.max_tx, rng)
                total_tx_used_m2 += tx_used
                m2_total_tx_attempts += tx_used
                m2_total_tbs_tried += 1
                if (not success) or (total_tx_used_m2 > max_total_tx_budget):
                    ok_m2 = False
                    break
            if ok_m2:
                m2_succ_pkts += 1
                m2_delivered_bytes += payload_size

        avg_tbs = float(np.mean(tb_counts))
        m1_upd = m1_succ_pkts / args.trials
        m2_upd = m2_succ_pkts / args.trials
        m1_avg_tx_tb = (m1_total_tx_attempts / m1_total_tbs_tried) if m1_total_tbs_tried > 0 else 0.0
        m2_avg_tx_tb = (m2_total_tx_attempts / m2_total_tbs_tried) if m2_total_tbs_tried > 0 else 0.0
        m1_goodput = m1_delivered_bytes / (m1_total_tx_attempts * args.tb_bytes) if m1_total_tx_attempts > 0 else 0.0
        m2_goodput = m2_delivered_bytes / (m2_total_tx_attempts * args.tb_bytes) if m2_total_tx_attempts > 0 else 0.0

        print(
            f"{snr:6.2f} | {m1_upd:8.4f} | {m1_avg_tx_tb:8.2f} | {m1_goodput:8.4f} | "
            f"{m2_upd:8.4f} | {m2_avg_tx_tb:8.2f} | {m2_goodput:8.4f} | {avg_tbs:8.2f}"
        )

        if th1_99 is None and m1_upd >= 0.99:
            th1_99 = float(snr)
        if th2_99 is None and m2_upd >= 0.99:
            th2_99 = float(snr)
        if th1_999 is None and m1_upd >= 0.999:
            th1_999 = float(snr)
        if th2_999 is None and m2_upd >= 0.999:
            th2_999 = float(snr)

        rows.append((float(snr), m1_upd, m2_upd, m1_avg_tx_tb, m2_avg_tx_tb, m1_goodput, m2_goodput, avg_tbs))

    print("-" * 110)
    print("[RESULT] Threshold summary")
    print(f"Mode 1 (QPSK)  -> 0.99: {th1_99} dB | 0.999: {th1_999} dB")
    print(f"Mode 2 (16QAM) -> 0.99: {th2_99} dB | 0.999: {th2_999} dB")

    csv_path = Path(args.csv_out)
    with open(csv_path, "w", encoding="utf-8") as file:
        file.write("snr_db,m1_upd_pass,m2_upd_pass,m1_tx_tb,m2_tx_tb,m1_goodput,m2_goodput,avg_num_tbs\n")
        for row in rows:
            file.write(",".join(f"{value:.6f}" for value in row) + "\n")

    print(f"[DONE] Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()
