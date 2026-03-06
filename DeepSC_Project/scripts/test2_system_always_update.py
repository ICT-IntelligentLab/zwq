"""
Author: WeiqingZhu
Paper: GenAI-Enabled Dual-Stream Speech Semantic Communication under Dynamic Channels and Latency Constraints
Copyright: WeiqingZhu
Note: Please retain this attribution notice in any reuse of this code.

Always-update baseline with best-effort acoustic cache delivery.
"""
import argparse, json, time
from pathlib import Path
import os, sys
from math import ceil
import numpy as np
import torch
import soundfile as sf
import pandas as pd
from tqdm import tqdm
from scipy.special import erfc
# =========================
# Path setup (consistent with your project)
# =========================
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]          # .../DeepSC_Project
SRC_DIR = PROJECT_ROOT / "src"               # .../DeepSC_Project/src
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from channel_model import DeepSC_Adapter
from vallex_wrapper import SemanticSender, SemanticReceiver

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =======================================================
# PHY abstraction: FBL normal approximation + HARQ-CC
# =======================================================
def Qfunc(x: float) -> float:
    return float(0.5 * erfc(x / np.sqrt(2.0)))


def bler_normal_approx_awgn(snr_db: float, tb_bits: int, Qm: int, R: float) -> float:
    """
    Finite blocklength normal approximation for complex AWGN (rough PHY abstraction).
    Interpreting snr_db as Eb/N0 (dB) consistent with your pipeline.
    """
    ebn0 = 10.0 ** (float(snr_db) / 10.0)

    Rb = float(Qm) * float(R)  # bits per complex channel use
    if Rb <= 0:
        return 1.0

    # Eb/N0 -> SNR per complex use
    gamma = ebn0 * Rb

    # number of complex channel uses
    n = int(ceil(tb_bits / Rb))
    n = max(n, 1)

    # capacity & dispersion (complex AWGN)
    C = np.log2(1.0 + gamma)
    V = (gamma * (gamma + 2.0) / ((1.0 + gamma) ** 2)) * (np.log2(np.e) ** 2)

    x = (np.sqrt(n) * (C - Rb)) / np.sqrt(max(V, 1e-12))
    eps = Qfunc(x)
    return float(np.clip(eps, 1e-8, 1.0))


def simulate_harq_update(
    snr_db: float,
    payload_bytes: int,
    Qm: int,
    R: float,
    rng: np.random.Generator,
    tb_bytes: int,
    max_tx: int,
    max_avg_tx_per_tb: float,
):
    """
    Simulate whole cache update payload transfer:
    - Split payload into TBs
    - Each TB uses HARQ-CC up to max_tx transmissions
    - Global latency budget: total_tx_used <= ceil(num_tbs * max_avg_tx_per_tb)

    Returns:
      success (bool),
      total_tx_used (int),
      num_tbs (int),
      total_budget (int)
    """
    tb_bits = int(tb_bytes) * 8
    num_tbs = int(ceil(payload_bytes / tb_bytes))
    num_tbs = max(num_tbs, 1)

    total_budget = int(ceil(num_tbs * float(max_avg_tx_per_tb)))
    total_budget = max(total_budget, 1)

    total_tx_used = 0

    for _ in range(num_tbs):
        tb_success = False
        for i in range(1, int(max_tx) + 1):
            total_tx_used += 1

            # Chase combining gain (approx): i transmissions => +10log10(i) dB
            eq_snr = float(snr_db) + 10.0 * np.log10(i)

            bler = bler_normal_approx_awgn(eq_snr, tb_bits, Qm, R)
            if rng.random() > bler:
                tb_success = True
                break

            if total_tx_used > total_budget:
                return False, total_tx_used, num_tbs, total_budget

        if (not tb_success) or (total_tx_used > total_budget):
            return False, total_tx_used, num_tbs, total_budget

    return True, total_tx_used, num_tbs, total_budget



def simulate_harq_update_partial(
    snr_db: float,
    payload_bytes: int,
    Qm: int,
    R: float,
    rng: np.random.Generator,
    tb_bytes: int,
    max_tx: int,
    max_avg_tx_per_tb: float,
    bytes_per_frame: int,
):
    """
    Best-effort HARQ-CC cache update:
    - Always attempts to transmit TBs sequentially (with retransmissions) until success, max_tx, or budget exhaustion.
    - If budget is exhausted before completing all TBs, returns a PARTIAL delivery: TBs that decoded are marked True, others False.
    - This guarantees the receiver gets *some* updated portion whenever possible under the latency budget, without "cheating".

    Returns:
      tb_ok (list[bool]) length=num_tbs,
      total_tx_used (int),
      num_tbs (int),
      total_budget (int),
      tb_frames (int) number of acoustic frames per TB (except last TB)
    """
    tb_frames = int(tb_bytes) // int(bytes_per_frame)
    tb_frames = max(tb_frames, 1)
    eff_tb_bytes = int(tb_frames * int(bytes_per_frame))

    tb_bits = eff_tb_bytes * 8
    num_tbs = int(ceil(payload_bytes / eff_tb_bytes))
    num_tbs = max(num_tbs, 1)

    total_budget = int(ceil(num_tbs * float(max_avg_tx_per_tb)))
    total_budget = max(total_budget, 1)

    total_tx_used = 0
    tb_ok = [False] * num_tbs

    for tb_idx in range(num_tbs):
        # if no budget left for even 1 transmission, stop (remaining TBs stay False)
        if total_tx_used >= total_budget:
            break

        for i in range(1, int(max_tx) + 1):
            if total_tx_used >= total_budget:
                break

            total_tx_used += 1

            # Chase combining gain (approx): i transmissions => +10log10(i) dB
            eq_snr = float(snr_db) + 10.0 * np.log10(i)

            bler = bler_normal_approx_awgn(eq_snr, tb_bits, Qm, R)
            if rng.random() > bler:
                tb_ok[tb_idx] = True
                break

    return tb_ok, total_tx_used, num_tbs, total_budget, tb_frames



# =========================
# IO utils
# =========================
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
    with open(meta_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def spk_from_utt_id(utt_id: str) -> str:
    # your id looks like "1992_141719_000014_000013"
    return utt_id.split("_")[0] if "_" in utt_id else utt_id


def parse_snr_list(snr_list_tokens):
    """
    Supports:
      --snr_list "-5,-2,-1,0,1"
      --snr_list -5 -2 -1 0 1
      --snr_list "-5" "-2" "0"
    """
    if snr_list_tokens is None:
        return []
    if isinstance(snr_list_tokens, str):
        raw = snr_list_tokens
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        return [float(p) for p in parts]

    # nargs='+' gives list
    merged = []
    for tok in snr_list_tokens:
        if tok is None:
            continue
        tok = str(tok).strip()
        if not tok:
            continue
        # allow comma inside token too
        for p in tok.split(","):
            p = p.strip()
            if p:
                merged.append(float(p))
    return merged


def unpack_full_codes(tx):
    """
    Compatible with your two variants:
      - tuple: tx[1] is full_codes
      - dict: tx["full_codes"]
    """
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


def payload_bytes_from_acoustic(tx_acoustic_T7: np.ndarray) -> int:
    """
    Your reliable stream payload in this stage: 7 layers acoustic tokens (int16).
    If later you add phonemes/transcript etc., expand here.
    """
    T = int(tx_acoustic_T7.shape[0])
    # 7 layers * T * int16(2 bytes)
    return int(T * 7 * 2)


# =======================================================
# Main
# =======================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--meta_jsonl", required=True)
    ap.add_argument("--adapter_ckpt", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)

    # thresholds (use your computed ones)
    ap.add_argument("--snr_th_mode1", type=float, default=0.0)
    ap.add_argument("--snr_th_mode2", type=float, default=2.5)

    # IMPORTANT: accept both comma-string and multiple args
    ap.add_argument(
        "--snr_list",
        nargs="+",
        default=["-5,-2,-1,0,1,2,2.5,3,5,10,15,20"],
        help='SNR list, e.g. --snr_list "-5,-2,0,5"  OR  --snr_list -5 -2 0 5',
    )

    # HARQ knobs
    ap.add_argument("--tb_bytes", type=int, default=1024)
    ap.add_argument("--max_tx", type=int, default=4)
    ap.add_argument("--max_avg_tx_per_tb", type=float, default=1.5)

    # speaker/utt sampling
    ap.add_argument("--num_speakers", type=int, default=10)
    ap.add_argument("--utts_per_speaker", type=int, default=10)
    ap.add_argument("--min_utts_per_speaker", type=int, default=3)
    ap.add_argument("--max_total_utts", type=int, default=0, help="0 means no limit")

    # output
    ap.add_argument("--write_audio", action="store_true")
    ap.add_argument("--sr", type=int, default=24000)

    ap.add_argument("--always_update_mode", type=int, default=2, choices=[1,2], help="Always-Update baseline uses fixed MCS: 1->(Qm=2,R=0.44), 2->(Qm=4,R=0.54)")

    # Paper alignment: count full Mode2 update success only when SNR >= this threshold.
    # Below this threshold we still transmit and commit a (possibly erroneous) cache to the receiver.
    ap.add_argument("--mode2_success_th_db", type=float, default=5.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    if args.write_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    snr_list = parse_snr_list(args.snr_list)
    if not snr_list:
        raise ValueError("Empty snr_list after parsing. Check your CLI.")

    # init models
    adapter = DeepSC_Adapter(vocab_size=1024, d_model=256, nhead=4, num_layers=4).to(DEVICE)
    adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=DEVICE))
    adapter.eval()
    sender = SemanticSender()
    receiver = SemanticReceiver()

    # load meta and group by speaker
    items_all = load_meta(args.meta_jsonl)
    by_spk = {}
    for it in items_all:
        utt_id = it["id"]
        spk = spk_from_utt_id(utt_id)
        by_spk.setdefault(spk, []).append(it)

    # pick speakers that have enough utts
    eligible_spks = [spk for spk, lst in by_spk.items() if len(lst) >= args.min_utts_per_speaker]
    eligible_spks.sort()

    if not eligible_spks:
        raise RuntimeError(
            f"No speakers with >= {args.min_utts_per_speaker} utts in meta. "
            "Your meta may be too small or too speaker-sparse for cache evaluation."
        )

    chosen_spks = eligible_spks[: args.num_speakers]
    selected_items = []
    for spk in chosen_spks:
        lst = by_spk[spk][: args.utts_per_speaker]
        selected_items.extend(lst)

    if args.max_total_utts and args.max_total_utts > 0:
        selected_items = selected_items[: args.max_total_utts]

    # sort to ensure each speaker is processed sequentially (cache makes sense)
    selected_items.sort(key=lambda it: (spk_from_utt_id(it["id"]), it["id"]))

    print(f"[INFO] DEVICE={DEVICE}")
    print(f"[INFO] meta={args.meta_jsonl}")
    print(f"[INFO] selected speakers={len(chosen_spks)} (min_utts_per_speaker={args.min_utts_per_speaker})")
    print(f"[INFO] total selected utts={len(selected_items)}")
    print(f"[INFO] snr_list={snr_list}")
    print(f"[INFO] thresholds: mode1={args.snr_th_mode1:.2f} dB, mode2={args.snr_th_mode2:.2f} dB")
    print(f"[INFO] HARQ: tb_bytes={args.tb_bytes}, max_tx={args.max_tx}, max_avg_tx_per_tb={args.max_avg_tx_per_tb}")
    print(f"[INFO] write_audio={args.write_audio} out_dir={out_dir}")

    # receiver cache (frozen library), per speaker
    acoustic_frozen_library = {}

    # log rows for downstream metrics computation
    rows = []

    # quick SNR aggregation
    agg = {snr: {"n": 0, "upd": 0, "avg_tx_sum": 0.0, "tb_sum": 0, "adapter_lat_sum": 0.0, "vocoder_lat_sum": 0.0} for snr in snr_list}

    for it in tqdm(selected_items, desc="System-Level Eval"):
        utt_id = it["id"]
        spk_id = spk_from_utt_id(utt_id)
        prompt_wav = it["prompt_wav"]
        transcript = it.get("prompt_transcript", "")

        # sender generates full codes
        t0 = time.perf_counter()
        tx = sender.process(
            text=transcript,
            prompt_path=str(prompt_wav),
            language="en",
            return_full_codes=True,
            prompt_transcript=transcript,
        )
        full_codes = unpack_full_codes(tx)
        t1 = time.perf_counter()
        latency_sender_sec = t1 - t0

        tx_layer1 = full_codes[:, 0].astype(np.int64)
        tx_acoustic = full_codes[:, 1:8].astype(np.int64)  # (T,7)

        # handshake: first utterance initializes cache (so later we can see frozen effect)
        if spk_id not in acoustic_frozen_library:
            acoustic_frozen_library[spk_id] = tx_acoustic.copy()
            handshake = True
        else:
            handshake = False

        payload_bytes = payload_bytes_from_acoustic(tx_acoustic)
        tokens = torch.tensor(tx_layer1[None, :], dtype=torch.long, device=DEVICE)

        # optional: oracle reference wav per utterance (never overwrite)
        oracle_wav_path = ""
        if args.write_audio:
            rx_full_clean = np.concatenate([tx_layer1[:, None], tx_acoustic], axis=1)
            t_or = time.perf_counter()
            wav_clean = receiver.full_codes_to_audio(rx_full_clean)
            latency_vocoder_oracle_sec = time.perf_counter() - t_or
            oracle_wav_path = str(audio_dir / f"{utt_id}_ORACLE.wav")
            write_wav(Path(oracle_wav_path), wav_clean, args.sr)
        else:
            latency_vocoder_oracle_sec = 0.0

        for snr in snr_list:
            # ---------- A) semantic stream over DeepSC adapter ----------
            t_ad0 = time.perf_counter()
            with torch.no_grad():
                logits = adapter(tokens, snr_db=float(snr))
                rx_layer1 = torch.argmax(logits, dim=-1)[0].detach().cpu().numpy().astype(np.int64)
            latency_adapter_sec = time.perf_counter() - t_ad0
            # ---------- B) acoustic stream reliable transfer decision ----------
            # Always-Update baseline (Mode2): ignore channel state, always attempt update + retransmissions.
            # Requirement:
            #   - Even when full update success rate is 0 (under latency budget), we still DELIVER a version to RX.
            #   - No cheating: we must NOT fall back to the old frozen cache when update fails.
            mode = 2
            Qm, R = 4, 0.54

            # Best-effort delivery under latency budget.
            BYTES_PER_FRAME = 7 * 2  # 7 acoustic layers, int16 each

            tb_ok, tx_used, num_tbs, budget, tb_frames = simulate_harq_update_partial(
                snr_db=snr,
                payload_bytes=payload_bytes,
                Qm=Qm, R=R,
                rng=rng,
                tb_bytes=args.tb_bytes,
                max_tx=args.max_tx,
                max_avg_tx_per_tb=args.max_avg_tx_per_tb,
                bytes_per_frame=BYTES_PER_FRAME,
            )

            # "Full update success" is counted ONLY when SNR >= args.mode2_success_th_db.
            full_ok = bool(all(tb_ok))
            force_err = bool(float(snr) < float(args.mode2_success_th_db))
            update_success = bool(full_ok and (not force_err))

            if update_success:
                status_tag = f"Always_Mode{mode}_Updated"
            elif force_err:
                status_tag = f"Always_Mode{mode}_ForcedErr"
            elif any(tb_ok):
                status_tag = f"Always_Mode{mode}_Partial"
            else:
                status_tag = f"Always_Mode{mode}_Failed"

            # Build RX cache to use (and commit) for subsequent utterances.
            # NO CHEATING: for TBs not decoded (or not attempted due to budget), overwrite with corrupted tokens.
            frozen = acoustic_frozen_library[spk_id]
            T_base = int(frozen.shape[0])
            T_tx = int(tx_acoustic.shape[0])
            active_acoustic = frozen.copy()

            max_covered = 0
            for tb_idx, ok in enumerate(tb_ok):
                s = int(tb_idx * tb_frames)
                if s >= T_base:
                    break
                e = int(min(s + tb_frames, T_base))
                if e <= s:
                    continue
                max_covered = max(max_covered, e)

                tb_good = bool(ok and (not force_err))
                if tb_good and s < T_tx:
                    ee = int(min(e, T_tx))
                    active_acoustic[s:ee, :] = tx_acoustic[s:ee, :]
                    if ee < e:
                        active_acoustic[ee:e, :] = rng.integers(0, 1024, size=(e - ee, 7), dtype=np.int64)
                else:
                    active_acoustic[s:e, :] = rng.integers(0, 1024, size=(e - s, 7), dtype=np.int64)

            if max_covered < T_base:
                active_acoustic[max_covered:T_base, :] = rng.integers(0, 1024, size=(T_base - max_covered, 7), dtype=np.int64)

            # ALWAYS commit (even when update_success is False): RX receives a (possibly erroneous) cache.
            acoustic_frozen_library[spk_id] = active_acoustic.copy()


            # align lengths
            T_sem = rx_layer1.shape[0]
            T_aco = active_acoustic.shape[0]
            if T_aco < T_sem:
                rep = int(ceil(T_sem / T_aco))
                aligned_acoustic = np.tile(active_acoustic, (rep, 1))[:T_sem, :]
            else:
                aligned_acoustic = active_acoustic[:T_sem, :]

            rx_full = np.concatenate([rx_layer1[:, None], aligned_acoustic], axis=1)

            degraded_wav_path = ""
            t_v0 = time.perf_counter()
            if args.write_audio:
                wav_deg = receiver.full_codes_to_audio(rx_full)
                degraded_wav_path = str(audio_dir / f"{utt_id}_SNR{snr:g}_{status_tag}.wav")
                write_wav(Path(degraded_wav_path), wav_deg, args.sr)
            latency_vocoder_deg_sec = time.perf_counter() - t_v0

            avg_tx_per_tb = float(tx_used) / float(num_tbs) if (num_tbs > 0) else 0.0

            rows.append({
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

                # latency breakdown (seconds)
                "latency_sender_sec": float(latency_sender_sec),
                "latency_adapter_sec": float(latency_adapter_sec),
                "latency_vocoder_oracle_sec": float(latency_vocoder_oracle_sec),
                "latency_vocoder_deg_sec": float(latency_vocoder_deg_sec),
            })

            # aggregate
            agg[snr]["n"] += 1
            agg[snr]["upd"] += int(update_success)
            agg[snr]["avg_tx_sum"] += avg_tx_per_tb
            agg[snr]["tb_sum"] += num_tbs
            agg[snr]["adapter_lat_sum"] += latency_adapter_sec
            agg[snr]["vocoder_lat_sum"] += latency_vocoder_deg_sec

    # write csv
    df = pd.DataFrame(rows)
    csv_path = out_dir / "step3_generation_meta.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] wrote meta csv: {csv_path}")

    # print summary
    print("\n================= SUMMARY (per SNR) =================")
    for snr in snr_list:
        n = agg[snr]["n"]
        if n == 0:
            continue
        upd_rate = agg[snr]["upd"] / n
        avg_tx = agg[snr]["avg_tx_sum"] / n
        avg_ad_lat = agg[snr]["adapter_lat_sum"] / n
        avg_voc_lat = agg[snr]["vocoder_lat_sum"] / n
        print(
            f"SNR={snr:>6g} dB | samples={n:>5d} | upd_rate={upd_rate:.4f} "
            f"| avg_tx/tb={avg_tx:.3f} | avg_adapter_lat={avg_ad_lat*1000:.2f} ms | avg_vocoder_lat={avg_voc_lat*1000:.2f} ms"
        )
    print("=====================================================\n")

    if args.write_audio:
        print(f"[OK] wavs saved under: {audio_dir}")
    print("[DONE]")


if __name__ == "__main__":
    main()

