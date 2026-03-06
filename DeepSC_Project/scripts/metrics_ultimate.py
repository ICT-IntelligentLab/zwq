#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: WeiqingZhu
Paper: GenAI-Enabled Dual-Stream Speech Semantic Communication under Dynamic Channels and Latency Constraints
Copyright: WeiqingZhu
Note: Please retain this attribution notice in any reuse of this code.

Compute objective and downstream metrics from step3 system outputs.
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Objective metrics
from pystoi import stoi
from pesq import pesq
import jiwer

# ASR + Speaker verification
from transformers import pipeline
from speechbrain.inference.speaker import SpeakerRecognition

# Plot
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Utilities: robust audio I/O
# ----------------------------
def load_audio_mono_resample(wav_path: str, target_sr: int = 16000):
    """
    Load wav -> mono -> resample to target_sr.
    Returns:
      wav_t: torch.Tensor shape [1, T] float32
      wav_np: np.ndarray shape [T] float32
      sr: int (target_sr)
    """
    wav_path = str(wav_path)
    wav, sr = torchaudio.load(wav_path)  # [C, T], float32
    if wav.ndim != 2:
        raise RuntimeError(f"Unexpected wav shape={wav.shape} for {wav_path}")

    # force mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)

    wav = wav.contiguous()
    wav_np = wav.squeeze(0).cpu().numpy().astype(np.float32)
    return wav, wav_np, target_sr


def align_min_len(a: np.ndarray, b: np.ndarray):
    """Align by truncating to min length."""
    n = min(len(a), len(b))
    return a[:n], b[:n], n


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


# ----------------------------
# Mode parsing / grouping
# ----------------------------
def status_to_mode(status_tag: str) -> int:
    s = (status_tag or "").lower()
    if "mode2" in s:
        return 2
    if "mode1" in s:
        return 1
    return 0


# ----------------------------
# Keying + resume
# ----------------------------
def make_row_key(row: dict) -> str:
    """
    A stable key to support resume/caching.
    Use degraded_wav path + snr + status_tag as primary identity.
    """
    degraded = str(row.get("degraded_wav", ""))
    snr = row.get("snr", row.get("snr_db", ""))
    status = str(row.get("status_tag", ""))
    spk = str(row.get("spk_id", ""))
    return f"{spk}||{snr}||{status}||{degraded}"


def load_existing_results(out_csv: Path):
    if out_csv.exists():
        df_old = pd.read_csv(out_csv)
        done = set(df_old["row_key"].astype(str).tolist()) if "row_key" in df_old.columns else set()
        return df_old, done
    return None, set()


def append_one_row(out_csv: Path, row_dict: dict):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df1 = pd.DataFrame([row_dict])
    if out_csv.exists():
        df1.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        df1.to_csv(out_csv, mode="w", header=True, index=False)


# ----------------------------
# Text normalization (WER)
# ----------------------------
WER_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])


# ----------------------------
# Plot helpers
# ----------------------------
def plot_metric_vs_snr(agg: pd.DataFrame, metric_col: str, ylabel: str, title: str, out_path: Path, is_percent=False):
    """
    Plot metric mean vs SNR with separate curves for mode=0/1/2.
    agg must contain columns: snr_db, mode, metric_col
    """
    fig = plt.figure()
    for mode in sorted(agg["mode"].unique()):
        sub = agg[agg["mode"] == mode].sort_values("snr_db")
        x = sub["snr_db"].to_numpy()
        y = sub[metric_col].to_numpy()
        if is_percent:
            y = 100.0 * y
        # default matplotlib color cycle (no manual colors)
        plt.plot(x, y, marker="o", label=f"mode={int(mode)}")

    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def plot_and_save_all(agg: pd.DataFrame, plot_dir: Path, prefix: str, formats=("pdf", "png")):
    """
    Save a set of standard system-level curves:
      - STOI_mean vs SNR
      - PESQ_mean vs SNR
      - WER_mean vs SNR (%)
      - SIM_pd_mean vs SNR
      - Lat_mean vs SNR (optional)
    """
    plot_dir.mkdir(parents=True, exist_ok=True)

    items = [
        ("STOI_mean", "STOI", "STOI vs SNR (system-level)", False),
        ("PESQ_mean", "PESQ", "PESQ vs SNR (system-level)", False),
        ("WER_mean", "WER (%)", "WER vs SNR (system-level)", True),
        ("SIM_pd_mean", "Speaker SIM (prompt->degraded)", "Speaker Similarity vs SNR", False),
    ]

    # latency curve (optional if exists)
    if "Lat_mean" in agg.columns:
        items.append(("Lat_mean", "Latency (s)", "End-to-end Latency vs SNR", False))

    for metric, ylabel, title, is_percent in items:
        if metric not in agg.columns:
            continue
        for fmt in formats:
            out_path = plot_dir / f"{prefix}_{metric}_vs_snr.{fmt}"
            plot_metric_vs_snr(
                agg=agg,
                metric_col=metric,
                ylabel=ylabel,
                title=title,
                out_path=out_path,
                is_percent=is_percent
            )


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True, help="Path to step3_generation_meta.csv")
    ap.add_argument("--out_csv", default="results/step4_final_metrics.csv", help="Detailed per-sample metrics CSV (appendable)")
    ap.add_argument("--summary_csv", default="results/step4_summary_table.csv", help="Aggregated summary table CSV")
    ap.add_argument("--summary_tex", default="results/step4_summary_table.tex", help="Aggregated summary LaTeX table")

    # plotting
    ap.add_argument("--plot", action="store_true", help="If set, generate plots after summary.")
    ap.add_argument("--plot_dir", default="results/plots_step4", help="Where to save plot files.")
    ap.add_argument("--plot_prefix", default="step4", help="Prefix for plot filenames.")
    ap.add_argument("--plot_formats", default="pdf,png", help="Comma separated formats, e.g. pdf,png")

    # audio + metrics configs
    ap.add_argument("--sr", type=int, default=16000, help="Resample SR for metrics/ASR/Speaker models")
    ap.add_argument("--min_dur_sec", type=float, default=0.25, help="Too-short audio => skip PESQ/STOI (keep NaN)")
    ap.add_argument("--pesq_mode", type=str, default="wb", choices=["wb", "nb"], help="PESQ mode. Use wb for 16k.")
    ap.add_argument("--whisper_model", type=str, default="openai/whisper-base", help="ASR model id")
    ap.add_argument("--speechbrain_source", type=str, default="speechbrain/spkrec-ecapa-voxceleb", help="Speaker model id")

    # speed knobs
    ap.add_argument("--max_rows", type=int, default=-1, help="Debug: limit number of rows to evaluate")
    ap.add_argument("--resume", action="store_true", help="Resume if out_csv exists (skip evaluated rows)")
    ap.add_argument("--skip_asr", action="store_true", help="Skip WER (faster)")
    ap.add_argument("--skip_pesq", action="store_true", help="Skip PESQ (faster)")
    ap.add_argument("--skip_stoi", action="store_true", help="Skip STOI (faster)")
    ap.add_argument("--skip_spk", action="store_true", help="Skip speaker similarity (faster)")
    ap.add_argument("--cache_asr_jsonl", default="results/cache_whisper_asr.jsonl", help="Cache file for ASR outputs by wav path")
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    out_csv = Path(args.out_csv)
    summary_csv = Path(args.summary_csv)
    summary_tex = Path(args.summary_tex)
    cache_asr_jsonl = Path(args.cache_asr_jsonl)

    assert csv_path.exists(), f"missing {csv_path}"

    df = pd.read_csv(csv_path)

    # ---------- Strict filtering ----------
    df = df[df["degraded_wav"].notna() & (df["degraded_wav"] != "")]
    df = df[df["oracle_wav"].notna() & (df["oracle_wav"] != "")]
    df = df[df["prompt_wav"].notna() & (df["prompt_wav"] != "")]

    def _exists(p): 
        return isinstance(p, str) and (p.strip() != "") and Path(p).exists()

    df = df[df["degraded_wav"].apply(_exists) & df["oracle_wav"].apply(_exists) & df["prompt_wav"].apply(_exists)]

    # normalize snr column name
    if "snr_db" in df.columns and "snr" not in df.columns:
        df["snr"] = df["snr_db"]

    # derive mode
    if "status_tag" not in df.columns:
        df["status_tag"] = ""
    df["mode"] = df["status_tag"].apply(status_to_mode)

    # optionally limit
    if args.max_rows > 0:
        df = df.head(args.max_rows)

    print(f"[INFO] Loaded {len(df)} valid rows from: {csv_path}")

    # ---------- Resume ----------
    done_keys = set()
    if args.resume:
        _, done_keys = load_existing_results(out_csv)
        if done_keys:
            print(f"[INFO] Resume enabled. Already done rows: {len(done_keys)}")

    # ---------- Load ASR cache ----------
    asr_cache = {}
    if (not args.skip_asr) and cache_asr_jsonl.exists():
        with open(cache_asr_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    asr_cache[str(obj["wav_path"])] = str(obj["text"])
                except Exception:
                    pass
        print(f"[INFO] Loaded ASR cache entries: {len(asr_cache)}")

    # ---------- Init models ----------
    asr_pipe = None
    verification = None

    if not args.skip_asr:
        print(f"[INIT] Loading ASR model: {args.whisper_model} (DEVICE={DEVICE})")
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=args.whisper_model,
            device=0 if DEVICE == "cuda" else -1,
        )

    if not args.skip_spk:
        print(f"[INIT] Loading Speaker model: {args.speechbrain_source} (DEVICE={DEVICE})")
        verification = SpeakerRecognition.from_hparams(
            source=args.speechbrain_source,
            savedir="tmp_speechbrain",
            run_opts={"device": DEVICE},
        )

    # ---------- Evaluate ----------
    min_len_samples = int(args.min_dur_sec * args.sr)
    n_eval = 0
    n_skip = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Step4 Metrics"):
        row_dict = row.to_dict()
        row_key = make_row_key(row_dict)
        row_dict["row_key"] = row_key

        if args.resume and (row_key in done_keys):
            n_skip += 1
            continue

        degraded_path = str(row_dict["degraded_wav"])
        oracle_path = str(row_dict["oracle_wav"])
        prompt_path = str(row_dict["prompt_wav"])
        gt_text = str(row_dict.get("transcript", "")).strip()

        # -------- load audio (mono + resample) --------
        try:
            _, deg_np, _ = load_audio_mono_resample(degraded_path, target_sr=args.sr)
            _, ora_np, _ = load_audio_mono_resample(oracle_path, target_sr=args.sr)
        except Exception as e:
            row_dict.update({
                "STOI": np.nan, "PESQ": np.nan, "WER": np.nan,
                "SIM_prompt_oracle": np.nan, "SIM_prompt_degraded": np.nan, "SIM_oracle_degraded": np.nan,
                "asr_text": "",
                "err": f"audio_load_failed: {type(e).__name__}: {e}"
            })
            append_one_row(out_csv, row_dict)
            done_keys.add(row_key)
            n_eval += 1
            continue

        # align for STOI/PESQ
        deg_align, ora_align, L = align_min_len(deg_np, ora_np)

        # durations
        row_dict["dur_degraded_sec"] = float(len(deg_np) / args.sr)
        row_dict["dur_oracle_sec"] = float(len(ora_np) / args.sr)
        row_dict["aligned_len"] = int(L)

        # -------- STOI / PESQ --------
        val_stoi = np.nan
        val_pesq = np.nan

        if L >= min_len_samples:
            if not args.skip_stoi:
                try:
                    val_stoi = float(stoi(ora_align, deg_align, args.sr, extended=False))
                except Exception:
                    val_stoi = np.nan

            if not args.skip_pesq:
                try:
                    val_pesq = float(pesq(args.sr, ora_align, deg_align, args.pesq_mode))
                except Exception:
                    val_pesq = np.nan

        row_dict["STOI"] = val_stoi
        row_dict["PESQ"] = val_pesq

        # -------- ASR / WER --------
        asr_text = ""
        wer_val = np.nan

        if not args.skip_asr:
            try:
                if degraded_path in asr_cache:
                    asr_text = asr_cache[degraded_path]
                else:
                    out = asr_pipe({"array": deg_np.astype(np.float32), "sampling_rate": args.sr})
                    asr_text = str(out["text"]).strip()

                    asr_cache[degraded_path] = asr_text
                    cache_asr_jsonl.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_asr_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"wav_path": degraded_path, "text": asr_text}, ensure_ascii=False) + "\n")

                if gt_text.strip():
                    wer_val = float(jiwer.wer(WER_TRANSFORM(gt_text), WER_TRANSFORM(asr_text)))
                else:
                    wer_val = np.nan
            except Exception as e:
                asr_text = ""
                wer_val = np.nan
                row_dict["asr_err"] = f"{type(e).__name__}: {e}"

        row_dict["asr_text"] = asr_text
        row_dict["WER"] = wer_val

        # -------- Speaker similarity --------
        sim_po = np.nan
        sim_pd = np.nan
        sim_od = np.nan

        if not args.skip_spk:
            try:
                s1, _ = verification.verify_files(prompt_path, oracle_path)
                s2, _ = verification.verify_files(prompt_path, degraded_path)
                s3, _ = verification.verify_files(oracle_path, degraded_path)
                sim_po = float(s1.item())
                sim_pd = float(s2.item())
                sim_od = float(s3.item())
            except Exception as e:
                row_dict["spk_err"] = f"{type(e).__name__}: {e}"

        row_dict["SIM_prompt_oracle"] = sim_po
        row_dict["SIM_prompt_degraded"] = sim_pd
        row_dict["SIM_oracle_degraded"] = sim_od

        # -------- latency (from step3 if exists) --------
        row_dict["latency_adapter_sec"] = safe_float(row_dict.get("latency_adapter_sec", np.nan))
        row_dict["latency_vocoder_sec"] = safe_float(row_dict.get("latency_vocoder_sec", row_dict.get("latency_vocoder_deg_sec", np.nan)))
        if np.isfinite(row_dict["latency_adapter_sec"]) and np.isfinite(row_dict["latency_vocoder_sec"]):
            row_dict["latency_total_sec"] = row_dict["latency_adapter_sec"] + row_dict["latency_vocoder_sec"]
        else:
            row_dict["latency_total_sec"] = np.nan

        row_dict["err"] = ""

        append_one_row(out_csv, row_dict)
        done_keys.add(row_key)
        n_eval += 1

    print(f"\n[DONE] Evaluated newly: {n_eval}, skipped(resume): {n_skip}")
    print(f"[DONE] Detailed CSV saved/updated: {out_csv}")

    # ---------- Aggregate summary ----------
    out_df = pd.read_csv(out_csv)

    if "snr_db" not in out_df.columns and "snr" in out_df.columns:
        out_df["snr_db"] = out_df["snr"]

    if "mode" not in out_df.columns and "status_tag" in out_df.columns:
        out_df["mode"] = out_df["status_tag"].apply(status_to_mode)

    agg = out_df.groupby(["snr_db", "mode"]).agg(
        N=("row_key", "count"),
        STOI_mean=("STOI", "mean"),
        STOI_std=("STOI", "std"),
        PESQ_mean=("PESQ", "mean"),
        PESQ_std=("PESQ", "std"),
        WER_mean=("WER", "mean"),
        WER_std=("WER", "std"),
        SIM_po_mean=("SIM_prompt_oracle", "mean"),
        SIM_po_std=("SIM_prompt_oracle", "std"),
        SIM_pd_mean=("SIM_prompt_degraded", "mean"),
        SIM_pd_std=("SIM_prompt_degraded", "std"),
        SIM_od_mean=("SIM_oracle_degraded", "mean"),
        SIM_od_std=("SIM_oracle_degraded", "std"),
        Lat_mean=("latency_total_sec", "mean"),
        Lat_std=("latency_total_sec", "std"),
    ).reset_index()

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(summary_csv, index=False)
    print(f"[DONE] Summary CSV: {summary_csv}")

    # ---------- LaTeX table ----------
    def fmt(x, nd=3):
        if pd.isna(x):
            return "--"
        return f"{x:.{nd}f}"

    summary_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_tex, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{System-level end-to-end speech evaluation aggregated by SNR and operating mode.}\n")
        f.write("\\label{tab:step4_system_metrics}\n")
        f.write("\\begin{tabular}{c|c|c|c|c|c|c}\n\\hline\n")
        f.write("SNR(dB) & mode & $N$ & STOI$\\uparrow$ & PESQ$\\uparrow$ & WER$\\downarrow$ & SIM(p\\!\\rightarrow\\!d)$\\uparrow$ \\\\\n\\hline\n")
        for _, r in agg.iterrows():
            snr = r["snr_db"]
            mode = int(r["mode"])
            N = int(r["N"])
            stoi_m = fmt(r["STOI_mean"], 4)
            pesq_m = fmt(r["PESQ_mean"], 3)
            wer_m = "--" if pd.isna(r["WER_mean"]) else f"{100.0 * r['WER_mean']:.2f}\\%"
            sim_pd_m = fmt(r["SIM_pd_mean"], 3)
            f.write(f"{snr:g} & {mode:d} & {N:d} & {stoi_m} & {pesq_m} & {wer_m} & {sim_pd_m} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"[DONE] Summary LaTeX table: {summary_tex}")

    # ---------- Plot ----------
    if args.plot:
        plot_dir = Path(args.plot_dir)
        fmts = tuple([x.strip() for x in args.plot_formats.split(",") if x.strip()])
        plot_and_save_all(agg=agg, plot_dir=plot_dir, prefix=args.plot_prefix, formats=fmts)
        print(f"[DONE] Plots saved to: {plot_dir}  (formats={fmts})")

    # ---------- Console preview ----------
    print("\n================= Summary Preview (mean) =================")
    show_cols = ["snr_db", "mode", "N", "STOI_mean", "PESQ_mean", "WER_mean", "SIM_pd_mean", "Lat_mean"]
    show_cols = [c for c in show_cols if c in agg.columns]
    print(agg[show_cols].sort_values(["snr_db", "mode"]).to_string(index=False))
    print("==========================================================\n")


if __name__ == "__main__":
    main()

