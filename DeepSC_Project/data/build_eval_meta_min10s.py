"""
Author: WeiqingZhu
Paper: GenAI-Enabled Dual-Stream Speech Semantic Communication under Dynamic Channels and Latency Constraints
Copyright: WeiqingZhu
Note: Please retain this attribution notice in any reuse of this code.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_ids(path):
    ids = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            item = line.strip()
            if item:
                ids.append(item)
    return ids


def crop_prompt_wav(src_wav, out_wav, prompt_sec=3.0, target_sr=24000):
    wav, sr = sf.read(str(src_wav), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)

    if sr != target_sr:
        raise RuntimeError(
            f"SR mismatch for {src_wav}: sr={sr}, expected {target_sr}. "
            "Please resample the source audio before building prompts."
        )

    sample_count = int(prompt_sec * target_sr)
    if len(wav) < sample_count:
        pad = np.zeros(sample_count - len(wav), dtype=np.float32)
        wav = np.concatenate([wav, pad], axis=0)
    else:
        wav = wav[:sample_count]

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), wav, target_sr, format="WAV", subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split_ids", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_utts", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt_sec", type=float, default=3.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    wav_dir = out_dir / "prompts_wav"
    meta_path = out_dir / "prompts_meta.jsonl"
    ids_path = out_dir / "eval_ids.txt"
    out_dir.mkdir(parents=True, exist_ok=True)

    split_ids = load_ids(args.split_ids)
    if len(split_ids) < args.num_utts:
        raise RuntimeError(f"split_ids too small: {len(split_ids)} < {args.num_utts}")

    random.seed(args.seed)
    picked = random.sample(split_ids, args.num_utts)
    ids_path.write_text("\n".join(picked) + "\n", encoding="utf-8")

    id2item = {}
    for item in read_jsonl(args.manifest):
        utt_id = item.get("id")
        if utt_id in picked:
            id2item[utt_id] = {
                "src_audio": item.get("src_audio"),
                "transcript": item.get("transcript", "").strip(),
            }

    missing = [utt_id for utt_id in picked if utt_id not in id2item]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} ids in manifest, example: {missing[:5]}")

    with open(meta_path, "w", encoding="utf-8") as file:
        for utt_id in picked:
            src_wav = Path(id2item[utt_id]["src_audio"])
            transcript = id2item[utt_id]["transcript"]
            out_wav = wav_dir / f"{utt_id}.wav"
            crop_prompt_wav(
                src_wav,
                out_wav,
                prompt_sec=args.prompt_sec,
                target_sr=24000,
            )
            record = {
                "id": utt_id,
                "prompt_wav": str(out_wav.resolve()),
                "prompt_transcript": transcript,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("[OK] out_dir =", out_dir)
    print("[OK] eval_ids =", ids_path, "count=", len(picked))
    print("[OK] meta =", meta_path)
    print("[OK] wav_dir =", wav_dir)


if __name__ == "__main__":
    main()
