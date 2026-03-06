"""
Author: WeiqingZhu
Paper: GenAI-Enabled Dual-Stream Speech Semantic Communication under Dynamic Channels and Latency Constraints
Copyright: WeiqingZhu
Note: Please retain this attribution notice in any reuse of this code.
"""

import glob
import os
import sys

import numpy as np
from tqdm import tqdm


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from vallex_wrapper import SemanticSender


PROMPT_DIR = "/root/autodl-tmp/VALL-E-X/prompts"
TEXT_CORPUS_PATH = "/root/autodl-tmp/VALL-E-X/unique_text_corpus.npy"
TRAIN_SAVE = os.path.join(config.DATA_DIR, "train_unique_1k.npy")
TEST_SAVE = os.path.join(config.DATA_DIR, "test_unique_1k.npy")


def mine():
    wav_files = sorted(glob.glob(os.path.join(PROMPT_DIR, "*.wav")))
    total_speakers = len(wav_files)
    if total_speakers == 0:
        print(f"[ERR] No prompt audio found in {PROMPT_DIR}.")
        return

    if not os.path.exists(TEXT_CORPUS_PATH):
        print(f"[ERR] Text corpus not found: {TEXT_CORPUS_PATH}")
        return

    text_corpus = np.load(TEXT_CORPUS_PATH, allow_pickle=True)
    total_texts = len(text_corpus)
    texts_per_speaker = total_texts // total_speakers

    print(f"[INFO] prompts={total_speakers}, texts={total_texts}")
    print(f"[INFO] texts_per_speaker={texts_per_speaker}")

    sender = SemanticSender()
    all_data_train = []
    all_data_test = []

    for index, wav_path in enumerate(wav_files):
        filename = os.path.basename(wav_path)
        try:
            spk_id = int(filename.split("_")[1])
        except (IndexError, ValueError):
            spk_id = 1 if index < 24 else 9

        is_train = spk_id <= 8
        dataset_type = "TRAIN" if is_train else "TEST"

        start_idx = index * texts_per_speaker
        end_idx = total_texts if index == total_speakers - 1 else start_idx + texts_per_speaker
        my_texts = text_corpus[start_idx:end_idx]

        print(
            f"[INFO] [{index + 1}/{total_speakers}] {filename} -> {dataset_type} "
            f"texts={start_idx}:{end_idx}"
        )

        for text in tqdm(my_texts, desc=f"Mining {filename}"):
            try:
                packet = sender.process(text, wav_path)
                tokens = packet["layer1_codes"].numpy().flatten()
                if len(tokens) > 10:
                    if is_train:
                        all_data_train.append(tokens)
                    else:
                        all_data_test.append(tokens)
            except Exception:
                continue

    np.save(TRAIN_SAVE, np.array(all_data_train, dtype=object))
    np.save(TEST_SAVE, np.array(all_data_test, dtype=object))

    print("[DONE] Dataset mining finished.")
    print(f"[DONE] Train samples: {len(all_data_train)} -> {TRAIN_SAVE}")
    print(f"[DONE] Test samples: {len(all_data_test)} -> {TEST_SAVE}")


if __name__ == "__main__":
    mine()
