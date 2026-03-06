# -*- coding: utf-8 -*-
"""
Author: WeiqingZhu
Paper: GenAI-Enabled Dual-Stream Speech Semantic Communication under Dynamic Channels and Latency Constraints
Copyright: WeiqingZhu
Note: Please retain this attribution notice in any reuse of this code.

Thin VALL-E X wrapper used by the DeepSC pipeline.
"""

import logging
import os
import sys
from typing import Any, Dict, Optional, Union

import numpy as np
import torch


logger = logging.getLogger("ValleMiddleware")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
_VALLEX_ROOT_DEFAULT = os.path.abspath(os.path.join(_PROJECT_ROOT, ".."))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

VALLEX_ROOT = _VALLEX_ROOT_DEFAULT
try:
    import config  # type: ignore

    if hasattr(config, "VALLEX_ROOT"):
        VALLEX_ROOT = os.path.abspath(getattr(config, "VALLEX_ROOT"))
except Exception:
    config = None  # type: ignore

if VALLEX_ROOT not in sys.path:
    sys.path.insert(0, VALLEX_ROOT)

try:
    import utils.generation as gen_pkg  # type: ignore
    from utils.generation import (  # type: ignore
        code2lang,
        lang2token,
        preload_models,
        text_collater,
        text_tokenizer,
    )
    from utils.prompt_making import make_prompt  # type: ignore
except Exception as exc:
    raise ImportError(
        f"Failed to import VALL-E-X modules from VALLEX_ROOT={VALLEX_ROOT}. "
        f"Make sure DeepSC_Project is inside the VALL-E-X repo and `utils/` exists. "
        f"Original: {repr(exc)}"
    )


class ValleEngine:
    _initialized = False

    @staticmethod
    def init():
        if ValleEngine._initialized:
            return

        logger.info("Starting VALL-E X engine.")
        old_cwd = os.getcwd()
        os.chdir(VALLEX_ROOT)
        try:
            preload_models()
        finally:
            os.chdir(old_cwd)

        if getattr(gen_pkg, "vocos", None) is None:
            raise RuntimeError(
                "utils.generation.preload_models() finished but gen_pkg.vocos is None. "
                "Check that Vocos is installed and preload_models() completed."
            )

        if getattr(gen_pkg, "model", None) is None:
            raise RuntimeError(
                "utils.generation.preload_models() finished but gen_pkg.model is None. "
                "Check that vallex-checkpoint.pt exists under VALL-E-X/checkpoints/."
            )

        ValleEngine._initialized = True
        logger.info("VALL-E X engine ready.")


class SemanticSender:
    """Generate acoustic tokens with VALL-E X."""

    def __init__(self):
        ValleEngine.init()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prompt_cache: Dict[str, Any] = {}

    def _load_or_make_prompt(self, prompt_path: str, transcript: Optional[str] = None):
        abs_prompt_path = os.path.abspath(prompt_path)
        if abs_prompt_path in self.prompt_cache:
            return self.prompt_cache[abs_prompt_path]

        if not os.path.exists(abs_prompt_path):
            raise FileNotFoundError(f"Prompt not found: {abs_prompt_path}")

        old_cwd = os.getcwd()
        os.chdir(VALLEX_ROOT)
        try:
            make_prompt(
                name="tx_prompt",
                audio_prompt_path=abs_prompt_path,
                transcript=transcript,
            )
            prompt_file = os.path.join("customs", "tx_prompt.npz")
            prompt_data = np.load(prompt_file, allow_pickle=True)

            audio_prompts = (
                torch.tensor(prompt_data["audio_tokens"]).to(self.device).to(torch.int32)
            )
            text_prompts = (
                torch.tensor(prompt_data["text_tokens"]).to(self.device).to(torch.int32)
            )
            lang_pr = code2lang[int(prompt_data["lang_code"])]
            enroll_x_lens = int(text_prompts.shape[-1])

            pack = (audio_prompts, text_prompts, lang_pr, enroll_x_lens)
            self.prompt_cache[abs_prompt_path] = pack
            return pack
        finally:
            os.chdir(old_cwd)

    @torch.no_grad()
    def process(
        self,
        text: str,
        prompt_path: str,
        language: str = "en",
        return_full_codes: bool = True,
        prompt_transcript: Optional[str] = None,
    ) -> Dict[str, Any]:
        audio_prompts, text_prompts, lang_pr, enroll_x_lens = self._load_or_make_prompt(
            prompt_path,
            transcript=prompt_transcript,
        )

        text = text.replace("\n", "").strip(" ")
        lang_token = lang2token[language]
        text_full = lang_token + text + lang_token

        phone_tokens, langs = text_tokenizer.tokenize(text=f"_{text_full}".strip())
        text_tokens, text_tokens_lens = text_collater([phone_tokens])
        text_tokens = text_tokens.to(self.device)
        text_tokens_lens = text_tokens_lens.to(self.device)

        if len(langs) != int(text_tokens.shape[1]):
            diff = int(text_tokens.shape[1]) - len(langs)
            if diff > 0:
                last_lang = langs[-1] if langs else "en"
                langs = langs + [last_lang] * diff
            else:
                langs = langs[: int(text_tokens.shape[1])]

        text_tokens = torch.cat([text_prompts, text_tokens], dim=-1)
        text_tokens_lens += enroll_x_lens

        full_codes = gen_pkg.model.inference(
            text_tokens,
            text_tokens_lens,
            audio_prompts,
            enroll_x_lens=enroll_x_lens,
            top_k=-100,
            temperature=1.0,
            prompt_language=lang_pr,
            text_language=langs,
        )

        output = {
            "layer1_codes": full_codes[:, :, 0].detach().cpu(),
            "audio_prompts": audio_prompts.detach().cpu(),
        }
        if return_full_codes:
            output["full_codes"] = full_codes.detach().cpu()
        return output


class SemanticReceiver:
    """Decode full 8-layer RVQ codes to waveform through Vocos."""

    def __init__(self):
        ValleEngine.init()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def full_codes_to_audio(
        self,
        full_codes: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        if isinstance(full_codes, np.ndarray):
            codes = torch.from_numpy(full_codes).long()
        else:
            codes = full_codes.long()

        if codes.ndim == 2:
            if codes.shape[-1] == 8:
                codes = codes.unsqueeze(0)
            elif codes.shape[0] == 8:
                codes = codes.transpose(0, 1).unsqueeze(0)
            else:
                raise ValueError(
                    f"2D codes must be (T,8) or (8,T), got {tuple(codes.shape)}"
                )
        elif codes.ndim == 3:
            if codes.shape[-1] == 8:
                pass
            elif codes.shape[1] == 8:
                codes = codes.transpose(1, 2)
            else:
                raise ValueError(
                    f"3D codes must be (B,T,8) or (B,8,T), got {tuple(codes.shape)}"
                )
        else:
            raise ValueError(f"codes.ndim must be 2 or 3, got {codes.ndim}")

        codes = codes.to(self.device)
        vocos = getattr(gen_pkg, "vocos", None)
        if vocos is None:
            raise RuntimeError("gen_pkg.vocos is None; preload_models() did not initialize vocos.")

        frames = codes.permute(2, 0, 1).contiguous()
        features = vocos.codes_to_features(frames)
        samples = vocos.decode(features, bandwidth_id=torch.tensor([2], device=self.device))
        return samples.squeeze().detach().cpu().numpy().astype(np.float32)
