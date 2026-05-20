"""
Singleton predictor that loads the model once at startup and exposes predict().

Thread safety: the model is set to eval() after loading and never mutated.
torch.no_grad() is applied per-call. Concurrent reads are safe.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# hanta_model.py lives at the API project root (one level above app/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hanta_model import MultiTaskHantaBERT  # noqa: E402

from app.config import settings
from app.schemas import ClassScore, PredictResponse, TaskPrediction

logger = logging.getLogger(__name__)


class HantaPredictor:
    def __init__(self) -> None:
        self._loaded = False
        self._error: str | None = None
        self.label_maps: dict[str, List[str]] = {}
        self.device: torch.device | None = None
        self.tokenizer = None
        self.model: MultiTaskHantaBERT | None = None

    # ── Startup ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        try:
            logger.info("Loading label maps from %s", settings.label_maps_path)
            with open(settings.label_maps_path, "r") as f:
                self.label_maps = json.load(f)

            n_species = len(self.label_maps["species"])
            n_host    = len(self.label_maps["host"])
            n_geo     = len(self.label_maps["geo"])

            if settings.device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(settings.device)
            logger.info("Using device: %s", self.device)

            logger.info("Loading tokenizer from %s", settings.dnabert_model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.dnabert_model_path, trust_remote_code=True
            )

            logger.info(
                "Instantiating MultiTaskHantaBERT(n_species=%d, n_host=%d, n_geo=%d)",
                n_species, n_host, n_geo,
            )
            self.model = MultiTaskHantaBERT(n_species, n_host, n_geo)

            weights_path = os.path.realpath(settings.model_weights_path)
            logger.info("Loading weights from %s", weights_path)
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self._loaded = True
            logger.info("Model loaded successfully.")

        except Exception as exc:
            self._error = str(exc)
            logger.exception("Failed to load model: %s", exc)
            raise

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._error

    # ── Inference ───────────────────────────────────────────────────────────

    def predict(self, sequence: str, top_n: int = 3) -> PredictResponse:
        if not self._loaded:
            raise RuntimeError("Model is not loaded.")

        clean_seq = sequence.upper().replace("U", "T")
        original_len = len(clean_seq)

        enc = self.tokenizer(
            clean_seq,
            max_length=settings.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        raw_ids = self.tokenizer(clean_seq, truncation=False)["input_ids"]
        was_truncated = len(raw_ids) > settings.max_length

        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        return PredictResponse(
            species=self._decode_task(outputs["species_logits"], self.label_maps["species"], top_n),
            host=self._decode_task(outputs["host_logits"],    self.label_maps["host"],    top_n),
            geo=self._decode_task(outputs["geo_logits"],      self.label_maps["geo"],     top_n),
            sequence_length=original_len,
            truncated=was_truncated,
        )

    # ── Helper ───────────────────────────────────────────────────────────────

    @staticmethod
    def _decode_task(
        logits: torch.Tensor,
        labels: List[str],
        top_n: int,
    ) -> TaskPrediction:
        probs = F.softmax(logits[0], dim=-1).cpu().tolist()
        n_safe = min(top_n, len(labels))
        ranked = sorted(zip(labels, probs), key=lambda x: x[1], reverse=True)
        top_class, top_conf = ranked[0]
        return TaskPrediction(
            predicted=top_class,
            confidence=round(top_conf, 6),
            top_n=[
                ClassScore(label=lbl, confidence=round(conf, 6))
                for lbl, conf in ranked[:n_safe]
            ],
        )


predictor = HantaPredictor()
