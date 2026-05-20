"""
Multi-task HantaBERT model — self-contained copy for HantaBERT-API.
Original: HantaBERT/model.py
Change: replaced `import config` with inline constant _MODEL_PATH.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, BertConfig

_MODEL_PATH = "zhihan1996/DNABERT-2-117M"


class MultiTaskHantaBERT(nn.Module):
    def __init__(self, n_species: int, n_host: int, n_geo: int):
        super().__init__()

        bert_config = BertConfig.from_pretrained(_MODEL_PATH)
        # Force PyTorch attention fallback — avoids Triton flash-attention OOM on non-A100 GPUs.
        bert_config.attention_probs_dropout_prob = 0.1
        self.encoder = AutoModel.from_pretrained(
            _MODEL_PATH,
            trust_remote_code=True,
            config=bert_config,
        )
        D = 768

        self.bottleneck = nn.Sequential(
            nn.Linear(D, D),
            nn.LayerNorm(D),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        self.species_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, n_species))
        self.host_head    = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, n_host))
        self.geo_head     = nn.Sequential(nn.Dropout(0.1), nn.Linear(D, n_geo))

    def _mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask   = attention_mask.unsqueeze(-1).float()
        summed = (hidden_states * mask).sum(dim=1)
        count  = mask.sum(dim=1).clamp(min=1e-9)
        return summed / count

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # DNABERT-2's custom bert_layers.py returns a raw tuple, not BaseModelOutput
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
        pooled = self._mean_pool(hidden, attention_mask)
        shared = self.bottleneck(pooled)

        return {
            "species_logits": self.species_head(shared),
            "host_logits":    self.host_head(shared),
            "geo_logits":     self.geo_head(shared),
            "embedding":      shared,
        }
