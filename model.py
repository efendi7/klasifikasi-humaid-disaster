"""
model.py
--------
Definisi arsitektur Model D (RoBERTa + BiLSTM + Cross-Attention)
dan fungsi loader yang di-cache oleh Streamlit.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import streamlit as st
from transformers import AutoTokenizer, AutoModel

from config import (
    ROBERTA_NAME, NUM_CLASSES, HIDDEN_DIM, DROPOUT,
    MODEL_PATH, LABEL_MAP_PATH, CLASS_LABELS,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Arsitektur ─────────────────────────────────────────────────────────────────

class RoBERTaCrossAttn(nn.Module):
    def __init__(
        self,
        model_name: str = ROBERTA_NAME,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()

        self.roberta    = AutoModel.from_pretrained(model_name)
        bert_h          = self.roberta.config.hidden_size

        self.bilstm     = nn.LSTM(
            bert_h, hidden_dim, num_layers=2,
            batch_first=True, bidirectional=True, dropout=dropout,
        )
        self.proj_lstm  = nn.Linear(hidden_dim * 2, bert_h)

        num_heads       = 8 if bert_h % 8 == 0 else 4
        self.cross_attn = nn.MultiheadAttention(
            bert_h, num_heads, dropout=dropout, batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(bert_h),
            nn.Linear(bert_h, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        bert_out            = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state

        lstm_out, _         = self.bilstm(bert_out)
        lstm_feat           = self.drop(self.proj_lstm(lstm_out))

        key_pad_mask        = attention_mask == 0
        fused, attn_weights = self.cross_attn(
            query=lstm_feat, key=bert_out, value=bert_out,
            key_padding_mask=key_pad_mask,
            need_weights=True, average_attn_weights=True,
        )

        mask_exp = attention_mask.unsqueeze(-1).float()
        pooled   = (fused * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

        return self.head(pooled), attn_weights


# ── Loader (cached) ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Memuat model RoBERTa-BiLSTM-CrossAttn… (hanya sekali)")
def load_model_and_tokenizer() -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_NAME)

    if LABEL_MAP_PATH.exists():
        with open(LABEL_MAP_PATH) as f:
            lm = json.load(f)
        id2label: dict[int, str] = {int(k): v for k, v in lm["id2label"].items()}
    else:
        id2label = dict(enumerate(CLASS_LABELS))

    model = RoBERTaCrossAttn(ROBERTA_NAME, len(id2label), HIDDEN_DIM, DROPOUT)

    # FIX: MODEL_PATH sekarang string (dari hf_hub_download), bukan Path
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        st.error(f"Checkpoint tidak ditemukan: `{model_path}`")
        st.stop()

    ckpt  = torch.load(model_path, map_location=DEVICE)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.to(DEVICE).eval()

    return tokenizer, model, id2label