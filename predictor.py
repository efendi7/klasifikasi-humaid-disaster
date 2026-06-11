"""
predictor.py
------------
Fungsi inferensi (single & batch) dan formatter teks hasil prediksi.
Tidak ada import streamlit — murni logika, mudah di-unit-test.
"""

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from config import CLASS_LABELS, CLASS_LABELS_DISPLAY, MAX_LENGTH
from model import DEVICE


# ── Tipe data hasil prediksi ───────────────────────────────────────────────────

@dataclass
class PredictionResult:
    label:          str
    confidence:     float
    entropy_norm:   float
    entropy_raw:    float
    probs:          dict[str, float]
    top3:           list[tuple[str, float]]
    tokens:         list[str]
    token_imp_norm: list[float]

    @property
    def display_label(self) -> str:
        return CLASS_LABELS_DISPLAY.get(self.label, self.label)

    @property
    def certainty_label(self) -> str:
        # Gabungkan entropy_norm + gap kelas #1 vs #2
        sorted_probs = sorted(self.probs.values(), reverse=True)
        gap = sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0)

        if self.entropy_norm < 0.30 and gap >= 0.15:
            return "Yakin"
        if self.entropy_norm < 0.65:
            return "Ragu"
        return "Tidak Yakin"

    def top_tokens(self, n: int = 5) -> list[tuple[str, float]]:
        """Kembalikan n token dengan bobot attention tertinggi (sudah bersih dari special tokens)."""
        def clean(tok: str) -> str:
            return tok.replace("Ġ", " ").replace("Ċ", " ").strip()

        pairs = [
            (clean(tok), imp)
            for tok, imp in zip(self.tokens, self.token_imp_norm)
            if tok not in ("<s>", "</s>", "<pad>") and clean(tok)
        ]
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:n]


# ── Inferensi ──────────────────────────────────────────────────────────────────

def predict_single(text: str, tokenizer, model, id2label: dict) -> PredictionResult:
    """Prediksi satu tweet. Returns PredictionResult."""
    enc = tokenizer(
        text, max_length=MAX_LENGTH, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        logits, attn_weights = model(input_ids, attention_mask)

    probs      = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred_idx   = int(np.argmax(probs))
    pred_label = id2label[pred_idx]
    confidence = float(probs[pred_idx])

    entropy_raw  = float(-np.sum(probs * np.log(probs + 1e-12)))
    entropy_norm = entropy_raw / math.log(len(probs))

    attn             = attn_weights.squeeze(0).cpu().numpy()
    token_importance = attn.mean(axis=0)
    seq_len          = int(attention_mask.sum().item())

    tokens       = tokenizer.convert_ids_to_tokens(input_ids[0, :seq_len].cpu().tolist())
    tok_imp      = token_importance[:seq_len]
    t_min, t_max = tok_imp.min(), tok_imp.max()
    tok_imp_norm = ((tok_imp - t_min) / (t_max - t_min + 1e-8)).tolist()

    top3_idx = np.argsort(probs)[::-1][:3]

    return PredictionResult(
        label          = pred_label,
        confidence     = confidence,
        entropy_norm   = entropy_norm,
        entropy_raw    = entropy_raw,
        probs          = {id2label[i]: float(probs[i]) for i in range(len(probs))},
        top3           = [(id2label[i], float(probs[i])) for i in top3_idx],
        tokens         = tokens,
        token_imp_norm = tok_imp_norm,
    )


def predict_batch(
    texts: list[str],
    tokenizer,
    model,
    id2label: dict,
    progress_callback=None,
) -> list[PredictionResult]:
    """
    Prediksi sejumlah tweet.

    Parameters
    ----------
    progress_callback : callable(current, total) | None
        Dipanggil setiap tweet selesai — untuk update progress bar.
    """
    results = []
    for i, text in enumerate(texts):
        results.append(predict_single(str(text), tokenizer, model, id2label))
        if progress_callback:
            progress_callback(i + 1, len(texts))
    return results


# ── Formatter ──────────────────────────────────────────────────────────────────

def format_result_as_text(tweet_text: str, result: PredictionResult) -> str:
    """
    Format hasil prediksi sebagai plain text untuk disalin / diunduh.

    Berisi:
    - Prediksi + confidence + entropy
    - Top-5 token paling diperhatikan model (Cross-Attention)
    - Distribusi probabilitas semua kelas dihapus (sudah ada di chart)
    """
    top_tokens = result.top_tokens(5)
    token_str  = "\n".join(
        f"  {i+1}. \"{word.strip()}\" — bobot {imp:.3f}"
        for i, (word, imp) in enumerate(top_tokens)
    )

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return (
        f"=== HASIL KLASIFIKASI TWEET BENCANA ===\n"
        f"Timestamp   : {ts}\n"
        f"Tweet       : {tweet_text}\n"
        f"-------------------------------------\n"
        f"Prediksi    : {result.display_label}\n"
        f"Confidence  : {result.confidence*100:.1f}%\n"
        f"Entropy     : {result.entropy_norm:.3f} ({result.certainty_label})\n"
        f"-------------------------------------\n"
        f"Top-5 Kata Paling Diperhatikan (Cross-Attention):\n"
        f"{token_str}\n"
        f"-------------------------------------\n"
        f"Model       : RoBERTa + BiLSTM + Cross-Attention + Focal Loss\n"
        f"by Muhammad Ma'mun Efendi\n"
        f"======================================\n"
    )
