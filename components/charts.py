"""
components/charts.py
--------------------
Fungsi render chart Plotly dan visualisasi token attention.
Semua fungsi menerima data murni (tidak bergantung pada session state).
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from config import CLASS_LABELS_DISPLAY, CLASS_COLORS
from predictor import PredictionResult


# ── Probability bar chart ──────────────────────────────────────────────────────

def render_probability_chart(result: PredictionResult) -> None:
    """Horizontal bar chart semua kelas, highlight kelas prediksi."""
    labels   = list(result.probs.keys())
    values   = [result.probs[l] for l in labels]
    displays = [CLASS_LABELS_DISPLAY.get(l, l) for l in labels]
    colors   = [
        "#E74C3C" if l == result.label else CLASS_COLORS.get(l, "#7F8C8D")
        for l in labels
    ]
    order = np.argsort(values)[::-1]

    fig = go.Figure(go.Bar(
        x=[values[i] for i in order],
        y=[displays[i] for i in order],
        orientation="h",
        marker_color=[colors[i] for i in order],
        text=[f"{values[i]*100:.1f}%" for i in order],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Probabilitas: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis=dict(
            range=[0, min(1.2, max(values) * 1.35)],
            tickformat=".0%",
            title="Probabilitas",
        ),
        yaxis=dict(automargin=True),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Token highlight ────────────────────────────────────────────────────────────

def render_token_highlight(result: PredictionResult) -> None:
    """Tampilkan token dengan warna berdasarkan bobot cross-attention."""

    def imp_to_rgba(imp: float) -> str:
        alpha = 0.08 + imp * 0.72
        return f"rgba(231,76,60,{alpha:.2f})"

    def clean_token(tok: str) -> str:
        return tok.replace("Ġ", " ").replace("Ċ", " ").strip()

    spans = []
    for tok, imp in zip(result.tokens, result.token_imp_norm):
        if tok in ("<s>", "</s>", "<pad>"):
            continue
        word = clean_token(tok)
        if not word:
            continue
        fw = "600" if imp > 0.6 else "400"
        spans.append(
            f'<span title="attention: {imp:.3f}" style="'
            f'background:{imp_to_rgba(imp)};border-radius:4px;'
            f'padding:2px 5px;margin:2px;display:inline-block;font-weight:{fw};">'
            f'{word}</span>'
        )

    legend = (
        '<div style="font-size:11px;color:#888;margin-top:4px;">'
        '<span style="background:rgba(231,76,60,0.08);padding:2px 7px;border-radius:3px;">rendah</span> '
        '&#8594; '
        '<span style="background:rgba(231,76,60,0.44);padding:2px 7px;border-radius:3px;">sedang</span> '
        '&#8594; '
        '<span style="background:rgba(231,76,60,0.80);padding:2px 7px;border-radius:3px;font-weight:600;">tinggi</span>'
        ' &nbsp; intensitas = bobot Cross-Attention'
        '</div>'
    )

    html = (
        '<div style="line-height:2.4;font-size:15px;padding:10px 0;">'
        + "".join(spans)
        + "</div>"
        + legend
    )
    st.markdown(html, unsafe_allow_html=True)


# ── Entropy card (full-width, lebih besar) ────────────────────────────────────

def render_entropy_card(result: PredictionResult) -> None:
    """Card metrik ketidakpastian model — full width, diperbesar."""
    entropy_norm = result.entropy_norm
    label        = result.certainty_label

    color_map = {
        "Yakin":       "#27AE60",
        "Ragu":        "#F39C12",
        "Tidak Yakin": "#E74C3C",
    }
    icon_map = {
        "Yakin":       "✅",
        "Ragu":        "⚠️",
        "Tidak Yakin": "❌",
    }
    desc_map = {
        "Yakin":       "Distribusi probabilitas sangat terpusat pada satu kelas. Model mengenali pola teks ini dengan baik.",
        "Ragu":        "Distribusi menyebar ke beberapa kelas. Model mempertimbangkan lebih dari satu kemungkinan.",
        "Tidak Yakin": "Distribusi probabilitas hampir merata di semua kelas. Teks mungkin ambigu atau di luar distribusi training.",
    }
    color   = color_map[label]
    icon    = icon_map[label]
    desc    = desc_map[label]
    bar_pct = int(entropy_norm * 100)

    # Entropy raw info
    entropy_raw = result.entropy_raw
    max_entropy = 2.303  # ln(10) untuk 10 kelas

    st.markdown(
        f'<div style="background:#1a1a1a;border:1.5px solid {color}55;'
        f'border-radius:14px;padding:20px 24px;margin-top:4px;">'

        # Judul
        f'<div style="font-size:11px;color:#888;text-transform:uppercase;'
        f'letter-spacing:0.08em;margin-bottom:12px;">Metrik Ketidakpastian Model (Entropy)</div>'

        # Label besar + skor
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">'
        f'<span style="font-size:36px;">{icon}</span>'
        f'<div>'
        f'<div style="font-size:26px;font-weight:700;color:{color};line-height:1;">{label}</div>'
        f'<div style="font-size:13px;color:#aaa;margin-top:3px;">'
        f'H = {entropy_raw:.4f} nat &nbsp;|&nbsp; H_norm = {entropy_norm:.4f}'
        f'</div>'
        f'</div>'
        f'</div>'

        # Progress bar
        f'<div style="background:#2a2a2a;border-radius:6px;height:10px;margin-bottom:10px;">'
        f'<div style="width:{bar_pct}%;background:linear-gradient(90deg,{color}88,{color});'
        f'border-radius:6px;height:10px;transition:width 0.4s;"></div>'
        f'</div>'

        # Skala
        f'<div style="display:flex;justify-content:space-between;'
        f'font-size:10px;color:#555;margin-bottom:14px;">'
        f'<span>0.000 (Pasti)</span><span>0.500</span><span>1.000 (Acak Sempurna)</span>'
        f'</div>'

        # Deskripsi
        f'<div style="font-size:13px;color:#ccc;line-height:1.7;'
        f'background:#111;border-radius:8px;padding:10px 14px;">'
        f'{desc}'
        f'</div>'

        f'</div>',
        unsafe_allow_html=True,
    )
