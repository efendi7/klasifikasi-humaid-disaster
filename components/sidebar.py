"""
components/sidebar.py
---------------------
Render sidebar kiri: info model dan daftar kelas yang didukung.
"""

import streamlit as st

from config import CLASS_LABELS, CLASS_LABELS_DISPLAY, CLASS_COLORS
from model import DEVICE, HIDDEN_DIM, DROPOUT
from config import ROBERTA_NAME, MAX_LENGTH, NUM_CLASSES


def render_sidebar() -> None:
    """Render seluruh konten sidebar."""
    with st.sidebar:
        _render_model_info()
        st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
        _render_class_list()


# ── Private helpers ────────────────────────────────────────────────────────────

def _render_model_info() -> None:
    st.markdown(
        '<div style="text-align:center;padding:12px 0 8px;">'
        '<div style="font-size:13px;font-weight:600;color:#ccc;">Informasi Model</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    rows = [
        ("Arsitektur",       "Model D (Penuh)"),
        ("Backbone",         ROBERTA_NAME),
        ("Lapisan Tambahan", "BiLSTM + Cross-Attn"),
        ("Fungsi Loss",      "Focal Loss"),
        ("Max Token",        str(MAX_LENGTH)),
        ("Jumlah Kelas",     str(NUM_CLASSES)),
        ("Device",           str(DEVICE).upper()),
    ]
    for key, val in rows:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:5px 0;border-bottom:1px solid #333;font-size:12px;">'
            f'<span style="color:#888;">{key}</span>'
            f'<span style="color:#ccc;font-weight:500;">{val}</span></div>',
            unsafe_allow_html=True,
        )


def _render_class_list() -> None:
    st.markdown(
        '<div style="font-size:12px;font-weight:600;color:#ccc;margin-bottom:8px;">'
        'Kelas yang Didukung</div>',
        unsafe_allow_html=True,
    )
    for cls in CLASS_LABELS:
        color   = CLASS_COLORS.get(cls, "#888")
        display = CLASS_LABELS_DISPLAY.get(cls, cls)
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;'
            f'padding:3px 0;font-size:11px;color:#aaa;">'
            f'<span style="width:8px;height:8px;border-radius:50%;'
            f'background:{color};flex-shrink:0;display:inline-block;"></span>'
            f'{display}</div>',
            unsafe_allow_html=True,
        )
