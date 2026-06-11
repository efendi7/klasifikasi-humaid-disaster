"""
components/tabs/single.py
-------------------------
Konten tab "Input Tunggal": text area, contoh tweet, tombol analisis.
"""

import streamlit as st

from config import EXAMPLE_TWEETS
from predictor import predict_single
from components.result_card import render_result_card


def render_tab_single(tokenizer, model, id2label: dict) -> None:
    """Render seluruh konten tab Input Tunggal."""
    st.markdown(
        '<div style="font-size:13px;color:#aaa;margin-bottom:12px;">'
        'Masukkan satu tweet untuk diklasifikasikan secara real-time.</div>',
        unsafe_allow_html=True,
    )

    tweet_text = _render_input_area()

    col_btn, col_clear, _ = st.columns([1, 1, 4])
    with col_btn:
        predict_clicked = st.button("Analisis Tweet", type="primary", use_container_width=True)
    with col_clear:
        st.button("Bersihkan", use_container_width=True)

    if predict_clicked:
        if not tweet_text.strip():
            st.warning("Masukkan teks tweet terlebih dahulu.")
            return

        with st.spinner("Menganalisis…"):
            result = predict_single(tweet_text.strip(), tokenizer, model, id2label)

        st.divider()
        render_result_card(tweet_text.strip(), result)


# ── Private ────────────────────────────────────────────────────────────────────

def _render_input_area() -> str:
    """Render text area + dropdown contoh tweet. Returns teks yang dimasukkan."""
    col_input, col_example = st.columns([3, 1])

    with col_example:
        st.markdown(
            '<div style="font-size:12px;color:#888;margin-bottom:4px;">Pilih contoh tweet:</div>',
            unsafe_allow_html=True,
        )
        options         = ["(tulis sendiri)"] + [f"Contoh {i+1}" for i in range(len(EXAMPLE_TWEETS))]
        selected        = st.selectbox("Contoh", options, label_visibility="collapsed")
        default_text    = (
            "" if selected == "(tulis sendiri)"
            else EXAMPLE_TWEETS[int(selected.split()[-1]) - 1]
        )

    with col_input:
        tweet_text = st.text_area(
            "Tweet",
            value=default_text,
            height=110,
            placeholder="Ketik atau tempel tweet di sini…",
            label_visibility="collapsed",
        )

    return tweet_text
