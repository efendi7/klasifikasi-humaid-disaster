"""
components/result_card.py
-------------------------
Render kartu hasil prediksi lengkap untuk tab Input Tunggal.

Layout order:
  1. Header prediksi (label + confidence)
  2. [col kiri] Probabilitas semua kelas | [col kanan] Metrik Ketidakpastian (Entropy)
  3. Ringkasan naratif — "Mengapa model memilih kelas ini?"
  4. Token highlight (Cross-Attention)
  5. Preview teks + tombol unduh  ← paling bawah
"""

from datetime import datetime

import streamlit as st

from config import CLASS_COLORS, CLASS_LABELS_DISPLAY
from predictor import PredictionResult, format_result_as_text
from components.charts import (
    render_probability_chart,
    render_token_highlight,
    render_entropy_card,
)


# ── Sub-renderers ──────────────────────────────────────────────────────────────

def _render_prediction_header(result: PredictionResult) -> None:
    """Header besar berisi label prediksi dan confidence score."""
    color = CLASS_COLORS.get(result.label, "#2C3E50")
    st.markdown(
        f'<div style="background:{color}22;border:1.5px solid {color}66;'
        f'border-radius:12px;padding:16px 20px;margin-bottom:16px;">'
        f'<div style="font-size:12px;color:#aaa;margin-bottom:4px;'
        f'text-transform:uppercase;letter-spacing:0.05em;">Hasil Prediksi</div>'
        f'<div style="font-size:24px;font-weight:700;color:{color};margin-bottom:6px;">'
        f'{result.display_label}</div>'
        f'<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">'
        f'<span style="font-size:14px;color:#ccc;">Confidence: '
        f'<strong style="color:{color};font-size:16px;">{result.confidence*100:.1f}%</strong></span>'
        f'<span style="font-size:12px;background:{color}33;color:{color};'
        f'padding:3px 10px;border-radius:20px;">{result.label}</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def _render_why_summary(result: PredictionResult) -> None:
    """
    Ringkasan naratif mengapa model memilih kelas ini.

    Catatan: RoBERTa (sebagai encoder) hanya menghasilkan representasi
    kontekstual — ia tidak menghasilkan penjelasan kausal secara langsung.
    Penjelasan di bawah adalah *post-hoc approximation* berbasis:
      - Confidence score (dari distribusi softmax)
      - Entropy (ketidakpastian distribusi)
      - Bobot Cross-Attention (kata mana yang paling diperhatikan model)
    """
    color      = CLASS_COLORS.get(result.label, "#2C3E50")
    top_tokens = result.top_tokens(3)
    token_str  = ", ".join(f'"{w.strip()}"' for w, _ in top_tokens)

    # Kalimat confidence
    conf_pct = result.confidence * 100
    if conf_pct >= 80:
        conf_phrase = f"dengan keyakinan tinggi ({conf_pct:.1f}%)"
    elif conf_pct >= 50:
        conf_phrase = f"dengan keyakinan sedang ({conf_pct:.1f}%)"
    else:
        conf_phrase = f"meski dengan keyakinan rendah ({conf_pct:.1f}%)"

    # Kalimat entropy — gabungkan entropy_norm DAN gap kelas #1 vs #2
    # agar narasi tidak kontradiktif saat dua kelas hampir seri
    sorted_probs = sorted(result.probs.values(), reverse=True)
    top1_prob    = sorted_probs[0]
    top2_prob    = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    gap          = top1_prob - top2_prob          # selisih kelas 1 vs 2

    cert = result.certainty_label

    if gap < 0.10:
        # Dua kelas hampir seri — override apapun label entropy-nya
        ent_phrase = (
            f"Namun distribusi probabilitas hampir seri antara dua kelas teratas "
            f"(selisih hanya {gap*100:.1f}%) — prediksi ini perlu diinterpretasikan "
            f"dengan hati-hati karena model sangat ragu di antara dua kemungkinan."
        )
    elif cert == "Yakin" and gap >= 0.20:
        ent_phrase = "Distribusi probabilitas terpusat kuat pada kelas ini, menunjukkan model tidak mempertimbangkan kelas lain secara signifikan."
    elif cert == "Yakin":
        ent_phrase = "Distribusi probabilitas cukup terpusat, meski kelas runner-up masih mendapat sebagian probabilitas."
    elif cert == "Ragu":
        ent_phrase = "Model masih mempertimbangkan beberapa kelas lain — interpretasi perlu dilakukan dengan hati-hati."
    else:
        ent_phrase = "Distribusi probabilitas menyebar merata; teks mungkin ambigu atau tidak umum dalam data latih."

    # Kalimat attention
    if top_tokens:
        attn_phrase = f"Kata-kata yang paling diperhatikan model (via Cross-Attention): {token_str}."
    else:
        attn_phrase = "Bobot Cross-Attention tidak tersedia untuk teks ini."

    # Peringatan epistemis
    note = (
        "<em>Catatan: RoBERTa sebagai encoder hanya <strong>memprediksi</strong>, "
        "bukan menjelaskan secara kausal. Ringkasan ini adalah aproksimasi post-hoc "
        "berdasarkan confidence, entropy, dan bobot Cross-Attention — bukan ground truth penjelasan model.</em>"
    )

    st.markdown(
        f'<div style="background:#111;border-left:3px solid {color};'
        f'border-radius:0 10px 10px 0;padding:16px 18px;margin:4px 0 4px;">'

        f'<div style="font-size:12px;color:#888;text-transform:uppercase;'
        f'letter-spacing:0.07em;margin-bottom:10px;">📋 Mengapa Model Memilih Kelas Ini?</div>'

        f'<div style="font-size:14px;color:#e0e0e0;line-height:1.8;margin-bottom:12px;">'
        f'Model mengklasifikasikan tweet ini sebagai '
        f'<strong style="color:{color};">{result.display_label}</strong> '
        f'{conf_phrase}. {ent_phrase} {attn_phrase}'
        f'</div>'

        f'<div style="font-size:11px;color:#666;line-height:1.6;border-top:1px solid #2a2a2a;padding-top:8px;">'
        f'{note}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _render_export_area(tweet_text: str, result: PredictionResult) -> None:
    """Preview teks full-width + tombol unduh — selalu di paling bawah."""
    # 1. Ambil format awal dari predictor
    base_result_text = format_result_as_text(tweet_text, result)

    # 2. Tambahkan daftar probabilitas semua kelas
    probs_text = "\n\n=== Probabilitas Semua Kelas ===\n"
    # Mengurutkan probabilitas dari yang tertinggi ke terendah
    sorted_probs = sorted(result.probs.items(), key=lambda item: item[1], reverse=True)
    for class_label, prob in sorted_probs:
        probs_text += f"- {class_label}: {prob * 100:.2f}%\n"

    # Gabungkan menjadi satu teks utuh
    result_text = base_result_text + probs_text

    st.markdown(
        '<div style="font-size:13px;font-weight:600;color:#ccc;margin-bottom:6px;">'
        'Ringkasan Hasil (salin atau unduh)'
        '</div>',
        unsafe_allow_html=True,
    )
    
    # 3. Tampilkan di text area (height sedikit ditambah agar muat lebih banyak baris)
    st.text_area(
        label="preview",
        value=result_text,
        height=260, 
        label_visibility="collapsed",
        key=f"preview_{hash(tweet_text)}",
    )

    col_dl, _ = st.columns([1, 3])
    with col_dl:
        st.download_button(
            label="⬇ Unduh Hasil (.txt)",
            data=result_text.encode("utf-8"),
            file_name=f"hasil_prediksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

# ── Public API ─────────────────────────────────────────────────────────────────

def render_result_card(tweet_text: str, result: PredictionResult) -> None:
    """
    Render kartu hasil prediksi lengkap.

    Urutan layout:
        header prediksi
        ├── [col kiri]  probabilitas semua kelas
        └── [col kanan] metrik ketidakpastian (entropy card — diperbesar)
        ringkasan naratif "Mengapa model memilih kelas ini?"
        token highlight (Cross-Attention)
        preview teks + unduh  ← paling bawah
    """
    _render_prediction_header(result)
    st.divider()

    col_chart, col_meta = st.columns([1.2, 1])
    with col_chart:
        st.markdown("**Probabilitas Semua Kelas**")
        render_probability_chart(result)
    with col_meta:
        render_entropy_card(result)

    st.divider()
    _render_why_summary(result)
    st.divider()

    st.markdown("**Penjelasan: Kata yang Paling Diperhatikan Model (Cross-Attention)**")
    st.caption(
        "Warna merah menunjukkan bobot Cross-Attention — "
        "kata lebih gelap = kontribusi lebih besar terhadap keputusan kelas ini."
    )
    render_token_highlight(result)

    st.divider()
    _render_export_area(tweet_text, result)
