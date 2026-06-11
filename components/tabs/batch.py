"""
components/tabs/batch.py
------------------------
Konten tab "Input Batch (CSV)": upload, proses, distribusi, export.
"""

from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config import CLASS_LABELS_DISPLAY, CLASS_COLORS
from predictor import predict_batch, PredictionResult


def render_tab_batch(tokenizer, model, id2label: dict) -> None:
    """Render seluruh konten tab Input Batch."""
    st.markdown(
        '<div style="font-size:13px;color:#aaa;margin-bottom:12px;">'
        'Upload file CSV berisi banyak tweet sekaligus. '
        'Kolom wajib: <code>tweet_text</code>. Encoding: UTF-8.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded is None:
        return

    df_input = _read_csv(uploaded)
    if df_input is None:
        return

    st.success(f"File berhasil dibaca — {len(df_input):,} tweet ditemukan.")
    st.dataframe(df_input.head(5), use_container_width=True)

    if not st.button(f"Proses {len(df_input):,} Tweet", type="primary"):
        return

    results   = _run_batch(df_input["tweet_text"].tolist(), tokenizer, model, id2label)
    df_result = _build_result_df(df_input, results, id2label)

    st.success(f"Selesai! {len(df_result):,} tweet diproses.")
    _render_distribution_chart(df_result)
    _render_preview_table(df_result)
    st.divider()
    _render_export_buttons(df_result)


# ── Private helpers ────────────────────────────────────────────────────────────

def _read_csv(uploaded_file) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Gagal membaca CSV: {exc}")
        return None

    if "tweet_text" not in df.columns:
        st.error(
            f"Kolom `tweet_text` tidak ditemukan. "
            f"Kolom tersedia: {list(df.columns)}"
        )
        return None

    return df


def _run_batch(
    texts: list[str],
    tokenizer,
    model,
    id2label: dict,
) -> list[PredictionResult]:
    progress_bar = st.progress(0, text="Memulai…")

    def on_progress(current: int, total: int) -> None:
        progress_bar.progress(current / total, text=f"Memproses {current}/{total}…")

    with st.spinner("Memproses batch…"):
        results = predict_batch(texts, tokenizer, model, id2label, on_progress)

    progress_bar.empty()
    return results


def _build_result_df(
    df_input: pd.DataFrame,
    results: list[PredictionResult],
    id2label: dict,
) -> pd.DataFrame:
    rows = []
    for res in results:
        top3_labels = [c for c, _ in res.top3]
        top3_probs  = [p for _, p in res.top3]
        row = {
            "predicted_label":    res.label,
            "predicted_display":  res.display_label,
            "confidence":         round(res.confidence, 4),
            "entropy_normalized": round(res.entropy_norm, 4),
            "top2_label":         top3_labels[1] if len(top3_labels) > 1 else "",
            "top2_prob":          round(top3_probs[1], 4) if len(top3_probs) > 1 else 0,
            "top3_label":         top3_labels[2] if len(top3_labels) > 2 else "",
            "top3_prob":          round(top3_probs[2], 4) if len(top3_probs) > 2 else 0,
        }
        for cls, prob in res.probs.items():
            row[f"prob_{cls}"] = round(prob, 4)
        rows.append(row)

    return pd.concat(
        [df_input.reset_index(drop=True), pd.DataFrame(rows)],
        axis=1,
    )


def _render_distribution_chart(df_result: pd.DataFrame) -> None:
    st.markdown("#### Distribusi Hasil Prediksi")

    dist = df_result["predicted_display"].value_counts().reset_index()
    dist.columns = ["Kelas", "Jumlah"]
    dist["Persentase"] = (dist["Jumlah"] / len(df_result) * 100).round(1)

    def _display_to_color(display_name: str) -> str:
        key = next(
            (k for k, v in CLASS_LABELS_DISPLAY.items() if v == display_name), ""
        )
        return CLASS_COLORS.get(key, "#7F8C8D")

    fig = go.Figure(go.Bar(
        y=dist["Kelas"],
        x=dist["Jumlah"],
        orientation="h",
        text=[f"{p}%" for p in dist["Persentase"]],
        textposition="outside",
        marker_color=[_display_to_color(c) for c in dist["Kelas"]],
    ))
    fig.update_layout(
        height=350,
        margin=dict(l=10, r=60, t=10, b=10),
        xaxis_title="Jumlah Tweet",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_preview_table(df_result: pd.DataFrame) -> None:
    st.markdown("#### Preview Hasil (10 baris pertama)")
    preview_cols = [
        "tweet_text", "predicted_display", "confidence",
        "entropy_normalized", "top2_label", "top2_prob",
    ]
    visible = [c for c in preview_cols if c in df_result.columns]
    st.dataframe(df_result[visible].head(10), use_container_width=True)


def _render_export_buttons(df_result: pd.DataFrame) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    col_csv, col_txt = st.columns(2)

    with col_csv:
        st.download_button(
            label="Unduh Hasil Lengkap (CSV)",
            data=df_result.to_csv(index=False).encode("utf-8"),
            file_name=f"hasil_prediksi_{ts}.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )

    with col_txt:
        st.download_button(
            label="Unduh Ringkasan (TXT)",
            data=_build_summary_text(df_result).encode("utf-8"),
            file_name=f"ringkasan_{ts}.txt",
            mime="text/plain",
            type="secondary",
            use_container_width=True,
        )


def _build_summary_text(df_result: pd.DataFrame) -> str:
    dist = df_result["predicted_display"].value_counts().reset_index()
    dist.columns = ["Kelas", "Jumlah"]
    dist["Persentase"] = (dist["Jumlah"] / len(df_result) * 100).round(1)

    lines = [
        "=== RINGKASAN HASIL KLASIFIKASI BATCH ===",
        f"Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total tweet : {len(df_result):,}",
        "-------------------------------------",
        "Distribusi Prediksi:",
        *[
            f"  {row['Kelas']:<30} {row['Jumlah']:>4} tweet ({row['Persentase']}%)"
            for _, row in dist.iterrows()
        ],
        "-------------------------------------",
        "Model       : RoBERTa + BiLSTM + Cross-Attention + Focal Loss",
        "by Muhammad Ma'mun Efendi",
        "==========================================",
    ]
    return "\n".join(lines)
