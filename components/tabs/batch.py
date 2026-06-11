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

    # ── Simpan ke session_state saat file baru di-upload ──────────────────────
    if uploaded is not None:
        df_parsed = _read_csv(uploaded)
        if df_parsed is not None:
            # Reset hasil lama jika file baru di-upload
            if st.session_state.get("batch_filename") != uploaded.name:
                st.session_state.pop("batch_result", None)
            st.session_state["batch_df"]       = df_parsed
            st.session_state["batch_filename"] = uploaded.name

    df_input = st.session_state.get("batch_df", None)

    if df_input is None:
        return

    st.success(f"File berhasil dibaca — {len(df_input):,} tweet ditemukan.")
    st.dataframe(df_input.head(5), use_container_width=True)
    if len(df_input) > 5:
        st.caption(f"Menampilkan 5 dari {len(df_input):,} baris. Semua baris akan diproses.")

    # ── Tombol proses ──────────────────────────────────────────────────────────
    if st.button(f"Proses {len(df_input):,} Tweet", type="primary"):
        results   = _run_batch(df_input["tweet_text"].tolist(), tokenizer, model, id2label)
        df_result = _build_result_df(df_input, results, id2label)
        st.session_state["batch_result"] = df_result

    df_result = st.session_state.get("batch_result", None)

    if df_result is None:
        return

    # ── Tampilkan hasil ────────────────────────────────────────────────────────
    st.success(f"Selesai! {len(df_result):,} tweet diproses.")
    _render_distribution_chart(df_result)
    _render_preview_table(df_result)
    st.divider()
    _render_export_buttons(df_result)


# ── Private helpers ────────────────────────────────────────────────────────────

# Nama kolom yang diterima sebagai teks tweet (urutan prioritas)
_CANDIDATE_COLUMNS = ["tweet_text", "text", "tweet", "content", "message"]


def _read_csv(uploaded_file) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Gagal membaca CSV: {exc}")
        return None

    # Cari kolom teks secara otomatis
    found_col = next((c for c in _CANDIDATE_COLUMNS if c in df.columns), None)

    if found_col is None:
        st.error(
            f"Kolom teks tidak ditemukan. "
            f"Pastikan CSV memiliki salah satu kolom berikut: {_CANDIDATE_COLUMNS}. "
            f"Kolom tersedia: {list(df.columns)}"
        )
        return None

    # Normalisasi: rename ke tweet_text agar sisa kode konsisten
    if found_col != "tweet_text":
        df = df.rename(columns={found_col: "tweet_text"})
        st.info(f"Kolom `{found_col}` dideteksi dan digunakan sebagai teks tweet.")

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
            "keterangan":         res.certainty_label,
            "entropy_normalized": round(res.entropy_norm, 4),
            "top2_label":         CLASS_LABELS_DISPLAY.get(top3_labels[1], top3_labels[1]) if len(top3_labels) > 1 else "",
            "top2_prob":          round(top3_probs[1], 4) if len(top3_probs) > 1 else 0,
            "top3_label":         CLASS_LABELS_DISPLAY.get(top3_labels[2], top3_labels[2]) if len(top3_labels) > 2 else "",
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

    fig = go.Figure(go.Pie(
        labels=dist["Kelas"],
        values=dist["Jumlah"],
        marker_colors=[_display_to_color(c) for c in dist["Kelas"]],
        textinfo="label+percent",
        hovertemplate="%{label}<br>%{value} tweet (%{percent})<extra></extra>",
        hole=0.35,
    ))
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_preview_table(df_result: pd.DataFrame) -> None:
    st.markdown("#### Preview Hasil (10 baris pertama)")
    preview_cols = [
        "tweet_text", "predicted_display", "confidence",
        "keterangan", "entropy_normalized", "top2_label", "top2_prob",
    ]
    visible = [c for c in preview_cols if c in df_result.columns]
    st.dataframe(df_result[visible].head(10), use_container_width=True, height=420)


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

    # ── Header & distribusi ────────────────────────────────────────────────────
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
    ]

    # ── Detail per baris ──────────────────────────────────────────────────────
    lines.append("Detail Per Tweet:")
    lines.append("")
    for i, row in df_result.iterrows():
        tweet   = str(row.get("tweet_text", "")).strip()
        label   = row.get("predicted_display", "-")
        conf    = float(row.get("confidence", 0)) * 100
        ket     = row.get("keterangan", "-")
        top2    = row.get("top2_label", "-")
        top2p   = float(row.get("top2_prob", 0)) * 100
        lines += [
            f"[{i+1:>3}] {tweet[:120]}{'...' if len(tweet) > 120 else ''}",
            f"       Prediksi  : {label} ({conf:.1f}%) — {ket}",
            f"       Alternatif: {top2} ({top2p:.1f}%)",
            "",
        ]

    lines += [
        "-------------------------------------",
        "Model       : RoBERTa + BiLSTM + Cross-Attention + Focal Loss",
        "by Muhammad Ma'mun Efendi",
        "==========================================",
    ]
    return "\n".join(lines)