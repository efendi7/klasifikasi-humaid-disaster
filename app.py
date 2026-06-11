"""
app.py
------
Entry point aplikasi Streamlit.
Hanya berisi konfigurasi halaman, header, dan orkestrasi tab.
Semua logika ada di modul terpisah.

Jalankan dengan:
    streamlit run app.py
"""

import streamlit as st

from model import load_model_and_tokenizer
from components.sidebar import render_sidebar
from components.tabs.single import render_tab_single
from components.tabs.batch import render_tab_batch


# ── CSS global ─────────────────────────────────────────────────────────────────

_GLOBAL_CSS = """
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
    font-size: 13px;
}
div[data-testid="stDownloadButton"] button {
    width: 100%;
    border-radius: 8px;
    font-size: 13px;
}
</style>
"""

_HEADER_HTML = """
<div style="padding: 20px 0 10px;">
  <div style="display:flex;align-items:flex-start;gap:16px;">
    <div style="background:linear-gradient(135deg,#E74C3C,#C0392B);
                border-radius:12px;padding:12px;flex-shrink:0;">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
           stroke="white" stroke-width="2.5" stroke-linecap="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
        <line x1="12" y1="9" x2="12" y2="13"/>
        <line x1="12" y1="17" x2="12.01" y2="17"/>
      </svg>
    </div>
    <div>
      <div style="font-size:11px;color:#888;text-transform:uppercase;
                  letter-spacing:0.08em;margin-bottom:4px;">
        Sistem Klasifikasi Informasi Bencana
      </div>
      <div style="font-size:20px;font-weight:700;color:#eee;line-height:1.3;">
        Implementasi Arsitektur RoBERTa Berbasis BiLSTM dan Cross-Attention<br>
        dengan Focal Loss pada Klasifikasi Multi-Kelas Tweet Bencana
      </div>
      <div style="font-size:12px;color:#888;margin-top:6px;">
        by <span style="color:#aaa;font-weight:500;">Muhammad Ma'mun Efendi</span>
        &nbsp;&bull;&nbsp; Model D (Arsitektur Penuh) &nbsp;&bull;&nbsp;
        RoBERTa + BiLSTM + Cross-Attention + Focal Loss
      </div>
    </div>
  </div>
</div>
"""

_FOOTER_HTML = """
<div style="text-align:center;padding:20px 0 8px;
            border-top:1px solid #333;margin-top:24px;">
  <div style="font-size:11px;color:#555;">
    Implementasi Arsitektur RoBERTa Berbasis BiLSTM dan Cross-Attention
    dengan Focal Loss pada Klasifikasi Multi-Kelas Tweet Bencana
  </div>
  <div style="font-size:11px;color:#444;margin-top:4px;">
    Muhammad Ma'mun Efendi &nbsp;&bull;&nbsp; Universitas Negeri Semarang
  </div>
</div>
"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Klasifikasi Tweet Bencana",
        page_icon=(
            "https://raw.githubusercontent.com/microsoft/fluentui-emoji"
            "/main/assets/Warning/3D/warning_3d.png"
        ),
        layout="wide",
    )

    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown(_HEADER_HTML, unsafe_allow_html=True)
    st.divider()

    tokenizer, model, id2label = load_model_and_tokenizer()

    render_sidebar()

    tab_single, tab_batch = st.tabs(["  Input Tunggal  ", "  Input Batch (CSV)  "])

    with tab_single:
        render_tab_single(tokenizer, model, id2label)

    with tab_batch:
        render_tab_batch(tokenizer, model, id2label)

    st.markdown(_FOOTER_HTML, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
