from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FEATURE_META, PRIMARY_FEATURES, THRESHOLD
from inference import load_artifacts, predict

_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

  /* ── base ── */
  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stApp"] {
    background-color: #f5f6f8 !important;
    color: #111827 !important;
    font-family: "Inter", "Segoe UI", system-ui, sans-serif !important;
  }

  /* main content area sits on white */
  [data-testid="stMain"],
  [data-testid="stMainBlockContainer"],
  .main .block-container {
    background-color: #ffffff !important;
    color: #111827 !important;
  }

  [data-testid="stHeader"] { background: transparent !important; }

  .block-container {
    max-width: 1080px !important;
    padding: 2.5rem 2.5rem 5rem !important;
  }

  /* ── typography helpers ── */
  .page-title {
    font-size: 1.45rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: -0.025em;
    margin: 0 0 0.3rem;
  }
  .page-subtitle {
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.6;
    margin: 0 0 2rem;
  }
  .section-label {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 1.75rem 0 0.6rem;
  }

  hr { border: none !important; border-top: 1px solid #e2e8f0 !important; margin: 1.75rem 0 !important; }

  /* ── ALL widget labels — force dark text ── */
  label,
  label p,
  [data-testid="stWidgetLabel"],
  [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] > div,
  [data-testid="stWidgetLabel"] > div > p,
  .stNumberInput label,
  .stSelectbox label,
  .stTextInput label {
    color: #1e293b !important;
    font-size: 0.76rem !important;
    font-weight: 500 !important;
  }

  /* ── number inputs ── */
  /* outer baseweb wrapper — owns the border, transparent bg so inner controls show */
  .stNumberInput [data-baseweb="input"],
  div[data-testid="stNumberInput"] [data-baseweb="input"] {
    background-color: transparent !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    padding: 0 !important;
    overflow: hidden;
  }
  .stNumberInput [data-baseweb="input"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.12) !important;
  }
  /* the actual input element — light bg + dark text */
  .stNumberInput input[type="number"],
  .stNumberInput [data-baseweb="input"] input {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    padding: 0.38rem 0.6rem !important;
    caret-color: #0f172a !important;
  }
  /* step buttons */
  .stNumberInput button,
  .stNumberInput [data-baseweb="input"] button {
    background-color: #f1f5f9 !important;
    color: #334155 !important;
    border: none !important;
    border-left: 1px solid #e2e8f0 !important;
  }
  .stNumberInput button:hover {
    background-color: #e2e8f0 !important;
  }

  /* ── selectbox ── */
  .stSelectbox > div > div[data-baseweb="select"] > div,
  .stSelectbox [data-baseweb="select"] > div {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 6px !important;
    font-size: 0.84rem !important;
  }
  /* dropdown option text */
  [data-baseweb="menu"] li,
  [role="option"] {
    color: #0f172a !important;
    background: #ffffff !important;
    font-size: 0.84rem !important;
  }
  [role="option"]:hover { background: #f1f5f9 !important; }

  /* ── expander ── */
  [data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: #fafafa !important;
    overflow: hidden !important;
  }
  [data-testid="stExpander"] summary,
  [data-testid="stExpander"] summary p {
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
  }
  /* content area inside expander */
  [data-testid="stExpander"] > div:last-child {
    background: #fafafa !important;
    padding: 0.5rem 0.75rem 1rem !important;
  }

  /* ── primary button ── */
  .stButton > button[kind="primary"],
  button[data-testid="baseButton-primary"] {
    background: #0f172a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 1.75rem !important;
    letter-spacing: 0.02em !important;
    transition: background 0.15s ease, transform 0.1s ease !important;
  }
  .stButton > button[kind="primary"]:hover,
  button[data-testid="baseButton-primary"]:hover {
    background: #1e293b !important;
    transform: translateY(-1px) !important;
  }

  /* ── result card ── */
  .result-card {
    border-radius: 10px;
    padding: 1.1rem 1.4rem;
    margin: 1rem 0 0.5rem;
  }
  .result-card.sepsis {
    background: #fff1f2;
    border: 1.5px solid #fda4af;
  }
  .result-card.clear {
    background: #f0fdf4;
    border: 1.5px solid #86efac;
  }
  .result-title {
    font-size: 0.95rem;
    font-weight: 700;
    margin: 0 0 0.2rem;
  }
  .result-card.sepsis .result-title { color: #be123c; }
  .result-card.clear  .result-title { color: #15803d; }
  .result-sub { font-size: 0.76rem; color: #475569; margin: 0; }

  /* ── confidence badge inside result card ── */
  .confidence {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    margin: 0.3rem 0 0;
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
  }
  .result-card.sepsis .confidence { color: #be123c; }
  .result-card.clear  .confidence { color: #15803d; }
</style>
"""


def _fmt(step: float) -> str:
    if step >= 1.0:
        return "%.0f"
    return f"%.{max(0, -int(math.floor(math.log10(step))))}f"


def feature_input(name: str, meta: dict, default: float | int) -> float | int:
    unit = meta["unit"]
    label = f"{meta['label']}  {('(' + unit + ')') if unit else ''}".strip()

    if meta["type"] == "binary":
        return int(st.selectbox(label, options=[0, 1], index=int(default), key=f"inp_{name}"))

    return float(
        st.number_input(
            label,
            min_value=float(meta["min"]),
            max_value=float(meta["max"]),
            value=float(default),
            step=float(meta["step"]),
            format=_fmt(meta["step"]),
            key=f"inp_{name}",
        )
    )


def render_result(proba: float, predicted: bool) -> None:
    card_cls = "sepsis" if predicted else "clear"
    verdict = "Sepsis" if predicted else "No Sepsis"
    confidence = f"{proba * 100:.10f}%"

    st.markdown(
        f"""
        <div class="result-card {card_cls}">
          <p class="result-title">{verdict}</p>
          <p class="confidence">{confidence} <span class="result-sub">confidence</span></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Sepsis Early Warning", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    st.markdown('<p class="page-title">Sepsis Early Warning System Prototype</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-subtitle">'
        "Fitur klinis utama merupakan fitur biologis terpenting untuk mendeteksi sepsis"
        "</p>",
        unsafe_allow_html=True,
    )

    try:
        model, scaler, feature_names = load_artifacts()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    unknown = [f for f in feature_names if f not in FEATURE_META]
    if unknown:
        st.error(f"Fitur tidak dikenali dalam metadata: {unknown}")
        st.stop()

    input_values: dict[str, float | int] = {}

    st.markdown('<p class="section-label">Fitur Klinis Utama</p>', unsafe_allow_html=True)
    cols = st.columns(4)
    for i, name in enumerate(PRIMARY_FEATURES):
        with cols[i % 4]:
            input_values[name] = feature_input(name, FEATURE_META[name], FEATURE_META[name]["default_sepsis"])

    st.markdown("<hr>", unsafe_allow_html=True)

    hidden = [f for f in feature_names if f not in PRIMARY_FEATURES]
    with st.expander(f"Fitur Tersembunyi  ({len(hidden)} fitur)"):
        st.markdown(
            '<p style="font-size:0.74rem;color:#64748b;margin:0.25rem 0 0.9rem;">'
            "Diisi dengan nilai fisiologis normal. Ubah jika diperlukan."
            "</p>",
            unsafe_allow_html=True,
        )
        hcols = st.columns(4)
        for i, name in enumerate(hidden):
            with hcols[i % 4]:
                input_values[name] = feature_input(name, FEATURE_META[name], FEATURE_META[name]["default_neutral"])

    st.markdown("<hr>", unsafe_allow_html=True)

    if st.button("Prediksi Sepsis", type="primary", use_container_width=True):
        proba, X = predict(model, scaler, input_values, feature_names)

        if not np.isfinite(X).all():
            st.error("Terdapat nilai tidak valid (NaN atau Inf). Periksa kembali input.")
            st.stop()

        render_result(proba, proba >= THRESHOLD)


if __name__ == "__main__":
    main()
