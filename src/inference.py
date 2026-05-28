from __future__ import annotations

import joblib
import numpy as np
import streamlit as st

from config import FEATURE_NAMES_PATH, MODEL_PATH, SCALER_PATH


@st.cache_resource
def load_artifacts() -> tuple:
    missing = [p for p in (MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH) if not p.exists()]
    if missing:
        paths = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            f"Artifact tidak ditemukan:\n{paths}\n\n"
            "Jalankan cell-mod-split dan cell-mod-train di notebooks/02_modeling_sepsis.ipynb "
            "terlebih dahulu untuk membuat scaler.pkl dan memastikan model tersedia."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names: list[str] = np.load(FEATURE_NAMES_PATH, allow_pickle=True).tolist()
    return model, scaler, feature_names


def predict(model, scaler, input_values: dict[str, float | int], feature_names: list[str]) -> tuple[float, np.ndarray]:
    X = np.array([[input_values[f] for f in feature_names]], dtype=np.float64)
    X_scaled = scaler.transform(X)
    proba = float(model.predict_proba(X_scaled)[0, 1])
    return proba, X
