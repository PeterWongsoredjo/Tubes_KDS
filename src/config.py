from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "model" / "sepsis_mlp_model.pkl"
SCALER_PATH = ROOT / "model" / "scaler.pkl"
FEATURE_NAMES_PATH = ROOT / "data" / "feature_names.npy"

THRESHOLD = 0.0732

PRIMARY_FEATURES = ["HR", "Temp", "Lactate", "O2Sat", "SBP", "Resp", "BUN", "SF_ratio"]

FEATURE_META: dict[str, dict] = {
    "AST":                 {"label": "AST",                     "unit": "U/L",      "type": "float",  "min": 0.0,   "max": 5000.0,  "step": 1.0,   "default_sepsis": 85.0,   "default_neutral": 30.0},
    "Alkalinephos":        {"label": "Alkaline Phosphatase",    "unit": "U/L",      "type": "float",  "min": 0.0,   "max": 5000.0,  "step": 1.0,   "default_sepsis": 120.0,  "default_neutral": 70.0},
    "BUN":                 {"label": "BUN",                     "unit": "mg/dL",    "type": "float",  "min": 0.0,   "max": 200.0,   "step": 0.1,   "default_sepsis": 45.0,   "default_neutral": 15.0},
    "BaseExcess":          {"label": "Base Excess",             "unit": "mEq/L",    "type": "float",  "min": -30.0, "max": 30.0,    "step": 0.1,   "default_sepsis": -6.0,   "default_neutral": 0.0},
    "BaseExcess_Measured": {"label": "Base Excess Measured",    "unit": "0/1",      "type": "binary", "min": 0,     "max": 1,       "step": 1,     "default_sepsis": 1,      "default_neutral": 0},
    "Bilirubin_direct":    {"label": "Bilirubin Direct",        "unit": "mg/dL",    "type": "float",  "min": 0.0,   "max": 50.0,    "step": 0.01,  "default_sepsis": 0.4,    "default_neutral": 0.3},
    "Bilirubin_total":     {"label": "Bilirubin Total",         "unit": "mg/dL",    "type": "float",  "min": 0.0,   "max": 100.0,   "step": 0.01,  "default_sepsis": 1.5,    "default_neutral": 0.8},
    "Calcium":             {"label": "Calcium",                 "unit": "mg/dL",    "type": "float",  "min": 0.0,   "max": 20.0,    "step": 0.1,   "default_sepsis": 7.8,    "default_neutral": 9.2},
    "Chloride":            {"label": "Chloride",                "unit": "mEq/L",    "type": "float",  "min": 70.0,  "max": 130.0,   "step": 0.1,   "default_sepsis": 110.0,  "default_neutral": 102.0},
    "Creatinine":          {"label": "Creatinine",              "unit": "mg/dL",    "type": "float",  "min": 0.0,   "max": 50.0,    "step": 0.01,  "default_sepsis": 2.1,    "default_neutral": 0.9},
    "DBP":                 {"label": "Diastolic BP",            "unit": "mmHg",     "type": "float",  "min": 0.0,   "max": 200.0,   "step": 0.1,   "default_sepsis": 50.0,   "default_neutral": 65.0},
    "EtCO2":               {"label": "EtCO2",                   "unit": "mmHg",     "type": "float",  "min": 0.0,   "max": 60.0,    "step": 0.1,   "default_sepsis": 28.0,   "default_neutral": 35.0},
    "FiO2":                {"label": "FiO2",                    "unit": "fraction", "type": "float",  "min": 0.21,  "max": 1.0,     "step": 0.01,  "default_sepsis": 0.4,    "default_neutral": 0.21},
    "Fibrinogen":          {"label": "Fibrinogen",              "unit": "mg/dL",    "type": "float",  "min": 0.0,   "max": 1000.0,  "step": 1.0,   "default_sepsis": 180.0,  "default_neutral": 300.0},
    "Glucose":             {"label": "Glucose",                 "unit": "mg/dL",    "type": "float",  "min": 0.0,   "max": 1000.0,  "step": 0.1,   "default_sepsis": 160.0,  "default_neutral": 100.0},
    "HCO3":                {"label": "HCO3",                    "unit": "mEq/L",    "type": "float",  "min": 0.0,   "max": 50.0,    "step": 0.1,   "default_sepsis": 16.0,   "default_neutral": 24.0},
    "HCO3_Measured":       {"label": "HCO3 Measured",           "unit": "0/1",      "type": "binary", "min": 0,     "max": 1,       "step": 1,     "default_sepsis": 1,      "default_neutral": 0},
    "HR":                  {"label": "Heart Rate",              "unit": "bpm",      "type": "float",  "min": 0.0,   "max": 300.0,   "step": 0.1,   "default_sepsis": 120.0,  "default_neutral": 80.0},
    "Hct":                 {"label": "Hematocrit",              "unit": "%",        "type": "float",  "min": 0.0,   "max": 70.0,    "step": 0.1,   "default_sepsis": 32.0,   "default_neutral": 42.0},
    "Hgb":                 {"label": "Hemoglobin",              "unit": "g/dL",     "type": "float",  "min": 0.0,   "max": 25.0,    "step": 0.1,   "default_sepsis": 10.5,   "default_neutral": 14.0},
    "Lactate":             {"label": "Lactate",                 "unit": "mmol/L",   "type": "float",  "min": 0.0,   "max": 30.0,    "step": 0.01,  "default_sepsis": 4.2,    "default_neutral": 1.0},
    "Lactate_Measured":    {"label": "Lactate Measured",        "unit": "0/1",      "type": "binary", "min": 0,     "max": 1,       "step": 1,     "default_sepsis": 1,      "default_neutral": 1},
    "Lactate_abnormal":    {"label": "Lactate Abnormal (>2.0)", "unit": "0/1",      "type": "binary", "min": 0,     "max": 1,       "step": 1,     "default_sepsis": 1,      "default_neutral": 0},
    "Lactate_delta6h":     {"label": "Lactate Delta 6h",        "unit": "mmol/L",   "type": "float",  "min": -30.0, "max": 30.0,    "step": 0.01,  "default_sepsis": 1.2,    "default_neutral": 0.0},
    "Lactate_max6h":       {"label": "Lactate Max 6h",          "unit": "mmol/L",   "type": "float",  "min": 0.0,   "max": 30.0,    "step": 0.01,  "default_sepsis": 4.5,    "default_neutral": 1.0},
    "MAP":                 {"label": "Mean Arterial Pressure",  "unit": "mmHg",     "type": "float",  "min": 0.0,   "max": 200.0,   "step": 0.1,   "default_sepsis": 62.0,   "default_neutral": 90.0},
    "Magnesium":           {"label": "Magnesium",               "unit": "mg/dL",    "type": "float",  "min": 0.0,   "max": 10.0,    "step": 0.1,   "default_sepsis": 1.5,    "default_neutral": 2.0},
    "O2Sat":               {"label": "O2 Saturation",           "unit": "%",        "type": "float",  "min": 0.0,   "max": 100.0,   "step": 0.1,   "default_sepsis": 88.0,   "default_neutral": 98.0},
    "O2Sat_Measured":      {"label": "O2Sat Measured",          "unit": "0/1",      "type": "binary", "min": 0,     "max": 1,       "step": 1,     "default_sepsis": 1,      "default_neutral": 1},
    "O2Sat_low":           {"label": "O2Sat Low (<92%)",        "unit": "0/1",      "type": "binary", "min": 0,     "max": 1,       "step": 1,     "default_sepsis": 1,      "default_neutral": 0},
    "O2Sat_min6h":         {"label": "O2Sat Min 6h",            "unit": "%",        "type": "float",  "min": 0.0,   "max": 100.0,   "step": 0.1,   "default_sepsis": 85.0,   "default_neutral": 98.0},
    "PTT":                 {"label": "PTT",                     "unit": "sec",      "type": "float",  "min": 0.0,   "max": 200.0,   "step": 0.1,   "default_sepsis": 48.0,   "default_neutral": 30.0},
    "PaCO2":               {"label": "PaCO2",                   "unit": "mmHg",     "type": "float",  "min": 0.0,   "max": 100.0,   "step": 0.1,   "default_sepsis": 32.0,   "default_neutral": 40.0},
    "PaCO2_Measured":      {"label": "PaCO2 Measured",          "unit": "0/1",      "type": "binary", "min": 0,     "max": 1,       "step": 1,     "default_sepsis": 1,      "default_neutral": 0},
    "Phosphate":           {"label": "Phosphate",               "unit": "mg/dL",    "type": "float",  "min": 0.0,   "max": 20.0,    "step": 0.1,   "default_sepsis": 4.2,    "default_neutral": 3.5},
    "Platelets":           {"label": "Platelets",               "unit": "10³/μL", "type": "float", "min": 0.0, "max": 1000.0, "step": 0.1, "default_sepsis": 95.0, "default_neutral": 200.0},
    "Potassium":           {"label": "Potassium",               "unit": "mEq/L",    "type": "float",  "min": 0.0,   "max": 10.0,    "step": 0.1,   "default_sepsis": 5.1,    "default_neutral": 4.0},
    "Resp":                {"label": "Respiratory Rate",        "unit": "/min",     "type": "float",  "min": 0.0,   "max": 60.0,    "step": 0.1,   "default_sepsis": 28.0,   "default_neutral": 16.0},
    "SBP":                 {"label": "Systolic BP",             "unit": "mmHg",     "type": "float",  "min": 0.0,   "max": 300.0,   "step": 0.1,   "default_sepsis": 85.0,   "default_neutral": 120.0},
    "SF_ratio":            {"label": "SF Ratio (SpO2/FiO2)",    "unit": "",         "type": "float",  "min": 0.0,   "max": 10000.0, "step": 0.1,   "default_sepsis": 150.0,  "default_neutral": 466.7},
    "SaO2":                {"label": "SaO2",                    "unit": "%",        "type": "float",  "min": 0.0,   "max": 100.0,   "step": 0.1,   "default_sepsis": 88.0,   "default_neutral": 98.0},
    "SaO2_Measured":       {"label": "SaO2 Measured",           "unit": "0/1",      "type": "binary", "min": 0,     "max": 1,       "step": 1,     "default_sepsis": 1,      "default_neutral": 0},
    "ShockIndex":          {"label": "Shock Index (HR/SBP)",    "unit": "",         "type": "float",  "min": 0.0,   "max": 10.0,    "step": 0.01,  "default_sepsis": 1.41,   "default_neutral": 0.7},
    "Temp":                {"label": "Temperature",             "unit": "°C",       "type": "float",  "min": 32.0,  "max": 42.0,    "step": 0.1,   "default_sepsis": 38.9,   "default_neutral": 37.0},
    "TroponinI":           {"label": "Troponin I",              "unit": "ng/mL",    "type": "float",  "min": 0.0,   "max": 50.0,    "step": 0.001, "default_sepsis": 0.08,   "default_neutral": 0.01},
    "WBC":                 {"label": "WBC",                     "unit": "10³/μL", "type": "float", "min": 0.0, "max": 200.0, "step": 0.1, "default_sepsis": 18.5, "default_neutral": 8.0},
    "pH":                  {"label": "pH",                      "unit": "",         "type": "float",  "min": 6.5,   "max": 8.0,     "step": 0.01,  "default_sepsis": 7.25,   "default_neutral": 7.4},
    "pH_Measured":         {"label": "pH Measured",             "unit": "0/1",      "type": "binary", "min": 0,     "max": 1,       "step": 1,     "default_sepsis": 1,      "default_neutral": 0},
}
