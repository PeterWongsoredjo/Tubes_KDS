<div align="center"> 
  <h1> Analisis Krisis Metabolik pada Sepsis: Sistem Peringatan Dini Berbasis MLP dan Explainable AI </h1>
  <h3> Tugas Besar IF3211 Komputasi Domain Spesifik </h3>

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-MLP-F7931E?logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-00BFFF?logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)

</div>


Sistem peringatan dini sepsis berbasis machine learning yang menggunakan arsitektur Multi-Layer Perceptron (MLP) dan SHAP untuk mendeteksi risiko sepsis dari data klinis pasien ICU secara per jam. Dataset yang digunakan adalah PhysioNet/CinC Challenge 2019.

---

## Daftar Isi

1. [Deskripsi Proyek](#1-deskripsi-proyek)
2. [Struktur Repositori](#2-struktur-repositori)
3. [Alur Pipeline](#3-alur-pipeline)
4. [Memulai](#4-memulai)
   - [Prasyarat](#prasyarat)
   - [Instalasi](#instalasi)
   - [Menjalankan Notebook](#menjalankan-notebook)
   - [Menjalankan Dashboard Streamlit](#menjalankan-dashboard-streamlit)
5. [Hasil Model](#5-hasil-model)
6. [Tim Pengembang](#6-tim-pengembang)

---

## 1. Deskripsi Proyek

Proyek ini mengembangkan sistem peringatan dini sepsis menggunakan data deret waktu klinis dari pasien ICU. Model MLP dilatih untuk memprediksi probabilitas sepsis pada setiap jam observasi, kemudian diinterpretasikan menggunakan SHAP KernelExplainer untuk mengidentifikasi fitur klinis yang paling berpengaruh.

Fitur utama sistem ini meliputi:

- Pemrosesan data klinis berdimensi tinggi dengan penanganan missing value berbasis bounded forward-fill
- Rekayasa fitur metabolik dan hipoksia seperti Lactate Max 6h, SF Ratio, dan Shock Index
- Evaluasi berlapis: per jam dan per pasien untuk meminimalkan false alarm
- Dashboard inferensi interaktif berbasis Streamlit dengan 48 fitur klinis

---

## 2. Struktur Repositori

```
Tubes_KDS/
├── data/
│   ├── Dataset.csv             # Dataset mentah PhysioNet/CinC Challenge 2019
│   ├── df_balanced.csv         # Dataset setelah downsampling pasien (rasio 1:2)
│   ├── feature_names.npy       # Nama 48 fitur yang digunakan model
│   ├── X_train.npy             # Data latih (sudah diskalakan)
│   ├── X_test.npy              # Data uji (sudah diskalakan)
│   ├── y_train.npy             # Label latih
│   ├── y_test.npy              # Label uji
│   ├── test_pid.npy            # ID pasien pada data uji
│   └── test_hour.npy           # Jam observasi pada data uji
│
├── model/
│   ├── sepsis_mlp_model.pkl    # Model MLP terlatih
│   └── scaler.pkl              # StandardScaler yang digunakan saat pelatihan
│
├── notebooks/
│   ├── 01_eda_sepsis.ipynb     # Eksplorasi data: pipeline preprocessing dan EDA
│   └── 02_modeling_sepsis.ipynb # Pelatihan model, evaluasi, dan analisis SHAP
│
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── loss_curve.png
│   ├── shap_summary.png
│   ├── patient_confusion_matrix.png
│   ├── patient_roc_curve.png
│   ├── patient_pr_curve.png
│   ├── eda_feature_correlation.png
│   ├── eda_balanced_correlation.png
│   └── full_feature_heatmap.png
│
├── src/
│   ├── config.py               # Konstanta, path artefak, dan metadata 48 fitur
│   ├── inference.py            # Fungsi pemuatan model dan prediksi
│   └── dashboard.py            # Aplikasi Streamlit (entry point)
│
├── requirements.txt
└── README.md
```

---

## 4. Panduan Menjalankan Dashboard Prototype

### Prasyarat

- Python 3.9 atau lebih baru
- pip

### Instalasi

Clone repositori dan instal dependensi:

```bash
git clone <url-repositori>
cd Tubes_KDS
pip install -r requirements.txt
pip install streamlit
```

### Menjalankan Notebook

Jalankan notebook secara berurutan dari direktori `notebooks/`:

```bash
jupyter notebook
```

1. Buka dan jalankan `01_eda_sepsis.ipynb` untuk menjalankan pipeline preprocessing dan EDA.
2. Buka dan jalankan `02_modeling_sepsis.ipynb` untuk melatih model, menyimpan artefak, dan menjalankan analisis SHAP.

Setelah `02_modeling_sepsis.ipynb` selesai dijalankan, artefak berikut akan tersedia:

```
model/sepsis_mlp_model.pkl
model/scaler.pkl
data/feature_names.npy
```

### Menjalankan Dashboard Streamlit

Pastikan ketiga artefak di atas sudah tersedia, kemudian jalankan dari direktori root repositori:

```bash
streamlit run src/dashboard.py
```

Dashboard akan terbuka di browser pada `http://localhost:8501`. Masukkan nilai fitur klinis pasien dan tekan tombol **Prediksi Sepsis** untuk mendapatkan hasil prediksi beserta tingkat kepercayaan model.

---

## 5. Hasil Model

| Metrik | Per Jam | Per Pasien |
|---|---|---|
| Recall | >= 0,75 (target) | tinggi |
| Threshold | 0,0732 | agregasi top-3 |
| Metode evaluasi | Per baris observasi | Rata-rata 3 probabilitas tertinggi per pasien |

Fitur paling berpengaruh berdasarkan analisis SHAP: Temp, SF Ratio, O2Sat Min 6h, Heart Rate, Chloride, FiO2, BUN, dan Hemoglobin.

---

## 6. Tim Penulis

| Nama | NIM |
|---|---|
| Nathaniel Jonathan Rusli | 13523013 |
| Maheswara Bayu Kaindra | 13523015 |
| Peter Wongsoredjo | 13523039 |
