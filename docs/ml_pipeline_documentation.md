# Dokumentasi Pipeline Machine Learning Hybrid (MSAR + XAI)

Dokumen ini menjelaskan secara teknis alur proses *Machine Learning* yang diterapkan pada proyek **"Explainable AI-Driven Credit Risk Dashboard using Hybrid MSAR Models"**. Pipeline ini dirancang dengan mengkombinasikan analisis makroekonomi (sistemik) dan analisis mikro (individu peminjam).

---

## Arsitektur Sistem

Pipeline ML terdiri dari empat modul (skrip) utama yang dieksekusi secara berurutan:
1. `01_msar_model.py` (Macro-Risk Engine)
2. `02_credit_model.py` (Micro-Risk Engine)
3. `03_shap_explainer.py` (Explainable AI Engine)
4. `04_hybrid_decision.py` (Hybrid Decision Engine)

Sistem ini didukung oleh file konfigurasi bersama `db_config.py` untuk menjembatani komunikasi ke database terpusat, yaitu **Supabase**.

---

## 1. Macro-Risk Engine (MSAR)

**Tujuan:** Mendeteksi *regime* / status ekonomi makro (Stabil vs Volatil) setiap bulan, karena status ekonomi sistemik sangat memengaruhi probabilitas gagal bayar massal.

- **Data Masukan:** 
  - Bersumber dari tabel `macro_monthly` di Supabase. 
  - Terdiri dari 60 baris data bulanan (Januari 2014 - Desember 2018).
  - Variabel Endogen (utama): `financial_stress_index` (Indeks tekanan finansial).
  - Variabel Eksogen (pendukung): `fed_rate` (Suku bunga The Fed).
- **Pemrosesan (Algoritma):**
  - Menggunakan algoritma **Markov-Switching Autoregressive (MSAR)** `MarkovAutoregression` dari pustaka `statsmodels`.
  - Spesifikasi model: 2 *Regimes* (k_regimes=2), *Autoregressive* orde 1 (order=1), dengan *switching variance* dan *switching AR*.
- **Hasil:**
  - Model mengklasifikasikan bulan ke dalam 2 *regime*:
    - **Regime 0 (STABIL):** Memiliki rata-rata indeks tekanan finansial sebesar 0.2838. Terdeteksi sebanyak 16 bulan (27.1%).
    - **Regime 1 (VOLATIL):** Memiliki rata-rata indeks tekanan finansial sebesar 0.6816. Terdeteksi sebanyak 43 bulan (72.9%).
- **Data Keluaran:**
  - Menyimpan hasil per bulan beserta probabilitas kemulusannya (*smoothed probabilities*) ke Supabase dalam tabel `msar_regime_results`.

---

## 2. Micro-Risk Engine (Prediksi Kredit Individu)

**Tujuan:** Memprediksi **Probabilitas Gagal Bayar (PD - Probability of Default)** untuk setiap peminjam menggunakan algoritma pemelajaran mesin berbasis *Ensemble Learning*.

- **Data Masukan:** 
  - Bersumber dari data lokal `accepted_2014_2018_cleaned.csv` berisi histori peminjaman *Lending Club*.
- **Pemrosesan (Preprocessing):**
  - **Pembersihan:** Membuang duplikat, menstandarkan format teks, membuang limit ekstrim (DTI > 100), memastikan fitur numerik/kategorikal valid. Dari **55.853**, tersisa **55.805** observasi bersih.
  - **Penentuan Target:** Kelas Positif (Gagal Bayar / *Charged Off*) sebesar 20.6%, Kelas Negatif (Lancar / *Fully Paid*) sebesar 79.4%.
  - **Imputasi:** Mengisi nilai numerik yang kosong dengan *Median* (nilai tengah).
  - **Encoding:** Menggunakan *LabelEncoder* untuk mengubah 5 fitur teks menjadi fitur numerik.
  - **Pembagian Data (Train-Test Split):** Menggunakan pembagian **80% Latih (44.644 baris)** dan **20% Uji (11.161 baris)** dengan stratifikasi agar proporsi gagal bayar seimbang.
- **Pemrosesan (Modelling):**
  - Melatih dua algoritma: **Random Forest Classifier** dan **LightGBM Classifier**.
  - **Hasil Evaluasi:** Random Forest memenangkan perbandingan dengan metrik **AUC-ROC: 0.7204** dan Akurasi keseluruhan 74.53%.
- **Data Keluaran:**
  - Fitur dan Target untuk *Test Set* disalin ke Supabase menjadi tabel `ml_x_test` dan `ml_y_test`.
  - Hasil probabilitas (`predict_proba`) disimpan di tabel `ml_credit_predictions`.
  - Berbagai *state* model disimpan lokal sebagai *pickle* (`.pkl`).

---

## 3. Explainable AI (SHAP)

**Tujuan:** Memberikan transparansi pada "Black Box" milik model Random Forest. Bertujuan untuk menjawab "Fitur apa yang membuat A disetujui, dan B ditolak?"

- **Data Masukan:** 
  - *Test set* murni yang ditarik dari Supabase (`ml_x_test` dan `ml_y_test`).
- **Pemrosesan (Algoritma):**
  - Menggunakan metode SHAP (*SHapley Additive exPlanations*) varian **TreeExplainer**, yang dioptimalkan khusus untuk model berbasis pohon (Random Forest).
  - Dilakukan penarikan sampel representatif sebanyak 500 observasi.
- **Hasil:**
  - **Global Importance:** Menemukan bahwa Suku Bunga (`int_rate`), Peringkat Pinjaman (`grade`), dan Tenor (`term`) adalah 3 faktor paling vital yang mendikte risiko gagal bayar secara makro.
  - **Local Interpretability:** Menyimpan nilai SHAP individu yang menjelaskan pengaruh setiap nilai variabel terhadap skor seseorang (menambah vs mengurangi risiko).
- **Data Keluaran:**
  - Tabel matriks dampak variabel disalin ke Supabase menjadi tabel `shap_values` dan ringkasan kepentingannya di `shap_global_importance`.

---

## 4. Hybrid Decision Engine (Sinergi Aturan)

**Tujuan:** Menghasilkan keputusan final (`DISETUJUI` / `DITOLAK`) dengan menggabungkan pandangan makroekonomi dan probabilitas kredit mikro individu.

- **Data Masukan:** 
  - `msar_regime_results` dari Supabase (Regime per bulan).
  - Keseluruhan data 55.853 peminjam dari CSV.
  - Model Kredit Random Forest ter- *load* dari lokal.
- **Pemrosesan (Penggabungan & Threshold Dinamis):**
  - Pertama, menghitung Probabilitas Gagal Bayar (PD) dari seluruh 55.853 peminjam dengan Random Forest.
  - Kedua, memetakan *Regime* ekonomi bulan pinjaman (*issue_month*) milik setiap pengguna dari data MSAR.
  - Ketiga, menerapkan **Aturan Ambang Batas (Threshold) Hybrid:**
    - Jika Regime = **STABIL**, maka persyaratannya lebih longgar: Pinjaman **DISETUJUI** jika Probabilitas Gagal Bayar $\le 40\%$.
    - Jika Regime = **VOLATIL/KRISIS**, maka persyaratannya sangat ketat demi menekan NPL: Pinjaman **DISETUJUI** jika Probabilitas Gagal Bayar $\le 20\%$.
- **Hasil Akhir:**
  - Sistem menolak **37.695 (67.5%)** pengajuan dan menyetujui **18.158 (32.5%)** pengajuan.
  - Terbukti secara simulasi bahwa ketika menggunakan pendekatan MSAR, sistem berani melonggarkan batas (dan mendapat tambahan potensi keuntungan) di masa yang stabil, serta membatasi persetujuan secara ekstrem di masa volatil untuk membatasi angka kredit macet mendadak.
- **Data Keluaran:**
  - Data mentah keseluruhan beserta output `pd_probability`, `threshold_applied`, `hybrid_decision`, dan `decision_reason` dimuat ke Supabase pada tabel **`hybrid_decisions`**.
  - Perbandingan berbagai simulasi skenario ambang batas disimpan dalam **`scenario_comparison`**.
