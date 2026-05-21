# ==========================================================
# 01_MSAR_MODEL.py
# Markov-Switching Autoregressive (MSAR) Model
# Mendeteksi Regime Ekonomi (Stabil vs Volatil/Krisis)
# dari data makroekonomi AS bulanan (2014-2018)
# ==========================================================

import pandas as pd
import numpy as np
import os
import warnings
import json

warnings.filterwarnings("ignore")

# ==========================================================
# KONFIGURASI
# ==========================================================

MACRO_DATA_PATH = "TrenEkonomiAS_2014_2018_EA.csv"
OUTPUT_DIR = "ml_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
# 1. LOAD & PARSE DATA MAKROEKONOMI
# ==========================================================

print("=" * 60)
print("TAHAP 1: MEMUAT DATA MAKROEKONOMI AS")
print("=" * 60)

macro_df = pd.read_csv(MACRO_DATA_PATH)

print(f"Jumlah baris   : {macro_df.shape[0]}")
print(f"Jumlah kolom   : {macro_df.shape[1]}")
print(f"Kolom          : {list(macro_df.columns)}")

# ----------------------------------------------------------
# Parsing format Indonesia (koma sebagai desimal, ada %)
# Contoh: "2,27%" -> 2.27
# ----------------------------------------------------------

def parse_indo_pct(value):
    """Mengubah format persentase Indonesia ke float."""
    if pd.isna(value):
        return np.nan
    val = str(value)
    val = val.replace("%", "").replace('"', "").strip()
    val = val.replace(",", ".")
    try:
        return float(val)
    except ValueError:
        return np.nan

for col in ["Fed-Funds-Rate", "Inflasi", "Delinquency_Rate"]:
    macro_df[col] = macro_df[col].apply(parse_indo_pct)

# ----------------------------------------------------------
# Parsing kolom tanggal (format: "Januari 2014")
# ----------------------------------------------------------

bulan_map = {
    "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
    "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
    "September": 9, "Oktober": 10, "November": 11, "Desember": 12
}

def parse_indo_date(val):
    """Mengubah 'Januari 2014' menjadi datetime."""
    parts = str(val).strip().split()
    if len(parts) == 2:
        bulan = bulan_map.get(parts[0], 1)
        tahun = int(parts[1])
        return pd.Timestamp(year=tahun, month=bulan, day=1)
    return pd.NaT

macro_df["date"] = macro_df["Bulan dan Tahun"].apply(parse_indo_date)
macro_df = macro_df.dropna(subset=["date"])
macro_df = macro_df.sort_values("date").reset_index(drop=True)
macro_df = macro_df.set_index("date")

# Pastikan frekuensi bulanan (diperlukan oleh statsmodels)
macro_df.index.freq = "MS"

print(f"\nRentang data   : {macro_df.index.min()} s/d {macro_df.index.max()}")
print(f"Total bulan    : {len(macro_df)}")
print("\nPreview data (5 baris pertama):")
print(macro_df[["Fed-Funds-Rate", "Inflasi", "Delinquency_Rate"]].head())

# ==========================================================
# 2. FITTING MODEL MSAR
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 2: MELATIH MODEL MSAR (Markov Switching AR)")
print("=" * 60)

from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

# Kita gunakan Delinquency_Rate sebagai variabel utama (endogen)
# karena paling merepresentasikan risiko gagal bayar agregat.
# Fed-Funds-Rate digunakan sebagai variabel eksogen (pengaruh eksternal).

endog = macro_df["Delinquency_Rate"].dropna()
exog = macro_df.loc[endog.index, ["Fed-Funds-Rate"]].copy()

print(f"Variabel Endogen    : Delinquency_Rate ({len(endog)} observasi)")
print(f"Variabel Eksogen    : Fed-Funds-Rate")
print(f"Jumlah Regime       : 2 (Stabil & Volatil)")
print(f"Order AR            : 1 (Autoregressive lag 1)")

# Fitting MSAR:
#   k_regimes=2  -> 2 regime (stabil vs volatil)
#   order=1      -> AR(1) pada setiap regime
#   switching_ar -> koefisien AR berbeda di tiap regime
#   switching_variance -> variansi error berbeda di tiap regime

model = MarkovAutoregression(
    endog=endog,
    k_regimes=2,
    order=1,
    switching_ar=True,
    switching_variance=True,
    exog=exog
)

result = model.fit(maxiter=500, em_iter=100)

print("\n--- RINGKASAN MODEL MSAR ---")
print(result.summary())

# ==========================================================
# 3. EKSTRAKSI REGIME
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 3: MENGEKSTRAKSI REGIME EKONOMI PER BULAN")
print("=" * 60)

# Smoothed probabilities: probabilitas berada di regime tertentu
# pada setiap titik waktu (sudah memperhitungkan seluruh data).
smoothed_probs = result.smoothed_marginal_probabilities

# Regime 0 dan Regime 1 diidentifikasi berdasarkan rata-rata
# Delinquency_Rate di masing-masing regime.
# Regime dengan rata-rata Delinquency lebih TINGGI = Volatil/Krisis.

regime_labels = result.smoothed_marginal_probabilities.iloc[:, 1] > 0.5
regime_series = regime_labels.astype(int)

# Tentukan mana regime stabil dan mana volatil
mean_regime_0 = endog[regime_series == 0].mean()
mean_regime_1 = endog[regime_series == 1].mean()

print(f"\nRata-rata Delinquency di Regime 0: {mean_regime_0:.4f}%")
print(f"Rata-rata Delinquency di Regime 1: {mean_regime_1:.4f}%")

# Labeling: regime dengan delinquency lebih tinggi = VOLATIL
if mean_regime_1 > mean_regime_0:
    label_map = {0: "STABIL", 1: "VOLATIL"}
else:
    label_map = {1: "STABIL", 0: "VOLATIL"}
    # Balik regime series
    regime_series = 1 - regime_series

print(f"Mapping Regime      : {label_map}")

# ==========================================================
# 4. MENYUSUN OUTPUT TABEL REGIME
# ==========================================================

regime_output = pd.DataFrame({
    "date": endog.index,
    "Delinquency_Rate": endog.values,
    "Fed_Funds_Rate": exog["Fed-Funds-Rate"].values,
    "regime_code": regime_series.values,
    "regime_label": regime_series.map(label_map).values,
    "prob_stabil": smoothed_probs.iloc[:, 0].values,
    "prob_volatil": smoothed_probs.iloc[:, 1].values
})

# Pastikan kolom prob sesuai dengan label yang benar
if mean_regime_1 <= mean_regime_0:
    regime_output["prob_stabil"] = smoothed_probs.iloc[:, 1].values
    regime_output["prob_volatil"] = smoothed_probs.iloc[:, 0].values

print("\n--- TABEL REGIME PER BULAN (Preview) ---")
print(regime_output[["date", "regime_label", "prob_stabil", "prob_volatil"]].head(12).to_string(index=False))

# Statistik distribusi regime
stabil_count = (regime_output["regime_label"] == "STABIL").sum()
volatil_count = (regime_output["regime_label"] == "VOLATIL").sum()
print(f"\nDistribusi Regime:")
print(f"  STABIL  : {stabil_count} bulan ({stabil_count/len(regime_output)*100:.1f}%)")
print(f"  VOLATIL : {volatil_count} bulan ({volatil_count/len(regime_output)*100:.1f}%)")

# ==========================================================
# 5. SIMPAN HASIL
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 5: MENYIMPAN HASIL")
print("=" * 60)

# Simpan tabel regime
regime_output.to_csv(f"{OUTPUT_DIR}/msar_regime_results.csv", index=False)
print(f"[SAVED] {OUTPUT_DIR}/msar_regime_results.csv")

# Simpan parameter model sebagai JSON
model_params = {
    "model": "MarkovAutoregression",
    "k_regimes": 2,
    "order": 1,
    "switching_ar": True,
    "switching_variance": True,
    "endog_variable": "Delinquency_Rate",
    "exog_variables": ["Fed-Funds-Rate"],
    "n_observations": int(len(endog)),
    "log_likelihood": float(result.llf),
    "aic": float(result.aic),
    "bic": float(result.bic),
    "regime_0_label": label_map[0],
    "regime_1_label": label_map[1],
    "regime_0_mean_delinquency": float(mean_regime_0),
    "regime_1_mean_delinquency": float(mean_regime_1),
    "months_stabil": int(stabil_count),
    "months_volatil": int(volatil_count)
}

with open(f"{OUTPUT_DIR}/msar_model_params.json", "w") as f:
    json.dump(model_params, f, indent=2)
print(f"[SAVED] {OUTPUT_DIR}/msar_model_params.json")

print("\n✅ Model MSAR selesai dilatih dan regime berhasil diekstraksi!")
print("   Output ini akan digunakan oleh 04_hybrid_decision.py")
