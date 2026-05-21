# ==========================================================
# 01_MSAR_MODEL.py
# Markov-Switching Autoregressive (MSAR) Model
# Mendeteksi Regime Ekonomi (Stabil vs Volatil/Krisis)
# dari data makroekonomi AS bulanan (2014-2018)
#
# DATA SOURCE : Supabase -> tabel "macro_monthly"
# DATA OUTPUT : Supabase -> tabel "msar_regime_results"
#               Lokal    -> ml_output/msar_model_params.json
# ==========================================================

import pandas as pd
import numpy as np
import os
import warnings
import json

warnings.filterwarnings("ignore")

from db_config import get_engine, save_to_supabase, read_from_supabase

# ==========================================================
# KONFIGURASI
# ==========================================================

OUTPUT_DIR = "ml_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nama tabel di Supabase
TABLE_MACRO_INPUT = "macro_monthly"
TABLE_REGIME_OUTPUT = "msar_regime_results"

# ==========================================================
# 1. LOAD & PARSE DATA MAKROEKONOMI DARI SUPABASE
# ==========================================================

print("=" * 60)
print("TAHAP 1: MEMUAT DATA MAKROEKONOMI AS DARI SUPABASE")
print("=" * 60)

engine = get_engine()
macro_df = read_from_supabase(TABLE_MACRO_INPUT, engine)

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
    if col in macro_df.columns:
        macro_df[col] = macro_df[col].apply(parse_indo_pct)

# ----------------------------------------------------------
# Parsing kolom tanggal (format: "Januari 2014")
# ----------------------------------------------------------

bulan_map = {
    "Januari": 1, "Februari": 2, "Maret": 3, "April": 4,
    "Mei": 5, "Juni": 6, "Juli": 7, "Agustus": 8,
    "September": 9, "Oktober": 10, "November": 11, "Desember": 12
}

# Cek nama kolom tanggal (bisa "Bulan dan Tahun" atau variasi lain)
date_col = None
for candidate in ["Bulan dan Tahun", "bulan_dan_tahun", "bulan dan tahun"]:
    if candidate in macro_df.columns:
        date_col = candidate
        break

if date_col is None:
    # Fallback: coba kolom pertama yang bertipe string
    str_cols = macro_df.select_dtypes(include=["object"]).columns
    if len(str_cols) > 0:
        date_col = str_cols[0]
        print(f"  [INFO] Menggunakan kolom '{date_col}' sebagai kolom tanggal")

def parse_indo_date(val):
    """Mengubah 'Januari 2014' menjadi datetime."""
    parts = str(val).strip().split()
    if len(parts) == 2:
        bulan = bulan_map.get(parts[0], None)
        if bulan:
            tahun = int(parts[1])
            return pd.Timestamp(year=tahun, month=bulan, day=1)
    # Fallback: coba pd.to_datetime langsung
    try:
        return pd.to_datetime(val)
    except Exception:
        return pd.NaT

macro_df["date"] = macro_df[date_col].apply(parse_indo_date)
macro_df = macro_df.dropna(subset=["date"])
macro_df = macro_df.sort_values("date").reset_index(drop=True)
macro_df = macro_df.set_index("date")

# Pastikan frekuensi bulanan (diperlukan oleh statsmodels)
macro_df.index.freq = "MS"

# ----------------------------------------------------------
# Memetakan Nama Kolom Baru dari Supabase
# ----------------------------------------------------------

# Deteksi kolom secara dinamis berdasarkan apa yang ada di Supabase
col_fed = "fed_rate" if "fed_rate" in macro_df.columns else "interest_rate"
col_inflasi = "inflation_rate" if "inflation_rate" in macro_df.columns else "Inflasi"

# Gunakan financial_stress_index sebagai indikator krisis (pengganti Delinquency_Rate)
if "financial_stress_index" in macro_df.columns:
    col_risk = "financial_stress_index"
elif "unemployment_rate" in macro_df.columns:
    col_risk = "unemployment_rate"
else:
    col_risk = "Delinquency_Rate"

print(f"\nPreview data (5 baris pertama):")
try:
    print(macro_df[[col_fed, col_inflasi, col_risk]].head())
except KeyError:
    print(macro_df.head())

# ==========================================================
# 2. FITTING MODEL MSAR
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 2: MELATIH MODEL MSAR (Markov Switching AR)")
print("=" * 60)

from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

endog = macro_df[col_risk].dropna()
exog = macro_df.loc[endog.index, [col_fed]].copy()

print(f"Variabel Endogen    : {col_risk} ({len(endog)} observasi)")
print(f"Variabel Eksogen    : {col_fed}")
print(f"Jumlah Regime       : 2 (Stabil & Volatil)")
print(f"Order AR            : 1 (Autoregressive lag 1)")

# Fitting MSAR:
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
aligned_endog = endog.loc[regime_series.index]
mean_regime_0 = aligned_endog[regime_series == 0].mean()
mean_regime_1 = aligned_endog[regime_series == 1].mean()

print(f"\nRata-rata Risk Indicator di Regime 0: {mean_regime_0:.4f}")
print(f"Rata-rata Risk Indicator di Regime 1: {mean_regime_1:.4f}")

# Labeling: regime dengan risk lebih tinggi = VOLATIL
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
    "date": regime_series.index,
    "delinquency_rate": aligned_endog.values,
    "fed_funds_rate": exog.loc[regime_series.index, col_fed].values,
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

# --- Simpan ke Supabase ---
save_to_supabase(regime_output, TABLE_REGIME_OUTPUT, engine)

# --- Simpan parameter model sebagai JSON (lokal) ---
model_params = {
    "model": "MarkovAutoregression",
    "k_regimes": 2,
    "order": 1,
    "switching_ar": True,
    "switching_variance": True,
    "endog_variable": col_risk,
    "exog_variables": [col_fed],
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
print(f"[LOCAL SAVED] {OUTPUT_DIR}/msar_model_params.json")

print("\n[DONE] Model MSAR selesai dilatih dan regime berhasil diekstraksi!")
print("   Output Supabase : tabel 'msar_regime_results'")
print("   Output Lokal    : ml_output/msar_model_params.json")
