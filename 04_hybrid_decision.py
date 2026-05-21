# ==========================================================
# 04_HYBRID_DECISION.py
# Hybrid Decision Engine
# Mengintegrasikan MSAR (Macro Regime) dengan
# Model Kredit (Micro PD) untuk menghasilkan keputusan
# akhir: DISETUJUI atau DITOLAK
# ==========================================================

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# KONFIGURASI
# ==========================================================

OUTPUT_DIR = "ml_output"
DATA_PATH = "accepted_2014_2018_cleaned.csv"

# Threshold Hybrid Decision Rule:
#   - Saat ekonomi STABIL  → toleransi PD lebih longgar
#   - Saat ekonomi VOLATIL → toleransi PD diperketat
THRESHOLD_STABIL = 0.40   # PD <= 40% → DISETUJUI
THRESHOLD_VOLATIL = 0.20  # PD <= 20% → DISETUJUI

# ==========================================================
# 1. LOAD SEMUA KOMPONEN
# ==========================================================

print("=" * 60)
print("TAHAP 1: MEMUAT SEMUA KOMPONEN MODEL")
print("=" * 60)

# --- 1a. Load Model Kredit ---
best_model = joblib.load(f"{OUTPUT_DIR}/best_credit_model.pkl")
feature_names = joblib.load(f"{OUTPUT_DIR}/feature_names.pkl")
imputer = joblib.load(f"{OUTPUT_DIR}/imputer.pkl")
label_encoders = joblib.load(f"{OUTPUT_DIR}/label_encoders.pkl")

with open(f"{OUTPUT_DIR}/credit_model_meta.json", "r") as f:
    credit_meta = json.load(f)

print(f"Model Kredit     : {credit_meta['best_model']}")
print(f"AUC-ROC          : {credit_meta['best_auc_roc']:.4f}")

# --- 1b. Load Regime MSAR ---
regime_df = pd.read_csv(f"{OUTPUT_DIR}/msar_regime_results.csv", parse_dates=["date"])

with open(f"{OUTPUT_DIR}/msar_model_params.json", "r") as f:
    msar_meta = json.load(f)

print(f"MSAR Regime      : {msar_meta['months_stabil']} bulan stabil, "
      f"{msar_meta['months_volatil']} bulan volatil")

# --- 1c. Load SHAP metadata ---
with open(f"{OUTPUT_DIR}/shap_meta.json", "r") as f:
    shap_meta = json.load(f)

print(f"SHAP Base Value  : {shap_meta['base_value']:.4f}")

# --- 1d. Load data lengkap ---
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()

print(f"Total peminjam   : {df.shape[0]}")

# ==========================================================
# 2. MAPPING REGIME KE SETIAP PEMINJAM
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 2: MENCOCOKKAN REGIME EKONOMI KE SETIAP PEMINJAM")
print("=" * 60)

# Parse tanggal di dataset pinjaman
# Format issue_d biasanya: "Dec-2015" atau "2015-12-01"
df["issue_date"] = pd.to_datetime(df["issue_d"], format="mixed", dayfirst=False)
df["issue_month"] = df["issue_date"].dt.to_period("M")

# Parse tanggal di regime
regime_df["month_period"] = regime_df["date"].dt.to_period("M")

# Buat mapping bulan → regime
regime_map = dict(zip(
    regime_df["month_period"].astype(str),
    regime_df["regime_label"]
))

regime_prob_stabil_map = dict(zip(
    regime_df["month_period"].astype(str),
    regime_df["prob_stabil"]
))

# Terapkan mapping
df["economic_regime"] = df["issue_month"].astype(str).map(regime_map)
df["prob_stabil"] = df["issue_month"].astype(str).map(regime_prob_stabil_map)

# Isi yang tidak ter-mapping (bulan di luar jangkauan MSAR)
df["economic_regime"] = df["economic_regime"].fillna("STABIL")
df["prob_stabil"] = df["prob_stabil"].fillna(0.5)

regime_dist = df["economic_regime"].value_counts()
print("Distribusi regime pada peminjam:")
for regime, count in regime_dist.items():
    print(f"  {regime}: {count} peminjam ({count/len(df)*100:.1f}%)")

# ==========================================================
# 3. PREDIKSI PD UNTUK SELURUH PEMINJAM
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 3: MENGHITUNG PROBABILITAS GAGAL BAYAR (PD)")
print("=" * 60)

# Ambil fitur yang dibutuhkan model
numeric_features = credit_meta["numeric_features"]
categorical_features = credit_meta["categorical_features"]
all_features = numeric_features + categorical_features

# Validasi: hanya fitur yang ada
numeric_features = [f for f in numeric_features if f in df.columns]
categorical_features = [f for f in categorical_features if f in df.columns]

X_all = df[numeric_features + categorical_features].copy()

# Preprocessing (sama dengan saat training)
X_all[numeric_features] = imputer.transform(X_all[numeric_features])

for col in categorical_features:
    X_all[col] = X_all[col].fillna("UNKNOWN").astype(str).str.strip()
    le = label_encoders[col]
    # Handle unseen labels
    known_classes = set(le.classes_)
    X_all[col] = X_all[col].apply(
        lambda x: x if x in known_classes else "UNKNOWN"
    )
    # Pastikan "UNKNOWN" ada di encoder
    if "UNKNOWN" not in le.classes_:
        le.classes_ = np.append(le.classes_, "UNKNOWN")
    X_all[col] = le.transform(X_all[col])

# Prediksi PD
pd_probability = best_model.predict_proba(X_all)[:, 1]
df["pd_probability"] = pd_probability

print(f"Rata-rata PD     : {pd_probability.mean():.4f} ({pd_probability.mean()*100:.1f}%)")
print(f"Median PD        : {np.median(pd_probability):.4f} ({np.median(pd_probability)*100:.1f}%)")
print(f"Min PD           : {pd_probability.min():.4f}")
print(f"Max PD           : {pd_probability.max():.4f}")

# ==========================================================
# 4. HYBRID DECISION RULE
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 4: MENERAPKAN HYBRID DECISION RULE")
print("=" * 60)

print(f"Threshold STABIL  : PD <= {THRESHOLD_STABIL*100:.0f}% → DISETUJUI")
print(f"Threshold VOLATIL : PD <= {THRESHOLD_VOLATIL*100:.0f}% → DISETUJUI")

def hybrid_decision(row):
    """
    Logika keputusan hybrid:
    Menggabungkan kondisi makroekonomi (MSAR regime)
    dengan risiko individu (PD) untuk menghasilkan
    keputusan akhir yang adaptif terhadap guncangan ekonomi.
    """
    regime = row["economic_regime"]
    pd_score = row["pd_probability"]

    if regime == "STABIL":
        threshold = THRESHOLD_STABIL
    else:  # VOLATIL
        threshold = THRESHOLD_VOLATIL

    if pd_score <= threshold:
        return "DISETUJUI"
    else:
        return "DITOLAK"

def decision_reason(row):
    """Menghasilkan penjelasan keputusan untuk setiap peminjam."""
    regime = row["economic_regime"]
    pd_score = row["pd_probability"]
    decision = row["hybrid_decision"]

    if regime == "STABIL":
        threshold = THRESHOLD_STABIL
    else:
        threshold = THRESHOLD_VOLATIL

    if decision == "DISETUJUI":
        return (
            f"PD ({pd_score*100:.1f}%) berada di bawah threshold "
            f"{threshold*100:.0f}% (Regime: {regime}). "
            f"Risiko dapat diterima."
        )
    else:
        return (
            f"PD ({pd_score*100:.1f}%) melebihi threshold "
            f"{threshold*100:.0f}% (Regime: {regime}). "
            f"Risiko terlalu tinggi untuk kondisi ekonomi saat ini."
        )

# Terapkan keputusan
df["hybrid_decision"] = df.apply(hybrid_decision, axis=1)
df["decision_reason"] = df.apply(decision_reason, axis=1)

# Tentukan threshold yang digunakan
df["threshold_applied"] = df["economic_regime"].apply(
    lambda r: THRESHOLD_STABIL if r == "STABIL" else THRESHOLD_VOLATIL
)

# ==========================================================
# 5. EVALUASI KEPUTUSAN HYBRID
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 5: EVALUASI HASIL KEPUTUSAN HYBRID")
print("=" * 60)

# Distribusi keputusan
decision_dist = df["hybrid_decision"].value_counts()
print("Distribusi Keputusan Hybrid:")
for decision, count in decision_dist.items():
    print(f"  {decision}: {count} ({count/len(df)*100:.1f}%)")

# Cross-tabulation: Keputusan vs Status Aktual
print("\n--- Cross-Tab: Keputusan Hybrid vs Status Aktual ---")
crosstab = pd.crosstab(
    df["hybrid_decision"],
    df["loan_status"],
    margins=True,
    margins_name="Total"
)
print(crosstab)

# Analisis per regime
print("\n--- Keputusan per Regime Ekonomi ---")
for regime in ["STABIL", "VOLATIL"]:
    subset = df[df["economic_regime"] == regime]
    if len(subset) == 0:
        continue
    approved = (subset["hybrid_decision"] == "DISETUJUI").sum()
    rejected = (subset["hybrid_decision"] == "DITOLAK").sum()
    actual_default = (subset["loan_status"] == "Charged Off").sum()
    print(f"\n  Regime: {regime}")
    print(f"    Total peminjam      : {len(subset)}")
    print(f"    DISETUJUI           : {approved} ({approved/len(subset)*100:.1f}%)")
    print(f"    DITOLAK             : {rejected} ({rejected/len(subset)*100:.1f}%)")
    print(f"    Aktual Gagal Bayar  : {actual_default} ({actual_default/len(subset)*100:.1f}%)")

    # Precision: dari yang ditolak, berapa persen memang benar gagal bayar?
    rejected_subset = subset[subset["hybrid_decision"] == "DITOLAK"]
    if len(rejected_subset) > 0:
        true_reject = (rejected_subset["loan_status"] == "Charged Off").sum()
        reject_precision = true_reject / len(rejected_subset)
        print(f"    Presisi Penolakan   : {reject_precision:.4f} ({reject_precision*100:.1f}%)")

# ==========================================================
# 6. SIMULASI SKENARIO
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 6: SIMULASI PERBANDINGAN SKENARIO")
print("=" * 60)

scenarios = {
    "Tanpa MSAR (Flat 30%)": 0.30,
    "MSAR Stabil (40%)": THRESHOLD_STABIL,
    "MSAR Volatil (20%)": THRESHOLD_VOLATIL,
}

scenario_results = []
for scenario_name, threshold in scenarios.items():
    approved = (df["pd_probability"] <= threshold).sum()
    rejected = (df["pd_probability"] > threshold).sum()

    # Dari yang disetujui, berapa yang ternyata gagal bayar?
    approved_mask = df["pd_probability"] <= threshold
    if approved_mask.sum() > 0:
        false_approvals = (
            (df[approved_mask]["loan_status"] == "Charged Off").sum()
        )
        false_approval_rate = false_approvals / approved_mask.sum()
    else:
        false_approvals = 0
        false_approval_rate = 0

    print(f"\n  Skenario: {scenario_name}")
    print(f"    Disetujui         : {approved} ({approved/len(df)*100:.1f}%)")
    print(f"    Ditolak           : {rejected} ({rejected/len(df)*100:.1f}%)")
    print(f"    Salah Setujui     : {false_approvals} ({false_approval_rate*100:.1f}%)")

    scenario_results.append({
        "scenario": scenario_name,
        "threshold": threshold,
        "approved": int(approved),
        "rejected": int(rejected),
        "false_approvals": int(false_approvals),
        "false_approval_rate": round(float(false_approval_rate), 4)
    })

# ==========================================================
# 7. SIMPAN HASIL AKHIR
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 7: MENYIMPAN HASIL AKHIR")
print("=" * 60)

# Kolom output yang akan disimpan
output_cols = [
    "id",
    "loan_amnt",
    "int_rate",
    "grade",
    "annual_inc",
    "dti",
    "fico_range_low",
    "purpose",
    "loan_status",
    "issue_d",
    "economic_regime",
    "prob_stabil",
    "pd_probability",
    "threshold_applied",
    "hybrid_decision",
    "decision_reason"
]

# Pastikan hanya kolom yang ada
output_cols = [c for c in output_cols if c in df.columns]

# Simpan keputusan hybrid per peminjam
hybrid_output = df[output_cols].copy()
hybrid_output.to_csv(
    f"{OUTPUT_DIR}/hybrid_decisions.csv",
    index=False
)
print(f"[SAVED] {OUTPUT_DIR}/hybrid_decisions.csv")

# Simpan ringkasan statistik
summary = {
    "total_borrowers": int(len(df)),
    "threshold_stabil": THRESHOLD_STABIL,
    "threshold_volatil": THRESHOLD_VOLATIL,
    "decisions": {
        "approved": int((df["hybrid_decision"] == "DISETUJUI").sum()),
        "rejected": int((df["hybrid_decision"] == "DITOLAK").sum()),
        "approval_rate_pct": round(
            (df["hybrid_decision"] == "DISETUJUI").mean() * 100, 2
        )
    },
    "regime_distribution": {
        regime: int(count) for regime, count
        in df["economic_regime"].value_counts().items()
    },
    "pd_statistics": {
        "mean": round(float(df["pd_probability"].mean()), 4),
        "median": round(float(df["pd_probability"].median()), 4),
        "std": round(float(df["pd_probability"].std()), 4),
        "min": round(float(df["pd_probability"].min()), 4),
        "max": round(float(df["pd_probability"].max()), 4)
    },
    "scenario_comparison": scenario_results,
    "model_components": {
        "msar": msar_meta,
        "credit_model": credit_meta["best_model"],
        "credit_auc": credit_meta["best_auc_roc"],
        "shap_base_value": shap_meta["base_value"]
    }
}

with open(f"{OUTPUT_DIR}/hybrid_summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"[SAVED] {OUTPUT_DIR}/hybrid_summary.json")

# Simpan skenario simulasi
scenario_df = pd.DataFrame(scenario_results)
scenario_df.to_csv(
    f"{OUTPUT_DIR}/scenario_comparison.csv",
    index=False
)
print(f"[SAVED] {OUTPUT_DIR}/scenario_comparison.csv")

print("\n" + "=" * 60)
print("✅ HYBRID DECISION ENGINE SELESAI!")
print("=" * 60)
print(f"""
Arsitektur Hybrid MSAR telah menghasilkan keputusan
adaptif untuk {len(df)} peminjam.

Ringkasan:
  DISETUJUI : {(df['hybrid_decision'] == 'DISETUJUI').sum()} peminjam
  DITOLAK   : {(df['hybrid_decision'] == 'DITOLAK').sum()} peminjam

Seluruh output tersimpan di folder: {OUTPUT_DIR}/
File-file ini siap digunakan oleh Dashboard Explainable AI.
""")
