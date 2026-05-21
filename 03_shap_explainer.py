# ==========================================================
# 03_SHAP_EXPLAINER.py
# Explainable AI menggunakan SHAP
# (SHapley Additive exPlanations)
# Menjelaskan MENGAPA model memprediksi seorang peminjam
# sebagai Gagal Bayar atau Lancar
#
# DATA SOURCE : Supabase -> tabel "ml_x_test"
#               Lokal    -> ml_output/best_credit_model.pkl
# DATA OUTPUT : Supabase -> tabel "shap_values",
#                           "shap_global_importance"
#               Lokal    -> ml_output/shap_meta.json,
#                           ml_output/shap_local_explanations.json
# ==========================================================

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings

warnings.filterwarnings("ignore")

from db_config import get_engine, save_to_supabase, read_from_supabase

# ==========================================================
# KONFIGURASI
# ==========================================================

OUTPUT_DIR = "ml_output"
SHAP_SAMPLE_SIZE = 500   # Jumlah sampel untuk kalkulasi SHAP

# Nama tabel di Supabase
TABLE_X_TEST = "ml_x_test"
TABLE_Y_TEST = "ml_y_test"
TABLE_SHAP_VALUES = "shap_values"
TABLE_SHAP_IMPORTANCE = "shap_global_importance"

# ==========================================================
# 1. LOAD MODEL & DATA
# ==========================================================

print("=" * 60)
print("TAHAP 1: MEMUAT MODEL DAN DATA TEST SET")
print("=" * 60)

engine = get_engine()

# Load model terbaik (lokal)
best_model = joblib.load(f"{OUTPUT_DIR}/best_credit_model.pkl")
feature_names = joblib.load(f"{OUTPUT_DIR}/feature_names.pkl")

# Load test set dari Supabase
X_test = read_from_supabase(TABLE_X_TEST, engine)
y_test = read_from_supabase(TABLE_Y_TEST, engine).squeeze()

# Load metadata model (lokal)
with open(f"{OUTPUT_DIR}/credit_model_meta.json", "r") as f:
    model_meta = json.load(f)

model_name = model_meta["best_model"]
print(f"Model dimuat     : {model_name}")
print(f"Jumlah fitur     : {len(feature_names)}")
print(f"Test set         : {X_test.shape[0]} baris")

# ==========================================================
# 2. INISIALISASI SHAP EXPLAINER
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 2: MENGINISIALISASI SHAP EXPLAINER")
print("=" * 60)

import shap

# Ambil sampel acak dari test set untuk efisiensi
if len(X_test) > SHAP_SAMPLE_SIZE:
    X_shap = X_test.sample(
        n=SHAP_SAMPLE_SIZE,
        random_state=42
    ).reset_index(drop=True)
else:
    X_shap = X_test.copy()

print(f"Sampel SHAP      : {len(X_shap)} baris")

# TreeExplainer sangat efisien untuk model berbasis pohon
explainer = shap.TreeExplainer(best_model)
print(f"Explainer        : TreeExplainer (optimized for {model_name})")

# ==========================================================
# 3. KALKULASI SHAP VALUES
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 3: MENGHITUNG SHAP VALUES")
print("(Proses ini membutuhkan beberapa menit...)")
print("=" * 60)

shap_values = explainer.shap_values(X_shap)

# Untuk klasifikasi biner, shap_values bisa berupa list [class_0, class_1]
# atau array 3D dengan bentuk (n_samples, n_features, n_classes).
# Kita ambil shap values untuk kelas 1 (Charged Off / Gagal Bayar)
if isinstance(shap_values, list):
    shap_values_default = shap_values[1]  # Kelas "Charged Off"
elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    shap_values_default = shap_values[:, :, 1]
else:
    shap_values_default = shap_values

print(f"Shape SHAP Values: {shap_values_default.shape}")
print("[OK] SHAP Values berhasil dihitung!")

# ==========================================================
# 4. ANALISIS GLOBAL: FITUR MANA YANG PALING BERPENGARUH?
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 4: ANALISIS GLOBAL FEATURE IMPORTANCE (SHAP)")
print("=" * 60)

# Rata-rata absolut SHAP value per fitur
mean_abs_shap = np.abs(shap_values_default).mean(axis=0)

global_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\nTop 15 Fitur Berpengaruh (Global SHAP Importance):")
print(global_importance.head(15).to_string(index=False))

# ==========================================================
# 5. ANALISIS LOKAL: PENJELASAN PER PEMINJAM
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 5: CONTOH PENJELASAN KEPUTUSAN PER PEMINJAM")
print("=" * 60)

# Prediksi probabilitas untuk sampel SHAP
proba = best_model.predict_proba(X_shap)[:, 1]

# Ambil 3 contoh: 1 gagal bayar, 1 lancar, 1 borderline
idx_default = np.argmax(proba)
idx_safe = np.argmin(proba)
idx_border = np.argmin(np.abs(proba - 0.3))

examples = [
    ("GAGAL BAYAR (PD Tertinggi)", idx_default),
    ("LANCAR (PD Terendah)", idx_safe),
    ("BORDERLINE (PD ~30%)", idx_border),
]

local_explanations = []

for label, idx in examples:
    pd_score = proba[idx]
    shap_row = shap_values_default[idx]

    # Top 5 faktor pendorong keputusan
    top_factors_idx = np.argsort(np.abs(shap_row))[::-1][:5]

    print(f"\n--- {label} ---")
    print(f"Probabilitas Gagal Bayar (PD): {pd_score:.4f} ({pd_score*100:.1f}%)")
    print(f"Top 5 Faktor Penyebab:")

    factors = []
    for rank, fi in enumerate(top_factors_idx, 1):
        fname = feature_names[fi]
        fval = X_shap.iloc[idx, fi]
        shap_val = shap_row[fi]
        direction = ">> Meningkatkan risiko" if shap_val > 0 else "<< Menurunkan risiko"
        print(f"  {rank}. {fname} = {fval:.2f} (SHAP: {shap_val:+.4f}) {direction}")

        factors.append({
            "rank": rank,
            "feature": fname,
            "value": round(float(fval), 4),
            "shap_value": round(float(shap_val), 4),
            "direction": "increase_risk" if shap_val > 0 else "decrease_risk"
        })

    local_explanations.append({
        "label": label,
        "pd_probability": round(float(pd_score), 4),
        "top_factors": factors
    })

# ==========================================================
# 6. SIMPAN HASIL SHAP
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 6: MENYIMPAN HASIL SHAP")
print("=" * 60)

# --- Simpan SHAP values ke Supabase ---
shap_df = pd.DataFrame(
    shap_values_default,
    columns=feature_names
)
save_to_supabase(shap_df, TABLE_SHAP_VALUES, engine)

# --- Simpan global importance ke Supabase ---
save_to_supabase(global_importance, TABLE_SHAP_IMPORTANCE, engine)

# --- Simpan contoh penjelasan lokal (lokal) ---
with open(f"{OUTPUT_DIR}/shap_local_explanations.json", "w") as f:
    json.dump(local_explanations, f, indent=2, ensure_ascii=False)
print(f"[LOCAL SAVED] {OUTPUT_DIR}/shap_local_explanations.json")

# --- Simpan metadata SHAP (lokal) ---
if isinstance(explainer.expected_value, (list, np.ndarray)):
    base_value = float(explainer.expected_value[1])
else:
    base_value = float(explainer.expected_value)

shap_meta = {
    "explainer_type": "TreeExplainer",
    "model_explained": model_name,
    "shap_sample_size": len(X_shap),
    "base_value": base_value,
    "n_features": len(feature_names),
    "top_10_global_features": global_importance.head(10).to_dict("records")
}

with open(f"{OUTPUT_DIR}/shap_meta.json", "w") as f:
    json.dump(shap_meta, f, indent=2, ensure_ascii=False)
print(f"[LOCAL SAVED] {OUTPUT_DIR}/shap_meta.json")

print("\n[DONE] Analisis SHAP (Explainable AI) selesai!")
print("   Output Supabase : tabel 'shap_values', 'shap_global_importance'")
print("   Output Lokal    : ml_output/shap_*.json")
