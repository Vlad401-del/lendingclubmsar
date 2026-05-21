# ==========================================================
# 02_CREDIT_MODEL.py
# Micro-Risk Engine: Prediksi Probabilitas Gagal Bayar (PD)
# Menggunakan Random Forest dan LightGBM
# ==========================================================

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# ==========================================================
# KONFIGURASI
# ==========================================================

DATA_PATH = "accepted_2014_2018_cleaned.csv"
OUTPUT_DIR = "ml_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ==========================================================
# 1. LOAD DATA
# ==========================================================

print("=" * 60)
print("TAHAP 1: MEMUAT DATASET LENDING CLUB")
print("=" * 60)

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Total baris  : {df.shape[0]}")
print(f"Total kolom  : {df.shape[1]}")

# ==========================================================
# 2. PENYIAPAN VARIABEL TARGET
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 2: MENYIAPKAN VARIABEL TARGET")
print("=" * 60)

# Hanya pinjaman yang sudah selesai
df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()

# Encode target: 1 = Charged Off (Gagal Bayar), 0 = Fully Paid (Lancar)
df["target"] = (df["loan_status"] == "Charged Off").astype(int)

print(f"Fully Paid   : {(df['target'] == 0).sum()} ({(df['target'] == 0).mean()*100:.1f}%)")
print(f"Charged Off  : {(df['target'] == 1).sum()} ({(df['target'] == 1).mean()*100:.1f}%)")

# ==========================================================
# 3. SELEKSI FITUR UNTUK PEMODELAN
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 3: SELEKSI FITUR")
print("=" * 60)

# Fitur numerik yang sudah tervalidasi penting dari Feature Importance
numeric_features = [
    "loan_amnt",        # Jumlah pinjaman
    "int_rate",         # Suku bunga
    "dti",              # Debt-to-Income ratio
    "annual_inc",       # Pendapatan tahunan
    "fico_range_low",   # Skor kredit FICO bawah
    "revol_bal",        # Saldo kredit bergulir
    "revol_util",       # Utilisasi kredit bergulir
    "total_acc",        # Jumlah total akun kredit
    "open_acc",         # Jumlah akun terbuka
    "pub_rec",          # Catatan publik negatif
    "delinq_2yrs",      # Keterlambatan 2 tahun terakhir
    "inq_last_6mths",   # Inquiry kredit 6 bulan terakhir
    "total_rev_hi_lim", # Batas kredit bergulir tertinggi
    "avg_cur_bal",      # Rata-rata saldo saat ini
    "bc_open_to_buy",   # Sisa limit kartu kredit
    "mort_acc",         # Jumlah akun mortgage
    "tot_cur_bal",      # Total saldo saat ini
]

# Fitur kategorikal
categorical_features = [
    "grade",              # Peringkat risiko (A-G)
    "home_ownership",     # Status kepemilikan rumah
    "verification_status", # Status verifikasi pendapatan
    "purpose",            # Tujuan pinjaman
    "term",               # Tenor pinjaman
]

# Filter hanya fitur yang ada di dataset
numeric_features = [f for f in numeric_features if f in df.columns]
categorical_features = [f for f in categorical_features if f in df.columns]
all_features = numeric_features + categorical_features

print(f"Fitur Numerik     : {len(numeric_features)}")
print(f"Fitur Kategorikal : {len(categorical_features)}")
print(f"Total Fitur       : {len(all_features)}")

# ==========================================================
# 4. PREPROCESSING
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 4: PREPROCESSING DATA")
print("=" * 60)

X = df[all_features].copy()
y = df["target"].copy()

# --- Numerik: Imputasi dengan median ---
imputer = SimpleImputer(strategy="median")
X[numeric_features] = imputer.fit_transform(X[numeric_features])

# --- Kategorikal: Imputasi + Label Encoding ---
label_encoders = {}

for col in categorical_features:
    X[col] = X[col].fillna("UNKNOWN").astype(str).str.strip()
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print("Imputasi numerik   : Median")
print("Encoding kategori  : LabelEncoder")
print(f"Shape X            : {X.shape}")
print(f"Shape y            : {y.shape}")

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain set          : {X_train.shape[0]} baris")
print(f"Test set           : {X_test.shape[0]} baris")

# Simpan nama fitur untuk SHAP nanti
feature_names = list(X.columns)
joblib.dump(feature_names, f"{OUTPUT_DIR}/feature_names.pkl")
joblib.dump(imputer, f"{OUTPUT_DIR}/imputer.pkl")
joblib.dump(label_encoders, f"{OUTPUT_DIR}/label_encoders.pkl")

# Simpan test set untuk SHAP
X_test.to_csv(f"{OUTPUT_DIR}/X_test.csv", index=False)
y_test.to_csv(f"{OUTPUT_DIR}/y_test.csv", index=False)

# ==========================================================
# 5. PELATIHAN MODEL 1: RANDOM FOREST
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 5A: MELATIH RANDOM FOREST")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced"  # Menangani ketimpangan kelas
)

rf_model.fit(X_train, y_train)

# Prediksi probabilitas
rf_proba = rf_model.predict_proba(X_test)[:, 1]
rf_pred = rf_model.predict(X_test)

# Evaluasi
rf_auc = roc_auc_score(y_test, rf_proba)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)

print(f"AUC-ROC     : {rf_auc:.4f}")
print(f"Accuracy    : {rf_acc:.4f}")
print(f"F1-Score    : {rf_f1:.4f}")
print(f"Precision   : {rf_precision:.4f}")
print(f"Recall      : {rf_recall:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=["Fully Paid", "Charged Off"]))
print(f"Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# ==========================================================
# 6. PELATIHAN MODEL 2: LIGHTGBM
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 5B: MELATIH LIGHTGBM")
print("=" * 60)

try:
    import lightgbm as lgb

    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        is_unbalance=True,  # Menangani ketimpangan kelas
        verbose=-1
    )

    lgb_model.fit(X_train, y_train)

    # Prediksi probabilitas
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    lgb_pred = lgb_model.predict(X_test)

    # Evaluasi
    lgb_auc = roc_auc_score(y_test, lgb_proba)
    lgb_acc = accuracy_score(y_test, lgb_pred)
    lgb_f1 = f1_score(y_test, lgb_pred)
    lgb_precision = precision_score(y_test, lgb_pred)
    lgb_recall = recall_score(y_test, lgb_pred)

    print(f"AUC-ROC     : {lgb_auc:.4f}")
    print(f"Accuracy    : {lgb_acc:.4f}")
    print(f"F1-Score    : {lgb_f1:.4f}")
    print(f"Precision   : {lgb_precision:.4f}")
    print(f"Recall      : {lgb_recall:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, lgb_pred, target_names=["Fully Paid", "Charged Off"]))
    print(f"Confusion Matrix:")
    print(confusion_matrix(y_test, lgb_pred))

    lgb_available = True

except ImportError:
    print("⚠️  LightGBM belum terinstall. Jalankan: pip install lightgbm")
    print("    Melanjutkan hanya dengan Random Forest...")
    lgb_available = False
    lgb_auc = 0.0

# ==========================================================
# 7. PEMILIHAN MODEL TERBAIK
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 6: PERBANDINGAN & PEMILIHAN MODEL TERBAIK")
print("=" * 60)

comparison = {
    "Random Forest": {
        "AUC-ROC": round(rf_auc, 4),
        "Accuracy": round(rf_acc, 4),
        "F1-Score": round(rf_f1, 4),
        "Precision": round(rf_precision, 4),
        "Recall": round(rf_recall, 4)
    }
}

if lgb_available:
    comparison["LightGBM"] = {
        "AUC-ROC": round(lgb_auc, 4),
        "Accuracy": round(lgb_acc, 4),
        "F1-Score": round(lgb_f1, 4),
        "Precision": round(lgb_precision, 4),
        "Recall": round(lgb_recall, 4)
    }

# Tampilkan tabel perbandingan
comp_df = pd.DataFrame(comparison).T
print(comp_df.to_string())

# Pilih model terbaik berdasarkan AUC-ROC
if lgb_available and lgb_auc > rf_auc:
    best_model = lgb_model
    best_name = "LightGBM"
    best_auc = lgb_auc
    best_proba = lgb_proba
else:
    best_model = rf_model
    best_name = "Random Forest"
    best_auc = rf_auc
    best_proba = rf_proba

print(f"\n🏆 Model Terbaik    : {best_name}")
print(f"   AUC-ROC Score    : {best_auc:.4f}")

# ==========================================================
# 8. SIMPAN HASIL
# ==========================================================

print("\n" + "=" * 60)
print("TAHAP 7: MENYIMPAN MODEL & HASIL")
print("=" * 60)

# Simpan model terbaik
joblib.dump(best_model, f"{OUTPUT_DIR}/best_credit_model.pkl")
print(f"[SAVED] {OUTPUT_DIR}/best_credit_model.pkl ({best_name})")

# Simpan Random Forest juga (untuk perbandingan)
joblib.dump(rf_model, f"{OUTPUT_DIR}/rf_model.pkl")
print(f"[SAVED] {OUTPUT_DIR}/rf_model.pkl")

if lgb_available:
    joblib.dump(lgb_model, f"{OUTPUT_DIR}/lgb_model.pkl")
    print(f"[SAVED] {OUTPUT_DIR}/lgb_model.pkl")

# Simpan prediksi probabilitas test set
pred_output = pd.DataFrame({
    "y_true": y_test.values,
    "pd_probability": best_proba
})
pred_output.to_csv(f"{OUTPUT_DIR}/credit_predictions.csv", index=False)
print(f"[SAVED] {OUTPUT_DIR}/credit_predictions.csv")

# Simpan metadata model
model_meta = {
    "best_model": best_name,
    "best_auc_roc": best_auc,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "n_features": len(all_features),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "train_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "target_distribution": {
        "fully_paid": int((y == 0).sum()),
        "charged_off": int((y == 1).sum()),
        "default_rate_pct": round((y == 1).mean() * 100, 2)
    },
    "comparison": comparison
}

with open(f"{OUTPUT_DIR}/credit_model_meta.json", "w") as f:
    json.dump(model_meta, f, indent=2)
print(f"[SAVED] {OUTPUT_DIR}/credit_model_meta.json")

print("\n✅ Model kredit selesai dilatih!")
print("   Output ini akan digunakan oleh 03_shap_explainer.py")
