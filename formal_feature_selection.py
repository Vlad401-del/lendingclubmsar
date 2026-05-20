import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

def formal_feature_selection():
    print("="*50)
    print("MEMULAI PROSES SELEKSI FITUR FORMAL")
    print("="*50)
    
    # 1. Load Data
    print("\nMemuat dataset...")
    df = pd.read_csv('accepted_2014_2018_sampled.csv', low_memory=False)
    initial_cols = df.shape[1]
    print(f"Total baris: {df.shape[0]}")
    print(f"Total kolom awal: {initial_cols}")

    # Kami hanya fokus pada pinjaman yang sudah selesai untuk akurasi credit scoring
    # Menyaring 'Fully Paid' dan 'Charged Off'
    df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
    print(f"Total baris setelah filter status selesai (Fully Paid/Charged Off): {df.shape[0]}")

    # ==========================================
    # TAHAP 1: FILTER MISSING VALUES (> 50%)
    # ==========================================
    print("\n[Tahap 1] Menghapus kolom dengan Missing Value > 50%...")
    missing_percentages = df.isnull().sum() / len(df)
    cols_to_drop_missing = missing_percentages[missing_percentages > 0.50].index.tolist()
    df.drop(columns=cols_to_drop_missing, inplace=True)
    print(f"Dihapus {len(cols_to_drop_missing)} kolom karena kosong melompong.")
    # print(f"Contoh yang dihapus: {cols_to_drop_missing[:5]}...")

    # ==========================================
    # TAHAP 2: FILTER VARIANS NOL (Zero Variance)
    # ==========================================
    print("\n[Tahap 2] Menghapus kolom yang hanya memiliki 1 nilai unik (Zero Variance)...")
    cols_to_drop_variance = [col for col in df.columns if df[col].nunique() <= 1]
    df.drop(columns=cols_to_drop_variance, inplace=True)
    print(f"Dihapus {len(cols_to_drop_variance)} kolom karena tidak memiliki variasi.")
    if len(cols_to_drop_variance) > 0:
        print(f"Kolom yang dihapus: {cols_to_drop_variance}")

    # ==========================================
    # TAHAP 3: UJI MULTIKOLINEARITAS (Korelasi)
    # ==========================================
    print("\n[Tahap 3] Menghapus kolom dengan korelasi tinggi (Multikolinearitas > 0.85)...")
    # Hanya kolom numerik
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # Matriks segitiga atas
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    cols_to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.85)]
    df.drop(columns=cols_to_drop_corr, inplace=True)
    print(f"Dihapus {len(cols_to_drop_corr)} kolom numerik karena saling menduplikasi (korelasi > 0.85).")
    print(f"Contoh yang dihapus: {cols_to_drop_corr[:5]}...")

    # ==========================================
    # TAHAP 4: UJI KEKUATAN PREDIKTIF (Random Forest)
    # ==========================================
    print("\n[Tahap 4] Mengukur Feature Importance dengan Random Forest...")
    
    # Ambil numerik saja untuk uji pentingnya fitur
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Drop kolom ID karena ID tidak memiliki kekuatan prediktif logis
    if 'id' in numeric_features: numeric_features.remove('id')
    if 'member_id' in numeric_features: numeric_features.remove('member_id')
        
    X = df[numeric_features].copy()
    y = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})
    
    # Imputasi sederhana untuk Random Forest
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
    rf.fit(X_imputed, y)
    
    # Ranking pentingnya fitur
    importances = pd.DataFrame({
        'Fitur': numeric_features,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 15 Fitur Paling Berpengaruh (Berdasarkan Random Forest):")
    print(importances.head(15).to_string(index=False))
    
    # Simpan dataset yang sudah bersih
    output_filename = 'accepted_2014_2018_cleaned.csv'
    df.to_csv(output_filename, index=False)
    print(f"\nSelesai! Sisa kolom: {df.shape[1]}")
    print(f"Dataset bersih (setelah seleksi formal) disimpan di: {output_filename}")

if __name__ == '__main__':
    formal_feature_selection()
