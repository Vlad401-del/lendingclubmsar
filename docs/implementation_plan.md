# Rencana Implementasi: Migrasi Pipeline ML ke Supabase

Proyek ini akan memigrasikan *pipeline Machine Learning* dari pembacaan/penulisan file CSV lokal menjadi koneksi langsung ke Supabase (PostgreSQL) menggunakan SQLAlchemy.

## ⚠️ User Review Required
Terdapat beberapa keputusan teknis penting mengenai bagaimana data dan model disimpan di Supabase. Mohon tinjau bagian **Open Questions** di bawah sebelum saya mengeksekusi perubahan kodenya.

## ❓ Open Questions
1. **Nama Tabel Input:** Anda menyebutkan sudah meng-import data ke Supabase. Apa nama persis tabel untuk data pinjaman (sebelumnya `accepted_2014_2018_cleaned.csv`) dan data makroekonomi (sebelumnya `TrenEkonomiAS_2014_2018_EA.csv`) di Supabase Anda? (Jika namanya belum diubah, saya akan asumsikan namanya `accepted_2014_2018_cleaned` dan `TrenEkonomiAS_2014_2018_EA`).
2. **Penyimpanan File Model (.pkl) dan Metadata (.json):** *Script* ML saat ini menghasilkan file model (`.pkl`) dan meta-informasi (`.json`). Memasukkan file *binary* (Pickle) ke dalam database PostgreSQL adalah *bad practice* karena membuat database lambat. **Apakah Anda setuju jika file `.csv` saja yang ditulis ke Supabase sebagai tabel, sedangkan file `.pkl` dan `.json` tetap disimpan secara lokal di folder `ml_output/`?**

## Proposed Changes

Perubahan arsitektur ini akan diterapkan pada keempat skrip ML. Setiap skrip akan di-update untuk memuat `dotenv` dan `sqlalchemy` guna membuat *engine* koneksi.

---

### Komponen 1: MSAR Model

#### [MODIFY] [01_msar_model.py](file:///d:/laragon/www/lendingclubmsar/01_msar_model.py)
*   **Data Ingestion:** Mengubah `pd.read_csv("TrenEkonomiAS_2014_2018_EA.csv")` menjadi `pd.read_sql("SELECT * FROM \"TrenEkonomiAS_2014_2018_EA\"", engine)`.
*   **Data Export:** Mengubah ekspor hasil deteksi *regime* dari `regime_output.to_csv(...)` menjadi `regime_output.to_sql('msar_regime_results', engine, if_exists='replace', index=False)`.

---

### Komponen 2: Credit Prediction Model

#### [MODIFY] [02_credit_model.py](file:///d:/laragon/www/lendingclubmsar/02_credit_model.py)
*   **Data Ingestion:** Mengubah pembacaan data *training* menjadi *query* SQL dari tabel `accepted_2014_2018_cleaned` (atau nama tabel yang Anda sebutkan).
*   **Data Export (Test Set):** Menyimpan `X_test` dan `y_test` (yang nantinya dipakai oleh SHAP) ke dalam Supabase sebagai tabel `ml_x_test` dan `ml_y_test` agar *pipeline* tetap berjalan mulus.
*   **Data Export (Predictions):** Menyimpan probabilitas prediksi `credit_predictions` sebagai tabel `ml_credit_predictions`.
*   *(File `.pkl` untuk model Random Forest / LightGBM akan tetap disimpan lokal).*

---

### Komponen 3: SHAP Explainer

#### [MODIFY] [03_shap_explainer.py](file:///d:/laragon/www/lendingclubmsar/03_shap_explainer.py)
*   **Data Ingestion:** Menarik `X_test` dari tabel `ml_x_test` di Supabase.
*   **Data Export:** Menyimpan hasil kalkulasi SHAP (yang akan sangat berguna untuk *dashboard*) ke dalam tabel `shap_values` dan `shap_global_importance` menggunakan `.to_sql()`.

---

### Komponen 4: Hybrid Decision Engine

#### [MODIFY] [04_hybrid_decision.py](file:///d:/laragon/www/lendingclubmsar/04_hybrid_decision.py)
*   **Data Ingestion:** Mengambil *regime* makro dari tabel `msar_regime_results` dan mengambil data profil peminjam dari tabel `accepted_2014_2018_cleaned`.
*   **Data Export:** Hasil keputusan akhir (DISETUJUI/DITOLAK) akan ditulis sebagai tabel utama `hybrid_decisions` di Supabase. Tabel inilah yang nantinya akan langsung di-kueri oleh *Dashboard Enterprise Architecture* Anda untuk divisualisasikan.

## Verification Plan

1. **Dry Run Koneksi:** Menguji koneksi `.env` pada skrip yang dimodifikasi.
2. **End-to-End Execution:** Menjalankan `01_msar_model.py` hingga `04_hybrid_decision.py` secara berurutan.
3. **Database Check:** Memastikan tabel-tabel baru (*regime*, *predictions*, *shap*, *decisions*) berhasil terbuat di dalam skema `public` Supabase.
