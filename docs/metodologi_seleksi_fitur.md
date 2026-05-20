# Metodologi Seleksi Fitur Formal (Formal Feature Selection)

Dokumen ini menjelaskan tahapan dan metodologi statistik yang digunakan untuk membersihkan dan menyeleksi atribut (fitur) pada dataset Lending Club (sampel 2014-2018). Proses ini sangat krusial dalam arsitektur **Explainable AI-Driven Enterprise Architecture** untuk memastikan bahwa model *Machine Learning* dan MSAR tidak terdistorsi oleh data yang kotor (*noise*), redundan, atau tidak relevan.

Proses seleksi fitur dilakukan menggunakan pustaka `pandas` dan `scikit-learn` melalui serangkaian *pipeline* formal berikut:

## 1. Penyaringan Status Pinjaman (Labeling)
Sebelum masuk ke penghapusan kolom, baris data disaring berdasarkan target variabel (`loan_status`).
*   **Tindakan:** Hanya mempertahankan pinjaman yang status akhirnya sudah pasti, yaitu **Fully Paid** (Lancar/Lunas) dan **Charged Off** (Gagal Bayar/Wanprestasi). Pinjaman yang berstatus *Current* (masih berjalan) atau *Late* dihapus.
*   **Alasan:** Pemodelan risiko kredit (*Credit Risk Scoring*) adalah permasalahan *Supervised Learning* (klasifikasi biner). Kita membutuhkan label data yang definitif untuk melatih algoritma dan menghitung keakuratannya.
*   **Hasil:** Dari total `101.499` baris sampel awal, tersisa **55.853 baris** data latih yang valid.

## 2. Tahap 1: Filter Nilai Kosong (*Missing Value Filter*)
Banyak dari 151 kolom awal di dataset Lending Club merupakan atribut opsional atau atribut sekunder yang tidak diisi oleh mayoritas peminjam.
*   **Tindakan:** Menghitung persentase nilai kosong (`NaN`) pada setiap kolom. Kolom yang memiliki persentase data kosong lebih dari **50%** dihapus dari dataset.
*   **Hasil:** Sebanyak **57 kolom** dihapus karena mayoritas datanya kosong. Mempertahankan kolom ini akan merusak model prediktif karena membutuhkan terlalu banyak imputasi buatan.

## 3. Tahap 2: Filter Varians Nol (*Zero Variance Filter*)
*   **Tindakan:** Menghitung jumlah nilai unik (*unique values*) pada setiap kolom yang tersisa. Jika sebuah kolom hanya memiliki 1 nilai unik untuk semua baris (misalnya semuanya bernilai `1`), kolom tersebut dihapus.
*   **Alasan:** Kolom tanpa variasi tidak memiliki kekuatan prediktif sama sekali dan tidak dapat membedakan antara peminjam yang gagal bayar maupun yang lunas.
*   **Hasil:** Sebanyak **5 kolom** dihapus (contoh: `pymnt_plan`, `out_prncp`, `out_prncp_inv`, `policy_code`, `hardship_flag`).

## 4. Tahap 3: Uji Multikolinearitas (Korelasi Pearson)
Beberapa fitur numerik dalam dataset merupakan representasi metrik yang identik secara fungsional (misalnya, `loan_amnt` memiliki nilai yang hampir persis sama dengan `funded_amnt`).
*   **Tindakan:** Membangun matriks korelasi *Pearson* antar seluruh fitur numerik. Jika korelasi absolut antara dua fitur melebih ambang batas **0.85** (> 85%), maka salah satu fitur tersebut didrop (dihapus).
*   **Alasan:** Memasukkan fitur-fitur yang multikolinear (saling tumpang tindih) akan menyebabkan model (seperti regresi linear atau MSAR) menjadi tidak stabil dan koefisien kepentingannya menjadi bias, yang pada akhirnya merusak proses *Explainable AI*.
*   **Hasil:** Sebanyak **12 kolom numerik** dihapus karena duplikasi (contoh: `funded_amnt`, `installment`, `total_pymnt_inv`).

## 5. Tahap 4: Uji Kekuatan Prediktif (*Feature Importance*)
Sebagai tahap verifikasi terakhir untuk melihat atribut mana yang benar-benar berharga secara statistik.
*   **Tindakan:** Melatih model `RandomForestClassifier` ringan menggunakan semua fitur numerik yang tersisa, kemudian mengekstrak metrik `feature_importances_`.
*   **Hasil Ekstraksi (Top 15 Fitur Berpengaruh):**
    Berdasarkan algoritma struktur pohon, fitur-fitur yang paling banyak dipakai untuk mendeteksi potensi gagal bayar adalah:
    1.  `recoveries` (Skor: ~0.31) - Penagihan paksa
    2.  `last_fico_range_low` (Skor: ~0.20) - Skor kredit terakhir
    3.  `last_fico_range_high` (Skor: ~0.15)
    4.  `last_pymnt_amnt` (Skor: ~0.15) - Nominal pembayaran terakhir
    5.  `total_pymnt` (Skor: ~0.06) - Total pengembalian
    6.  `loan_amnt` (Skor: ~0.03) - Jumlah yang dipinjam
    7.  `int_rate` (Skor: ~0.01) - Suku Bunga Pinjaman
    8.  `total_rec_late_fee` - Total denda
    9.  `total_rec_int` - Total bunga
    10. `fico_range_low` - Skor kredit saat awal mendaftar
    11. `total_rev_hi_lim`
    12. `bc_open_to_buy`
    13. `dti` (*Debt-to-Income Ratio*)
    14. `avg_cur_bal`
    15. `revol_bal`

---

## Kesimpulan Akhir
Melalui tahapan metodologi seleksi fitur di atas, dataset `accepted_2014_2018_sampled.csv` yang sebelumnya kotor dan berdimensi tinggi (151 kolom) telah berhasil disusutkan menjadi dataset yang solid dan valid secara statistik, dengan spesifikasi akhir:
*   **Total Kolom (Fitur):** 77 kolom
*   **Total Baris (Observasi):** 55.853 baris
*   **File Output:** `accepted_2014_2018_cleaned.csv`

Dataset akhir ini sudah siap digunakan sebagai komponen input mikro pada model deteksi risiko (*Hybrid MSAR*) berbasis makroekonomi AS.
