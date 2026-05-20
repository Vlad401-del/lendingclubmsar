# Profil Dataset: Explainable AI-Driven Enterprise Architecture

Dokumen ini memuat ringkasan spesifikasi dataset yang digunakan dalam arsitektur sistem pemodelan *Fintech Lending Risk* menggunakan metode *Hybrid MSAR*. Proyek ini menggunakan dua dataset utama yang digabungkan: data ekonomi makro (sebagai representasi guncangan eksternal) dan data mikrokredit pinjaman P2P.

---

## 1. Dataset Makroekonomi Amerika Serikat (2014-2018)

*   **Deskripsi:** Dataset sekunder ini berisi indikator ekonomi makro Amerika Serikat secara bulanan. Data ditarik secara otomatis menggunakan API dari *Federal Reserve Economic Data* (FRED). Dalam arsitektur sistem, dataset ini berfungsi sebagai *input* untuk model MSAR (*Markov-Switching Autoregressive*) guna mendeteksi rezim kondisi ekonomi (misalnya: masa stabil vs. krisis/volatilitas tinggi) yang dapat memengaruhi risiko gagal bayar secara agregat.
*   **Jumlah Data:** 60 baris pengamatan (merepresentasikan 60 bulan berturut-turut dari Januari 2014 hingga Desember 2018).
*   **Atribut (Kolom):**
    1.  `Bulan dan Tahun`: Periode pencatatan data dalam format "Bulan Tahun" (misal: Januari 2014).
    2.  `Fed-Funds-Rate`: Suku bunga acuan AS (*Effective Federal Funds Rate*) dalam persentase, bertindak sebagai proksi likuiditas dan biaya modal.
    3.  `Inflasi`: Tingkat inflasi AS secara *Year-over-Year* (YoY) berdasarkan *Consumer Price Index* (CPI), mencerminkan tekanan daya beli masyarakat.
    4.  `Delinquency_Rate`: Persentase tingkat keterlambatan pembayaran kredit konsumsi secara agregat nasional di AS (dianalogikan seperti rasio TWP90 di Indonesia).

---

## 2. Dataset Mikrokredit P2P Lending (Lending Club)

*   **Deskripsi:** Dataset ini bersumber dari *Lending Club*, sebuah platform *Peer-to-Peer* (P2P) *Lending* di Amerika Serikat. Data ini berisi informasi mendetail mengenai profil demografis, finansial, dan riwayat kredit dari peminjam individu, serta status keberhasilan pembayaran pinjamannya. Dataset ini digunakan untuk melatih algoritma *Machine Learning* dalam memprediksi status pinjaman dan dijelaskan keputusannya menggunakan metode *Explainable AI* (XAI).
*   **Jumlah Data:** 
    *   **Data Asli:** ~2,26 Juta baris (2007 - 2018).
    *   **Data Pre-processing:** ~100.000 baris (merupakan hasil 5% *random sampling* khusus untuk pinjaman yang diterbitkan antara tahun 2014 hingga 2018). Tujuan reduksi ukuran ini adalah untuk mengoptimalkan kinerja *dashboard* dan memastikan proses kalkulasi *Explainable AI* (seperti SHAP) dapat berjalan tanpa hambatan (*bottleneck*) memori.
*   **Atribut Utama (Setelah Seleksi Fitur):**
    1.  `id`: Identifier unik untuk setiap pinjaman yang disalurkan.
    2.  `loan_amnt`: Nominal jumlah pinjaman yang diajukan oleh peminjam.
    3.  `term`: Tenor atau lama masa cicilan (biasanya 36 bulan atau 60 bulan).
    4.  `int_rate`: Persentase suku bunga tahunan yang dibebankan pada pinjaman.
    5.  `installment`: Nominal cicilan rutin yang harus dibayar peminjam setiap bulannya.
    6.  `grade`: Peringkat risiko awal (A hingga G) yang diberikan oleh platform.
    7.  `emp_length`: Lama pengalaman kerja peminjam dalam satuan tahun.
    8.  `home_ownership`: Status kepemilikan rumah (misalnya: *RENT*, *OWN*, *MORTGAGE*).
    9.  `annual_inc`: Total pendapatan kotor tahunan yang dilaporkan peminjam.
    10. `verification_status`: Status apakah sumber pendapatan tersebut telah diverifikasi oleh platform atau belum.
    11. `issue_d`: Bulan dan tahun spesifik di mana dana pinjaman dicairkan. Parameter ini digunakan sebagai kunci untuk *merge* (penggabungan) dengan dataset makroekonomi.
    12. `loan_status`: **(Variabel Target)** Status pelunasan pinjaman saat ini, di mana klasifikasi utama yang diamati adalah *Fully Paid* (Lancar) dan *Charged Off* (Gagal Bayar/Wanprestasi).
    13. `purpose`: Kategori tujuan penggunaan dana pinjaman (misal: konsolidasi utang, modal usaha, dll).
    14. `dti`: *Debt-to-Income Ratio*, rasio proporsi cicilan utang bulanan terhadap total pendapatan bulanan peminjam.
    15. `fico_range_low` & `fico_range_high`: Batas bawah dan batas atas dari skor kredit FICO peminjam saat aplikasi pinjaman dibuat (merepresentasikan riwayat kredit eksternal).
