# Rencana Desain Dashboard Streamlit

Karena inti dari proyek Anda adalah membuktikan berfungsinya arsitektur **Hybrid (MSAR + XAI)**, Dashboard harus menonjolkan bagaimana kondisi makroekonomi (sistemik) memengaruhi keputusan akhir terhadap individu (peminjam), serta menjelaskan "mengapa" keputusan itu dibuat.

## User Review Required

> [!IMPORTANT]
> Mohon tinjau rancangan halaman/tab Dashboard di bawah ini. Apakah ada fitur tambahan yang ingin Anda masukkan, atau ada metrik spesifik yang menjadi fokus utama Anda?

## Proposed Changes (Struktur Dashboard)

Kita akan membuat dashboard berbasis **Multi-Page** atau **Multi-Tab** menggunakan Streamlit agar tidak menumpuk di satu halaman. Berikut adalah usulan strukturnya:

### Halaman 1: 📊 Executive Overview
Halaman beranda untuk melihat gambaran besar bisnis saat ini.
- **Key Performance Indicators (KPI):** Total Pengajuan, Total Disetujui, Total Ditolak, dan Tingkat Penolakan (*Rejection Rate*).
- **Grafik Utama:** Distribusi Keputusan Hybrid (Disetujui vs Ditolak).
- **Komparasi Regime:** *Bar chart* perbandingan persentase persetujuan pinjaman ketika ekonomi sedang **Stabil** vs **Volatil**.

### Halaman 2: 🌍 Macro-Risk Engine (MSAR)
Fokus pada kondisi makroekonomi sistemik.
- **Grafik Time Series:** Tren *Financial Stress Index* dan Suku Bunga (*Fed Rate*) dari waktu ke waktu.
- **Highlight Regime:** Area grafik akan diberi warna latar belakang yang berbeda (misal merah transparan) pada rentang waktu yang terdeteksi sebagai krisis/volatil oleh MSAR.

### Halaman 3: 🤖 Micro-Risk & Feature Importance
Fokus pada model AI kredit secara keseluruhan.
- **Distribusi Skor PD:** *Histogram* yang menampilkan sebaran Probabilitas Gagal Bayar (PD) dari seluruh peminjam.
- **Global SHAP (Feature Importance):** *Bar chart* Horizontal dari tabel `shap_global_importance` yang menunjukkan 10 fitur paling krusial dalam menentukan risiko peminjam (misal: Suku Bunga, Peringkat, dll).

### Halaman 4: 🔍 Explainable AI (Analisis Individu)
Fitur unggulan proyek ini: membedah profil satu peminjam.
- **Pencarian Peminjam:** *Dropdown* untuk memilih satu ID Peminjam secara acak.
- **Kartu Profil:** Menampilkan ringkasan pinjaman (jumlah pinjaman, tujuan, peringkat, rasio utang).
- **Keputusan Final:** Menampilkan Skor PD, Status Regime bulan tersebut, Threshold yang berlaku, dan **Keputusan Final**.
- **SHAP Waterfall/Bar Chart:** Visualisasi *Explainable AI* yang menunjukkan dengan sangat transparan fitur apa saja yang **meningkatkan** dan **menurunkan** risiko bagi peminjam tersebut secara spesifik.

### Halaman 5: 🎛️ Scenario Simulation
Halaman interaktif khusus pengambil kebijakan.
- Menggunakan tabel `scenario_comparison`.
- **Tabel Simulasi:** Perbandingan metrik jika perusahaan tidak menggunakan MSAR vs Menggunakan MSAR (Stabil 40% / Volatil 20%).
- *(Opsional)* **Slider Interaktif:** Memungkinkan Anda menggeser-geser angka threshold PD dan melihat bagaimana dampaknya terhadap jumlah persetujuan dan potensi *default* secara langsung.

## Verification Plan

- Saya akan membuat file `streamlit_app.py`.
- Aplikasi ini akan terkoneksi langsung dengan membaca database **Supabase** secara *real-time*.
- Kita akan melakukan tes jalankan secara lokal menggunakan perintah `streamlit run streamlit_app.py`.
