# Dashboard Overhaul — Walkthrough

## Ringkasan Perubahan

File yang diubah: [DashboardStreamLit.py](file:///d:/laragon/www/lendingclubmsar/Data%20EA%20ZAQY/DashboardStreamLit.py)

Dashboard di-rewrite sepenuhnya dari ~472 baris menjadi ~750+ baris dengan perbaikan menyeluruh.

---

## Perubahan Global

| Aspek | Sebelum | Sesudah |
|---|---|---|
| **Styling** | Default Streamlit | Custom CSS (Inter font, gradient cards, dark sidebar, colored banners) |
| **Emoji** | Banyak emoji di setiap judul | Minimal (hanya 1 di page icon) |
| **Plotly Theme** | Default, inkonsisten | Konsisten via `apply_layout()` helper + `COLORS` palette |
| **Deprecation** | `use_container_width=True` (deprecated) | `width="stretch"` |
| **Data Sources** | 4 tabel | 10 tabel (6 tabel baru dimanfaatkan) |
| **Bahasa** | Full English | Campuran Indonesia-Inggris, penjelasan sederhana |

### Tabel Baru yang Digunakan

| Tabel | Halaman | Kegunaan |
|---|---|---|
| `macro_monthly` | Macro-Risk Engine | Data bulanan lengkap (60 baris) dengan `fed_rate` dan `financial_stress_index` |
| `msar_regime_results` | Macro-Risk Engine | Overlay regime shading + timeline |
| `shap_global_importance` | Micro-Risk | SHAP importance asli dari model |
| `shap_values` | XAI | Kontribusi SHAP per individu peminjam |
| `hybrid_decisions` | XAI | Profil + `decision_reason` narasi |
| `scenario_comparison` | Scenario Simulation | Data skenario riil dari model |

---

## Perubahan per Halaman

### 1. Executive Overview
- Ditambahkan **info box** penjelasan halaman dalam bahasa sederhana
- **8 KPI metrics** (dari 4): tambah approval rate, rata-rata pinjaman, rata-rata bunga, jumlah grade
- **Donut chart** (bukan pie) dengan angka total di tengah
- Ditambahkan **distribusi per risk grade** (warna: hijau/kuning/merah sesuai risiko)
- Ditambahkan **top 8 tujuan pinjaman** (horizontal bar chart)

### 2. Macro-Risk Engine
- **Sumber data diganti** dari `macroeconomic_indicators` (3 baris, kolom tidak lengkap) ke `macro_monthly` (60 baris, semua kolom ada)
- **3 chart sekarang muncul semua**: Financial Stress Index, Fed Rate, Inflation Rate
- Semua chart menggunakan `lines+markers` + `hovermode="x unified"` → **hover menampilkan nilai pasti**
- **Regime shading** merah muda di belakang chart untuk periode VOLATIL
- Ditambahkan **Regime Timeline** bar chart di bawah
- **4 KPI metrics** di atas (rata-rata & max stress index, fed rate, inflasi)
- Penjelasan sederhana untuk setiap indikator

### 3. Micro-Risk & Feature Importance
- Histogram PD dengan **garis threshold 0.5** dan warna lebih baik
- **4 KPI**: rata-rata PD, median PD, jumlah risiko rendah/tinggi
- **SHAP Global Importance** dari tabel `shap_global_importance` (bukan hardcoded)
- **Label fitur diterjemahkan** ke bahasa Indonesia (e.g., "int_rate" → "Tingkat Bunga (Interest Rate)")
- Gradient color pada bar chart

### 4. Explainable AI (XAI)
- **Pencarian lebih baik**: `text_input` untuk filter + `selectbox` yang menampilkan ringkasan (ID + Grade + Amount + Decision)
- **Decision banner** berwarna (hijau DITERIMA / merah DITOLAK)
- **8 metric cards** profil peminjam (loan amount, interest rate, grade, income, DTI, FICO, purpose, regime)
- **3 metric cards** parameter keputusan (PD, threshold, prob stabil)
- **Alasan keputusan dari database**: Kolom `decision_reason` dari `hybrid_decisions` ditampilkan dalam kotak kuning
- **Narasi auto-generated**: Template Python yang membangun narasi berdasarkan profil (income context, DTI analysis, FICO category, regime context, conclusion)
- **SHAP contribution chart per individu**: Data dari `shap_values`, top 10 fitur, bar merah (meningkatkan risiko) vs hijau (menurunkan risiko)

### 5. Scenario Simulation
- Slider threshold dengan **penjelasan kontekstual** yang berubah sesuai nilai
- Data dari **`scenario_comparison`** (riil dari model, bukan hardcoded)
- **Tabel data** skenario
- **Grouped bar chart** Disetujui vs Ditolak per skenario
- **False Approval Rate chart** terpisah
- **Info box** cara membaca hasil

---

## Verifikasi

- Dry-run: `python DashboardStreamLit.py` → **exit code 0, no errors**
- Semua 10 tabel database berhasil di-load
- Deprecation warning `use_container_width` sudah diatasi
