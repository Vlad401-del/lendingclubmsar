# ==========================================================
# STREAMLIT DASHBOARD v2.0
# Explainable AI-Driven Enterprise Architecture
# Hybrid MSAR + XAI Dashboard
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Hybrid MSAR + XAI Dashboard",
    layout="wide",
    page_icon="⬡"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }

    h1 {
        font-size: 1.8rem !important;
        color: #1a1a2e !important;
        border-bottom: 3px solid #4361ee;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem !important;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    div[data-testid="stMetric"] label {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #495057 !important;
    }

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #1a1a2e !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #e0e0e0 !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        border-bottom: none !important;
    }

    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #4361ee;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.6;
        color: #333;
    }

    .decision-approved {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border: 2px solid #28a745;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .decision-approved h2 {
        color: #155724 !important;
        border-bottom: none !important;
        margin: 0 !important;
    }

    .decision-rejected {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border: 2px solid #dc3545;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .decision-rejected h2 {
        color: #721c24 !important;
        border-bottom: none !important;
        margin: 0 !important;
    }

    .narrative-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 1rem 0;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    .section-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #4361ee;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .footer-text {
        text-align: center;
        color: #adb5bd;
        font-size: 0.8rem;
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# COLOR PALETTE & PLOTLY LAYOUT
# ==========================================================

COLORS = {
    "primary": "#4361ee",
    "secondary": "#3a0ca3",
    "success": "#2ec4b6",
    "danger": "#e63946",
    "warning": "#f4a261",
    "info": "#4cc9f0",
    "dark": "#1a1a2e",
}

PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", size=13),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=40, t=50, b=40),
    hoverlabel=dict(
        bgcolor="white",
        font_size=13,
        font_family="Inter, sans-serif"
    ),
)


def apply_layout(fig, title="", height=420):
    """Apply consistent plotly layout to a figure."""
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, family="Inter, sans-serif"),
        ),
        height=height,
        **PLOTLY_LAYOUT,
    )
    return fig

# ==========================================================
# SUPABASE CONNECTION
# ==========================================================

DB_CONN = (
    "postgresql://postgres.ddbepkvfyhgaikpzxelg:EA0supabase"
    "@aws-1-ap-south-1.pooler.supabase.com:6543/postgres"
    "?sslmode=require"
)

engine = create_engine(DB_CONN)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_table(table_name):
    return pd.read_sql(f"SELECT * FROM {table_name}", con=engine)


@st.cache_data
def get_row_count(table_name):
    df = pd.read_sql(f"SELECT COUNT(*) FROM {table_name}", con=engine)
    return int(df.iloc[0, 0])


# Core tables
fact_accepted = load_table("fact_accepted_loan")
total_rejected = get_row_count("fact_rejected_loan")
dim_risk = load_table("dim_risk_grade")

# Macro & MSAR tables
macro_monthly = load_table("macro_monthly")
msar_regime = load_table("msar_regime_results")
forecast_df = load_table("msar_forecasting")

# ML & XAI tables
shap_global = load_table("shap_global_importance")
shap_values = load_table("shap_values")
hybrid_decisions = load_table("hybrid_decisions")
scenario_comp = load_table("scenario_comparison")

# ==========================================================
# FEATURE LABEL TRANSLATIONS
# ==========================================================

FEATURE_LABELS = {
    "int_rate": "Tingkat Bunga (Interest Rate)",
    "grade": "Grade Risiko",
    "term": "Jangka Waktu (Term)",
    "dti": "Rasio Utang/Pendapatan (DTI)",
    "loan_amnt": "Jumlah Pinjaman (Loan Amount)",
    "annual_inc": "Pendapatan Tahunan",
    "fico_range_low": "Skor Kredit FICO",
    "revol_bal": "Saldo Revolving",
    "revol_util": "Utilisasi Revolving",
    "total_acc": "Total Akun Kredit",
    "open_acc": "Akun Kredit Terbuka",
    "pub_rec": "Catatan Publik (Public Record)",
    "delinq_2yrs": "Tunggakan 2 Tahun Terakhir",
    "inq_last_6mths": "Inquiry 6 Bulan Terakhir",
    "total_rev_hi_lim": "Limit Kredit Total",
    "avg_cur_bal": "Rata-rata Saldo",
    "bc_open_to_buy": "Sisa Limit Kartu Kredit",
    "mort_acc": "Akun Mortgage",
    "tot_cur_bal": "Total Saldo Saat Ini",
    "home_ownership": "Status Kepemilikan Rumah",
    "verification_status": "Status Verifikasi",
    "purpose": "Tujuan Pinjaman",
}

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.markdown("### Navigasi")

page = st.sidebar.radio(
    "Pilih Halaman",
    [
        "Executive Overview",
        "Macro-Risk Engine",
        "Micro-Risk & Feature Importance",
        "Explainable AI",
        "Scenario Simulation",
    ],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption("Hybrid MSAR + XAI System v2.0")

# ==========================================================
# PAGE 1 : EXECUTIVE OVERVIEW
# ==========================================================

if page == "Executive Overview":

    st.title("Executive Overview")

    st.markdown(
        """
    <div class="info-box">
        Halaman ini menampilkan ringkasan performa penyaluran kredit secara keseluruhan.
        Anda dapat melihat jumlah total pengajuan, tingkat persetujuan dan penolakan,
        serta sebaran pinjaman berdasarkan grade risiko dan tujuan penggunaan.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # --- KPI Metrics ---
    total_accepted = len(fact_accepted)
    total_application = total_accepted + total_rejected
    rejection_rate = (total_rejected / total_application) * 100
    approval_rate = 100.0 - rejection_rate

    avg_loan = (
        fact_accepted["loan_amnt"].mean()
        if "loan_amnt" in fact_accepted.columns
        else 0
    )
    avg_int = (
        fact_accepted["int_rate"].mean()
        if "int_rate" in fact_accepted.columns
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pengajuan", f"{total_application:,}")
    col2.metric("Pinjaman Disetujui", f"{total_accepted:,}")
    col3.metric("Pinjaman Ditolak", f"{total_rejected:,}")
    col4.metric("Tingkat Penolakan", f"{rejection_rate:.1f}%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Tingkat Persetujuan", f"{approval_rate:.1f}%")
    col6.metric("Rata-rata Pinjaman", f"${avg_loan:,.0f}")
    col7.metric("Rata-rata Bunga", f"{avg_int:.2f}%")
    col8.metric("Jumlah Grade Risiko", f"{len(dim_risk)}")

    st.divider()

    # --- Charts side-by-side ---
    col_left, col_right = st.columns(2)

    with col_left:
        decision_df = pd.DataFrame(
            {
                "Keputusan": ["Disetujui", "Ditolak"],
                "Jumlah": [total_accepted, total_rejected],
            }
        )

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=decision_df["Keputusan"],
                    values=decision_df["Jumlah"],
                    hole=0.55,
                    marker=dict(
                        colors=[COLORS["success"], COLORS["danger"]]
                    ),
                    textinfo="label+percent",
                    textfont=dict(size=14),
                    hovertemplate=(
                        "<b>%{label}</b><br>"
                        "Jumlah: %{value:,}<br>"
                        "Persentase: %{percent}"
                        "<extra></extra>"
                    ),
                )
            ]
        )
        fig = apply_layout(fig, "Rasio Persetujuan vs Penolakan")
        fig.update_layout(
            annotations=[
                dict(
                    text=f"<b>{total_application:,}</b><br>Total",
                    x=0.5,
                    y=0.5,
                    font_size=16,
                    showarrow=False,
                )
            ]
        )
        st.plotly_chart(fig, width="stretch")

    with col_right:
        if "grade_id" in fact_accepted.columns and len(dim_risk) > 0:
            grade_merged = fact_accepted.merge(
                dim_risk[["grade_id", "grade", "risk_level"]],
                on="grade_id",
                how="left",
            )
            grade_counts = (
                grade_merged["grade"]
                .value_counts()
                .sort_index()
                .reset_index()
            )
            grade_counts.columns = ["Grade", "Jumlah"]

            color_map = {}
            for g in grade_counts["Grade"].unique():
                if g in ("A", "B"):
                    color_map[g] = COLORS["success"]
                elif g in ("C", "D"):
                    color_map[g] = COLORS["warning"]
                else:
                    color_map[g] = COLORS["danger"]

            fig2 = px.bar(
                grade_counts,
                x="Grade",
                y="Jumlah",
                color="Grade",
                color_discrete_map=color_map,
                text="Jumlah",
            )
            fig2.update_traces(
                texttemplate="%{text:,}",
                textposition="outside",
                hovertemplate=(
                    "<b>Grade %{x}</b><br>"
                    "Jumlah: %{y:,}"
                    "<extra></extra>"
                ),
            )
            fig2 = apply_layout(
                fig2, "Distribusi Pinjaman per Risk Grade"
            )
            fig2.update_layout(
                showlegend=False,
                xaxis_title="Risk Grade",
                yaxis_title="Jumlah Pinjaman",
            )
            st.plotly_chart(fig2, width="stretch")

    # --- Purpose Distribution ---
    if "purpose" in fact_accepted.columns:
        purpose_counts = (
            fact_accepted["purpose"]
            .value_counts()
            .head(8)
            .reset_index()
        )
        purpose_counts.columns = ["Tujuan Pinjaman", "Jumlah"]

        fig3 = px.bar(
            purpose_counts,
            x="Jumlah",
            y="Tujuan Pinjaman",
            orientation="h",
            color="Jumlah",
            color_continuous_scale=[
                "#4cc9f0",
                "#4361ee",
                "#3a0ca3",
            ],
            text="Jumlah",
        )
        fig3.update_traces(
            texttemplate="%{text:,}",
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Jumlah: %{x:,}"
                "<extra></extra>"
            ),
        )
        fig3 = apply_layout(
            fig3, "Top 8 Tujuan Penggunaan Pinjaman", height=380
        )
        fig3.update_layout(
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig3, width="stretch")

# ==========================================================
# PAGE 2 : MACRO-RISK ENGINE
# ==========================================================

elif page == "Macro-Risk Engine":

    st.title("Macro-Risk Engine (MSAR)")

    st.markdown(
        """
    <div class="info-box">
        Halaman ini menampilkan tren indikator ekonomi makro bulanan dari 2014 hingga 2018.
        Model MSAR (Markov-Switching Autoregressive) menggunakan data ini untuk mendeteksi
        apakah ekonomi sedang <b>stabil</b> atau <b>volatil/krisis</b>.
        Area merah muda menandakan periode <b>ekonomi volatil</b>.
        <br><br>
        <i>Arahkan kursor ke grafik untuk melihat nilai pasti pada setiap titik waktu.</i>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Prepare data
    macro_monthly["date"] = pd.to_datetime(macro_monthly["date"])
    msar_regime["date"] = pd.to_datetime(msar_regime["date"])

    volatile_dates = msar_regime.loc[
        msar_regime["regime_label"] == "VOLATIL", "date"
    ].tolist()

    # --- KPI row ---
    col1, col2, col3, col4 = st.columns(4)

    if "financial_stress_index" in macro_monthly.columns:
        col1.metric(
            "Rata-rata Stress Index",
            f"{macro_monthly['financial_stress_index'].mean():.2f}",
        )
        col2.metric(
            "Stress Index Tertinggi",
            f"{macro_monthly['financial_stress_index'].max():.2f}",
        )

    if "fed_rate" in macro_monthly.columns:
        col3.metric(
            "Rata-rata Fed Rate",
            f"{macro_monthly['fed_rate'].mean():.2f}%",
        )

    if "inflation_rate" in macro_monthly.columns:
        col4.metric(
            "Rata-rata Inflasi",
            f"{macro_monthly['inflation_rate'].mean():.2f}%",
        )

    st.divider()

    # Helper: add MSAR regime shading
    def add_regime_shading(fig, v_dates):
        for vd in v_dates:
            fig.add_vrect(
                x0=vd - pd.Timedelta(days=15),
                x1=vd + pd.Timedelta(days=15),
                fillcolor="rgba(230, 57, 70, 0.08)",
                layer="below",
                line_width=0,
            )
        return fig

    # --- Chart 1: Financial Stress Index ---
    if "financial_stress_index" in macro_monthly.columns:
        st.markdown(
            '<p class="section-header">Financial Stress Index</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Mengukur tingkat tekanan di pasar keuangan. "
            "Nilai lebih tinggi = tekanan lebih besar terhadap sistem keuangan."
        )

        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=macro_monthly["date"],
                y=macro_monthly["financial_stress_index"],
                mode="lines+markers",
                name="Financial Stress Index",
                line=dict(color=COLORS["danger"], width=2.5),
                marker=dict(size=5),
                hovertemplate=(
                    "<b>%{x|%B %Y}</b><br>"
                    "Stress Index: %{y:.2f}"
                    "<extra></extra>"
                ),
            )
        )
        fig1 = add_regime_shading(fig1, volatile_dates)
        fig1 = apply_layout(fig1, "")
        fig1.update_layout(
            xaxis_title="Periode",
            yaxis_title="Index Value",
            hovermode="x unified",
            showlegend=False,
        )
        st.plotly_chart(fig1, width="stretch")

    # --- Chart 2: Fed Rate ---
    if "fed_rate" in macro_monthly.columns:
        st.markdown(
            '<p class="section-header">'
            "Federal Funds Rate (Suku Bunga Acuan AS)"
            "</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Suku bunga yang ditetapkan bank sentral AS (The Fed). "
            "Saat naik, biaya pinjaman meningkat dan risiko gagal bayar "
            "cenderung naik."
        )

        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(
                x=macro_monthly["date"],
                y=macro_monthly["fed_rate"],
                mode="lines+markers",
                name="Fed Rate",
                line=dict(color=COLORS["primary"], width=2.5),
                marker=dict(size=5),
                hovertemplate=(
                    "<b>%{x|%B %Y}</b><br>"
                    "Fed Rate: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )
        fig2 = add_regime_shading(fig2, volatile_dates)
        fig2 = apply_layout(fig2, "")
        fig2.update_layout(
            xaxis_title="Periode",
            yaxis_title="Rate (%)",
            hovermode="x unified",
            showlegend=False,
        )
        st.plotly_chart(fig2, width="stretch")

    # --- Chart 3: Inflation Rate ---
    if "inflation_rate" in macro_monthly.columns:
        st.markdown(
            '<p class="section-header">Tingkat Inflasi</p>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Inflasi yang tinggi menggerus daya beli masyarakat "
            "dan meningkatkan kemungkinan gagal bayar pinjaman."
        )

        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=macro_monthly["date"],
                y=macro_monthly["inflation_rate"],
                mode="lines+markers",
                name="Inflation Rate",
                line=dict(color=COLORS["warning"], width=2.5),
                marker=dict(size=5),
                hovertemplate=(
                    "<b>%{x|%B %Y}</b><br>"
                    "Inflasi: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )
        fig3 = add_regime_shading(fig3, volatile_dates)
        fig3 = apply_layout(fig3, "")
        fig3.update_layout(
            xaxis_title="Periode",
            yaxis_title="Rate (%)",
            hovermode="x unified",
            showlegend=False,
        )
        st.plotly_chart(fig3, width="stretch")

    # --- Regime Timeline ---
    st.divider()
    st.markdown(
        '<p class="section-header">'
        "Timeline Regime Ekonomi (Hasil Deteksi MSAR)"
        "</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Menampilkan kapan ekonomi terdeteksi stabil vs volatil "
        "berdasarkan model MSAR."
    )

    regime_chart = msar_regime.copy()

    fig_r = px.bar(
        regime_chart,
        x="date",
        y=[1] * len(regime_chart),
        color="regime_label",
        color_discrete_map={
            "STABIL": COLORS["success"],
            "VOLATIL": COLORS["danger"],
        },
        labels={"regime_label": "Regime", "date": "Periode"},
    )
    fig_r.update_traces(
        hovertemplate=(
            "<b>%{x|%B %Y}</b><br>"
            "Regime: %{data.name}"
            "<extra></extra>"
        )
    )
    fig_r = apply_layout(fig_r, "", height=200)
    fig_r.update_layout(
        yaxis=dict(showticklabels=False, title=""),
        xaxis_title="Periode",
        bargap=0,
    )
    st.plotly_chart(fig_r, width="stretch")

# ==========================================================
# PAGE 3 : MICRO-RISK & FEATURE IMPORTANCE
# ==========================================================

elif page == "Micro-Risk & Feature Importance":

    st.title("Micro-Risk & Feature Importance")

    st.markdown(
        """
    <div class="info-box">
        <b>Probability of Default (PD)</b> adalah skor 0–1 yang menunjukkan
        seberapa besar kemungkinan peminjam gagal membayar.
        Semakin tinggi nilainya, semakin berisiko.
        <br><br>
        <b>Feature Importance</b> menunjukkan variabel mana yang paling
        berpengaruh terhadap prediksi risiko gagal bayar menurut model.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # --- PD Distribution ---
    st.markdown(
        '<p class="section-header">'
        "Distribusi Probability of Default (PD)"
        "</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Sebaran skor risiko gagal bayar dari seluruh pinjaman yang "
        "disetujui. Garis merah menunjukkan batas threshold 0.5."
    )

    if "default_probability" in fact_accepted.columns:
        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=fact_accepted["default_probability"],
                nbinsx=40,
                marker=dict(
                    color=COLORS["primary"],
                    line=dict(color="white", width=0.5),
                ),
                opacity=0.85,
                hovertemplate=(
                    "<b>PD Range: %{x}</b><br>"
                    "Jumlah: %{y:,}"
                    "<extra></extra>"
                ),
            )
        )
        fig.add_vline(
            x=0.5,
            line_dash="dash",
            line_color=COLORS["danger"],
            line_width=2,
            annotation_text="Threshold 0.5",
            annotation_position="top right",
            annotation_font_color=COLORS["danger"],
        )
        fig = apply_layout(fig, "", height=400)
        fig.update_layout(
            xaxis_title="Probability of Default",
            yaxis_title="Jumlah Peminjam",
            bargap=0.05,
        )
        st.plotly_chart(fig, width="stretch")

        # PD summary metrics
        pd_data = fact_accepted["default_probability"]
        high_risk = int((pd_data >= 0.5).sum())
        low_risk = int((pd_data < 0.5).sum())

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rata-rata PD", f"{pd_data.mean():.3f}")
        col2.metric("Median PD", f"{pd_data.median():.3f}")
        col3.metric("Risiko Rendah (PD < 0.5)", f"{low_risk:,}")
        col4.metric("Risiko Tinggi (PD >= 0.5)", f"{high_risk:,}")

    st.divider()

    # --- SHAP Global Feature Importance ---
    st.markdown(
        '<p class="section-header">'
        "Global Feature Importance (SHAP)"
        "</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Berdasarkan metode SHAP (SHapley Additive exPlanations), "
        "variabel berikut paling berpengaruh terhadap prediksi model."
    )

    if len(shap_global) > 0:
        shap_sorted = shap_global.sort_values(
            "mean_abs_shap", ascending=True
        ).copy()
        shap_sorted["label"] = (
            shap_sorted["feature"]
            .map(FEATURE_LABELS)
            .fillna(shap_sorted["feature"])
        )

        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=shap_sorted["mean_abs_shap"],
                y=shap_sorted["label"],
                orientation="h",
                marker=dict(
                    color=shap_sorted["mean_abs_shap"],
                    colorscale=[
                        [0, "#4cc9f0"],
                        [0.5, "#4361ee"],
                        [1, "#3a0ca3"],
                    ],
                    line=dict(color="white", width=0.5),
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "SHAP Importance: %{x:.4f}"
                    "<extra></extra>"
                ),
            )
        )
        fig2 = apply_layout(
            fig2, "", height=max(400, len(shap_sorted) * 28)
        )
        fig2.update_layout(
            xaxis_title="Mean |SHAP Value|",
            yaxis_title="",
            showlegend=False,
        )
        st.plotly_chart(fig2, width="stretch")

# ==========================================================
# PAGE 4 : EXPLAINABLE AI (XAI)
# ==========================================================

elif page == "Explainable AI":

    st.title("Explainable AI (XAI)")

    st.markdown(
        """
    <div class="info-box">
        Halaman ini menjelaskan <b>mengapa</b> model AI membuat keputusan tertentu
        untuk setiap pengajuan pinjaman. Cari atau pilih ID peminjam untuk melihat
        profil lengkap, keputusan model, dan faktor-faktor yang mempengaruhi keputusan.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # --- Borrower Search ---
    st.markdown(
        '<p class="section-header">Cari Peminjam</p>',
        unsafe_allow_html=True,
    )

    col_search, col_select = st.columns([1, 2])

    with col_search:
        search_input = st.text_input(
            "Ketik ID Pinjaman",
            placeholder="Contoh: 63919847",
            help="Masukkan sebagian atau seluruh ID pinjaman untuk mencari",
        )

    all_ids = hybrid_decisions["id"].dropna().unique()

    if search_input:
        filtered_ids = [
            lid for lid in all_ids if str(search_input) in str(lid)
        ]
    else:
        filtered_ids = list(all_ids[:100])

    with col_select:
        id_labels = {}
        for lid in filtered_ids[:50]:
            row = hybrid_decisions[hybrid_decisions["id"] == lid]
            if len(row) > 0:
                r = row.iloc[0]
                grade = r.get("grade", "?")
                amt = r.get("loan_amnt", 0)
                dec = r.get("hybrid_decision", "?")
                label = (
                    f"ID {lid}  |  Grade {grade}"
                    f"  |  ${amt:,.0f}  |  {dec}"
                )
                id_labels[label] = lid

        if id_labels:
            selected_label = st.selectbox(
                "Pilih Peminjam",
                list(id_labels.keys()),
                help=(
                    "Daftar difilter berdasarkan pencarian "
                    "di kolom sebelah kiri"
                ),
            )
            selected_id = id_labels[selected_label]
        else:
            st.warning("Tidak ditemukan peminjam dengan ID tersebut.")
            selected_id = None

    # --- Display selected borrower ---
    if selected_id is not None:
        selected_data = hybrid_decisions[
            hybrid_decisions["id"] == selected_id
        ]

        if len(selected_data) > 0:
            b = selected_data.iloc[0]

            st.divider()

            # --- Decision Banner ---
            decision = b.get("hybrid_decision", "")
            if decision == "DITERIMA":
                st.markdown(
                    '<div class="decision-approved">'
                    "<h2>DITERIMA (APPROVED)</h2>"
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="decision-rejected">'
                    "<h2>DITOLAK (REJECTED)</h2>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("")

            # --- Borrower Profile ---
            st.markdown(
                '<p class="section-header">Profil Peminjam</p>',
                unsafe_allow_html=True,
            )

            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Jumlah Pinjaman",
                f"${b.get('loan_amnt', 0):,.0f}",
            )
            col2.metric(
                "Tingkat Bunga",
                f"{b.get('int_rate', 0):.2f}%",
            )
            col3.metric("Grade Risiko", f"{b.get('grade', 'N/A')}")
            col4.metric(
                "Pendapatan Tahunan",
                f"${b.get('annual_inc', 0):,.0f}",
            )

            col5, col6, col7, col8 = st.columns(4)
            col5.metric(
                "Rasio Utang (DTI)", f"{b.get('dti', 0):.2f}"
            )
            col6.metric(
                "Skor FICO", f"{b.get('fico_range_low', 0):.0f}"
            )
            col7.metric(
                "Tujuan Pinjaman",
                f"{b.get('purpose', 'N/A')}",
            )
            col8.metric(
                "Regime Ekonomi",
                f"{b.get('economic_regime', 'N/A')}",
            )

            st.divider()

            # --- Decision Parameters ---
            st.markdown(
                '<p class="section-header">Parameter Keputusan</p>',
                unsafe_allow_html=True,
            )

            col_a, col_b, col_c = st.columns(3)
            col_a.metric(
                "Probability of Default",
                f"{b.get('pd_probability', 0):.1%}",
            )
            col_b.metric(
                "Threshold Diterapkan",
                f"{b.get('threshold_applied', 0):.0%}",
            )
            col_c.metric(
                "Prob. Ekonomi Stabil",
                f"{b.get('prob_stabil', 0):.4f}",
            )

            st.divider()

            # --- Decision Reason (from database) ---
            st.markdown(
                '<p class="section-header">'
                "Alasan Keputusan (Dari Model)"
                "</p>",
                unsafe_allow_html=True,
            )

            reason = b.get("decision_reason", "Tidak tersedia")
            st.markdown(
                f'<div class="narrative-box">'
                f"<b>Penjelasan model:</b><br>{reason}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # --- Auto-generated Narrative ---
            st.markdown(
                '<p class="section-header">Narasi Penjelasan</p>',
                unsafe_allow_html=True,
            )

            pd_score = b.get("pd_probability", 0)
            threshold = b.get("threshold_applied", 0.5)
            regime = b.get("economic_regime", "UNKNOWN")
            dti_val = b.get("dti", 0)
            fico_val = b.get("fico_range_low", 0)
            int_rate_val = b.get("int_rate", 0)
            loan_amnt_val = b.get("loan_amnt", 0)
            annual_inc_val = b.get("annual_inc", 0)
            grade_val = b.get("grade", "")
            purpose_val = b.get("purpose", "")

            parts = []

            # Profile summary
            parts.append(
                f"Peminjam ini mengajukan pinjaman sebesar "
                f"**${loan_amnt_val:,.0f}** "
                f"untuk tujuan **{purpose_val.replace('_', ' ')}** "
                f"dengan tingkat bunga **{int_rate_val:.2f}%** "
                f"(Grade **{grade_val}**)."
            )

            # Income & DTI
            if annual_inc_val > 0:
                lti = (loan_amnt_val / annual_inc_val) * 100
                parts.append(
                    f"Dengan pendapatan tahunan "
                    f"**${annual_inc_val:,.0f}**, "
                    f"rasio pinjaman terhadap pendapatan adalah "
                    f"**{lti:.1f}%** "
                    f"dan rasio utang (DTI) sebesar **{dti_val:.2f}**."
                )
                if dti_val > 30:
                    parts.append(
                        "Rasio DTI di atas 30 menunjukkan beban utang "
                        "yang cukup tinggi relatif terhadap pendapatan."
                    )

            # FICO
            if fico_val >= 740:
                fico_cat = "sangat baik (Excellent)"
            elif fico_val >= 670:
                fico_cat = "baik (Good)"
            elif fico_val >= 580:
                fico_cat = "cukup (Fair)"
            else:
                fico_cat = "rendah (Poor)"
            parts.append(
                f"Skor kredit FICO **{fico_val:.0f}** termasuk "
                f"kategori **{fico_cat}**."
            )

            # Regime
            if regime == "VOLATIL":
                parts.append(
                    f"Pada saat pengajuan, ekonomi terdeteksi "
                    f"**VOLATIL** sehingga model menerapkan "
                    f"threshold lebih ketat "
                    f"(**{threshold:.0%}**) untuk mengurangi "
                    f"risiko gagal bayar massal."
                )
            else:
                parts.append(
                    f"Pada saat pengajuan, ekonomi terdeteksi "
                    f"**STABIL** sehingga model menerapkan "
                    f"threshold standar (**{threshold:.0%}**)."
                )

            # Conclusion
            if pd_score >= threshold:
                parts.append(
                    f"**Kesimpulan:** Skor risiko gagal bayar "
                    f"(PD = **{pd_score:.1%}**) melebihi batas "
                    f"threshold (**{threshold:.0%}**), sehingga "
                    f"pengajuan ini **ditolak** oleh model."
                )
            else:
                parts.append(
                    f"**Kesimpulan:** Skor risiko gagal bayar "
                    f"(PD = **{pd_score:.1%}**) berada di bawah "
                    f"batas threshold (**{threshold:.0%}**), "
                    f"sehingga pengajuan ini **disetujui** oleh model."
                )

            st.markdown("\n\n".join(parts))

            st.divider()

            # --- Feature Contribution Chart (SHAP) ---
            st.markdown(
                '<p class="section-header">'
                "Kontribusi Fitur terhadap Keputusan"
                "</p>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Batang hijau = menurunkan risiko (faktor positif). "
                "Batang merah = meningkatkan risiko (faktor negatif)."
            )

            # Match by positional index
            idx_pos = selected_data.index[0]

            if idx_pos < len(shap_values):
                shap_row = shap_values.iloc[idx_pos]
                shap_dict = shap_row.to_dict()
                shap_items = sorted(
                    shap_dict.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:10]

                contrib_df = pd.DataFrame(
                    {
                        "Feature": [
                            FEATURE_LABELS.get(s[0], s[0])
                            for s in shap_items
                        ],
                        "SHAP Value": [s[1] for s in shap_items],
                    }
                )
                contrib_df = contrib_df.sort_values("SHAP Value")

                bar_colors = [
                    COLORS["danger"] if v > 0 else COLORS["success"]
                    for v in contrib_df["SHAP Value"]
                ]

                fig_c = go.Figure()
                fig_c.add_trace(
                    go.Bar(
                        x=contrib_df["SHAP Value"],
                        y=contrib_df["Feature"],
                        orientation="h",
                        marker=dict(
                            color=bar_colors,
                            line=dict(color="white", width=0.5),
                        ),
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "SHAP Value: %{x:.4f}<br>"
                            "<i>Positif = menaikkan risiko<br>"
                            "Negatif = menurunkan risiko</i>"
                            "<extra></extra>"
                        ),
                    )
                )
                fig_c.add_vline(
                    x=0, line_color="#aaa", line_width=1
                )
                fig_c = apply_layout(
                    fig_c,
                    "",
                    height=max(350, len(contrib_df) * 35),
                )
                fig_c.update_layout(
                    xaxis_title="SHAP Value",
                    yaxis_title="",
                    showlegend=False,
                )
                st.plotly_chart(fig_c, width="stretch")
            else:
                st.info(
                    "Data SHAP detail tidak tersedia untuk "
                    "peminjam ini."
                )

# ==========================================================
# PAGE 5 : SCENARIO SIMULATION
# ==========================================================

elif page == "Scenario Simulation":

    st.title("Scenario Simulation")

    st.markdown(
        """
    <div class="info-box">
        Halaman ini memungkinkan Anda menguji dampak perubahan kebijakan risiko.
        Dengan menggeser slider <b>threshold</b>, Anda dapat melihat bagaimana
        batas toleransi risiko yang berbeda mempengaruhi jumlah pinjaman yang
        disetujui dan ditolak.
        <br><br>
        <b>Threshold</b> adalah batas skor risiko. Peminjam dengan PD di atas
        threshold akan <b>ditolak</b>, dan yang di bawahnya akan <b>disetujui</b>.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # --- Interactive Threshold ---
    st.markdown(
        '<p class="section-header">Simulasi Threshold</p>',
        unsafe_allow_html=True,
    )

    threshold = st.slider(
        "Atur Batas Risiko (Threshold PD)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help=(
            "Geser ke kiri = lebih ketat (lebih banyak ditolak). "
            "Geser ke kanan = lebih longgar (lebih banyak disetujui)."
        ),
    )

    if "default_probability" in fact_accepted.columns:
        approved = fact_accepted[
            fact_accepted["default_probability"] < threshold
        ]
        rejected_sim = fact_accepted[
            fact_accepted["default_probability"] >= threshold
        ]
        sim_approval_rate = (
            len(approved) / max(len(fact_accepted), 1)
        ) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Disetujui", f"{len(approved):,}")
        col2.metric("Ditolak", f"{len(rejected_sim):,}")
        col3.metric(
            "Tingkat Persetujuan", f"{sim_approval_rate:.1f}%"
        )

        # Contextual explanation
        if threshold < 0.3:
            st.info(
                "Threshold sangat ketat. Hanya peminjam dengan risiko "
                "sangat rendah yang disetujui. Mengurangi kerugian tapi "
                "juga mengurangi volume pinjaman secara signifikan."
            )
        elif threshold < 0.5:
            st.info(
                "Threshold cukup ketat. Keseimbangan antara mengurangi "
                "risiko dan mempertahankan volume pinjaman."
            )
        elif threshold < 0.7:
            st.info(
                "Threshold moderat. Lebih banyak pinjaman disetujui, "
                "namun risiko gagal bayar juga lebih tinggi."
            )
        else:
            st.warning(
                "Threshold sangat longgar. Banyak peminjam berisiko "
                "tinggi akan disetujui. Potensi kerugian meningkat "
                "signifikan."
            )

    st.divider()

    # --- Real Scenario Comparison ---
    st.markdown(
        '<p class="section-header">'
        "Perbandingan Skenario (Hasil Model)"
        "</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Tabel dan grafik di bawah menampilkan hasil perbandingan nyata "
        "dari 3 skenario yang diuji oleh model MSAR."
    )

    if len(scenario_comp) > 0:
        # Display table
        display_df = scenario_comp.copy()
        display_df.columns = [
            "Skenario",
            "Threshold",
            "Disetujui",
            "Ditolak",
            "False Approvals",
            "False Approval Rate",
        ]
        display_df["False Approval Rate"] = (
            display_df["False Approval Rate"]
            .apply(lambda x: f"{x * 100:.2f}%")
        )
        st.dataframe(
            display_df, use_container_width=True, hide_index=True
        )

        # Grouped bar chart
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="Disetujui",
                x=scenario_comp["scenario"],
                y=scenario_comp["approved"],
                marker_color=COLORS["success"],
                text=scenario_comp["approved"].apply(
                    lambda x: f"{x:,}"
                ),
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Disetujui: %{y:,}"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Bar(
                name="Ditolak",
                x=scenario_comp["scenario"],
                y=scenario_comp["rejected"],
                marker_color=COLORS["danger"],
                text=scenario_comp["rejected"].apply(
                    lambda x: f"{x:,}"
                ),
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Ditolak: %{y:,}"
                    "<extra></extra>"
                ),
            )
        )

        fig = apply_layout(
            fig,
            "Perbandingan Jumlah Disetujui vs Ditolak per Skenario",
            height=450,
        )
        fig.update_layout(
            barmode="group",
            xaxis_title="Skenario",
            yaxis_title="Jumlah Pinjaman",
        )
        st.plotly_chart(fig, width="stretch")

        # False approval rate chart
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=scenario_comp["scenario"],
                y=scenario_comp["false_approval_rate"] * 100,
                marker=dict(
                    color=[
                        COLORS["success"],
                        COLORS["warning"],
                        COLORS["primary"],
                    ],
                    line=dict(color="white", width=1),
                ),
                text=scenario_comp["false_approval_rate"].apply(
                    lambda x: f"{x * 100:.2f}%"
                ),
                textposition="outside",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "False Approval Rate: %{y:.2f}%"
                    "<extra></extra>"
                ),
            )
        )
        fig2 = apply_layout(
            fig2,
            "False Approval Rate per Skenario",
            height=380,
        )
        fig2.update_layout(
            xaxis_title="Skenario",
            yaxis_title="False Approval Rate (%)",
            showlegend=False,
        )
        st.plotly_chart(fig2, width="stretch")

        st.markdown(
            """
        <div class="info-box">
            <b>Cara membaca:</b> <i>False Approval Rate</i> menunjukkan
            persentase pinjaman yang disetujui namun akhirnya gagal bayar.
            Skenario <b>MSAR Volatil (20%)</b> memiliki false approval rate
            paling rendah karena menggunakan threshold paling ketat saat
            ekonomi tidak stabil.
        </div>
        """,
            unsafe_allow_html=True,
        )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.markdown(
    """
<div class="footer-text">
    Hybrid MSAR + XAI Dashboard v2.0 &mdash;
    Explainable AI-Driven Enterprise Architecture
    for Fintech Lending Risk
</div>
""",
    unsafe_allow_html=True,
)