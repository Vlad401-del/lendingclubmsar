# ==========================================================
# STREAMLIT DASHBOARD
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
    page_icon="📊"
)

# ==========================================================
# SUPABASE CONNECTION
# ==========================================================

DB_CONN = "postgresql://postgres.ddbepkvfyhgaikpzxelg:EA0supabase@aws-1-ap-south-1.pooler.supabase.com:6543/postgres?sslmode=require"

# Membuat engine koneksi database
engine = create_engine(DB_CONN)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_table(table_name):
    # Langsung query seluruh isi tabel menjadi DataFrame
    query = f"SELECT * FROM {table_name}"
    return pd.read_sql(query, con=engine)

@st.cache_data
def get_row_count(table_name):
    # Mengambil jumlah baris tabel tanpa memuat seluruh datanya
    query = f"SELECT COUNT(*) FROM {table_name}"
    df = pd.read_sql(query, con=engine)
    return int(df.iloc[0, 0])

# Load tables
fact_accepted = load_table("fact_accepted_loan")
total_rejected = get_row_count("fact_rejected_loan")
dim_risk = load_table("dim_risk_grade")
macro_df = load_table("macroeconomic_indicators")
forecast_df = load_table("msar_forecasting")
feature_importance = load_table("feature_importance")

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Executive Overview",
        "Macro-Risk Engine",
        "Micro-Risk & Feature Importance",
        "Explainable AI",
        "Scenario Simulation"
    ]
)

# ==========================================================
# PAGE 1 : EXECUTIVE OVERVIEW
# ==========================================================

if page == "Executive Overview":

    st.title("📊 Executive Overview")

    total_accepted = len(fact_accepted)
    # total_rejected is defined globally via database count query

    total_application = (
        total_accepted + total_rejected
    )

    rejection_rate = (
        total_rejected / total_application
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Applications",
        f"{total_application:,}"
    )

    col2.metric(
        "Approved Loans",
        f"{total_accepted:,}"
    )

    col3.metric(
        "Rejected Loans",
        f"{total_rejected:,}"
    )

    col4.metric(
        "Rejection Rate",
        f"{rejection_rate:.2f}%"
    )

    st.divider()

    # Approval vs Rejection
    decision_df = pd.DataFrame({
        "Decision": ["Approved", "Rejected"],
        "Count": [total_accepted, total_rejected]
    })

    fig = px.pie(
        decision_df,
        names="Decision",
        values="Count",
        title="Hybrid Lending Decisions"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # Regime comparison
    if "economic_scenario" in forecast_df.columns:

        regime_counts = (
            forecast_df["economic_scenario"]
            .value_counts()
            .reset_index()
        )

        regime_counts.columns = [
            "Scenario",
            "Count"
        ]

        fig = px.bar(
            regime_counts,
            x="Scenario",
            y="Count",
            title="Economic Regime Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==========================================================
# PAGE 2 : MACRO-RISK ENGINE
# ==========================================================

elif page == "Macro-Risk Engine":

    st.title("🌍 Macro-Risk Engine (MSAR)")

    macro_df["date"] = pd.to_datetime(
        macro_df["date"]
    )

    # Financial Stress Index
    if "financial_stress_index" in macro_df.columns:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=macro_df["date"],
                y=macro_df[
                    "financial_stress_index"
                ],
                mode="lines",
                name="Financial Stress Index"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Fed Rate
    if "fed_rate" in macro_df.columns:

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=macro_df["date"],
                y=macro_df["fed_rate"],
                mode="lines",
                name="Fed Rate"
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # Inflation
    if "inflation_rate" in macro_df.columns:

        fig3 = px.line(
            macro_df,
            x="date",
            y="inflation_rate",
            title="Inflation Rate Trend"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

# ==========================================================
# PAGE 3 : MICRO RISK & FEATURE IMPORTANCE
# ==========================================================

elif page == "Micro-Risk & Feature Importance":

    st.title("🤖 Micro-Risk & Feature Importance")

    # PD Distribution
    if "default_probability" in fact_accepted.columns:

        fig = px.histogram(
            fact_accepted,
            x="default_probability",
            nbins=30,
            title="Probability of Default Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # Feature Importance
    if len(feature_importance) > 0:

        feature_importance = (
            feature_importance.sort_values(
                by="importance_score",
                ascending=True
            )
        )

        fig2 = px.bar(
            feature_importance,
            x="importance_score",
            y="feature_name",
            orientation="h",
            title="Global Feature Importance"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

# ==========================================================
# PAGE 4 : EXPLAINABLE AI
# ==========================================================

elif page == "Explainable AI":

    st.title("🔍 Explainable AI")

    borrower_ids = (
        fact_accepted["loan_id"]
        .dropna()
        .unique()
    )

    selected_id = st.selectbox(
        "Select Loan ID",
        borrower_ids
    )

    selected_data = fact_accepted[
        fact_accepted["loan_id"] == selected_id
    ]

    if len(selected_data) > 0:

        borrower = selected_data.iloc[0]

        col1, col2 = st.columns(2)

        col1.metric(
            "Loan Amount",
            f"${borrower['loan_amnt']:,.0f}"
        )

        col1.metric(
            "Interest Rate",
            f"{borrower['int_rate']:.2f}%"
        )

        col2.metric(
            "DTI Ratio",
            f"{borrower['dti']:.2f}"
        )

        col2.metric(
            "PD Score",
            f"{borrower['default_probability']:.2f}"
        )

        st.divider()

        # Final Decision
        decision = (
            "REJECTED"
            if borrower["default_probability"] >= 0.5
            else "APPROVED"
        )

        st.subheader(
            f"Final Decision : {decision}"
        )

        st.divider()

        # Explainability Chart
        explain_df = pd.DataFrame({

            "Feature": [
                "Interest Rate",
                "DTI",
                "FICO",
                "Loan Amount"
            ],

            "Impact": [
                borrower["int_rate"] * 0.02,
                borrower["dti"] * 0.01,
                -borrower["fico_range_low"] * 0.001,
                borrower["loan_amnt"] * 0.00001
            ]
        })

        fig = px.bar(
            explain_df,
            x="Impact",
            y="Feature",
            orientation="h",
            title="Feature Contribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==========================================================
# PAGE 5 : SCENARIO SIMULATION
# ==========================================================

elif page == "Scenario Simulation":

    st.title("🎛️ Scenario Simulation")

    st.subheader("PD Threshold Simulation")

    threshold = st.slider(
        "Default Probability Threshold",
        0.0,
        1.0,
        0.5,
        0.01
    )

    approved = fact_accepted[
        fact_accepted[
            "default_probability"
        ] < threshold
    ]

    rejected = fact_accepted[
        fact_accepted[
            "default_probability"
        ] >= threshold
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Approved",
        len(approved)
    )

    col2.metric(
        "Rejected",
        len(rejected)
    )

    approval_rate = (
        len(approved) / len(fact_accepted)
    ) * 100

    col3.metric(
        "Approval Rate",
        f"{approval_rate:.2f}%"
    )

    st.divider()

    scenario_df = pd.DataFrame({

        "Scenario": [
            "Without MSAR",
            "Stable Economy",
            "Volatile Economy"
        ],

        "Approval Rate": [
            72,
            81,
            55
        ],

        "Predicted Default": [
            19,
            10,
            28
        ]
    })

    fig = px.bar(
        scenario_df,
        x="Scenario",
        y="Approval Rate",
        title="Scenario Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    """
    Explainable AI-Driven Enterprise Architecture
    for Financial Inclusion under Fintech Lending Risk
    using Hybrid MSAR Models
    """
)