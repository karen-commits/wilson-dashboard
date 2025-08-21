# streamlit_app.py
# Wilson Dashboard — sample Streamlit mockup
# -------------------------------------------------
# This app lets you explore and score EV charging/infrastructure companies
# using the Wilson methodology (weights are editable in the sidebar).
# You can load your own CSV or use the demo dataset generated on the fly.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

st.set_page_config(page_title="Wilson Dashboard", layout="wide")

# ---- Brand / Theme ----
# Set EVPower Insights color palette (derived from logo blues)
PALETTE = ["#1F3A8A", "#2563EB", "#60A5FA", "#93C5FD", "#0EA5E9"]  # dark→light blues
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PALETTE)

# Try to load logo (place a file named 'logo.png' in repo root)
with st.sidebar:
    st.image("logo.png", caption="EVPower Insights", use_container_width=True)
    st.write("")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.title("Controls")

st.sidebar.subheader("Data source")
upload = st.sidebar.file_uploader(
    "Upload CSV (optional)", type=["csv"],
    help=(
        "Expected columns: company,funding_momentum,partnership_velocity,"
        "deployment_footprint,team_signals,policy_positioning,region,segment,date"
    ),
)

st.sidebar.markdown("---")
st.sidebar.subheader("Wilson Weights (%)")
# Default weights per method: 30, 20, 20, 15, 15
w_funding = st.sidebar.number_input("Funding Momentum", 0, 100, 30, step=1)
w_partner = st.sidebar.number_input("Partnership Velocity", 0, 100, 20, step=1)
w_deploy = st.sidebar.number_input("Deployment Footprint", 0, 100, 20, step=1)
w_team = st.sidebar.number_input("Team Signals", 0, 100, 15, step=1)
w_policy = st.sidebar.number_input("Policy Positioning", 0, 100, 15, step=1)

# Normalize to sum to 100 while preserving ratios (safety if edited)
weights = np.array([w_funding, w_partner, w_deploy, w_team, w_policy], dtype=float)
if weights.sum() == 0:
    weights = np.array([30,20,20,15,15], dtype=float)
weights = 100 * weights / weights.sum()

st.sidebar.caption(f"Weights normalized to: {weights.round(1).tolist()} (%)")

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
start_default = datetime.now() - timedelta(days=365)
end_default = datetime.now()
start_date, end_date = st.sidebar.date_input(
    "Date range", [start_default.date(), end_default.date()]
)
region_filter = st.sidebar.multiselect(
    "Region",
    options=["US", "EU", "LATAM", "APAC"],
    default=[],
)
segment_filter = st.sidebar.multiselect(
    "Segment",
    options=["Hardware", "Software", "Network", "Utility", "Fleet"],
    default=[],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Display")
show_top_n = st.sidebar.slider("Top N companies", 5, 50, 15)

# -----------------------------
# Data loading / demo generator
# -----------------------------

def demo_data(n=60, seed=7):
    rng = np.random.default_rng(seed)
    companies = [f"Company {chr(65+i//3)}{i%3}" for i in range(n)]
    regions = rng.choice(["US", "EU", "LATAM", "APAC"], size=n)
    segments = rng.choice(["Hardware", "Software", "Network", "Utility", "Fleet"], size=n)
    dates = rng.integers(0, 365, size=n)
    base = datetime.now() - timedelta(days=365)

    # Generate raw (0-100) indicators, skewed to look realistic
    funding = np.clip(rng.normal(55, 18, size=n), 0, 100)
    partner = np.clip(rng.normal(48, 15, size=n), 0, 100)
    deploy = np.clip(rng.normal(52, 17, size=n), 0, 100)
    team = np.clip(rng.normal(50, 14, size=n), 0, 100)
    policy = np.clip(rng.normal(46, 16, size=n), 0, 100)

    df = pd.DataFrame({
        "company": companies,
        "funding_momentum": funding,
        "partnership_velocity": partner,
        "deployment_footprint": deploy,
        "team_signals": team,
        "policy_positioning": policy,
        "region": regions,
        "segment": segments,
        "date": [base + timedelta(days=int(d)) for d in dates],
    })
    return df

if upload is not None:
    df = pd.read_csv(upload)
    # Attempt to coerce date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
else:
    df = demo_data()

# -----------------------------
# Filtering
# -----------------------------

mask = (df["date"].dt.date >= start_date) & (df["date"].dt.date <= end_date)
if region_filter:
    mask &= df["region"].isin(region_filter)
if segment_filter:
    mask &= df["segment"].isin(segment_filter)

fdf = df.loc[mask].copy()

# -----------------------------
# Scoring
# -----------------------------

# Ensure all metrics exist; if missing, create zeros
for col in [
    "funding_momentum",
    "partnership_velocity",
    "deployment_footprint",
    "team_signals",
    "policy_positioning",
]:
    if col not in fdf.columns:
        fdf[col] = 0.0

# Normalize each indicator to 0-100 if not already in that scale
indicator_cols = [
    "funding_momentum",
    "partnership_velocity",
    "deployment_footprint",
    "team_signals",
    "policy_positioning",
]

norm = {}
for col in indicator_cols:
    col_min, col_max = fdf[col].min(), fdf[col].max()
    if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
        norm[col] = np.zeros(len(fdf))
    else:
        norm[col] = 100 * (fdf[col] - col_min) / (col_max - col_min)

norm_df = pd.DataFrame(norm)

# Weighted score
w = weights / 100.0
fdf["WilsonScore"] = (
    norm_df["funding_momentum"] * w[0]
    + norm_df["partnership_velocity"] * w[1]
    + norm_df["deployment_footprint"] * w[2]
    + norm_df["team_signals"] * w[3]
    + norm_df["policy_positioning"] * w[4]
)

# -----------------------------
# Header & KPI cards
# -----------------------------

st.columns([1,6])[0].image("logo.png", width=120)
st.title("Wilson Dashboard — EV Charging & Infrastructure")
st.caption(
    "Interactive scoring of companies using Funding Momentum, Partnership Velocity, "
    "Deployment Footprint, Team Signals, and Policy Positioning."
)

colA, colB, colC, colD = st.columns(4)
colA.metric("Companies (filtered)", f"{len(fdf):,}")
colB.metric("Avg Wilson Score", f"{fdf['WilsonScore'].mean():.1f}")
colC.metric("Top Score", f"{fdf['WilsonScore'].max():.1f}")
colD.metric("Date range", f"{start_date} → {end_date}")

st.markdown("---")

# -----------------------------
# Top N table
# -----------------------------

ranked = fdf.sort_values("WilsonScore", ascending=False).head(show_top_n)

st.subheader(f"Top {show_top_n} Companies")
st.dataframe(
    ranked[[
        "company", "region", "segment", "WilsonScore",
        "funding_momentum", "partnership_velocity", "deployment_footprint",
        "team_signals", "policy_positioning",
    ]]
    .rename(columns={
        "company": "Company",
        "region": "Region",
        "segment": "Segment",
        "WilsonScore": "Wilson Score",
        "funding_momentum": "Funding",
        "partnership_velocity": "Partnerships",
        "deployment_footprint": "Deployment",
        "team_signals": "Team",
        "policy_positioning": "Policy",
    })
    .style.format({
        "Wilson Score": "{:.1f}",
        "Funding": "{:.1f}",
        "Partnerships": "{:.1f}",
        "Deployment": "{:.1f}",
        "Team": "{:.1f}",
        "Policy": "{:.1f}",
    })
)

# -----------------------------
# Charts
# -----------------------------

st.subheader("Visualizations")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Wilson Score — Top companies**")
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.0))
    ax1.bar(ranked["company"], ranked["WilsonScore"], color=PALETTE[1])
    ax1.set_ylabel("Wilson Score")
    ax1.set_xticklabels(ranked["company"], rotation=45, ha="right")
    st.pyplot(fig1, clear_figure=True)

with col2:
    st.markdown("**Average indicators by segment (filtered)**")
    seg_group = (
        fdf.groupby("segment")[indicator_cols]
        .mean()
        .sort_values("funding_momentum", ascending=False)
    )
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.0))
    seg_group.plot(kind="bar", ax=ax2)  # uses matplotlib palette
    ax2.set_ylabel("Average (0–100)")
    ax2.legend(title="Indicators", bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig2, clear_figure=True)

# -----------------------------
# Radar (Spider) chart — Company comparison
# -----------------------------

st.markdown("---")
st.subheader("Radar chart — Compare indicators by company")

# Add normalized columns to fdf for easy access
for c in indicator_cols:
    fdf[c + "_norm"] = norm_df[c].values

# Let user pick companies from the current Top N list
options = ranked["company"].tolist()
default_sel = options[:3] if len(options) >= 3 else options
selected_companies = st.multiselect(
    "Select companies to compare (from Top N table)", options=options, default=default_sel
)

if selected_companies:
    labels = [
        "Funding",
        "Partnerships",
        "Deployment",
        "Team",
        "Policy",
    ]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close circle

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 100)

    for name in selected_companies:
        row = fdf.loc[fdf["company"] == name].head(1)
        if row.empty:
            continue
        values = [
            float(row["funding_momentum_norm"].iloc[0]),
            float(row["partnership_velocity_norm"].iloc[0]),
            float(row["deployment_footprint_norm"].iloc[0]),
            float(row["team_signals_norm"].iloc[0]),
            float(row["policy_positioning_norm"].iloc[0]),
        ]
        values += values[:1]
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.1)

    ax.set_title("Normalized indicators (0–100)")
# apply palette to radar by cycling colors already set in rcParams
    st.pyplot(fig, clear_figure=True)
else:
    st.info("Select one or more companies from the Top N list to see the radar chart.")

# -----------------------------
# Details & download
# -----------------------------

st.subheader("All filtered data")
st.dataframe(fdf.sort_values("WilsonScore", ascending=False))

csv = fdf.sort_values("WilsonScore", ascending=False).to_csv(index=False)
st.download_button("Download filtered data (CSV)", csv, "wilson_filtered.csv", "text/csv")

st.info(
    "Tip: Upload your own CSV with your tracked metrics to override the demo data. "
    "Use the sidebar to tweak weights and filters, then export the ranked results."
)
