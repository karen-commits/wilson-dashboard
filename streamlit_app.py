# streamlit_app.py
# Wilson Dashboard — sample Streamlit mockup with logo and color palette

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Wilson Dashboard", layout="wide")

# -----------------------------
# Custom branding (logo + colors)
# -----------------------------

# Load logo safely (supports either 'logo.png' or 'Newlogo.png')
import os
from PIL import Image
logo_paths = ["logo.png", "Newlogo.png"]
logo_path = next((p for p in logo_paths if os.path.exists(p)), None)
if logo_path:
    st.image(Image.open(logo_path), width=200, caption="EVPower Insights")
else:
    st.caption("(Upload a logo file named 'logo.png' to repo root to show branding)")

# Define brand color palette (from EVPower Insights logo)
brand_colors = {
    "primary": "#1A44A3",   # dark blue
    "secondary": "#4C73FF", # lighter blue
    "accent": "#000000"     # black (for text)
}

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

    # Generate raw (0-100) indicators
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

for col in [
    "funding_momentum",
    "partnership_velocity",
    "deployment_footprint",
    "team_signals",
    "policy_positioning",
]:
    if col not in fdf.columns:
        fdf[col] = 0.0

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

st.title("Wilson Dashboard — EV Charging & Infrastructure")
st.caption("Interactive scoring of companies using Wilson Methodology metrics.")

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
    # Use matplotlib for color control
    import matplotlib.pyplot as plt
    fig1, ax1 = plt.subplots(figsize=(6.4, 3.6))
    ax1.bar(ranked["company"], ranked["WilsonScore"], color=brand_colors["primary"])
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
    # Use matplotlib for color control
import matplotlib.pyplot as plt
fig2, ax2 = plt.subplots(figsize=(6.4, 3.6))
seg_group.plot(kind="bar", ax=ax2)
for idx, patch in enumerate(ax2.patches):
    patch.set_facecolor(brand_colors["secondary"]) if idx % len(seg_group.columns) == 0 else None
ax2.set_ylabel("Average (0–100)")
ax2.legend(title="Indicators", bbox_to_anchor=(1.02, 1), loc="upper left")
st.pyplot(fig2, clear_figure=True)

# -----------------------------
# Radar (Spider) chart — Company comparison
# -----------------------------

st.markdown("---")
st.subheader("Radar chart — Compare indicators by company")

for c in indicator_cols:
    fdf[c + "_norm"] = norm_df[c].values

options = ranked["company"].tolist()
default_sel = options[:3] if len(options) >= 3 else options
selected_companies = st.multiselect(
    "Select companies to compare (from Top N table)", options=options, default=default_sel
)

if selected_companies:
    import plotly.graph_objects as go

    categories = ["Funding", "Partnerships", "Deployment", "Team", "Policy"]

    fig = go.Figure()
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
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories + categories[:1],
            fill='toself',
            name=name
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select one or more companies from the Top N list to see the radar chart.")

# -----------------------------
# Details & download
# -----------------------------

st.subheader("All filtered data")
st.dataframe(fdf.sort_values("WilsonScore", ascending=False))

csv = fdf.sort_values("WilsonScore", ascending=False).to_csv(index=False)
st.download_button("Download filtered data (CSV)", csv, "wilson_filtered.csv", "text/csv")

st.info("Tip: Upload your own CSV to override demo data. Use sidebar to tweak weights and filters.")
