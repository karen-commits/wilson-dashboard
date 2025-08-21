# Wilson Dashboard

An interactive dashboard built with **Streamlit** to evaluate EV
charging and infrastructure companies using the **Wilson Score
Methodology**.

## 🚀 Features

-   Upload your own CSV dataset or use demo data.
-   Adjustable weights for the five Wilson metrics:
    -   Funding Momentum (30%)
    -   Partnership Velocity (20%)
    -   Deployment Footprint (20%)
    -   Team Signals (15%)
    -   Policy Positioning (15%)
-   Interactive filters by date, region, and segment.
-   Ranking table with Wilson Scores.
-   Charts for top companies and segment averages.
-   Download filtered results as CSV.

## 📦 Requirements

The project uses the following Python packages: - `streamlit` -
`pandas` - `numpy`

Optional for extended features: - `matplotlib` - `openpyxl`

## ▶️ How to Run Locally

1.  Clone this repository:

    ``` bash
    git clone https://github.com/karen-commits/wilson-dashboard.git
    cd wilson-dashboard
    ```

2.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Run the app:

    ``` bash
    streamlit run streamlit_app.py
    ```

## 🌐 Deploy on Streamlit Cloud

This app can be deployed directly via [Streamlit
Cloud](https://streamlit.io/cloud).\
Just connect your GitHub repo and set: - **Repository:**
`karen-commits/wilson-dashboard` - **Branch:** `main` - **Main file
path:** `streamlit_app.py`

## 📊 Demo Data

A demo dataset is automatically generated if no CSV is uploaded.\
Expected CSV columns if you upload your own data:

    company, funding_momentum, partnership_velocity,
    deployment_footprint, team_signals, policy_positioning,
    region, segment, date

------------------------------------------------------------------------

✨ Built with [Streamlit](https://streamlit.io).
