import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import streamlit as st
import sys
import os
sys.path.append(os.path.abspath("src"))
from visualization import plot_monthly_avg

# Ignore future warnings from seaborn
warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_style("whitegrid")

@st.cache_data
def load_data():
    df_benin = pd.read_csv("data/clean_benin.csv")
    df_togo = pd.read_csv("data/clean_togo.csv")
    df_sierraleone = pd.read_csv("data/clean_sierraleone.csv")

    for df in [df_benin, df_togo, df_sierraleone]:
        for col in ['time', 'Time', 'date', 'Date', 'datetime', 'Timestamp']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                break

    df_benin['Country'] = 'Benin'
    df_togo['Country'] = 'Togo'
    df_sierraleone['Country'] = 'Sierra Leone'

    df_all = pd.concat([df_benin, df_togo, df_sierraleone], ignore_index=True)
    return df_all

# Load data
df_all = load_data()

# Sidebar: Country and Metric Filters
st.sidebar.header("Filters")
all_countries = df_all['Country'].unique().tolist()
selected_countries = st.sidebar.multiselect(
    "Select countries:",
    options=all_countries,
    default=all_countries
)

metrics_options = ["GHI", "DNI", "DHI"]
selected_metrics = st.sidebar.multiselect(
    "Select indicators:",
    options=metrics_options,
    default=metrics_options
)

# Filter by country
filtered_df = df_all[df_all['Country'].isin(selected_countries)] if selected_countries else pd.DataFrame()

# Main Title
st.title("Cross-Country Solar Energy Comparison Dashboard")

# Section: Distribution Comparison
st.header("☀️ Country Comparison: Solar Energy Potential Based on Summary Statistics")
if filtered_df.empty:
    st.warning("Please select at least one country to display the charts.")
else:
    for metric in selected_metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=filtered_df, x='Country', y=metric, palette='Set2', ax=ax)
        ax.set_title(f"{metric} Distribution by Country")
        ax.set_xlabel("")
        ax.set_ylabel(f"{metric} (W/m²)")
        plt.tight_layout()
        st.pyplot(fig)

# Average GHI bar chart
if not filtered_df.empty:
    avg_ghi = filtered_df.groupby('Country')['GHI'].mean().sort_values(ascending=False)
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    ax_bar.bar(avg_ghi.index, avg_ghi.values, color="#1f77b4")
    ax_bar.set_title("Average GHI by Country")
    ax_bar.set_ylabel("GHI (W/m²)")
    ax_bar.set_xlabel("Country")
    ax_bar.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig_bar)

# Monthly trend plots per country
st.header("📆 Monthly Trends of Solar and Environmental Metrics")
for country in all_countries:
    if st.checkbox(f"Show monthly trend for {country}"):
        df_country = df_all[df_all['Country'] == country]
        if not df_country.empty:
            st.subheader(f"📍 {country}")
            plot_monthly_avg(
                df_country,
                columns=selected_metrics,
                country_name=country
            )
