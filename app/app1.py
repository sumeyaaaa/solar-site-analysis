import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import streamlit as st
import sys
import os

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
                df.rename(columns={col: 'Timestamp'}, inplace=True)
                break

    df_benin['Country'] = 'Benin'
    df_togo['Country'] = 'Togo'
    df_sierraleone['Country'] = 'Sierra Leone'

    return df_benin, df_togo, df_sierraleone

def plot_monthly_avg(df, columns, country_name=''):
    """
    Plot monthly average values for specified columns.

    Parameters:
        df (pd.DataFrame): DataFrame with datetime index.
        columns (list): List of column names to group and plot.
        country_name (str): Optional title prefix.
    """
    df = df.copy()
    df['Month'] = pd.to_datetime(df['Timestamp']).dt.to_period('M')

    monthly_avg = df.groupby('Month')[columns].mean()

    plt.figure(figsize=(12, 6))
    for col in columns:
        plt.plot(monthly_avg.index.to_timestamp(), monthly_avg[col], label=col)

    plt.title(f'{country_name} – Monthly Averages of Irradiance & Environment')
    plt.xlabel('Month')
    plt.ylabel('Average Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(plt)

# Load data
df_benin, df_togo, df_sierraleone = load_data()
df_all = pd.concat([df_benin, df_togo, df_sierraleone], ignore_index=True)

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
country_df_map = {
    'Benin': df_benin,
    'Togo': df_togo,
    'Sierra Leone': df_sierraleone
}
for country in all_countries:
    if st.checkbox(f"Show monthly trend for {country}"):
        df_country = country_df_map[country]
        if not df_country.empty:
            st.subheader(f"📍 {country}")
            plot_monthly_avg(
                df_country,
                columns=selected_metrics,
                country_name=country
            )
