import streamlit as st
import matplotlib.pyplot as plt
import os

# Update to actual plot path
plot_dir = "plots"

# Set page title
st.title("Cross-Country Solar Energy Comparison Dashboard")

# Sidebar: Country Filter
st.sidebar.header("Filters")
countries = ["Benin", "Togo", "Sierra Leone"]
selected_countries = st.sidebar.multiselect(
    "Select countries:",
    options=countries,
    default=countries
)

metrics_options = ["GHI", "DNI", "DHI"]
selected_metrics = st.sidebar.multiselect(
    "Select indicators:",
    options=metrics_options,
    default=metrics_options
)

# Show explanation
st.header("☀️ Country Comparison: Solar Energy Potential Based on Summary Statistics")
st.markdown("""
This dashboard compares solar irradiance indicators across three countries: **Benin**, **Togo**, and **Sierra Leone**.

The visualizations are based on:
- **Global Horizontal Irradiance (GHI)**
- **Direct Normal Irradiance (DNI)**
- **Diffuse Horizontal Irradiance (DHI)**

These metrics help evaluate the suitability of each country for different types of solar power systems.
""")

# Display pre-generated plots
plot_paths = {
    "ghi_boxplot": f"{plot_dir}/ghi_boxplot.png",
    "dni_boxplot": f"{plot_dir}/dni_boxplot.png",
    "dhi_boxplot": f"{plot_dir}/dhi_boxplot.png",
    "benin_monthly_avg": f"{plot_dir}/benin_monthly_avg.png",
    "togo_monthly_avg": f"{plot_dir}/togo_monthly_avg.png",
    "sierra_leone_monthly_avg": f"{plot_dir}/sierra_leone_monthly_avg.png",
}

# Display distribution plots
st.subheader("📊 Boxplots by Country")
if selected_countries:
    if any(c in selected_countries for c in ["Benin", "Togo", "Sierra Leone"]):
        st.image(plot_paths["ghi_boxplot"], caption="GHI Distribution by Country", use_container_width=True)
        st.image(plot_paths["dni_boxplot"], caption="DNI Distribution by Country", use_container_width=True)
        st.image(plot_paths["dhi_boxplot"], caption="DHI Distribution by Country", use_container_width=True)
else:
    st.warning("Please select at least one country to view the boxplots.")

# Monthly Trends
st.header("📆 Monthly Trends of Solar and Environmental Metrics")
if "Benin" in selected_countries:
    st.subheader("📍 Benin")
    st.image(plot_paths["benin_monthly_avg"], use_container_width=True)
if "Togo" in selected_countries:
    st.subheader("📍 Togo")
    st.image(plot_paths["togo_monthly_avg"], use_container_width=True)
if "Sierra Leone" in selected_countries:
    st.subheader("📍 Sierra Leone")
    st.image(plot_paths["sierra_leone_monthly_avg"], use_container_width=True)
