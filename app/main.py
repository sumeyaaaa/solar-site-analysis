import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import streamlit as st

# Ignore future warnings from seaborn
warnings.filterwarnings("ignore", category=FutureWarning)

# Set plot style for consistency
sns.set_style("whitegrid")

@st.cache_data
def load_data():
    # Load datasets (assuming CSV files are in the same directory as this app)
    df_benin = pd.read_csv("data\clean_benin.csv")
    df_togo = pd.read_csv("data\clean_togo.csv")
    df_sierraleone = pd.read_csv("data\clean_sierraleone.csv")
    # Attempt to parse datetime if a time/date column exists
    for df in [df_benin, df_togo, df_sierraleone]:
        for col in ['time', 'Time', 'date', 'Date', 'datetime', 'Timestamp']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                break
    # Add country labels
    df_benin['Country'] = 'Benin'
    df_togo['Country'] = 'Togo'
    df_sierraleone['Country'] = 'Sierra Leone'
    # Combine all data
    df_all = pd.concat([df_benin, df_togo, df_sierraleone], ignore_index=True)
    return df_all

# Load data (uses caching to avoid re-reading on each interaction)
df_all = load_data()

# Sidebar controls for interactivity
st.sidebar.header("Filters")
# Country selection
all_countries = df_all['Country'].unique().tolist()
selected_countries = st.sidebar.multiselect(
    "Select countries:",
    options=all_countries,
    default=all_countries
)
# Indicator (metric) selection
metrics_options = ["GHI", "DNI", "DHI"]
selected_metrics = st.sidebar.multiselect(
    "Select indicators:",
    options=metrics_options,
    default=metrics_options
)
# Date range selection (if date/time column exists)
date_col = None
for col in ['time', 'Time', 'date', 'Date', 'datetime', 'Timestamp']:
    if col in df_all.columns:
        date_col = col
        break
if date_col:
    # Determine min and max dates for the slider
    min_date = pd.to_datetime(df_all[date_col].min())
    max_date = pd.to_datetime(df_all[date_col].max())
    try:
        # If datetime, convert to date for display
        min_date_val = min_date.date()
        max_date_val = max_date.date()
    except AttributeError:
        min_date_val = min_date
        max_date_val = max_date
    # Date range input
    start_date, end_date = st.sidebar.date_input(
        "Date range:",
        value=(min_date_val, max_date_val)
    )
else:
    start_date, end_date = None, None

# Filter data according to selections
if selected_countries:
    filtered_df = df_all[df_all['Country'].isin(selected_countries)]
else:
    filtered_df = pd.DataFrame()  # empty if no country selected

if date_col and start_date is not None and end_date is not None and not filtered_df.empty:
    # If date range is selected, filter the data
    # Convert to date (without time) for comparison if needed
    if pd.api.types.is_datetime64_any_dtype(filtered_df[date_col]):
        date_only = filtered_df[date_col].dt.date
    else:
        date_only = pd.to_datetime(filtered_df[date_col]).dt.date
    filtered_df = filtered_df[(date_only >= start_date) & (date_only <= end_date)]

# Main title
st.title("Cross-Country Solar Energy Comparison Dashboard")

# Section 1: Summary statistics and distributions
st.header("☀️ Country Comparison: Solar Energy Potential Based on Summary Statistics")
# If no country is selected, prompt the user
if filtered_df.empty:
    st.warning("Please select at least one country to display the charts.")
else:
    # Plot distribution charts for each selected metric
    for metric in selected_metrics:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=filtered_df, x='Country', y=metric, palette='Set2', ax=ax)
        ax.set_title(f"{metric} Distribution by Country")
        ax.set_xlabel("")  # no label for x-axis (country names already shown)
        ax.set_ylabel(f"{metric} (W/m²)")
        plt.tight_layout()
        st.pyplot(fig)

# Explanation and summary (from notebook)
st.markdown("""
## ☀️ Country Comparison: Solar Energy Potential Based on Summary Statistics

Based on the summary statistics of **Global Horizontal Irradiance (GHI)**, **Direct Normal Irradiance (DNI)**, and **Diffuse Horizontal Irradiance (DHI)** for **Benin**, **Sierra Leone**, and **Togo**, the most promising country for solar investment is **Benin**.

### 📋 Summary Table:

| Metric              | Benin      | Sierra Leone | Togo       |
| ------------------- | ---------- | ------------ | ---------- |
| **GHI Mean (W/m²)** | **241.74** | 198.70       | 230.98     |
| **DNI Mean (W/m²)** | **167.44** | 104.66       | 149.37     |
| **DHI Mean (W/m²)** | 112.39     | 112.44       | **112.42** |

---

### 📌 Key Points:

- **🔆 Higher GHI Mean:**  
  Benin has the highest Global Horizontal Irradiance (GHI) average, indicating more solar energy on a flat surface.  
  → More consistent overall sunlight for traditional solar panel setups.

- **☀️ Strongest DNI (Direct Beam Irradiance):**  
  Benin also leads in DNI mean, the key indicator for concentrated solar power (CSP) systems and tracking panels.  
  → Ideal conditions for high-efficiency solar technologies.

- **🌥️ DHI is Nearly Identical:**  
  All three countries show similar DHI (~112 W/m²), meaning cloud-diffused sunlight doesn't majorly differentiate them.

- **📈 Trade-off in Variability:**  
  While Benin shows slightly **higher standard deviations**, its **high mean values** make it favorable overall for stable energy production.

---

### 📊 Conclusion:

**Benin** is the **most promising region** for solar energy development due to:

✅ The **highest average GHI and DNI values**  
✅ Suitability for both **traditional PV systems** and **CSP systems**  
✅ Variability is present but acceptable given the strong overall irradiance

---

### ✅ Recommendation:

- **Primary Focus:** Benin  
- **Secondary Option:** Togo  
- **Less Favorable (for now):** Sierra Leone, due to noticeably lower irradiance values
""")

# Section 2: Statistical testing of GHI differences
st.header("📊 Statistical Testing of GHI Across Benin, Sierra Leone, and Togo")
st.markdown("""
## 📊 Statistical Testing of GHI Across Benin, Sierra Leone, and Togo

To assess whether the **Global Horizontal Irradiance (GHI)** significantly differs across the three countries—**Benin**, **Sierra Leone**, and **Togo**—we used two statistical methods:

---

### 🔹 1. One-Way ANOVA (Analysis of Variance)

#### 🔍 Method Overview:
One-Way ANOVA is a **parametric test** that compares the **means** of three or more independent groups. It tests the **null hypothesis** that all group means are equal.

- **Test Statistic (F):**  
  Measures the ratio of variation **between group means** to the variation **within groups**.
  
  \[
  F = \frac{\text{variance between groups}}{\text{variance within groups}}
  \]
  
- A high F-value suggests that the group means differ more than would be expected by chance.

- **Assumptions:**
  - Normal distribution within groups
  - Equal variance (homoscedasticity)
  - Independence of observations

#### 🧪 Results:
- **F-statistic** = `2677.69`
- **p-value** = `0.0`

#### 📌 Interpretation of p-value:
The **p-value** indicates the probability of obtaining an F-statistic this large **if the null hypothesis were true** (i.e., if all countries had the same mean GHI).

- A **p-value of 0.0** means there is an **extremely low probability** (near zero) that the observed differences are due to chance.
- ✅ Therefore, we **reject the null hypothesis**.
- **Conclusion:** The **mean GHI values** of the three countries are **statistically significantly different**.

---

### 🔹 2. Kruskal–Wallis H-Test

#### 🔍 Method Overview:
The Kruskal–Wallis test is a **non-parametric alternative** to ANOVA. It compares **medians and distributions** of more than two groups without assuming normality.

- Instead of comparing means, it ranks all values across groups and tests if those ranks differ significantly between the groups.
- Useful when the data may be skewed, non-normal, or have unequal variances.

- **Test Statistic (H):**  
  Calculated based on the sum of ranks for each group.
  
  \[
  H = \frac{12}{N(N+1)} \sum \left( \frac{R_i^2}{n_i} \right) - 3(N+1)
  \]\n
  where:
  - \( R_i \): sum of ranks in group \( i \)
  - \( n_i \): size of group \( i \)
  - \( N \): total number of observations

#### 🧪 Results:
- **H-statistic** = `1769.18`
- **p-value** = `0.0`

#### 📌 Interpretation of p-value:
This p-value tells us the likelihood that all samples come from the **same distribution**.

- Again, a **p-value of 0.0** means the differences observed in the distributions of GHI are **highly significant**.
- ✅ So we **reject the null hypothesis**.
- **Conclusion:** The **distributions of GHI** are **not the same** across the three countries.

---

### ✅ Final Insight:

Both statistical tests—despite having different assumptions—strongly agree:

> 🔹 There is a **statistically significant difference** in GHI across Benin, Sierra Leone, and Togo.

This validates the need to:
- Analyze each country independently
- Avoid one-size-fits-all solar strategies
- Prioritize high-GHI regions like **Benin** for solar investments

These results form the foundation for **data-driven regional ranking** in solar energy planning.
""")

# Average GHI bar chart (visualizing the ranking)
if not filtered_df.empty:
    # Calculate average GHI for each selected country in the filtered data
    avg_ghi = filtered_df.groupby('Country')['GHI'].mean()
    # Sort countries by GHI for better visual ranking
    avg_ghi = avg_ghi.sort_values(ascending=False)
    fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
    ax_bar.bar(avg_ghi.index, avg_ghi.values, color="#1f77b4")
    ax_bar.set_title("Average GHI by Country")
    ax_bar.set_ylabel("GHI (W/m²)")
    ax_bar.set_xlabel("Country")
    ax_bar.yaxis.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig_bar)
