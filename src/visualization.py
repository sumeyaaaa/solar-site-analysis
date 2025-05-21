import matplotlib.pyplot as plt
import pandas as pd

def plot_hourly_irradiance(df, country_name):
    """
    Plot hourly average GHI, DNI, and DHI over time for a given country's DataFrame.
    Assumes the DataFrame has a 'Timestamp' column and the three irradiance columns.
    """
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)
    
    df_hourly = df[['GHI', 'DNI', 'DHI']].resample('H').mean()

    plt.figure(figsize=(15, 5))
    plt.plot(df_hourly.index, df_hourly['GHI'], label='GHI', color='gold')
    plt.plot(df_hourly.index, df_hourly['DNI'], label='DNI', color='red')
    plt.plot(df_hourly.index, df_hourly['DHI'], label='DHI', color='skyblue')
    
    plt.title(f'Solar Irradiance Components Over Time - {country_name} (Hourly Avg)')
    plt.xlabel('Timestamp')
    plt.ylabel('Irradiance (W/m²)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_nighttime_irradiance(df, country_name, irradiance_cols=['GHI', 'DNI', 'DHI']):
    """
    Plot nighttime values of irradiance components and highlight negative readings.

    Parameters:
        df (DataFrame): The input DataFrame with 'Timestamp' and irradiance columns.
        country_name (str): For use in plot titles.
        irradiance_cols (list): List of irradiance columns to analyze.
    """
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    for col in irradiance_cols:
        # Resample to hourly averages
        df_hourly = df[[col]].resample('H').mean()
        df_hourly['hour'] = df_hourly.index.hour
        df_hourly['is_night'] = ~df_hourly['hour'].between(6, 18)
        df_hourly['is_negative'] = df_hourly[col] < 0

        df_night = df_hourly[df_hourly['is_night']]
        negative_values = df_night[df_night['is_negative']]

        # Plot
        plt.figure(figsize=(15, 5))
        plt.plot(df_night.index, df_night[col], label=f'{col} at Night', color='blue', linewidth=1)
        plt.scatter(negative_values.index, negative_values[col], color='red', label='Negative Values', zorder=3)

        plt.title(f'{country_name}: Nighttime {col} with Negative Values Highlighted')
        plt.xlabel('Timestamp')
        plt.ylabel(f'{col} (W/m²)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
import pandas as pd
import matplotlib.pyplot as plt

def plot_negative_irradiance_counts_by_time(df, country_name, irradiance_cols=['GHI', 'DNI', 'DHI']):
    """
    Classify hourly irradiance data into time periods and plot negative value counts for each.

    Parameters:
        df (pd.DataFrame): DataFrame with a 'Timestamp' column and irradiance measurements.
        country_name (str): For title context.
        irradiance_cols (list): List of irradiance columns to check.
    """
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.set_index('Timestamp', inplace=True)

    # Resample to hourly averages
    df_hourly = df[irradiance_cols].resample('H').mean()
    df_hourly['hour'] = df_hourly.index.hour

    # Classify into time-of-day categories
    def classify_period(hour):
        if 11 <= hour <= 13:
            return 'Midday'
        elif 6 <= hour <= 10:
            return 'Day'
        elif 0 <= hour <= 5:
            return 'Midnight'
        else:
            return 'Night'

    df_hourly['time_category'] = df_hourly['hour'].apply(classify_period)

    # Count negative values by time category
    negative_counts = {}
    for col in irradiance_cols:
        mask = df_hourly[col] < 0
        counts = (
            df_hourly[mask]['time_category']
            .value_counts()
            .reindex(['Midday', 'Day', 'Night', 'Midnight'], fill_value=0)
        )
        negative_counts[col] = counts

    # Create DataFrame and plot
    neg_counts_df = pd.DataFrame(negative_counts)
    neg_counts_df.plot(kind='bar', figsize=(10, 6), color=['orange', 'crimson', 'purple'])
    plt.title(f'Negative Value Counts by Time Category - {country_name}')
    plt.xlabel('Time Category')
    plt.ylabel('Number of Negative Hours')
    plt.legend(title='Irradiance Type')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

def plot_trend(df, column, country_name='', window=24, title_suffix='Trend Over Time'):
    """
    Plot raw and smoothed time-series trend for a given column.
    
    Parameters:
        df (pd.DataFrame): DataFrame with datetime index.
        column (str): Column to plot.
        country_name (str): Optional prefix for the chart title.
        window (int): Rolling window size (e.g. 24 for hourly smoothing).
        title_suffix (str): Suffix text for the plot title.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df[column], color='lightgray', alpha=0.5, label='Raw')

    smoothed = df[column].rolling(window=window, min_periods=1).mean()
    plt.plot(df.index, smoothed, color='crimson', linewidth=2, label=f'{column} ({window}-pt avg)')

    plt.title(f'{country_name} – {column} {title_suffix}')
    plt.xlabel('Timestamp')
    plt.ylabel(column)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
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
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_cleaning_effect_on_module_irradiance(df, cleaning_col='Cleaning', mod_cols=['ModA', 'ModB']):
    """
    Plots the average module irradiance (ModA, ModB) before and after cleaning.

    Parameters:
        df (pd.DataFrame): DataFrame containing cleaning flags and module readings.
        cleaning_col (str): Column indicating cleaning events (e.g., 0 = before, 1 = after).
        mod_cols (list): List of module columns to compare (default ['ModA', 'ModB']).
        
    Returns:
        pd.DataFrame: Grouped average irradiance values.
    """
    # Group and calculate average
    avg_by_cleaning = df.groupby(cleaning_col)[mod_cols].mean()

    # Plot
    avg_by_cleaning.plot(kind='bar', figsize=(8, 5), color=['orange', 'skyblue'])
    plt.title("Average ModA and ModB Readings Before (0) and After (1) Cleaning")
    plt.ylabel("Average Irradiance (W/m²)")
    plt.xlabel("Cleaning Flag")
    plt.xticks(ticks=[0, 1], labels=['Before Cleaning (0)', 'After Cleaning (1)'], rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return avg_by_cleaning
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation_heatmap(df, columns, title='Correlation Heatmap'):
    """
    Plots a correlation heatmap for selected numerical columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of numerical columns to include in the correlation matrix.
        title (str): Title of the heatmap plot.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    df_clean = df[columns].dropna()
    corr_matrix = df_clean.corr()

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(title)
    plt.tight_layout()
    plt.show()

    return corr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_multiple_scatter_plots(df, scatter_pairs, ncols=2, title=None):
    """
    Creates a grid of scatter plots for given x–y pairs.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        scatter_pairs (list of tuples): Each tuple is (x_col, y_col, plot_title).
        ncols (int): Number of columns in the subplot grid.
        title (str or None): Optional overall title.

    Returns:
        None
    """
    nplots = len(scatter_pairs)
    nrows = (nplots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = axes.flatten()

    for idx, (x, y, subtitle) in enumerate(scatter_pairs):
        sns.scatterplot(data=df, x=x, y=y, ax=axes[idx])
        axes[idx].set_title(subtitle)
    
    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

def plot_ghi_ws_histograms(df, ghi_col='GHI', ws_col='WS', bins=50):
    """
    Plots side-by-side histograms with KDE for GHI and Wind Speed (WS).

    Parameters:
        df (pd.DataFrame): Input DataFrame with GHI and WS columns.
        ghi_col (str): Name of the GHI column.
        ws_col (str): Name of the Wind Speed column.
        bins (int): Number of histogram bins.

    Returns:
        None
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    sns.histplot(df[ghi_col].dropna(), bins=bins, kde=True, ax=ax[0], color='orange')
    ax[0].set_title('Histogram of GHI')
    ax[0].set_xlabel(ghi_col)

    sns.histplot(df[ws_col].dropna(), bins=bins, kde=True, ax=ax[1], color='skyblue')
    ax[1].set_title('Histogram of Wind Speed (WS)')
    ax[1].set_xlabel(ws_col)

    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyze_rh_temperature_radiation(df, columns=['RH', 'Tamb', 'GHI', 'DNI', 'DHI']):
    """
    Plots:
    1. A heatmap of correlations between RH, Tamb, and solar radiation.
    2. A scatter plot of Relative Humidity vs Ambient Temperature.

    Parameters:
        df (pd.DataFrame): DataFrame containing the specified columns.
        columns (list): Columns to include in analysis (default: RH, Tamb, GHI, DNI, DHI).
    
    Returns:
        pd.DataFrame: Correlation matrix used in the heatmap.
    """
    df_clean = df[columns].dropna()

    # 1. Correlation Heatmap
    corr = df_clean.corr()
    plt.figure(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Between RH, Temperature and Solar Radiation")
    plt.tight_layout()
    plt.show()

    # 2. RH vs Tamb Scatter Plot
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df_clean, x='RH', y='Tamb', alpha=0.4)
    plt.title('Relative Humidity vs Ambient Temperature')
    plt.xlabel('Relative Humidity (%)')
    plt.ylabel('Ambient Temperature (°C)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return corr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rh_temp_and_radiation_correlation(df, columns=['RH', 'Tamb', 'GHI', 'DNI', 'DHI']):
    """
    Plots:
    1. Correlation heatmap of RH, Tamb, and solar radiation columns.
    2. Scatter plot of Relative Humidity vs Ambient Temperature.

    Parameters:
        df (pd.DataFrame): DataFrame containing relevant environmental and solar radiation columns.
        columns (list): List of columns to include in the correlation and scatter plots.

    Returns:
        pd.DataFrame: Correlation matrix for the provided columns.
    """
    # Drop missing values
    df_clean = df[columns].dropna()

    # 1. Correlation Matrix Heatmap
    corr_matrix = df_clean.corr()
    plt.figure(figsize=(8, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Between RH, Temperature, and Solar Radiation")
    plt.tight_layout()
    plt.show()

    # 2. Scatter Plot: RH vs Tamb
    if 'RH' in df_clean.columns and 'Tamb' in df_clean.columns:
        plt.figure(figsize=(7, 5))
        sns.scatterplot(data=df_clean, x='RH', y='Tamb', alpha=0.4)
        plt.title('Relative Humidity vs Ambient Temperature')
        plt.xlabel('Relative Humidity (%)')
        plt.ylabel('Ambient Temperature (°C)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return corr_matrix

import matplotlib.pyplot as plt
import pandas as pd

def plot_bubble_chart(df, x='Tamb', y='GHI', bubble_col='RH', max_bubble_size=300, title=None):
    """
    Plots a bubble chart for GHI vs Tamb, with bubble size scaled by a third variable (e.g., RH or BP).

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        x (str): Column for x-axis (e.g., 'Tamb').
        y (str): Column for y-axis (e.g., 'GHI').
        bubble_col (str): Column to scale bubble size (e.g., 'RH' or 'BP').
        max_bubble_size (int): Maximum bubble size after scaling.
        title (str): Plot title.

    Returns:
        None
    """
    df_clean = df[[x, y, bubble_col]].dropna().copy()

    # Scale the bubble size for visual impact
    df_clean['bubble_size'] = df_clean[bubble_col] / df_clean[bubble_col].max() * max_bubble_size

    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df_clean[x], df_clean[y],
                s=df_clean['bubble_size'],
                alpha=0.4,
                color='skyblue',
                edgecolors='gray')

    plt.title(title or f'{y} vs {x} with Bubble Size = {bubble_col}')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
