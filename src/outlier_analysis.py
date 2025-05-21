import pandas as pd
from collections import defaultdict

def summarize_outliers_by_time(df_outlier, timestamp_col='Timestamp', outlier_col='z_outlier_cols'):
    """
    Summarizes how many times each column was flagged as an outlier,
    grouped by time-of-day intervals (e.g., midday, midnight).
    
    Parameters:
        df_outlier (pd.DataFrame): DataFrame with outlier rows, including a timestamp and a string column of outlier column names.
        timestamp_col (str): Name of the timestamp column.
        outlier_col (str): Name of the column containing comma-separated outlier column names.
    
    Returns:
        pd.DataFrame: A summary table showing counts of outliers per column across defined time intervals.
    """
    df = df_outlier.copy()
    df['Hour'] = pd.to_datetime(df[timestamp_col]).dt.hour

    # Define time intervals
    time_intervals = {
        "Midday (11AM - 1PM)": range(11, 14),
        "Midnight (11PM - 1AM)": [23, 0, 1],
        "After Midday (1PM - 11PM)": range(14, 23),
        "After Midnight (1AM - 11AM)": range(2, 11)
    }

    results = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        hour = row['Hour']
        cols = row[outlier_col].split(', ')

        # Determine interval
        if hour in time_intervals["Midday (11AM - 1PM)"]:
            interval = "Midday (11AM - 1PM)"
        elif hour in time_intervals["Midnight (11PM - 1AM)"]:
            interval = "Midnight (11PM - 1AM)"
        elif hour in time_intervals["After Midnight (1AM - 11AM)"]:
            interval = "After Midnight (1AM - 11AM)"
        elif hour in time_intervals["After Midday (1PM - 11PM)"]:
            interval = "After Midday (1PM - 11PM)"
        else:
            interval = "Other Times"

        for col in cols:
            results[col][interval] += 1

    return pd.DataFrame(results).fillna(0).astype(int).T
import pandas as pd
import matplotlib.pyplot as plt

def plot_monthly_outlier_distribution(df_outlier, timestamp_col='Timestamp', outlier_col='z_outlier_cols'):
    """
    Plots the monthly distribution of Z-score outliers for each sensor column.
    
    Parameters:
        df_outlier (pd.DataFrame): Outlier dataframe with Timestamp and comma-separated column name flags.
        timestamp_col (str): Name of the timestamp column (default: 'Timestamp').
        outlier_col (str): Name of the outlier columns flag (default: 'z_outlier_cols').
        
    Returns:
        pd.DataFrame: Monthly outlier frequency table by column.
    """
    df = df_outlier.copy()

    # Ensure timestamp is datetime
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    df['Month'] = df[timestamp_col].dt.month

    # Expand multiple outlier columns into rows
    outlier_expanded = df.assign(
        z_outlier_cols=df[outlier_col].str.split(', ')
    ).explode(outlier_col)

    # Group and count
    monthly_counts = (
        outlier_expanded
        .groupby([outlier_col, 'Month'])
        .size()
        .unstack(fill_value=0)
    )

    # Plot one bar chart per column
    for col in monthly_counts.index:
        plt.figure(figsize=(14, 4))
        monthly_counts.loc[col].plot(kind='bar', color='skyblue')
        plt.title(f'Monthly Distribution of Outliers for {col}', fontsize=14)
        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Number of Outliers', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    return monthly_counts

import pandas as pd

def replace_outliers_with_monthly_median(df, z_outlier_mask, cols, timestamp_col='Timestamp'):
    """
    Replaces Z-score outlier values in specified columns with the column's monthly median.

    Parameters:
        df (pd.DataFrame): The input DataFrame with a timestamp column and sensor columns.
        z_outlier_mask (pd.DataFrame): Boolean mask (same shape as df[cols]) where True = outlier.
        cols (list): List of columns to clean.
        timestamp_col (str): Name of the timestamp column to extract month from.

    Returns:
        pd.DataFrame: Updated DataFrame with outliers replaced by monthly medians.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df['month'] = df[timestamp_col].dt.month

    for col in cols:
        # Compute monthly median for the column
        monthly_medians = df.groupby('month')[col].transform('median')

        # Replace outliers with corresponding monthly median
        df.loc[z_outlier_mask[col], col] = monthly_medians[z_outlier_mask[col]]

    df.drop(columns='month', inplace=True)
    return df
