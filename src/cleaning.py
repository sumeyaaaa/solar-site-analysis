import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd

def replace_negative_irradiance_with_hourly_medians(df, cols=['GHI', 'DNI', 'DHI']):
    """
    Replaces negative values in GHI, DNI, and DHI columns of the original dataset based on:
    - 'Night' and 'Midnight': set to 0
    - 'Day': replace with median of that hour across all days (hour-wise median)
    
    Parameters:
        df (pd.DataFrame): Original DataFrame with 'Timestamp' and irradiance columns
        cols (list): List of irradiance columns to clean
        
    Returns:
        df (pd.DataFrame): The same DataFrame with negative values replaced in-place
    """
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['hour'] = df['Timestamp'].dt.hour

    # Time category classification per row
    def classify_period(hour):
        if 11 <= hour <= 13:
            return 'Midday'
        elif 6 <= hour <= 10:
            return 'Day'
        elif 0 <= hour <= 5:
            return 'Midnight'
        else:
            return 'Night'
    
    df['time_category'] = df['hour'].apply(classify_period)

    for col in cols:
        # Step 1: Calculate hourly median for non-negative daytime values
        hour_medians = (
            df[(df['time_category'] == 'Day') & (df[col] >= 0)]
            .groupby('hour')[col].median()
        )

        # Step 2: Apply cleaning
        def clean_value(row):
            if row[col] < 0:
                if row['time_category'] in ['Night', 'Midnight']:
                    return 0
                elif row['time_category'] == 'Day':
                    return hour_medians.get(row['hour'], 0)
            return row[col]

        df[col] = df.apply(clean_value, axis=1)

    # Drop helper columns if desired
    df.drop(columns=['hour', 'time_category'], inplace=True)
    return df
import pandas as pd
import numpy as np
from scipy.stats import zscore

def detect_zscore_outliers(df, columns, timestamp_col='Timestamp', z_thresh=3):
    """
    Detects rows with Z-score outliers across specified columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to check for outliers.
        timestamp_col (str): The name of the timestamp column (default 'Timestamp').
        z_thresh (float): Z-score threshold for identifying outliers.

    Returns:
        pd.DataFrame: DataFrame with rows containing outliers,
                      with an extra column 'z_outlier_cols' listing affected columns.
    """
    df = df.copy()

    # Reset index in case Timestamp is in the index
    df.reset_index(inplace=True)

    # Ensure timestamp is in datetime format
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
    else:
        raise ValueError(f"'{timestamp_col}' column not found in DataFrame.")

    # Optional: Add month column for grouping or future use
    df['month'] = df[timestamp_col].dt.month

    # Step 1: Compute Z-scores for selected columns
    z_scores = df[columns].apply(zscore)

    # Step 2: Create mask where Z-score > threshold
    z_outlier_mask = np.abs(z_scores) > z_thresh

    # Step 3: Identify rows with any outlier
    rows_with_outliers = z_outlier_mask.any(axis=1)

    # Step 4: Extract affected rows + columns
    outlier_info = df.loc[rows_with_outliers, [timestamp_col] + columns].copy()
    outlier_info['z_outlier_cols'] = z_outlier_mask[rows_with_outliers].apply(
        lambda row: ', '.join(row[row].index.tolist()), axis=1
    )

    return outlier_info
