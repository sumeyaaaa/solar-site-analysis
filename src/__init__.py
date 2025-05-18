import os
import pandas as pd

# Base data path (relative from the root of the repo)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def load_csv(file_name):
    """
    Load an Csv file from the data directory.

    Args:
        file_name (str): The filename (the 3 datas there like 'benin-malanville.csv')

    Returns:
         Loaded DataFrame
    """
    file_path = os.path.join(DATA_DIR, file_name)
    return pd.read_csv(file_path)

def summary_statistics(df):
    return df.describe()

def missing_value_report(df, threshold=0.05):
    missing_counts = df.isna().sum()
    total = len(df)
    high_nulls = missing_counts[missing_counts / total > threshold].index.tolist()

    return {
        "missing_counts": missing_counts,
        "high_null_columns": high_nulls
    }
