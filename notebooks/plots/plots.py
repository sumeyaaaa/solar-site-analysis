import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# Add the root to sys.path so Python can find the src folder
import sys
import os
sys.path.append(os.path.abspath(".."))

from src import load_csv

# Load the Benin dataset
df_sierraleone = load_csv('clean_sierraleone.csv')
df_togo = load_csv('clean_togo.csv')
df_benin = load_csv('clean_benin.csv')
# Combine datasets
df_benin['Country'] = 'Benin'
df_togo['Country'] = 'Togo'
df_sierraleone['Country'] = 'Sierra Leone'
df_all = pd.concat([df_benin, df_togo, df_sierraleone], ignore_index=True)

# Set plot style
sns.set(style="whitegrid")

# Create 'plots' directory if it doesn't exist
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

# GHI boxplot
df_ghi = df_all[['GHI', 'Country']]
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_ghi, x='Country', y='GHI', palette='Set2')
plt.title('GHI Distribution by Country')
plt.ylabel('GHI (W/m²)')
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'ghi_boxplot.png'))
plt.close()

# DNI boxplot
df_dni = df_all[['DNI', 'Country']]
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_dni, x='Country', y='DNI', palette='Set2')
plt.title('DNI Distribution by Country')
plt.ylabel('DNI (W/m²)')
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'dni_boxplot.png'))
plt.close()

# DHI boxplot
df_dhi = df_all[['DHI', 'Country']]
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_dhi, x='Country', y='DHI', palette='Set2')
plt.title('DHI Distribution by Country')
plt.ylabel('DHI (W/m²)')
plt.xlabel('')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'dhi_boxplot.png'))
plt.close()


import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_monthly_avg(df, columns, country_name=''):
    """
    Plot monthly average values for specified columns and save the plot as an image file.

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

    # Ensure the 'plots' directory exists
    os.makedirs('plots', exist_ok=True)

    # Save the plot as an image file
    filename = f'plots/{country_name.lower().replace(" ", "_")}_monthly_avg.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
plot_monthly_avg(df_benin, columns=['GHI', 'DNI', 'DHI'], country_name='Benin')
plot_monthly_avg(df_togo, columns=['GHI', 'DNI', 'DHI'], country_name='Togo')
plot_monthly_avg(df_sierraleone, columns=['GHI', 'DNI', 'DHI'], country_name='Sierra Leone')
