"""
AI Preparedness Index (AIPI) and GDP Analysis for ASEAN Countries

This script analyzes the relationship between IMF AI Preparedness Index (AIPI)
and GDP per capita across ASEAN nations, including regression analysis and
visualization of the relationship.

Data Source: IMF API
- AIPI: https://www.imf.org/external/datamapper/api/v1/AI_PI
- GDP per capita: https://www.imf.org/external/datamapper/api/v1/NGDPDPC

Output: Results saved to results/regression_output.txt
"""

import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import statsmodels.api as sm
from adjustText import adjust_text
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Configuration
ASEAN_ISO3 = ["IDN", "SGP", "MYS", "THA", "VNM", "PHL", "KHM", "LAO", "MMR", "BRN"]
YEAR_RANGE = (2020, 2023)
TARGET_YEAR = 2023

# Create results directory if it doesn't exist
os.makedirs('results/figures', exist_ok=True)


def fetch_imf_aipi_data(iso3_list, year_min, year_max):
    """
    Fetch AI Preparedness Index data from IMF API.
    
    Args:
        iso3_list (list): List of ISO3 country codes
        year_min (int): Minimum year
        year_max (int): Maximum year
    
    Returns:
        pd.DataFrame: DataFrame with columns [iso3, year, aipi]
    """
    url = "https://www.imf.org/external/datamapper/api/v1/AI_PI"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching AIPI data: {e}")
        return pd.DataFrame()
    
    rows = []
    values = data.get("values", {}).get("AI_PI", {})
    
    for iso3, year_dict in values.items():
        if iso3 not in iso3_list:
            continue
        for year, value in year_dict.items():
            year_int = int(year)
            if year_min <= year_int <= year_max and value is not None:
                rows.append({
                    "iso3": iso3,
                    "year": year_int,
                    "aipi": float(value)
                })
    
    return pd.DataFrame(rows)


def fetch_imf_gdp_data(iso3_list):
    """
    Fetch GDP per capita data from IMF API.
    
    Args:
        iso3_list (list): List of ISO3 country codes
    
    Returns:
        pd.DataFrame: DataFrame with columns [iso3, year, gdp_pc]
    """
    rows = []
    
    for iso3 in iso3_list:
        url = f"https://www.imf.org/external/datamapper/api/v1/NGDPDPC/{iso3}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            country_data = data.get("values", {}).get("NGDPDPC", {}).get(iso3, {})
            
            for year, value in country_data.items():
                if value is not None:
                    rows.append({
                        "iso3": iso3,
                        "year": int(year),
                        "gdp_pc": float(value)
                    })
        except requests.RequestException as e:
            print(f"Warning: Error fetching data for {iso3}: {e}")
            continue
    
    return pd.DataFrame(rows)


def merge_datasets(aipi_df, gdp_df):
    """
    Merge AIPI and GDP datasets.
    
    Args:
        aipi_df (pd.DataFrame): AIPI data
        gdp_df (pd.DataFrame): GDP data
    
    Returns:
        pd.DataFrame: Merged dataset
    """
    return pd.merge(aipi_df, gdp_df, on=["iso3", "year"], how="inner")


def add_country_names(df):
    """
    Map ISO3 codes to full country names.
    
    Args:
        df (pd.DataFrame): DataFrame with iso3 column
    
    Returns:
        pd.DataFrame: DataFrame with added country column
    """
    iso3_to_country = {
        "IDN": "Indonesia",
        "SGP": "Singapore",
        "MYS": "Malaysia",
        "THA": "Thailand",
        "VNM": "Vietnam",
        "PHL": "Philippines",
        "KHM": "Cambodia",
        "LAO": "Laos",
        "MMR": "Myanmar",
        "BRN": "Brunei"
    }
    
    df = df.copy()
    df["country"] = df["iso3"].map(iso3_to_country)
    return df


def fit_baseline_model(df):
    """
    Fit baseline OLS regression model: log(GDP) ~ AIPI
    
    Args:
        df (pd.DataFrame): Dataset with aipi and log_gdp_pc columns
    
    Returns:
        statsmodels.regression.linear_model.RegressionResults: Fitted model
    """
    X = df[["aipi"]].copy()
    X = sm.add_constant(X)
    y = df["log_gdp_pc"]
    
    model = sm.OLS(y, X).fit()
    return model


def fit_interaction_model(df):
    """
    Fit interaction model to test differential effects by AIPI level.
    
    Model: log(GDP) ~ AIPI + low_aipi + AIPI*low_aipi
    
    Args:
        df (pd.DataFrame): Dataset with aipi and log_gdp_pc columns
    
    Returns:
        statsmodels.regression.linear_model.RegressionResults: Fitted model
    """
    df = df.copy()
    median_aipi = df["aipi"].median()
    df["low_aipi"] = (df["aipi"] < median_aipi).astype(int)
    df["aipi_low_interaction"] = df["aipi"] * df["low_aipi"]
    
    X = df[["aipi", "low_aipi", "aipi_low_interaction"]]
    X = sm.add_constant(X)
    y = df["log_gdp_pc"]
    
    model = sm.OLS(y, X).fit()
    return model


def plot_scatter_with_regression(df, model):
    """
    Create scatter plot with regression line and country labels.
    Saves figure to results/figures/regression_plot.png
    
    Args:
        df (pd.DataFrame): Dataset with aipi, log_gdp_pc, and country columns
        model: Fitted OLS model
    """
    X = df[["aipi"]].copy()
    X = sm.add_constant(X)
    predictions = model.predict(X)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(df["aipi"], df["log_gdp_pc"], s=100, alpha=0.6, edgecolors='black')
    plt.plot(df["aipi"].sort_values(), predictions.sort_values(), 
             color='red', linewidth=2, label='OLS Fit')
    
    texts = []
    for i, row in df.iterrows():
        texts.append(plt.text(row["aipi"], row["log_gdp_pc"], 
                             row["country"], fontsize=9))
    
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5))
    
    plt.xlabel("AI Preparedness Index (AIPI)", fontsize=11)
    plt.ylabel("Log GDP per capita (2023)", fontsize=11)
    plt.title("AI Preparedness and Productivity in ASEAN", fontsize=13, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    
    # Save figure
    plt.savefig('results/figures/regression_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved: results/figures/regression_plot.png")
    plt.show()


def plot_interaction_model(df, model_interaction):
    """
    Create scatter plot with separate regression lines for high vs low AIPI groups.
    Saves figure to results/figures/interaction_plot.png
    
    Args:
        df (pd.DataFrame): Dataset with aipi, log_gdp_pc, and country columns
        model_interaction: Fitted interaction model
    """
    df_plot = df.copy()
    median_aipi = df_plot["aipi"].median()
    df_plot["aipi_group"] = df_plot["aipi"].apply(
        lambda x: "High AI Preparedness" if x >= median_aipi else "Low AI Preparedness"
    )
    
    plt.figure(figsize=(12, 7))
    
    # Plot points colored by group
    for group, color in [("Low AI Preparedness", '#FF9500'), ("High AI Preparedness", '#1f77b4')]:
        df_group = df_plot[df_plot["aipi_group"] == group]
        plt.scatter(df_group["aipi"], df_group["log_gdp_pc"], 
                   s=120, alpha=0.7, edgecolors='black', linewidth=1,
                   label=group, color=color)
    
    # Create prediction lines for each group
    aipi_range = np.linspace(df_plot["aipi"].min() - 0.5, 
                             df_plot["aipi"].max() + 0.5, 100)
    
    # Low AIPI group - fit line only in low range
    low_aipi_dummy = (aipi_range < median_aipi).astype(int)
    interaction = aipi_range * low_aipi_dummy
    X_low = np.column_stack([np.ones(len(aipi_range)), aipi_range, 
                             low_aipi_dummy, interaction])
    y_low = model_interaction.predict(X_low)
    # Only plot line where low_aipi_dummy is 1
    low_mask = aipi_range < median_aipi
    plt.plot(aipi_range[low_mask], y_low[low_mask], color='#FF9500', linewidth=3, 
            label='Low AI Preparedness')
    
    # High AIPI group - fit line only in high range
    high_aipi_dummy = (aipi_range >= median_aipi).astype(int)
    interaction_high = aipi_range * (1 - high_aipi_dummy)
    X_high = np.column_stack([np.ones(len(aipi_range)), aipi_range, 
                              1 - high_aipi_dummy, interaction_high])
    y_high = model_interaction.predict(X_high)
    # Only plot line where high_aipi_dummy is 1
    high_mask = aipi_range >= median_aipi
    plt.plot(aipi_range[high_mask], y_high[high_mask], color='#1f77b4', linewidth=3, 
            label='High AI Preparedness')
    
    # Add country labels
    texts = []
    for i, row in df_plot.iterrows():
        texts.append(plt.text(row["aipi"], row["log_gdp_pc"], 
                             row["iso3"], fontsize=10, fontweight='bold'))
    
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
               expand_points=(1.5, 1.5))
    
    plt.xlabel("AI Preparedness Index (AIPI)", fontsize=12, fontweight='bold')
    plt.ylabel("Log GDP per capita 2023", fontsize=12, fontweight='bold')
    plt.title("Diverging Productivity Effects of AI Preparedness", fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11, framealpha=0.95)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    # Save figure
    plt.savefig('results/figures/interaction_plot.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved: results/figures/interaction_plot.png")
    plt.show()


def generate_summary_statistics(df):
    """
    Generate summary statistics table for 2023.
    
    Args:
        df (pd.DataFrame): Full dataset
    
    Returns:
        pd.DataFrame: Summary statistics for 2023
    """
    df_2023 = df[df["year"] == TARGET_YEAR].copy()
    df_2023 = df_2023.sort_values("gdp_pc", ascending=False)
    
    summary = df_2023[["country", "aipi", "gdp_pc", "log_gdp_pc"]].copy()
    summary = summary.rename(columns={
        "country": "Country",
        "aipi": "AIPI Score",
        "gdp_pc": "GDP per Capita (USD)",
        "log_gdp_pc": "Log GDP per Capita"
    })
    
    return summary


def save_results_to_file(df, model_baseline, model_interaction, summary_stats):
    """
    Save all results and statistics to a formatted text file for use in FINDINGS.md
    
    Args:
        df (pd.DataFrame): Full dataset
        model_baseline: Baseline regression model
        model_interaction: Interaction regression model
        summary_stats (pd.DataFrame): Summary statistics
    """
    
    with open('results/regression_output.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("AI PREPAREDNESS INDEX (AIPI) AND GDP ANALYSIS - ASEAN REGION\n")
        f.write("="*80 + "\n")
        f.write(f"Analysis Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Summary Statistics
        f.write("SUMMARY STATISTICS (2023)\n")
        f.write("-"*80 + "\n")
        f.write(summary_stats.to_string(index=False))
        f.write("\n\n")
        
        # Descriptive Statistics
        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"Number of Countries: {df['iso3'].nunique()}\n")
        f.write(f"Number of Years: {df['year'].nunique()}\n")
        f.write(f"Total Observations: {len(df)}\n\n")
        
        f.write("AIPI Statistics (All Years):\n")
        f.write(df['aipi'].describe().to_string())
        f.write("\n\n")
        
        f.write("GDP per Capita Statistics (All Years, USD):\n")
        f.write(df['gdp_pc'].describe().to_string())
        f.write("\n\n")
        
        f.write("Log GDP per Capita Statistics:\n")
        f.write(df['log_gdp_pc'].describe().to_string())
        f.write("\n\n")
        
        # Baseline Regression
        f.write("="*80 + "\n")
        f.write("BASELINE OLS REGRESSION RESULTS\n")
        f.write("="*80 + "\n")
        f.write("Model: log(GDP per capita) ~ AIPI\n")
        f.write("Dependent Variable: Log GDP per capita\n")
        f.write("-"*80 + "\n\n")
        f.write(str(model_baseline.summary()))
        f.write("\n\n")
        
        # Extract key coefficients
        f.write("KEY FINDINGS FROM BASELINE MODEL:\n")
        f.write("-"*80 + "\n")
        intercept = model_baseline.params['const']
        aipi_coef = model_baseline.params['aipi']
        aipi_pval = model_baseline.pvalues['aipi']
        r_squared = model_baseline.rsquared
        
        f.write(f"Intercept: {intercept:.4f}\n")
        f.write(f"AIPI Coefficient: {aipi_coef:.4f}\n")
        f.write(f"AIPI P-value: {aipi_pval:.6f}\n")
        f.write(f"R-squared: {r_squared:.4f}\n")
        f.write(f"Adjusted R-squared: {model_baseline.rsquared_adj:.4f}\n")
        f.write(f"\nInterpretation: A 1-unit increase in AIPI is associated with a {aipi_coef*100:.2f}% ")
        f.write(f"increase in GDP per capita.\n")
        if aipi_pval < 0.05:
            f.write(f"Statistical Significance: YES (p-value = {aipi_pval:.6f} < 0.05)\n")
        else:
            f.write(f"Statistical Significance: NO (p-value = {aipi_pval:.6f} > 0.05)\n")
        f.write("\n\n")
        
        # Interaction Regression
        f.write("="*80 + "\n")
        f.write("INTERACTION MODEL RESULTS\n")
        f.write("="*80 + "\n")
        f.write("Model: log(GDP) ~ AIPI + low_aipi + (AIPI × low_aipi)\n")
        f.write("Tests differential effects by AIPI level\n")
        f.write("-"*80 + "\n\n")
        f.write(str(model_interaction.summary()))
        f.write("\n\n")
        
        # Correlation Analysis
        f.write("="*80 + "\n")
        f.write("CORRELATION ANALYSIS\n")
        f.write("="*80 + "\n")
        correlation = df['aipi'].corr(df['log_gdp_pc'])
        f.write(f"Pearson Correlation (AIPI vs Log GDP): {correlation:.4f}\n")
        f.write("\n\n")
        
        # Data Quality Notes
        f.write("="*80 + "\n")
        f.write("DATA QUALITY & NOTES\n")
        f.write("="*80 + "\n")
        f.write(f"Countries Included: {', '.join(df['country'].unique())}\n")
        f.write(f"Time Period: {df['year'].min()}-{df['year'].max()}\n")
        f.write(f"Data Source: IMF DataMapper API\n")
        f.write(f"AIPI Indicator: AI_PI\n")
        f.write(f"GDP Indicator: NGDPDPC (GDP per capita, current prices USD)\n")
        f.write("\n\n")
        
        # Footer
        f.write("="*80 + "\n")
        f.write("For detailed interpretation and context, see FINDINGS.md\n")
        f.write("="*80 + "\n")
    
    print("✓ Results saved: results/regression_output.txt")


def main():
    """Main analysis pipeline."""
    
    print("="*80)
    print("AI PREPAREDNESS INDEX (AIPI) AND GDP ANALYSIS - ASEAN REGION")
    print("="*80)
    
    # Fetch data
    print("\n1. Fetching AIPI data...")
    aipi_df = fetch_imf_aipi_data(ASEAN_ISO3, YEAR_RANGE[0], YEAR_RANGE[1])
    print(f"   ✓ Retrieved {len(aipi_df)} AIPI observations")
    
    print("2. Fetching GDP data...")
    gdp_df = fetch_imf_gdp_data(ASEAN_ISO3)
    print(f"   ✓ Retrieved {len(gdp_df)} GDP observations")
    
    # Merge and prepare
    print("3. Merging datasets...")
    df = merge_datasets(aipi_df, gdp_df)
    df = add_country_names(df)
    df["log_gdp_pc"] = np.log(df["gdp_pc"])
    print(f"   ✓ Final dataset: {len(df)} observations")
    
    # Generate summary statistics
    print("4. Generating summary statistics...")
    summary_stats = generate_summary_statistics(df)
    print("   ✓ Summary statistics generated")
    
    # Baseline regression
    print("5. Fitting baseline OLS model: log(GDP) ~ AIPI")
    model_baseline = fit_baseline_model(df)
    print("   ✓ Baseline model complete")
    print(f"   R-squared: {model_baseline.rsquared:.4f}")
    
    # Interaction model
    print("6. Fitting interaction model...")
    model_interaction = fit_interaction_model(df)
    print("   ✓ Interaction model complete")
    print(f"   R-squared: {model_interaction.rsquared:.4f}")
    
    # Save results to file
    print("7. Saving results to file...")
    save_results_to_file(df, model_baseline, model_interaction, summary_stats)
    
    # Visualization
    print("8. Creating visualizations...")
    plot_scatter_with_regression(df, model_baseline)
    plot_interaction_model(df, model_interaction)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nOutput Files Generated:")
    print("  • results/regression_output.txt - Full regression results and statistics")
    print("  • results/figures/regression_plot.png - Baseline model visualization")
    print("  • results/figures/interaction_plot.png - Interaction model visualization")
    print("\nNext Step: Use regression_output.txt to populate FINDINGS.md")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()