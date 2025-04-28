import pandas as pd
import matplotlib.pyplot as plt

# Load all STICS output files
ghi = pd.read_csv("stics_run_ghi.csv")
row_43 = pd.read_csv("stics_run_row_43.csv")
row_61 = pd.read_csv("stics_run_row_61.csv")
row_79 = pd.read_csv("stics_run_row_79.csv")

# Create a dictionary for easy iteration
runs = {
    "Nominal (GHI)": ghi,
    "Row 43": row_43,
    "Row 61": row_61,
    "Row 79": row_79
}

# Initialize a summary DataFrame
summary = pd.DataFrame(columns=["Run", "Total Yield (g/m2)", "Days to Maturity", "Max LAI"])

# Extract meaningful data
yield_results = []
maturity_days = []
labels = []
max_lai_values = []

for name, df in runs.items():
    total_yield = df['mafruit_rec'].sum()

    if 'mat' in df.columns:
        mat_date = df.loc[df['mat'] > 0].index.min()
        if pd.isna(mat_date):
            maturity_day = None
        else:
            maturity_day = df.loc[mat_date, 'doy']
    else:
        maturity_day = None

    max_lai = df['lai'].max()

    yield_results.append(total_yield)
    maturity_days.append(maturity_day)
    labels.append(name)
    max_lai_values.append(max_lai)

    summary = pd.concat([
        summary,
        pd.DataFrame({
            "Run": [name],
            "Total Yield (g/m2)": [total_yield],
            "Days to Maturity": [maturity_day],
            "Max LAI": [max_lai]
        })
    ], ignore_index=True)

# Save summary table
summary.to_csv("summary_results.csv", index=False)

# Yield Comparison Plot
plt.figure(figsize=(10,6))
plt.bar(labels, yield_results)
plt.ylabel("Total Yield (g/m²)")
plt.title("Impact of Solar Panels on Crop Yield")
plt.xticks(rotation=15)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("yield_comparison.png", dpi=300)
plt.show()

# LAI Evolution Plot (Corrected for Year 2 only)
plt.figure(figsize=(14,6))

for name, df in runs.items():
    df_second_year = df.iloc[365:].copy()  # Select second year
    df_second_year['simulation_day'] = range(1, len(df_second_year) + 1)
    plt.plot(df_second_year['simulation_day'], df_second_year['lai'], label=name)

plt.xlabel("Day of Simulation Year")
plt.ylabel("Leaf Area Index (LAI)")
plt.title("LAI Across Different Scenarios")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("lai_comparison.png", dpi=300)
plt.show()

print("Saved: summary_results.csv, yield_comparison.png, lai_comparison.png")
