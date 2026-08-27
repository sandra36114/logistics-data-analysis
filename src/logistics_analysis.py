import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# Logistics Data Analysis - Week 1
# ==========================================

# Load dataset
# Replace the filename with your actual dataset file.
df = pd.read_csv("data/smart_logistics_dataset.csv")

# ------------------------------------------
# 1. Basic Data Exploration
# ------------------------------------------

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

print("\nShape after removing duplicates:", df.shape)


# ------------------------------------------
# 2. Logistics KPI Analysis
# ------------------------------------------

# Calculate delay rate if the column exists
if "Logistics_Delay" in df.columns:
    delay_rate = df["Logistics_Delay"].mean() * 100
    print("\nLogistics Delay Rate:",
          round(delay_rate, 2), "%")


# Calculate average waiting time
if "Waiting_Time" in df.columns:
    average_waiting_time = df["Waiting_Time"].mean()
    print("Average Waiting Time:",
          round(average_waiting_time, 2))


# Calculate average asset utilization
if "Asset_Utilization" in df.columns:
    asset_utilization = df["Asset_Utilization"].mean()
    print("Average Asset Utilization:",
          round(asset_utilization, 2))


# ------------------------------------------
# 3. Traffic and Waiting-Time Analysis
# ------------------------------------------

if "Traffic_Status" in df.columns and "Waiting_Time" in df.columns:

    traffic_summary = (
        df.groupby("Traffic_Status")["Waiting_Time"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\nAverage Waiting Time by Traffic Status:")
    print(traffic_summary)

    traffic_summary.plot(
        kind="bar",
        title="Average Waiting Time by Traffic Status"
    )

    plt.xlabel("Traffic Status")
    plt.ylabel("Average Waiting Time")
    plt.tight_layout()
    plt.show()


# ------------------------------------------
# 4. Inventory Analysis
# ------------------------------------------

if "Inventory_Level" in df.columns:

    print("\nInventory Statistics:")
    print(df["Inventory_Level"].describe())

    df["Inventory_Level"].plot(
        kind="hist",
        bins=20,
        title="Inventory Level Distribution"
    )

    plt.xlabel("Inventory Level")
    plt.tight_layout()
    plt.show()


# ------------------------------------------
# 5. Correlation Analysis
# ------------------------------------------

numeric_columns = df.select_dtypes(
    include="number"
).columns

if len(numeric_columns) > 1:

    correlation = df[numeric_columns].corr()

    print("\nCorrelation Matrix:")
    print(correlation)

    plt.figure(figsize=(10, 6))
    plt.imshow(correlation, aspect="auto")
    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.columns)),
        correlation.columns
    )

    plt.title("Logistics Variable Correlation")
    plt.tight_layout()
    plt.show()


print("\nAnalysis completed successfully.")
