import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------

file_path = "data/smart_logistics_dataset.csv"

df = pd.read_csv(file_path)

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# --------------------------------------------------
# 2. Data Preparation
# --------------------------------------------------

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    errors="coerce"
)

# Create useful time features
df["Month"] = df["Timestamp"].dt.month
df["Hour"] = df["Timestamp"].dt.hour
df["Date"] = df["Timestamp"].dt.date

# --------------------------------------------------
# 3. Basic Exploratory Data Analysis
# --------------------------------------------------

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATES ==========")
print(df.duplicated().sum())

# --------------------------------------------------
# 4. Central Tendency
# --------------------------------------------------

numeric_columns = [
    "Inventory_Level",
    "Temperature",
    "Humidity",
    "Waiting_Time",
    "User_Transaction_Amount",
    "User_Purchase_Frequency",
    "Asset_Utilization",
    "Demand_Forecast",
    "Logistics_Delay"
]

print("\n========== CENTRAL TENDENCY ==========")

for column in numeric_columns:
    print(f"\n{column}")
    print("Mean:", df[column].mean())
    print("Median:", df[column].median())
    print("Standard Deviation:", df[column].std())
    print("Minimum:", df[column].min())
    print("Maximum:", df[column].max())

# --------------------------------------------------
# 5. Shipment Status Analysis
# --------------------------------------------------

print("\n========== SHIPMENT STATUS ==========")
print(df["Shipment_Status"].value_counts())

# --------------------------------------------------
# 6. Traffic Status Analysis
# --------------------------------------------------

print("\n========== TRAFFIC STATUS ==========")
print(df["Traffic_Status"].value_counts())

# --------------------------------------------------
# 7. Delay Reason Analysis
# --------------------------------------------------

print("\n========== DELAY REASONS ==========")
print(df["Logistics_Delay_Reason"].value_counts())

# --------------------------------------------------
# 8. Correlation Analysis
# --------------------------------------------------

correlation = df[numeric_columns].corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation)

# --------------------------------------------------
# Create Visualization Folder
# --------------------------------------------------

import os

os.makedirs("visualizations", exist_ok=True)

# --------------------------------------------------
# Visualization 1: Logistics Delay Distribution
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    df["Logistics_Delay"].dropna(),
    bins=20
)

plt.title("Distribution of Logistics Delay")
plt.xlabel("Logistics Delay")
plt.ylabel("Number of Records")
plt.tight_layout()

plt.savefig(
    "visualizations/delivery_delay_distribution.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# Visualization 2: Shipping Status
# --------------------------------------------------

plt.figure(figsize=(8, 5))

df["Shipment_Status"].value_counts().plot(
    kind="bar"
)

plt.title("Shipment Status Distribution")
plt.xlabel("Shipment Status")
plt.ylabel("Number of Shipments")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "visualizations/shipment_status.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# Visualization 3: Traffic vs Waiting Time
# --------------------------------------------------

traffic_waiting = (
    df.groupby("Traffic_Status")["Waiting_Time"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

traffic_waiting.plot(kind="bar")

plt.title("Average Waiting Time by Traffic Status")
plt.xlabel("Traffic Status")
plt.ylabel("Average Waiting Time")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "visualizations/traffic_vs_waiting_time.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# Visualization 4: Inventory vs Demand Forecast
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Inventory_Level"],
    df["Demand_Forecast"],
    alpha=0.6
)

plt.title("Inventory Level vs Demand Forecast")
plt.xlabel("Inventory Level")
plt.ylabel("Demand Forecast")

plt.tight_layout()

plt.savefig(
    "visualizations/inventory_vs_demand.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# Visualization 5: Correlation Heatmap
# --------------------------------------------------

plt.figure(figsize=(11, 8))

plt.imshow(
    correlation,
    interpolation="nearest",
    aspect="auto"
)

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

plt.title("Correlation Matrix of Logistics Variables")

plt.tight_layout()

plt.savefig(
    "visualizations/correlation_heatmap.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# Visualization 6: Asset Utilization
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.hist(
    df["Asset_Utilization"].dropna(),
    bins=20
)

plt.title("Distribution of Asset Utilization")
plt.xlabel("Asset Utilization")
plt.ylabel("Number of Records")

plt.tight_layout()

plt.savefig(
    "visualizations/asset_utilization.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 9. Additional Analytical Results
# --------------------------------------------------

print("\n========== TRAFFIC AND DELAY ANALYSIS ==========")

traffic_delay = (
    df.groupby("Traffic_Status")["Logistics_Delay"]
    .mean()
    .sort_values(ascending=False)
)

print(traffic_delay)

print("\n========== ASSET UTILIZATION AND DELAY ==========")

utilization_delay = df[
    ["Asset_Utilization", "Logistics_Delay"]
].corr()

print(utilization_delay)

print("\n========== WAITING TIME AND DELAY ==========")

waiting_delay = df[
    ["Waiting_Time", "Logistics_Delay"]
].corr()

print(waiting_delay)

print("\n========== ANALYSIS COMPLETED ==========")

print("Visualizations saved in:")
print("visualizations/")

# --------------------------------------------------
# Visualization 7: Traffic Status vs Delay Rate
# --------------------------------------------------

traffic_delay_rate = (
    df.groupby("Traffic_Status")["Logistics_Delay"]
    .mean()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8, 5))

(traffic_delay_rate * 100).plot(kind="bar")

plt.title("Logistics Delay Rate by Traffic Status")
plt.xlabel("Traffic Status")
plt.ylabel("Delay Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "visualizations/traffic_vs_delay_rate.png",
    dpi=300
)

plt.close()

print("\nTraffic Delay Rate (%):")
print(traffic_delay_rate * 100)