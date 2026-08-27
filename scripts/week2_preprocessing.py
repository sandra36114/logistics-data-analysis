import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# 1. Load the logistics dataset
# --------------------------------------------------

file_path = "data/smart_logistics_dataset.csv"

df = pd.read_csv(file_path)

print("Original Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# --------------------------------------------------
# 2. Initial Data Inspection
# --------------------------------------------------

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# --------------------------------------------------
# 3. Remove Duplicate Records
# --------------------------------------------------

df = df.drop_duplicates().reset_index(drop=True)

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)

# --------------------------------------------------
# 4. Handle Missing Values
# --------------------------------------------------

# Fill numerical missing values with median
numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill categorical missing values with "Unknown"
categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    df[column] = df[column].fillna("Unknown")

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# --------------------------------------------------
# 5. Standardize Text/Categorical Values
# --------------------------------------------------

for column in categorical_columns:
    df[column] = df[column].astype(str).str.strip()

# --------------------------------------------------
# 6. Detect Outliers Using IQR
# --------------------------------------------------

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(
        f"\n{column} - Potential Outliers:",
        len(outliers)
    )

# --------------------------------------------------
# 7. Normalize Numerical Data
# --------------------------------------------------

scaler = MinMaxScaler()

df_normalized = df.copy()

df_normalized[numeric_columns] = scaler.fit_transform(
    df_normalized[numeric_columns]
)

print("\nNormalized Data:")
print(df_normalized.head())

# --------------------------------------------------
# 8. Final Data Quality Check
# --------------------------------------------------

print("\nFinal Dataset Shape:", df.shape)

print("\nFinal Missing Values:")
print(df.isnull().sum())

print("\nFinal Duplicate Count:")
print(df.duplicated().sum())

# --------------------------------------------------
# 9. Save Cleaned Dataset
# --------------------------------------------------

output_file = "data/logistics_cleaned.csv"

df.to_csv(output_file, index=False)

print("\nCleaned dataset saved successfully to:")
print(output_file)