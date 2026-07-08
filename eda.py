import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
df = pd.read_csv(r"C:\Users\Janan\Downloads\TESLA PROJECT\combined.csv")  # Change path if needed

# -----------------------------
# Basic Information
# -----------------------------
print("=" * 50)
print("First 5 Rows")
print(df.head())

print("=" * 50)
print("Last 5 Rows")
print(df.tail())

print("=" * 50)
print("Dataset Shape")
print(df.shape)

print("=" * 50)
print("Column Names")
print(df.columns.tolist())

print("=" * 50)
print("Data Types")
print(df.dtypes)

print("=" * 50)
print("Dataset Info")
print(df.info())

print("=" * 50)
print("Summary Statistics")
print(df.describe(include="all"))

# -----------------------------
# Missing Values
# -----------------------------
print("=" * 50)
print("Missing Values")
print(df.isnull().sum())

plt.figure(figsize=(10,5))
msno.bar(df)
plt.show()

# -----------------------------
# Duplicate Rows
# -----------------------------
print("=" * 50)
print("Duplicate Rows :", df.duplicated().sum())

# -----------------------------
# Correlation Heatmap
# -----------------------------
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12,8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.show()

# -----------------------------
# Histograms
# -----------------------------
for col in numeric_df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(col)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Boxplots
# -----------------------------
for col in numeric_df.columns:
    plt.figure(figsize=(6,3))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot - {col}")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Pairplot
# -----------------------------
if len(numeric_df.columns) <= 8:
    sns.pairplot(numeric_df)
    plt.show()

# -----------------------------
# Countplots for Categorical Data
# -----------------------------
cat_cols = df.select_dtypes(include=["object"]).columns

for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(
        y=df[col],
        order=df[col].value_counts().index[:10]
    )
    plt.title(col)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Outlier Detection (IQR)
# -----------------------------
print("=" * 50)
print("Outlier Count")

for col in numeric_df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"{col} : {len(outliers)} outliers")

# -----------------------------
# Unique Values
# -----------------------------
print("=" * 50)

for col in df.columns:
    print(f"{col} : {df[col].nunique()} unique values")

# -----------------------------
# Save Clean Dataset
# -----------------------------
df.to_csv("cleaned_dataset.csv", index=False)

print("=" * 50)
print("EDA Completed Successfully")
