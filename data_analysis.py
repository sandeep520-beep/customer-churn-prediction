import pandas as pd

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.xls")

# Convert TotalCharges into numeric values
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Check missing values after conversion
print("\nMISSING VALUES AFTER CONVERSION")
print(df.isnull().sum())

# Count missing TotalCharges
print("\nMISSING TOTAL CHARGES")
print(df["TotalCharges"].isnull().sum())

# Display data types
print("\nUPDATED DATA TYPES")
print(df.dtypes)

# Display statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())