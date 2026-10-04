import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.xls")

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove rows with missing values
df = df.dropna()

print("Cleaned dataset shape:", df.shape)

# Set visualization style
sns.set_style("whitegrid")

# Graph 1: Churn distribution
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Churn")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")

plt.savefig("reports/churn_distribution.png")
plt.show()

# Graph 2: Tenure vs Churn
plt.figure(figsize=(7, 5))

sns.boxplot(data=df, x="Churn", y="tenure")

plt.title("Customer Tenure vs Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")

plt.savefig("reports/tenure_vs_churn.png")
plt.show()

# Graph 3: Monthly Charges vs Churn
plt.figure(figsize=(7, 5))

sns.boxplot(data=df, x="Churn", y="MonthlyCharges")

plt.title("Monthly Charges vs Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")

plt.savefig("reports/monthly_charges_vs_churn.png")
plt.show()

print("All graphs saved successfully!")