import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px
import os

output_dir = "visuals_1"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("/Users/fatoumata-binta/Desktop/fbj_business_customer_project/Data/bank-_1_.csv")
print(df.head(5))

print(df.shape)
print(df.info())
print(df.head)

print (df.isnull().sum())

# Remove unnecessary column "."
df = df.drop(columns=["."])

df = df.drop(columns=["VALUE DATE"])

# Remove column CHQ.NO. (almost always empty)
df = df.drop(columns=["CHQ.NO."])

# Clean Account No (remove the ')
df["Account No"] = df["Account No"].str.replace("'", "", regex=False)

df["Account No"] = df["Account No"].astype(str).str.strip()  # remove spaces
df["Account No"] = df["Account No"].str.replace("’", "", regex=False)  # retire les apostrophes spéciales
df["Account No"] = df["Account No"].str.replace(" ", "", regex=False)  # retire les espaces internes
# ...existing code...

# Convert DATE into proper datetime format
df["DATE"] = pd.to_datetime(df["DATE"], format="%d-%b-%y", errors="coerce")

# Convert amounts to float
for col in ["WITHDRAWAL AMT", "DEPOSIT AMT", "BALANCE AMT"]:
    df[col] = df[col].str.replace(",", "", regex=False)  # remove commas
    df[col] = pd.to_numeric(df[col], errors="coerce")

    print(df.info())
print(df.head())

# Remove duplicates 
print(df.duplicated().sum())   # number of duplicated rows
df = df.drop_duplicates()

# Add an ID column starting at 1 and incrementing by 1
df["id"] = range(1, len(df) + 1)

# Check
print(df.head())

# Put ID as the first column
cols = ["id"] + [col for col in df.columns if col != "id"]
df = df[cols]

print(df.head())

print(df.isna().sum())   # missing values
print(df.describe())     # stats on numeric columns
print(df["Account No"].nunique())  # number of distinct accounts

# Save cleaned data into a CSV file
df.to_csv("bank_clean.csv", index=False)

# --- EDA ---

# Structure and data quality

# Dimension
print("Shape:", df.shape)

# Column types
print(df.dtypes)

# Missing values 
print(df.isnull().sum())

# Number of distinct accounts 
print("distinct accounts:", df["Account No"].nunique)


# Descriptive analysis 

print("Total number of transactions:", len(df))

# Financial totals
total_deposit = df["DEPOSIT AMT"].sum()
total_withdrawal = df["WITHDRAWAL AMT"].sum()
balance_amount = df["BALANCE AMT"].mean()

print("total deposit:", total_deposit)
print("total withdrawal:", total_withdrawal)
print("balance amount:", balance_amount)

# Top 10 accounts by transaction volume
print(df["Account No"].value_counts().head(10))


import os
import textwrap
import matplotlib.pyplot as plt
import pandas as pd

output_dir = "visuals_1"
os.makedirs(output_dir, exist_ok=True)

# take top 10 accounts
top_accounts = df["Account No"].value_counts().head(10)

# ---- 1) Diagnostic: print raw labels (use repr to see invisibles) ----
print("=== Raw top accounts index ===")
for i, lab in enumerate(top_accounts.index.astype(str)):
    print(i, repr(lab), "len=", len(lab))
print("=== End diagnostic ===\n")

# ---- 2) Clean labels ----
labels = pd.Series(top_accounts.index.astype(str)).str.strip().str.replace("'", "", regex=False)
print("Cleaned labels:", labels.tolist())

# If you want to use last 6 digits, check duplicates first
last6 = labels.str[-6:]
if last6.duplicated().any():
    print("Warning: using last 6 digits will create duplicate labels:")
    print(last6[last6.duplicated()])
else:
    print("Last6 labels OK (no duplicates).")

# helper to wrap a long label every `width` chars to avoid overlap
def wrap_label(s, width=8):
    s = str(s)
    if len(s) <= width:
        return s
    wrapped = textwrap.fill(s, width=width)
    return wrapped

# wrapped labels (break to multiple lines)
labels_wrapped = labels.apply(lambda s: wrap_label(s, width=8))

# ---- 3) Horizontal bar chart (best for long labels) ----
plt.figure(figsize=(10,6))
top_h = top_accounts.copy()
top_h.index = labels_wrapped.values  # use wrapped labels for readability
ax = top_h.plot(kind="barh", legend=False)
ax.invert_yaxis()  # put largest on top
plt.title("Top 10 Accounts by Transaction Count")
plt.xlabel("Number of transactions")
plt.ylabel("")  # account shown on y-axis
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_10_accounts_horizontal.png"))
plt.close()

# ---- 4) Vertical bar chart with rotated labels and adjusted bottom ----
plt.figure(figsize=(12,6))
top_v = top_accounts.copy()
top_v.index = labels.values  # use full cleaned labels (not wrapped)
ax = top_v.plot(kind="bar", legend=False)
plt.title("Top 10 Accounts by Transaction Count")
plt.xlabel("Account No")
plt.ylabel("Number of transactions")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.subplots_adjust(bottom=0.30)  # make space for long labels
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_10_accounts_vertical.png"))
plt.close()

# ---- 5) Option: truncated last 6 digits (only if no duplicates) ----
if not last6.duplicated().any():
    plt.figure(figsize=(12,6))
    top_l6 = top_accounts.copy()
    top_l6.index = last6.values
    top_l6.plot(kind="bar", legend=False)
    plt.title("Top 10 Accounts (last 6 digits)")
    plt.xlabel("Account (last 6 digits)")
    plt.ylabel("Number of transactions")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_10_accounts_last6.png"))
    plt.close()
else:
    print("Skipped last6 plot because truncation would create duplicate labels.")

print("Saved: horizontal, vertical and (maybe) last6 plots in", output_dir)


# Create a new column Net Amount
df['Net Amount'] = df['DEPOSIT AMT'].fillna(0) - df['WITHDRAWAL AMT'].fillna(0)

# Check total balance
total_net = df['Net Amount'].sum()
print("Net balance:", total_net)


# Create a "Month" column
df["Month"] = df["DATE"].dt.to_period("M")

# Number of transactions per month 
transactions_par_mois = df["Month"].value_counts().sort_index()
transactions_par_mois.plot(kind="bar", figsize=(14,5), title="Transactions per month")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "transactions_per_month.png")) # type: ignore
plt.close()


# Balance evolution (for one account)
for account in df["Account No"].unique()[:1]: # tracking only the 1st account 
    df_acc = df[df["Account No"] == account].sort_values("DATE")
    plt.figure(figsize=(14,5))
    plt.plot(df_acc["DATE"], df_acc["BALANCE AMT"])
    plt.title(f"Balance evolution - Account {account}")
    plt.xlabel("Date")
    plt.ylabel("Balance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"balance_account_{account}.png"))
    plt.close()


# Transaction analysis 

# Top labels
top_labels = df["TRANSACTION DETAILS"].value_counts().head(20)
print(top_labels)

top_labels.plot(kind="barh", figsize=(12,6), title="Top 20 transaction types")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_20_transactions.png"))
plt.close()


# Distribution of deposits and withdrawals 
plt.figure(figsize=(12,5))
plt.hist(df["DEPOSIT AMT"].dropna(), bins=50, alpha=0.6, label="Deposit")
plt.hist(df["WITHDRAWAL AMT"].dropna(), bins=50, alpha=0.6, label="Withdrawal")
plt.legend()
plt.title("Amount distributions")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "amount_distribution.png"))
plt.close()


# Anomaly detection 

# Create a column "Amount" taking the greater of withdrawal or deposit
df["Amount"] = df[["WITHDRAWAL AMT", "DEPOSIT AMT"]].max(axis=1)

# Detect the top 1% largest amounts
seuil = df["Amount"].quantile(0.99)
outliers = df[df["Amount"] > seuil]

print("Number of potential anomalies:", len(outliers))
print(outliers.head())
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "potential_anomalies.png"))
plt.close()

# Correlations 

import seaborn as sns

plt.figure(figsize=(6,4))
sns.heatmap(df[["WITHDRAWAL AMT", "DEPOSIT AMT", "BALANCE AMT"]].corr(), annot=True, cmap="coolwarm")
plt.title("Correlations between amounts and balance")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlations.png"))
plt.close()


# Transactions per day 
transactions_par_jour = df["DATE"].value_counts().sort_index()
transactions_par_jour.plot(kind="line", figsize=(14,5), title="Transactions per day")


# Top accounts by transaction volume
top_accounts = df["Account No"].value_counts().head(10)
top_accounts.plot(kind="bar", figsize=(12,6), title="Top 10 accounts per transactions")

plt.show()

import matplotlib
matplotlib.use("Agg")  # non-interactive mode

plt.figure(figsize=(10,6))
sns.boxplot(data=df[["DEPOSIT AMT", "WITHDRAWAL AMT"]], orient="h")
plt.title("Boxplot of Deposit and Withdrawal Amounts")
plt.xlabel("Amount")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplot_amounts.png"))
plt.close()


plt.figure(figsize=(12,5))
plt.hist(df["DEPOSIT AMT"].dropna(), bins=100, alpha=0.6, label="Deposit")
plt.hist(df["WITHDRAWAL AMT"].dropna(), bins=100, alpha=0.6, label="Withdrawal")
plt.xscale("log")  # échelle logarithmique
plt.legend()
plt.title("Distribution of Deposit and Withdrawal Amounts (Log Scale)")
plt.xlabel("Amount (log scale)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "distribution_amounts_log.png"))
plt.close()


top_accounts = df["Account No"].value_counts().head(10)

plt.figure(figsize=(10,6))
top_accounts.plot(kind="barh")

plt.title("Top 10 Accounts per Transactions")
plt.xlabel("Number of Transactions")
plt.ylabel("Account No")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_10_accounts_horizontal.png"))
plt.close()
