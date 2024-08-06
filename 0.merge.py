import os
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def read_excel_file(file_path: str, header: int = None) -> pd.DataFrame:
    csv_file = file_path.replace(".xlsx", ".csv")

    if not os.path.exists(csv_file):
        print("Converting excel to csv...")
        if header:
            df = pd.read_excel(file_path, header=header)
        else:
            df = pd.read_excel(file_path)

        df.to_csv(csv_file, index=False)
        print(f"  {file_path} -> {csv_file}")
        return df
    else:
        print(f"  Reading {csv_file}")
        return pd.read_csv(csv_file, low_memory=False)

ROOT_DIR = "data"
RANDOM_STATE = 110

# Read files
X_Dam = read_excel_file(os.path.join(ROOT_DIR, "Dam dispensing.xlsx"), header=1)
X_AutoClave = read_excel_file(os.path.join(ROOT_DIR, "Auto clave.xlsx"), header=1)
X_Fill1 = read_excel_file(os.path.join(ROOT_DIR, "Fill1 dispensing.xlsx"), header=1)
X_Fill2 = read_excel_file(os.path.join(ROOT_DIR, "Fill2 dispensing.xlsx"), header=1)
y = pd.read_csv(os.path.join(ROOT_DIR, "train_y.csv"))

# Function to rename columns
def rename_columns(df, prefix):
    new_columns = {}
    for col in df.columns:
        if col == 'Set ID':
            new_columns[col] = col
        else:
            new_columns[col] = f"{prefix}_{col}"
    return df.rename(columns=new_columns)

# Rename columns for each dataframe
X_Dam = rename_columns(X_Dam, 'Dam dispensing')
X_AutoClave = rename_columns(X_AutoClave, 'Auto clave')
X_Fill1 = rename_columns(X_Fill1, 'Fill1 dispensing')
X_Fill2 = rename_columns(X_Fill2, 'Fill2 dispensing')

# Merge X
X = pd.merge(X_Dam, X_AutoClave, on="Set ID")
X = pd.merge(X, X_Fill1, on="Set ID")
X = pd.merge(X, X_Fill2, on="Set ID")
X = X.drop(X[X.duplicated(subset="Set ID")].index).reset_index(drop=True)

# Merge X and y
df_merged = pd.merge(X, y, "inner", on="Set ID")

# Drop columns with more than half of the values missing
drop_cols = []
for column in df_merged.columns:
    if (df_merged[column].notnull().sum() // 2) < df_merged[column].isnull().sum():
        drop_cols.append(column)
df_merged = df_merged.drop(drop_cols, axis=1)

# Drop Lot ID
df_merged = df_merged.drop('Dam dispensing_LOT ID', axis=1)

print(df_merged.columns)

# Save the merged dataframe to CSV
output_path = os.path.join(ROOT_DIR, 'MERGED.csv')
df_merged.to_csv(output_path, index=False)
print(f"Merged data saved to {output_path}")