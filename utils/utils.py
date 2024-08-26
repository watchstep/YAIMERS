import os
import random
import numpy as np
import pandas as pd

def get_columns_group(df:pd.DateOffset):
    grouped_columns = {}
    for col in df.columns:
        key = col.split('_')[-1]
        if key not in grouped_columns:
            grouped_columns[key] = []
        grouped_columns[key].append(col)
    
    return grouped_columns

# utils.columns_to_txt(df, col_name="Fill1")
def columns_to_txt(df:pd.DataFrame, col_name:str, version=1):
    df_col = df.loc[:, df.columns.str.contains(f"_{col_name}")]
    
    with open(os.path.join("./columns", f"{col_name}_v{version}.txt"), "w") as f:
        for col in df_col.columns:
            f.write('"'+ col + '"' + "\n")
    
            
# plt.figure(figsize=(8, 5))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('Correlation Matrix');