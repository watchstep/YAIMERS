import os
import random
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer

from utils import *

ROOT_DIR = "./data"
RANDOM_SEED = 110
np.random.seed(RANDOM_SEED)

def preprocessing_test(file_name:str, version=2):
    # read csv
    df = pd.read_csv(os.path.join(ROOT_DIR, f"{file_name}.csv"))
    
    # NaN
    nan_columns = df.columns[df.isnull().any()]
    df.drop(nan_columns, axis=1, inplace=True)
    
    # Duplicates
    CAT = ["Dam", "AutoClave", "Fill1", "Fill2"]
    
    wip_line = [f"Wip Line_{cat}" for cat in CAT]
    df.drop(wip_line, axis=1, inplace=True)
    
    process_desc = [f"Process Desc._{cat}" for cat in CAT]
    df.drop(process_desc, axis=1, inplace=True)
    
    for cat in CAT:
        col = f"Equipment_{cat}"
        if cat == "AutoClave": 
            df.drop(col, axis=1, inplace=True) 
        else:
            df[col] = df[col].str.split("#", expand=True)[1] # dtype: object
    
    df["Model.Suffix"] = df["Model.Suffix_Dam"]
    model_suffix = [f"Model.Suffix_{cat}" for cat in CAT]
    df.drop(model_suffix, axis=1, inplace=True)

    insp_seq_no = [f"Insp. Seq No._{cat}" for cat in CAT]
    df.drop(insp_seq_no, axis=1, inplace=True)
    
    insp_jude_code = [f"Insp Judge Code_{cat}" for cat in CAT]
    df.drop(insp_jude_code, axis=1, inplace=True)
    
    df["Workorder"] = df["Workorder_Dam"]
    work_order = [f"Workorder_{cat}" for cat in CAT]
    df.drop(work_order, axis=1, inplace=True)
    
    df["Workorder Category"] = df["Workorder"].str.split('-', expand=True)[0].str[:4]
    
    # AutoClave
    df.rename(columns={"1st Pressure 1st Pressure Unit Time_AutoClave": "1st Pressure Unit Time_AutoClave"}, inplace=True)
    
    df.drop(["1st Pressure Judge Value_AutoClave",
             "2nd Pressure Judge Value_AutoClave",
             "3rd Pressure Judge Value_AutoClave"], axis=1, inplace=True)
    
    pressure_unit_time = df[["1st Pressure Unit Time_AutoClave", 
                             "2nd Pressure Unit Time_AutoClave",
                             "3rd Pressure Unit Time_AutoClave",]]
    df["Mean Pressure Unit Time_AutoClave"] = pressure_unit_time.apply("mean", axis=1).astype('int64')
    

    pressure_collect_result = df[["1st Pressure Collect Result_AutoClave", 
                                  "2nd Pressure Collect Result_AutoClave",
                                  "3rd Pressure Collect Result_AutoClave",]]
    df["Mean Pressure Collect Result_AutoClave"] = pressure_collect_result.apply("mean", axis=1).round(3)
    
    # Dam
    df.drop(["CURE END POSITION Θ Collect Result_Dam", "CURE STANDBY POSITION Z Collect Result_Dam"], axis=1, inplace=True)
    
    df.drop(["CURE START POSITION Z Collect Result_Dam",
             "CURE STANDBY POSITION X Collect Result_Dam",
             "CURE STANDBY POSITION Θ Collect Result_Dam"], axis=1, inplace=True)
    
    kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["DISCHARGED TIME OF RESIN(Stage1) Collect Result Bins_Dam"] = kb.fit_transform(df["DISCHARGED TIME OF RESIN(Stage1) Collect Result_Dam"].values.reshape(-1,1))
    df["DISCHARGED TIME OF RESIN(Stage2) Collect Result Bins_Dam"] = kb.fit_transform(df["DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam"].values.reshape(-1,1))
    df["DISCHARGED TIME OF RESIN(Stage3) Collect Result Bins_Dam"] = kb.fit_transform(df["DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["Dispense Volume(Stage1) Collect Result Bins_Dam"] = kb.fit_transform(df["Dispense Volume(Stage1) Collect Result_Dam"].values.reshape(-1,1))
    df["Dispense Volume(Stage2) Collect Result Bins_Dam"] = kb.fit_transform(df["Dispense Volume(Stage2) Collect Result_Dam"].values.reshape(-1,1))
    df["Dispense Volume(Stage3) Collect Result Bins_Dam"] = kb.fit_transform(df["Dispense Volume(Stage3) Collect Result_Dam"].values.reshape(-1,1))
    
    ############
    kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result Bins_Dam"] = kb.fit_transform(df["HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Dam"].values.reshape(-1,1))
    df["HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result Bins_Dam"] = kb.fit_transform(df["HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Dam"].values.reshape(-1,1))
    
    df["HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result Bins_Dam"] = kb.fit_transform(df["HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Dam"].values.reshape(-1,1))
    df["HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result Bins_Dam"] = kb.fit_transform(df["HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Dam"].values.reshape(-1,1))
    df["HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result Bins_Dam"] = kb.fit_transform(df["HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result_Dam"].values.reshape(-1,1))
    
    df["HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result Bins_Dam"] = kb.fit_transform(df["HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Dam"].values.reshape(-1,1))
    df["HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result Bins_Dam"] = kb.fit_transform(df["HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Dam"].values.reshape(-1,1))
    df["HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result Bins_Dam"] = kb.fit_transform(df["HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Dam"].values.reshape(-1,1))
    
    ############
    kb = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans', random_state=RANDOM_SEED)
    
    df["HEAD Standby Position X Collect Result Bins_Dam"] = kb.fit_transform(df["HEAD Standby Position X Collect Result_Dam"].values.reshape(-1,1))
    
    ############
    kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["Machine Tact time Collect Result Bins_Dam"] = kb.fit_transform(df["Machine Tact time Collect Result_Dam"].values.reshape(-1,1))
    
    ############
    kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["PalletID Collect Result Bins_Dam"] = kb.fit_transform(df["PalletID Collect Result_Dam"].values.reshape(-1,1))
    
    #############
    kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["Production Qty Collect Result Bins_Dam"] = kb.fit_transform(df["Production Qty Collect Result_Dam"].values.reshape(-1,1))
    
    ##############
    kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='kmeans', random_state=RANDOM_SEED)
    
    df["Receip No Collect Result Bins_Dam"] = kb.fit_transform(df["Receip No Collect Result_Dam"].values.reshape(-1,1))
    
    circle2_3_dam =  ["Stage1 Circle3 Distance Speed Collect Result_Dam",
                      "Stage2 Circle3 Distance Speed Collect Result_Dam",
                      "Stage3 Circle3 Distance Speed Collect Result_Dam",
                      "Stage1 Circle4 Distance Speed Collect Result_Dam",
                      "Stage2 Circle4 Distance Speed Collect Result_Dam",
                      "Stage3 Circle4 Distance Speed Collect Result_Dam"]
    
    df.drop(circle2_3_dam, axis=1, inplace=True)
    
    stage1_line_dam = df[["Stage1 Line1 Distance Speed Collect Result_Dam",
                      "Stage1 Line2 Distance Speed Collect Result_Dam",
                      "Stage1 Line3 Distance Speed Collect Result_Dam",
                      "Stage1 Line4 Distance Speed Collect Result_Dam"]]
    
    stage2_line_dam = df[["Stage2 Line1 Distance Speed Collect Result_Dam",
                      "Stage2 Line2 Distance Speed Collect Result_Dam",
                      "Stage2 Line3 Distance Speed Collect Result_Dam",
                      "Stage2 Line4 Distance Speed Collect Result_Dam"]]
    
    stage3_line_dam = df[["Stage3 Line1 Distance Speed Collect Result_Dam",
                      "Stage3 Line2 Distance Speed Collect Result_Dam",
                      "Stage3 Line3 Distance Speed Collect Result_Dam",
                      "Stage3 Line4 Distance Speed Collect Result_Dam"]]
    
    df["Mean Stage1 Line Distance Speed Collect Result_Dam"] = stage1_line_dam.apply("mean", axis=1).astype('int64')
    df["Mean Stage2 Line Distance Speed Collect Result_Dam"] = stage2_line_dam.apply("mean", axis=1).astype('int64')
    df["Mean Stage3 Line Distance Speed Collect Result_Dam"] = stage3_line_dam.apply("mean", axis=1).astype('int64')
    
    # Fill1
    kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["DISCHARGED TIME OF RESIN(Stage1) Collect Result Bins_Fill1"] = kb.fit_transform(df["DISCHARGED TIME OF RESIN(Stage1) Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["DISCHARGED TIME OF RESIN(Stage2) Collect Result Bins_Fill1"] = kb.fit_transform(df["DISCHARGED TIME OF RESIN(Stage2) Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["DISCHARGED TIME OF RESIN(Stage3) Collect Result Bins_Fill1"] = kb.fit_transform(df["DISCHARGED TIME OF RESIN(Stage3) Collect Result_Fill1"].values.reshape(-1,1))

    kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["Dispense Volume(Stage1) Collect Result Bins_Fill1"] = kb.fit_transform(df["Dispense Volume(Stage1) Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["Dispense Volume(Stage2) Collect Result Bins_Fill1"] = kb.fit_transform(df["Dispense Volume(Stage2) Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["Dispense Volume(Stage3) Collect Result Bins_Fill1"] = kb.fit_transform(df["Dispense Volume(Stage3) Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result Bins_Fill1"] = kb.fit_transform(df["HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result Bins_Fill1"] = kb.fit_transform(df["HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result Bins_Fill1"] = kb.fit_transform(df["HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result Bins_Fill1"] = kb.fit_transform(df["HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result Bins_Fill1"] = kb.fit_transform(df["HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result Bins_Fill1"] = kb.fit_transform(df["HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result Bins_Fill1"] = kb.fit_transform(df["HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result Bins_Fill1"] = kb.fit_transform(df["HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans', random_state=RANDOM_SEED)
    
    df["HEAD Standby Position X Collect Result Bins_Fill1"] = kb.fit_transform(df["HEAD Standby Position X Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='kmeans', random_state=RANDOM_SEED)
    
    df["Machine Tact time Collect Result Bins_Fill1"] = kb.fit_transform(df["Machine Tact time Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["PalletID Collect Result Bins_Fill1"] = kb.fit_transform(df["PalletID Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["Production Qty Collect Result Bins_Fill1"] = kb.fit_transform(df["Production Qty Collect Result_Fill1"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans', random_state=RANDOM_SEED)
    
    df["Receip No Collect Result Bins_Fill1"] = kb.fit_transform(df["Receip No Collect Result_Fill1"].values.reshape(-1,1))
    
    # Fill2
    df.drop(["CURE STANDBY POSITION X Collect Result_Fill2",
             "CURE START POSITION Θ Collect Result_Fill2",
             "CURE STANDBY POSITION Θ Collect Result_Fill2",
             "CURE END POSITION Θ Collect Result_Fill2",], axis=1, inplace=True)
    
    df.drop(["CURE STANDBY POSITION Z Collect Result_Fill2"], axis=1, inplace=True)
    
    df.drop(["DISCHARGED TIME OF RESIN(Stage1) Collect Result_Fill2",
             "DISCHARGED TIME OF RESIN(Stage2) Collect Result_Fill2",
             "DISCHARGED TIME OF RESIN(Stage3) Collect Result_Fill2"], axis=1, inplace=True)
    df.drop(["DISCHARGED SPEED OF RESIN Collect Result_Fill2"], axis=1, inplace=True)
    
    dispense_fill2 = ["Dispense Volume(Stage1) Collect Result_Fill2",
                      "Dispense Volume(Stage2) Collect Result_Fill2",
                      "Dispense Volume(Stage3) Collect Result_Fill2"]
    df.drop(dispense_fill2, axis=1, inplace=True)
    
    df.drop(["HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Fill2"], axis=1, inplace=True)
    
    df.drop("Head Purge Position Y Collect Result_Fill2", axis=1, inplace=True)
    
    kb = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["Machine Tact time Collect Result Bins_Fill2"] = kb.fit_transform(df["Machine Tact time Collect Result_Fill2"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["PalletID Collect Result Bins_Fill2"] = kb.fit_transform(df["PalletID Collect Result_Fill2"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile', random_state=RANDOM_SEED)
    
    df["Production Qty Collect Result Bins_Fill2"] = kb.fit_transform(df["Production Qty Collect Result_Fill2"].values.reshape(-1,1))
    
    kb = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='kmeans', random_state=RANDOM_SEED)
    
    df["Receip No Collect Result Bins_Fill2"] = kb.fit_transform(df["Receip No Collect Result_Fill2"].values.reshape(-1,1))
    
    # drop 
    df.drop(["Workorder"], axis=1, inplace=True)
    
    # save csv
    df.to_csv(os.path.join(ROOT_DIR, f"{file_name}_v{version}.csv"), index=False)
    
preprocessing_test("test")