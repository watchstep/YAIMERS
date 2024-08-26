############################# IMPORT LIBRARY  #################################
import os
import random
import re
from tqdm.notebook import tqdm
from collections import Counter
from datetime import datetime
import argparse

import numpy as np
import pandas as pd

# https://contrib.scikit-learn.org/category_encoders/index.html
import category_encoders as ce
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif, chi2
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score, precision_score, recall_score

import optuna
import warnings
warnings.filterwarnings('ignore')

from utils import *


#######################   CONFIG  #######################
parser = argparse.ArgumentParser(description='Anomaly Detection')

parser.add_argument('--data_path', type=str, default='./data')
parser.add_argument('--seed',type=int, default=110)

parser.add_argument('--model', type=str, default='cat')

parser.add_argument('-en', '--encoder', type=str, default='js')
parser.add_argument('-s', '--scaler', type=str, default='ss')

downsample_options = {1:"nearmiss", 2:"cluster", 3:"allknn", 4:"oneside", 5:"tomek"}
parser.add_argument('-ds', '--downsampling', type=int, default=5)

upsample_options = {1: "random", 2:"smote", 3:"adasyn", 4:"smotenc", 5:"smoten", 6:"borderline", 7:"kmeans", 8:"svm"}
parser.add_argument('-us', '--upsampling', type=int, default=4)

parser.add_argument('--fs_mode', type=bool, default=False, help='feature selection T:auto F:manual')
parser.add_argument('--estimator', type=str, default='extra', help="using for feature selection")
parser.add_argument('--selector', type=str, default='sfm', help='auto feature selector')

parser.add_argument('--check_all', type=bool, default=False)
parser.add_argument('--tune_mode', type=bool, default=True, help='optuna tuning')

config = parser.parse_args([])

np.random.seed(config.seed)

#######################   LOAD DATA  #######################
df_tr = pd.read_csv(os.path.join(config.data_path, "train_v1.csv"))
df_te = pd.read_csv(os.path.join(config.data_path, "test_v1.csv"))
df_list = [df_tr, df_te]

# Workorder (test에 있는데, train에는 없는 경우가 있어 그냥 제외)
# 대신 Workorder Categeory 사용
for df in df_list:
    df.drop(["Workorder"], axis=1, inplace=True)


############################  FEATURE HANDLING  ###########################
## CATEGORICAL FEATURES
cat_features = ["Equipment_Dam",
                "Equipment_Fill1",
                "Equipment_Fill2",
                "Model.Suffix",
                "Workorder Category",
                "Chamber Temp. Judge Value_AutoClave"]

## BINNING FEATURES
bins_features = df_tr.columns[df_tr.columns.str.contains(r".*Bins.*")].tolist()
# Bins 열 만드는 데 사용된 열
from_bins_features = [re.sub(r'\s*Bins\s*', '', f).strip() for f in bins_features]

cat_features.extend(bins_features)

for df in df_list:
    df[cat_features] = df[cat_features].astype("category")

## NUMERICAL FEATURES
num_features = df_tr.select_dtypes(exclude=["category"]).columns.to_list()
num_features.remove("target")

## ALL FEATURES
all_features = num_features + cat_features

## TARGET ENCODING
df_tr["target"] = df_tr["target"].map({"Normal": 0, "AbNormal": 1})

## DATA SPLITTING 
X_tr, y_tr = df_tr.drop("target", axis=1), df_tr["target"]
X_te = df_te.drop("Set ID", axis=1)


#############################  FEATURE ENCODING/SCALING ###########################
## ENCODING
if config.encoder == "le":
    le = LabelEncoder()
    for cat_feature in cat_features:
        X_tr[cat_feature] = le.fit_transform(X_tr[cat_feature])
        X_te[cat_feature] = le.transform(X_te[cat_feature])
        
elif config.encoder == "js":
    js = ce.JamesSteinEncoder(cols=cat_features)
    
    X_tr = js.fit_transform(X_tr, y_tr)
    X_te = js.transform(X_te)
    
elif config.encoder == "woe":
    woe = ce.WOEEncoder(cols=cat_features)
    
    X_tr = woe.fit_transform(X_tr, y_tr)
    X_te = woe.transform(X_te)

## SCALING
if config.scaler == "mms":
    mms = MinMaxScaler()
    X_tr[num_features] = mms.fit_transform(X_tr[num_features])
    X_te[num_features] = mms.transform(X_te[num_features])
    
elif config.scaler == "ss":
    ss = StandardScaler()
    X_tr[num_features] = ss.fit_transform(X_tr[num_features])
    X_te[num_features] = ss.transform(X_te[num_features])
    
elif config.scaler == "qt":
    qt = QuantileTransformer(random_state=config.seed) # n_quantiles = 1000
    
    X_tr[num_features] = qt.fit_transform(X_tr[num_features])
    X_te[num_features] = qt.transform(X_te[num_features])


#################################  DOWN SAMPLING  ###############################
downsampled_df_tr = resampling.downsample(X_tr, y_tr, method=downsample_options[config.downsampling], random_seed=config.seed)


#################################  UP SAMPLING  ###############################
cat_idx = [downsampled_df_tr.columns.get_loc(col) for col in cat_features]
X_tr = downsampled_df_tr.drop("target", axis=1)
y_tr = downsampled_df_tr["target"]

upsampled_df_tr = resampling.upsample(X_tr, y_tr, cat_idx=cat_idx, method=upsample_options[config.upsampling], random_seed=config.seed)

## RESAMPLED DATA
# X_tr = downsampled_df_tr.drop("target", axis=1)
# y_tr = downsampled_df_tr["target"]

X_tr = upsampled_df_tr.drop("target", axis=1)
y_tr = upsampled_df_tr["target"]


################ MODEL ############### 
classifiers = {
    "cat": CatBoostClassifier(random_state=config.seed, auto_class_weights="Balanced"),
    "lgbm": LGBMClassifier(random_state=config.seed,),
    "xgb": XGBClassifier(random_state=config.seed, eval_metric='auc', objective="binary:logistic"),
    "ada": AdaBoostClassifier(random_state=config.seed),
    "rfc": RandomForestClassifier(random_state=config.seed, class_weight='balanced'),
    "lr": LogisticRegression(random_state=config.seed),
    "extra": ExtraTreesClassifier(random_state=config.seed)
}

###############################  FEATURE SELECTION  ############################
if config.fs_mode:
    estimator = classifiers[config.estimator]
    estimator.fit(X_tr, y_tr)
    
    selectors = {
        'rfe': RFE(estimator=estimator, n_features_to_select=50),
        'sfm': SelectFromModel(estimator=estimator, threshold="mean"),
        'kbest': SelectKBest(score_func=f_classif,),
    }
    
    selector = selectors[config.selector]
    
    X_tr_selec = selector.fit_transform(X_tr, y_tr)
    X_te_selec = selector.transform(X_te)
    
else:
    # 기존 열 대신 Bins 열 사용
    selected_features = [feature for feature in all_features if feature not in from_bins_features]
    
    X_tr_selec = X_tr[selected_features]
    X_te_selec = X_te[selected_features]
    
print("FEATRUE SELECTION")
print("Before ", X_tr.shape)
print("After ", X_tr_selec.shape, end='\n')

###############################  EVALUATION  ############################
stk = StratifiedKFold(n_splits=5, random_state=config.seed, shuffle=True)
rstk = RepeatedStratifiedKFold(n_splits=5, random_state=config.seed)

if config.check_all:
    classifiers_lst = list(classifiers.values())
    
    score_dic = {}
    for clf_name, clf in classifiers.items():
        scores = cross_val_score(clf, X_tr_selec, y_tr, scoring="f1", cv=stk)
        score_dic[clf_name] = scores.mean()
    
    print("MODEL CHECK")
    print(score_dic, end='\n')
    
else:
    scores = cross_val_score(classifiers["cat"], X_tr_selec, y_tr, scoring="f1", cv=stk)
    
    print("MODEL CHECK")
    print('F1', scores.mean(), end='\n')

def catboost_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.0001, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.5, 1.0),
        'od_type': trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
        'od_wait': trial.suggest_int("od_wait", 10, 50),
    }

    cat_clf = CatBoostClassifier(**params, random_state=config.seed, auto_class_weights="Balanced",) # eval_metric="TotalF1"
    
    stk = StratifiedKFold(n_splits=5, random_state=config.seed, shuffle=True)
    f1_scores = np.empty(5)
    
    for idx, (tr_idx, val_idx) in enumerate(stk.split(X_tr_selec, y_tr)):
        X_tr_fold, X_val_fold = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        y_tr_fold, y_val_fold = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        
        cat_clf.fit(X_tr_fold, y_tr_fold, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=50, verbose=False)
        y_pred_fold = cat_clf.predict(X_val_fold)
        f1_scores[idx] = f1_score(y_val_fold, y_pred_fold)

    return np.mean(f1_scores) 
    
if config.tune_mode and config.model == "cat":
    
    cat_study = optuna.create_study(direction='maximize')
    cat_study.optimize(catboost_objective, n_trials=15)
    
    cat_best_params = cat_study.best_params
    cat_best_score = cat_study.best_value
    
    print("CatBoost Best Hyperparams: ", cat_best_params)
    print("CatBoost Best F1 Score: ", cat_best_score, end='\n')
    
    final_clf = CatBoostClassifier(**cat_best_params, random_state=config.seed, auto_class_weights="Balanced",)
    
elif config.tune_mode and config.model == "lgbm":
    
    lgbm_study = optuna.create_study(direction='maximize')
    lgbm_study.optimize(tuning.lgbm_objective(X_tr_selec, y_tr), n_trials=15)
    
    lgbm_best_params = lgbm_study.best_params
    lgbm_best_score= lgbm_study.best_value
    
    print("LGBM Best Hyperparams: ",lgbm_best_params)
    print("LGBM Best F1 Score: ", lgbm_best_score, end='\n')
    
    final_clf = LGBMClassifier(**lgbm_best_params, random_state=config.seed,)
    
else:
    final_clf = classifiers[config.model]
    
################################################################
#####################     SUBMISSION   #########################
################################################################
final_clf.fit(X_tr_selec, y_tr, ) # use_best_model=True
final_preds = final_clf.predict(X_te_selec)

df_sub = pd.read_csv(os.path.join(config.data_path, "submission.csv"))
df_sub["target"] = final_preds
df_sub["target"] = df_sub["target"].map({0 : "Normal", 1 : "AbNormal"})

print('=============================')
print(df_sub["target"].value_counts())

curr_date = datetime.now().strftime("%m-%d_%H-%M-%S")
# df_sub.to_csv(os.path.join(config.data_path, f"submission_{curr_date}.csv"), index=False)
df_sub.to_csv(os.path.join(config.data_path, f"le_mms_tomek_smoten_cat.csv"))


# {'n_estimators': 275, 'learning_rate': 0.12690114610168876, 'max_depth': 9, 'subsample': 0.6224073667490215, 'colsample_bylevel': 0.5427716173358497, 'od_type': 'Iter', 'od_wait': 19}.