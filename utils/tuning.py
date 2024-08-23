import os
import random
import re
from tqdm.notebook import tqdm
from collections import Counter
from datetime import datetime
import argparse

import numpy as np
import pandas as pd

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

parser = argparse.ArgumentParser()
parser.add_argument('--seed',type=int, default=110)
config = parser.parse_args([])

def catboost_objective(trial, X_tr, y_tr):
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
    
    for idx, (tr_idx, val_idx) in enumerate(stk.split(X_tr, y_tr)):
        X_tr_fold, X_val_fold = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        y_tr_fold, y_val_fold = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        
        cat_clf.fit(X_tr_fold, y_tr_fold, eval_set=[(X_val_fold, y_val_fold)], early_stopping_rounds=50, verbose=False)
        y_pred_fold = cat_clf.predict(X_val_fold)
        f1_scores[idx] = f1_score(y_val_fold, y_pred_fold)

    return np.mean(f1_scores)

def lgbm_objective(trial, X_tr, y_tr):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'num_leaves': trial.suggest_int('num_leaves', 10, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
    }
    
    lgbm_clf = LGBMClassifier(**params, random_state=config.seed)
    
    stk = StratifiedKFold(n_splits=5, random_state=config.seed, shuffle=True)
    f1_scores = np.empty(5)
    
    for idx, (tr_idx, val_idx) in enumerate(stk.split(X_tr, y_tr)):
        X_tr_fold, X_val_fold = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        y_tr_fold, y_val_fold = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]
        
        lgbm_clf.fit(X_tr_fold, y_tr_fold, early_stopping_rounds=50,)
        y_pred_fold = lgbm_clf.predict(X_val_fold)
        f1_scores[idx] = f1_score(y_val_fold, y_pred_fold)

    return np.mean(f1_scores)