import os
import random
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.utils import resample
from imblearn.under_sampling import (NearMiss,
                                     ClusterCentroids,
                                     AllKNN,
                                     OneSidedSelection,
                                     TomekLinks)
from imblearn.over_sampling import (RandomOverSampler,
                                    SMOTE,
                                    ADASYN,
                                    SMOTENC,
                                    SMOTEN,
                                    BorderlineSMOTE,
                                    KMeansSMOTE,
                                    SVMSMOTE)
from imblearn.combine import SMOTEENN
# https://imbalanced-learn.org/stable/references/index.html

def random_downsample(df, random_seed, sample_ratio=1.0):
    df_normal = df[df["target"] == 0] 
    df_abnormal = df[df["target"] == 1]
    
    downsampled = resample(
        df_normal,
        replace=False,
        n_samples=int(len(df_abnormal) * sample_ratio),
        random_state=random_seed
    )
    
    downsampled_df = pd.concat([df_abnormal, downsampled])
    
    return downsampled_df

def downsample(X, y, method, random_seed):
    # NearMiss
    if method == "nearmiss":
        # sampling_strategy="auto"
        nm = NearMiss(sampling_strategy=0.4)
        X_downsampled, y_downsampled = nm.fit_resample(X, y)
    # ClusterCentroids
    elif method == "cluster":
        cc = ClusterCentroids(random_state=random_seed)
        X_downsampled, y_downsampled = cc.fit_resample(X, y)
    # AllKNN
    elif method == "allknn":
        allknn = AllKNN()
        X_downsampled, y_downsampled = allknn.fit_resample(X, y)
    # OneSidedSelection
    elif method == "oneside":
        oss = OneSidedSelection(random_state=random_seed)
        X_downsampled, y_downsampled = oss.fit_resample(X, y)
    # Tomeklinks
    elif method == "tomek":
        tl = TomekLinks()
        X_downsampled, y_downsampled = tl.fit_resample(X, y)
    
    X_downsampled_df= pd.DataFrame(X_downsampled, columns=X.columns)
    y_downsampled_df = pd.Series(y_downsampled, name="target") 
    downsampled_df = pd.concat([X_downsampled_df, y_downsampled_df], axis=1)
    
    print('DOWN SAMPLING')
    print('=============')
    print('Original dataset shape %s' % Counter(y))
    print('Resampled dataset shape %s' % Counter(y_downsampled), end='\n')
    
    return downsampled_df


def upsample(X, y, cat_idx, method, random_seed):
    
    if method == "random":
        ros = RandomOverSampler(random_state=random_seed)
        X_upsampled, y_upsampled = ros.fit_resample(X, y)
        
    # SMOTE
    elif method == "smote":
        smote = SMOTE(random_state=random_seed)
        X_upsampled, y_upsampled = smote.fit_resample(X, y)
        
    # ADASYN
    elif method == "adasyn":
        adasyn = ADASYN(random_state=random_seed)
        X_upsampled, y_upsampled = adasyn.fit_resample(X, y)
        
    # SMOTE-NC
    elif method == "smotenc":
        smotenc = SMOTENC(random_state=random_seed, sampling_strategy="auto", categorical_features=cat_idx)
        X_upsampled, y_upsampled = smotenc.fit_resample(X, y)
        
    elif method == "smoten":
        smoten = SMOTEN(random_state=random_seed, sampling_strategy="auto", k_neighbors=5)
        X_upsampled, y_upsampled = smoten.fit_resample(X, y)
        
    elif method == "borderline":
        borderline_smote = BorderlineSMOTE(random_state=random_seed)
        X_upsampled, y_upsampled = borderline_smote.fit_resample(X, y)
        
    elif method == "kmeans":
        kmeans_smote = KMeansSMOTE(random_state=random_seed, sampling_strategy="auto", k_neighbors=5)
        X_upsampled, y_upsampled = kmeans_smote.fit_resample(X, y)
        
    elif method == "svm":
        svm_smote = SVMSMOTE(random_state=42)
        X_upsampled, y_upsampled = svm_smote.fit_resample(X, y)
        
    X_upsampled_df= pd.DataFrame(X_upsampled, columns=X.columns)
    y_upsampled_df = pd.Series(y_upsampled, name="target") 
    upsampled_df = pd.concat([X_upsampled_df, y_upsampled_df], axis=1)
    
    print('UP SAMPLNG')
    print('==========')
    print('Original dataset shape %s' % Counter(y))
    print('Resampled dataset shape %s' % Counter(y_upsampled), end='\n')
    
    return upsampled_df