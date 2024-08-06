import pandas as pd
from imblearn.under_sampling import OneSidedSelection
from imblearn.over_sampling import ADASYN

def oss_resample(features, target, RANDOM_STATE, step_ratio=0.02):
    """
    Perform One-Sided Selection (OSS) under-sampling.
    """
    oss = OneSidedSelection(random_state=RANDOM_STATE)
    X_oss, y_oss = oss.fit_resample(features, target)
    
    print('Performing OSS...')

    initial_count_major = (y_oss == 'Normal').sum()
    initial_count_minor = (y_oss == 'AbNormal').sum()

    target_count_major = int(initial_count_major * (1 - step_ratio))
    count_major = initial_count_major
    print('Target ratio: ', target_count_major/initial_count_minor)

    while count_major > target_count_major:
        X_oss, y_oss = oss.fit_resample(X_oss, y_oss)
        count_major = (y_oss == 'Normal').sum()
        count_minor = (y_oss == 'AbNormal').sum()
        print('count_major: ', count_major, 'count_minor: ', count_minor, 'Updated ratio:', count_major/count_minor)

    return X_oss, y_oss

def adasyn_resample(features, target, RANDOM_STATE, step_ratio=0.05):
    """
    Perform ADASYN over-sampling.
    """
    adasyn = ADASYN(random_state=RANDOM_STATE)

    X_adasyn, y_adasyn = adasyn.fit_resample(features, target)
    
    print('Performing ADASYN...')
    
    initial_count_major = (y_adasyn == 'Normal').sum()
    initial_count_minor = (y_adasyn == 'AbNormal').sum()

    target_count_minor = int(initial_count_minor * (1 + step_ratio))
    count_minor = initial_count_minor
    print('initial count minor: ', initial_count_minor)
    print('Target ratio: ', initial_count_major/target_count_minor)

    return X_adasyn, y_adasyn

def sampling(df, RANDOM_STATE, target_ratio=1.5):
    """
    Perform under-sampling and over-sampling to adjust the class distribution.
    """
    
    # Store original column names
    original_columns = df.columns.tolist()
    feature_columns = [col for col in original_columns if col != 'target']
    print(original_columns)
    # Separate the target column
    target = df['target']
    features = df.drop(columns=['target'])

    # Initial class distribution
    count_major = (target == 'Normal').sum()
    count_minor = (target == 'AbNormal').sum()
    current_ratio = count_major / count_minor
    print('Initial class distribution - Normal:', count_major, 'AbNormal:', count_minor)
    print('Initial ratio:', current_ratio)

    while current_ratio > target_ratio:
        # Perform One-Sided Selection (OSS) under-sampling
        X_oss, y_oss = oss_resample(features, target, RANDOM_STATE, step_ratio=0.02)
        df_oss = pd.DataFrame(X_oss)
        df_oss['target'] = y_oss

        # Perform ADASYN over-sampling
        X_adasyn, y_adasyn = adasyn_resample(df_oss.drop(columns=['target']), df_oss['target'], RANDOM_STATE, step_ratio=0.05)
        df_adasyn = pd.DataFrame(X_adasyn)
        df_adasyn['target'] = y_adasyn

        features = df_adasyn.drop(columns=['target'])
        target = df_adasyn['target']

        # Update class distribution
        count_major = (target == 'Normal').sum()
        count_minor = (target == 'AbNormal').sum()
        current_ratio = count_major / count_minor
        print('Updated class distribution - Normal:', count_major, 'AbNormal:', count_minor)
        print('Updated ratio:', current_ratio)

    # Rename columns to match original names
    df_adasyn.columns = original_columns
    print(df_adasyn.columns)
    # Sort by date if needed
    if 'Collect Date - Dam' in df_adasyn.columns:
        df_adasyn = df_adasyn.sort_values(by=["Collect Date - Dam"])

    # Print the final counts
    print(df_adasyn['target'].value_counts())

    return df_adasyn

# Example usage:
file_path = 'CLEANED_PROCESSED.csv'
RANDOM_STATE = 42

df_merged = pd.read_csv(file_path)
print('Data merging...')
sampled_df = sampling(df_merged, RANDOM_STATE, target_ratio=1.5)

processed_file_path = './SAMPLED_PROCESSED.csv'
sampled_df.to_csv(processed_file_path, index=False)

print(f"Sampled data saved to {processed_file_path}")