# scripts/preprocessing.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
pd.set_option('future.no_silent_downcasting', True)


def preprocess_parkinsons(df):
    df = df.copy()

    # Drop 'name' column
    df = df.drop(['name'], axis=1)

    # Final selected features
    features = [
        'PPE', 'MDVP:Fo(Hz)', 'spread1',
        'MDVP:Flo(Hz)', 'Jitter:DDP', 'MDVP:Fhi(Hz)', 'spread2'
    ]
    X = df[features]
    y = df['status']

    return X, y

def preprocess_kidney(df):
    df = df.copy()

    # Replace and fix classification labels
    df['classification'] = df['classification'].replace({'ckd': 1, 'notckd': 0, 'ckd\t': 1, 'notckd\t': 0})
    
    # Drop 'id' column if it exists
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Encode categorical features
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    # Final selected features
    features = ['hemo', 'sc', 'sg', 'pcv', 'al', 'dm']
    X = df[features]
    y = df['classification']

    return X, y

def preprocess_liver(df):
    df = df.copy()

    # Map 'Dataset' to 1 and 0
    df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})

    # Fill missing values
    df = df.fillna(df.median(numeric_only=True))

    # Encode Gender
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

    # Final selected features
    features = [
        'Alkaline_Phosphotase', 'Aspartate_Aminotransferase', 'Alamine_Aminotransferase',
        'Age', 'Total_Bilirubin', 'Total_Protiens', 'Albumin',
        'Direct_Bilirubin', 'Albumin_and_Globulin_Ratio'
    ]
    X = df[features]
    y = df['Dataset']

    return X, y
