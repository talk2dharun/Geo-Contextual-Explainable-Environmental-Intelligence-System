"""
GEEIS - Data Processing Module
Handles loading, cleaning, normalization, and feature engineering
for the water quality dataset.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


# Feature columns used by the model
FEATURE_COLUMNS = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

# Target column
TARGET_COLUMN = 'Potability'

# Multi-class labels
CLASS_LABELS = {0: 'Unsafe', 1: 'Moderate', 2: 'Safe'}


def load_dataset(filepath):
    """Load the water quality CSV dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    """
    Clean the dataset:
    - Handle missing values using group-based median imputation
    - Remove duplicates
    - Clip outliers using IQR method
    """
    df = df.copy()

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Fill missing values with group median (by Potability)
    for col in FEATURE_COLUMNS:
        if df[col].isnull().sum() > 0:
            df[col] = df.groupby(TARGET_COLUMN)[col].transform(
                lambda x: x.fillna(x.median())
            )

    # Clip outliers using IQR
    for col in FEATURE_COLUMNS:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def create_multiclass_target(df):
    """
    Create multi-class target labels:
    - Safe (2): Potability = 1 and good feature ranges
    - Moderate (1): Borderline values
    - Unsafe (0): Potability = 0
    
    Uses a heuristic based on WHO guidelines for pH, turbidity, etc.
    """
    df = df.copy()

    # Start with binary labels
    df['Quality_Class'] = df[TARGET_COLUMN].apply(lambda x: 2 if x == 1 else 0)

    # Refine: if Potability == 0 but features are borderline, mark as Moderate
    borderline_mask = (
        (df[TARGET_COLUMN] == 0) &
        (df['ph'].between(6.0, 9.0)) &
        (df['Turbidity'] <= 5.0) &
        (df['Conductivity'] <= 500)
    )
    df.loc[borderline_mask, 'Quality_Class'] = 1

    return df


def normalize_features(df, scaler=None, fit=True):
    """
    Normalize feature columns using StandardScaler.
    Returns the normalized DataFrame and the scaler object.
    """
    df = df.copy()

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])
    else:
        df[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])

    return df, scaler


def prepare_data(filepath, models_dir='models'):
    """
    Full data preparation pipeline:
    1. Load dataset
    2. Clean data
    3. Create multi-class target
    4. Normalize features
    5. Split into train/test sets
    6. Save scaler
    
    Returns: X_train, X_test, y_train, y_test, scaler, raw_df
    """
    # Load
    df = load_dataset(filepath)
    raw_df = df.copy()

    # Clean
    df = clean_data(df)

    # Create multi-class labels
    df = create_multiclass_target(df)

    # Normalize
    df_normalized, scaler = normalize_features(df, fit=True)

    # Split
    X = df_normalized[FEATURE_COLUMNS]
    y = df_normalized['Quality_Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save scaler
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)

    return X_train, X_test, y_train, y_test, scaler, raw_df


def augment_features_with_weather(features_dict, weather_data):
    """
    Augment prediction features with weather context.
    Adds temperature, humidity, and rainfall effects.
    """
    augmented = features_dict.copy()

    if weather_data:
        temp = weather_data.get('temperature', 25)
        humidity = weather_data.get('humidity', 50)
        rain = weather_data.get('rainfall', 0)

        # Temperature affects bacterial growth and chemical reactions
        temp_factor = 1.0 + (temp - 25) * 0.02
        augmented['Organic_carbon'] = augmented.get('Organic_carbon', 14.0) * temp_factor

        # High humidity can increase contamination pathways
        humidity_factor = 1.0 + (humidity - 50) * 0.005
        augmented['Turbidity'] = augmented.get('Turbidity', 4.0) * humidity_factor

        # Rainfall can dilute or increase contamination
        if rain > 10:
            augmented['Solids'] = augmented.get('Solids', 22000) * 1.15
            augmented['Turbidity'] = augmented.get('Turbidity', 4.0) * 1.2

    return augmented
