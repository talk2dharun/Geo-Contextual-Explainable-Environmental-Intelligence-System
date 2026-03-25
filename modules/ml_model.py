"""
GEEIS - Machine Learning Model Module
Handles training, evaluation, and prediction using XGBoost classifier
with Explainable AI (SHAP) capabilities.
"""

import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import shap
import joblib

from .data_processing import FEATURE_COLUMNS, CLASS_LABELS


def train_model(X_train, y_train, models_dir='models'):
    """
    Train an XGBoost classifier for water quality prediction.
    Returns: trained model
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        objective='multi:softprob',
        num_class=len(CLASS_LABELS)
    )

    model.fit(X_train, y_train)

    # Save model
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'xgboost_model.joblib')
    joblib.dump(model, model_path)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    Returns: dict with accuracy, precision, recall, f1, 
             classification_report text, and confusion matrix
    """
    y_pred = model.predict(X_test)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'classification_report': classification_report(
            y_test, y_pred,
            target_names=list(CLASS_LABELS.values()),
            zero_division=0
        ),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'predictions': y_pred
    }

    return results


def predict_quality(model, scaler, features_dict):
    """
    Make a single prediction from a features dictionary.
    Returns: predicted class label, class probabilities, raw prediction
    """
    # Build feature array in correct order
    feature_values = []
    for col in FEATURE_COLUMNS:
        feature_values.append(features_dict.get(col, 0.0))

    feature_array = np.array([feature_values])

    # Scale features
    feature_scaled = scaler.transform(feature_array)

    # Predict
    prediction = model.predict(feature_scaled)[0]
    probabilities = model.predict_proba(feature_scaled)[0]

    label = CLASS_LABELS.get(prediction, 'Unknown')

    return label, probabilities, prediction


def get_shap_explanation(model, scaler, features_dict):
    """
    Generate SHAP explanation for a single prediction.
    Returns: shap_values, feature_names, feature_values
    """
    # Build feature array
    feature_values = []
    for col in FEATURE_COLUMNS:
        feature_values.append(features_dict.get(col, 0.0))

    feature_array = np.array([feature_values])
    feature_scaled = scaler.transform(feature_array)

    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_scaled)

    return shap_values, FEATURE_COLUMNS, feature_values, explainer.expected_value


def get_feature_importance(model):
    """
    Get global feature importance from the trained model.
    Returns: DataFrame with feature names and importance scores
    """
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

    return importance_df


def load_trained_model(models_dir='models'):
    """
    Load a previously trained model and scaler.
    Returns: model, scaler or None, None if not found
    """
    model_path = os.path.join(models_dir, 'xgboost_model.joblib')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler

    return None, None
