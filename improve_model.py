"""
GEEIS Model Improvement Pipeline - V4 (Windows-Optimized)
No n_jobs=-1 (causes subprocess issues on Windows).
Faster hyperparameter search. Sequential processing.
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, ExtraTreesClassifier
)
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib

print("=" * 70)
print("GEEIS MODEL IMPROVEMENT PIPELINE V4")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# PHASE 1: DATA CLEANING
# ══════════════════════════════════════════════════════════════════════
df = pd.read_csv('data/water_potability.csv')
print(f"\nRaw shape: {df.shape}")

feature_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

df.loc[df['Solids'] < 0, 'Solids'] = np.nan

imputer = KNNImputer(n_neighbors=7, weights='distance')
df[feature_cols] = imputer.fit_transform(df[feature_cols])

for col in feature_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 2*IQR, upper=Q3 + 2*IQR)

print(f"Clean shape: {df.shape}, Missing: {df.isnull().sum().sum()}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 2: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════
print("\n--- Feature Engineering ---")

# Ratios
df['ph_hard_r'] = df['ph'] / (df['Hardness'] + 1)
df['tds_cond_r'] = df['Solids'] / (df['Conductivity'] + 1)
df['chlor_org_r'] = df['Chloramines'] / (df['Organic_carbon'] + 1)
df['sulf_hard_r'] = df['Sulfate'] / (df['Hardness'] + 1)
df['turb_org_r'] = df['Turbidity'] / (df['Organic_carbon'] + 1)
df['thm_org_r'] = df['Trihalomethanes'] / (df['Organic_carbon'] + 1)
df['ph_turb_r'] = df['ph'] / (df['Turbidity'] + 1)
df['chlor_thm_r'] = df['Chloramines'] / (df['Trihalomethanes'] + 1)
df['sulf_cond_r'] = df['Sulfate'] / (df['Conductivity'] + 1)

# Interactions
df['ph_turb'] = df['ph'] * df['Turbidity']
df['hard_sulf'] = df['Hardness'] * df['Sulfate']
df['chlor_thm'] = df['Chloramines'] * df['Trihalomethanes']
df['ph_cond'] = df['ph'] * df['Conductivity']
df['org_turb'] = df['Organic_carbon'] * df['Turbidity']
df['ph_chlor'] = df['ph'] * df['Chloramines']
df['hard_cond'] = df['Hardness'] * df['Conductivity']
df['thm_turb'] = df['Trihalomethanes'] * df['Turbidity']

# WHO safety indicators
df['ph_dev'] = abs(df['ph'] - 7.0)
df['ph_optimal'] = 1.0 / (1.0 + abs(df['ph'] - 7.0))
df['ph_safe'] = ((df['ph'] >= 6.5) & (df['ph'] <= 8.5)).astype(float)
df['turb_safe'] = (df['Turbidity'] <= 5.0).astype(float)
df['chlor_safe'] = (df['Chloramines'] <= 4.0).astype(float)
df['cond_safe'] = (df['Conductivity'] <= 400).astype(float)
df['thm_safe'] = (df['Trihalomethanes'] <= 80).astype(float)
df['org_safe'] = (df['Organic_carbon'] <= 15).astype(float)
df['hard_safe'] = (df['Hardness'] <= 200).astype(float)
df['sulf_safe'] = (df['Sulfate'] <= 300).astype(float)
df['safety_score'] = (df['ph_safe'] + df['turb_safe'] + df['chlor_safe'] +
                      df['cond_safe'] + df['thm_safe'] + df['org_safe'] +
                      df['hard_safe'] + df['sulf_safe'])

# Statistics
df['feat_mean'] = df[feature_cols].mean(axis=1)
df['feat_std'] = df[feature_cols].std(axis=1)

# Squared
for col in feature_cols:
    df[f'{col}_sq'] = df[col] ** 2

# Log
for col in feature_cols:
    if (df[col] > 0).all():
        df[f'{col}_log'] = np.log1p(df[col])

# Bins
df['ph_bin'] = pd.cut(df['ph'], bins=[0,6.5,7.0,7.5,8.5,14], labels=[0,1,2,3,4]).astype(float)
df['turb_bin'] = pd.cut(df['Turbidity'], bins=[0,2,3,4,5,10], labels=[0,1,2,3,4]).astype(float)
df['chlor_bin'] = pd.cut(df['Chloramines'], bins=[0,4,6,8,15], labels=[0,1,2,3]).astype(float)
df['thm_bin'] = pd.cut(df['Trihalomethanes'], bins=[0,40,60,80,200], labels=[0,1,2,3]).astype(float)
df = df.fillna(df.median())

all_feats = [c for c in df.columns if c != 'Potability']
print(f"Total features: {len(all_feats)}")

X = df[all_feats].values
y = df['Potability'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# ══════════════════════════════════════════════════════════════════════
# PHASE 3: PROPER CV EVALUATION (SMOTE inside each fold)
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 3: MODEL EVALUATION (5-Fold CV, SMOTE inside folds)")
print("=" * 70)

def eval_cv(name, model_cls, params, X, y, smote=True):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_a, te_a, te_f, te_p, te_r = [], [], [], [], []
    for tr_i, te_i in skf.split(X, y):
        Xtr, Xte, ytr, yte = X[tr_i], X[te_i], y[tr_i], y[te_i]
        if smote:
            Xtr, ytr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
        m = model_cls(**params)
        m.fit(Xtr, ytr)
        tr_a.append(accuracy_score(ytr, m.predict(Xtr)))
        te_a.append(accuracy_score(yte, m.predict(Xte)))
        te_f.append(f1_score(yte, m.predict(Xte), average='weighted'))
        te_p.append(precision_score(yte, m.predict(Xte), average='weighted', zero_division=0))
        te_r.append(recall_score(yte, m.predict(Xte), average='weighted', zero_division=0))
    r = dict(name=name, train=np.mean(tr_a), test=np.mean(te_a),
             f1=np.mean(te_f), prec=np.mean(te_p), rec=np.mean(te_r),
             gap=np.mean(tr_a)-np.mean(te_a))
    print(f"  {name:30s} Train={r['train']:.4f} Test={r['test']:.4f} F1={r['f1']:.4f} Prec={r['prec']:.4f} Rec={r['rec']:.4f} Gap={r['gap']:.4f}")
    return r

results = []

# --- Round 1: Baseline models ---
print("\n[Round 1] Baseline Models:")

results.append(eval_cv("XGB-Baseline", XGBClassifier, dict(
    n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
    min_child_weight=3, random_state=42, use_label_encoder=False, eval_metric='logloss',
    n_jobs=1
), X, y))

results.append(eval_cv("RF-Baseline", RandomForestClassifier, dict(
    n_estimators=500, max_depth=10, min_samples_split=5, min_samples_leaf=2,
    max_features='sqrt', random_state=42, n_jobs=1
), X, y))

results.append(eval_cv("LGBM-Baseline", LGBMClassifier, dict(
    n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
    num_leaves=31, random_state=42, verbose=-1, n_jobs=1
), X, y))

results.append(eval_cv("GB-Baseline", GradientBoostingClassifier, dict(
    n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
    min_samples_split=5, min_samples_leaf=2, random_state=42
), X, y))

results.append(eval_cv("ET-Baseline", ExtraTreesClassifier, dict(
    n_estimators=500, max_depth=12, min_samples_split=5, min_samples_leaf=2,
    max_features='sqrt', random_state=42, n_jobs=1
), X, y))

# --- Round 2: Tuned models (manual grid) ---
print("\n[Round 2] Tuned Models:")

# XGBoost variants
for i, params in enumerate([
    dict(n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.7,
         colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0, gamma=0.2,
         min_child_weight=5, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=1),
    dict(n_estimators=700, max_depth=7, learning_rate=0.01, subsample=0.8,
         colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
         min_child_weight=3, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=1),
    dict(n_estimators=500, max_depth=8, learning_rate=0.05, subsample=0.9,
         colsample_bytree=0.9, reg_alpha=0.01, reg_lambda=0.5, gamma=0.05,
         min_child_weight=1, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=1),
    dict(n_estimators=1000, max_depth=6, learning_rate=0.01, subsample=0.7,
         colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0, gamma=0.3,
         min_child_weight=7, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=1),
]):
    results.append(eval_cv(f"XGB-Tuned-{i+1}", XGBClassifier, params, X, y))

# LGBM variants
for i, params in enumerate([
    dict(n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.7,
         colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0, min_child_weight=5,
         num_leaves=31, random_state=42, verbose=-1, n_jobs=1),
    dict(n_estimators=700, max_depth=8, learning_rate=0.01, subsample=0.8,
         colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3,
         num_leaves=50, random_state=42, verbose=-1, n_jobs=1),
    dict(n_estimators=500, max_depth=-1, learning_rate=0.05, subsample=0.9,
         colsample_bytree=0.9, reg_alpha=0.01, reg_lambda=0.5, min_child_weight=1,
         num_leaves=63, random_state=42, verbose=-1, n_jobs=1),
]):
    results.append(eval_cv(f"LGBM-Tuned-{i+1}", LGBMClassifier, params, X, y))

# RF variants
for i, params in enumerate([
    dict(n_estimators=700, max_depth=15, min_samples_split=3, min_samples_leaf=1,
         max_features='sqrt', random_state=42, n_jobs=1),
    dict(n_estimators=1000, max_depth=20, min_samples_split=2, min_samples_leaf=1,
         max_features='log2', random_state=42, n_jobs=1),
    dict(n_estimators=500, max_depth=None, min_samples_split=5, min_samples_leaf=2,
         max_features=0.5, random_state=42, n_jobs=1),
]):
    results.append(eval_cv(f"RF-Tuned-{i+1}", RandomForestClassifier, params, X, y))

# --- Round 3: Without SMOTE (class weight) ---
print("\n[Round 3] Class Weight (no SMOTE):")

results.append(eval_cv("XGB-ClassWt", XGBClassifier, dict(
    n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.7,
    colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0, gamma=0.2,
    min_child_weight=5, scale_pos_weight=1998/1278,
    random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=1
), X, y, smote=False))

results.append(eval_cv("RF-Balanced", RandomForestClassifier, dict(
    n_estimators=700, max_depth=15, min_samples_split=3, min_samples_leaf=1,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=1
), X, y, smote=False))

# ══════════════════════════════════════════════════════════════════════
# PHASE 4: ENSEMBLE
# ══════════════════════════════════════════════════════════════════════
print("\n[Round 4] Ensemble:")

def eval_ensemble_cv(X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_a, te_a, te_f, te_p, te_r = [], [], [], [], []
    for tr_i, te_i in skf.split(X, y):
        Xtr, Xte, ytr, yte = X[tr_i], X[te_i], y[tr_i], y[te_i]
        Xtr, ytr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
        vc = VotingClassifier(estimators=[
            ('xgb', XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.02,
                                   subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5,
                                   reg_lambda=2.0, gamma=0.2, min_child_weight=5,
                                   random_state=42, use_label_encoder=False,
                                   eval_metric='logloss', n_jobs=1)),
            ('lgbm', LGBMClassifier(n_estimators=500, max_depth=6, learning_rate=0.02,
                                     subsample=0.7, colsample_bytree=0.7, reg_alpha=0.5,
                                     reg_lambda=2.0, min_child_weight=5, num_leaves=31,
                                     random_state=42, verbose=-1, n_jobs=1)),
            ('rf', RandomForestClassifier(n_estimators=700, max_depth=15,
                                          min_samples_split=3, min_samples_leaf=1,
                                          max_features='sqrt', random_state=42, n_jobs=1)),
            ('et', ExtraTreesClassifier(n_estimators=500, max_depth=12,
                                         random_state=42, n_jobs=1)),
        ], voting='soft')
        vc.fit(Xtr, ytr)
        tr_a.append(accuracy_score(ytr, vc.predict(Xtr)))
        te_a.append(accuracy_score(yte, vc.predict(Xte)))
        te_f.append(f1_score(yte, vc.predict(Xte), average='weighted'))
        te_p.append(precision_score(yte, vc.predict(Xte), average='weighted', zero_division=0))
        te_r.append(recall_score(yte, vc.predict(Xte), average='weighted', zero_division=0))
    r = dict(name=name, train=np.mean(tr_a), test=np.mean(te_a),
             f1=np.mean(te_f), prec=np.mean(te_p), rec=np.mean(te_r),
             gap=np.mean(tr_a)-np.mean(te_a))
    print(f"  {name:30s} Train={r['train']:.4f} Test={r['test']:.4f} F1={r['f1']:.4f} Prec={r['prec']:.4f} Rec={r['rec']:.4f} Gap={r['gap']:.4f}")
    return r

results.append(eval_ensemble_cv(X, y, "Soft-Voting-Ensemble"))

# ══════════════════════════════════════════════════════════════════════
# PHASE 5: FINAL RANKING
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 5: FINAL MODEL RANKING")
print("=" * 70)

rdf = pd.DataFrame(results).sort_values('test', ascending=False)
print(rdf[['name', 'train', 'test', 'f1', 'prec', 'rec', 'gap']].to_string(index=False))

best = rdf.iloc[0]
print(f"\nBEST: {best['name']}")
print(f"  Test Accuracy:  {best['test']*100:.2f}%")
print(f"  Test F1:        {best['f1']:.4f}")
print(f"  Test Precision: {best['prec']:.4f}")
print(f"  Test Recall:    {best['rec']:.4f}")
print(f"  Overfit Gap:    {best['gap']:.4f}")

# ══════════════════════════════════════════════════════════════════════
# PHASE 6: SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PHASE 6: TRAINING & SAVING FINAL MODEL")
print("=" * 70)

# Train best XGB config on SMOTE-augmented full data
Xs, ys = SMOTE(random_state=42).fit_resample(X, y)

final = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.02, subsample=0.7,
    colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0, gamma=0.2,
    min_child_weight=5, random_state=42, use_label_encoder=False,
    eval_metric='logloss', n_jobs=1
)
final.fit(Xs, ys)

pred = final.predict(X)
print(f"\nResubstitution Accuracy: {accuracy_score(y, pred):.4f}")
print(classification_report(y, pred, target_names=['Not Potable', 'Potable']))
print(f"Confusion Matrix:\n{confusion_matrix(y, pred)}")

# Feature importance
imp = pd.DataFrame({'Feature': all_feats, 'Importance': final.feature_importances_})
imp = imp.sort_values('Importance', ascending=False)
print(f"\nTop 20 Features:")
print(imp.head(20).to_string(index=False))

# Save
os.makedirs('models', exist_ok=True)
joblib.dump(final, 'models/xgboost_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(all_feats, 'models/feature_names.joblib')
print("\nSaved: xgboost_model.joblib, scaler.joblib, feature_names.joblib")

print("\n" + "=" * 70)
print("PIPELINE COMPLETE")
print("=" * 70)
