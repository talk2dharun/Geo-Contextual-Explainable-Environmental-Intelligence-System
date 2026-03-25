"""Quick verification of improvement results"""
import os, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load and process data
df = pd.read_csv('data/water_potability.csv')
feature_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
df.loc[df['Solids'] < 0, 'Solids'] = np.nan
imputer = KNNImputer(n_neighbors=7, weights='distance')
df[feature_cols] = imputer.fit_transform(df[feature_cols])
for col in feature_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 2*IQR, upper=Q3 + 2*IQR)

# Feature engineering (same as main pipeline)
df['ph_hard_r'] = df['ph'] / (df['Hardness'] + 1)
df['tds_cond_r'] = df['Solids'] / (df['Conductivity'] + 1)
df['chlor_org_r'] = df['Chloramines'] / (df['Organic_carbon'] + 1)
df['sulf_hard_r'] = df['Sulfate'] / (df['Hardness'] + 1)
df['turb_org_r'] = df['Turbidity'] / (df['Organic_carbon'] + 1)
df['thm_org_r'] = df['Trihalomethanes'] / (df['Organic_carbon'] + 1)
df['ph_turb_r'] = df['ph'] / (df['Turbidity'] + 1)
df['chlor_thm_r'] = df['Chloramines'] / (df['Trihalomethanes'] + 1)
df['sulf_cond_r'] = df['Sulfate'] / (df['Conductivity'] + 1)
df['ph_turb'] = df['ph'] * df['Turbidity']
df['hard_sulf'] = df['Hardness'] * df['Sulfate']
df['chlor_thm'] = df['Chloramines'] * df['Trihalomethanes']
df['ph_cond'] = df['ph'] * df['Conductivity']
df['org_turb'] = df['Organic_carbon'] * df['Turbidity']
df['ph_chlor'] = df['ph'] * df['Chloramines']
df['hard_cond'] = df['Hardness'] * df['Conductivity']
df['thm_turb'] = df['Trihalomethanes'] * df['Turbidity']
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
df['feat_mean'] = df[feature_cols].mean(axis=1)
df['feat_std'] = df[feature_cols].std(axis=1)
for col in feature_cols:
    df[col+'_sq'] = df[col] ** 2
for col in feature_cols:
    if (df[col] > 0).all():
        df[col+'_log'] = np.log1p(df[col])
df['ph_bin'] = pd.cut(df['ph'], bins=[0,6.5,7.0,7.5,8.5,14], labels=[0,1,2,3,4]).astype(float)
df['turb_bin'] = pd.cut(df['Turbidity'], bins=[0,2,3,4,5,10], labels=[0,1,2,3,4]).astype(float)
df['chlor_bin'] = pd.cut(df['Chloramines'], bins=[0,4,6,8,15], labels=[0,1,2,3]).astype(float)
df['thm_bin'] = pd.cut(df['Trihalomethanes'], bins=[0,40,60,80,200], labels=[0,1,2,3]).astype(float)
df = df.fillna(df.median())

all_feats = [c for c in df.columns if c != 'Potability']
X = df[all_feats].values
y = df['Potability'].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Open output file
out = open('FINAL_RESULTS.txt', 'w')

def log(msg):
    print(msg)
    out.write(msg + '\n')

log("=" * 70)
log("GEEIS MODEL IMPROVEMENT - FINAL RESULTS")
log("=" * 70)
log("")
log("Dataset: {} samples, {} features".format(X.shape[0], X.shape[1]))
log("Target distribution: 0={}, 1={}".format(np.sum(y==0), np.sum(y==1)))
log("")

# Best model config (from pipeline run)
best_config = dict(
    n_estimators=1000, max_depth=6, learning_rate=0.01, subsample=0.7,
    colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=5.0, gamma=0.3,
    min_child_weight=7, random_state=42, use_label_encoder=False,
    eval_metric='logloss', n_jobs=1
)

# Full 5-fold CV evaluation
log("5-FOLD STRATIFIED CROSS-VALIDATION (SMOTE inside each fold)")
log("-" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (tr_i, te_i) in enumerate(skf.split(X, y)):
    Xtr, Xte = X[tr_i], X[te_i]
    ytr, yte = y[tr_i], y[te_i]
    Xtr, ytr = SMOTE(random_state=42).fit_resample(Xtr, ytr)
    
    model = XGBClassifier(**best_config)
    model.fit(Xtr, ytr)
    
    tr_pred = model.predict(Xtr)
    te_pred = model.predict(Xte)
    
    tr_acc = accuracy_score(ytr, tr_pred)
    te_acc = accuracy_score(yte, te_pred)
    te_f1 = f1_score(yte, te_pred, average='weighted')
    te_prec = precision_score(yte, te_pred, average='weighted', zero_division=0)
    te_rec = recall_score(yte, te_pred, average='weighted', zero_division=0)
    
    fold_results.append({
        'fold': fold+1, 'train_acc': tr_acc, 'test_acc': te_acc,
        'f1': te_f1, 'prec': te_prec, 'rec': te_rec
    })
    log("Fold {}: Train={:.4f}  Test={:.4f}  F1={:.4f}  Prec={:.4f}  Rec={:.4f}".format(
        fold+1, tr_acc, te_acc, te_f1, te_prec, te_rec))

avg_train = np.mean([r['train_acc'] for r in fold_results])
avg_test = np.mean([r['test_acc'] for r in fold_results])
avg_f1 = np.mean([r['f1'] for r in fold_results])
avg_prec = np.mean([r['prec'] for r in fold_results])
avg_rec = np.mean([r['rec'] for r in fold_results])
gap = avg_train - avg_test

log("")
log("AVERAGE ACROSS 5 FOLDS:")
log("  Train Accuracy:  {:.4f} ({:.2f}%)".format(avg_train, avg_train*100))
log("  Test Accuracy:   {:.4f} ({:.2f}%)".format(avg_test, avg_test*100))
log("  Test F1 Score:   {:.4f}".format(avg_f1))
log("  Test Precision:  {:.4f}".format(avg_prec))
log("  Test Recall:     {:.4f}".format(avg_rec))
log("  Overfit Gap:     {:.4f}".format(gap))

# Train final model on SMOTE-augmented full data
log("")
log("=" * 70)
log("FINAL MODEL (Trained on full SMOTE-augmented data)")
log("=" * 70)

Xs, ys = SMOTE(random_state=42).fit_resample(X, y)
final_model = XGBClassifier(**best_config)
final_model.fit(Xs, ys)

# Resubstitution
pred = final_model.predict(X)
resub_acc = accuracy_score(y, pred)
log("")
log("Resubstitution Accuracy: {:.4f} ({:.2f}%)".format(resub_acc, resub_acc*100))
log("")
log("Classification Report:")
log(classification_report(y, pred, target_names=['Not Potable', 'Potable']))
log("Confusion Matrix:")
cm = confusion_matrix(y, pred)
log(str(cm))

# Feature importance
imp = pd.DataFrame({'Feature': all_feats, 'Importance': final_model.feature_importances_})
imp = imp.sort_values('Importance', ascending=False)
log("")
log("TOP 20 MOST IMPORTANT FEATURES:")
for i, row in imp.head(20).iterrows():
    log("  {}: {:.6f}".format(row['Feature'], row['Importance']))

# Save
os.makedirs('models', exist_ok=True)
joblib.dump(final_model, 'models/xgboost_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(all_feats, 'models/feature_names.joblib')

log("")
log("Models saved to models/ directory")
log("")
log("=" * 70)
log("COMPLETE")
log("=" * 70)

out.close()
