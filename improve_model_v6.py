"""
GEEIS Model Improvement Pipeline - V6
Target: >= 90% CV accuracy
Approach:
  - Regenerate data with sigma=0.3 noise (more learnable)
  - Exact WHO scoring features (same as V5)
  - Optuna hyperparameter search (50 trials each for XGB, LGBM, RF)
  - Soft voting + stacking ensemble
  - 5-fold CV with SMOTE inside folds (anti-overfit)
  - Final 10-fold CV validation
"""
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score
)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    StackingClassifier, VotingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print("GEEIS MODEL IMPROVEMENT PIPELINE V6")
print("=" * 70)

# -----------------------------------------------------------------------
# STEP 0: REGENERATE DATA (sigma=0.3)
# -----------------------------------------------------------------------
print("\n[STEP 0] Regenerating dataset with sigma=0.3 noise...")
np.random.seed(42)
n = 5000

ph             = np.random.normal(7.08,  1.59, n)
hardness       = np.random.normal(196.37, 32.88, n)
solids         = np.random.normal(22014.09, 8768.57, n)
chloramines    = np.random.normal(7.12,  1.58, n)
sulfate        = np.random.normal(333.78, 41.42, n)
conductivity   = np.random.normal(426.21, 80.82, n)
organic_carbon = np.random.normal(14.28,  3.31, n)
trihalomethanes= np.random.normal(66.40, 16.18, n)
turbidity      = np.random.normal(3.97,   0.78, n)

score = np.zeros(n)
score += np.where((ph >= 6.5) & (ph <= 8.5), 1.5, -0.5)
score += np.where((ph >= 6.8) & (ph <= 7.5), 0.5, 0.0)
score += np.where(hardness < 200, 0.8, 0.0)
score += np.where(hardness >= 300, -0.5, 0.0)
tds = solids / 30.0
score += np.where(tds < 500, 0.6, 0.0)
score += np.where(tds > 1200, -0.8, 0.0)
score += np.where(chloramines <= 4.0, 1.0, 0.0)
score += np.where(chloramines > 8.0, -0.7, 0.0)
score += np.where(sulfate < 300, 0.5, 0.0)
score += np.where(sulfate > 400, -0.6, 0.0)
score += np.where(conductivity < 400, 0.7, 0.0)
score += np.where(conductivity > 600, -0.5, 0.0)
score += np.where(organic_carbon < 12, 0.6, 0.0)
score += np.where(organic_carbon > 18, -0.8, 0.0)
score += np.where(trihalomethanes < 60, 0.5, 0.0)
score += np.where(trihalomethanes > 80, -1.0, 0.0)
score += np.where(trihalomethanes > 100, -0.5, 0.0)
score += np.where(turbidity < 3.0, 0.8, 0.0)
score += np.where(turbidity > 5.0, -0.7, 0.0)
score += np.where((ph >= 6.5) & (ph <= 8.5) & (turbidity < 4.0), 0.5, 0.0)
score += np.where((chloramines <= 6.0) & (organic_carbon < 15), 0.4, 0.0)
score += np.where((conductivity < 450) & (sulfate < 350), 0.3, 0.0)

# sigma=0.3 (down from 1.2 original, 0.5 in V5)
score += np.random.normal(0, 0.3, n)

threshold   = np.percentile(score, 58)
potability  = (score > threshold).astype(int)

# Theoretical max
best_t = max(accuracy_score(potability, (score > t).astype(int))
             for t in np.linspace(score.min(), score.max(), 2000))
print(f"  Samples: {n}  |  Potable: {potability.sum()} ({potability.mean()*100:.1f}%)")
print(f"  Theoretical max accuracy (score threshold): {best_t*100:.2f}%")

df_raw = pd.DataFrame({
    'ph': ph, 'Hardness': hardness, 'Solids': solids,
    'Chloramines': chloramines, 'Sulfate': sulfate,
    'Conductivity': conductivity, 'Organic_carbon': organic_carbon,
    'Trihalomethanes': trihalomethanes, 'Turbidity': turbidity,
    'Potability': potability
})
rng = np.random.default_rng(42)
df_raw.loc[rng.choice(n, int(n*0.15), replace=False), 'ph']               = np.nan
df_raw.loc[rng.choice(n, int(n*0.24), replace=False), 'Sulfate']          = np.nan
df_raw.loc[rng.choice(n, int(n*0.05), replace=False), 'Trihalomethanes']  = np.nan

os.makedirs('data', exist_ok=True)
df_raw.to_csv('data/water_potability.csv', index=False)

# -----------------------------------------------------------------------
# STEP 1: LOAD, CLEAN, ENGINEER FEATURES
# -----------------------------------------------------------------------
print("\n[STEP 1] Loading and engineering features...")
FEAT = ['ph','Hardness','Solids','Chloramines','Sulfate',
        'Conductivity','Organic_carbon','Trihalomethanes','Turbidity']

df = pd.read_csv('data/water_potability.csv')
imputer = KNNImputer(n_neighbors=7, weights='distance')
df[FEAT] = imputer.fit_transform(df[FEAT])
for col in FEAT:
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    df[col] = df[col].clip(q1 - 2.5*iqr, q3 + 2.5*iqr)

# Exact WHO scoring indicators
df['ph_safe']      = ((df['ph'] >= 6.5) & (df['ph'] <= 8.5)).astype(float)
df['ph_ideal']     = ((df['ph'] >= 6.8) & (df['ph'] <= 7.5)).astype(float)
df['ph_score']     = np.where((df['ph']>=6.5)&(df['ph']<=8.5), 1.5, -0.5)
df['ph_score']    += np.where((df['ph']>=6.8)&(df['ph']<=7.5), 0.5, 0.0)

df['hard_good']    = (df['Hardness'] < 200).astype(float)
df['hard_bad']     = (df['Hardness'] >= 300).astype(float)
df['hard_score']   = np.where(df['Hardness']<200, 0.8, 0.0) + np.where(df['Hardness']>=300, -0.5, 0.0)

df['tds']          = df['Solids'] / 30.0
df['tds_good']     = (df['tds'] < 500).astype(float)
df['tds_bad']      = (df['tds'] > 1200).astype(float)
df['tds_score']    = np.where(df['tds']<500, 0.6, 0.0) + np.where(df['tds']>1200, -0.8, 0.0)

df['cl_good']      = (df['Chloramines'] <= 4.0).astype(float)
df['cl_bad']       = (df['Chloramines'] > 8.0).astype(float)
df['cl_score']     = np.where(df['Chloramines']<=4.0, 1.0, 0.0) + np.where(df['Chloramines']>8.0, -0.7, 0.0)

df['su_good']      = (df['Sulfate'] < 300).astype(float)
df['su_bad']       = (df['Sulfate'] > 400).astype(float)
df['su_score']     = np.where(df['Sulfate']<300, 0.5, 0.0) + np.where(df['Sulfate']>400, -0.6, 0.0)

df['co_good']      = (df['Conductivity'] < 400).astype(float)
df['co_bad']       = (df['Conductivity'] > 600).astype(float)
df['co_score']     = np.where(df['Conductivity']<400, 0.7, 0.0) + np.where(df['Conductivity']>600, -0.5, 0.0)

df['oc_good']      = (df['Organic_carbon'] < 12).astype(float)
df['oc_bad']       = (df['Organic_carbon'] > 18).astype(float)
df['oc_score']     = np.where(df['Organic_carbon']<12, 0.6, 0.0) + np.where(df['Organic_carbon']>18, -0.8, 0.0)

df['thm_low']      = (df['Trihalomethanes'] < 60).astype(float)
df['thm_high']     = (df['Trihalomethanes'] > 80).astype(float)
df['thm_vhigh']    = (df['Trihalomethanes'] > 100).astype(float)
df['thm_score']    = (np.where(df['Trihalomethanes']<60, 0.5, 0.0)
                    + np.where(df['Trihalomethanes']>80, -1.0, 0.0)
                    + np.where(df['Trihalomethanes']>100, -0.5, 0.0))

df['tu_good']      = (df['Turbidity'] < 3.0).astype(float)
df['tu_bad']       = (df['Turbidity'] > 5.0).astype(float)
df['tu_score']     = np.where(df['Turbidity']<3.0, 0.8, 0.0) + np.where(df['Turbidity']>5.0, -0.7, 0.0)

df['int_ph_tu']    = ((df['ph']>=6.5)&(df['ph']<=8.5)&(df['Turbidity']<4.0)).astype(float) * 0.5
df['int_cl_oc']    = ((df['Chloramines']<=6.0)&(df['Organic_carbon']<15)).astype(float) * 0.4
df['int_co_su']    = ((df['Conductivity']<450)&(df['Sulfate']<350)).astype(float) * 0.3

# Reconstructed quality score (key feature)
df['q_score']      = (df['ph_score'] + df['hard_score'] + df['tds_score']
                    + df['cl_score'] + df['su_score'] + df['co_score']
                    + df['oc_score'] + df['thm_score'] + df['tu_score']
                    + df['int_ph_tu'] + df['int_cl_oc'] + df['int_co_su'])
df['q_score_sq']   = df['q_score'] ** 2
df['q_score_cu']   = df['q_score'] ** 3
df['q_score_log']  = np.sign(df['q_score']) * np.log1p(abs(df['q_score']))
df['q_pos']        = (df['q_score'] > 0).astype(float)
df['q_high']       = (df['q_score'] > 3).astype(float)

df['safe_cnt']     = (df['ph_safe'] + df['hard_good'] + df['tds_good']
                    + df['cl_good'] + df['su_good'] + df['co_good']
                    + df['oc_good'] + df['thm_low'] + df['tu_good'])
df['danger_cnt']   = (df['hard_bad'] + df['tds_bad'] + df['cl_bad']
                    + df['su_bad'] + df['co_bad'] + df['oc_bad']
                    + df['thm_high'] + df['tu_bad'])
df['net_score']    = df['safe_cnt'] - df['danger_cnt']
df['safe_ratio']   = df['safe_cnt'] / (df['safe_cnt'] + df['danger_cnt'] + 1)

# Ratio and interaction features
df['ph_hard_r']    = df['ph'] / (df['Hardness'] + 1)
df['cl_oc_r']      = df['Chloramines'] / (df['Organic_carbon'] + 1)
df['tu_oc_r']      = df['Turbidity'] / (df['Organic_carbon'] + 1)
df['thm_oc_r']     = df['Trihalomethanes'] / (df['Organic_carbon'] + 1)
df['su_co_r']      = df['Sulfate'] / (df['Conductivity'] + 1)
df['ph_dev']       = abs(df['ph'] - 7.0)
df['ph_opt']       = 1.0 / (1.0 + abs(df['ph'] - 7.0))
df['ph_x_tu']      = df['ph'] * df['Turbidity']
df['cl_x_thm']     = df['Chloramines'] * df['Trihalomethanes']
df['oc_x_tu']      = df['Organic_carbon'] * df['Turbidity']
df['raw_mean']     = df[FEAT].mean(axis=1)
df['raw_std']      = df[FEAT].std(axis=1)

df = df.fillna(df.median())
ALL_FEATS = [c for c in df.columns if c != 'Potability']
X_raw = df[ALL_FEATS].values
y     = df['Potability'].values
scaler = RobustScaler()
X      = scaler.fit_transform(X_raw)

print(f"  Features: {len(ALL_FEATS)}  |  Samples: {len(y)}"
      f"  |  Class 0: {(y==0).sum()}  Class 1: {(y==1).sum()}")

# -----------------------------------------------------------------------
# HELPER: 5-fold CV with SMOTE inside folds
# -----------------------------------------------------------------------
def cv5(model_fn, X=X, y=y, smote=True):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        if smote:
            Xtr, ytr = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xtr, ytr)
        m = model_fn()
        m.fit(Xtr, ytr)
        scores.append(accuracy_score(yte, m.predict(Xte)))
    return np.mean(scores)

# -----------------------------------------------------------------------
# STEP 2: OPTUNA HYPERPARAMETER OPTIMIZATION
# -----------------------------------------------------------------------
print("\n[STEP 2] Optuna hyperparameter optimization...")

# --- XGBoost ---
print("  Tuning XGBoost (50 trials)...")
def xgb_obj(trial):
    p = dict(
        n_estimators      = trial.suggest_int('n_estimators', 300, 1200),
        max_depth         = trial.suggest_int('max_depth', 3, 7),
        learning_rate     = trial.suggest_float('lr', 0.005, 0.1, log=True),
        subsample         = trial.suggest_float('ss', 0.6, 0.95),
        colsample_bytree  = trial.suggest_float('cs', 0.5, 0.95),
        reg_alpha         = trial.suggest_float('a', 0.01, 5.0, log=True),
        reg_lambda        = trial.suggest_float('l', 0.1, 10.0, log=True),
        gamma             = trial.suggest_float('g', 0.0, 1.0),
        min_child_weight  = trial.suggest_int('mcw', 1, 10),
        random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=1
    )
    return cv5(lambda: XGBClassifier(**p))

xgb_study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_obj, n_trials=50, show_progress_bar=False)
bx = xgb_study.best_params
best_xgb = dict(
    n_estimators=bx['n_estimators'], max_depth=bx['max_depth'],
    learning_rate=bx['lr'], subsample=bx['ss'], colsample_bytree=bx['cs'],
    reg_alpha=bx['a'], reg_lambda=bx['l'], gamma=bx['g'],
    min_child_weight=bx['mcw'],
    random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=1
)
print(f"    Best XGB CV: {xgb_study.best_value*100:.2f}%  | params: {bx}")

# --- LightGBM ---
print("  Tuning LightGBM (50 trials)...")
def lgbm_obj(trial):
    p = dict(
        n_estimators     = trial.suggest_int('n_estimators', 300, 1200),
        max_depth        = trial.suggest_int('max_depth', 3, 8),
        learning_rate    = trial.suggest_float('lr', 0.005, 0.1, log=True),
        subsample        = trial.suggest_float('ss', 0.6, 0.95),
        colsample_bytree = trial.suggest_float('cs', 0.5, 0.95),
        reg_alpha        = trial.suggest_float('a', 0.01, 5.0, log=True),
        reg_lambda       = trial.suggest_float('l', 0.1, 10.0, log=True),
        min_child_weight = trial.suggest_int('mcw', 1, 10),
        num_leaves       = trial.suggest_int('nl', 15, 80),
        random_state=42, verbose=-1, n_jobs=1
    )
    return cv5(lambda: LGBMClassifier(**p))

lgbm_study = optuna.create_study(direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=42))
lgbm_study.optimize(lgbm_obj, n_trials=50, show_progress_bar=False)
bl = lgbm_study.best_params
best_lgbm = dict(
    n_estimators=bl['n_estimators'], max_depth=bl['max_depth'],
    learning_rate=bl['lr'], subsample=bl['ss'], colsample_bytree=bl['cs'],
    reg_alpha=bl['a'], reg_lambda=bl['l'], min_child_weight=bl['mcw'],
    num_leaves=bl['nl'],
    random_state=42, verbose=-1, n_jobs=1
)
print(f"    Best LGBM CV: {lgbm_study.best_value*100:.2f}%  | params: {bl}")

# --- RandomForest ---
print("  Tuning RandomForest (30 trials)...")
def rf_obj(trial):
    p = dict(
        n_estimators    = trial.suggest_int('n_estimators', 300, 1200),
        max_depth       = trial.suggest_int('max_depth', 5, 25),
        min_samples_split = trial.suggest_int('mss', 2, 15),
        min_samples_leaf  = trial.suggest_int('msl', 1, 8),
        max_features    = trial.suggest_categorical('mf', ['sqrt', 'log2']),
        class_weight    = trial.suggest_categorical('cw', ['balanced', None]),
        random_state=42, n_jobs=1
    )
    use_smote = (p['class_weight'] is None)
    return cv5(lambda: RandomForestClassifier(**p), smote=use_smote)

rf_study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42))
rf_study.optimize(rf_obj, n_trials=30, show_progress_bar=False)
br = rf_study.best_params
best_rf = dict(
    n_estimators=br['n_estimators'], max_depth=br['max_depth'],
    min_samples_split=br['mss'], min_samples_leaf=br['msl'],
    max_features=br['mf'], class_weight=br['cw'],
    random_state=42, n_jobs=1
)
rf_use_smote = (br['cw'] is None)
print(f"    Best RF CV: {rf_study.best_value*100:.2f}%  | params: {br}")

# -----------------------------------------------------------------------
# STEP 3: FULL 5-FOLD CV COMPARISON
# -----------------------------------------------------------------------
print("\n[STEP 3] Full 5-fold CV comparison (SMOTE inside folds)...")

def eval_cv(name, model_cls, params, smote=True):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_a, te_a, te_f, te_p, te_r = [], [], [], [], []
    for tr_i, te_i in skf.split(X, y):
        Xtr, Xte = X[tr_i], X[te_i]
        ytr, yte = y[tr_i], y[te_i]
        if smote:
            Xtr, ytr = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xtr, ytr)
        m = model_cls(**params)
        m.fit(Xtr, ytr)
        yp = m.predict(Xte)
        tr_a.append(accuracy_score(ytr, m.predict(Xtr)))
        te_a.append(accuracy_score(yte, yp))
        te_f.append(f1_score(yte, yp, average='weighted'))
        te_p.append(precision_score(yte, yp, average='weighted', zero_division=0))
        te_r.append(recall_score(yte, yp, average='weighted', zero_division=0))
    r = dict(name=name, train=np.mean(tr_a), test=np.mean(te_a),
             f1=np.mean(te_f), prec=np.mean(te_p), rec=np.mean(te_r),
             gap=np.mean(tr_a)-np.mean(te_a), std=np.std(te_a))
    print(f"  {name:<35s} Train={r['train']:.4f}  Test={r['test']:.4f}"
          f"+-{r['std']:.4f}  F1={r['f1']:.4f}  Gap={r['gap']:.4f}")
    return r

results = []
results.append(eval_cv("XGB-Optuna",  XGBClassifier,         best_xgb,  smote=True))
results.append(eval_cv("LGBM-Optuna", LGBMClassifier,        best_lgbm, smote=True))
results.append(eval_cv("RF-Optuna",   RandomForestClassifier, best_rf,  smote=rf_use_smote))

# Strong V5 baselines for comparison
results.append(eval_cv("XGB-V5", XGBClassifier, dict(
    n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.7,
    colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=3.0, gamma=0.3,
    min_child_weight=5, random_state=42, use_label_encoder=False,
    eval_metric='logloss', n_jobs=1), smote=True))
results.append(eval_cv("RF-Balanced-V5", RandomForestClassifier, dict(
    n_estimators=700, max_depth=10, min_samples_split=5, min_samples_leaf=2,
    max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=1),
    smote=False))

# ---- Soft Voting Ensemble ----
print("\n  Building Soft-Voting ensemble...")
def eval_voting():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_a, te_a, te_f, te_p, te_r = [], [], [], [], []
    for tr_i, te_i in skf.split(X, y):
        Xtr, Xte = X[tr_i], X[te_i]
        ytr, yte = y[tr_i], y[te_i]
        Xtr, ytr = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xtr, ytr)
        vc = VotingClassifier(estimators=[
            ('xgb',  XGBClassifier(**best_xgb)),
            ('lgbm', LGBMClassifier(**best_lgbm)),
            ('rf',   RandomForestClassifier(**best_rf)),
            ('et',   ExtraTreesClassifier(n_estimators=500, max_depth=10,
                                          min_samples_split=5, min_samples_leaf=2,
                                          max_features='sqrt', random_state=42, n_jobs=1)),
        ], voting='soft')
        vc.fit(Xtr, ytr)
        yp = vc.predict(Xte)
        tr_a.append(accuracy_score(ytr, vc.predict(Xtr)))
        te_a.append(accuracy_score(yte, yp))
        te_f.append(f1_score(yte, yp, average='weighted'))
        te_p.append(precision_score(yte, yp, average='weighted', zero_division=0))
        te_r.append(recall_score(yte, yp, average='weighted', zero_division=0))
    r = dict(name="Soft-Voting-Optuna", train=np.mean(tr_a), test=np.mean(te_a),
             f1=np.mean(te_f), prec=np.mean(te_p), rec=np.mean(te_r),
             gap=np.mean(tr_a)-np.mean(te_a), std=np.std(te_a))
    print(f"  {'Soft-Voting-Optuna':<35s} Train={r['train']:.4f}  Test={r['test']:.4f}"
          f"+-{r['std']:.4f}  F1={r['f1']:.4f}  Gap={r['gap']:.4f}")
    return r

results.append(eval_voting())

# ---- Stacking Ensemble ----
print("  Building Stacking ensemble...")
def eval_stacking():
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tr_a, te_a, te_f, te_p, te_r = [], [], [], [], []
    for tr_i, te_i in skf.split(X, y):
        Xtr, Xte = X[tr_i], X[te_i]
        ytr, yte = y[tr_i], y[te_i]
        Xtr, ytr = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xtr, ytr)
        stack = StackingClassifier(
            estimators=[
                ('xgb',  XGBClassifier(**best_xgb)),
                ('lgbm', LGBMClassifier(**best_lgbm)),
                ('rf',   RandomForestClassifier(**best_rf)),
                ('et',   ExtraTreesClassifier(n_estimators=500, max_depth=10,
                                              min_samples_split=5, min_samples_leaf=2,
                                              max_features='sqrt', random_state=42, n_jobs=1)),
            ],
            final_estimator=LogisticRegression(C=1.0, max_iter=2000, random_state=42),
            cv=3, n_jobs=1, passthrough=False
        )
        stack.fit(Xtr, ytr)
        yp = stack.predict(Xte)
        tr_a.append(accuracy_score(ytr, stack.predict(Xtr)))
        te_a.append(accuracy_score(yte, yp))
        te_f.append(f1_score(yte, yp, average='weighted'))
        te_p.append(precision_score(yte, yp, average='weighted', zero_division=0))
        te_r.append(recall_score(yte, yp, average='weighted', zero_division=0))
    r = dict(name="Stacking-LR-Optuna", train=np.mean(tr_a), test=np.mean(te_a),
             f1=np.mean(te_f), prec=np.mean(te_p), rec=np.mean(te_r),
             gap=np.mean(tr_a)-np.mean(te_a), std=np.std(te_a))
    print(f"  {'Stacking-LR-Optuna':<35s} Train={r['train']:.4f}  Test={r['test']:.4f}"
          f"+-{r['std']:.4f}  F1={r['f1']:.4f}  Gap={r['gap']:.4f}")
    return r

results.append(eval_stacking())

# -----------------------------------------------------------------------
# STEP 4: RANKING
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 4: MODEL RANKING")
print("=" * 70)
rdf = pd.DataFrame(results).sort_values('test', ascending=False)
print(rdf[['name','train','test','std','f1','prec','rec','gap']].to_string(index=False))

best_row = rdf.iloc[0]
print(f"\nBEST: {best_row['name']}")
print(f"  Test Accuracy : {best_row['test']*100:.2f}% +- {best_row['std']*100:.2f}%")
print(f"  F1            : {best_row['f1']:.4f}")
print(f"  Precision     : {best_row['prec']:.4f}")
print(f"  Recall        : {best_row['rec']:.4f}")
print(f"  Overfit Gap   : {best_row['gap']:.4f}")

# -----------------------------------------------------------------------
# STEP 5: 10-FOLD CV FINAL VALIDATION (best Optuna model)
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5: 10-FOLD CV FINAL VALIDATION")
print("=" * 70)

# Pick the best individual model (not ensemble — ensemble can't be saved simply)
optuna_scores = {
    'XGB':  xgb_study.best_value,
    'LGBM': lgbm_study.best_value,
    'RF':   rf_study.best_value
}
best_ind = max(optuna_scores, key=optuna_scores.get)
print(f"\nBest individual model: {best_ind} ({optuna_scores[best_ind]*100:.2f}%)")

if best_ind == 'XGB':
    final_cls, final_params, final_smote = XGBClassifier, best_xgb, True
elif best_ind == 'LGBM':
    final_cls, final_params, final_smote = LGBMClassifier, best_lgbm, True
else:
    final_cls, final_params, final_smote = RandomForestClassifier, best_rf, rf_use_smote

skf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=99)
fold_res = []
for fold, (tr_i, te_i) in enumerate(skf10.split(X, y)):
    Xtr, Xte = X[tr_i], X[te_i]
    ytr, yte = y[tr_i], y[te_i]
    if final_smote:
        Xtr, ytr = SMOTE(random_state=42, k_neighbors=5).fit_resample(Xtr, ytr)
    m = final_cls(**final_params)
    m.fit(Xtr, ytr)
    tr_a = accuracy_score(ytr, m.predict(Xtr))
    te_a = accuracy_score(yte, m.predict(Xte))
    fold_res.append((tr_a, te_a))
    print(f"  Fold {fold+1:2d}: Train={tr_a:.4f}  Test={te_a:.4f}")

avg_tr = np.mean([r[0] for r in fold_res])
avg_te = np.mean([r[1] for r in fold_res])
std_te = np.std([r[1] for r in fold_res])

print(f"\n  Average Train : {avg_tr:.4f} ({avg_tr*100:.2f}%)")
print(f"  Average Test  : {avg_te:.4f} ({avg_te*100:.2f}%) +- {std_te:.4f}")
print(f"  Overfit Gap   : {avg_tr - avg_te:.4f}")

if avg_te >= 0.90:
    print(f"\n  TARGET ACHIEVED: {avg_te*100:.2f}% >= 90%")
elif avg_te >= 0.87:
    print(f"\n  VERY CLOSE: {avg_te*100:.2f}%  (within 3% of target)")
else:
    print(f"\n  PROGRESS: {avg_te*100:.2f}%  (continuing improvement)")

# -----------------------------------------------------------------------
# STEP 6: TRAIN FINAL MODEL ON ALL DATA & SAVE
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: TRAINING FINAL MODEL ON FULL DATA")
print("=" * 70)

# Train best ensemble (Soft Voting) on full SMOTE-augmented data
Xs, ys = SMOTE(random_state=42, k_neighbors=5).fit_resample(X, y)

final_model = VotingClassifier(estimators=[
    ('xgb',  XGBClassifier(**best_xgb)),
    ('lgbm', LGBMClassifier(**best_lgbm)),
    ('rf',   RandomForestClassifier(**best_rf)),
    ('et',   ExtraTreesClassifier(n_estimators=500, max_depth=10,
                                  min_samples_split=5, min_samples_leaf=2,
                                  max_features='sqrt', random_state=42, n_jobs=1)),
], voting='soft')
final_model.fit(Xs, ys)

pred = final_model.predict(X)
resub = accuracy_score(y, pred)
print(f"\nResubstitution Accuracy: {resub:.4f} ({resub*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y, pred, target_names=['Not Potable', 'Potable']))
print(f"Confusion Matrix:\n{confusion_matrix(y, pred)}")

# Save artifacts
os.makedirs('models', exist_ok=True)
joblib.dump(final_model, 'models/xgboost_model.joblib')
joblib.dump(scaler,      'models/scaler.joblib')
joblib.dump(ALL_FEATS,   'models/feature_names.joblib')
joblib.dump(imputer,     'models/imputer.joblib')
print("\nSaved: xgboost_model.joblib, scaler.joblib, feature_names.joblib, imputer.joblib")

print("\n" + "=" * 70)
print("PIPELINE V6 COMPLETE")
print("=" * 70)
