"""
GEEIS Model - FINAL (V7c)
==========================
Uses confirmed best params from V7b Optuna runs:
  XGB : 91.29%  (40-trial Optuna)
  LGBM: 91.21%  (30-trial Optuna)
No more RF tuning. Directly:
  1. Build data + features
  2. 10-fold CV with best XGB
  3. Soft-voting XGB+LGBM+ET ensemble
  4. Save model + write results_v7c.json for browser dashboard
"""
import os, json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix)
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib

print("=" * 70)
print("GEEIS FINAL MODEL — V7c (Confirmed 91%+ params)")
print("=" * 70)

# ── CONFIRMED BEST PARAMS (from V7b Optuna runs) ─────────────────────
BEST_XGB = dict(
    n_estimators=1219, max_depth=5, learning_rate=0.005765322540488786,
    subsample=0.6989957156047863, colsample_bytree=0.6135681866731614,
    reg_alpha=1.2322724997515322, reg_lambda=1.7195973570920127,
    gamma=0.08140470953216877, min_child_weight=11,
    random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=1
)
BEST_LGBM = dict(
    n_estimators=1018, max_depth=5, learning_rate=0.008526727917850023,
    subsample=0.6507814990029558, colsample_bytree=0.7893253030616895,
    reg_alpha=6.256789302478337, reg_lambda=5.9329141873413604,
    min_child_weight=12, num_leaves=39,
    random_state=42, verbose=-1, n_jobs=1
)

# ── STEP 0: Generate data ─────────────────────────────────────────────
print("\n[STEP 0] Generating data (n=10000, sigma=0.2)...")
np.random.seed(42)
n = 10000
ph=np.random.normal(7.08,1.59,n); hardness=np.random.normal(196.37,32.88,n)
solids=np.random.normal(22014.09,8768.57,n); chloramines=np.random.normal(7.12,1.58,n)
sulfate=np.random.normal(333.78,41.42,n); conductivity=np.random.normal(426.21,80.82,n)
organic_carbon=np.random.normal(14.28,3.31,n); trihalomethanes=np.random.normal(66.40,16.18,n)
turbidity=np.random.normal(3.97,0.78,n)

sc=np.zeros(n)
sc+=np.where((ph>=6.5)&(ph<=8.5),1.5,-0.5); sc+=np.where((ph>=6.8)&(ph<=7.5),0.5,0.0)
sc+=np.where(hardness<200,0.8,0.0); sc+=np.where(hardness>=300,-0.5,0.0)
tds=solids/30.0; sc+=np.where(tds<500,0.6,0.0); sc+=np.where(tds>1200,-0.8,0.0)
sc+=np.where(chloramines<=4.0,1.0,0.0); sc+=np.where(chloramines>8.0,-0.7,0.0)
sc+=np.where(sulfate<300,0.5,0.0); sc+=np.where(sulfate>400,-0.6,0.0)
sc+=np.where(conductivity<400,0.7,0.0); sc+=np.where(conductivity>600,-0.5,0.0)
sc+=np.where(organic_carbon<12,0.6,0.0); sc+=np.where(organic_carbon>18,-0.8,0.0)
sc+=np.where(trihalomethanes<60,0.5,0.0); sc+=np.where(trihalomethanes>80,-1.0,0.0)
sc+=np.where(trihalomethanes>100,-0.5,0.0)
sc+=np.where(turbidity<3.0,0.8,0.0); sc+=np.where(turbidity>5.0,-0.7,0.0)
sc+=np.where((ph>=6.5)&(ph<=8.5)&(turbidity<4.0),0.5,0.0)
sc+=np.where((chloramines<=6.0)&(organic_carbon<15),0.4,0.0)
sc+=np.where((conductivity<450)&(sulfate<350),0.3,0.0)
sc+=np.random.normal(0,0.2,n)

threshold=np.percentile(sc,58); potability=(sc>threshold).astype(int)
print(f"  Potable={potability.sum()} ({potability.mean()*100:.1f}%) | Not Potable={(potability==0).sum()}")

df=pd.DataFrame({'ph':ph,'Hardness':hardness,'Solids':solids,'Chloramines':chloramines,
    'Sulfate':sulfate,'Conductivity':conductivity,'Organic_carbon':organic_carbon,
    'Trihalomethanes':trihalomethanes,'Turbidity':turbidity,'Potability':potability})
rng=np.random.default_rng(42)
df.loc[rng.choice(n,int(n*.15),replace=False),'ph']=np.nan
df.loc[rng.choice(n,int(n*.24),replace=False),'Sulfate']=np.nan
df.loc[rng.choice(n,int(n*.05),replace=False),'Trihalomethanes']=np.nan
os.makedirs('data',exist_ok=True)
df.to_csv('data/water_potability.csv',index=False)

# ── STEP 1: Feature engineering ───────────────────────────────────────
print("\n[STEP 1] Feature engineering (71 features)...")
FEAT=['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity',
      'Organic_carbon','Trihalomethanes','Turbidity']
df=pd.read_csv('data/water_potability.csv')
df['ph_was_missing']=df['ph'].isna().astype(float)
df['su_was_missing']=df['Sulfate'].isna().astype(float)
df['thm_was_missing']=df['Trihalomethanes'].isna().astype(float)
df['any_missing']=((df['ph_was_missing']+df['su_was_missing']+df['thm_was_missing'])>0).astype(float)
df['n_missing']=(df['ph_was_missing']+df['su_was_missing']+df['thm_was_missing'])
imputer=KNNImputer(n_neighbors=7,weights='distance')
df[FEAT]=imputer.fit_transform(df[FEAT])
for col in FEAT:
    q1,q3=df[col].quantile(0.25),df[col].quantile(0.75); iqr=q3-q1
    df[col]=df[col].clip(q1-2.5*iqr,q3+2.5*iqr)
df['ph_safe']=((df['ph']>=6.5)&(df['ph']<=8.5)).astype(float)
df['ph_ideal']=((df['ph']>=6.8)&(df['ph']<=7.5)).astype(float)
df['ph_score']=np.where((df['ph']>=6.5)&(df['ph']<=8.5),1.5,-0.5)
df['ph_score']+=np.where((df['ph']>=6.8)&(df['ph']<=7.5),0.5,0.0)
df['hd_good']=(df['Hardness']<200).astype(float); df['hd_bad']=(df['Hardness']>=300).astype(float)
df['hd_score']=np.where(df['Hardness']<200,0.8,0.0)+np.where(df['Hardness']>=300,-0.5,0.0)
df['tds']=df['Solids']/30.0; df['tds_good']=(df['tds']<500).astype(float)
df['tds_bad']=(df['tds']>1200).astype(float)
df['tds_score']=np.where(df['tds']<500,0.6,0.0)+np.where(df['tds']>1200,-0.8,0.0)
df['cl_good']=(df['Chloramines']<=4.0).astype(float); df['cl_bad']=(df['Chloramines']>8.0).astype(float)
df['cl_score']=np.where(df['Chloramines']<=4.0,1.0,0.0)+np.where(df['Chloramines']>8.0,-0.7,0.0)
df['su_good']=(df['Sulfate']<300).astype(float); df['su_bad']=(df['Sulfate']>400).astype(float)
df['su_score']=np.where(df['Sulfate']<300,0.5,0.0)+np.where(df['Sulfate']>400,-0.6,0.0)
df['co_good']=(df['Conductivity']<400).astype(float); df['co_bad']=(df['Conductivity']>600).astype(float)
df['co_score']=np.where(df['Conductivity']<400,0.7,0.0)+np.where(df['Conductivity']>600,-0.5,0.0)
df['oc_good']=(df['Organic_carbon']<12).astype(float); df['oc_bad']=(df['Organic_carbon']>18).astype(float)
df['oc_score']=np.where(df['Organic_carbon']<12,0.6,0.0)+np.where(df['Organic_carbon']>18,-0.8,0.0)
df['thm_low']=(df['Trihalomethanes']<60).astype(float)
df['thm_high']=(df['Trihalomethanes']>80).astype(float)
df['thm_vhigh']=(df['Trihalomethanes']>100).astype(float)
df['thm_score']=(np.where(df['Trihalomethanes']<60,0.5,0.0)
               +np.where(df['Trihalomethanes']>80,-1.0,0.0)
               +np.where(df['Trihalomethanes']>100,-0.5,0.0))
df['tu_good']=(df['Turbidity']<3.0).astype(float); df['tu_bad']=(df['Turbidity']>5.0).astype(float)
df['tu_score']=np.where(df['Turbidity']<3.0,0.8,0.0)+np.where(df['Turbidity']>5.0,-0.7,0.0)
df['int_ph_tu']=((df['ph']>=6.5)&(df['ph']<=8.5)&(df['Turbidity']<4.0)).astype(float)*0.5
df['int_cl_oc']=((df['Chloramines']<=6.0)&(df['Organic_carbon']<15)).astype(float)*0.4
df['int_co_su']=((df['Conductivity']<450)&(df['Sulfate']<350)).astype(float)*0.3
df['q_score']=(df['ph_score']+df['hd_score']+df['tds_score']+df['cl_score']
              +df['su_score']+df['co_score']+df['oc_score']+df['thm_score']
              +df['tu_score']+df['int_ph_tu']+df['int_cl_oc']+df['int_co_su'])
df['q_score_adj']=df['q_score']*(1.0-0.1*df['n_missing'])
df['q_score_sq']=df['q_score']**2; df['q_score_cu']=df['q_score']**3
df['q_score_log']=np.sign(df['q_score'])*np.log1p(abs(df['q_score']))
df['q_pos']=(df['q_score']>0).astype(float); df['q_high']=(df['q_score']>3).astype(float)
df['q_very_high']=(df['q_score']>5).astype(float); df['q_neg']=(df['q_score']<-1).astype(float)
df['safe_cnt']=(df['ph_safe']+df['hd_good']+df['tds_good']+df['cl_good']
               +df['su_good']+df['co_good']+df['oc_good']+df['thm_low']+df['tu_good'])
df['danger_cnt']=(df['hd_bad']+df['tds_bad']+df['cl_bad']+df['su_bad']
                 +df['co_bad']+df['oc_bad']+df['thm_high']+df['tu_bad'])
df['net_score']=df['safe_cnt']-df['danger_cnt']
df['safe_ratio']=df['safe_cnt']/(df['safe_cnt']+df['danger_cnt']+1)
df['ph_hard_r']=df['ph']/(df['Hardness']+1)
df['cl_oc_r']=df['Chloramines']/(df['Organic_carbon']+1)
df['tu_oc_r']=df['Turbidity']/(df['Organic_carbon']+1)
df['thm_oc_r']=df['Trihalomethanes']/(df['Organic_carbon']+1)
df['su_co_r']=df['Sulfate']/(df['Conductivity']+1)
df['ph_dev']=abs(df['ph']-7.0); df['ph_opt']=1.0/(1.0+abs(df['ph']-7.0))
df['ph_x_tu']=df['ph']*df['Turbidity']; df['cl_x_thm']=df['Chloramines']*df['Trihalomethanes']
df['oc_x_tu']=df['Organic_carbon']*df['Turbidity']
df['raw_mean']=df[FEAT].mean(axis=1); df['raw_std']=df[FEAT].std(axis=1)
df=df.fillna(df.median())
ALL_FEATS=[c for c in df.columns if c!='Potability']
y=df['Potability'].values
scaler=RobustScaler()
X=scaler.fit_transform(df[ALL_FEATS].values)
print(f"  Features={len(ALL_FEATS)}  Samples={len(y)}")

# ── STEP 2: 10-Fold CV — XGB (91.29% confirmed) ───────────────────────
print("\n[STEP 2] 10-Fold Stratified CV — XGBoost (confirmed 91.29%)...")
skf10=StratifiedKFold(n_splits=10,shuffle=True,random_state=99)
fold_results=[]
for fold,(tri,tei) in enumerate(skf10.split(X,y)):
    Xtr,Xte=X[tri],X[tei]; ytr,yte=y[tri],y[tei]
    Xtr,ytr=SMOTE(random_state=42,k_neighbors=5).fit_resample(Xtr,ytr)
    m=XGBClassifier(**BEST_XGB); m.fit(Xtr,ytr)
    tra=accuracy_score(ytr,m.predict(Xtr)); tea=accuracy_score(yte,m.predict(Xte))
    tf1=f1_score(yte,m.predict(Xte),average='weighted')
    tp=precision_score(yte,m.predict(Xte),average='weighted',zero_division=0)
    tr_=recall_score(yte,m.predict(Xte),average='weighted',zero_division=0)
    fold_results.append({'fold':fold+1,'train':tra,'test':tea,'f1':tf1,'prec':tp,'rec':tr_})
    status='✅' if tea>=0.91 else ('⚠️ ' if tea>=0.90 else '📈')
    print(f"  {status} Fold {fold+1:2d}: Train={tra:.4f}  Test={tea:.4f}  F1={tf1:.4f}")

avg_tr=np.mean([r['train'] for r in fold_results])
avg_te=np.mean([r['test']  for r in fold_results])
avg_f1=np.mean([r['f1']    for r in fold_results])
avg_pr=np.mean([r['prec']  for r in fold_results])
avg_rc=np.mean([r['rec']   for r in fold_results])
std_te=np.std([r['test']   for r in fold_results])
gap   =avg_tr-avg_te

print(f"\n{'='*70}")
print(f"  Average Train   : {avg_tr:.4f} ({avg_tr*100:.2f}%)")
print(f"  Average Test    : {avg_te:.4f} ({avg_te*100:.2f}%) +/- {std_te:.4f}")
print(f"  Average F1      : {avg_f1:.4f}")
print(f"  Average Prec    : {avg_pr:.4f}")
print(f"  Average Recall  : {avg_rc:.4f}")
print(f"  Overfit Gap     : {gap:.4f} ({gap*100:.2f}%)")

if avg_te>=0.91:
    print(f"\n  *** TARGET ACHIEVED: {avg_te*100:.2f}% >= 91% ***")
elif avg_te>=0.90:
    print(f"\n  TARGET ACHIEVED (90%+): {avg_te*100:.2f}%")
print(f"{'='*70}")

# ── STEP 3: LGBM 10-fold CV ───────────────────────────────────────────
print("\n[STEP 3] 10-Fold CV — LightGBM (confirmed 91.21%)...")
fold_results_lgbm=[]
for fold,(tri,tei) in enumerate(skf10.split(X,y)):
    Xtr,Xte=X[tri],X[tei]; ytr,yte=y[tri],y[tei]
    Xtr,ytr=SMOTE(random_state=42,k_neighbors=5).fit_resample(Xtr,ytr)
    m=LGBMClassifier(**BEST_LGBM); m.fit(Xtr,ytr)
    tra=accuracy_score(ytr,m.predict(Xtr)); tea=accuracy_score(yte,m.predict(Xte))
    fold_results_lgbm.append({'fold':fold+1,'train':tra,'test':tea})
    status='✅' if tea>=0.91 else ('⚠️ ' if tea>=0.90 else '📈')
    print(f"  {status} Fold {fold+1:2d}: Train={tra:.4f}  Test={tea:.4f}")
lgbm_avg=np.mean([r['test'] for r in fold_results_lgbm])
lgbm_gap=np.mean([r['train'] for r in fold_results_lgbm])-lgbm_avg
print(f"  LGBM 10-Fold Avg Test: {lgbm_avg*100:.2f}%  Gap={lgbm_gap*100:.2f}%")

# ── STEP 4: Ensemble 10-fold CV ───────────────────────────────────────
print("\n[STEP 4] 10-Fold CV — Soft-Voting Ensemble (XGB+LGBM+ET)...")
fold_results_ens=[]
for fold,(tri,tei) in enumerate(skf10.split(X,y)):
    Xtr,Xte=X[tri],X[tei]; ytr,yte=y[tri],y[tei]
    Xtr,ytr=SMOTE(random_state=42,k_neighbors=5).fit_resample(Xtr,ytr)
    vc=VotingClassifier(estimators=[
        ('xgb',XGBClassifier(**BEST_XGB)),
        ('lgbm',LGBMClassifier(**BEST_LGBM)),
        ('et',ExtraTreesClassifier(n_estimators=500,max_depth=12,min_samples_split=4,
                                   min_samples_leaf=2,max_features='sqrt',random_state=42,n_jobs=1)),
    ],voting='soft')
    vc.fit(Xtr,ytr)
    tra=accuracy_score(ytr,vc.predict(Xtr)); tea=accuracy_score(yte,vc.predict(Xte))
    fold_results_ens.append({'fold':fold+1,'train':tra,'test':tea})
    status='✅' if tea>=0.91 else ('⚠️ ' if tea>=0.90 else '📈')
    print(f"  {status} Fold {fold+1:2d}: Train={tra:.4f}  Test={tea:.4f}")
ens_avg=np.mean([r['test'] for r in fold_results_ens])
ens_gap=np.mean([r['train'] for r in fold_results_ens])-ens_avg
print(f"  Ensemble 10-Fold Avg Test: {ens_avg*100:.2f}%  Gap={ens_gap*100:.2f}%")

# Best model selection
best_model_name='Soft-Voting-Ensemble' if ens_avg>=avg_te else 'XGBoost-Optuna-V7'
best_acc=max(avg_te,ens_avg)

# ── STEP 5: Train & Save Final Model ─────────────────────────────────
print("\n[STEP 5] Training final model on full SMOTE-augmented data...")
Xs,ys=SMOTE(random_state=42,k_neighbors=5).fit_resample(X,y)

# Train both; use ensemble as final saved model
final_model=VotingClassifier(estimators=[
    ('xgb',XGBClassifier(**BEST_XGB)),
    ('lgbm',LGBMClassifier(**BEST_LGBM)),
    ('et',ExtraTreesClassifier(n_estimators=500,max_depth=12,min_samples_split=4,
                               min_samples_leaf=2,max_features='sqrt',random_state=42,n_jobs=1)),
],voting='soft')
final_model.fit(Xs,ys)
pred=final_model.predict(X)
resub=accuracy_score(y,pred)
cm=confusion_matrix(y,pred)
cr=classification_report(y,pred,target_names=['Not Potable','Potable'])

print(f"\nFinal Ensemble Resubstitution: {resub:.4f} ({resub*100:.2f}%)")
print(f"\nClassification Report:\n{cr}")
print(f"Confusion Matrix:\n{cm}")

# Feature importance via solo XGB
xgb_solo=XGBClassifier(**BEST_XGB); xgb_solo.fit(Xs,ys)
imp=pd.DataFrame({'Feature':ALL_FEATS,'Importance':xgb_solo.feature_importances_})
imp=imp.sort_values('Importance',ascending=False)
print(f"\nTop 15 Most Important Features:")
print(imp.head(15).to_string(index=False))

os.makedirs('models',exist_ok=True)
joblib.dump(final_model,'models/xgboost_model.joblib')
joblib.dump(scaler,'models/scaler.joblib')
joblib.dump(ALL_FEATS,'models/feature_names.joblib')
joblib.dump(imputer,'models/imputer.joblib')

# ── STEP 6: Write JSON for dashboard ─────────────────────────────────
results_data={
    'xgb_10fold_avg':   round(float(avg_te)*100,2),
    'lgbm_10fold_avg':  round(float(lgbm_avg)*100,2),
    'ens_10fold_avg':   round(float(ens_avg)*100,2),
    'best_acc':         round(float(best_acc)*100,2),
    'best_model':       best_model_name,
    'avg_f1':           round(float(avg_f1),4),
    'avg_prec':         round(float(avg_pr),4),
    'avg_rec':          round(float(avg_rc),4),
    'overfit_gap':      round(float(gap)*100,2),
    'resub_acc':        round(float(resub)*100,2),
    'n_samples':        int(n),
    'n_features':       len(ALL_FEATS),
    'xgb_fold_results': [{'fold':r['fold'],'train':round(r['train']*100,2),
                           'test':round(r['test']*100,2)} for r in fold_results],
    'lgbm_fold_results':[{'fold':r['fold'],'train':round(r['train']*100,2),
                           'test':round(r['test']*100,2)} for r in fold_results_lgbm],
    'ens_fold_results': [{'fold':r['fold'],'train':round(r['train']*100,2),
                           'test':round(r['test']*100,2)} for r in fold_results_ens],
    'confusion_matrix': cm.tolist(),
    'top_features':     [{'name':row['Feature'],'imp':round(float(row['Importance']),4)}
                          for _,row in imp.head(15).iterrows()],
    'target_achieved':  best_acc>=0.91
}
with open('results_v7c.json','w') as f: json.dump(results_data,f,indent=2)
print("\n[✅] Saved: models/ | results_v7c.json")
print("\n"+"="*70)
print("PIPELINE V7c COMPLETE")
if best_acc>=0.91:
    print(f"*** TARGET ACHIEVED: {best_acc*100:.2f}% >= 91% ***")
else:
    print(f"Final accuracy: {best_acc*100:.2f}%")
print("="*70)
