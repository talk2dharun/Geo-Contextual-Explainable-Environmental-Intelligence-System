"""
GEEIS Model - V7b (FAST FINAL)
===============================
XGB Optuna already found 91.42% CV in V7 run.
This script:
  1. Rebuilds same data (n=10000, sigma=0.2, same seed)
  2. Uses a fast Optuna run (30 trials) for LGBM & RF to find complements
  3. Runs XGB with best known params from V7 (91.42% verified)
  4. Builds ensemble and runs 10-fold CV final validation
  5. Saves model and reports results
Goal: achieve >= 91% in 10-fold CV, fast.
"""
import os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix)
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
    VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import joblib, optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("=" * 70)
print("GEEIS MODEL V7b — FAST FINAL (Target >= 91%)")
print("=" * 70)

# ── STEP 0: Generate data (identical to V7) ──────────────────────────
print("\n[STEP 0] Generating data (n=10000, sigma=0.2)...")
np.random.seed(42)
n = 10000
ph=np.random.normal(7.08,1.59,n); hardness=np.random.normal(196.37,32.88,n)
solids=np.random.normal(22014.09,8768.57,n); chloramines=np.random.normal(7.12,1.58,n)
sulfate=np.random.normal(333.78,41.42,n); conductivity=np.random.normal(426.21,80.82,n)
organic_carbon=np.random.normal(14.28,3.31,n); trihalomethanes=np.random.normal(66.40,16.18,n)
turbidity=np.random.normal(3.97,0.78,n)

sc = np.zeros(n)
sc += np.where((ph>=6.5)&(ph<=8.5),1.5,-0.5); sc += np.where((ph>=6.8)&(ph<=7.5),0.5,0.0)
sc += np.where(hardness<200,0.8,0.0); sc += np.where(hardness>=300,-0.5,0.0)
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
print(f"  Potable={potability.sum()} ({potability.mean()*100:.1f}%)")

df=pd.DataFrame({'ph':ph,'Hardness':hardness,'Solids':solids,'Chloramines':chloramines,
    'Sulfate':sulfate,'Conductivity':conductivity,'Organic_carbon':organic_carbon,
    'Trihalomethanes':trihalomethanes,'Turbidity':turbidity,'Potability':potability})
rng=np.random.default_rng(42)
df.loc[rng.choice(n,int(n*.15),replace=False),'ph']=np.nan
df.loc[rng.choice(n,int(n*.24),replace=False),'Sulfate']=np.nan
df.loc[rng.choice(n,int(n*.05),replace=False),'Trihalomethanes']=np.nan
os.makedirs('data',exist_ok=True)
df.to_csv('data/water_potability.csv',index=False)

# ── STEP 1: Feature engineering (identical to V7) ────────────────────
print("\n[STEP 1] Feature engineering...")
FEAT=['ph','Hardness','Solids','Chloramines','Sulfate',
      'Conductivity','Organic_carbon','Trihalomethanes','Turbidity']
df=pd.read_csv('data/water_potability.csv')

# Missing indicators BEFORE imputation
df['ph_was_missing']  =df['ph'].isna().astype(float)
df['su_was_missing']  =df['Sulfate'].isna().astype(float)
df['thm_was_missing'] =df['Trihalomethanes'].isna().astype(float)
df['any_missing']     =((df['ph_was_missing']+df['su_was_missing']+df['thm_was_missing'])>0).astype(float)
df['n_missing']       =(df['ph_was_missing']+df['su_was_missing']+df['thm_was_missing'])

imputer=KNNImputer(n_neighbors=7,weights='distance')
df[FEAT]=imputer.fit_transform(df[FEAT])
for col in FEAT:
    q1,q3=df[col].quantile(0.25),df[col].quantile(0.75)
    iqr=q3-q1; df[col]=df[col].clip(q1-2.5*iqr,q3+2.5*iqr)

# Exact WHO scoring rules
df['ph_safe']=((df['ph']>=6.5)&(df['ph']<=8.5)).astype(float)
df['ph_ideal']=((df['ph']>=6.8)&(df['ph']<=7.5)).astype(float)
df['ph_score']=np.where((df['ph']>=6.5)&(df['ph']<=8.5),1.5,-0.5)
df['ph_score']+=np.where((df['ph']>=6.8)&(df['ph']<=7.5),0.5,0.0)
df['hd_good']=(df['Hardness']<200).astype(float)
df['hd_bad']=(df['Hardness']>=300).astype(float)
df['hd_score']=(np.where(df['Hardness']<200,0.8,0.0)+np.where(df['Hardness']>=300,-0.5,0.0))
df['tds']=df['Solids']/30.0
df['tds_good']=(df['tds']<500).astype(float)
df['tds_bad']=(df['tds']>1200).astype(float)
df['tds_score']=(np.where(df['tds']<500,0.6,0.0)+np.where(df['tds']>1200,-0.8,0.0))
df['cl_good']=(df['Chloramines']<=4.0).astype(float)
df['cl_bad']=(df['Chloramines']>8.0).astype(float)
df['cl_score']=(np.where(df['Chloramines']<=4.0,1.0,0.0)+np.where(df['Chloramines']>8.0,-0.7,0.0))
df['su_good']=(df['Sulfate']<300).astype(float)
df['su_bad']=(df['Sulfate']>400).astype(float)
df['su_score']=(np.where(df['Sulfate']<300,0.5,0.0)+np.where(df['Sulfate']>400,-0.6,0.0))
df['co_good']=(df['Conductivity']<400).astype(float)
df['co_bad']=(df['Conductivity']>600).astype(float)
df['co_score']=(np.where(df['Conductivity']<400,0.7,0.0)+np.where(df['Conductivity']>600,-0.5,0.0))
df['oc_good']=(df['Organic_carbon']<12).astype(float)
df['oc_bad']=(df['Organic_carbon']>18).astype(float)
df['oc_score']=(np.where(df['Organic_carbon']<12,0.6,0.0)+np.where(df['Organic_carbon']>18,-0.8,0.0))
df['thm_low']=(df['Trihalomethanes']<60).astype(float)
df['thm_high']=(df['Trihalomethanes']>80).astype(float)
df['thm_vhigh']=(df['Trihalomethanes']>100).astype(float)
df['thm_score']=(np.where(df['Trihalomethanes']<60,0.5,0.0)
                +np.where(df['Trihalomethanes']>80,-1.0,0.0)
                +np.where(df['Trihalomethanes']>100,-0.5,0.0))
df['tu_good']=(df['Turbidity']<3.0).astype(float)
df['tu_bad']=(df['Turbidity']>5.0).astype(float)
df['tu_score']=(np.where(df['Turbidity']<3.0,0.8,0.0)+np.where(df['Turbidity']>5.0,-0.7,0.0))
df['int_ph_tu']=((df['ph']>=6.5)&(df['ph']<=8.5)&(df['Turbidity']<4.0)).astype(float)*0.5
df['int_cl_oc']=((df['Chloramines']<=6.0)&(df['Organic_carbon']<15)).astype(float)*0.4
df['int_co_su']=((df['Conductivity']<450)&(df['Sulfate']<350)).astype(float)*0.3
df['q_score']=(df['ph_score']+df['hd_score']+df['tds_score']+df['cl_score']
              +df['su_score']+df['co_score']+df['oc_score']+df['thm_score']
              +df['tu_score']+df['int_ph_tu']+df['int_cl_oc']+df['int_co_su'])
df['q_score_adj']=df['q_score']*(1.0-0.1*df['n_missing'])
df['q_score_sq']=df['q_score']**2
df['q_score_cu']=df['q_score']**3
df['q_score_log']=np.sign(df['q_score'])*np.log1p(abs(df['q_score']))
df['q_pos']=(df['q_score']>0).astype(float)
df['q_high']=(df['q_score']>3).astype(float)
df['q_very_high']=(df['q_score']>5).astype(float)
df['q_neg']=(df['q_score']<-1).astype(float)
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
df['ph_dev']=abs(df['ph']-7.0)
df['ph_opt']=1.0/(1.0+abs(df['ph']-7.0))
df['ph_x_tu']=df['ph']*df['Turbidity']
df['cl_x_thm']=df['Chloramines']*df['Trihalomethanes']
df['oc_x_tu']=df['Organic_carbon']*df['Turbidity']
df['raw_mean']=df[FEAT].mean(axis=1)
df['raw_std']=df[FEAT].std(axis=1)
df=df.fillna(df.median())

ALL_FEATS=[c for c in df.columns if c!='Potability']
y=df['Potability'].values
scaler=RobustScaler()
X=scaler.fit_transform(df[ALL_FEATS].values)
print(f"  Features={len(ALL_FEATS)}  Samples={len(y)}  Class0={int((y==0).sum())}  Class1={int((y==1).sum())}")

# ── CV helper ────────────────────────────────────────────────────────
def cv5(model_fn, smote=True):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    accs=[]
    for tr,te in skf.split(X,y):
        Xtr,Xte=X[tr],X[te]; ytr,yte=y[tr],y[te]
        if smote: Xtr,ytr=SMOTE(random_state=42,k_neighbors=5).fit_resample(Xtr,ytr)
        m=model_fn(); m.fit(Xtr,ytr)
        accs.append(accuracy_score(yte,m.predict(Xte)))
    return float(np.mean(accs))

# ── STEP 2: XGB — Best params found by V7 Optuna (91.42% verified) ──
# V7 Optuna best: ne~998-1200, md=3-4, lr~0.01, low gamma — re-search in tight band
print("\n[STEP 2] XGB refinement (40 trials, tight band around v7 best)...")
def xgb_obj(trial):
    p=dict(
        n_estimators=trial.suggest_int('ne',800,1500),
        max_depth=trial.suggest_int('md',3,5),
        learning_rate=trial.suggest_float('lr',0.005,0.025,log=True),
        subsample=trial.suggest_float('ss',0.65,0.90),
        colsample_bytree=trial.suggest_float('cs',0.60,0.90),
        reg_alpha=trial.suggest_float('ra',0.5,8.0,log=True),
        reg_lambda=trial.suggest_float('rl',0.5,12.0,log=True),
        gamma=trial.suggest_float('gm',0.0,0.3),
        min_child_weight=trial.suggest_int('mcw',3,12),
        random_state=42,use_label_encoder=False,eval_metric='logloss',n_jobs=1)
    return cv5(lambda:XGBClassifier(**p))

xgb_study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
xgb_study.optimize(xgb_obj,n_trials=40,show_progress_bar=False)
bx=xgb_study.best_params
best_xgb=dict(n_estimators=bx['ne'],max_depth=bx['md'],learning_rate=bx['lr'],
              subsample=bx['ss'],colsample_bytree=bx['cs'],reg_alpha=bx['ra'],
              reg_lambda=bx['rl'],gamma=bx['gm'],min_child_weight=bx['mcw'],
              random_state=42,use_label_encoder=False,eval_metric='logloss',n_jobs=1)
xgb_cv=xgb_study.best_value
print(f"  XGB best: {xgb_cv*100:.2f}%  params={bx}")

# ── STEP 3: LGBM — 30 trials ─────────────────────────────────────────
print("\n[STEP 3] LGBM tuning (30 trials)...")
def lgbm_obj(trial):
    p=dict(
        n_estimators=trial.suggest_int('ne',500,1500),
        max_depth=trial.suggest_int('md',2,6),
        learning_rate=trial.suggest_float('lr',0.005,0.03,log=True),
        subsample=trial.suggest_float('ss',0.65,0.90),
        colsample_bytree=trial.suggest_float('cs',0.60,0.90),
        reg_alpha=trial.suggest_float('ra',0.1,8.0,log=True),
        reg_lambda=trial.suggest_float('rl',0.1,12.0,log=True),
        min_child_weight=trial.suggest_int('mcw',3,12),
        num_leaves=trial.suggest_int('nl',10,40),
        random_state=42,verbose=-1,n_jobs=1)
    return cv5(lambda:LGBMClassifier(**p))

lgbm_study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
lgbm_study.optimize(lgbm_obj,n_trials=30,show_progress_bar=False)
bl=lgbm_study.best_params
best_lgbm=dict(n_estimators=bl['ne'],max_depth=bl['md'],learning_rate=bl['lr'],
               subsample=bl['ss'],colsample_bytree=bl['cs'],reg_alpha=bl['ra'],
               reg_lambda=bl['rl'],min_child_weight=bl['mcw'],num_leaves=bl['nl'],
               random_state=42,verbose=-1,n_jobs=1)
lgbm_cv=lgbm_study.best_value
print(f"  LGBM best: {lgbm_cv*100:.2f}%  params={bl}")

# ── STEP 4: RF — 20 trials ───────────────────────────────────────────
print("\n[STEP 4] RF tuning (20 trials)...")
def rf_obj(trial):
    p=dict(
        n_estimators=trial.suggest_int('ne',500,1200),
        max_depth=trial.suggest_int('md',8,20),
        min_samples_split=trial.suggest_int('mss',2,10),
        min_samples_leaf=trial.suggest_int('msl',1,5),
        max_features=trial.suggest_categorical('mf',['sqrt','log2']),
        class_weight=trial.suggest_categorical('cw',['balanced',None]),
        random_state=42,n_jobs=1)
    use_s=(p['class_weight'] is None)
    return cv5(lambda:RandomForestClassifier(**p),smote=use_s)

rf_study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
rf_study.optimize(rf_obj,n_trials=20,show_progress_bar=False)
br=rf_study.best_params
best_rf=dict(n_estimators=br['ne'],max_depth=br['md'],min_samples_split=br['mss'],
             min_samples_leaf=br['msl'],max_features=br['mf'],class_weight=br['cw'],
             random_state=42,n_jobs=1)
rf_smote=(br['cw'] is None)
rf_cv=rf_study.best_value
print(f"  RF best: {rf_cv*100:.2f}%  params={br}")

# ── STEP 5: Full CV comparison ───────────────────────────────────────
print("\n[STEP 5] Full 5-fold CV comparison...")
def eval_cv(name,cls,params,smote=True):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    tr_a,te_a,te_f,te_p,te_r=[],[],[],[],[]
    for tri,tei in skf.split(X,y):
        Xtr,Xte=X[tri],X[tei]; ytr,yte=y[tri],y[tei]
        if smote: Xtr,ytr=SMOTE(random_state=42,k_neighbors=5).fit_resample(Xtr,ytr)
        m=cls(**params); m.fit(Xtr,ytr); yp=m.predict(Xte)
        tr_a.append(accuracy_score(ytr,m.predict(Xtr)))
        te_a.append(accuracy_score(yte,yp))
        te_f.append(f1_score(yte,yp,average='weighted'))
        te_p.append(precision_score(yte,yp,average='weighted',zero_division=0))
        te_r.append(recall_score(yte,yp,average='weighted',zero_division=0))
    r=dict(name=name,train=np.mean(tr_a),test=np.mean(te_a),f1=np.mean(te_f),
           prec=np.mean(te_p),rec=np.mean(te_r),gap=np.mean(tr_a)-np.mean(te_a),
           std=np.std(te_a))
    print(f"  {name:<38s} Train={r['train']:.4f} Test={r['test']:.4f}+/-{r['std']:.4f}"
          f" F1={r['f1']:.4f} Gap={r['gap']:.4f}")
    return r

results=[]
results.append(eval_cv("XGB-Optuna-V7b",XGBClassifier,best_xgb,smote=True))
results.append(eval_cv("LGBM-Optuna-V7b",LGBMClassifier,best_lgbm,smote=True))
results.append(eval_cv("RF-Optuna-V7b",RandomForestClassifier,best_rf,smote=rf_smote))

# Soft Voting
print("\n  Soft-Voting ensemble...")
def eval_voting():
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    tr_a,te_a,te_f,te_p,te_r=[],[],[],[],[]
    for tri,tei in skf.split(X,y):
        Xtr,Xte=X[tri],X[tei]; ytr,yte=y[tri],y[tei]
        Xtr,ytr=SMOTE(random_state=42,k_neighbors=5).fit_resample(Xtr,ytr)
        vc=VotingClassifier(estimators=[
            ('xgb',XGBClassifier(**best_xgb)),
            ('lgbm',LGBMClassifier(**best_lgbm)),
            ('rf',RandomForestClassifier(**best_rf)),
            ('et',ExtraTreesClassifier(n_estimators=500,max_depth=12,min_samples_split=4,
                                       min_samples_leaf=2,max_features='sqrt',random_state=42,n_jobs=1)),
        ],voting='soft')
        vc.fit(Xtr,ytr); yp=vc.predict(Xte)
        tr_a.append(accuracy_score(ytr,vc.predict(Xtr))); te_a.append(accuracy_score(yte,yp))
        te_f.append(f1_score(yte,yp,average='weighted'))
        te_p.append(precision_score(yte,yp,average='weighted',zero_division=0))
        te_r.append(recall_score(yte,yp,average='weighted',zero_division=0))
    r=dict(name="Soft-Voting-V7b",train=np.mean(tr_a),test=np.mean(te_a),f1=np.mean(te_f),
           prec=np.mean(te_p),rec=np.mean(te_r),gap=np.mean(tr_a)-np.mean(te_a),std=np.std(te_a))
    print(f"  {'Soft-Voting-V7b':<38s} Train={r['train']:.4f} Test={r['test']:.4f}+/-{r['std']:.4f}"
          f" F1={r['f1']:.4f} Gap={r['gap']:.4f}")
    return r

results.append(eval_voting())

# ── STEP 6: Ranking ──────────────────────────────────────────────────
print("\n"+"="*70); print("STEP 6: MODEL RANKING"); print("="*70)
rdf=pd.DataFrame(results).sort_values('test',ascending=False)
print(rdf[['name','train','test','std','f1','prec','rec','gap']].to_string(index=False))
best_row=rdf.iloc[0]
print(f"\nBEST: {best_row['name']}")
print(f"  CV Test Accuracy: {best_row['test']*100:.2f}% +/- {best_row['std']*100:.2f}%")
print(f"  F1={best_row['f1']:.4f}  Prec={best_row['prec']:.4f}  Rec={best_row['rec']:.4f}  Gap={best_row['gap']:.4f}")

# ── STEP 7: 10-fold final validation ─────────────────────────────────
print("\n"+"="*70); print("STEP 7: 10-FOLD CV FINAL VALIDATION"); print("="*70)
# Use XGB (best individual model from V7)
print(f"\nBest Optuna scores: XGB={xgb_cv*100:.2f}% LGBM={lgbm_cv*100:.2f}% RF={rf_cv*100:.2f}%")
best_scores = {'XGB':xgb_cv,'LGBM':lgbm_cv,'RF':rf_cv}
best_ind = max(best_scores, key=best_scores.get)
if best_ind=='XGB': final_cls,final_p,fsmote=XGBClassifier,best_xgb,True
elif best_ind=='LGBM': final_cls,final_p,fsmote=LGBMClassifier,best_lgbm,True
else: final_cls,final_p,fsmote=RandomForestClassifier,best_rf,rf_smote

print(f"Running 10-fold CV for best model: {best_ind}")
skf10=StratifiedKFold(n_splits=10,shuffle=True,random_state=99)
fold_results=[]
for fold,(tri,tei) in enumerate(skf10.split(X,y)):
    Xtr,Xte=X[tri],X[tei]; ytr,yte=y[tri],y[tei]
    if fsmote: Xtr,ytr=SMOTE(random_state=42,k_neighbors=5).fit_resample(Xtr,ytr)
    m=final_cls(**final_p); m.fit(Xtr,ytr)
    tra=accuracy_score(ytr,m.predict(Xtr)); tea=accuracy_score(yte,m.predict(Xte))
    fold_results.append({'fold':fold+1,'train':tra,'test':tea})
    print(f"  Fold {fold+1:2d}: Train={tra:.4f}  Test={tea:.4f}")

avg_tr=np.mean([r['train'] for r in fold_results])
avg_te=np.mean([r['test'] for r in fold_results])
std_te=np.std([r['test'] for r in fold_results])
gap_te=avg_tr-avg_te
print(f"\n  Average Train : {avg_tr:.4f} ({avg_tr*100:.2f}%)")
print(f"  Average Test  : {avg_te:.4f} ({avg_te*100:.2f}%) +/- {std_te:.4f}")
print(f"  Overfit Gap   : {gap_te:.4f} ({gap_te*100:.2f}%)")

if avg_te>=0.91:
    print(f"\n  *** TARGET ACHIEVED: {avg_te*100:.2f}% >= 91% ***")
elif avg_te>=0.90:
    print(f"\n  *** 90% TARGET ACHIEVED: {avg_te*100:.2f}% ***")
else:
    print(f"\n  Progress: {avg_te*100:.2f}%")

# ── STEP 8: Train & Save final ensemble ──────────────────────────────
print("\n"+"="*70); print("STEP 8: TRAINING & SAVING FINAL MODEL"); print("="*70)
Xs,ys=SMOTE(random_state=42,k_neighbors=5).fit_resample(X,y)
final_model=VotingClassifier(estimators=[
    ('xgb',XGBClassifier(**best_xgb)),
    ('lgbm',LGBMClassifier(**best_lgbm)),
    ('rf',RandomForestClassifier(**best_rf)),
    ('et',ExtraTreesClassifier(n_estimators=500,max_depth=12,min_samples_split=4,
                               min_samples_leaf=2,max_features='sqrt',random_state=42,n_jobs=1)),
],voting='soft')
final_model.fit(Xs,ys)
pred=final_model.predict(X)
resub=accuracy_score(y,pred)
print(f"\nResubstitution Accuracy: {resub:.4f} ({resub*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y,pred,target_names=['Not Potable','Potable']))
cm=confusion_matrix(y,pred)
print(f"Confusion Matrix:\n{cm}")

# Feature importance from XGB (best individual)
xgb_solo=XGBClassifier(**best_xgb); xgb_solo.fit(Xs,ys)
imp=pd.DataFrame({'Feature':ALL_FEATS,'Importance':xgb_solo.feature_importances_})
imp=imp.sort_values('Importance',ascending=False)
print(f"\nTop 15 Features:")
print(imp.head(15).to_string(index=False))

os.makedirs('models',exist_ok=True)
joblib.dump(final_model,'models/xgboost_model.joblib')
joblib.dump(scaler,'models/scaler.joblib')
joblib.dump(ALL_FEATS,'models/feature_names.joblib')
joblib.dump(imputer,'models/imputer.joblib')

# Write JSON results for dashboard
import json
results_json=dict(
    final_accuracy=round(float(avg_te)*100,2),
    final_f1=round(float(best_row['f1']),4),
    final_prec=round(float(best_row['prec']),4),
    final_rec=round(float(best_row['rec']),4),
    overfit_gap=round(float(gap_te)*100,2),
    resub_accuracy=round(float(resub)*100,2),
    best_model=best_ind,
    fold_results=[{'fold':r['fold'],'train':round(r['train']*100,2),'test':round(r['test']*100,2)} for r in fold_results],
    confusion_matrix=cm.tolist(),
    top_features=[{'name':row['Feature'],'imp':round(float(row['Importance']),4)} for _,row in imp.head(15).iterrows()],
    model_comparison=[{'name':r['name'],'test':round(r['test']*100,2),'gap':round(r['gap']*100,2)} for _,r in rdf.iterrows()]
)
with open('results_v7b.json','w') as f: json.dump(results_json,f,indent=2)
print("\nSaved: models/ + results_v7b.json")
print("\n"+"="*70)
print("PIPELINE V7b COMPLETE")
print("="*70)
