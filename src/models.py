from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score

NUMERIC = ["Prorated Billed Amount (Excl GST)", "Prorated Debit Amount (Excl tax)", "Debit/Billed Ratio",
           "Previous Month Billing", "Site Historical Average", "Site Historical Std", "Rolling 3M Average",
           "MoM Billing Change %", "Difference From Site Average %", "Month", "Quarter"]
CATEGORICAL = ["Circle Code", "Vendor Code"]

def prepare_taxonomy(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy(); x.columns = [str(c).strip() for c in x.columns]
    if not {"Dispute Head", "Dispute Sub-Category"}.issubset(x.columns):
        raise ValueError("Dispute taxonomy requires Dispute Head and Dispute Sub-Category")
    x["Dispute Head"] = x["Dispute Head"].astype(str).str.strip().str.upper()
    x["Dispute Sub-Category"] = x["Dispute Sub-Category"].astype(str).str.strip()
    return x.drop_duplicates()

def valid_reasons(taxonomy, head):
    return taxonomy.loc[taxonomy["Dispute Head"].eq(str(head).upper()), "Dispute Sub-Category"].tolist()

@dataclass
class DisputeClassifier:
    threshold: float = 0.60
    models: dict = field(default_factory=dict)
    taxonomy: pd.DataFrame | None = None

    def _pipe(self, num, cat):
        tr=[]
        if num: tr.append(("n", Pipeline([("i",SimpleImputer(strategy="median"))]), num))
        if cat: tr.append(("c", Pipeline([("i",SimpleImputer(strategy="most_frequent")),("o",OneHotEncoder(handle_unknown="ignore"))]), cat))
        prep=ColumnTransformer(tr)
        clf=RandomForestClassifier(n_estimators=300,min_samples_leaf=2,class_weight="balanced_subsample",random_state=42,n_jobs=-1)
        return Pipeline([("prep",prep),("clf",clf)])

    def fit(self, df, taxonomy):
        if "Dispute Type" not in df: raise ValueError("Training data needs actual Dispute Type")
        self.taxonomy=prepare_taxonomy(taxonomy); self.models={}
        for head in ["EB","DG"]:
            allowed=set(valid_reasons(self.taxonomy, head))
            s=df[df["Expense Nature"].astype(str).str.upper().eq(head) & df["Dispute Type"].isin(allowed)].copy()
            if len(s)<8 or s["Dispute Type"].nunique()<2: continue
            num=[c for c in NUMERIC if c in s and pd.to_numeric(s[c],errors="coerce").notna().any()]
            cat=[c for c in CATEGORICAL if c in s and s[c].notna().any()]
            X=s[num+cat].copy(); pipe=self._pipe(num,cat); pipe.fit(X,s["Dispute Type"].astype(str))
            self.models[head]=(pipe,num,cat)
        if not self.models: raise ValueError("No trainable EB/DG classifier could be built")
        return self

    def predict(self, df):
        out=df.copy(); out["Predicted Dispute Type"]="Manual Review Required"; out["Prediction Confidence"]=np.nan
        out["Second Candidate"]=pd.NA; out["Second Candidate Confidence"]=np.nan; out["Prediction Status"]="No eligible model"
        for head,(pipe,num,cat) in self.models.items():
            mask=out["Expense Nature"].astype(str).str.upper().eq(head)
            if not mask.any(): continue
            X=out.loc[mask,num+cat]; p=pipe.predict_proba(X); classes=pipe.named_steps["clf"].classes_
            order=np.argsort(p,axis=1)[:,::-1]; top=order[:,0]; prob=p[np.arange(len(p)),top]
            pred=classes[top]; second=classes[order[:,1]] if p.shape[1]>1 else np.array([None]*len(p))
            second_p=p[np.arange(len(p)),order[:,1]] if p.shape[1]>1 else np.full(len(p),np.nan)
            ok=prob>=self.threshold
            out.loc[mask,"Predicted Dispute Type"]=np.where(ok,pred,"Manual Review Required")
            out.loc[mask,"Prediction Confidence"]=prob; out.loc[mask,"Second Candidate"]=second
            out.loc[mask,"Second Candidate Confidence"]=second_p; out.loc[mask,"Prediction Status"]=np.where(ok,"Auto-predicted","Low confidence")
        return out

def add_anomalies(df, contamination=0.10):
    out=df.copy(); candidates=[c for c in NUMERIC if c in out and pd.to_numeric(out[c],errors="coerce").notna().any()]
    use=[c for c in candidates if c not in {"Month","Quarter"}][:8]
    if not use:
        out["Is Anomaly"]=False; out["Anomaly Percentile"]=0.0; out["Anomaly Evidence"]="No numeric features"; return out
    X=out[use].apply(pd.to_numeric,errors="coerce").fillna(out[use].apply(pd.to_numeric,errors="coerce").median()).fillna(0)
    iso=IsolationForest(contamination=contamination,random_state=42).fit(X); score=-iso.score_samples(X)
    out["Is Anomaly"]=iso.predict(X).eq(-1) if isinstance(iso.predict(X),pd.Series) else iso.predict(X)==-1
    out["Anomaly Percentile"]=pd.Series(score,index=out.index).rank(pct=True)
    def evidence(r):
        parts=[]
        for c in ["MoM Billing Change %","Difference From Site Average %","Debit/Billed Ratio"]:
            if c in out and pd.notna(r.get(c)): parts.append(f"{c}: {r[c]:.1f}")
        return "; ".join(parts[:3]) or "Multivariate billing deviation"
    out["Anomaly Evidence"]=out.apply(evidence,axis=1)
    return out

def evaluate_time_split(df, taxonomy, threshold=0.60):
    months=pd.to_datetime(df["Month-Year"],errors="coerce")
    uniq=sorted(months.dropna().dt.to_period("M").unique())
    if len(uniq)<4: return {"error":"Need at least four months"},pd.DataFrame()
    cut=uniq[max(1,int(len(uniq)*0.75)-1)]; train=df[months.dt.to_period("M")<=cut]; test=df[months.dt.to_period("M")>cut]
    model=DisputeClassifier(threshold).fit(train,taxonomy); pred=model.predict(test)
    scored=pred[pred["Dispute Type"].notna() & pred["Predicted Dispute Type"].ne("Manual Review Required")]
    m={"train_rows":len(train),"test_rows":len(test),"coverage":len(scored)/max(1,test["Dispute Type"].notna().sum())}
    if not scored.empty:
        y=scored["Dispute Type"].astype(str); p=scored["Predicted Dispute Type"].astype(str)
        m.update(accuracy=accuracy_score(y,p),macro_f1=f1_score(y,p,average="macro",zero_division=0),weighted_f1=f1_score(y,p,average="weighted",zero_division=0))
    return m,pred
