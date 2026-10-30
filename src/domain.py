import numpy as np
import pandas as pd

LOW_PCT= 0.5
HIGH_PCT= 99.5

DOMAIN_AXES= ["bp_rp","mg"]

#Global envelope is the union of per-class envelopes
def domain_measure(train: pd.DataFrame,axes=None) -> dict:
    axes= axes or DOMAIN_AXES
    out= {"global": {},"per_class": {}}
    for cls,s in train.groupby("lum_class"):
        out["per_class"][cls]= {
            a: (float(np.percentile(s[a].dropna(),LOW_PCT)),
                float(np.percentile(s[a].dropna(),HIGH_PCT)))
            for a in axes }
    for a in axes:
        lows= [b[a][0] for b in out["per_class"].values()]
        highs= [b[a][1] for b in out["per_class"].values()]
        out["global"][a]= (float(min(lows)),float(max(highs)))
    return out

#True where a star falls inside the global envelope of the training set
def in_domain(df: pd.DataFrame,domain: dict,axes=None) -> pd.Series:
    axes= axes or DOMAIN_AXES
    keep= pd.Series(True,index=df.index)
    for a in axes:
        lo,hi= domain["global"][a]
        keep &= df[a].between(lo,hi)
    return keep

#Stricter envelope of assigned class,so it names the failure mode
def prediction_is_supported(df: pd.DataFrame,domain: dict,pred_col: str= "predicted",axes=None) -> pd.Series:
    axes= axes or DOMAIN_AXES
    ok= pd.Series(False,index=df.index)
    for cls,bounds in domain["per_class"].items():
        m= df[pred_col]== cls
        if not m.any():
            continue
        good= m.copy()
        for a in axes:
            lo,hi= bounds[a]
            good &= df[a].between(lo,hi)
        ok |= good
    return ok

def describe(domain: dict) -> None:
    print(f"\nTraining domain (percentiles {LOW_PCT}-{HIGH_PCT} of the labelled set):")
    for a,(lo,hi) in domain["global"].items():
        print(f"global {a:8s}: {lo:7.2f} to {hi:7.2f} per class")
    for cls,bounds in sorted(domain["per_class"].items()):
        parts= "".join(f"{a} {lo:6.2f}..{hi:6.2f}" for a,(lo,hi) in bounds.items())
        print(f"{cls:12s}: {parts}")
