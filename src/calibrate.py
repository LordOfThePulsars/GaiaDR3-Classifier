import pickle
import sys

from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (brier_score_loss,classification_report,f1_score,log_loss)
from sklearn.model_selection import GroupKFold

import config
from train_model import RANDOM_STATE,TRAINABLE,make_model,sky_blocks

FEATURES_IN= config.DATA_PROCESSED/"features.parquet"
OOF_OUT= config.DATA_PROCESSED/"oof_predictions.parquet"
MODEL_OUT= config.DATA_PROCESSED/"calibrated_model.pkl"
REL_FIG= config.FIGURES/"reliability_diagram.png"

N_FOLDS= 4
CALIB_FRACTION= 0.25

#sklearn 1.6 replaced with Frozen estimator
def prefit_calibrator_fit(base,X,y):
    try:
        from sklearn.frozen import FrozenEstimator
        calib= CalibratedClassifierCV(FrozenEstimator(base),method="isotonic")
    except ImportError:  #sklearn < 1.6
        calib= CalibratedClassifierCV(base,method="isotonic",cv="prefit")
    calib.fit(X,y)
    return calib

def out_of_fold(df: pd.DataFrame,classes: list[str]):
    X= df[config.FEATURES].values
    y= df.lum_class.values
    groups= sky_blocks(df.source_id).values

    n,k= len(df),len(classes)
    raw= np.zeros((n,k))
    cal= np.zeros((n,k))

    gkf= GroupKFold(n_splits=N_FOLDS)
    for i,(tr,te) in enumerate(gkf.split(X,y,groups),start=1):

        #Nested patch splits, calibrator never sees main model batches
        inner_groups= groups[tr]
        uniq= np.unique(inner_groups)
        rng= np.random.default_rng(RANDOM_STATE + i)
        rng.shuffle(uniq)
        n_cal= max(1,int(CALIB_FRACTION * len(uniq)))
        cal_patches= set(uniq[:n_cal])
        is_cal= np.array([g in cal_patches for g in inner_groups])

        fit_idx,cal_idx= tr[~is_cal],tr[is_cal]

        base= make_model().fit(X[fit_idx],y[fit_idx])
        raw[te]= base.predict_proba(X[te])

        calib= prefit_calibrator_fit(base,X[cal_idx],y[cal_idx])
        cal[te]= calib.predict_proba(X[te])

        print(f"fold {i}/{N_FOLDS}: fit {len(fit_idx):,} | calib {len(cal_idx):,} | predict {len(te):,}")

    return raw,cal

#Equal-count bins, width piles on top
def reliability(y_true_bin,prob,n_bins=12):
    edges= np.quantile(prob,np.linspace(0,1,n_bins + 1))
    edges= np.unique(edges)
    centres,observed,counts= [],[],[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        is_last= hi== edges[-1]
        m= (prob >= lo) & (prob <= hi if is_last else prob < hi)
        if m.sum() < 20:
            continue
        centres.append(prob[m].mean())
        observed.append(y_true_bin[m].mean())
        counts.append(int(m.sum()))
    return np.array(centres),np.array(observed),np.array(counts)

def reliability_plot(y,raw,cal,classes,path):
    fig,axes= plt.subplots(1,2,figsize=(13,5.8))

    for ax,prob,title in ((axes[0],raw,"Raw LightGBM"),(axes[1],cal,"Isotonic calibrated")):
        ax.plot([0,1],[0,1],color="#888888",ls="--",lw=1.2,label="perfect calibration")
        for ci,cls in enumerate(classes):
            binary= (y== cls).astype(float)
            c,o,_= reliability(binary,prob[:,ci])
            if len(c):
                ax.plot(c,o,"o-",ms=5,lw=1.4,label=cls)
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Observed Frequency")
        ax.set_title(title)
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.legend(fontsize=8,loc="upper left")

    fig.suptitle("Reliability diagram")
    fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

#Brier score and log loss put a number on the reliability diagram
def scores(name,y,prob,classes):
    briers= []
    for ci,cls in enumerate(classes):
        briers.append(brier_score_loss((y== cls).astype(int),prob[:,ci]))
    ll= log_loss(y,prob,labels=classes)
    print(f"{name:22s}  mean Brier {np.mean(briers):.5f}   log loss {ll:.5f}")
    return np.mean(briers),ll

def main() -> None:
    print("out-of-fold calibration")

    if not FEATURES_IN.exists():
        raise SystemExit(f"Missing {FEATURES_IN}. Run the feature engineering phase.")
    df= pd.read_parquet(FEATURES_IN)
    df= df[df.lum_class.isin(TRAINABLE)].dropna(subset=config.FEATURES).copy()
    df= df.reset_index(drop=True)
    classes= sorted(df.lum_class.unique())
    print(f"{len(df):,} labelled stars,classes {classes}")
    print(f"{sky_blocks(df.source_id).nunique()} distinct sky patches\n")

    print(f"Running {N_FOLDS}-fold group-aware cross-validation")
    raw,cal= out_of_fold(df,classes)
    y= df.lum_class.values
    pred_raw= np.array(classes)[raw.argmax(axis=1)]
    pred_cal= np.array(classes)[cal.argmax(axis=1)]

    print("\nout-of-fold performance")
    print(classification_report(y,pred_cal,labels=classes,zero_division=0,digits=3))
    m= np.isin(y,["dwarf","evolved"])
    sub_f1= f1_score(y[m],pred_cal[m],labels=["dwarf","evolved"],average="macro",zero_division=0)
    print(f"Macro F1,dwarf vs evolved: {sub_f1:.3f}")

    print("\ncalibration quality:")
    scores("raw",y,raw,classes)
    scores("calibrated",y,cal,classes)

    reliability_plot(y,raw,cal,classes,REL_FIG)

    conf= cal.max(axis=1)
    wrong= pred_cal != y
    print("\nanomaly pool by confidence:")
    de= m & wrong
    for thr in (0.0,0.7,0.8,0.9,0.95,0.99):
        n= int((de & (conf > thr)).sum())
        print(f"Confidence > {thr:.2f} : {n:>5,} misclassified")

    n90= int((de & (conf > 0.90)).sum())
    print(f"\nanomalies at p>0.90: {n90:,}")

    out= df[["source_id","bp_rp","mg","log_v_tan","lum_class"]].copy()
    out["predicted"]= pred_cal
    out["predicted_raw"]= pred_raw
    out["confidence"]= conf
    out["confidence_raw"]= raw.max(axis=1)
    for ci,cls in enumerate(classes):
        out[f"p_{cls}"]= cal[:,ci]
    out["is_wrong"]= wrong
    out["sky_patch"]= sky_blocks(df.source_id).values
    for c in config.DIAGNOSTICS:
        if c in df.columns:
            out[c]= df[c].values
    for c in ("teff_gspspec","logg_cal","delta_logg","v_tan","parallax","ra","dec","phot_g_mean_mag"):
        if c in df.columns:
            out[c]= df[c].values
    out.to_parquet(OOF_OUT,index=False)
    print(f"\nWrote {OOF_OUT}  ({len(out):,} rows)")

    X,groups= df[config.FEATURES].values,sky_blocks(df.source_id).values
    uniq= np.unique(groups)
    rng= np.random.default_rng(RANDOM_STATE)
    rng.shuffle(uniq)
    cal_patches= set(uniq[:max(1,int(CALIB_FRACTION * len(uniq)))])
    is_cal= np.array([g in cal_patches for g in groups])

    base= make_model().fit(X[~is_cal],y[~is_cal])
    final= prefit_calibrator_fit(base,X[is_cal],y[is_cal])

    with open(MODEL_OUT,"wb") as fh:
        pickle.dump({"model": final,"features": config.FEATURES,"classes": classes},fh)
    print(f"Wrote {MODEL_OUT}")

    config.log_provenance(
        "calibration",
        rows=len(out), n_folds=N_FOLDS,
        method="isotonic,calibrator fitted onsky patches",split="GroupKFold on HEALPix level-3 patches from source_id",
        anomalies_at_0p90=n90,classes=classes,)

if __name__== "__main__":
    main()
