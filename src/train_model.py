import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pickle
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,balanced_accuracy_score, f1_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config

FEATURES_IN= config.DATA_PROCESSED/"features.parquet"
MODEL_OUT= config.DATA_PROCESSED/"model.pkl"
TEST_OUT= config.DATA_PROCESSED/"test_predictions.parquet"
CM_FIG= config.FIGURES/"confusion_matrix.png"
IMP_FIG= config.FIGURES/"feature_importance.png"

TRAINABLE= ["dwarf", "evolved", "white_dwarf"]
TEST_FRACTION= 0.25
HEALPIX_DIVISOR= 2**53
RANDOM_STATE= 42

EXTRAPOLATION_COLOUR= 1.7

#Imported by calibrate so the calibrated model matches the evaluated one
def make_model():
    return LGBMClassifier(
        n_estimators=600, learning_rate=0.05, num_leaves=31,
        min_child_samples=40, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.9, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,)

#source_id encodes position, this gives the HEALPIX patch
def sky_blocks(source_id: pd.Series) -> pd.Series:
    return source_id // HEALPIX_DIVISOR

#Split on patches: neighbouring stars share extinction, so a random split leaks.
def split(df: pd.DataFrame):
    groups= sky_blocks(df.source_id)
    gss= GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION,random_state=RANDOM_STATE)
    tr,te= next(gss.split(df, df.lum_class, groups))
    train,test= df.iloc[tr], df.iloc[te]
    n_tr= groups.iloc[tr].nunique()
    n_te= groups.iloc[te].nunique()
    overlap= set(groups.iloc[tr]) & set(groups.iloc[te])
    print(f"training {len(train):,} stars in {n_tr} sky patches")
    print(f"testing  {len(test):,} stars in {n_te} sky patches")
    print(f"Amount of Overlap: {len(overlap)}")
    if overlap:
        raise SystemExit("Sky patches appear on both sets, leakage present!!.")
    return train, test

def report(name, y_true, y_pred, labels):
    print(f"\n {name}")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=3))
    print(f"balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.3f}")
    print(f"macro-F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.3f}")
    sub_f1= np.nan
    m= np.isin(y_true,["dwarf", "evolved"])
    if m.sum():
        sub_f1= f1_score(y_true[m], y_pred[m], labels=["dwarf", "evolved"], average="macro", zero_division=0)
        print(f"Dwarf vs evolved only: {sub_f1:.3f} with respect to Macro-F1")
    return sub_f1

def confusion_plot(y_true, y_pred, labels, path):
    cm= confusion_matrix(y_true, y_pred, labels=labels)
    cmn= cm / cm.sum(axis=1, keepdims=True)
    fig, ax= plt.subplots(figsize=(6.6, 5.6))
    im= ax.imshow(cmn, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=20)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cmn[i,j]:.3f}\n({cm[i,j]:,})", ha="center",va="center", fontsize=8,color="white" if cmn[i, j] < 0.5 else "black")
    fig.colorbar(im, ax=ax, label="True Class Fraction")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

def importance_plot(model, features, path):
    imp= model.feature_importances_
    order= np.argsort(imp)
    fig, ax= plt.subplots(figsize=(7, 4))
    ax.barh([features[i] for i in order], imp[order], color="#8b7ff0")
    ax.set_xlabel("Gain")
    ax.set_title("Feature Importance")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

def colour_breakdown(test, y_pred):
    print(f"\nPerformance by colour (extrapolation beyond BP-RP= {EXTRAPOLATION_COLOUR}):")
    edges= [-1.0, 0.5, 1.0, 1.3, EXTRAPOLATION_COLOUR, 10.0]
    for lo,hi in zip(edges[:-1], edges[1:]):
        m= test.bp_rp.between(lo, hi).values
        if m.sum() < 50:
            continue
        yt,yp= test.lum_class.values[m], y_pred[m]
        sub= np.isin(yt, ["dwarf", "evolved"])
        f1= (f1_score(yt[sub], yp[sub], labels=["dwarf", "evolved"],average="macro", zero_division=0) if sub.sum() > 20 else np.nan)
        tag= "  extrapolation" if lo >= EXTRAPOLATION_COLOUR else ""
        print(f"BP-RP {lo:5.1f} to {hi:5.1f}: n={int(m.sum()):>6,} dwarf/evolved F1= {f1:.3f}{tag}")

def main() -> None:
    if not FEATURES_IN.exists():
        raise SystemExit(f"Missing {FEATURES_IN}. Run Phase 3 first.")
    df= pd.read_parquet(FEATURES_IN)
    df= df[df.lum_class.isin(TRAINABLE)].copy()
    df= df.dropna(subset=config.FEATURES)
    print(f"{len(df):,} labelled stars with complete features.")
    print(f"Features: {config.FEATURES}")

    labels= sorted(df.lum_class.unique())
    print(f"Classes: {dict(df.lum_class.value_counts())}")
    train, test= split(df)

    Xtr, ytr= train[config.FEATURES].values, train.lum_class.values
    Xte, yte= test[config.FEATURES].values, test.lum_class.values

    dummy= DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    report("Majority-class baseline", yte, dummy.predict(Xte), labels)

    simple= ["bp_rp", "mg"]
    lr= make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,class_weight="balanced"))
    lr.fit(train[simple].values, ytr)
    lr_f1= report("Logistic regression", yte,lr.predict(test[simple].values), labels)

    clf= make_model()
    clf.fit(Xtr, ytr)

    y_pred= clf.predict(Xte)
    gbm_f1= report("LightGBM", yte, y_pred, labels)
    gain= gbm_f1 - lr_f1
    print(f"\nLGBM - LR dwarf/evolved F1: {gain:+.3f}")
    colour_breakdown(test, y_pred)

    confusion_plot(yte, y_pred, labels, CM_FIG)
    importance_plot(clf, config.FEATURES, IMP_FIG)

    proba= clf.predict_proba(Xte)
    out= test[["source_id","bp_rp","mg","log_v_tan", "lum_class"]].copy()
    out["predicted"]= y_pred
    out["confidence"]= proba.max(axis=1)
    for i, c in enumerate(clf.classes_):
        out[f"p_{c}"]= proba[:, i]
    for c in config.DIAGNOSTICS:
        if c in test.columns:
            out[c]= test[c].values
    out.to_parquet(TEST_OUT, index=False)
    print(f"Wrote {TEST_OUT}  ({len(out):,} test stars)")

    with open(MODEL_OUT, "wb") as fh:
        pickle.dump({"model": clf, "features": config.FEATURES,"classes": list(clf.classes_)}, fh)
    print(f"Wrote {MODEL_OUT}")

    config.log_provenance(
        "model",
        n_train=len(train), n_test=len(test),split="GroupShuffleSplit on HEALPix level-3 sky patches from source_id",
        features=config.FEATURES,classes=labels,lr_baseline_f1_dwarf_evolved=float(lr_f1),lgbm_f1_dwarf_evolved=float(gbm_f1),
        note="white_dwarf labels are geometric and therefore circular";"dwarf-vs-evolved is the meaningful result", )

if __name__== "__main__":
    main()
