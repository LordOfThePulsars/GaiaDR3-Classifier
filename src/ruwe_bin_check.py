import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import contextlib
import io
import pickle

import numpy as np
import pandas as pd
from scipy import stats

import attribute
import config
from domain import domain_measure

SPLIT= 1.4
SPLITS= [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2]

#Pooling across a fixed split ignores the inner edgees
BINNINGS= {
    "published ": [1.0, 1.2, 1.4, 1.7, 2.0, 2.4, 3.0],"coarse ": [1.0, 1.4, 2.0, 3.0],
    "two bins only": [1.0, 1.4, 3.0], "fine ": [1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.9, 2.2, 2.6, 3.0],
    "edges off 1.4": [1.0, 1.25, 1.45, 1.75, 2.1, 2.5, 3.0],"equal-count (4)": None,
}

def rebin(dwarfs: pd.DataFrame, edges: list[float]) -> pd.DataFrame:
    rows= []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s= dwarfs[dwarfs.ruwe.between(lo, hi, inclusive="left")]
        if len(s) < 40:
            continue
        rows.append(dict(lo=lo, hi=hi, n=len(s), n_err=int((s.predicted== "evolved").sum())))
    return pd.DataFrame(rows)

def ratio_across(dw: pd.DataFrame, edges: list[float], split: float= SPLIT):
    dr= rebin(dw, edges)
    if not len(dr):
        return None
    lo, hi= dr[dr.hi <= split], dr[dr.lo >= split]
    if not len(lo) or not len(hi):
        return None
    n_lo, k_lo= int(lo.n.sum()), int(lo.n_err.sum())
    n_hi, k_hi= int(hi.n.sum()), int(hi.n_err.sum())
    r_lo, r_hi= k_lo / n_lo, k_hi / n_hi
    _, p= stats.fisher_exact([[k_hi, n_hi - k_hi], [k_lo, n_lo - k_lo]])
    return len(dr), n_lo, r_lo, n_hi, r_hi, r_hi / max(r_lo, 1e-12), p

def split_at(dw: pd.DataFrame, split: float):
    return ratio_across(dw, [1.0, split, 3.0], split=split)

def main() -> None:
    with open(config.DATA_PROCESSED / "calibrated_model.pkl", "rb") as fh:
        bundle= pickle.load(fh)
    model, classes= bundle["model"], bundle["classes"]

    field= pd.read_parquet(config.DATA_PROCESSED / "features.parquet")
    dom= domain_measure(field[field.lum_class.isin(classes)])

    #dose_response prints its own report
    with contextlib.redirect_stdout(io.StringIO()):
        ridge= attribute.ms_ridge(field, verbose=False)
        _, both= attribute.dose_response(model, classes, dom, field, ridge)

    dw= both[both.lum_class== "dwarf"].dropna(subset=["ruwe"])
    dw= dw[dw.ruwe.between(1.0, 3.0)]
    print(f"dwarfs in domain, RUWE 1.0-3.0: {len(dw):,}")
    print(f"split: {SPLIT}\n")

    BINNINGS["equal-count (4)"]= [float(x) for x in np.quantile(dw.ruwe, [0, .25, .50, .75, 1.0])]

    print(f"{'binning':20s}{'bins':>5s}{'n<1.4':>8s}{'err%':>8s}{'n>1.4':>8s}{'err%':>8s}{'ratio':>8s}{'Fisher p':>11s}")
    ratios= []
    for name, edges in BINNINGS.items():
        res= ratio_across(dw, edges)
        if res is None:
            print(f"{name:20s}{'--':>5s}")
            continue
        nb, n_lo, r_lo, n_hi, r_hi, ratio, p= res
        ratios.append(ratio)
        print(f"{name:20s}{nb:5d}{n_lo:8,d}{100*r_lo:8.2f}{n_hi:8,d}{100*r_hi:8.2f}{ratio:8.1f}{p:11.1e}")
    print(f"min {min(ratios):.1f}  max {max(ratios):.1f}  spread {max(ratios)-min(ratios):.1f}")

    print(f"\n{'split':>7s}{'n below':>10s}{'err%':>8s}{'n above':>10s}{'err%':>8s}{'ratio':>8s}{'Fisher p':>11s}")
    sweep= []
    for split in SPLITS:
        res= split_at(dw, split)
        if res is None:
            print(f"{split:7.1f}{'--':>10s}")
            continue
        _, n_lo, r_lo, n_hi, r_hi, ratio, p= res
        sweep.append(ratio)
        print(f"{split:7.1f}{n_lo:10,d}{100*r_lo:8.2f}{n_hi:10,d}{100*r_hi:8.2f}{ratio:8.1f}{p:11.1e}")
    print(f"min {min(sweep):.1f}  max {max(sweep):.1f}  median {np.median(sweep):.1f}")

    err= (dw.predicted== "evolved").astype(int)
    rho, p_rho= stats.spearmanr(dw.ruwe, err)
    mw= stats.mannwhitneyu(dw.loc[err== 1, "ruwe"], dw.loc[err== 0, "ruwe"], alternative="greater")
    n1, n0= int(err.sum()), int((1 - err).sum())
    print(f"\nSpearman rho: {rho:+.4f}  p {p_rho:.2e}")
    print(f"Mann-Whitney p: {mw.pvalue:.2e}")
    print(f"AUC: {mw.statistic/(n1*n0):.3f}  ({n1} errors, {n0:,} correct)")

if __name__== "__main__":
    main()
