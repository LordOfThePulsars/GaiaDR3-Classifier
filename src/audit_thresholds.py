import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import config
from build_labels_mod import BUFFER,GIANT_LOGG_MAX,SUBGIANT_DELTA
from attribute import BRIGHT_EXCESS,MAX_BINARY_EXCESS

#Equal-mass pair is twice as luminous,using magnitude laws
EQUAL_MASS_BINARY= 2.5 * np.log10(2)

#Clips rather than giving NaN,so every dwarf gives a residual
def mode_ridge(d: pd.DataFrame):
    edges= np.arange(0.4,2.05,0.06)
    cs,ms= [],[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        v= d.loc[d.bp_rp.between(lo,hi),"mg"].dropna()
        if len(v) < 40:
            continue
        grid= np.linspace(v.min(),v.max(),300)
        try:
            peak= float(grid[np.argmax(gaussian_kde(v.values)(grid))])
        except Exception:
            peak= float(v.median())
        cs.append(0.5 * (lo + hi))
        ms.append(peak)
    cs,ms= np.asarray(cs),np.asarray(ms)
    coef= np.polyfit(cs,ms,3)

    def ridge(x):
        return np.polyval(coef,np.clip(np.asarray(x,float),cs.min(),cs.max()))
    return ridge

def main() -> None:
    lab_p= config.DATA_PROCESSED/"labels.parquets"
    feat_p= config.DATA_PROCESSED/"features.parquet"
    for p in (lab_p,feat_p):
        if not p.exists():
            raise SystemExit(f"Missing {p}")

    lab= pd.read_parquet(lab_p)
    feat= pd.read_parquet(feat_p)
    print("Buffer vs log-g scatter")
    d= lab.loc[lab.logg_cal > GIANT_LOGG_MAX,"delta_logg"].dropna()
    sigma_g= 0.5 * (d.quantile(.84) - d.quantile(.16))
    print(f"delta_logg scatter (1 sigma): {sigma_g:.3f} dex")
    print(f"BUFFER {BUFFER:.3f}")
    print(f"half sigma: {sigma_g/2:.3f}")

    print("\nGIANT_LOGG_MAX")
    g,dl= lab.logg_cal,lab.delta_logg
    by_giant= g < GIANT_LOGG_MAX
    by_delta= dl < SUBGIANT_DELTA - BUFFER
    only_giant= int((by_giant & ~by_delta).sum())
    only_delta= int((~by_giant & by_delta).sum())
    both= int((by_giant & by_delta).sum())
    print(f"evolved by giant cut: {only_giant:>7,}")
    print(f"evolved by delta cut: {only_delta:>7,}")
    print(f"evolved by both: {both:>7,}")
    frac= 100 * only_giant / max(both + only_giant + only_delta,1)
    print(f"Decided by giant cut: {frac:.2f}%")

    print("\nPhotometric scatter")
    dw= feat[(feat.lum_class== "dwarf") & feat.mg.notna() & feat.bp_rp.notna()]
    ridge= mode_ridge(dw)
    resid= (dw.mg - ridge(dw.bp_rp)).dropna()

    faint= resid[resid > 0]
    sigma_m= float(faint.quantile(0.683))
    print(f"single star M_G scatter: {sigma_m:.3f} mag")

    print(f"BRIGHT_EXCESS  current {BRIGHT_EXCESS:.3f}   derived  {2*sigma_m:.3f}")
    print(f"MAX_BINARY_EXCESS current {MAX_BINARY_EXCESS:.3f} derived ({EQUAL_MASS_BINARY:.3f} + 1 sigma) {EQUAL_MASS_BINARY + sigma_m:.3f}")

    rows= [ ("BUFFER",BUFFER,sigma_g / 2),("BRIGHT_EXCESS",BRIGHT_EXCESS,2 * sigma_m),("MAX_BINARY_EXCESS",MAX_BINARY_EXCESS,EQUAL_MASS_BINARY + sigma_m),]
    print(f"{'constant':22s}{'current':>10s}{'derived':>10s}{'change':>10s}")
    for name,cur,der in rows:
        print(f" {name:22s}{cur:10.3f}{der:10.3f}{der-cur:+10.3f}")
    print(f"{'GIANT_LOGG_MAX':22s}{GIANT_LOGG_MAX:10.3f}{only_giant:10,}")

if __name__== "__main__":
    main()
