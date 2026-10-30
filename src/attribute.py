import pickle
import sys

from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


import config
from build_features import derive
from build_labels_mod import (BUFFER,GIANT_LOGG_MAX,SUBGIANT_DELTA,logg_calibrate,ridge_fit)
from domain import in_domain,domain_measure

ANOM_IN= config.DATA_PROCESSED/"anomalies.parquet"
CTRL_IN= config.DATA_PROCESSED/"controls.parquet"
FEATURES_IN= config.DATA_PROCESSED/"features.parquet"
BIN_SRC= config.DATA_RAW/"gaia_binaries.parquet"
BIN_LAB= config.DATA_RAW/"gsp_spec_binaries.parquet"
MODEL_IN= config.DATA_PROCESSED/"calibrated_model.pkl"

FIG_DOSE= config.FIGURES/"binary_dose_response.png"
FIG_DIAG= config.FIGURES/"diagnostic_distributions.png"
OUT= config.DATA_PROCESSED/"attribution.parquet"

RUWE_BINS= [1.0,1.2,1.4,1.7,2.0,2.4,3.0]

#Priority order with rather strict thresholds to ensure proper order
METAL_POOR= -0.4
HIGH_VTAN= 100.0
BRIGHT_EXCESS= 0.368
MAX_BINARY_EXCESS= 0.937

#Measured 1 sigma of delta_logg
LOGG_SCATTER= 0.175

#KS test being used instead of t-test given skew dist sits on the tail
def ks(a: pd.Series,b: pd.Series,name: str) -> dict:
    a,b= a.dropna(),b.dropna()
    if len(a) < 8 or len(b) < 8:
        return {"name": name,"n_a": len(a),"n_b": len(b),"p": np.nan}
    st,p= stats.ks_2samp(a,b)
    pooled= np.sqrt((a.var(ddof=1) + b.var(ddof=1))/ 2)
    return {
        "name": name,"n_a": len(a),"n_b": len(b),
        "median_a": float(a.median()),"median_b": float(b.median()),
        "diff": float(a.median() - b.median()),
        "ks": float(st),"p": float(p),
        "cohens_d": float((a.mean() - b.mean())/ pooled) if pooled else np.nan,}

def show_ks(rows: list[dict]) -> None:
    print(f"{'diagnostic':26s}{'anomalies':>11s}{'controls':>11s}{'diff':>9s}{'KS':>7s}{'p':>11s}{'d':>7s}")
    for r in rows:
        if np.isnan(r.get("p",np.nan)):
            print(f"{r['name']:26s} too few values (n={r['n_a']},{r['n_b']})")
            continue
        star= " ***" if r["p"] < 0.001 else ("  *" if r["p"] < 0.05 else "")
        print(f"{r['name']:26s}{r['median_a']:11.3f}{r['median_b']:11.3f}{r['diff']:+9.3f}{r['ks']:7.3f}{r['p']:11.2e}{r['cohens_d']:+7.2f}{star}")

#Mode being more convenient here rather than median as it gets off focus else
def ms_ridge(field: pd.DataFrame,verbose: bool= True):
    d= field[(field.lum_class=="dwarf") & field.mg.notna()
              & field.bp_rp.notna()]
    edges= np.arange(0.4,2.05,0.06)
    cs,ms= [],[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        v= d.loc[d.bp_rp.between(lo,hi),"mg"]
        if len(v) < 40:
            continue
        grid= np.linspace(v.min(),v.max(),300)
        try:
            peak= float(grid[np.argmax(stats.gaussian_kde(v.values)(grid))])
        except Exception:
            peak= float(v.median())
        cs.append(0.5 * (lo + hi))
        ms.append(peak)

    cs,ms= np.asarray(cs),np.asarray(ms)
    coef= np.polyfit(cs,ms,3)

    #NaN outside the fitted colour range rather than fitted expected value
    def ridge(colour):
        x= np.asarray(colour,dtype=float)
        y= np.polyval(coef,np.clip(x,cs.min(),cs.max()))
        return np.where((x < cs.min() - 0.1) | (x > cs.max() + 0.1),np.nan,y)

    if verbose:
        print("\nMS ridge:")
        for c in (0.6,0.8,1.0,1.2,1.5):
            if cs.min() <= c <= cs.max():
                print(f"BP-RP {c:.1f} -> M_G {float(ridge(c)):.2f}")
    return ridge

#Model not trained such that RUWE is the only variable being changed
def dose_response(model,classes,dom,field,ridge) -> pd.DataFrame:
    rows= []

    clean= field[field.lum_class.isin(["dwarf","evolved"])].copy()
    clean= clean[in_domain(clean,dom)]
    if "ruwe" not in clean.columns:
        raise SystemExit("features.parquet has no ruwe column; re-run build labels mod")

    #Joining high RUWE source to their respective spectroscopic labels
    src= pd.read_parquet(BIN_SRC)
    lab= pd.read_parquet(BIN_LAB)
    lab= lab[lab.flags_gspspec.astype(str).str.startswith(config.STRICT_FLAG_PREFIX)]

    b= src.merge(lab[["source_id","teff_gspspec","logg_gspspec"]],on="source_id",how="inner")
    b= derive(b)

    #Labelling binaries with same ridge offset scheme as the field
    b["logg_cal"]= logg_calibrate(b.logg_gspspec)
    kiel,*_= ridge_fit(b.teff_gspspec,b.logg_cal)
    b["delta_logg"]= b.logg_cal - kiel(b.teff_gspspec.values)
    b["lum_class"]= np.where(
        (b.logg_cal < GIANT_LOGG_MAX) | (b.delta_logg < SUBGIANT_DELTA - BUFFER),"evolved",
        np.where(b.delta_logg > SUBGIANT_DELTA + BUFFER,"dwarf","ambiguous"))
    b= b[b.lum_class.isin(["dwarf","evolved"])]
    b= b[in_domain(b,dom)]

    both= pd.concat([
        clean[["source_id","ruwe","bp_rp","mg","bp_g","g_rp","log_v_tan","lum_class"]],
        b[["source_id","ruwe","bp_rp","mg","bp_g","g_rp","log_v_tan","lum_class"]] ],
         ignore_index=True).dropna(subset=config.FEATURES)

    proba= model.predict_proba(both[config.FEATURES].values)
    both["predicted"]= np.array(classes)[proba.argmax(axis=1)]
    both["confidence"]= proba.max(axis=1)
    both["excess"]= ridge(both.bp_rp) - both.mg

    dwarfs= both[both.lum_class=="dwarf"]
    for lo,hi in zip(RUWE_BINS[:-1],RUWE_BINS[1:]):
        s= dwarfs[dwarfs.ruwe.between(lo,hi,inclusive="left")]
        if len(s) < 40:
            continue
        err= s.predicted=="evolved"
        conf= err & (s.confidence > 0.9)
        #Wilson interval stays honest near zero where the normal approx goes negative
        n,k= len(s),int(err.sum())
        p= k/ n
        z= 1.96
        denom= 1 + z**2/ n
        centre= (p + z**2/ (2 * n))/ denom
        halfw= z * np.sqrt(p * (1 - p)/ n + z**2/ (4 * n**2))/ denom
        rows.append(dict(ruwe_lo=lo,ruwe_hi=hi,n=n,n_err=k,rate=100 * p,
                         lo95=100 * (centre - halfw),hi95=100 * (centre + halfw),n_conf=int(conf.sum()),
                         median_excess=float(s.loc[err,"excess"].median()) if k else np.nan))
    return pd.DataFrame(rows),both

def dose_plot(dr: pd.DataFrame,path: Path) -> None:
    fig,ax= plt.subplots(figsize=(8,5.2))
    x= 0.5 * (dr.ruwe_lo + dr.ruwe_hi)
    ax.errorbar(x,dr.rate,yerr=[dr.rate - dr.lo95,dr.hi95 - dr.rate],fmt="o-",color="#8b7ff0",capsize=4,lw=1.6)
    ax.axvline(1.4,color="#ff6b6b",ls="--",lw=1.2,label="Gaia Query Cut")
    ax.set_xlabel("RUWE")
    ax.set_ylabel("Dwarf-to-evolved misclassification [%]")
    ax.set_title("Binary dose-response")
    ax.legend(fontsize=9)
    fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

def diagnostics_plot(anom,ctrl,path):
    cols= [("mh_gspspec","[M/H]  [dex]"),("v_tan",r"$v_{\rm tan}$  [km/s]"),("ruwe","RUWE"),("phot_bp_rp_excess_factor","BP/RP excess")]
    have= [(c,l) for c,l in cols if c in anom.columns and anom[c].notna().sum() > 8]
    if not have:
        return
    fig,axes= plt.subplots(1,len(have),figsize=(4.6 * len(have),4.4))
    axes= np.atleast_1d(axes)
    for ax,(c,lab) in zip(axes,have):
        a,b= anom[c].dropna(),ctrl[c].dropna()
        lo,hi= np.percentile(pd.concat([a,b]),[1,99])
        bins= np.linspace(lo,hi,34)
        ax.hist(b,bins=bins,density=True,alpha=0.55,color="#3fb8af",label=f"controls ({len(b):,})")
        ax.hist(a,bins=bins,density=True,alpha=0.55,color="#ff6b6b",label=f"anomalies ({len(a):,})")
        ax.set_xlabel(lab)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Density")
    fig.suptitle("Withheld diagnostics: anomalies vs colour-magnitude controls")
    fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

#One cause per object, to prevent double count
def attribute(anom: pd.DataFrame,ridge) -> pd.DataFrame:
    a= anom.copy()
    a["excess"]= ridge(a.bp_rp) - a.mg
    cause= pd.Series("unexplained",index=a.index,dtype=object)

    too_bright= a.lum_class=="dwarf"
    too_faint= a.lum_class=="evolved"
    if "delta_logg" in a.columns:
        near= (a.delta_logg - SUBGIANT_DELTA).abs()< LOGG_SCATTER
        cause[(cause=="unexplained") & near]= "near label boundary"
    if "ruwe" in a.columns:
        cause[(cause=="unexplained") & too_bright & (a.ruwe > 1.4) & (a.excess.between(0.05,MAX_BINARY_EXCESS))]= "unresolved binary"
        cause[(cause=="unexplained") & too_bright & (a.ruwe > 1.4)]= "high RUWE" 
    if "non_single_star" in a.columns:
        cause[(cause=="unexplained") & (a.non_single_star.fillna(0) > 0)]= "known non-single"
    if "mh_gspspec" in a.columns:
        cause[(cause=="unexplained") & too_faint & (a.mh_gspspec < METAL_POOR)]= "metal-poor subdwarf"
    if "v_tan" in a.columns:
        cause[(cause=="unexplained") & too_faint &(a.v_tan > HIGH_VTAN)]= "halo kinematics"
    cause[(cause=="unexplained") & too_bright &(a.excess > MAX_BINARY_EXCESS)]= "over-luminous beyond any binary "
    cause[(cause=="unexplained") & too_bright & (a.excess > BRIGHT_EXCESS)]= "mildly over-luminous"
    if "phot_variable_flag" in a.columns:
        cause[(cause=="unexplained") &(a.phot_variable_flag.astype(str)=="VARIABLE")]= "variable star"

    a["cause"]= cause
    return a

def main() -> None:
    for p in (ANOM_IN,CTRL_IN,FEATURES_IN,MODEL_IN):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run the earlier modules first.")

    with open(MODEL_IN,"rb") as fh:
        bundle= pickle.load(fh)
    model,classes= bundle["model"],bundle["classes"]

    anom= pd.read_parquet(ANOM_IN)
    ctrl= pd.read_parquet(CTRL_IN)
    field= pd.read_parquet(FEATURES_IN)
    print(f"{len(anom):,} anomalies,{len(ctrl):,} matched controls")

    dom= domain_measure(field[field.lum_class.isin(classes)])
    ridge= ms_ridge(field)

    if not (BIN_SRC.exists() and BIN_LAB.exists()):
        print("Run query_binaries.py,data missing")
        dr= pd.DataFrame()
    else:
        dr,_= dose_response(model,classes,dom,field,ridge)
        if len(dr):
            print("\ndwarf -> evolved by RUWE:")
            print(f"{'RUWE':>12s}{'n':>8s}{'errors':>8s}{'rate %':>9s}{'95% CI':>16s}{'excess mag':>12s}")
            for _,r in dr.iterrows():
                ci= f"[{r.lo95:.2f},{r.hi95:.2f}]"
                ex= f"{r.median_excess:+.3f}" if not np.isnan(r.median_excess) else "  --"
                print(f"{r.ruwe_lo:5.1f}-{r.ruwe_hi:<5.1f}{int(r.n):>8,}{int(r.n_err):>8,}{r.rate:>9.2f}{ci:>16s}{ex:>12s}")

            below= dr[dr.ruwe_hi <= 1.4]
            above= dr[dr.ruwe_lo >= 1.4]
            if len(below) and len(above):
                r_lo= below.n_err.sum()/ below.n.sum()
                r_hi= above.n_err.sum()/ above.n.sum()
                tab= [[int(above.n_err.sum()),int(above.n.sum() - above.n_err.sum())],
                       [int(below.n_err.sum()),int(below.n.sum() - below.n_err.sum())]]
                odds,p= stats.fisher_exact(tab)
                print(f"\nRUWE < 1.4: {100*r_lo:.2f}% RUWE 1.4-3: {100*r_hi:.2f}%")
                print(f"ratio {r_hi/max(r_lo,1e-9):.1f}x odds ratio {odds:.1f}   Fisher p= {p:.2e}")
                mono= dr.rate.is_monotonic_increasing
                print(f"Monotonic across all RUWE bins: {mono}")
            dose_plot(dr,FIG_DOSE)

    print("\n Diagnostics against controls")

    for direction,true_cls in (("too bright","dwarf"),("too faint","evolved")):
        a= anom[anom.lum_class==true_cls]
        c= ctrl[ctrl.lum_class==true_cls]
        print(f"\n{direction}   n={len(a):,} vs {len(c):,} controls")
        if len(a) < 8:
            print("too few to test reliably")
            continue
        rows= [ks(a[col],c[col],col) for col in
                ("mh_gspspec","v_tan","ruwe","phot_bp_rp_excess_factor","alphafe_gspspec")
                if col in a.columns and col in c.columns]
        show_ks([r for r in rows if r])
    diagnostics_plot(anom,ctrl,FIG_DIAG)

    att= attribute(anom,ridge)
    counts= att.cause.value_counts()
    n_att= max(len(att),1)

    tb= att[att.lum_class=="dwarf"]
    if len(tb):
        print("\n Excess above MS:")
        for q in (0.25,0.50,0.75,0.95):
            print(f"{int(q*100):2d}th percentile: {tb.excess.quantile(q):+.2f} mag")
        n_ok= int((tb.excess <= MAX_BINARY_EXCESS).sum())
        print(f"within the {MAX_BINARY_EXCESS} mag binary ceiling: {n_ok:,} of {len(tb):,} ({100*n_ok/len(tb):.0f}%)")
    print()
    for cause,n in counts.items():
        print(f"{cause:32s}: {n:>4,}  ({100*n/n_att:5.1f}%)")

    unexp= int(counts.get("unexplained",0))
    print(f"\nAttributed: {len(att)-unexp:,}/ {len(att):,} ({100*(len(att)-unexp)/n_att:.1f}%)")
    print(f"unexplained: {unexp:,} ({100*unexp/n_att:.1f}%)")
    att.to_parquet(OUT,index=False)
    print(f"\nWrote {OUT}")

    config.log_provenance(
        "attribution",
        n_anomalies=len(att),
        causes={k: int(v) for k,v in counts.items()},
        unexplained_fraction=float(unexp/ n_att),
        thresholds=dict(metal_poor=METAL_POOR,high_vtan=HIGH_VTAN,bright_excess=BRIGHT_EXCESS),
        note="causes assigned in priority order,one per object", )

if __name__=="__main__":
    main()
