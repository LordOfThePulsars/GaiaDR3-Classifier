import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import config

LABELS_IN= config.DATA_RAW/"gsp_spec_labels.parquet"
OUTPUT= config.DATA_PROCESSED/"labels.parquets"
DELTA_FIG= config.FIGURES/"delta_logg.png"
KIEL_FIG= config.FIGURES/"kiel_diagram.png"

STRICT_FLAG_PREFIX= config.STRICT_FLAG_PREFIX

#Validity domain of the Recio-Blanco et al. 2023 
LOGG_VALID= (0.0,5.5)

#Ridge is fitted only where dwarfs and dominant
RIDGE_TEFF_RANGE= (4200,6800)
RIDGE_MIN_STARS= 40
RIDGE_BIN_WIDTH= 150
RIDGE_SEARCH_LOGG= (3.6,5.0)
RIDGE_POLY_DEGREE= 3

#Kiel diagram shows one continuous sequence
MERGE_EVOLVED= True

GIANT_LOGG_MAX= 3.5

#Dwarfs and subgiants are separated by offset from the ridge,~0.3+ dex
SUBGIANT_DELTA= -0.35

#Scatter is 0.2-0.3 dex
BUFFER= 0.12

#Log g correction from Recio-Blanco et al. 2023 Table 3,Eq. 1
def logg_calibrate(logg: pd.Series) -> pd.Series:
    g= logg.clip(*LOGG_VALID)
    return g + 0.4496 - 0.0036*g - 0.0224 * g**2

#Mode per bin,not median
def ridge_fit(teff: pd.Series,logg: pd.Series):
    lo,hi= RIDGE_TEFF_RANGE
    edges= np.arange(lo,hi+RIDGE_BIN_WIDTH,RIDGE_BIN_WIDTH)
    grid= np.linspace(*RIDGE_SEARCH_LOGG,400)
    centres,modes= [],[]
    for i in range(len(edges) - 1):
        m= (teff >= edges[i]) & (teff < edges[i + 1])
        v= logg[m]
        v= v[(v > RIDGE_SEARCH_LOGG[0]) & (v < RIDGE_SEARCH_LOGG[1])].dropna()
        if len(v)<RIDGE_MIN_STARS:
            continue
        try:
            peak= grid[np.argmax(gaussian_kde(v.values)(grid))]
        except Exception:
            peak= float(np.median(v))
        centres.append(0.5 *(edges[i] + edges[i + 1]))
        modes.append(peak)
    centres= np.asarray(centres)
    modes= np.asarray(modes)
    if len(centres)<RIDGE_POLY_DEGREE + 1:
        raise SystemExit("Not enough temperature bins in order to fit a ridge")
    #Fit in scaled temperature so the polynomial is well conditioned
    t= centres/5750.0
    coef= np.polyfit(t,modes,RIDGE_POLY_DEGREE)
    edge_lo,edge_hi= np.polyval(coef,lo/5750.0),np.polyval(coef,hi /5750.0)

    #Held flat outside the fitted range rather than extrapolated
    def ridge(teff_vals):
        x= np.asarray(teff_vals,dtype=float)
        y= np.polyval(coef,np.clip(x,lo,hi) / 5750.0)
        return np.where(x < lo,edge_lo,np.where(x > hi,edge_hi,y))

    return ridge,centres,modes,coef

#Luminosity class,with an ambiguous band straddling the boundary
def class_assign(df:pd.DataFrame)->pd.Series:
    out= pd.Series("ambiguous",index=df.index,dtype=object)
    g,d= df["logg_cal"],df["delta_logg"]

    if MERGE_EVOLVED:
        #Dropped: audit_thresholds found it decides 0 stars the delta cut misses.
        out[d< SUBGIANT_DELTA - BUFFER]= "evolved"
        out[d> SUBGIANT_DELTA + BUFFER]= "dwarf"
    else:
        out[g< GIANT_LOGG_MAX - BUFFER]= "giant"
        out[(g> GIANT_LOGG_MAX + BUFFER)&(d< SUBGIANT_DELTA - BUFFER)]= "subgiant"
        out[(g> GIANT_LOGG_MAX + BUFFER)&(d> SUBGIANT_DELTA + BUFFER)]= "dwarf"
    out[g.isna() | d.isna()]= "ambiguous"
    return out

def kiel_plot(df,ridge,centres,modes,path):
    fig,ax= plt.subplots(figsize=(8.5,7))
    m= df.teff_gspspec.notna() & df.logg_cal.notna()
    hb= ax.hexbin(df.loc[m,"teff_gspspec"],df.loc[m,"logg_cal"],gridsize=150,bins="log",cmap="magma",mincnt=1)
    tt= np.linspace(3400,7800,400)
    ax.plot(tt,ridge(tt),color="#3fb8af",lw=2.0,label="MS ridge (fitted)")
    ax.plot(tt,ridge(tt) + SUBGIANT_DELTA,color="#3fb8af",lw=1.2,ls="--",label=f"ridge {SUBGIANT_DELTA:+.2f} dex ")
    ax.scatter(centres,modes,s=16,color="#d5a535",zorder=5,label="binned modes")
    ax.axhline(GIANT_LOGG_MAX,color="#e82828",lw=1.2,ls=":",label=f"Given giant cut (log g= {GIANT_LOGG_MAX})")
    ax.set_xlabel(r"$T_{\rm eff}$ [K]")
    ax.set_ylabel(r"Calibrated log-g [dex]")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.legend(loc="lower right",framealpha=0.9,fontsize=9)
    ax.set_title("Kiel diagram with fitted main-sequence ridge")
    fig.colorbar(hb,ax=ax,label="stars per bin")
    fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

def delta_plot(df,path):
    fig,ax= plt.subplots(figsize=(9,5))
    d= df.loc[df.logg_cal > GIANT_LOGG_MAX,"delta_logg"].dropna()
    ax.hist(d,bins=220,range=(-2.0,1.0),color="#8b7ff0",log=True)
    ax.axvline(0,color="#3fb8af",lw=1.5,label="ridge")
    ax.axvline(SUBGIANT_DELTA,color="#ff6b6b",lw=1.5,ls="--",label=f"subgiant threshold ({SUBGIANT_DELTA})")
    ax.axvspan(SUBGIANT_DELTA-BUFFER,SUBGIANT_DELTA+BUFFER,color="#ff6b6b",alpha=0.15,label="ambiguous buffer")
    ax.set_xlabel(r"$\Delta \log g$  [dex]")
    ax.set_ylabel("Stars")
    ax.set_title("Main Sequence ridge offset for non-giants")
    ax.legend()
    fig.savefig(path,dpi=175,bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

def main()-> None:
    if not LABELS_IN.exists():
        raise SystemExit(f"Missing {LABELS_IN}")
    df= pd.read_parquet(LABELS_IN)
    print(f"Number of stars that follow GSP-Spec Parameters: {len(df):,}")
    df= df[df["flags_gspspec"].astype(str).str.startswith(STRICT_FLAG_PREFIX)].copy()
    print(f"Pass strict GSP-Spec quality flags {len(df):,}")

    df["logg_cal"]= logg_calibrate(df["logg_gspspec"])
    delta_cal= df.logg_cal - df.logg_gspspec
    print("\ncalibration shift:")
    print(f"logg< 3.0: {delta_cal[df.logg_gspspec <3].median():+.3f} dex")
    print(f"logg> 4.0: {delta_cal[df.logg_gspspec >4].median():+.3f} dex")

    ridge,centres,modes,coef= ridge_fit(df.teff_gspspec,df.logg_cal)
    print(f"{len(centres)} temperature bins used ({RIDGE_TEFF_RANGE[0]}-{RIDGE_TEFF_RANGE[1]} K)")
    print("ridge log g:")
    for t in (6500,6000,5750,5000,4500):
        print(f"  {t} K: {float(ridge(t)):.3f}")
    print(f"solar check: {float(ridge(5772)):.3f}  (true 4.44)")
    df["delta_logg"]= df.logg_cal - ridge(df.teff_gspspec.values)
    d= df.loc[df.logg_cal > GIANT_LOGG_MAX,"delta_logg"].dropna()
    print("\noffset from ridge:")
    print(f"median:{d.median():+.3f} dex")
    print(f"16th/84th percentile: {d.quantile(.16):+.3f} / {d.quantile(.84):+.3f}")
    print(f"scatter (half 16-84 range): {0.5*(d.quantile(.84)-d.quantile(.16)):.3f} dex")

    df["lum_class"]= class_assign(df)
    counts= df.lum_class.value_counts()
    print("\nclass balance:")
    for cls in counts.index:
        n= int(counts[cls])
        print(f"{cls:10s}:{n:>7,}  ({100*n/len(df):5.1f}%)")
    trainable= int((df.lum_class != "ambiguous").sum())
    print(f"\nTrainable: {trainable:,}")
    print("\nmedian Teff by class:")
    for cls in [c for c in counts.index if c != "ambiguous"]:
        s= df.loc[df.lum_class==cls,"teff_gspspec"]
        if len(s):
            print(f"{cls:10s}:{s.median():>7.0f} K  (n={len(s):,})")
    print("\nmetallicity by class:")
    for cls in [c for c in counts.index if c != "ambiguous"]:
        s= df.loc[df.lum_class== cls,"mh_gspspec"].dropna()
        if len(s):
            print(f"{cls:10s}: median [M/H] {s.median():+.2f} 5th pct {s.quantile(.05):+.2f}")

    kiel_plot(df,ridge,centres,modes,KIEL_FIG)
    delta_plot(df,DELTA_FIG)

    keep= ["source_id","teff_gspspec","logg_gspspec","logg_cal","delta_logg","mh_gspspec","alphafe_gspspec","lum_class","spectraltype_esphs","flags_gspspec"]
    out= df[[c for c in keep if c in df.columns]].copy()
    config.DATA_PROCESSED.mkdir(parents=True,exist_ok=True)
    out.to_parquet(OUTPUT,index=False)
    print(f"\nWrote {OUTPUT} ({OUTPUT.stat().st_size/1e6:.2f} MB)")

    config.log_provenance(
        "labels",
        rows=len(out), trainable_rows=trainable,class_counts={k: int(v) for k,v in counts.items()},
        calibration="Recio-Blanco et al. 2023 Table 3 Eq. 1",ridge_method="KDE mode per Teff bin,degree-3 polynomial in Teff/5750",
        ridge_coefficients=[float(c) for c in coef],ridge_teff_range=RIDGE_TEFF_RANGE,giant_logg_max=GIANT_LOGG_MAX,subgiant_delta=SUBGIANT_DELTA,
        buffer_dex=BUFFER,merge_evolved=MERGE_EVOLVED,target_variable="lum_class",output=str(OUTPUT) )

if __name__== "__main__":
    main()
