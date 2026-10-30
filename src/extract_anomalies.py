import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
from domain import describe,in_domain,domain_measure

OOF_IN= config.DATA_PROCESSED/"oof_predictions.parquet"
FEATURES_IN= config.DATA_PROCESSED/"features.parquet"
ANOM_OUT= config.DATA_PROCESSED/"anomalies.parquet"
CTRL_OUT= config.DATA_PROCESSED/"controls.parquet"
FIG= config.FIGURES/"anomalies_hr.png"

CONFIDENCE= 0.90
SPECTROSCOPIC= ["dwarf","evolved"]

#Larger k tightens the null but forces looser matches from a finite pool.
N_CONTROLS= 5

TOL_COLOUR= 0.10
TOL_MG= 0.40

#Without matching nothing is falsifiable, everything correlates with HR position.
def controls_match(anom: pd.DataFrame,pool: pd.DataFrame,k: int= N_CONTROLS) -> pd.DataFrame:
    used= set()
    rows= []

    p_colour= pool.bp_rp.values
    p_mg= pool.mg.values
    pidx= pool.index.values

    for _,a in anom.iterrows():
        #Scaled by tolerance so both axes count equally
        dc= (p_colour - a.bp_rp) / TOL_COLOUR
        dm= (p_mg - a.mg) / TOL_MG
        ok= (np.abs(dc) <= 1.0) & (np.abs(dm) <= 1.0)
        ok &= (pool.lum_class.values== a.lum_class)
        if not ok.any():
            continue

        d= np.hypot(dc,dm)
        #Rejected candidates sort last at infinity
        order= np.argsort(np.where(ok,d,np.inf))
        taken= 0
        for j in order:
            if not ok[j]:
                break
            if pidx[j] in used:
                continue
            used.add(pidx[j])
            r= pool.loc[pidx[j]].copy()
            r["matched_to"]= a.source_id
            rows.append(r)
            taken += 1
            if taken >= k:
                break
    return pd.DataFrame(rows)

#If the colour-magnitude difference survives,tolerarances are notstrict enough
def balance_report(anom: pd.DataFrame,ctrl: pd.DataFrame) -> None:
    print("\nmatching balance:")
    print(f"{'':10s}{'anomalies':>12s}{'controls':>12s}{'diff':>10s}")
    for col in ("bp_rp","mg"):
        a,c= anom[col].median(),ctrl[col].median()
        print(f"{col:10s}{a:12.3f}{c:12.3f}{a-c:+10.3f}")
    for cls in SPECTROSCOPIC:
        na= int((anom.lum_class== cls).sum())
        nc= int((ctrl.lum_class== cls).sum())
        print(f"{cls:10s}{na:12,}{nc:12,}")

def plot(anom,ctrl,field,path):
    fig,axes= plt.subplots(1,2,figsize=(14,7))

    for ax,direction,title in (
        (axes[0],("dwarf","evolved"),"dwarf called evolved\n(too bright for its colour)"),
        (axes[1],("evolved","dwarf"),"evolved called dwarf\n(too faint for its colour)") ):
        true_cls,pred_cls= direction
        s= anom[(anom.lum_class== true_cls) & (anom.predicted== pred_cls)]
        ax.hexbin(field.bp_rp,field.mg,gridsize=260,bins="log",cmap="Greys",mincnt=1,alpha=0.45,linewidths=0)
        c= ctrl[ctrl.lum_class== true_cls]
        ax.scatter(c.bp_rp,c.mg,s=8,c="#3fb8af",alpha=0.5,label=f"controls ({len(c):,})",linewidths=0)
        ax.scatter(s.bp_rp,s.mg,s=26,c="#ff6b6b",edgecolors="k",linewidths=0.3,label=f"anomalies ({len(s):,})")
        ax.set_xlabel(r"$G_{BP} - G_{RP}$")
        ax.set_title(title)
        ax.set_xlim(0.3,2.0)
        ax.set_ylim(9,-2)
        ax.legend(fontsize=9,loc="lower left")

    axes[0].set_ylabel(r"$M_G$")
    fig.suptitle("Confident misclassifications and matched controls")
    fig.savefig(path,dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

def main() -> None:

    for p in (OOF_IN,FEATURES_IN):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run the earlier modules first.")

    df= pd.read_parquet(OOF_IN)
    print(f"Loaded {len(df):,} out-of-fold predictions.")

    train= pd.read_parquet(FEATURES_IN)
    train= train[train.lum_class.isin(df.lum_class.unique())]
    dom= domain_measure(train)
    describe(dom)

    n0= len(df)
    df= df[df.lum_class.isin(SPECTROSCOPIC)]
    print("\nfiltering")
    print(f"spectroscopic classes only: {len(df):,} of {n0:,}")

    df["in_domain"]= in_domain(df,dom)
    df= df[df.in_domain]
    print(f"inside the training domain: {len(df):,}")

    wrong= df[df.predicted != df.lum_class]
    print(f"misclassified: {len(wrong):,}")

    anom= wrong[wrong.confidence > CONFIDENCE].copy()
    print(f"confident (p > {CONFIDENCE}): {len(anom):,}  (Anomalies)")


    print("\ndirection:")
    for true_cls,pred_cls in (("dwarf","evolved"),("evolved","dwarf")):
        n= int(((anom.lum_class== true_cls) & (anom.predicted== pred_cls)).sum())
        print(f"{true_cls:8s} -> {pred_cls:8s}: {n:>4,}")

    correct= df[(df.predicted== df.lum_class) &(~df.source_id.isin(anom.source_id))]
    print(f"\nControl pool: {len(correct):,} correctly classified in-domain stars.")
    print(f"Matching each anomaly to {N_CONTROLS} controls of the same true class")
    print(f"within {TOL_COLOUR} mag in colour and {TOL_MG} mag in M_G")

    ctrl= controls_match(anom,correct)
    matched= ctrl.matched_to.nunique() if len(ctrl) else 0
    print(f"{len(ctrl):,} controls matched to {matched:,} of {len(anom):,} anomalies ({100*matched/max(len(anom),1):.1f}%)")


    #Unmatched anomalies are dropped
    anom= anom[anom.source_id.isin(ctrl.matched_to)] if len(ctrl) else anom
    balance_report(anom,ctrl)

    print("\nwithheld diagnostics:")
    for col in ("ruwe","mh_gspspec","v_tan","phot_bp_rp_excess_factor"):
        if col not in anom.columns:
            continue
        a= anom[col].dropna()
        c= ctrl[col].dropna()
        if len(a) < 10 or len(c) < 10:
            continue
        print(f"{col:26s}: anomalies {a.median():8.3f}  controls {c.median():8.3f} ratio {a.median()/c.median():5.2f}")

    field= pd.read_parquet(FEATURES_IN,columns=["bp_rp","mg"])
    plot(anom,ctrl,field,FIG)

    anom.to_parquet(ANOM_OUT,index=False)
    ctrl.to_parquet(CTRL_OUT,index=False)
    print(f"Wrote {ANOM_OUT}  ({len(anom):,} anomalies)")
    print(f"Wrote {CTRL_OUT}  ({len(ctrl):,} controls)")

    config.log_provenance(
        "anomalies",
        n_anomalies=len(anom),n_controls=len(ctrl),
        confidence_threshold=CONFIDENCE,
        n_controls_per_anomaly=N_CONTROLS,
        tolerance_colour=TOL_COLOUR,tolerance_mg=TOL_MG,
        filters="in-domain,spectroscopic classes only,confident errors",
        note="controls matched on colour and M_G within true class",)

if __name__== "__main__":
    main()
