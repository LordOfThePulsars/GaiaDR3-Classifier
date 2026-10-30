import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

import config
from build_features import derive, flag_white_dwarfs
from domain import (describe, in_domain, domain_measure,prediction_is_supported)
from query_gaia import connect, run_with_retry
from train_model import EXTRAPOLATION_COLOUR

MODEL_IN= config.DATA_PROCESSED/"calibrated_model.pkl"
FIELD_IN= config.DATA_PROCESSED/"features.parquet"
RAW_OUT= config.DATA_RAW/"clusters.parquet"
OUT= config.DATA_PROCESSED/"cluster_predictions.parquet"
FIG= config.FIGURES/"cluster_hr.png"

CLUSTERS= {
    "Hyades": dict(
        ra=66.75, dec=15.87, radius_deg=15.0, parallax_range=(16.0, 30.0), pmra_range=(70.0, 140.0), pmdec_range=(-60.0, -5.0), age_myr=700, expected_giants=4,),
    "Pleiades": dict(
        ra=56.75, dec=24.12, radius_deg=5.0, parallax_range=(6.5, 8.3),pmra_range=(15.0, 25.0), pmdec_range=(-52.0, -38.0),age_myr=125, expected_giants=0,),}

COLUMNS= """
    source_id, ra, dec, l, b, parallax, parallax_error, parallax_over_error,
    pmra, pmdec, pm, ruwe, astrometric_excess_noise,radial_velocity, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
    bp_rp, bp_g, g_rp, phot_bp_rp_excess_factor, phot_variable_flag, non_single_star, random_index"""

#RUWE cut deliberately omitted so unresolved binaries survive.
def cluster_query(c: dict) -> str:
    return f"""
        SELECT {COLUMNS}
        FROM gaiadr3.gaia_source
        WHERE 1= CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {c['ra']}, {c['dec']}, {c['radius_deg']}))
          AND parallax BETWEEN {c['parallax_range'][0]} AND {c['parallax_range'][1]}AND pmra  BETWEEN {c['pmra_range'][0]}  AND {c['pmra_range'][1]}
          AND pmdec BETWEEN {c['pmdec_range'][0]} AND {c['pmdec_range'][1]}AND parallax_over_error > {config.PARALLAX_OVER_ERROR}
          AND phot_bp_rp_excess_factor BETWEEN {config.EXCESS_FACTOR_RANGE[0]} AND {config.EXCESS_FACTOR_RANGE[1]}AND bp_rp IS NOT NULL AND phot_g_mean_mag IS NOT NULL """

def fetch() -> pd.DataFrame:
    if RAW_OUT.exists():
        df= pd.read_parquet(RAW_OUT)
        print(f"Using cached {RAW_OUT} ({len(df):,} stars)")
        return df
    gaia, _= connect()
    frames= []
    for name, c in CLUSTERS.items():
        job= run_with_retry(lambda: gaia.launch_job_async(cluster_query(c)))
        part= job.get_results().to_pandas()
        part["cluster"]= name
        frames.append(part)
        print(f"{name:9s}: {len(part):>5,} candidate members")

    df= pd.concat(frames, ignore_index=True)
    df.to_parquet(RAW_OUT, index=False)
    config.log_provenance("clusters", rows=len(df), selection=CLUSTERS, note="astrometric selection; RUWE cut omitted so unresolved binaries survive")
    return df

#Inspection only; nothing is measured off this figure.
def plot(df: pd.DataFrame, field: pd.DataFrame, path: Path) -> None:
    fig, axes= plt.subplots(1, 2, figsize=(14, 7.5), sharex=True, sharey=True)
    colours= {"dwarf": "#3fb8af", "evolved": "#ffd166","white_dwarf": "#8b7ff0"}
    for ax, name in zip(axes, CLUSTERS):
        s= df[df.cluster== name]
        ax.hexbin(field.bp_rp, field.mg, gridsize=260, bins="log",cmap="Greys", mincnt=1, alpha=0.4, linewidths=0)
        for cls, col in colours.items():
            m= s.predicted== cls
            if m.sum():
                ax.scatter(s.loc[m, "bp_rp"], s.loc[m, "mg"], s=14, c=col,edgecolors="k", linewidths=0.2,label=f"{cls} ({int(m.sum())})")
        ax.axvline(EXTRAPOLATION_COLOUR, color="#ff6b6b", ls="--", lw=1.1,label=f"training limit (BP-RP {EXTRAPOLATION_COLOUR})")
        ax.set_xlabel(r"$G_{BP} - G_{RP}$")
        ax.set_title(f"{name} ({CLUSTERS[name]['age_myr']} Myr, n={len(s):,})")
        ax.legend(fontsize=8, loc="lower left")
        ax.set_xlim(-0.5, 4.0)
        ax.set_ylim(16, -2)

    axes[0].set_ylabel(r"$M_G$")
    fig.suptitle("Cluster members Classification over Field Density")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

def main() -> None:

    for p in (MODEL_IN, FIELD_IN):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run the earlier phases first.")

    with open(MODEL_IN, "rb") as fh:
        bundle= pickle.load(fh)
    model, classes= bundle["model"], bundle["classes"]
    print(f"Loading calibrated model, classes {classes}")

    df= derive(fetch())
    df= df.dropna(subset=config.FEATURES).reset_index(drop=True)

    proba= model.predict_proba(df[config.FEATURES].values)
    df["predicted"]= np.array(classes)[proba.argmax(axis=1)]
    df["confidence"]= proba.max(axis=1)
    df["is_wd_region"]= flag_white_dwarfs(df)

    train= pd.read_parquet(FIELD_IN)
    train= train[train.lum_class.isin(classes)]
    dom= domain_measure(train)
    describe(dom)

    df["in_domain"]= in_domain(df, dom)
    df["supported"]= prediction_is_supported(df, dom)

    for name, c in CLUSTERS.items():
        s_all= df[df.cluster== name]
        s= s_all[s_all.in_domain]
        print(f"\n{name}  ({c['age_myr']} Myr)")
        print(f"Members retrieved: {len(s_all):,}")
        print(f"Inside  training domain: {len(s):,}  ({100*len(s)/len(s_all):.1f}%)")
        print(f"Outside (model extrapolating): {len(s_all)-len(s):,}")
        print("\nin-domain predictions:")
        counts= s.predicted.value_counts()
        for cls in classes:
            n= int(counts.get(cls, 0))
            print(f"{cls:12s}: {n:>5,}  ({100*n/max(len(s),1):5.2f}%)")

        ev= s[(s.predicted== "evolved") & (~s.is_wd_region)]
        ev_conf= ev[ev.confidence > 0.9]
        print(f"\nevolved predictions:{len(ev):,} ({len(ev_conf):,} at confidence > 0.9)")
        print(f"expected giants: {c['expected_giants']}")

        red_ev= ev[ev.bp_rp > 0.7]
        blue_ev= ev[ev.bp_rp <= 0.7]
        print(f"of which red: {len(red_ev):,}")
        print(f"of which blue: {len(blue_ev):,}")

        if len(red_ev):
            print("brightest red evolved:")
            for _, r in red_ev.nsmallest(6, "mg").iterrows():
                print(f"G={r.phot_g_mean_mag:5.2f}  M_G={r.mg:5.2f} BP-RP={r.bp_rp:4.2f}  p={r.confidence:.3f}")

        if c["expected_giants"]== 0:
            fp= 100.0 * len(red_ev) / max(len(s), 1)
            print(f"\nfalse positive rate (red only): {fp:.2f}%")

        oo= s_all[~s_all.in_domain]
        if len(oo):
            print(f"\nOUT-OF-DOMAIN behaviour ({len(oo):,} stars:")
            for cls, n in oo.predicted.value_counts().items():
                print(f"  {cls:12s}: {int(n):>5,}")
            print(f"colour range: BP-RP {oo.bp_rp.min():.2f} to {oo.bp_rp.max():.2f}")
        unsupported= s_all[~s_all.supported]
        print(f"\nPredictions outside class envelope {len(unsupported):,} / {len(s_all):,} ({100*len(unsupported)/len(s_all):.1f}%)")
        if len(unsupported):
            for cls, n in unsupported.predicted.value_counts().items():
                sub= unsupported[unsupported.predicted== cls]
                print(f"{cls:12s}: {int(n):>5,} BP-RP {sub.bp_rp.min():.2f}-{sub.bp_rp.max():.2f}, M_G {sub.mg.min():.1f}-{sub.mg.max():.1f}")

    field= pd.read_parquet(FIELD_IN, columns=["bp_rp", "mg"])
    plot(df, field, FIG)

    keep= ["source_id", "cluster", "ra", "dec", "parallax", "pm", "ruwe", "phot_g_mean_mag", "bp_rp", "mg", "log_v_tan", "v_tan",
            "predicted", "confidence", "is_wd_region", "in_domain","supported",  "phot_bp_rp_excess_factor", "non_single_star", "radial_velocity"]
    df[[c for c in keep if c in df.columns]].to_parquet(OUT, index=False)
    print(f"Wrote {OUT}")

if __name__== "__main__":
    main()
