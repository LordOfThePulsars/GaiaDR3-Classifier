import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import config

FIELD_IN= config.DATA_RAW/"gaia_field.parquet"
LABELS_IN= config.DATA_PROCESSED/"labels.parquets"
OUTPUT= config.DATA_PROCESSED/"features.parquet"

#Measured by white dwarf cut calibration as density median
WD_INTERCEPT= 6.13
WD_COLOUR_MAX= 1.5
WD_SLOPE= 2.27

#Single copy of these formulas
def derive(df: pd.DataFrame) -> pd.DataFrame:
    df["mg"]= df.phot_g_mean_mag + 5.0 * np.log10(df.parallax) - 10.0
    df["v_tan"]= config.KAPPA * df.pm/ df.parallax
    df["log_v_tan"]= np.log10(df.v_tan.clip(lower=0.1))
    df["dist_pc"]= 1000.0 / df.parallax
    return df

#Geometric cut in the HR diagram used only when no spectroscopic labels exist
def flag_white_dwarfs(df: pd.DataFrame) -> pd.Series:
    return (df.mg > WD_SLOPE * df.bp_rp + WD_INTERCEPT) & (df.bp_rp < WD_COLOUR_MAX)

def diagnostics(df: pd.DataFrame,labelled: pd.DataFrame) -> None:
    print("Feature diagnostics")
    print(f"\nField sample: {len(df):,}")
    print(f"With a class label: {len(labelled):,} ({100*len(labelled)/len(df):.1f}%)")

    print("\nDerived feature ranges")
    for c in ("mg","bp_rp","v_tan","log_v_tan"):
        v= df[c].dropna()
        print(f"{c:12s}: {v.min():8.2f} to {v.max():8.2f}   median {v.median():7.2f}")
    print(f"\nv_tan median: {df.v_tan.median():.1f} km/s")
    print(f"v_tan 99th pct: {df.v_tan.quantile(0.99):.1f} km/s")

    print("\nClass balance:")
    counts= labelled.lum_class.value_counts()
    for cls,n in counts.items():
        print(f"{cls:10s}: {int(n):>7,}  ({100*n/len(labelled):5.1f}%)")

    if len(counts) > 1:
        ratio= counts.max() / counts.min()
        print(f"\nImbalance ratio: {ratio:.0f}:1")

    print("\nMedian feature values by class")
    hdr= "{:10s}" + "{:>12s}" * len(config.FEATURES)
    print(hdr.format("class",*config.FEATURES))
    for cls in counts.index:
        s= labelled[labelled.lum_class== cls]
        row= "{:10s}".format(cls) + "".join(f"{s[f].median():12.2f}"for f in config.FEATURES)
        print(row)

    #Nulls here that would leak their way in training
    nulls= labelled[config.FEATURES].isna().sum()
    bad= nulls[nulls > 0]
    print(f"\nNulls in feature: {'none' if bad.empty else dict(bad)}")

    #Withheld from training, to be used later on
    print("\nwithheld diagnostics")
    for c in config.DIAGNOSTICS:
        if c in labelled.columns:
            cov= 100.0 * labelled[c].notna().sum() / len(labelled)
            print(f"{c:28s}: {cov:5.1f}%  coverage")

def main() -> None:
    for p in (FIELD_IN,LABELS_IN):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Redo earlier modules first.")

    df= pd.read_parquet(FIELD_IN)
    print(f"Field sample: {len(df):,} stars")
    lab= pd.read_parquet(LABELS_IN)
    print(f"Labels: {len(lab):,} stars")
    df= derive(df)
    is_wd= flag_white_dwarfs(df)
    print(f"White dwarf candidates: {int(is_wd.sum()):,}")

    df= df.merge(lab,on="source_id",how="left")
    df["lum_class"]= df["lum_class"].fillna("unlabelled")
    wd_and_spec= int((is_wd & (df.lum_class != "unlabelled")).sum())

    if wd_and_spec:
        print(f"{wd_and_spec} accounting for WD that have spectra.")
    df.loc[is_wd & (df.lum_class== "unlabelled"),"lum_class"]= "white_dwarf"
    labelled= df[~df.lum_class.isin(["unlabelled","ambiguous"])].copy()

    df.to_parquet(OUTPUT,index=False)
    print(f" \nWrote {OUTPUT}  ({OUTPUT.stat().st_size/1e6:.1f} MB)")

    config.log_provenance(
        "features",
        rows=len(df),
        labelled_rows=len(labelled),
        class_counts={k: int(v) for k,v in labelled.lum_class.value_counts().items()},
        features=config.FEATURES,
        diagnostics_withheld=config.DIAGNOSTICS,
        wd_selection=f"mg > {WD_SLOPE}*bp_rp + {WD_INTERCEPT},bp_rp < {WD_COLOUR_MAX}", output=str(OUTPUT),)

if __name__== "__main__":
    main()
