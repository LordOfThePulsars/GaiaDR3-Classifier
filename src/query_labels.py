import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import config
from query_gaia import (connect, random_index_max_load, run_with_retry, where_clause)

CHUNK_DIR= config.DATA_RAW/"chunks_labels"
OUTPUT= config.DATA_RAW/"gsp_spec_labels.parquet"

#Two-table join costs far more per row than say for example a single table scan
ROWS_PER_CHUNK= 25_000

#Used if the COUNT query itself times out
FALLBACK_CHUNKS= 12

#gspphot columns come from the photometry the model sees
COLUMNS= """
    ap.source_id,
    ap.teff_gspspec, ap.teff_gspspec_lower, ap.teff_gspspec_upper,ap.logg_gspspec, ap.logg_gspspec_lower, ap.logg_gspspec_upper,
    ap.mh_gspspec, ap.mh_gspspec_lower, ap.mh_gspspec_upper, ap.alphafe_gspspec, ap.flags_gspspec,ap.spectraltype_esphs, ap.teff_gspphot, ap.logg_gspphot, ap.mh_gspphot"""

STRICT_FLAG_PREFIX= config.STRICT_FLAG_PREFIX

#Restates the query_gaia.py cuts inside the join, so no table upload is needed
def count_query(rand_max: int) -> str:
    return f"""
        SELECT COUNT(*) AS n
        FROM gaiadr3.gaia_source AS gsJOIN gaiadr3.astrophysical_parameters AS ap
        ON gs.source_id= ap.source_id WHERE {where_clause()}
        AND ap.teff_gspspec IS NOT NULLAND gs.random_index < {rand_max}
    """

def chunk_query(rand_max: int, lo: int, hi: int)-> str:
    return f"""
        SELECT {COLUMNS}
        FROM gaiadr3.gaia_source AS gs JOIN gaiadr3.astrophysical_parameters AS ap
        ON gs.source_id= ap.source_id WHERE {where_clause()}
        AND ap.teff_gspspec IS NOT NULLAND gs.random_index < {rand_max}
        AND gs.random_index >= {lo} AND gs.random_index < {hi}
    """

#Same caching as the Gaia query, smaller chunks because of the join cost invovled within this query
def chunks_fetch(gaia, rand_max: int, edges: list[int])-> pd.DataFrame:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    frames= []
    n= len(edges)- 1

    for i in range(n):
        path= CHUNK_DIR/ f"labels_{i:03d}.parquet"
        if path.exists():
            part= pd.read_parquet(path)
            print(f"[{i+1:3d}/{n}] cached {len(part):>7,} rows")
            frames.append(part)
            continue
        q= chunk_query(rand_max, edges[i], edges[i + 1])
        job= run_with_retry(lambda: gaia.launch_job_async(q))
        part= job.get_results().to_pandas()
        part.to_parquet(path, index=False)
        frames.append(part)
        print(f"[{i+1:3d}/{n}] fetched  {len(part):>7,} rows")

    return pd.concat(frames, ignore_index=True)

def diagnostics_check(df: pd.DataFrame, field: pd.DataFrame)-> None:
    print("GSP-Spec labels")

    n_field, n= len(field), len(df)
    print(f"\nField Sample from initial Gaia Query: {n_field:,}")
    print(f"With the GSP-SPEC parameters: {n:,} ({100*n/n_field:.1f}%)")

    flags= df["flags_gspspec"].astype(str)
    strict= flags.str.startswith(STRICT_FLAG_PREFIX)
    print(f"Passing the Strict quality checks: {int(strict.sum()):,}  ({100*strict.mean():.1f}% of matched)")

    print("\nflag prefix length:")
    for k in (0,5,9,11,13):
        keep= flags.str.startswith("0"*k)  if k else flags.notna()
        print(f"first {k:2d} flags zero: {int(keep.sum()):>7,}")

    s= df[strict] if strict.sum() > 100 else df
    tag= "strict subset" if strict.sum() > 100 else "all matched"
    print(f"\nParameter ranges are ({tag}):")
    for col, unit in (("teff_gspspec","K"), ("logg_gspspec","dex"), ("mh_gspspec","dex")):
        v= s[col].dropna()
        if len(v):
            print(f"{col:20s}:{v.min():8.2f} to {v.max():8.2f} {unit} median {v.median():.2f}")

    #Indicative only given the fact that uncalibrated log g carries a known systematic offset.
    lg= s["logg_gspspec"].dropna()
    if len(lg):
        print("\nprovisional log-g split:")
        print(f"giant (log-g < 3.5):   {int((lg < 3.5).sum()):>7,}")
        print(f"subgiant (log-g 3.5-4.0):   {int(((lg >= 3.5) & (lg < 4.0)).sum()):>7,}")
        print(f"dwarf (log-g > 4.0):   {int((lg >= 4.0).sum()):>7,}")

    #RVS reaches only the bright end, so the labelled set is magnitude-limited.
    merged= field.merge(df[["source_id"]].drop_duplicates(), on="source_id", how="left",indicator=True)
    has= merged["_merge"]== "both"
    print("\nselection function:")
    print(f"G of labelled stars: median {field.loc[has.values, 'phot_g_mean_mag'].median():.2f}")
    print(f"G of unlabelled stars: median {field.loc[~has.values, 'phot_g_mean_mag'].median():.2f}")

    if "spectraltype_esphs" in df.columns:
        vc= df["spectraltype_esphs"].value_counts().head(6)
        if len(vc):
            print(f"\nESP-ELS spectral types: {dict(vc)}")

def main() -> None:
    field_path= config.DATA_RAW / "gaia_field.parquet"
    if not field_path.exists():
        raise SystemExit(f"Missing {field_path}. Re-Run The Gaia Query.")
    field= pd.read_parquet(field_path, columns=["source_id","phot_g_mean_mag"])
    print(f"Gaia Query sample:{len(field):,} stars.")
    rand_max= random_index_max_load()
    gaia, authed= connect()
    total= None
    try:
        job= run_with_retry(lambda: gaia.launch_job_async(count_query(rand_max)), attempts=2)
        total= int(job.get_results()["n"][0])
        print(f"{total:,} of {len(field):,} have spectroscopic parameters ({100*total/len(field):.1f}%).")
    except Exception as exc:
        print(f"Count failed ({type(exc).__name__}).")
        print(f"Using a fixed {FALLBACK_CHUNKS}chunk plan instead.")

    if total== 0:
        raise SystemExit("No possible matches, ensure the Gaia Query ran through properly.")

    if total is None:
        n_chunks= FALLBACK_CHUNKS
    else:
        n_chunks= max(1, int(np.ceil(total/ ROWS_PER_CHUNK)))
    edges= [int(round(x)) for x in np.linspace(0, rand_max, n_chunks + 1)]

    df= chunks_fetch(gaia, rand_max, edges)
    print(f"\nRetrieved {len(df):,} labelled stars.")
    if total is None:
        print(f"True coverage is  {100*len(df)/len(field):.1f}% of field sample")
    df.to_parquet(OUTPUT,index=False)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size/1e6:.1f} MB)")

    flags= df["flags_gspspec"].astype(str)
    config.log_provenance(
        "gsp_spec_labels",
        rows= len(df), field_rows= len(field), coverage_fraction= len(df) / len(field),
        strict_flag_rows= int(flags.str.startswith(STRICT_FLAG_PREFIX).sum()), random_index_max= rand_max,authenticated= authed,
        label_source= "GSP-Spec (RVS spectra)", output=str(OUTPUT))
    diagnostics_check(df, field)

if __name__== "__main__":
    main()
