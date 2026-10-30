import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import config

GAIA_DR3_NROWS= 1_811_709_771
OUTPUT= config.DATA_RAW/"gaia_field.parquet"
CHUNK_DIR= config.DATA_RAW/"chunks"
PROVENANCE= config.LOGS/"gaia_field.provenance.json"

ROWS_PER_CHUNK_AUTH= 250_000
ROWS_PER_CHUNK_ANON= 200_000

COLUMNS= """
    source_id, ra, dec, l, b, parallax, parallax_error, parallax_over_error,pmra, pmdec, pm, ruwe, astrometric_excess_noise, radial_velocity, radial_velocity_error,
    phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,bp_rp, bp_g, g_rp, phot_bp_rp_excess_factor, phot_variable_flag,non_single_star, random_index"""

#Shared by count and data queries
def where_clause() -> str:
    lo, hi= config.EXCESS_FACTOR_RANGE
    return f"""
        parallax > {config.PARALLAX_MIN} AND parallax_over_error > {config.PARALLAX_OVER_ERROR}
        AND ruwe < {config.RUWE_MAX} AND phot_bp_rp_excess_factor BETWEEN {lo} AND {hi}
        AND bp_rp IS NOT NULL AND phot_g_mean_mag IS NOT NULL"""

#Read back rather than hard-coded eto ensure ensure later phases cover same stars
def random_index_max_load() -> int:
    if not PROVENANCE.exists():
        raise SystemExit(f"Unable to find {PROVENANCE}.\nRun the query_gaia.py file again")
    rec= json.loads(PROVENANCE.read_text())
    span= rec.get("random_index_span")
    if span is None:
        raise SystemExit("Provenance file has no random_index_span, re-run the query_gaia.py file again")
    print(f"Gaia query phase used random_index <{span:,} ({rec['rows']:,} stars)")
    return int(span)

def count_query() -> str:
    return f"SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE {where_clause()}"

def chunk_query(lo: int, hi: int) -> str:
    return f"""
        SELECT {COLUMNS}
        FROM gaiadr3.gaia_source WHERE {where_clause()}
        AND random_index >= {lo} AND random_index < {hi} """

def connect():
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT= -1
    cred= config.GAIA_CREDENTIALS
    if cred.exists():
        try:
            Gaia.login(credentials_file= str(cred))
            print("authenticated")
            return Gaia, True
        except Exception as exc:
            print(f"Login has failed ({type(exc).__name__}), continuing anonymously.")
    else:
        print("anonymous")
    return Gaia, False

#Exponential backoff
def run_with_retry(fn, attempts: int= 4, base_delay: float= 15.0):
    for i in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if i== attempts:
                raise
            delay= base_delay * (2 ** (i - 1))
            print(f"Attempt {i} failed ({type(exc).__name__}), retrying in {delay:.0f}s")
            time.sleep(delay)
    return None

#Cached chunks, so a timeout costs one chunk and a restart resumes.
def chunks_fetch(gaia, edges: list[int]) -> pd.DataFrame:
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    frames= []
    n_chunks= len(edges) - 1
    for i in range(n_chunks):
        lo, hi= edges[i], edges[i + 1]
        path= CHUNK_DIR / f"chunk_{i:03d}.parquet"
        if path.exists():
            part= pd.read_parquet(path)
            print(f"[{i+1:3d}/{n_chunks}] cached {len(part):>7,} rows")
            frames.append(part)
            continue
        job= run_with_retry(lambda: gaia.launch_job_async(chunk_query(lo, hi)))
        part= job.get_results().to_pandas()
        part.to_parquet(path, index=False)
        frames.append(part)
        print(f"[{i+1:3d}/{n_chunks}] fetched {len(part):>7,} rows")
    return pd.concat(frames, ignore_index=True)

def diagnostics(df: pd.DataFrame) -> None:
    print("field sample")
    n= len(df)
    print(f"\nRows retrieved: {n:,}")
    dup= int(df.source_id.duplicated().sum())
    print(f"Duplicate source_id: {dup}")

    frac_err= 1.0 / df["parallax_over_error"]
    print(f"\nMedian fractional plx error: {np.median(frac_err)*100:.2f} %")
    print(f"90th percentile: {np.percentile(frac_err, 90)*100:.2f} %")

    dist= 1000.0 / df["parallax"]
    print(f"\nDistance range (parsecs): {dist.min():.1f} to {dist.max():.1f}")
    print(f"Median distance (parsecs): {np.median(dist):.1f}")

    print(f"\nG magnitude range: {df.phot_g_mean_mag.min():.2f} to {df.phot_g_mean_mag.max():.2f}")
    print(f"BP-RP colour range: {df.bp_rp.min():.2f} to {df.bp_rp.max():.2f}")

    print("\nnull counts:")
    for c in ["parallax", "bp_rp", "phot_g_mean_mag", "pm", "ruwe"]:
        nulls= int(df[c].isna().sum())
        print(f"{c:28s}: {nulls:,}")

    print("\ndiagnostic coverage:")
    for c in ["radial_velocity", "non_single_star"]:
        if c in df.columns:
            print(f"{c:28s}: {100.0*df[c].notna().sum()/n:.1f} %")

    #Narrow range here would mean the subsample is one sky patch
    print(f"\nSky coverage: RA {df.ra.min():.1f}-{df.ra.max():.1f} deg, Dec {df.dec.min():.1f} to {df.dec.max():.1f} deg")
    print(f"Median |b|: {np.median(np.abs(df.b)):.1f} deg")

def main() -> None:
    gaia, authed= connect()

    job= run_with_retry(lambda: gaia.launch_job(count_query()))
    total= int(job.get_results()["n"][0])
    radius= 1000.0 / config.PARALLAX_MIN
    print(f"{total:,} Sources within {radius:.0f} parsecs pass all cuts.")

    target= config.TARGET_ROWS
    if total <= target:
        frac= 1.0
        print(f"Below the {target:,} count")
    else:
        frac= min(1.0, 1.05 * target / total)   #5% over-request
        print(f"Above target, taking random {frac*100:.2f}% subsample.")
    span= int(frac * GAIA_DR3_NROWS)
    expected= min(total, int(target * 1.05))
    per_chunk= ROWS_PER_CHUNK_AUTH if authed else ROWS_PER_CHUNK_ANON
    n_chunks= max(1, int(np.ceil(expected / per_chunk)))
    edges= [int(round(x)) for x in np.linspace(0, span, n_chunks + 1)]
    mode= "authenticated" if authed else "anonymous"
    print(f"\nDownloading in {n_chunks} chunks, {mode} mode, About ~{per_chunk:,} rows each.")
    df= chunks_fetch(gaia, edges)
    print(f"\nRetrieved {len(df):,} rows.")
    df= df.drop_duplicates(subset="source_id").reset_index(drop=True)
    df.to_parquet(OUTPUT, index=False)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size/1e6:.1f} MB)")
    config.log_provenance(
        "gaia_field",
        rows=len(df),
        total_passing_cuts=total, subsample_fraction=frac,
        random_index_span=span,  n_chunks=n_chunks,
        authenticated=authed,parallax_min= config.PARALLAX_MIN,
        parallax_over_error= config.PARALLAX_OVER_ERROR,
        ruwe_max= config.RUWE_MAX, excess_factor_range= config.EXCESS_FACTOR_RANGE,output=str(OUTPUT), )

    print(f"Written provenance to {config.LOGS / 'gaia_field.provenance.json'}")
    diagnostics(df)

if __name__== "__main__":
    main()
