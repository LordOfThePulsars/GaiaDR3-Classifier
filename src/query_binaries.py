import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config
from query_gaia import (COLUMNS as SOURCE_COLUMNS, connect,random_index_max_load, run_with_retry)
from query_labels import COLUMNS as LABEL_COLUMNS

#Complement of the Phase 1 cut
RUWE_MIN= 1.4
RUWE_MAX= 3.0

ROWS_PER_CHUNK_SOURCES= 150_000
ROWS_PER_CHUNK_LABELS= 25_000
FALLBACK_CHUNKS= 8

SOURCES_OUT= config.DATA_RAW/"gaia_binaries.parquet"
LABELS_OUT= config.DATA_RAW/"gsp_spec_binaries.parquet"
CHUNK_SOURCES= config.DATA_RAW/"chunks_binaries"
CHUNK_LABELS= config.DATA_RAW/"chunks_binary_labels"

#Prefix lets one clause serve the plain query
def _where(rand_max: int, prefix: str= "") -> str:
    p= f"{prefix}." if prefix else ""
    lo, hi= config.EXCESS_FACTOR_RANGE
    return f"""
        {p}parallax > {config.PARALLAX_MIN}
        AND {p}parallax_over_error > {config.PARALLAX_OVER_ERROR}
        AND {p}ruwe >= {RUWE_MIN} AND {p}ruwe <= {RUWE_MAX}
        AND {p}phot_bp_rp_excess_factor BETWEEN {lo} AND {hi} AND {p}bp_rp IS NOT NULL
        AND {p}phot_g_mean_mag IS NOT NULL AND {p}random_index < {rand_max}"""

def count_sources(rand_max: int) -> str:
    return f"SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE {_where(rand_max)}"

def count_labels(rand_max: int) -> str:
    return f"""
        SELECT COUNT(*) AS n
        FROM gaiadr3.gaia_source AS gs
        JOIN gaiadr3.astrophysical_parameters AS ap
          ON gs.source_id= ap.source_id
        WHERE {_where(rand_max, prefix='gs')}
          AND ap.teff_gspspec IS NOT NULL """

def chunk_sources(rand_max: int, lo: int, hi: int) -> str:
    return f"""
        SELECT {SOURCE_COLUMNS}
        FROM gaiadr3.gaia_source
        WHERE {_where(rand_max)}
          AND random_index >= {lo} AND random_index < {hi}"""

def chunk_labels(rand_max: int, lo: int, hi: int) -> str:
    return f"""
        SELECT {LABEL_COLUMNS}
        FROM gaiadr3.gaia_source AS gs
        JOIN gaiadr3.astrophysical_parameters AS ap
          ON gs.source_id= ap.source_id
        WHERE {_where(rand_max, prefix='gs')}
          AND gs.random_index >= {lo} AND gs.random_index < {hi}
          AND ap.teff_gspspec IS NOT NULL """

def fetch(gaia, builder, rand_max, edges, chunk_dir: Path, tag: str):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    frames, n= [], len(edges) - 1
    for i in range(n):
        path= chunk_dir / f"{tag}_{i:03d}.parquet"
        if path.exists():
            part= pd.read_parquet(path)
            print(f"[{i+1:2d}/{n}] cached   {len(part):>7,} rows")
            frames.append(part)
            continue
        q= builder(rand_max, edges[i], edges[i + 1])
        job= run_with_retry(lambda: gaia.launch_job_async(q))
        part= job.get_results().to_pandas()
        part.to_parquet(path, index=False)
        frames.append(part)
        print(f"[{i+1:2d}/{n}] fetched {len(part):>7,} rows")
    return pd.concat(frames, ignore_index=True)

#Count only sizes the chunk plan
def counted(gaia, query: str, label: str) -> int | None:
    print(f"\nCounting {label}")
    try:
        job= run_with_retry(lambda: gaia.launch_job_async(query), attempts=2)
        n= int(job.get_results()["n"][0])
        print(f"{n:,}")
        return n
    except Exception as exc:
        print(f"Count failed ({type(exc).__name__}); using a fixed {FALLBACK_CHUNKS}-chunk plan.")
        return None

def main() -> None:
    print("high-RUWE complement")
    rand_max= random_index_max_load()
    print(f"Fetching the same volume with RUWE between {RUWE_MIN} and {RUWE_MAX}.")

    gaia, _= connect()
    total= counted(gaia, count_sources(rand_max), f"stars with RUWE {RUWE_MIN}-{RUWE_MAX}")
    n_chunks= (FALLBACK_CHUNKS if total is None else max(1, int(np.ceil(total / ROWS_PER_CHUNK_SOURCES))))
    edges= [int(round(x)) for x in np.linspace(0, rand_max, n_chunks + 1)]

    print(f"\nDownloading sources in {n_chunks} chunks")
    src= fetch(gaia, chunk_sources, rand_max, edges, CHUNK_SOURCES, "bin")
    src= src.drop_duplicates(subset="source_id").reset_index(drop=True)
    src.to_parquet(SOURCES_OUT, index=False)
    print(f"Wrote {SOURCES_OUT}  ({len(src):,} stars, {SOURCES_OUT.stat().st_size/1e6:.1f} MB)")

    total_lab= counted(gaia, count_labels(rand_max), "those with GSP-Spec labels")
    n_chunks_lab= (FALLBACK_CHUNKS if total_lab is None else max(1, int(np.ceil(total_lab / ROWS_PER_CHUNK_LABELS))))
    edges_lab= [int(round(x)) for x in np.linspace(0, rand_max, n_chunks_lab + 1)]

    print(f"\nDownloading labels in {n_chunks_lab} chunks")
    lab= fetch(gaia, chunk_labels, rand_max, edges_lab, CHUNK_LABELS, "binlab")
    lab= lab.drop_duplicates(subset="source_id").reset_index(drop=True)
    lab.to_parquet(LABELS_OUT, index=False)
    print(f"Wrote {LABELS_OUT}  ({len(lab):,} labelled)")

    field= pd.read_parquet(config.DATA_RAW / "gaia_field.parquet",columns=["source_id"])
    print(f"\nGaia Query sample (RUWE < 1.4): {len(field):,}")
    print(f"Binary sample (RUWE 1.4-3): {len(src):,}")
    frac= len(src) / (len(src) + len(field))
    print(f"Binary fraction of the union: {100*frac:.1f}%")

    print("\nRUWE distribution:")
    for q in (0.25, 0.50, 0.75, 0.95):
        print(f"{int(q*100):2d}th percentile: {src.ruwe.quantile(q):.2f}")
    print(f"\nLabelled binaries: {len(lab):,} ({100*len(lab)/len(src):.1f}% of the binary sample)")
    flags= lab["flags_gspspec"].astype(str)
    strict= int(flags.str.startswith(config.STRICT_FLAG_PREFIX).sum())
    print(f"passing strict flags: {strict:,}")

    #Must be zero given their complementary nature
    overlap= len(set(src.source_id) & set(field.source_id))
    print(f"\nOverlap with Gaia Query: {overlap}")

    config.log_provenance(
        "gaia_binaries",
        sources=len(src), labelled=len(lab), strict_flag_rows=strict,
        ruwe_range=(RUWE_MIN, RUWE_MAX),
        random_index_max=rand_max,
        binary_fraction_of_union=float(frac),
        note="Complement of the gaia query ruwe cut, union of the two samples is the unbiased population.",)

if __name__== "__main__":
    main()
