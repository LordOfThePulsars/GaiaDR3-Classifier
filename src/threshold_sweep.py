import contextlib
import io
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
import attribute
import build_features
import build_labels_mod
import calibrate
import extract_anomalies

DELTAS= [-0.25, -0.35, -0.45, -0.55, -0.65]
CONFIDENCES= [0.80, 0.90, 0.99]

OUT_CSV= config.DATA_PROCESSED / "threshold_sweep.csv"
FIG= config.FIGURES / "threshold_sweep.png"

#Both namespaces one doing labels and other highlighting boundaries
def set_delta(value: float) -> None:
    build_labels_mod.SUBGIANT_DELTA= value
    attribute.SUBGIANT_DELTA= value

def set_confidence(value: float) -> None:
    extract_anomalies.CONFIDENCE= value

#Labels, features and calibration depend only on delta.
def chain_run(quiet: bool= True) -> dict:
    sink= io.StringIO()
    ctx= contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    out= {}
    with ctx:
        build_labels_mod.main()
        build_features.main()
        calibrate.main()
        for conf in CONFIDENCES:
            set_confidence(conf)
            extract_anomalies.main()
            attribute.main()
            rec= {}
            for name in ("labels", "features", "calibration", "attribution"):
                q= config.LOGS / f"{name}.provenance.json"
                if q.exists():
                    rec[name]= json.loads(q.read_text())
            out[conf]= rec
    return out

def collect(rec: dict) -> dict:
    lab= rec.get("labels", {}).get("class_counts", {})
    feat= rec.get("features", {}).get("class_counts", {})
    att= rec.get("attribution", {})
    causes= att.get("causes", {})
    n_att= att.get("n_anomalies", 0) or 1

    row= {
        "n_dwarf": lab.get("dwarf", 0),"n_evolved": lab.get("evolved", 0),
        "n_ambiguous": lab.get("ambiguous", 0),"n_wd": feat.get("white_dwarf", 0),
        "n_anomalies": att.get("n_anomalies", 0),"unexplained": att.get("unexplained_fraction", np.nan),}
    for k, v in causes.items():
        row[f"frac_{k.replace(' ', '_')}"]= v / n_att
    return row

def plot(df: pd.DataFrame, path: Path) -> None:
    frac_cols= [c for c in df.columns if c.startswith("frac_")]
    confs= sorted(df.confidence.unique())
    fig, axes= plt.subplots(1, 3, figsize=(17, 5))

    base= df[df.confidence == confs[0]].sort_values("delta")
    axes[0].plot(base.delta, base.n_evolved, "o-", color="#ffd166", label="evolved")
    axes[0].plot(base.delta, base.n_ambiguous, "s-", color="#888888", label="ambiguous")
    axes[0].set_ylabel("stars")
    axes[0].set_title("Class balance")
    axes[0].legend(fontsize=9)

    for c in confs:
        sub= df[df.confidence == c].sort_values("delta")
        axes[1].plot(sub.delta, sub.n_anomalies, "o-", label=f"p > {c:.2f}")
    axes[1].axhline(60, color="#ff6b6b", ls=":", lw=1.2)
    axes[1].set_ylabel("confident errors")
    axes[1].set_title("Anomaly count")
    axes[1].legend(fontsize=9)

    mid= confs[len(confs) // 2]
    sub= df[df.confidence == mid].sort_values("delta")
    for c in frac_cols:
        axes[2].plot(sub.delta, 100 * sub[c], "o-", label=c[5:].replace("_", " ")[:28])
    axes[2].set_ylabel("% of anomalies")
    axes[2].set_title(f"Attributed cause (p > {mid:.2f})")
    axes[2].legend(fontsize=7)

    for ax in axes:
        ax.set_xlabel(r"SUBGIANT_DELTA  [dex]")

    fig.suptitle("Sensitivity to the dwarf/evolved threshold and confidence cut")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    original= build_labels_mod.SUBGIANT_DELTA
    print(f"Sweeping SUBGIANT_DELTA over {DELTAS}")
    print(f"Current value: {original}\n")

    rows= []
    for d in DELTAS:
        set_delta(d)
        try:
            per_conf= chain_run()
        except Exception as exc:
            print(f"delta {d:+.2f}: FAILED ({type(exc).__name__}: {exc})")
            continue
        gathered= {conf: collect(rec) for conf, rec in per_conf.items()}
        for conf, row in gathered.items():
            row["delta"]= d
            row["confidence"]= conf
            rows.append(row)

        line= "  ".join(f"p>{c:.2f}: {r['n_anomalies']:>4,}" for c, r in gathered.items())
        n_evolved= next(iter(gathered.values()))["n_evolved"]
        print(f"delta {d:+.2f}  evolved {n_evolved:>6,}  {line}")

    set_delta(original)
    if not rows:
        raise SystemExit("No successful runs.")

    df= pd.DataFrame(rows).sort_values(["confidence", "delta"]).reset_index(drop=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")

    print("\nclass balance and anomalies:")
    print(df[["delta", "n_dwarf", "n_evolved", "n_ambiguous", "n_anomalies", "unexplained"]].to_string(index=False))

    frac_cols= [c for c in df.columns if c.startswith("frac_")]
    spread= {c: float(df[c].max() - df[c].min()) for c in frac_cols}
    if frac_cols:
        print("\nattributed cause fractions:")
        print(df[["confidence", "delta"] + frac_cols].round(3).to_string(index=False))
        worst= max(spread, key=spread.get)
        print(f"\nlargest swing: {worst[5:].replace('_',' ')}  {100*spread[worst]:.1f} pp")

    plot(df, FIG)
    print(f"Wrote {FIG}")

    config.log_provenance(
        "threshold_sweep",deltas=DELTAS, adopted=original, cause_spread={k: round(v, 4) for k, v in spread.items()}, )


if __name__ == "__main__":
    main()
