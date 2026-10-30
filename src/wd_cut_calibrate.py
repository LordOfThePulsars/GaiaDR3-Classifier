import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import config
from build_features import WD_COLOUR_MAX

FEATURES_IN= config.DATA_PROCESSED / "features.parquet"
FIG= config.FIGURES / "wd_cut_calibration.png"

#Original hand-placed cut, kept only for the figure's 
ORIGINAL_SLOPE, ORIGINAL_INTERCEPT= 3.0, 10.0

COLOUR_RANGE= (-0.3, 1.3)
BIN_WIDTH= 0.1
MIN_PER_BIN= 200
MS_WINDOW= (0.0, 9.0)
WD_WINDOW= (9.5, 16.5)

#Shallow minimum means the sequences are not separate
MIN_DEPTH= 0.50
EDGE_TOL= 0.15

def find_valley(mg: np.ndarray) -> dict | None:
    mg= mg[(mg > -2) & (mg < 18)]
    if len(mg) < MIN_PER_BIN:
        return None

    grid= np.linspace(-1, 17, 700)
    dens= gaussian_kde(mg, bw_method=0.12)(grid)

    ms= (grid >= MS_WINDOW[0]) & (grid <= MS_WINDOW[1])
    wd= (grid >= WD_WINDOW[0]) & (grid <= WD_WINDOW[1])
    if not ms.any() or not wd.any():
        return None

    ms_peak, wd_peak= grid[ms][np.argmax(dens[ms])], grid[wd][np.argmax(dens[wd])]
    ms_h, wd_h= dens[ms].max(), dens[wd].max()

    between= (grid > ms_peak) & (grid < wd_peak)
    if between.sum() < 10:
        return None
    valley= grid[between][np.argmin(dens[between])]
    depth= 1.0 - dens[between].min() / min(ms_h, wd_h)

    if (abs(ms_peak - MS_WINDOW[1]) < EDGE_TOL or abs(ms_peak - MS_WINDOW[0]) < EDGE_TOL
            or abs(wd_peak - WD_WINDOW[0]) < EDGE_TOL or abs(wd_peak - WD_WINDOW[1]) < EDGE_TOL):
        return None
    if not np.isfinite(depth) or depth < MIN_DEPTH:
        return None

    return dict(ms_peak=ms_peak, wd_peak=wd_peak, valley=valley,depth=depth, n=len(mg))

def plot(df, v, slope, intercept, path):
    fig, axes= plt.subplots(1, 2, figsize=(14, 6))

    hb= axes[0].hexbin(df.bp_rp, df.mg, gridsize=260, bins="log", cmap="magma",mincnt=1, linewidths=0, extent=(-0.7, 2.0, -2, 17))
    xx= np.linspace(*COLOUR_RANGE, 100)
    axes[0].plot(xx, slope * xx + intercept, color="#3fb8af", lw=2.2,label=f"fitted: {slope:.2f}(BP-RP) + {intercept:.2f}")
    axes[0].plot(xx, ORIGINAL_SLOPE * xx + ORIGINAL_INTERCEPT, color="#ff6b6b",lw=1.6, ls="--", label=f"hand-chosen: {ORIGINAL_SLOPE}(BP-RP) + {ORIGINAL_INTERCEPT}")
    axes[0].scatter(v.colour, v.valley, s=34, c="#ffd166", zorder=5,edgecolors="k", linewidths=0.4, label="measured minima")
    axes[0].set_xlabel(r"$G_{BP} - G_{RP}$")
    axes[0].set_ylabel(r"$M_G$")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, loc="lower left")
    fig.colorbar(hb, ax=axes[0], label="stars per bin")

    #One representative bin, to show the bimodality the fit relies on
    mid= v.iloc[len(v) // 2]
    sub= df.loc[df.bp_rp.between(mid.colour - BIN_WIDTH / 2,mid.colour + BIN_WIDTH / 2), "mg"]
    grid= np.linspace(-1, 17, 700)
    axes[1].plot(grid, gaussian_kde(sub.values, bw_method=0.12)(grid),color="#8b7ff0", lw=1.8)
    axes[1].axvline(mid.valley, color="#3fb8af", lw=2.0,label=f"valley at $M_G$= {mid.valley:.2f}")
    for pk in (mid.ms_peak, mid.wd_peak):
        axes[1].axvline(pk, color="#888888", ls=":", lw=1.4)
    axes[1].set_xlabel(r"$M_G$")
    axes[1].set_ylabel("density")
    axes[1].set_title(f"BP-RP= {mid.colour:.2f}  (n= {int(mid.n):,})")
    axes[1].legend(fontsize=9)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    if not FEATURES_IN.exists():
        raise SystemExit(f"Missing {FEATURES_IN}. Run build_features.py first.")
    df= pd.read_parquet(FEATURES_IN, columns=["bp_rp", "mg"]).dropna()
    print(f"{len(df):,} stars")

    edges= np.arange(COLOUR_RANGE[0], COLOUR_RANGE[1] + BIN_WIDTH, BIN_WIDTH)
    rows, rejected= [], 0
    for lo, hi in zip(edges[:-1], edges[1:]):
        res= find_valley(df.loc[df.bp_rp.between(lo, hi), "mg"].values)
        if res is None:
            rejected += 1
            continue
        res["colour"]= 0.5 * (lo + hi)
        rows.append(res)

    if len(rows) < 4:
        raise SystemExit("Too few usable colour bins to fit a boundary.")
    v= pd.DataFrame(rows)

    print(f"{len(v)} bins usable, {rejected} rejected\n")
    print(f"{'BP-RP':>7s}{'n':>9s}{'MS':>8s}{'WD':>8s}{'valley':>9s}{'depth':>8s}")
    for _, r in v.iterrows():
        print(f"{r.colour:7.2f}{int(r.n):>9,}{r.ms_peak:8.2f}{r.wd_peak:8.2f}{r.valley:9.2f}{r.depth:8.3f}")

    slope, intercept= np.polyfit(v.colour, v.valley, 1, w=np.sqrt(v.n.values))
    rms= float(np.sqrt(np.mean((v.valley - (slope * v.colour + intercept))**2)))

    print(f"\nM_G= {slope:.3f}(BP-RP) + {intercept:.3f}   RMS {rms:.3f} mag")
    print(f"mean valley depth {v.depth.mean():.3f}")
    print(f"margin to MS {float((v.valley - v.ms_peak).mean()):.2f} mag, to WD {float((v.wd_peak - v.valley).mean()):.2f} mag")

    #Flat response here is what justifies trusting the single fitted value.
    print("\nsensitivity:")
    base= int(((df.mg > slope * df.bp_rp + intercept) & (df.bp_rp < WD_COLOUR_MAX)).sum())
    for shift in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
        n= int(((df.mg > slope * df.bp_rp + intercept + shift) & (df.bp_rp < WD_COLOUR_MAX)).sum())
        print(f"{intercept + shift:6.2f}: {n:>7,}  {100*(n-base)/base:+6.2f}%")

    plot(df, v, slope, intercept, FIG)
    print(f"\nWrote {FIG}")

    config.log_provenance(
        "wd_cut_calibration",
        fitted_slope=float(slope), fitted_intercept=float(intercept), rms_scatter=rms, mean_valley_depth=float(v.depth.mean()),
        n_bins=len(v), hand_chosen=(ORIGINAL_SLOPE, ORIGINAL_INTERCEPT),method="density minimum between bimodal peaks per colour bin",)

if __name__== "__main__":
    main()
