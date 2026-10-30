import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from build_features import WD_COLOUR_MAX, WD_INTERCEPT, WD_SLOPE

FEATURES_IN= config.DATA_PROCESSED/"features.parquet"
FIG_MAIN= config.FIGURES/"hr_diagram.png"
FIG_CLASSES= config.FIGURES/"hr_classes.png"
FIG_WD= config.FIGURES/"hr_white_dwarf_cut.png"
FIG_BINARY= config.FIGURES/"hr_binary_sequence.png"

CLASS_COLOURS= {"dwarf": "#2dfaec","evolved": "#A27D1D","white_dwarf": "#8b7ff0","ambiguous": "#888888",}

#Equal-mass pair is 2.5*log10(2) mag brighter, no unresolved binary exceeds it.
BINARY_OFFSET= -0.753

def _style(ax):
    ax.set_xlabel(r"$G_{BP}-G_{RP}$ [mag]")
    ax.set_ylabel(r"$M_G$  [mag]")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()

def field_plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax= plt.subplots(figsize=(8.5, 10))
    hb= ax.hexbin(df.bp_rp, df.mg, gridsize=380,bins="log",cmap="magma", mincnt=1, linewidths=0)
    _style(ax)
    ax.set_title(f"Gaia DR3 Sample Hertzsprung-Russell diagram\n {len(df):,} stars within 200 pc")
    fig.colorbar(hb,ax=ax,label="stars per bin",pad=0.02)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

#Field density in grey with the labelled classes
def classes_plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax= plt.subplots(figsize=(9, 10))
    ax.hexbin(df.bp_rp, df.mg, gridsize=340, bins="log",cmap="Greys", mincnt=1, alpha=0.55, linewidths=0)

    #Large classes are subsampled 
    for cls in ("dwarf", "evolved", "white_dwarf"):
        s= df[df.lum_class== cls]
        if not len(s):
            continue
        if len(s)>12_000:
            s= s.sample(12_000, random_state=0)
        ax.scatter(s.bp_rp, s.mg, s=1.6, alpha=0.45,c=CLASS_COLOURS[cls],label=f"{cls} (n={int((df.lum_class==cls).sum()):,})",linewidths=0, rasterized=True)
    _style(ax)
    leg= ax.legend(loc="lower left", framealpha=0.92, markerscale=8)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax.set_title("Labelled classes over the field density")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

def wd_cut_plot(df: pd.DataFrame, path: Path) -> None:
    fig, axes= plt.subplots(1, 2, figsize=(14, 6.5))
    m= df.bp_rp < 2.0
    hb= axes[0].hexbin(df.loc[m, "bp_rp"], df.loc[m, "mg"], gridsize=260,bins="log", cmap="magma", mincnt=1, linewidths=0)
    xx= np.linspace(-0.7, WD_COLOUR_MAX, 200)
    axes[0].plot(xx, WD_SLOPE * xx + WD_INTERCEPT, color="#3fb8af", lw=2.0,label=f"WD cut: $M_G$ > {WD_SLOPE}(BP-RP) + {WD_INTERCEPT}")
    axes[0].axvline(WD_COLOUR_MAX, color="#3fb8af", lw=1.4, ls="--",label=f"colour limit {WD_COLOUR_MAX}")
    axes[0].legend(loc="lower left", fontsize=9, framealpha=0.9)
    axes[0].set_title("White dwarf selection boundary")
    _style(axes[0])
    fig.colorbar(hb, ax=axes[0], label="stars per bin")

    resid= df.mg - (WD_SLOPE*df.bp_rp + WD_INTERCEPT)
    sel= df.bp_rp < WD_COLOUR_MAX
    axes[1].hist(resid[sel], bins=300, range=(-12, 8), color="#8b7ff0", log=True)
    axes[1].axvline(0, color="#3fb8af", lw=2.0, label="the cut")
    axes[1].set_xlabel(r"$M_G$ - (cut line)  [mag]")
    axes[1].set_ylabel("stars (log scale)")
    axes[1].set_title("Residual about white dwarf cut line")
    axes[1].legend()

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

def binary_sequence_plot(df: pd.DataFrame, path: Path)-> None:
    fig, axes= plt.subplots(1, 2, figsize=(14, 6.5))

    m= (df.bp_rp.between(0.6, 2.2)) & (df.mg.between(3.0, 10.0))
    s= df[m]
    hb= axes[0].hexbin(s.bp_rp, s.mg, gridsize=300, bins="log",cmap="magma", mincnt=1, linewidths=0)
    axes[0].set_title(f"Main sequence zoom ({len(s):,} stars)")
    _style(axes[0])
    fig.colorbar(hb, ax=axes[0], label="stars per bin")

    #Local median M_G per colour bin
    bins= np.arange(0.6, 2.21, 0.05)
    idx= np.digitize(s.bp_rp, bins)
    ridge= s.groupby(idx).mg.transform("median")
    resid= s.mg - ridge

    axes[1].hist(resid, bins=200, range=(-2.0, 2.0), color="#8b7ff0", log=True)
    axes[1].axvline(0, color="#3fb8af", lw=1.6, label="single-star ridge")
    axes[1].axvline(BINARY_OFFSET, color="#ff6b6b", lw=1.6, ls="--",label=f"equal-mass binary ({BINARY_OFFSET} mag)")
    axes[1].set_xlabel(r"$M_G$ - local median   [mag]")
    axes[1].set_ylabel("stars (log scale)")
    axes[1].set_title("Residual about Main Sequence ridge")
    axes[1].legend(fontsize=9)

    #Magnitudes run backwards negative residual means a brighter ridge
    n_above= int((resid < -0.5).sum())
    print(f"stars >0.5 mag above the local ridge: {n_above:,} ({100*n_above/len(s):.1f}% of the zoom region)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")

def main() -> None:
    if not FEATURES_IN.exists():
        raise SystemExit(f"Missing {FEATURES_IN}. Run the feature assignment phase first.")
    df= pd.read_parquet(FEATURES_IN)
    df= df[df.mg.notna() & df.bp_rp.notna()]
    print(f"stars with colour and magnitude: {len(df):,}")

    field_plot(df, FIG_MAIN)
    classes_plot(df, FIG_CLASSES)
    wd_cut_plot(df, FIG_WD)
    binary_sequence_plot(df, FIG_BINARY)

    clump= df[(df.bp_rp.between(1.0, 1.4)) & (df.mg.between(-0.5, 1.5))]
    print(f"\nRed clump: {len(clump):,} stars")
    gap= df[(df.bp_rp.between(0.0, 1.0)) & (df.mg.between(8.0, 10.0))]
    wd= df[df.lum_class== "white_dwarf"]
    print(f"White dwarfs: {len(wd):,} stars")
    print(f"Gap region: {len(gap):,} stars")
    print(f"Gap/WD ratio: {len(gap)/max(len(wd),1):.3f}")

if __name__== "__main__":
    main()
