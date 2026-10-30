import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

import contextlib
import io
import json
import pickle
import numpy as np
import pandas as pd
from scipy import stats

import attribute
import build_features
import build_labels_mod
import calibrate
import config
import extract_anomalies
from domain import domain_measure

OUT_CSV= config.DATA_PROCESSED / "buffer_sweep.csv"

#Attribute file binds buffer at import
def set_buffer(value: float) -> None:
    build_labels_mod.BUFFER= value
    attribute.BUFFER= value

def chain_run() -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        build_labels_mod.main()
        build_features.main()
        calibrate.main()
        extract_anomalies.main()

#Pooling error rate either side of RUWE 1.4,
def dose_ratio() -> tuple:
    with open(config.DATA_PROCESSED/"calibrated_model.pkl","rb") as fh:
        bundle= pickle.load(fh)
    model,classes= bundle["model"],bundle["classes"]
    field= pd.read_parquet(config.DATA_PROCESSED / "features.parquet")
    dom= domain_measure(field[field.lum_class.isin(classes)])
    with contextlib.redirect_stdout(io.StringIO()):
        ridge= attribute.ms_ridge(field,verbose=False)
        dr,_= attribute.dose_response(model,classes,dom,field,ridge)
    if not len(dr):
        return np.nan,np.nan,np.nan,np.nan
    below,above= dr[dr.ruwe_hi <= 1.4],dr[dr.ruwe_lo >= 1.4]
    if not len(below) or not len(above):
        return np.nan,np.nan,np.nan,np.nan
    r_lo= below.n_err.sum() / below.n.sum()
    r_hi= above.n_err.sum() / above.n.sum()
    tab= [[int(above.n_err.sum()),int(above.n.sum() - above.n_err.sum())],[int(below.n_err.sum()),int(below.n.sum() - below.n_err.sum())]]
    _,p= stats.fisher_exact(tab)
    return 100*r_lo,100*r_hi,r_hi / max(r_lo,1e-12),p

def main() -> None:
    buf= float(sys.argv[1])
    set_buffer(buf)
    chain_run()

    lab= json.loads((config.LOGS/ "labels.provenance.json").read_text())
    anom= json.loads((config.LOGS/ "anomalies.provenance.json").read_text())
    counts= lab["class_counts"]
    rate_lo,rate_hi,ratio,p= dose_ratio()

    row= dict(buffer=buf,
              dwarf=counts.get("dwarf",0),evolved=counts.get("evolved",0),
              ambiguous=counts.get("ambiguous",0), trainable=lab["trainable_rows"],
              anomalies=anom["n_anomalies"],rate_below=rate_lo,rate_above=rate_hi,ratio=ratio,fisher_p=p)

    df= pd.DataFrame([row])
    if OUT_CSV.exists():
        df= pd.concat([pd.read_csv(OUT_CSV),df],ignore_index=True)
    df= df.drop_duplicates(subset="buffer",keep="last").sort_values("buffer")
    df.to_csv(OUT_CSV,index=False)

    print(f"{buf:.3f}{row['dwarf']:9,}{row['evolved']:8,}{row['ambiguous']:9,} "f"{row['anomalies']:7,}{rate_lo:8.2f}{rate_hi:8.2f}{ratio:8.1f}{p:11.1e}")

if __name__== "__main__":
    main()
