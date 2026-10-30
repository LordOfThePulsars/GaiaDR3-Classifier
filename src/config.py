import json
import os
from datetime import datetime,timezone
from pathlib import Path

PROJECT_ROOT= Path(__file__).resolve().parent.parent
DATA_RAW= PROJECT_ROOT/"data"/"raw"
DATA_PROCESSED= PROJECT_ROOT/"data"/"processed"
DATA_EXTERNAL= PROJECT_ROOT/"data"/"external"
FIGURES= PROJECT_ROOT/"figures"
LOGS= PROJECT_ROOT/"logs"
for _d in (DATA_RAW,DATA_PROCESSED,DATA_EXTERNAL,FIGURES,LOGS):
    _d.mkdir(parents=True,exist_ok=True)

#Kept outside the repository so it can never be committed
GAIA_CREDENTIALS= Path(os.environ.get("GAIA_CREDENTIALS",Path.home()/".gaia_credentials"))

#Sample definition
PARALLAX_MIN= 5.0
PARALLAX_OVER_ERROR= 20
RUWE_MAX= 1.4
EXCESS_FACTOR_RANGE= (1.0,1.5)
TARGET_ROWS= 500_000

#Converting proper motion into tangential velocity
KAPPA= 4.740470446

#First 13 flag characters must be zero.
STRICT_FLAG_PREFIX= "0"*13

#Reference only
CLUSTER_DISTANCE_PC= {"Hyades": 47.0,"Pleiades": 136.2}

FEATURES= ["bp_rp","bp_g","g_rp","mg","log_v_tan"]
DIAGNOSTICS= ["ruwe","phot_bp_rp_excess_factor","phot_variable_flag","mh_gspspec","radial_velocity","b"]

def log_provenance(name: str,**metadata) -> Path:
    record= {"name": name,"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),**metadata}
    path= LOGS/f"{name}.provenance.json"
    path.write_text(json.dumps(record,indent=2,default=str))
    return path
