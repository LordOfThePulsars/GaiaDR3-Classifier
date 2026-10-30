import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS= []

def record(label: str, ok, detail: str= "") -> None:
    tag= {True: "Pass", False: "Fail", None: "Skip"}[ok]
    print(f"[{tag}] {label}" + (f"-- {detail}"if detail else ""))
    RESULTS.append((label, ok, detail))

print("\npython")
vi= sys.version_info
record(f"Python {vi.major}.{vi.minor}.{vi.micro}",
       vi >= (3, 10),
       ""if vi >= (3, 10) else "astroquery/astropy need >=3.10")

print("\npackages")
REQUIRED= ["numpy", "pandas", "scipy", "pyarrow","astropy", "astroquery","sklearn", "lightgbm","matplotlib"]
for mod in REQUIRED:
    try:
        m= importlib.import_module(mod)
        record(mod, True, getattr(m, "__version__", "version unknown"))
    except ImportError as exc:
        record(mod, False, str(exc))

print("\npaths")
config= None
try:
    import config
except Exception as exc:
    record("import config", False, str(exc))

if config is not None:
    for name in ("DATA_RAW", "DATA_PROCESSED", "DATA_EXTERNAL", "FIGURES", "LOGS"):
        p= getattr(config, name)
        probe= p / ".write_probe"
        try:
            probe.write_text("ok")
            probe.unlink()
            record(f"{name} writable", True, str(p))
        except OSError as exc:
            record(f"{name} writable", False, str(exc))

print("\ncredentials")
cred_ok= False
if config is not None:
    cred= config.GAIA_CREDENTIALS
    if not cred.exists():
        record("credentials file", False, f"not found at {cred}")
    else:
        lines= [ln for ln in cred.read_text().splitlines() if ln.strip()]
        if len(lines) < 2:
            record("credentials file", False, "expected username on line 1, password on line 2")
        else:
            mode= cred.stat().st_mode & 0o777
            if mode & 0o077:
                record("credentials permissions", False, f"mode {oct(mode)} -- run: chmod 600 {cred}")
            else:
                record("credentials permissions", True, oct(mode))
            cred_ok= True
            record("credentials file", True, f"user '{lines[0]}'")

print("\narchive")
gaia= None
try:
    from astroquery.gaia import Gaia
    gaia= Gaia
    job= Gaia.launch_job(
        "SELECT TOP 5 source_id, parallax, bp_rp "
        "FROM gaiadr3.gaia_source WHERE parallax > 100"
    )
    n= len(job.get_results())
    record("anonymous TAP query", n== 5, f"{n} rows returned")
except Exception as exc:
    record("anonymous TAP query", False, f"{type(exc).__name__}: {exc}")

print("\nlogin")
if gaia is None:
    record("login", None,"astroquery  unavailable")
elif not cred_ok:
    record("login", None,"no personal credentials file")
else:
    try:
        gaia.login(credentials_file=str(config.GAIA_CREDENTIALS))
        record("login", True, "authenticated")
        try:
            jobs= gaia.list_async_jobs()
            record("async job listing", True, f"{len(jobs)} existing job(s) on account")
        except Exception as exc:
            record("async job listing", None, str(exc))
        gaia.logout()
    except Exception as exc:
        record("login", False, f"{type(exc).__name__}: {exc}")

failed= [label for label, ok, _ in RESULTS if ok is False]
skipped= [label for label, ok, _ in RESULTS if ok is None]

if failed:
    print(f"failed: {', '.join(failed)}")
if skipped:
    print(f"skipped: {', '.join(skipped)}")
sys.exit(1 if failed else 0)
