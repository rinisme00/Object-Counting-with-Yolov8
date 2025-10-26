from pathlib import Path
import csv, os

def _read_results_csv(csv_path):
    rows = []
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"results.csv not found: {p}")
    with p.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean = {}
            for k, v in row.items():
                try:
                    clean[k] = float(v) if v not in ("", None) else v
                except Exception:
                    clean[k] = v
            rows.append(clean)
    return rows

def log_to_comet(results_csv, run_name="run", params=None,
                 api_key=None, workspace=None, project_name=None):
    try:
        from comet_ml import Experiment
    except Exception as e:
        print(f"[WARN] comet_ml not installed: {e}")
        return

    key = api_key or os.getenv("COMET_API_KEY")
    if not key:
        print("[WARN] COMET_API_KEY not set; skipping Comet logging.")
        return

    ws   = workspace    or os.getenv("COMET_WORKSPACE")
    proj = project_name or os.getenv("COMET_PROJECT_NAME", "ultralytics")

    exp = Experiment(api_key=key, workspace=ws, project_name=proj,
                     auto_param_logging=False, auto_metric_logging=False)
    exp.set_name(run_name)

    if params:
        exp.log_parameters(params)

    for r in _read_results_csv(results_csv):
        step = int(r.get("epoch", 0))
        for k, v in r.items():
            if k == "epoch" or not isinstance(v, (int, float)):
                continue
            exp.log_metric(k, v, step=step)

    exp.end()
    print("[LOGGER] Comet logging complete.")