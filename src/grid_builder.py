import sqlite3
import numpy as np
import glob
import os
import json
import math
import yaml
from pathlib import Path
from datetime import datetime

# Import shared grid definition (supports both `python3 src/grid_builder.py` and `python -m` usage).
try:
    from src import orpi_grid as og
except Exception:
    import orpi_grid as og

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
SAMPLES_DIR = DATA_DIR / "samples"
DB_PATH = DATA_DIR / "orpi.db"
SCENARIO_DIR = DATA_DIR / "scenarios"
FEATURES_DIR = DATA_DIR / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Grid Definition
INC_BIN_SIZE = og.INC_BIN_SIZE_DEG    # degrees
RAAN_BIN_SIZE = og.RAAN_BIN_SIZE_DEG  # degrees

MU_EARTH_KM3_S2 = 398600.4418
EARTH_RADIUS_KM = 6378.135

def init_db():
    """Initialize SQLite database and table."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS orbital_cells (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT,
            alt_bin_start REAL,
            inc_bin_start REAL,
            raan_bin_start REAL,
            sample_count INTEGER,
            unique_objects INTEGER, -- Optional
            n_eff REAL,             -- Effective occupancy (mean objects in cell)
            vrel_proxy REAL,        -- Relative-energy proxy (km/s)
            occ_sigma REAL,         -- Occupancy std-dev across time window (objects)
            risk_sigma REAL,        -- Std-dev of (n*proxy) across time window
            trend_3y REAL,          -- Legacy: scenario delta (primary scenario file)
            trend_5y REAL,          -- Legacy: scenario delta (secondary scenario file, or fallback)
            trend_total REAL,       -- Delta risk at scenario target date (declarative)
            trend_annual REAL       -- Annualized delta risk (per year) assuming linear deployment
        )
    ''')
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS orpi_batches (
            batch_id TEXT PRIMARY KEY,
            created_at TEXT,
            scenario_file TEXT,
            scenario_name TEXT,
            scenario_target_date TEXT,
            scenario_years_to_target REAL,
            scenario_description TEXT,
            scenario_deployment_profile TEXT
        )
        """
    )
    # Lightweight "migration" for existing DBs.
    c.execute("PRAGMA table_info(orbital_cells)")
    cols = {row[1] for row in c.fetchall()}
    def add_col(name, sql_type):
        if name not in cols:
            c.execute(f"ALTER TABLE orbital_cells ADD COLUMN {name} {sql_type}")
    add_col("unique_objects", "INTEGER")
    add_col("n_eff", "REAL")
    add_col("vrel_proxy", "REAL")
    add_col("occ_sigma", "REAL")
    add_col("risk_sigma", "REAL")
    add_col("trend_3y", "REAL")
    add_col("trend_5y", "REAL")
    add_col("trend_total", "REAL")
    add_col("trend_annual", "REAL")
    c.execute('CREATE INDEX IF NOT EXISTS idx_batch ON orbital_cells (batch_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_pos ON orbital_cells (alt_bin_start, inc_bin_start, raan_bin_start)')
    conn.commit()
    return conn

def get_latest_samples():
    """Finds the latest .npz file."""
    list_of_files = glob.glob(str(SAMPLES_DIR / "samples_*.npz"))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)

def _edges(min_v, max_v, step):
    return np.arange(min_v, max_v + step, step, dtype=np.float32)

def _bin_start(value, min_v, step):
    # Align to our edges starting at min_v (not 0).
    return min_v + math.floor((value - min_v) / step) * step

def _v_circ_km_s(alt_km):
    r = EARTH_RADIUS_KM + alt_km
    return math.sqrt(MU_EARTH_KM3_S2 / r)

def _inc_factor(inc_centers_deg, weights, target_inc_deg):
    # Expected sin(dI/2) over the local inclination mixture at an altitude.
    if not weights or sum(weights) <= 0:
        return 0.0
    s = 0.0
    wsum = float(sum(weights))
    for inc, w in zip(inc_centers_deg, weights):
        d = abs(target_inc_deg - inc) * math.pi / 180.0
        s += float(w) * math.sin(d / 2.0)
    return s / wsum

def _load_scenario(scenario_path):
    if not scenario_path or not Path(scenario_path).exists():
        return {}
    with open(scenario_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _scenario_to_delta_cells(scenario, alt_edges, inc_edges, raan_edges):
    if not scenario:
        return {}
    consts = scenario.get("constellations", []) or []
    if not consts:
        return {}

    n_raan = len(raan_edges) - 1
    deltas = {}
    for c in consts:
        try:
            count = int(c.get("count", 0))
        except Exception:
            count = 0
        orbit = c.get("orbit", {}) or {}
        alt = float(orbit.get("altitude_km"))
        inc = float(orbit.get("inclination_deg"))
        if count <= 0:
            continue
        # Find bin indices.
        ai = int(np.digitize([alt], alt_edges)[0] - 1)
        ii = int(np.digitize([inc], inc_edges)[0] - 1)
        if ai < 0 or ai >= len(alt_edges) - 1:
            continue
        if ii < 0 or ii >= len(inc_edges) - 1:
            continue
        per = count / float(n_raan)
        for ki in range(n_raan):
            key = (ai, ii, ki)
            deltas[key] = deltas.get(key, 0.0) + per
    return deltas

def _years_to_target(target_date_str: str):
    if not target_date_str:
        return None
    s = str(target_date_str).strip()
    if not s:
        return None
    # Accept "YYYY-MM-DD" (recommended) or full ISO strings.
    for fmt in ("%Y-%m-%d", None):
        try:
            if fmt:
                target_dt = datetime.strptime(s, fmt)
            else:
                target_dt = datetime.fromisoformat(s)
            break
        except Exception:
            target_dt = None
    if target_dt is None:
        return None
    now = datetime.utcnow()
    years = (target_dt - now).total_seconds() / (365.25 * 86400.0)
    if years <= 0:
        return None
    return float(years)

def build_grid(sample_file, conn, scenario_3y="announced_2028.yaml", scenario_5y=None):
    print(f"📦 Processing {os.path.basename(sample_file)}...")
    
    # Load data
    data = np.load(sample_file)
    # shapes: (N_sats, N_times)
    alts_2d = data['altitude'].astype(np.float32)
    incs_2d = data['inclination'].astype(np.float32)
    raans_2d = data['raan'].astype(np.float32)

    if alts_2d.ndim != 2:
        raise ValueError("Expected altitude to be a 2D array (N_sats, N_times)")
    n_sats, n_steps = alts_2d.shape

    print(f"   Building per-cell features from {n_sats} satellites x {n_steps} timesteps...")

    # Define bins (shared with API scoring)
    alt_edges = og.ALT_EDGES_KM
    inc_edges = og.INC_EDGES_DEG
    raan_edges = og.RAAN_EDGES_DEG

    # Accumulators per (i,j,k): sum counts across timesteps, and sum of squares (for sigma).
    sum_counts = {}
    sumsq_counts = {}

    for t in range(n_steps):
        alt_t = alts_2d[:, t]
        valid = ~np.isnan(alt_t)
        if not np.any(valid):
            continue
        alt_v = alt_t[valid]
        inc_v = incs_2d[:, t][valid]
        raan_v = raans_2d[:, t][valid]

        H = np.histogramdd(
            (alt_v, inc_v, raan_v),
            bins=(alt_edges, inc_edges, raan_edges)
        )[0]
        nz = np.nonzero(H)
        counts = H[nz]
        for i, j, k, c in zip(nz[0], nz[1], nz[2], counts):
            key = (int(i), int(j), int(k))
            v = float(c)
            sum_counts[key] = sum_counts.get(key, 0.0) + v
            sumsq_counts[key] = sumsq_counts.get(key, 0.0) + v * v

        if (t + 1) % max(1, n_steps // 6) == 0:
            print(f"   ... timestep {t+1}/{n_steps}")

    print(f"   Populated cells: {len(sum_counts)}")

    # Base N_eff and occupancy sigma
    n_eff = {}
    occ_sigma = {}
    for key, s in sum_counts.items():
        mean = s / float(n_steps)
        n_eff[key] = mean
        var = (sumsq_counts.get(key, 0.0) / float(n_steps)) - (mean * mean)
        occ_sigma[key] = math.sqrt(max(0.0, var))

    # Build inclination-mixture per altitude (aggregated across RAAN)
    alt_inc_w = {}
    for (ai, ii, _ki), mean in n_eff.items():
        alt_start = float(alt_edges[ai])
        inc_start = float(inc_edges[ii])
        alt_inc_w.setdefault(alt_start, {})
        alt_inc_w[alt_start][inc_start] = alt_inc_w[alt_start].get(inc_start, 0.0) + float(mean)

    inc_factor_map = {}
    for alt_start, inc_map in alt_inc_w.items():
        inc_starts = sorted(inc_map.keys())
        inc_centers = [x + INC_BIN_SIZE / 2.0 for x in inc_starts]
        weights = [inc_map[x] for x in inc_starts]
        for inc_start, inc_center in zip(inc_starts, inc_centers):
            inc_factor_map[(alt_start, inc_start)] = _inc_factor(inc_centers, weights, inc_center)

    # vrel proxy per cell (km/s)
    vrel_proxy = {}
    for (ai, ii, ki), mean in n_eff.items():
        alt_start = float(alt_edges[ai])
        inc_start = float(inc_edges[ii])
        alt_center = float(alt_edges[ai] + alt_edges[ai + 1]) / 2.0
        v = _v_circ_km_s(alt_center)
        g = inc_factor_map.get((alt_start, inc_start), 0.0)
        vrel_proxy[(ai, ii, ki)] = 2.0 * v * g

    # Risk sigma per cell
    risk_sigma = {k: occ_sigma.get(k, 0.0) * vrel_proxy.get(k, 0.0) for k in n_eff.keys()}

    # Trend via declarative scenarios (base vs scenario file)
    scenario_3y_path = SCENARIO_DIR / scenario_3y if scenario_3y else None
    scenario_5y_path = SCENARIO_DIR / scenario_5y if scenario_5y else None
    scenario3 = _load_scenario(scenario_3y_path)
    scenario5 = _load_scenario(scenario_5y_path) if scenario_5y else {}

    delta3 = _scenario_to_delta_cells(scenario3, alt_edges, inc_edges, raan_edges)
    delta5 = _scenario_to_delta_cells(scenario5, alt_edges, inc_edges, raan_edges) if scenario5 else None
    if delta5 is None:
        delta5 = delta3

    def scenario_vrel(delta_cells):
        # Recompute vrel proxy under scenario-adjusted inclination mixture (scenario changes weights).
        # Only inclinations at a given altitude matter.
        alt_inc_w_s = {a: dict(m) for a, m in alt_inc_w.items()}
        for (ai, ii, _ki), d in delta_cells.items():
            alt_start = float(alt_edges[ai])
            inc_start = float(inc_edges[ii])
            alt_inc_w_s.setdefault(alt_start, {})
            alt_inc_w_s[alt_start][inc_start] = alt_inc_w_s[alt_start].get(inc_start, 0.0) + float(d)
        inc_factor_s = {}
        for alt_start, inc_map in alt_inc_w_s.items():
            inc_starts = sorted(inc_map.keys())
            inc_centers = [x + INC_BIN_SIZE / 2.0 for x in inc_starts]
            weights = [inc_map[x] for x in inc_starts]
            for inc_start, inc_center in zip(inc_starts, inc_centers):
                inc_factor_s[(alt_start, inc_start)] = _inc_factor(inc_centers, weights, inc_center)
        vrel_s = {}
        keys = set(n_eff.keys()) | set(delta_cells.keys())
        for (ai, ii, ki) in keys:
            alt_start = float(alt_edges[ai])
            inc_start = float(inc_edges[ii])
            alt_center = float(alt_edges[ai] + alt_edges[ai + 1]) / 2.0
            v = _v_circ_km_s(alt_center)
            g = inc_factor_s.get((alt_start, inc_start), 0.0)
            vrel_s[(ai, ii, ki)] = 2.0 * v * g
        return vrel_s

    vrel3 = scenario_vrel(delta3)
    vrel5 = scenario_vrel(delta5)

    trend_3y = {}
    trend_5y = {}
    all_keys = set(n_eff.keys()) | set(delta3.keys()) | set(delta5.keys())
    for key in all_keys:
        base_n = n_eff.get(key, 0.0)
        base_v = vrel_proxy.get(key, 0.0)
        base_risk = base_n * base_v

        n3 = base_n + float(delta3.get(key, 0.0))
        r3 = n3 * float(vrel3.get(key, base_v))
        trend_3y[key] = r3 - base_risk

        n5 = base_n + float(delta5.get(key, 0.0))
        r5 = n5 * float(vrel5.get(key, base_v))
        trend_5y[key] = r5 - base_risk

    # Scenario meta + annualization (per-year delta) if a target date is provided.
    scenario_name = (scenario3.get("scenario_name") or (Path(scenario_3y).stem if scenario_3y else "")).strip()
    scenario_desc = (scenario3.get("description") or "").strip()
    scenario_target_date = (scenario3.get("target_date") or "").strip()
    scenario_profile = (scenario3.get("deployment_profile") or "linear").strip()
    years_to_target = _years_to_target(scenario_target_date)

    trend_total = dict(trend_3y)
    if years_to_target and scenario_profile.lower() == "linear":
        trend_annual = {k: float(v) / float(years_to_target) for k, v in trend_total.items()}
    else:
        # If no target date is provided, keep the same units (delta-at-target).
        trend_annual = dict(trend_total)

    # Summary for quick sanity checks
    n_eff_vals = np.array(list(n_eff.values()), dtype=np.float32)
    summary = {
        "batch_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "n_sats": int(n_sats),
        "n_steps": int(n_steps),
        "n_cells": int(len(sum_counts)),
        "n_eff": {
            "min": float(np.min(n_eff_vals)) if n_eff_vals.size else 0.0,
            "p50": float(np.percentile(n_eff_vals, 50)) if n_eff_vals.size else 0.0,
            "p95": float(np.percentile(n_eff_vals, 95)) if n_eff_vals.size else 0.0,
            "max": float(np.max(n_eff_vals)) if n_eff_vals.size else 0.0,
        },
        "scenario": {
            "file": str(scenario_3y or ""),
            "name": scenario_name,
            "target_date": scenario_target_date,
            "years_to_target": float(years_to_target) if years_to_target else None,
            "deployment_profile": scenario_profile,
            "description": scenario_desc,
        },
    }
    
    # Prepare batch insert
    # edges[0][i] is the start of bin i
    batch_id = summary["batch_id"]
    
    rows = []
    for (ai, ii, ki) in all_keys:
        alt_start = float(alt_edges[ai])
        inc_start = float(inc_edges[ii])
        raan_start = float(raan_edges[ki])
        s_count = int(round(sum_counts.get((ai, ii, ki), 0.0)))
        rows.append((
            batch_id,
            alt_start,
            inc_start,
            raan_start,
            s_count,
            None,
            float(n_eff.get((ai, ii, ki), 0.0)),
            float(vrel_proxy.get((ai, ii, ki), 0.0)),
            float(occ_sigma.get((ai, ii, ki), 0.0)),
            float(risk_sigma.get((ai, ii, ki), 0.0)),
            float(trend_3y.get((ai, ii, ki), 0.0)),
            float(trend_5y.get((ai, ii, ki), 0.0)),
            float(trend_total.get((ai, ii, ki), 0.0)),
            float(trend_annual.get((ai, ii, ki), 0.0)),
        ))
    
    print("   Inserting into DB...")
    c = conn.cursor()
    c.executemany('''
        INSERT INTO orbital_cells (
            batch_id, alt_bin_start, inc_bin_start, raan_bin_start,
            sample_count, unique_objects,
            n_eff, vrel_proxy, occ_sigma, risk_sigma, trend_3y, trend_5y, trend_total, trend_annual
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', rows)

    # Store batch-level scenario meta for the API/UI.
    c.execute(
        """
        INSERT OR REPLACE INTO orpi_batches (
            batch_id, created_at,
            scenario_file, scenario_name, scenario_target_date, scenario_years_to_target,
            scenario_description, scenario_deployment_profile
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            batch_id,
            datetime.utcnow().isoformat(timespec="seconds"),
            str(scenario_3y or ""),
            scenario_name,
            scenario_target_date,
            float(years_to_target) if years_to_target else None,
            scenario_desc,
            scenario_profile,
        ),
    )
    conn.commit()
    
    out_summary = FEATURES_DIR / f"features_{batch_id}.json"
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Batch {batch_id} complete. {len(rows)} rows inserted.")
    print(f"🧾 Summary: {out_summary}")

def main():
    conn = init_db()
    latest_file = get_latest_samples()
    
    if latest_file:
        scenario_3y = os.getenv("ORPI_SCENARIO_3Y", "announced_2028.yaml")
        scenario_5y = os.getenv("ORPI_SCENARIO_5Y", "") or None
        build_grid(latest_file, conn, scenario_3y=scenario_3y, scenario_5y=scenario_5y)
    else:
        print("❌ No sample files found in data/samples/")
    
    conn.close()

if __name__ == "__main__":
    main()
