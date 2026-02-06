from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import os
import sqlite3
import numpy as np
from pathlib import Path
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta, timezone
import re
from collections import Counter, defaultdict
import math
import io

# Import shared grid definition (supports both `python3 src/api.py` and uvicorn `src.api:app`).
try:
    from src import orpi_grid as og
except Exception:
    import orpi_grid as og

# PDF brief renderer (pure python via reportlab).
try:
    from src import brief_pdf as bp
except Exception:
    import brief_pdf as bp

app = FastAPI(title="ORPI API", description="Orbital Risk Pressure Index Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "orpi.db"
TLE_FILE = DATA_DIR / "tle" / "latest.txt"

# --- In-Memory TLE Cache for Search ---
TLE_CACHE = []
SAT_ID_MAP = {}
FAMILY_COUNTS = Counter()
FAMILY_INDEX = defaultdict(list)  # family -> [{id,name}, ...]
TLE_FRESHNESS_STATS = None

# --- ORPI comparative model ---
ORPI_MODEL = None  # lazy-loaded at import/startup

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")

def infer_family(name: str) -> str:
    if not name:
        return ""
    n = name.upper().strip()
    # Strip any leading 0 from 3LE already handled, but keep this defensive.
    if n.startswith("0 "):
        n = n[2:].strip()
    # Split on whitespace first, then split token on '-' to catch STARLINK-XXXX.
    parts = n.split()
    first = parts[0] if parts else n
    first = first.split("-")[0]
    first = _NON_ALNUM.sub("", first)
    # Keep it short and meaningful.
    if len(first) < 2:
        return ""
    return first

def _parse_tle_epoch_utc_from_line1(line1: str):
    """
    Parse TLE line 1 epoch fields YYDDD.DDDDDDDD into UTC datetime.
    Returns None when parsing fails.
    """
    try:
        yy = int(line1[18:20])
        day = float(line1[20:32])
    except Exception:
        return None

    year = 2000 + yy if yy < 57 else 1900 + yy
    try:
        dt0 = datetime(year, 1, 1, tzinfo=timezone.utc)
        return dt0 + timedelta(days=day - 1.0)
    except Exception:
        return None

def _tle_freshness_stats_from_cache(cache):
    if not cache:
        return None
    now = datetime.now(timezone.utc)
    ages = []
    for sat in cache:
        l1 = sat.get("l1")
        if not l1:
            continue
        epoch = _parse_tle_epoch_utc_from_line1(l1)
        if epoch is None:
            continue
        age_days = (now - epoch).total_seconds() / 86400.0
        if math.isfinite(age_days):
            ages.append(age_days)
    if not ages:
        return None
    arr = np.array(ages, dtype=np.float64)
    return {
        "count": int(arr.size),
        "p50_days": float(np.percentile(arr, 50)),
        "p90_days": float(np.percentile(arr, 90)),
        "max_days": float(np.max(arr)),
    }

def load_tle_cache():
    """Lengths TLEs into memory for fast search."""
    global TLE_CACHE, SAT_ID_MAP, FAMILY_COUNTS, FAMILY_INDEX, TLE_FRESHNESS_STATS
    print("⏳ Loading TLE cache...")
    if not TLE_FILE.exists():
        print("❌ No TLE file found.")
        return

    with open(TLE_FILE, 'r') as f:
        lines = f.readlines()
    
    SAT_ID_MAP = {}
    FAMILY_COUNTS = Counter()
    FAMILY_INDEX = defaultdict(list)
    i = 0
    cache = []
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
            
        # Try to parse 3-line format (Name + Line1 + Line2)
        # Or 2-line format (Line1 + Line2) - standard CelesTrak/SpaceTrack GP raw often has 0/1/2
        
        # Heuristic: Check if line starts with '1 ' -> it's line 1
        name = "UNKNOWN"
        l1 = ""
        l2 = ""
        
        if line.startswith('1 ') and len(line) > 60:
             # 2-line format
             l1 = line
             if i+1 < len(lines): l2 = lines[i+1].strip()
             # Name is unknown or previous line? 
             # In raw GP files from ingest.py (SpaceTrack), it usually sends 2 lines ONLY if format=tle.
             # But usually Name is available in metadata or if we requested it. 
             # Our ingest.py `st.gp(..., format='tle')` returns raw TLEs.
             # Space-Track raw TLEs are just Line 1 and Line 2. The property name is missing unless '3le' format.
             # However, Line 1 contains the ID.
             # We will use ID as name if name is missing.
             try:
                 sat_id_guess = int(l1[2:7])
                 name = f"SAT-{sat_id_guess:05d}"
             except Exception:
                 name = "SAT-UNKNOWN"
             i += 2
        else:
             # 3-line format
             # Space-Track "3le" includes a leading "0 " on the name line.
             name = line[2:].strip() if line.startswith("0 ") else line
             if i+2 < len(lines):
                 l1 = lines[i+1].strip()
                 l2 = lines[i+2].strip()
             i += 3
        
        if l1 and l2:
            try:
                # Parse ID from Line 1
                sat_id = int(l1[2:7])
                entry = {"id": sat_id, "name": name, "l1": l1, "l2": l2}
                cache.append(entry)
                SAT_ID_MAP[sat_id] = entry
            except:
                pass
    
    TLE_CACHE = cache

    # Build a lightweight "family/operator" index for dropdown browsing.
    # Heuristic: first token split by whitespace or '-' (e.g. STARLINK-1234 -> STARLINK).
    for sat in TLE_CACHE:
        fam = infer_family(sat["name"])
        if fam:
            FAMILY_COUNTS[fam] += 1
            FAMILY_INDEX[fam].append({"id": sat["id"], "name": sat["name"]})

    TLE_FRESHNESS_STATS = _tle_freshness_stats_from_cache(TLE_CACHE)
    if TLE_FRESHNESS_STATS:
        print(
            "✅ Loaded "
            f"{len(TLE_CACHE)} satellites into cache. "
            f"TLE freshness p90={TLE_FRESHNESS_STATS['p90_days']:.2f} days."
        )
    else:
        print(f"✅ Loaded {len(TLE_CACHE)} satellites into cache.")

# Load on startup
load_tle_cache()

# --- Helpers ---

def percentile_from_sorted(sorted_arr: np.ndarray, x: float) -> float:
    if sorted_arr is None or sorted_arr.size == 0:
        return 0.0
    left = int(np.searchsorted(sorted_arr, x, side="left"))
    right = int(np.searchsorted(sorted_arr, x, side="right"))
    rank = (left + right) / 2.0
    return float(100.0 * rank / float(sorted_arr.size))

def _percentiles_for_values(sorted_arr: np.ndarray, values: np.ndarray) -> np.ndarray:
    if sorted_arr is None or sorted_arr.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    left = np.searchsorted(sorted_arr, values, side="left")
    right = np.searchsorted(sorted_arr, values, side="right")
    rank = (left + right) / 2.0
    return (100.0 * rank / float(sorted_arr.size)).astype(np.float32)

def load_orpi_model():
    """
    Build percentile-based ORPI v0 normalization tables from the latest batch.
    Uses aggregated (alt_bin_start, inc_bin_start) zones (summing across RAAN bins).

    ORPI v1 is a comparative index (0-100) based on percentiles.
    It combines:
      - pressure block: exposure + congestion + geometry
      - volatility block: normalized instability (sigma / pressure)
      - growth block: annualized declarative scenario delta
    """
    global ORPI_MODEL
    if not DB_PATH.exists():
        ORPI_MODEL = None
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT batch_id FROM orbital_cells ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        if not row:
            ORPI_MODEL = None
            return
        batch_id = row["batch_id"]

        c.execute("PRAGMA table_info(orbital_cells)")
        cols = {r[1] for r in c.fetchall()}
        trend_total_col = "trend_total" if "trend_total" in cols else "trend_3y"
        trend_annual_col = "trend_annual" if "trend_annual" in cols else trend_total_col

        c.execute(
            f"""
            SELECT
              alt_bin_start,
              inc_bin_start,
              SUM(sample_count) AS sample_count_sum,
              SUM(n_eff) AS n_eff_sum,
              SUM(n_eff * vrel_proxy) / NULLIF(SUM(n_eff), 0) AS vrel_weighted,
              SUM(n_eff * vrel_proxy) AS pressure_mean,
              SUM(risk_sigma) AS risk_sigma_sum,
              SUM({trend_total_col}) AS trend_total_sum,
              SUM({trend_annual_col}) AS trend_annual_sum
            FROM orbital_cells
            WHERE batch_id = ?
            GROUP BY alt_bin_start, inc_bin_start
            """,
            (batch_id,),
        )
        rows = c.fetchall()

        # Optional: batch-level scenario meta (nice for UI explainability).
        scenario_meta = None
        try:
            c.execute(
                """
                SELECT
                  scenario_file,
                  scenario_name,
                  scenario_target_date,
                  scenario_years_to_target,
                  scenario_description,
                  scenario_deployment_profile
                FROM orpi_batches
                WHERE batch_id = ?
                """,
                (batch_id,),
            )
            m = c.fetchone()
            if m:
                scenario_meta = {
                    "file": m["scenario_file"] or "",
                    "name": m["scenario_name"] or "",
                    "target_date": m["scenario_target_date"] or "",
                    "years_to_target": float(m["scenario_years_to_target"]) if m["scenario_years_to_target"] is not None else None,
                    "description": m["scenario_description"] or "",
                    "deployment_profile": m["scenario_deployment_profile"] or "",
                }
        except sqlite3.OperationalError:
            scenario_meta = None
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not rows:
        ORPI_MODEL = None
        return

    density = np.array([float(r["n_eff_sum"] or 0.0) for r in rows], dtype=np.float64)
    coverage = np.array([float(r["sample_count_sum"] or 0.0) for r in rows], dtype=np.float64)
    vrel = np.array([float(r["vrel_weighted"] or 0.0) for r in rows], dtype=np.float64)
    pressure = np.array([float(r["pressure_mean"] or 0.0) for r in rows], dtype=np.float64)
    sigma = np.array([float(r["risk_sigma_sum"] or 0.0) for r in rows], dtype=np.float64)
    growth_annual_raw = np.array([float(r["trend_annual_sum"] or 0.0) for r in rows], dtype=np.float64)
    growth = np.maximum(growth_annual_raw, 0.0)

    eps = 1e-6
    exposure_raw = np.log1p(np.maximum(pressure, 0.0))
    congestion_raw = np.log1p(np.maximum(density, 0.0))
    geometry_raw = np.maximum(vrel, 0.0)
    volatility_raw = sigma / (np.maximum(pressure, 0.0) + eps)
    growth_raw = growth

    density_s = np.sort(density.astype(np.float32))
    coverage_s = np.sort(coverage.astype(np.float32))
    vrel_s = np.sort(vrel.astype(np.float32))
    pressure_s = np.sort(pressure.astype(np.float32))
    sigma_s = np.sort(sigma.astype(np.float32))
    exposure_s = np.sort(exposure_raw.astype(np.float32))
    congestion_s = np.sort(congestion_raw.astype(np.float32))
    geometry_s = np.sort(geometry_raw.astype(np.float32))
    volatility_s = np.sort(volatility_raw.astype(np.float32))
    growth_s = np.sort(growth_raw.astype(np.float32))

    # Weights can be overridden via env vars if needed; defaults are conservative.
    # They must sum to 1.0.
    w_pressure = float(os.getenv("ORPI_W_PRESSURE", "0.62"))
    w_volatility = float(os.getenv("ORPI_W_VOLATILITY", "0.14"))
    w_growth = float(os.getenv("ORPI_W_GROWTH", "0.24"))
    w_sum = w_pressure + w_volatility + w_growth
    if w_sum <= 0:
        w_pressure, w_volatility, w_growth = 0.62, 0.14, 0.24
    else:
        w_pressure, w_volatility, w_growth = w_pressure / w_sum, w_volatility / w_sum, w_growth / w_sum

    # Calibrated defaults to improve inclination-family separability at same altitude
    # while preserving robustness of the pressure block.
    w_exp = float(os.getenv("ORPI_W_PRESSURE_EXPOSURE", "0.50"))
    w_cong = float(os.getenv("ORPI_W_PRESSURE_CONGESTION", "0.10"))
    w_geo = float(os.getenv("ORPI_W_PRESSURE_GEOMETRY", "0.40"))
    w_block = w_exp + w_cong + w_geo
    if w_block <= 0:
        w_exp, w_cong, w_geo = 0.50, 0.10, 0.40
    else:
        w_exp, w_cong, w_geo = w_exp / w_block, w_cong / w_block, w_geo / w_block

    exp_pct = _percentiles_for_values(exposure_s, exposure_raw.astype(np.float32))
    cong_pct = _percentiles_for_values(congestion_s, congestion_raw.astype(np.float32))
    geo_pct = _percentiles_for_values(geometry_s, geometry_raw.astype(np.float32))
    pressure_block_pct = (w_exp * exp_pct + w_cong * cong_pct + w_geo * geo_pct).astype(np.float32)
    vol_pct = _percentiles_for_values(volatility_s, volatility_raw.astype(np.float32))
    growth_pct = _percentiles_for_values(growth_s, growth_raw.astype(np.float32))

    orpi_scores = (w_pressure * pressure_block_pct + w_volatility * vol_pct + w_growth * growth_pct).astype(np.float32)
    orpi_s = np.sort(orpi_scores)

    ORPI_MODEL = {
        "version": "ORPI_v1",
        "batch_id": batch_id,
        "n_zones": int(len(rows)),
        "weights": {"pressure": w_pressure, "volatility": w_volatility, "growth": w_growth},
        "pressure_block_weights": {"exposure": w_exp, "congestion": w_cong, "geometry": w_geo},
        "density_sorted": density_s,
        "coverage_sorted": coverage_s,
        "vrel_sorted": vrel_s,
        "pressure_sorted": pressure_s,
        "sigma_sorted": sigma_s,
        "exposure_sorted": exposure_s,
        "congestion_sorted": congestion_s,
        "geometry_sorted": geometry_s,
        "volatility_sorted": volatility_s,
        "growth_sorted": growth_s,
        "trend_total_col": trend_total_col,
        "trend_annual_col": trend_annual_col,
        "scenario": scenario_meta,
        "orpi_sorted": orpi_s,
    }

def _rating_from_percentile(p: float) -> str:
    if p >= 90:
        return "CRITICAL"
    if p >= 70:
        return "HIGH"
    if p >= 40:
        return "MODERATE"
    return "LOW"

def _underwriting_stance(percentile: float) -> str:
    p = float(percentile)
    if p >= 95:
        return "Decline by default or cap capacity to a minimum with strict exclusions."
    if p >= 85:
        return "Write only with tight terms: higher deductible, tighter sub-limits, and collision/debris wording."
    if p >= 65:
        return "Write with moderate constraints and explicit monitoring/mitigation clauses."
    return "Standard terms are generally acceptable, subject to normal underwriting checks."

def _freshness_score_from_p90_days(p90_days: float):
    """
    Map catalog freshness to a 0-100 confidence score.
    100 when p90 <= 3d, 0 when p90 >= 30d, linear in between.
    """
    if p90_days is None or not math.isfinite(p90_days):
        return 50.0
    lo = 3.0
    hi = 30.0
    if p90_days <= lo:
        return 100.0
    if p90_days >= hi:
        return 0.0
    return float(100.0 * (hi - p90_days) / (hi - lo))

def _stability_score_for_cell(conn, model, alt_bin, inc_bin, trend_total_col, trend_annual_col, window=4):
    """
    Compute a simple inter-batch stability confidence for the target cell.
    Uses ORPI score spread over recent batches + cell persistence.
    """
    c = conn.cursor()
    c.execute(
        "SELECT batch_id FROM orbital_cells GROUP BY batch_id ORDER BY batch_id DESC LIMIT ?",
        (int(max(2, window)),),
    )
    batch_ids = [r[0] for r in c.fetchall()]
    if not batch_ids:
        return {"score": 50.0, "n_observed": 0, "n_window": 0, "orpi_range": None, "orpi_std": None}

    vals = []
    for bid in batch_ids:
        c.execute(
            f"""
            SELECT
                SUM(n_eff) as n_eff_sum,
                SUM(n_eff * vrel_proxy) / NULLIF(SUM(n_eff), 0) as vrel_weighted,
                SUM(n_eff * vrel_proxy) as pressure_mean,
                SUM(risk_sigma) as risk_sigma_sum,
                SUM({trend_total_col}) as trend_total_sum,
                SUM({trend_annual_col}) as trend_annual_sum
            FROM orbital_cells
            WHERE batch_id = ? AND alt_bin_start = ? AND inc_bin_start = ?
            """,
            (bid, alt_bin, inc_bin),
        )
        row = c.fetchone()
        if not row:
            continue
        n_eff = float(row[0]) if row[0] is not None else 0.0
        vrel_w = float(row[1]) if row[1] is not None else 0.0
        pressure_mean = float(row[2]) if row[2] is not None else 0.0
        sigma = float(row[3]) if row[3] is not None else 0.0
        growth = max(0.0, float(row[5]) if row[5] is not None else 0.0)

        exposure_raw = math.log1p(max(0.0, pressure_mean))
        congestion_raw = math.log1p(max(0.0, n_eff))
        geometry_raw = max(0.0, vrel_w)
        volatility_raw = sigma / (max(0.0, pressure_mean) + 1e-6)

        exposure_pct = percentile_from_sorted(model["exposure_sorted"], exposure_raw)
        congestion_pct = percentile_from_sorted(model["congestion_sorted"], congestion_raw)
        geometry_pct = percentile_from_sorted(model["geometry_sorted"], geometry_raw)
        volatility_pct = percentile_from_sorted(model["volatility_sorted"], volatility_raw)
        growth_pct = percentile_from_sorted(model["growth_sorted"], growth)

        pw = model.get("pressure_block_weights") or {"exposure": 0.50, "congestion": 0.10, "geometry": 0.40}
        pressure_pct = float(
            pw["exposure"] * exposure_pct
            + pw["congestion"] * congestion_pct
            + pw["geometry"] * geometry_pct
        )
        w = model["weights"]
        score = float(w["pressure"] * pressure_pct + w["volatility"] * volatility_pct + w["growth"] * growth_pct)
        vals.append(max(0.0, min(100.0, score)))

    n_obs = len(vals)
    n_win = len(batch_ids)
    if n_obs <= 1:
        persistence = 100.0 * (float(n_obs) / float(max(1, n_win)))
        return {
            "score": float(50.0 + 0.5 * persistence),
            "n_observed": n_obs,
            "n_window": n_win,
            "orpi_range": None,
            "orpi_std": None,
        }

    arr = np.array(vals, dtype=np.float64)
    score_range = float(np.max(arr) - np.min(arr))
    score_std = float(np.std(arr))
    # 0 spread => 100 confidence; >=20 points spread => 0 on this axis.
    spread_score = float(max(0.0, min(100.0, 100.0 * (1.0 - score_range / 20.0))))
    persistence = float(100.0 * n_obs / float(max(1, n_win)))
    # Blend spread and persistence.
    final = float(0.75 * spread_score + 0.25 * persistence)
    return {
        "score": final,
        "n_observed": n_obs,
        "n_window": n_win,
        "orpi_range": score_range,
        "orpi_std": score_std,
    }

def _orpi_justification(components: dict, cell: dict) -> str:
    # One-sentence explanation for why an orbit is high/low.
    parts = [
        ("pressure", components["pressure"]["percentile"]),
        ("volatility", components["volatility"]["percentile"]),
        ("growth", components["growth"]["percentile"]),
    ]
    parts.sort(key=lambda x: x[1], reverse=True)
    top1, top2 = parts[0], parts[1]
    return (
        f"Highest score drivers are {top1[0]} (P{top1[1]:.0f}) and {top2[0]} (P{top2[1]:.0f}) "
        f"in cell alt={cell['alt_bin_start']:.0f}km / inc={cell['inc_bin_start']:.0f}deg."
    )

load_orpi_model()

def bbox_grid(min_alt, max_alt):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT batch_id FROM orbital_cells ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    if not row: return []
    batch_id = row['batch_id']
    
    query = """
        SELECT alt_bin_start, inc_bin_start, raan_bin_start,
               sample_count, n_eff, vrel_proxy, occ_sigma, risk_sigma, trend_3y, trend_5y
        FROM orbital_cells 
        WHERE batch_id = ? AND alt_bin_start BETWEEN ? AND ?
        AND sample_count > 5
    """
    c.execute(query, (batch_id, min_alt, max_alt))
    rows = c.fetchall()
    conn.close()
    data = []
    for r in rows:
        data.append({
            "alt": r['alt_bin_start'],
            "inc": r['inc_bin_start'],
            "raan": r['raan_bin_start'],
            "count": r['sample_count'],
            "n_eff": r['n_eff'],
            "vrel_proxy": r['vrel_proxy'],
            "occ_sigma": r['occ_sigma'],
            "risk_sigma": r['risk_sigma'],
            "trend_3y": r['trend_3y'],
            "trend_5y": r['trend_5y'],
        })
    return data

def propagate_sat(l1, l2, duration_mins=90, step_mins=1):
    """Propagate a single satellite to get its path (TEME position vectors in km)."""
    sat = Satrec.twoline2rv(l1, l2)

    now = datetime.utcnow()
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)

    duration_mins = float(duration_mins)
    step_mins = float(step_mins)
    if duration_mins <= 0:
        duration_mins = 90.0
    if step_mins <= 0:
        step_mins = 1.0

    minutes = np.arange(0.0, duration_mins + 1e-6, step_mins, dtype=np.float64)
    if minutes.size < 2:
        minutes = np.array([0.0, duration_mins], dtype=np.float64)

    fr_offsets = minutes / 1440.0
    e, r, _v = sat.sgp4_array(np.full(minutes.size, jd, dtype=np.float64), fr + fr_offsets)
    valid = (e == 0)
    rr = r[valid]
    # Convert to a plain list for JSON. (Each element is [x,y,z] km, TEME.)
    return rr.tolist()

def _orbit_period_minutes(satrec: Satrec):
    try:
        n = float(getattr(satrec, "no_kozai", 0.0))
    except Exception:
        n = 0.0
    if not (n > 1e-12):
        return None
    return float(2.0 * math.pi / n)

# --- Endpoints ---

@app.get("/health")
def health():
    model = ORPI_MODEL
    return {
        "status": "ok",
        "cached_sats": len(TLE_CACHE),
        "grid": {"alt_km": [og.ALT_MIN_KM, og.ALT_MAX_KM], "inc_deg": [og.INC_MIN_DEG, og.INC_MAX_DEG]},
        "orpi_model": {
            "version": model.get("version", "ORPI_v0"),
            "batch_id": model["batch_id"],
            "zones": model["n_zones"],
            "weights": model.get("weights"),
            "pressure_block_weights": model.get("pressure_block_weights"),
            "scenario": model.get("scenario"),
            "tle_freshness": TLE_FRESHNESS_STATS,
        }
        if model
        else None,
    }

@app.get("/api/grid/latest")
def get_grid_layer(min_alt: float = 200, max_alt: float = 2000):
    data = bbox_grid(min_alt, max_alt)
    return {"count": len(data), "data": data}

@app.get("/api/satellites/search")
def search_satellites(q: str = Query(..., min_length=1)):
    """Search for satellites by Name or ID."""
    q = q.upper()
    results = []
    # Prefer exact NORAD ID matches when q is numeric, so the dropdown is usable.
    if q.isdigit():
        sat_id = int(q)
        sat = SAT_ID_MAP.get(sat_id)
        if sat:
            results.append({"id": sat["id"], "name": sat["name"]})

    # Simple linear search (for ~30k objects it's fast enough)
    count = len(results)
    for sat in TLE_CACHE:
        if count >= 10:
            break
        if results and sat["id"] == results[0]["id"]:
            continue
        if q in sat['name'] or q in str(sat['id']):
            results.append({"id": sat['id'], "name": sat['name']})
            count += 1
    return results

@app.get("/api/satellites/families")
def list_families(q: str = "", limit: int = 30):
    """
    List common satellite "families/operators" inferred from TLE names.
    Intended for a UI dropdown (e.g. INTELSAT, STARLINK, ONEWEB...).
    """
    qq = (q or "").upper().strip()
    items = []
    for fam, cnt in FAMILY_COUNTS.most_common():
        if qq and qq not in fam:
            continue
        items.append({"family": fam, "count": int(cnt)})
        if len(items) >= max(1, min(limit, 200)):
            break
    return items

@app.get("/api/satellites/by_family")
def satellites_by_family(family: str = Query(..., min_length=2), q: str = "", limit: int = 50):
    """
    Return satellites in a given family (for a second dropdown).
    Optional q filters within that family.
    """
    fam = (family or "").upper().strip()
    qq = (q or "").upper().strip()
    candidates = FAMILY_INDEX.get(fam, [])
    out = []
    for sat in candidates:
        if qq and qq not in sat["name"].upper() and qq not in str(sat["id"]):
            continue
        out.append(sat)
        if len(out) >= max(1, min(limit, 500)):
            break
    return out

@app.get("/api/satellites/{sat_id}/orbit")
def get_orbit_path(sat_id: int):
    """Get the propagated path (XYZ km) for a specific satellite."""
    sat = SAT_ID_MAP.get(sat_id)
    if not sat:
        raise HTTPException(status_code=404, detail="Satellite not found")
    
    satrec = Satrec.twoline2rv(sat['l1'], sat['l2'])
    inc_deg = float(satrec.inclo) * 180.0 / math.pi
    raan_deg = float(satrec.nodeo) * 180.0 / math.pi
    period_mins = _orbit_period_minutes(satrec) or 100.0
    # Keep it interactive: cap at 48h so deep-space objects don't return huge payloads.
    period_mins = float(max(10.0, min(period_mins, 2880.0)))
    # Aim ~360 points, but keep steps reasonable.
    target_points = 360.0
    step_mins = period_mins / target_points
    step_mins = float(max(0.25, min(step_mins, 10.0)))

    # Propagate
    points_km = propagate_sat(sat['l1'], sat['l2'], duration_mins=period_mins, step_mins=step_mins)
    
    # Also calculate current elements for display
    # (Simplified)
    return {
        "id": sat['id'],
        "name": sat['name'],
        "period_minutes": round(period_mins, 3),
        "mean_motion_rev_per_day": round(1440.0 / period_mins, 6) if period_mins > 0 else None,
        "inclination_deg": inc_deg,
        "raan_deg": raan_deg,
        "points": points_km # List of [x, y, z] in km
    }

@app.get("/api/orpi/score")
def calculate_score(altitude: float, inclination: float):
    model = ORPI_MODEL
    if not model:
        raise HTTPException(status_code=503, detail="ORPI model not available. Run grid_builder.py then restart API.")

    alt_bin = og.bin_start_from_edges(altitude, og.ALT_EDGES_KM)
    if alt_bin is None:
        raise HTTPException(
            status_code=400,
            detail=f"Altitude out of ORPI range ({og.ALT_MIN_KM:.0f}-{og.ALT_MAX_KM:.0f} km).",
        )
    inc_bin = og.bin_start_from_edges(inclination, og.INC_EDGES_DEG)
    if inc_bin is None:
        raise HTTPException(
            status_code=400,
            detail=f"Inclination out of ORPI range ({og.INC_MIN_DEG:.0f}-{og.INC_MAX_DEG:.0f} deg).",
        )

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    batch_id = model["batch_id"]
    trend_total_col = model.get("trend_total_col") or "trend_3y"
    trend_annual_col = model.get("trend_annual_col") or trend_total_col

    c.execute(
        f"""
        SELECT
            SUM(sample_count) as sample_count_sum,
            SUM(n_eff) as n_eff_sum,
            SUM(n_eff * vrel_proxy) / NULLIF(SUM(n_eff), 0) as vrel_weighted,
            SUM(n_eff * vrel_proxy) as pressure_mean,
            SUM(risk_sigma) as risk_sigma_sum,
            SUM({trend_total_col}) as trend_total_sum,
            SUM({trend_annual_col}) as trend_annual_sum
        FROM orbital_cells
        WHERE batch_id = ? AND alt_bin_start = ? AND inc_bin_start = ?
        """,
        (batch_id, alt_bin, inc_bin)
    )
    res = c.fetchone()
    sample_count_sum = float(res[0]) if res and res[0] is not None else 0.0
    n_eff_sum = float(res[1]) if res and res[1] is not None else 0.0
    vrel_weighted = float(res[2]) if res and res[2] is not None else 0.0
    pressure_mean = float(res[3]) if res and res[3] is not None else 0.0
    risk_sigma_sum = float(res[4]) if res and res[4] is not None else 0.0
    trend_total_sum = float(res[5]) if res and res[5] is not None else 0.0
    trend_annual_sum = float(res[6]) if res and res[6] is not None else 0.0

    # ORPI v1: pressure block + normalized volatility + scenario growth.
    density_pct = percentile_from_sorted(model["density_sorted"], n_eff_sum)
    vrel_pct = percentile_from_sorted(model["vrel_sorted"], vrel_weighted)
    sigma_pct = percentile_from_sorted(model["sigma_sorted"], risk_sigma_sum)

    exposure_raw = math.log1p(max(0.0, pressure_mean))
    congestion_raw = math.log1p(max(0.0, n_eff_sum))
    geometry_raw = max(0.0, vrel_weighted)
    volatility_raw = risk_sigma_sum / (max(0.0, pressure_mean) + 1e-6)

    exposure_pct = percentile_from_sorted(model["exposure_sorted"], exposure_raw)
    congestion_pct = percentile_from_sorted(model["congestion_sorted"], congestion_raw)
    geometry_pct = percentile_from_sorted(model["geometry_sorted"], geometry_raw)
    volatility_pct = percentile_from_sorted(model["volatility_sorted"], volatility_raw)

    pw = model.get("pressure_block_weights") or {"exposure": 0.50, "congestion": 0.10, "geometry": 0.40}
    pressure_pct = float(
        pw["exposure"] * exposure_pct
        + pw["congestion"] * congestion_pct
        + pw["geometry"] * geometry_pct
    )

    growth_value = max(0.0, trend_annual_sum)
    growth_pct = percentile_from_sorted(model["growth_sorted"], growth_value)

    w = model["weights"]
    orpi_score = float(w["pressure"] * pressure_pct + w["volatility"] * volatility_pct + w["growth"] * growth_pct)
    orpi_score = max(0.0, min(100.0, orpi_score))
    percentile = percentile_from_sorted(model["orpi_sorted"], orpi_score)
    rating = _rating_from_percentile(percentile)

    cell = {"batch_id": batch_id, "alt_bin_start": alt_bin, "inc_bin_start": inc_bin}
    components = {
        "pressure": {"value": pressure_mean, "percentile": round(pressure_pct, 2), "score": round(pressure_pct, 2)},
        "volatility": {"value": risk_sigma_sum, "percentile": round(volatility_pct, 2), "score": round(volatility_pct, 2)},
        "growth": {"value": growth_value, "percentile": round(growth_pct, 2), "score": round(growth_pct, 2)},
    }

    freshness_stats = TLE_FRESHNESS_STATS
    freshness_score = _freshness_score_from_p90_days(
        freshness_stats.get("p90_days") if freshness_stats else None
    )
    coverage_pct = percentile_from_sorted(model.get("coverage_sorted"), sample_count_sum)
    stability_meta = _stability_score_for_cell(
        conn=conn,
        model=model,
        alt_bin=alt_bin,
        inc_bin=inc_bin,
        trend_total_col=trend_total_col,
        trend_annual_col=trend_annual_col,
        window=4,
    )
    stability_score = float(stability_meta.get("score", 50.0))
    # Confidence weights kept simple and explicit.
    confidence_score = float(
        0.40 * freshness_score
        + 0.35 * coverage_pct
        + 0.25 * stability_score
    )
    confidence_score = max(0.0, min(100.0, confidence_score))
    conn.close()

    return {
        "model": {
            "name": "ORPI",
            "version": model.get("version", "ORPI_v0"),
            "architecture": "percentile-ensemble",
            "pressure_block_weights": pw,
        },
        "params": {"alt": altitude, "inc": inclination},
        "cell": cell,
        "features": {
            "n_eff_sum": round(n_eff_sum, 3),
            "vrel_mean_proxy_km_s": round(vrel_weighted, 3),
            "pressure_mean": round(pressure_mean, 3),
            "risk_sigma": round(risk_sigma_sum, 3),
            "volatility_ratio": round(volatility_raw, 6),
            "trend_total": round(trend_total_sum, 3),
            "trend_annual": round(trend_annual_sum, 3),
        },
        "drivers": {
            "density": {"value": round(n_eff_sum, 3), "percentile": round(density_pct, 2)},
            "vrel": {"value": round(vrel_weighted, 3), "percentile": round(vrel_pct, 2)},
            "sigma": {"value": round(risk_sigma_sum, 3), "percentile": round(sigma_pct, 2)},
        },
        "subcomponents": {
            "pressure": {
                "exposure": {"raw": round(exposure_raw, 6), "percentile": round(exposure_pct, 2)},
                "congestion": {"raw": round(congestion_raw, 6), "percentile": round(congestion_pct, 2)},
                "geometry": {"raw": round(geometry_raw, 6), "percentile": round(geometry_pct, 2)},
            },
            "volatility": {"raw": round(volatility_raw, 6), "percentile": round(volatility_pct, 2)},
            "growth": {"raw": round(growth_value, 6), "percentile": round(growth_pct, 2)},
        },
        "components": components,
        "weights": w,
        "scenario": model.get("scenario"),
        "orpi_score": round(orpi_score, 2),
        "percentile": round(percentile, 2),
        "rating": rating,
        "confidence_score": round(confidence_score, 2),
        "confidence": {
            "freshness": {
                "score": round(freshness_score, 2),
                "p90_tle_age_days": round(float(freshness_stats["p90_days"]), 3) if freshness_stats else None,
            },
            "coverage": {
                "score": round(coverage_pct, 2),
                "sample_count_sum": int(round(sample_count_sum)),
            },
            "stability": {
                "score": round(stability_score, 2),
                "n_observed_batches": int(stability_meta.get("n_observed", 0)),
                "n_window_batches": int(stability_meta.get("n_window", 0)),
                "orpi_range": round(float(stability_meta["orpi_range"]), 3) if stability_meta.get("orpi_range") is not None else None,
                "orpi_std": round(float(stability_meta["orpi_std"]), 3) if stability_meta.get("orpi_std") is not None else None,
            },
        },
        "underwriting_stance": _underwriting_stance(percentile),
        "justification": _orpi_justification(components, cell),
    }


@app.get("/api/brief/{sat_id}.pdf")
def generate_brief_pdf(sat_id: int):
    """
    One-page underwriting brief (PDF) for a satellite (NORAD).
    Intended for underwriters/brokers: shareable + auditable snapshot.
    """
    sat = SAT_ID_MAP.get(sat_id)
    if not sat:
        raise HTTPException(status_code=404, detail="Satellite not found")

    # Orbit + metadata (one orbital period, ~360 points).
    satrec = Satrec.twoline2rv(sat["l1"], sat["l2"])
    inc_deg = float(satrec.inclo) * 180.0 / math.pi
    raan_deg = float(satrec.nodeo) * 180.0 / math.pi
    period_mins = _orbit_period_minutes(satrec) or 100.0
    period_mins = float(max(10.0, min(period_mins, 2880.0)))
    step_mins = float(max(0.25, min(period_mins / 360.0, 10.0)))
    points_km = propagate_sat(sat["l1"], sat["l2"], duration_mins=period_mins, step_mins=step_mins)
    alts, alt_min, alt_mean, alt_max = bp.orbit_altitude_series_km(points_km, earth_radius_km=6378.137)

    tle_epoch = bp.satrec_epoch_utc(satrec)
    tle_age_days = None
    if isinstance(tle_epoch, datetime):
        try:
            tle_age_days = (datetime.now(timezone.utc) - tle_epoch).total_seconds() / 86400.0
        except Exception:
            tle_age_days = None

    # ORPI scoring at mean altitude + inclination. If out-of-range, still produce a PDF with N/A.
    orpi = None
    orpi_error = None
    if alt_mean is not None and math.isfinite(float(alt_mean)):
        try:
            orpi = calculate_score(altitude=float(alt_mean), inclination=float(inc_deg))
        except HTTPException as e:
            orpi_error = str(getattr(e, "detail", e))
        except Exception as e:
            orpi_error = str(e)
    else:
        orpi_error = "Altitude unavailable from propagation."

    ctx = {
        "generated_at": datetime.now(timezone.utc),
        "sat": {
            "id": sat_id,
            "name": sat.get("name") or f"SAT-{sat_id:05d}",
            "inc_deg": inc_deg,
            "raan_deg": raan_deg,
            "period_minutes": period_mins,
            "mean_motion_rev_per_day": (1440.0 / period_mins) if period_mins > 0 else None,
            "tle_epoch_utc": tle_epoch,
            "tle_age_days": tle_age_days,
        },
        "orbit": {
            "altitudes_km": alts,
            "alt_min_km": alt_min,
            "alt_mean_km": alt_mean,
            "alt_max_km": alt_max,
        },
        "orpi": orpi,
        "orpi_error": orpi_error,
    }

    try:
        pdf_bytes = bp.render_orpi_brief_pdf(ctx)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"PDF renderer error: {e}")

    headers = {"Content-Disposition": f'attachment; filename="ORPI_Brief_{sat_id}.pdf"'}
    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers=headers)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
