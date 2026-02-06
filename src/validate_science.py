import argparse
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

try:
    from src import orpi_grid as og
except Exception:
    import orpi_grid as og

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "orpi.db"
TLE_FILE = DATA_DIR / "tle" / "latest.txt"


def percentile_from_sorted(sorted_arr: np.ndarray, x: float) -> float:
    if sorted_arr is None or sorted_arr.size == 0:
        return 0.0
    left = int(np.searchsorted(sorted_arr, x, side="left"))
    right = int(np.searchsorted(sorted_arr, x, side="right"))
    rank = (left + right) / 2.0
    return float(100.0 * rank / float(sorted_arr.size))


def build_orpi_v1_model(rows):
    density = np.array([float(r["n_eff_sum"] or 0.0) for r in rows], dtype=np.float64)
    vrel = np.array([float(r["vrel_weighted"] or 0.0) for r in rows], dtype=np.float64)
    pressure = np.array([float(r["pressure_mean"] or 0.0) for r in rows], dtype=np.float64)
    sigma = np.array([float(r["risk_sigma_sum"] or 0.0) for r in rows], dtype=np.float64)
    growth_annual = np.array([max(0.0, float(r["trend_annual_sum"] or 0.0)) for r in rows], dtype=np.float64)

    exposure_raw = np.log1p(np.maximum(pressure, 0.0))
    congestion_raw = np.log1p(np.maximum(density, 0.0))
    geometry_raw = np.maximum(vrel, 0.0)
    volatility_raw = sigma / (np.maximum(pressure, 0.0) + 1e-6)

    model = {
        "weights": {"pressure": 0.62, "volatility": 0.14, "growth": 0.24},
        "pressure_block_weights": {"exposure": 0.50, "congestion": 0.10, "geometry": 0.40},
        "density_sorted": np.sort(density.astype(np.float32)),
        "vrel_sorted": np.sort(vrel.astype(np.float32)),
        "pressure_sorted": np.sort(pressure.astype(np.float32)),
        "sigma_sorted": np.sort(sigma.astype(np.float32)),
        "exposure_sorted": np.sort(exposure_raw.astype(np.float32)),
        "congestion_sorted": np.sort(congestion_raw.astype(np.float32)),
        "geometry_sorted": np.sort(geometry_raw.astype(np.float32)),
        "volatility_sorted": np.sort(volatility_raw.astype(np.float32)),
        "growth_sorted": np.sort(growth_annual.astype(np.float32)),
    }

    p_scores = []
    for r in rows:
        n_eff = float(r["n_eff_sum"] or 0.0)
        vrel_w = float(r["vrel_weighted"] or 0.0)
        pressure_mean = float(r["pressure_mean"] or 0.0)
        sigma_sum = float(r["risk_sigma_sum"] or 0.0)
        growth = max(0.0, float(r["trend_annual_sum"] or 0.0))

        exposure_pct = percentile_from_sorted(model["exposure_sorted"], math.log1p(max(0.0, pressure_mean)))
        congestion_pct = percentile_from_sorted(model["congestion_sorted"], math.log1p(max(0.0, n_eff)))
        geometry_pct = percentile_from_sorted(model["geometry_sorted"], max(0.0, vrel_w))
        p_block = (
            0.50 * exposure_pct
            + 0.10 * congestion_pct
            + 0.40 * geometry_pct
        )
        vol_pct = percentile_from_sorted(model["volatility_sorted"], sigma_sum / (max(0.0, pressure_mean) + 1e-6))
        growth_pct = percentile_from_sorted(model["growth_sorted"], growth)

        score = (
            model["weights"]["pressure"] * p_block
            + model["weights"]["volatility"] * vol_pct
            + model["weights"]["growth"] * growth_pct
        )
        p_scores.append(float(max(0.0, min(100.0, score))))

    model["orpi_sorted"] = np.sort(np.array(p_scores, dtype=np.float32))
    return model


def score_cell(model, row):
    if row is None:
        return None

    n_eff = float(row["n_eff_sum"] or 0.0)
    vrel_w = float(row["vrel_weighted"] or 0.0)
    pressure_mean = float(row["pressure_mean"] or 0.0)
    sigma_sum = float(row["risk_sigma_sum"] or 0.0)
    growth = max(0.0, float(row["trend_annual_sum"] or 0.0))

    exposure_pct = percentile_from_sorted(model["exposure_sorted"], math.log1p(max(0.0, pressure_mean)))
    congestion_pct = percentile_from_sorted(model["congestion_sorted"], math.log1p(max(0.0, n_eff)))
    geometry_pct = percentile_from_sorted(model["geometry_sorted"], max(0.0, vrel_w))
    pressure_pct = 0.50 * exposure_pct + 0.10 * congestion_pct + 0.40 * geometry_pct

    volatility_raw = sigma_sum / (max(0.0, pressure_mean) + 1e-6)
    volatility_pct = percentile_from_sorted(model["volatility_sorted"], volatility_raw)
    growth_pct = percentile_from_sorted(model["growth_sorted"], growth)

    score = (
        model["weights"]["pressure"] * pressure_pct
        + model["weights"]["volatility"] * volatility_pct
        + model["weights"]["growth"] * growth_pct
    )
    score = float(max(0.0, min(100.0, score)))
    return {
        "orpi": score,
        "percentile": percentile_from_sorted(model["orpi_sorted"], score),
        "pressure_pct": pressure_pct,
        "volatility_pct": volatility_pct,
        "growth_pct": growth_pct,
        "n_eff": n_eff,
        "vrel": vrel_w,
        "pressure": pressure_mean,
        "sigma": sigma_sum,
    }


def parse_tle_epoch(line1: str):
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


def tle_age_stats(path: Path):
    if not path.exists():
        return None
    now = datetime.now(timezone.utc)
    ages = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line.startswith("1 "):
                continue
            epoch = parse_tle_epoch(line)
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


def fetch_batch_ids(conn, limit=6):
    c = conn.cursor()
    # Prefer metadata order when available.
    try:
        c.execute("SELECT batch_id FROM orpi_batches ORDER BY created_at DESC LIMIT ?", (limit,))
        ids = [r[0] for r in c.fetchall()]
        if ids:
            return ids
    except Exception:
        pass

    c.execute("SELECT batch_id FROM orbital_cells GROUP BY batch_id ORDER BY batch_id DESC LIMIT ?", (limit,))
    return [r[0] for r in c.fetchall()]


def batch_rows(conn, batch_id):
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        """
        SELECT
          alt_bin_start,
          inc_bin_start,
          SUM(n_eff) AS n_eff_sum,
          SUM(n_eff * vrel_proxy) / NULLIF(SUM(n_eff), 0) AS vrel_weighted,
          SUM(n_eff * vrel_proxy) AS pressure_mean,
          SUM(risk_sigma) AS risk_sigma_sum,
          SUM(COALESCE(trend_annual, trend_3y)) AS trend_annual_sum
        FROM orbital_cells
        WHERE batch_id = ?
        GROUP BY alt_bin_start, inc_bin_start
        """,
        (batch_id,),
    )
    return c.fetchall()


def row_for_orbit(rows_index, alt_km, inc_deg):
    alt_bin = og.bin_start_from_edges(float(alt_km), og.ALT_EDGES_KM)
    inc_bin = og.bin_start_from_edges(float(inc_deg), og.INC_EDGES_DEG)
    if alt_bin is None or inc_bin is None:
        return None, alt_bin, inc_bin
    return rows_index.get((float(alt_bin), float(inc_bin))), float(alt_bin), float(inc_bin)


def pass_fail(value, threshold, op):
    if value is None:
        return "NA"
    if op == ">=":
        return "PASS" if value >= threshold else "FAIL"
    if op == "<=":
        return "PASS" if value <= threshold else "FAIL"
    return "NA"


def format_num(v, nd=2):
    if v is None:
        return "NA"
    return f"{float(v):.{nd}f}"


def rolling_windows(values, size):
    n = len(values)
    if size <= 0 or n < size:
        return
    for start in range(0, n - size + 1):
        yield values[start : start + size]


def rolling_backtest(batch_ids_oldest_first, batch_scores, targets):
    """
    Compute mini temporal backtest metrics over rolling windows.
    Evaluates stability and separability using reference orbits.
    """
    results = []
    n_batches = len(batch_ids_oldest_first)
    if n_batches < 3:
        return results

    max_window = min(6, n_batches)
    for window_size in range(3, max_window + 1):
        for window_ids in rolling_windows(batch_ids_oldest_first, window_size):
            cell_ranges = []
            cell_stds = []
            sep1_vals = []
            sep2_vals = []
            for bid in window_ids:
                scored = batch_scores.get(bid, {})
                a = scored.get("A_550_53")
                b = scored.get("B_550_97")
                c = scored.get("C_800_53")
                if a and b:
                    sep1_vals.append(abs(a["percentile"] - b["percentile"]))
                if a and c:
                    sep2_vals.append(abs(a["percentile"] - c["percentile"]))

            for name in targets:
                vals = []
                for bid in window_ids:
                    scored = batch_scores.get(bid, {})
                    s = scored.get(name)
                    if s:
                        vals.append(float(s["orpi"]))
                if len(vals) >= 2:
                    arr = np.array(vals, dtype=np.float64)
                    cell_ranges.append(float(np.max(arr) - np.min(arr)))
                    cell_stds.append(float(np.std(arr)))

            mean_range = float(np.mean(cell_ranges)) if cell_ranges else None
            max_range = float(np.max(cell_ranges)) if cell_ranges else None
            mean_std = float(np.mean(cell_stds)) if cell_stds else None
            min_sep1 = float(np.min(sep1_vals)) if sep1_vals else None
            min_sep2 = float(np.min(sep2_vals)) if sep2_vals else None

            is_pass = (
                max_range is not None
                and min_sep1 is not None
                and min_sep2 is not None
                and max_range <= 20.0
                and min_sep1 >= 10.0
                and min_sep2 >= 10.0
            )
            results.append(
                {
                    "window_size": int(window_size),
                    "batches": list(window_ids),
                    "mean_range": mean_range,
                    "max_range": max_range,
                    "mean_std": mean_std,
                    "min_sep1": min_sep1,
                    "min_sep2": min_sep2,
                    "status": "PASS" if is_pass else "FAIL",
                }
            )
    return results


def summarize_rolling_backtest(rolling_results):
    by_window = {}
    for row in rolling_results:
        w = int(row["window_size"])
        by_window.setdefault(w, []).append(row)

    summary = {}
    for w, rows in sorted(by_window.items()):
        count = len(rows)
        pass_count = sum(1 for r in rows if r["status"] == "PASS")

        mean_ranges = [r["mean_range"] for r in rows if r["mean_range"] is not None]
        max_ranges = [r["max_range"] for r in rows if r["max_range"] is not None]
        mean_stds = [r["mean_std"] for r in rows if r["mean_std"] is not None]
        sep1 = [r["min_sep1"] for r in rows if r["min_sep1"] is not None]
        sep2 = [r["min_sep2"] for r in rows if r["min_sep2"] is not None]

        summary[w] = {
            "windows": count,
            "pass_rate": (float(pass_count) / float(count)) if count > 0 else None,
            "mean_of_mean_range": float(np.mean(mean_ranges)) if mean_ranges else None,
            "worst_max_range": float(np.max(max_ranges)) if max_ranges else None,
            "mean_of_mean_std": float(np.mean(mean_stds)) if mean_stds else None,
            "worst_min_sep1": float(np.min(sep1)) if sep1 else None,
            "worst_min_sep2": float(np.min(sep2)) if sep2 else None,
            "status": "PASS" if count > 0 and pass_count == count else ("NA" if count == 0 else "FAIL"),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate ORPI scientific validation report")
    parser.add_argument("--output", default="docs/validation_reports/latest.md", help="Output markdown file")
    parser.add_argument("--batch-limit", type=int, default=6, help="How many latest batches to analyze")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any gate fails")
    parser.add_argument("--allow-missing-db", action="store_true", help="Exit zero if DB is missing")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not DB_PATH.exists():
        msg = "# ORPI Validation Report\n\nDatabase missing. Run grid builder first.\n"
        out_path.write_text(msg, encoding="utf-8")
        if args.allow_missing_db:
            print(f"Wrote report: {out_path} (DB missing, allowed)")
            return
        raise SystemExit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    batch_ids = fetch_batch_ids(conn, limit=max(1, args.batch_limit))
    if not batch_ids:
        conn.close()
        raise SystemExit("No batch found in orbital_cells")

    targets = {
        "A_550_53": (550.0, 53.0),
        "B_550_97": (550.0, 97.0),
        "C_800_53": (800.0, 53.0),
        "D_800_97": (800.0, 97.0),
    }

    # Precompute models/scores for each analyzed batch.
    batch_scores = {}
    latest = None
    scored = {}
    for bid in batch_ids:
        rows = batch_rows(conn, bid)
        if not rows:
            continue
        model = build_orpi_v1_model(rows)
        rows_index = {(float(r["alt_bin_start"]), float(r["inc_bin_start"])): r for r in rows}

        current = {}
        for key, (alt, inc) in targets.items():
            row, alt_bin, inc_bin = row_for_orbit(rows_index, alt, inc)
            out = score_cell(model, row)
            if out is not None:
                out["alt_bin"] = alt_bin
                out["inc_bin"] = inc_bin
            current[key] = out
        batch_scores[bid] = current

        if latest is None:
            latest = bid
            scored = current

    if latest is None:
        conn.close()
        raise SystemExit("No usable batch rows found in orbital_cells")

    # Keep order from oldest to latest for time-series windows.
    batch_ids_ordered = list(reversed([b for b in batch_ids if b in batch_scores]))

    sep1 = None
    sep2 = None
    if scored["A_550_53"] and scored["B_550_97"]:
        sep1 = abs(scored["A_550_53"]["percentile"] - scored["B_550_97"]["percentile"])
    if scored["A_550_53"] and scored["C_800_53"]:
        sep2 = abs(scored["A_550_53"]["percentile"] - scored["C_800_53"]["percentile"])

    # Temporal stability: evaluate the same reference cells over last batches.
    stability_cells = ["A_550_53", "B_550_97", "C_800_53"]
    stability = {}
    for name in stability_cells:
        series = []
        for bid in batch_ids_ordered:
            s = batch_scores[bid].get(name)
            if s:
                series.append((bid, s["orpi"], s["percentile"]))
        if series:
            vals = np.array([x[1] for x in series], dtype=np.float64)
            stability[name] = {
                "n": int(vals.size),
                "std": float(np.std(vals)),
                "range": float(np.max(vals) - np.min(vals)),
                "last": float(vals[-1]),
                "series": series,
            }

    rolling_results = rolling_backtest(
        batch_ids_oldest_first=batch_ids_ordered,
        batch_scores=batch_scores,
        targets=targets.keys(),
    )
    rolling_summary = summarize_rolling_backtest(rolling_results)

    tle_stats = tle_age_stats(TLE_FILE)

    gates = []
    gates.append(("Separability 550km/53deg vs 550km/97deg (delta percentile >= 10)", sep1, 10.0, ">="))
    gates.append(("Separability 550km/53deg vs 800km/53deg (delta percentile >= 10)", sep2, 10.0, ">="))
    if tle_stats:
        gates.append(("TLE freshness p90 <= 30 days", tle_stats["p90_days"], 30.0, "<="))

    # Stability gates for latest baseline cells.
    for name, s in stability.items():
        gates.append((f"Temporal stability {name} ORPI range <= 20", s["range"], 20.0, "<="))
    for window_size, s in rolling_summary.items():
        if s["pass_rate"] is not None:
            gates.append(
                (
                    f"Rolling backtest W{window_size} pass rate >= 80%",
                    100.0 * float(s["pass_rate"]),
                    80.0,
                    ">=",
                )
            )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append("# ORPI Scientific Validation Report")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append(f"Latest batch: `{latest}`")
    lines.append(f"Batches analyzed: {len(batch_ids_ordered)}")
    lines.append("")

    lines.append("## 1. Distinguishability Checks")
    lines.append("")
    lines.append("Target requirement: model should separate 53deg vs 97deg and 550km vs 800km.")
    lines.append("")
    lines.append("| Pair | Delta percentile | Status |")
    lines.append("|---|---:|---|")
    lines.append(f"| 550/53 vs 550/97 | {format_num(sep1)} | {pass_fail(sep1, 10.0, '>=')} |")
    lines.append(f"| 550/53 vs 800/53 | {format_num(sep2)} | {pass_fail(sep2, 10.0, '>=')} |")
    lines.append("")

    lines.append("### Reference Orbit Scores (latest batch)")
    lines.append("")
    lines.append("| Orbit | ORPI | Percentile | Pressure Pctl | Volatility Pctl | Growth Pctl |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, v in scored.items():
        if not v:
            lines.append(f"| {name} | NA | NA | NA | NA | NA |")
            continue
        lines.append(
            f"| {name} | {format_num(v['orpi'])} | {format_num(v['percentile'])} | {format_num(v['pressure_pct'])} | {format_num(v['volatility_pct'])} | {format_num(v['growth_pct'])} |"
        )
    lines.append("")

    lines.append("## 2. Temporal Stability")
    lines.append("")
    if not stability:
        lines.append("Not enough history to compute temporal stability.")
    else:
        lines.append("| Orbit | Samples | ORPI std dev | ORPI range | Last ORPI | Status |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for name, s in stability.items():
            status = pass_fail(s["range"], 20.0, "<=")
            lines.append(
                f"| {name} | {s['n']} | {format_num(s['std'])} | {format_num(s['range'])} | {format_num(s['last'])} | {status} |"
            )
    lines.append("")

    lines.append("## 3. Rolling Backtest (3-6 snapshots)")
    lines.append("")
    if not rolling_results:
        lines.append("Not enough history to run rolling windows (need at least 3 snapshots).")
    else:
        lines.append("| Window size | Windows | Pass rate | Mean ORPI std dev | Mean ORPI range | Worst ORPI range | Worst sep 550/53 vs 550/97 | Worst sep 550/53 vs 800/53 | Status |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for window_size, s in rolling_summary.items():
            pass_rate = (100.0 * s["pass_rate"]) if s["pass_rate"] is not None else None
            lines.append(
                f"| {window_size} | {s['windows']} | {format_num(pass_rate)}% | {format_num(s['mean_of_mean_std'])} | {format_num(s['mean_of_mean_range'])} | {format_num(s['worst_max_range'])} | {format_num(s['worst_min_sep1'])} | {format_num(s['worst_min_sep2'])} | {s['status']} |"
            )
        lines.append("")
        lines.append("### Rolling windows detail")
        lines.append("")
        lines.append("| Window | Batches | Max ORPI range | Min sep 550/53 vs 550/97 | Min sep 550/53 vs 800/53 | Status |")
        lines.append("|---|---|---:|---:|---:|---|")
        for row in rolling_results:
            window_label = f"W{row['window_size']}"
            batches_label = ", ".join(str(b) for b in row["batches"])
            lines.append(
                f"| {window_label} | `{batches_label}` | {format_num(row['max_range'])} | {format_num(row['min_sep1'])} | {format_num(row['min_sep2'])} | {row['status']} |"
            )
    lines.append("")

    lines.append("## 4. Catalog Freshness (TLE)")
    lines.append("")
    if not tle_stats:
        lines.append("No TLE freshness stats available (missing or unparsable TLE file).")
    else:
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        lines.append(f"| TLE count | {tle_stats['count']} |")
        lines.append(f"| p50 age (days) | {format_num(tle_stats['p50_days'])} |")
        lines.append(f"| p90 age (days) | {format_num(tle_stats['p90_days'])} |")
        lines.append(f"| max age (days) | {format_num(tle_stats['max_days'])} |")
        lines.append(f"| Freshness gate p90 <= 30d | {pass_fail(tle_stats['p90_days'], 30.0, '<=')} |")
    lines.append("")

    lines.append("## 5. Gate Summary")
    lines.append("")
    lines.append("| Gate | Value | Threshold | Status |")
    lines.append("|---|---:|---:|---|")
    failures = 0
    for label, value, threshold, op in gates:
        st = pass_fail(value, threshold, op)
        if st == "FAIL":
            failures += 1
        lines.append(f"| {label} | {format_num(value)} | {op} {threshold:.2f} | {st} |")

    lines.append("")
    lines.append("## 6. Notes")
    lines.append("")
    lines.append("- ORPI v1 remains a comparative index, not a conjunction event predictor.")
    lines.append("- Growth block depends on declarative scenarios and should be tracked with explicit scenario versioning.")
    lines.append("- Future model upgrades should add covariance and maneuver uncertainty models.")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    conn.close()

    print(f"Wrote report: {out_path}")
    if args.strict and failures > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
