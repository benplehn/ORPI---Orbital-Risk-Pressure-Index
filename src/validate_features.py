import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "orpi.db"


def latest_batch(conn):
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT batch_id FROM orbital_cells ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    return row["batch_id"] if row else None


def agg_alt_inc(conn, batch_id, alt_bin_start, inc_bin_start):
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("PRAGMA table_info(orbital_cells)")
    cols = {r[1] for r in c.fetchall()}
    trend_total_col = "trend_total" if "trend_total" in cols else "trend_3y"
    trend_annual_col = "trend_annual" if "trend_annual" in cols else trend_total_col
    c.execute(
        f"""
        SELECT
          SUM(n_eff) AS n_eff_sum,
          SUM(n_eff * vrel_proxy) / NULLIF(SUM(n_eff), 0) AS vrel_w,
          SUM(n_eff * vrel_proxy) AS pressure_mean,
          SUM(risk_sigma) AS risk_sigma,
          SUM({trend_total_col}) AS trend_total,
          SUM({trend_annual_col}) AS trend_annual
        FROM orbital_cells
        WHERE batch_id=? AND alt_bin_start=? AND inc_bin_start=?
        """,
        (batch_id, alt_bin_start, inc_bin_start),
    )
    r = c.fetchone()
    if not r:
        return None
    return {
        "n_eff_sum": float(r["n_eff_sum"] or 0.0),
        "vrel_w": float(r["vrel_w"] or 0.0),
        "pressure_mean": float(r["pressure_mean"] or 0.0),
        "risk_sigma": float(r["risk_sigma"] or 0.0),
        "trend_total": float(r["trend_total"] or 0.0),
        "trend_annual": float(r["trend_annual"] or 0.0),
    }


def main():
    if not DB_PATH.exists():
        raise SystemExit(f"DB missing: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    batch_id = latest_batch(conn)
    if not batch_id:
        raise SystemExit("No batch found.")

    print(f"batch_id={batch_id}")

    # Gate checks requested:
    # - 53 deg vs 97 deg
    # - 550 km vs 800 km
    combos = [
        (550.0, 52.0),
        (550.0, 96.0),
        (800.0, 52.0),
        (800.0, 96.0),
    ]
    for alt, inc in combos:
        m = agg_alt_inc(conn, batch_id, alt, inc)
        print(
            f"alt={alt:6.1f} inc={inc:6.1f} -> "
            f"n_eff={m['n_eff_sum']:.3f} vrel={m['vrel_w']:.3f} "
            f"pressure={m['pressure_mean']:.3f} sigma={m['risk_sigma']:.3f} "
            f"delta={m['trend_total']:.3f} delta/yr={m['trend_annual']:.3f}"
        )

    conn.close()


if __name__ == "__main__":
    main()
