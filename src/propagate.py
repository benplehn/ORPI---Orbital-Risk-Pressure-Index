import os
import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
TLE_FILE = DATA_DIR / "tle" / "latest.txt"
SAMPLES_DIR = DATA_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Constants
EARTH_RADIUS_KM = 6378.135
MIN_ALTITUDE_KM = 200.0   # Filter out decaying objects
MAX_ALTITUDE_KM = 2000.0  # Focus on LEO

def load_tles(filename):
    """Parses a TLE file and returns a list of Satrec objects."""
    satrecs = []
    sat_ids = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Simple TLE parser (assumes 3 lines per satellite or 2 lines)
    # Space-Track GP raw often has 2 lines. 
    # Check if Line 0 exists or if it starts with '1 '
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        if line.startswith('1 '):
            # strict 2-line format
            l1 = line
            l2 = lines[i+1].strip()
            i += 2
        elif line.startswith('2 '):
            # Should not happen if strictly ordered, skip
            i += 1
            continue
        else:
            # 3-line format (Header + 1 + 2)
            # l0 = line
            if i + 2 < len(lines):
                l1 = lines[i+1].strip()
                l2 = lines[i+2].strip()
                i += 3
            else:
                break
        
        try:
            sat = Satrec.twoline2rv(l1, l2)
            satrecs.append(sat)
            sat_ids.append(sat.satnum)
        except Exception:
            continue

    return satrecs, np.array(sat_ids)

def propagate_window(satrecs, sat_ids, start_time, duration_hours=24, step_minutes=30):
    """
    Propagates satellites over a time window.
    Returns arrays of: [Altitudes, Inclinations, RAANs]
    """
    print(f"🚀 Propagating {len(satrecs)} objects over {duration_hours}h (step: {step_minutes}m)...")
    
    steps = int((duration_hours * 60) / step_minutes) + 1
    
    # Pre-compute time inputs for SGP4 (vectorized if possible, but python loop for `sgp4` usually required per sat)
    # Actually Satrec.sgp4_array allows multiple times for ONE satellite.
    # We will loop over satellites and compute arrays of times.
    
    jd_start, fr_start = jday(start_time.year, start_time.month, start_time.day, 
                              start_time.hour, start_time.minute, start_time.second)
    
    # Generate time deltas
    minutes = np.arange(0, duration_hours * 60 + 0.1, step_minutes)
    fr_offsets = minutes / 1440.0
    
    # Storage (N_sats x N_steps) - using float32 to save RAM
    n_sats = len(satrecs)
    n_steps = len(minutes)
    
    altitudes = np.zeros((n_sats, n_steps), dtype=np.float32)
    inclinations = np.zeros((n_sats, n_steps), dtype=np.float32)
    raans = np.zeros((n_sats, n_steps), dtype=np.float32)
    
    # Process
    # SGP4 gives position (r) and velocity (v) in TEME
    # Osculating elements are good, but for "Distribution" we want Alt/Inc
    # Inclination is in sat.inclo (mean) but osculating inclination varies slightly.
    # Let's compute osculating elements from position/velocity? 
    # Or just use the position magnitude for altitude and keep mean inclination?
    # User asked for "distributions" - osculating altitude is CRITICAL (eccentricity).
    # Osculating Inclination is less critical but good to have.
    # We will compute position magnitude -> Altitude.
    # We will use the stored Mean Inclination (sat.inclo) as approx OR compute from state vector.
    # Computing from state vector is expensive in Python loop.
    # Let's verify requirement: "Objectif : extraire des distributions".
    # Osculating altitude is the biggest variable. Mean Inclination is stable enough for "distributions".
    
    count = 0
    for idx, sat in enumerate(satrecs):
        # Propagate for all time steps
        # jd is constant, fr varies
        e, r, v = sat.sgp4_array(np.full(n_steps, jd_start), fr_start + fr_offsets)
        
        # r is (N_steps, 3) in km
        # Calculate Altitude
        # r magnitude
        
        # Check for errors (e != 0)
        # We handle this by setting NaN or 0 if e != 0
        valid_mask = (e == 0)
        
        if np.any(valid_mask):
             r_valid = r[valid_mask]
             r_norm = np.linalg.norm(r_valid, axis=1)
             alt = r_norm - EARTH_RADIUS_KM
             altitudes[idx, valid_mask] = alt
             
             # Store Mean Inclination & RAAN (converted to degrees)
             # Note: These are MEAN elements at epoch, not osculating at t.
             # SGP4 does not easily return osculating orbital elements without conversion.
             # For a "Risk Pressure Index", the Altitude distribution IS the dynamic part.
             # Inclination plane is the static part (mostly).
             inclinations[idx, :] = np.degrees(sat.inclo)
             raans[idx, :] = np.degrees(sat.nodeo) 
             
             # If we wanted osculating Inc/RAAN we would need rv2coe conversion.
             # Given "simple" requirement, Mean Inclination/RAAN is acceptable for "Planes".
             # Altitude needs to be dynamic. 
        else:
            altitudes[idx, :] = np.nan

        count += 1
        if count % 1000 == 0:
            print(f"   Processed {count}/{n_sats} objects...")

    return minutes, sat_ids, altitudes, inclinations, raans

def main():
    if not TLE_FILE.exists():
        print(f"❌ TLE file not found: {TLE_FILE}")
        return

    print("Step 1: Loading TLEs...")
    satrecs, sat_ids = load_tles(TLE_FILE)
    print(f"   Loaded {len(satrecs)} objects.")

    print("Step 2: Propagating...")
    start_time = datetime.utcnow()
    # 24h window, 30 min steps = 48 points per sat.
    # 30k sats * 48 = 1.4M points. Lightweight.
    minutes, ids, alts, incs, raans = propagate_window(satrecs, sat_ids, start_time, duration_hours=24, step_minutes=30)
    
    # Filtering: Remove dataset rows that are completely NaN or deep space
    # (Optional, but keeps file clean)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = SAMPLES_DIR / f"samples_{timestamp}.npz"
    
    print(f"Step 3: Saving to {out_file}...")
    np.savez_compressed(out_file, 
                        sat_ids=ids, 
                        time_minutes=minutes, 
                        altitude=alts, 
                        inclination=incs, 
                        raan=raans)
    
    print("✅ Done.")

if __name__ == "__main__":
    main()
