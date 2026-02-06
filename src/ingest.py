import os
import datetime
import requests
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
TLE_DIR = DATA_DIR / "tle"
SCENARIO_DIR = DATA_DIR / "scenarios"

# Ensure directories exist
TLE_DIR.mkdir(parents=True, exist_ok=True)
SCENARIO_DIR.mkdir(parents=True, exist_ok=True)

def fetch_celestrak_tles(catalog="active"):
    """
    Fetch TLEs from CelesTrak (Public, no login required).
    catalog options: 'active', 'stations', 'starlink', etc.
    """
    print(f"📡 Fetching TLEs from CelesTrak ({catalog})...")
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={catalog}&FORMAT=tle"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"❌ Error fetching from CelesTrak: {e}")
        return None

def fetch_spacetrack_tles(username, password):
    """
    Fetch TLEs from Space-Track.org using raw requests (Session).
    Fetches the full catalog of objects (gp class) with specific columns or full TLE.
    """
    base_url = "https://www.space-track.org"
    auth_url = f"{base_url}/ajaxauth/login"
    
    # IMPORTANT:
    # The `current/Y` predicate on class=gp can return 500 errors on Space-Track.
    # We avoid it completely and instead:
    # - pull recent element sets (JSON includes TLE_LINE0/1/2 => name + 2 lines)
    # - order by NORAD_CAT_ID then EPOCH desc
    # - keep the first record per NORAD (latest)
    epoch_days = int(os.getenv("SPACETRACK_EPOCH_DAYS", "21"))
    limit_rows = int(os.getenv("SPACETRACK_LIMIT_ROWS", "200000"))
    query_url = (
        f"{base_url}/basicspacedata/query/"
        f"class/gp/decay_date/null-val/EPOCH/>now-{epoch_days}/"
        "orderby/NORAD_CAT_ID,EPOCH desc/format/json"
    )

    print(f"🔐 Authenticating with Space-Track as {username}...")
    
    session = requests.Session()
    
    try:
        # 1. Login
        login_payload = {
            "identity": username,
            "password": password
        }
        resp = session.post(auth_url, data=login_payload, timeout=30)
        resp.raise_for_status()
        
        if resp.status_code == 200:
             # Check if login was actually successful (Space-Track returns 200 even on failure sometimes, but cookie is key)
             # usually if cookie is set it's good.
             pass
        
        print("📡 Fetching recent GP element sets (JSON) and building 3LE with names...")

        seen = set()
        out_lines = []
        total_rows = 0

        # Space-Track's `offset/` can intermittently 500; avoid pagination and use a large limit.
        data_resp = session.get(f"{query_url}/limit/{limit_rows}", timeout=240)
        data_resp.raise_for_status()

        text = data_resp.text
        if "The Login-password you provided was incorrect" in text:
            print("❌ Authentication failed (Bad password).")
            return None

        try:
            rows = data_resp.json()
        except Exception:
            print("❌ Unexpected response format from Space-Track (expected JSON).")
            return None

        if not rows:
            print("⚠️ Received empty response from Space-Track.")
            return None

        total_rows = len(rows)
        for r in rows:
            norad_raw = r.get("NORAD_CAT_ID")
            if norad_raw is None:
                continue
            try:
                norad = int(norad_raw)
            except Exception:
                continue
            if norad in seen:
                continue
            seen.add(norad)

            l0 = (r.get("TLE_LINE0") or "").strip()
            l1 = (r.get("TLE_LINE1") or "").strip()
            l2 = (r.get("TLE_LINE2") or "").strip()
            if not (l0 and l1 and l2):
                continue
            out_lines.extend([l0, l1, l2])

        if not out_lines:
            print("⚠️ Received no usable TLE lines from Space-Track.")
            return None

        print(f"✅ Space-Track rows fetched: {total_rows} | unique NORAD kept: {len(seen)}")
        return "\n".join(out_lines) + "\n"

    except requests.RequestException as e:
        print(f"❌ Error fetching from Space-Track (Network): {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None
    finally:
        session.close()

def save_tles(tle_content, source):
    """
    Save TLE content to a versioned file.
    Format: tle_{source}_{YYYYMMDD_HHMMSS}.txt
    """
    if not tle_content:
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tle_{source}_{timestamp}.txt"
    filepath = TLE_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(tle_content)
    
    # Also save a 'latest.txt'
    latest_path = TLE_DIR / "latest.txt"
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(tle_content)
    
    print(f"✅ TLEs saved to: {filepath}")
    return filepath

def main():
    print("🚀 ORPI Data Ingestion Service")
    
    st_user = os.getenv("SPACETRACK_USER")
    st_pass = os.getenv("SPACETRACK_PASSWORD")

    if st_user and st_pass:
        print("👉 Using Space-Track")
        tles = fetch_spacetrack_tles(st_user, st_pass)
        source = "spacetrack"
        if not tles:
            print("⚠️ Space-Track indisponible. Fallback vers CelesTrak (public).")
            tles = fetch_celestrak_tles("active")
            source = "celestrak"
    else:
        print("⚠️ No Space-Track credentials found.")
        print("👉 Defaulting to CelesTrak")
        tles = fetch_celestrak_tles("active")
        source = "celestrak"

    if tles:
        path = save_tles(tles, source)
        line_count = len(tles.strip().split('\n'))
        # TLE is usually 2 or 3 lines per object.
        # With Space-Track format=3le and CelesTrak FORMAT=tle we expect 3-line sets.
        est_objects = line_count // 3 if " 0 " in ("\n" + tles[:2000]) or line_count % 3 == 0 else line_count // 2
        print(f"📊 Stats: ~{est_objects} objects imported (Lines: {line_count}).")
    else:
        print("❌ Failed to ingest data.")

if __name__ == "__main__":
    main()
