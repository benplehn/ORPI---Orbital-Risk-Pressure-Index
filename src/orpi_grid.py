import numpy as np

# Shared grid definition used by the offline pipeline (grid_builder.py)
# and the online API scoring (api.py).
#
# Goal: keep LEO resolution fine, while keeping the overall grid size
# manageable when extending to MEO/GEO.

# Altitude bin bands: (start_km, end_km, step_km)
# End is inclusive for edge generation, and the last edge is the exclusive max.
ALT_BANDS_KM = [
    (200.0, 2000.0, 10.0),      # LEO
    (2000.0, 20000.0, 50.0),    # MEO
    (20000.0, 42000.0, 100.0),  # GEO / HEO up to ~GEO altitude
]

INC_MIN_DEG = 0.0
INC_MAX_DEG = 180.0
INC_BIN_SIZE_DEG = 2.0

RAAN_MIN_DEG = 0.0
RAAN_MAX_DEG = 360.0
RAAN_BIN_SIZE_DEG = 10.0


def _edges_from_band(start: float, end: float, step: float) -> np.ndarray:
    """Return edges including both endpoints for an arithmetic grid."""
    start = float(start)
    end = float(end)
    step = float(step)
    if step <= 0:
        raise ValueError("step must be > 0")
    n = int(round((end - start) / step))
    if n < 1:
        raise ValueError("band too small")
    return (start + step * np.arange(n + 1, dtype=np.float32)).astype(np.float32)


def edges_alt_km() -> np.ndarray:
    parts = []
    for i, (a0, a1, da) in enumerate(ALT_BANDS_KM):
        e = _edges_from_band(a0, a1, da)
        if i > 0:
            # Remove the first edge to avoid duplicates at boundaries (e.g. 2000).
            e = e[1:]
        parts.append(e)
    edges = np.concatenate(parts).astype(np.float32)
    # Defensive: ensure strict monotonicity.
    edges = np.unique(edges)
    edges.sort()
    return edges


def edges_uniform(min_v: float, max_v: float, step: float) -> np.ndarray:
    min_v = float(min_v)
    max_v = float(max_v)
    step = float(step)
    n = int(round((max_v - min_v) / step))
    return (min_v + step * np.arange(n + 1, dtype=np.float32)).astype(np.float32)


ALT_EDGES_KM = edges_alt_km()
INC_EDGES_DEG = edges_uniform(INC_MIN_DEG, INC_MAX_DEG, INC_BIN_SIZE_DEG)
RAAN_EDGES_DEG = edges_uniform(RAAN_MIN_DEG, RAAN_MAX_DEG, RAAN_BIN_SIZE_DEG)

ALT_MIN_KM = float(ALT_EDGES_KM[0])
ALT_MAX_KM = float(ALT_EDGES_KM[-1])  # exclusive


def bin_start_from_edges(value: float, edges: np.ndarray):
    """
    Return the bin start edge for value, or None if out of range.
    Range is [edges[0], edges[-1]).
    """
    v = float(value)
    if v < float(edges[0]) or v >= float(edges[-1]):
        return None
    idx = int(np.searchsorted(edges, v, side="right") - 1)
    if idx < 0 or idx >= len(edges) - 1:
        return None
    return float(edges[idx])

