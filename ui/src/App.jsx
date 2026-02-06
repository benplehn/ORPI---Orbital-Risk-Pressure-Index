import React, { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Stars, Sphere } from '@react-three/drei';
import axios from 'axios';
import {
  Search,
  Eye,
  EyeOff,
  Trash2,
  Sparkles,
  Grid3x3,
  Crosshair,
  Info,
  Download
} from 'lucide-react';
import clsx from 'clsx';
import * as THREE from 'three';

// === Constants ===
const COLORS = ['#00f0ff', '#ff5faf', '#7bff5f', '#ffd447', '#ff7b3f'];
const EARTH_RADIUS_KM = 6378.137;
const EARTH_RADIUS_UNITS = 5; // Scene radius we use for the globe
const SCALE = EARTH_RADIUS_UNITS / EARTH_RADIUS_KM;
const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

function computeAltStatsKm(pointsKm) {
  if (!Array.isArray(pointsKm) || pointsKm.length === 0) return null;
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;
  for (const p of pointsKm) {
    if (!p || p.length < 3) continue;
    const r = Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    const alt = r - EARTH_RADIUS_KM;
    if (!Number.isFinite(alt)) continue;
    sum += alt;
    if (alt < min) min = alt;
    if (alt > max) max = alt;
  }
  if (!Number.isFinite(sum) || !Number.isFinite(min) || !Number.isFinite(max)) return null;
  const mean = sum / pointsKm.length;
  return { mean, min, max };
}

// === 3D COMPONENTS ===

function Planet() {
  const earthRef = useRef();
  const haloRef = useRef();

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    if (earthRef.current) earthRef.current.rotation.y = t * 0.08;
    if (haloRef.current) haloRef.current.rotation.y = t * 0.03;
  });

  return (
    <group rotation={[0, 0, 0.3]}>
      <Sphere ref={earthRef} args={[EARTH_RADIUS_UNITS, 96, 96]}>
        <meshPhysicalMaterial
          color="#020713"
          emissive="#041b3d"
          emissiveIntensity={0.35}
          roughness={0.15}
          metalness={0.75}
          clearcoat={1}
          clearcoatRoughness={0.08}
        />
      </Sphere>

      <Sphere args={[EARTH_RADIUS_UNITS * 1.01, 48, 48]}>
        <meshBasicMaterial color="#0ea5e9" wireframe transparent opacity={0.15} />
      </Sphere>

      <Sphere args={[EARTH_RADIUS_UNITS * 1.04, 64, 64]}>
        <meshBasicMaterial
          color="#38bdf8"
          transparent
          opacity={0.12}
          side={THREE.BackSide}
          blending={THREE.AdditiveBlending}
        />
      </Sphere>

      <Sphere ref={haloRef} args={[EARTH_RADIUS_UNITS * 1.06, 48, 48]}>
        <meshStandardMaterial
          color="#a855f7"
          transparent
          opacity={0.06}
          depthWrite={false}
          emissive="#a855f7"
          emissiveIntensity={0.25}
        />
      </Sphere>
    </group>
  );
}

function OrbitalShells({ altitudes }) {
  return (
    <group>
      {altitudes.map((alt, idx) => {
        const radius = EARTH_RADIUS_UNITS + SCALE * alt;
        return (
          <Sphere key={alt} args={[radius, 64, 64]}>
            <meshBasicMaterial
              color={idx % 2 === 0 ? '#22d3ee' : '#8b5cf6'}
              wireframe
              transparent
              opacity={0.08}
            />
          </Sphere>
        );
      })}
    </group>
  );
}

function EquatorialRing() {
  return (
    <mesh rotation={[Math.PI / 2, 0, 0]}>
      <torusGeometry args={[EARTH_RADIUS_UNITS * 1.35, 0.02, 16, 128]} />
      <meshBasicMaterial color="#22d3ee" transparent opacity={0.3} />
    </mesh>
  );
}

function OrbitPath({ points, color, visible }) {
  const scaledPoints = useMemo(() => {
    if (!Array.isArray(points)) return [];
    const out = [];
    for (const p of points) {
      if (!p || p.length < 3) continue;
      const x = Number(p[0]);
      const y = Number(p[1]);
      const z = Number(p[2]);
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
      // TEME is Z-up; R3F/three.js is Y-up. Preserve handedness by flipping one axis.
      out.push(new THREE.Vector3(x * SCALE, z * SCALE, -y * SCALE));
    }
    return out;
  }, [points]);

  const curve = useMemo(() => {
    if (scaledPoints.length < 2) return null;
    let closed = false;
    if (scaledPoints.length >= 6) {
      const a = scaledPoints[0];
      const b = scaledPoints[scaledPoints.length - 1];
      const r = a.length();
      const d = a.distanceTo(b);
      closed = r > 1e-6 && (d / r) < 0.03;
    }
    return new THREE.CatmullRomCurve3(scaledPoints, closed, 'catmullrom', 0.05);
  }, [scaledPoints]);

  const geomCore = useMemo(() => {
    if (!curve) return null;
    const tubularSegments = Math.max(64, scaledPoints.length * 2);
    return new THREE.TubeGeometry(curve, tubularSegments, 0.010, 10, Boolean(curve.closed));
  }, [curve, scaledPoints.length]);

  const geomGlow = useMemo(() => {
    if (!curve) return null;
    const tubularSegments = Math.max(64, scaledPoints.length * 2);
    return new THREE.TubeGeometry(curve, tubularSegments, 0.020, 10, Boolean(curve.closed));
  }, [curve, scaledPoints.length]);

  useEffect(() => {
    return () => {
      if (geomCore) geomCore.dispose();
      if (geomGlow) geomGlow.dispose();
    };
  }, [geomCore, geomGlow]);

  if (!visible || !geomCore || !geomGlow) return null;

  return (
    <group>
      <mesh geometry={geomGlow}>
        <meshBasicMaterial
          color={color}
          transparent
          opacity={0.12}
          blending={THREE.AdditiveBlending}
          depthTest={false}
          depthWrite={false}
          toneMapped={false}
        />
      </mesh>
      <mesh geometry={geomCore}>
        <meshBasicMaterial color={color} toneMapped={false} />
      </mesh>
    </group>
  );
}

function MovingSatellite({ points, color, visible, satId, onTelemetry }) {
  const ref = useRef();
  const scaledPoints = useMemo(() => {
    if (!Array.isArray(points)) return [];
    return points.map((p) => new THREE.Vector3(p[0] * SCALE, p[2] * SCALE, -p[1] * SCALE));
  }, [points]);

  // Deterministic per-satellite phase so multiple sats don't overlap.
  const phase = useMemo(() => {
    const n = scaledPoints.length || 1;
    return satId % n;
  }, [satId, scaledPoints.length]);

  const tmp = useMemo(() => new THREE.Vector3(), []);

  useFrame(({ clock }) => {
    if (!visible || !ref.current) return;
    const n = scaledPoints.length;
    if (n < 2) return;

    // Visual speed only (not affecting SGP4 accuracy): loop the path in ~22s.
    const loopSeconds = 22;
    const pointsPerSecond = n / loopSeconds;

    const u = (phase + clock.getElapsedTime() * pointsPerSecond) % n;
    const i0 = Math.floor(u);
    const i1 = (i0 + 1) % n;
    const a = u - i0;

    tmp.copy(scaledPoints[i0]).lerp(scaledPoints[i1], a);
    ref.current.position.copy(tmp);

    const p0 = points[i0];
    const p1 = points[i1];
    if (p0 && p1 && onTelemetry) {
      const x = p0[0] + (p1[0] - p0[0]) * a;
      const y = p0[1] + (p1[1] - p0[1]) * a;
      const z = p0[2] + (p1[2] - p0[2]) * a;
      const r = Math.sqrt(x * x + y * y + z * z);
      const altKm = r - EARTH_RADIUS_KM;
      onTelemetry(satId, altKm, tmp.x, tmp.y, tmp.z, u);
    }
  });

  if (!visible || scaledPoints.length === 0) return null;

  return (
    <group ref={ref} position={scaledPoints[phase] || undefined}>
      <Sphere args={[0.088, 16, 16]}>
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.6} />
      </Sphere>
      {/* Outer glow shell */}
      <Sphere args={[0.14, 16, 16]}>
        <meshBasicMaterial color={color} transparent opacity={0.12} blending={THREE.AdditiveBlending} />
      </Sphere>
    </group>
  );
}

function SpaceScene({ satellites, showShells, focusSatId, onTelemetry, autoRotate }) {
  const controlsRef = useRef();
  const camera = useThree((s) => s.camera);

  useEffect(() => {
    if (!focusSatId) return;
    if (!controlsRef.current) return;
    const sat = Array.isArray(satellites) ? satellites.find((s) => s.id === focusSatId) : null;
    const altKm = sat && Number.isFinite(sat.alt_mean_km) ? sat.alt_mean_km : null;
    if (!Number.isFinite(altKm)) return;

    // Earth-centric: snap the view distance so the full orbit is visible (LEO -> close, GEO -> zoomed out).
    const orbitRadius = EARTH_RADIUS_UNITS + SCALE * altKm;
    const desired = Math.max(12, Math.min(130, orbitRadius * 2.2));

    const dir = camera.position.clone().normalize();
    if (dir.lengthSq() < 1e-8) dir.set(1, 0.2, 1).normalize();
    camera.position.copy(dir.multiplyScalar(desired));
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
  }, [focusSatId, satellites, camera]);

  return (
    <>
      <color attach="background" args={['#02030a']} />
      <ambientLight intensity={0.35} />
      <pointLight position={[50, 30, 20]} intensity={2.4} color="#9cdcff" />
      <pointLight position={[-30, -20, -10]} intensity={1.6} color="#6b21a8" />
      <Stars radius={420} depth={70} count={9000} factor={4.5} saturation={0} fade speed={0.4} />

      <Planet />
      <EquatorialRing />
      {showShells && <OrbitalShells altitudes={[200, 400, 600, 800, 1000, 1200]} />}

      {satellites.map((sat) => (
        <group key={sat.id}>
          <OrbitPath points={sat.path} color={sat.color} visible={sat.visible} />
          <MovingSatellite points={sat.path} color={sat.color} visible={sat.visible} satId={sat.id} onTelemetry={onTelemetry} />
        </group>
      ))}

      <OrbitControls
        ref={controlsRef}
        // When focusing on a satellite, allow much closer zoom to inspect altitude deltas.
        minDistance={focusSatId ? 1.6 : 6.2}
        maxDistance={focusSatId ? 110 : 140}
        enablePan={false}
        zoomToCursor
        zoomSpeed={0.85}
        enableDamping
        dampingFactor={0.08}
        autoRotate={autoRotate && !focusSatId}
        autoRotateSpeed={0.45}
      />
    </>
  );
}

// === UI HELPERS ===

function Badge({ icon: Icon, label }) {
  return (
    <span className="badge">
      {React.createElement(Icon, { size: 13 })} {label}
    </span>
  );
}

// === MAIN APP ===

export default function App() {
  const [targetMode, setTargetMode] = useState('norad'); // 'norad' | 'family'
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [selected, setSelected] = useState([]); // {id,name,path,color,visible,inclination_deg,raan_deg}
  const [loading, setLoading] = useState(false);
  const [showLeftDeck, setShowLeftDeck] = useState(true);
  const [showRightDeck, setShowRightDeck] = useState(true);
  const [errorMsg, setErrorMsg] = useState('');
  const [apiOk, setApiOk] = useState(null); // null | boolean
  const [focusSatId, setFocusSatId] = useState(null);
  const [pendingAdds, setPendingAdds] = useState({}); // { [satId]: true }
  const pendingAddsRef = useRef(new Set());
  const selectedIds = useMemo(() => new Set(selected.map((s) => s.id)), [selected]);

  const noradWrapRef = useRef(null);
  const [noradOpen, setNoradOpen] = useState(false);

  // Fixed for now: no UI toggles (simpler for underwriter workflows).
  const showShells = false;
  const autoRotate = false;

  const telemetryRef = useRef({});
  const [telemetry, setTelemetry] = useState({});

  const onTelemetry = useCallback((satId, altKm, x, y, z, u) => {
    const store = telemetryRef.current || (telemetryRef.current = {});
    const t = store[satId] || (store[satId] = {});
    t.alt_km = altKm;
    t.x = x;
    t.y = y;
    t.z = z;
    t.u = u;
  }, []);

  // Family/operator browse
  const [familyQuery, setFamilyQuery] = useState('');
  const [familyResults, setFamilyResults] = useState([]); // {family,count}[]
  const [familyLoading, setFamilyLoading] = useState(false);
  const [selectedFamily, setSelectedFamily] = useState('');

  const [familySatQuery, setFamilySatQuery] = useState('');
  const [familySatResults, setFamilySatResults] = useState([]); // {id,name}[]
  const [familySatLoading, setFamilySatLoading] = useState(false);
  const familyWrapRef = useRef(null);
  const familySatWrapRef = useRef(null);
  const [familyOpen, setFamilyOpen] = useState(false);
  const [familySatOpen, setFamilySatOpen] = useState(false);

  // Debounced search
  useEffect(() => {
    if (targetMode !== 'norad') return;
    const handler = setTimeout(() => {
      const q = query.trim();
      if (q.length < 1) {
        setResults([]);
        setLoading(false);
        setErrorMsg('');
        setNoradOpen(false);
        return;
      }
      setLoading(true);
      axios
        .get(`${API_BASE}/api/satellites/search?q=${encodeURIComponent(q)}`)
        .then((res) => {
          setResults(res.data || []);
          setNoradOpen(true);
        })
        .catch((err) => {
          console.error('search failed', err);
          setErrorMsg('Backend indisponible (search). Demarre FastAPI (par defaut :8000).');
        })
        .finally(() => setLoading(false));
    }, 350);
    return () => clearTimeout(handler);
  }, [query, targetMode]);

  useEffect(() => {
    const onDown = (e) => {
      const t = e.target;
      if (noradWrapRef.current && noradWrapRef.current.contains(t)) {
        // keep open
      } else {
        setNoradOpen(false);
      }
      if (familyWrapRef.current && familyWrapRef.current.contains(t)) {
        // keep open
      } else {
        setFamilyOpen(false);
      }
      if (familySatWrapRef.current && familySatWrapRef.current.contains(t)) {
        // keep open
      } else {
        setFamilySatOpen(false);
      }
    };
    window.addEventListener('pointerdown', onDown);
    return () => window.removeEventListener('pointerdown', onDown);
  }, []);

  useEffect(() => {
    axios
      .get(`${API_BASE}/health`)
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false));
  }, []);

  // Pull live satellite telemetry from the 3D loop at a low rate for the UI.
  useEffect(() => {
    const id = setInterval(() => {
      const snap = telemetryRef.current || {};
      const next = {};
      for (const [k, v] of Object.entries(snap)) {
        next[k] = { ...v };
      }
      setTelemetry(next);
    }, 450);
    return () => clearInterval(id);
  }, []);

  // Debounced family list
  useEffect(() => {
    if (targetMode !== 'family') return;
    const handler = setTimeout(() => {
      const q = familyQuery.trim();
      setFamilyLoading(true);
      axios
        .get(`${API_BASE}/api/satellites/families?limit=30&q=${encodeURIComponent(q)}`)
        .then((res) => {
          setFamilyResults(res.data || []);
          setFamilyOpen(true);
        })
        .catch((err) => {
          console.error('families failed', err);
          setErrorMsg("Impossible de charger la liste d'operateurs (API).");
        })
        .finally(() => setFamilyLoading(false));
    }, 250);
    return () => clearTimeout(handler);
  }, [familyQuery, targetMode]);

  // Debounced satellites within family
  useEffect(() => {
    if (targetMode !== 'family') return;
    if (!selectedFamily) return;
    const handler = setTimeout(() => {
      const q = familySatQuery.trim();
      setFamilySatLoading(true);
      axios
        .get(
          `${API_BASE}/api/satellites/by_family?family=${encodeURIComponent(selectedFamily)}&limit=50&q=${encodeURIComponent(q)}`
        )
        .then((res) => {
          setFamilySatResults(res.data || []);
          setFamilySatOpen(true);
        })
        .catch((err) => {
          console.error('by_family failed', err);
          setErrorMsg("Impossible de charger les satellites de l'operateur (API).");
        })
        .finally(() => setFamilySatLoading(false));
    }, 250);
    return () => clearTimeout(handler);
  }, [familySatQuery, selectedFamily, targetMode]);

  const addSatellite = useCallback(
    (sat) => {
      const id = sat?.id;
      if (!Number.isFinite(Number(id))) return;
      const satId = Number(id);
      if (selectedIds.has(satId)) return;
      if (pendingAddsRef.current.has(satId)) return;

      pendingAddsRef.current.add(satId);
      setPendingAdds((prev) => ({ ...prev, [satId]: true }));
      setErrorMsg('');

      axios
        .get(`${API_BASE}/api/satellites/${satId}/orbit`)
        .then((res) => {
          const pts = res.data.points || [];
          if (!Array.isArray(pts) || pts.length === 0) {
            setErrorMsg(`Orbite vide pour ID ${satId}.`);
            return;
          }
          const altStats = computeAltStatsKm(pts);
          setSelected((prev) => {
            if (prev.find((s) => s.id === satId)) return prev;
            const color = COLORS[prev.length % COLORS.length];
            return [
              ...prev,
              {
                id: res.data.id,
                name: res.data.name,
                path: pts,
                color,
                visible: true,
                alt_mean_km: altStats ? altStats.mean : null,
                alt_min_km: altStats ? altStats.min : null,
                alt_max_km: altStats ? altStats.max : null,
                inclination_deg: res.data.inclination_deg,
                raan_deg: res.data.raan_deg
              }
            ];
          });

          // Keep dropdown open so users can add multiple sats smoothly.
          setResults((prev) => (Array.isArray(prev) ? prev.filter((r) => r.id !== satId) : prev));
          setFamilySatResults((prev) => (Array.isArray(prev) ? prev.filter((r) => r.id !== satId) : prev));

          // Auto-focus the first selected satellite (less clicks for non-technical users).
          setFocusSatId((cur) => (cur ? cur : res.data.id));
        })
        .catch((err) => {
          console.error('orbit fetch failed', err);
          setErrorMsg(`Impossible de recuperer l'orbite pour ID ${satId} (API).`);
        })
        .finally(() => {
          pendingAddsRef.current.delete(satId);
          setPendingAdds((prev) => {
            const next = { ...prev };
            delete next[satId];
            return next;
          });
        });
    },
    [selectedIds]
  );

  const pickFamily = (fam) => {
    setSelectedFamily(fam);
    setFamilyQuery(fam);
    setFamilyResults([]);
    setFamilyOpen(false);
    setFamilySatQuery('');
    setFamilySatResults([]);
    setFamilySatOpen(true);
    setErrorMsg('');
  };

  const pickFamilySatellite = (sat) => {
    addSatellite(sat);
    // Keep the dropdown/query so users can add multiple satellites quickly.
    setFamilySatOpen(true);
  };

  const toggleSatellite = (id) => {
    setSelected((prev) =>
      prev.map((s) => (s.id === id ? { ...s, visible: !s.visible } : s))
    );
  };

  const removeSatellite = (id) => {
    setSelected((prev) => prev.filter((s) => s.id !== id));
    if (focusSatId === id) setFocusSatId(null);
    if (telemetryRef.current) delete telemetryRef.current[id];
  };

  const toggleFocus = (id) => {
    setFocusSatId((cur) => (cur === id ? null : id));
  };

  // ORPI score + percentiles + explanation for the currently focused satellite.
  const [orpi, setOrpi] = useState(null);
  const [orpiLoading, setOrpiLoading] = useState(false);
  const lastOrpiKeyRef = useRef('');
  const pendingOrpiKeyRef = useRef('');

  const focusedSat = useMemo(() => selected.find((s) => s.id === focusSatId) || null, [selected, focusSatId]);
  const focusedTel = focusSatId ? telemetry[String(focusSatId)] : null;
  const focusedAltForScore = useMemo(() => {
    if (!focusedSat) return null;
    if (Number.isFinite(focusedSat.alt_mean_km)) return focusedSat.alt_mean_km;
    if (focusedTel && Number.isFinite(focusedTel.alt_km)) return focusedTel.alt_km;
    return null;
  }, [focusedSat, focusedTel]);
  const orpiInRange = useMemo(() => {
    return Number.isFinite(focusedAltForScore) && focusedAltForScore >= 200 && focusedAltForScore < 42000;
  }, [focusedAltForScore]);

  useEffect(() => {
    if (!focusedSat) {
      const handler = setTimeout(() => setOrpi(null), 0);
      return () => clearTimeout(handler);
    }
    const altForScore = focusedAltForScore;
    if (!Number.isFinite(altForScore) || !Number.isFinite(focusedSat.inclination_deg)) {
      const handler = setTimeout(() => setOrpi(null), 0);
      return () => clearTimeout(handler);
    }
    if (!orpiInRange) {
      const handler = setTimeout(() => setOrpi(null), 0);
      return () => clearTimeout(handler);
    }

    const altKey = Math.round(altForScore);
    const incKey = Math.round(focusedSat.inclination_deg * 10) / 10;
    const key = `${focusedSat.id}_${altKey}_${incKey}`;
    if (key === lastOrpiKeyRef.current) return;
    if (key === pendingOrpiKeyRef.current) return;
    pendingOrpiKeyRef.current = key;

    const handler = setTimeout(() => {
      lastOrpiKeyRef.current = key;
      setOrpiLoading(true);
      axios
        .get(`${API_BASE}/api/orpi/score?altitude=${encodeURIComponent(altForScore)}&inclination=${encodeURIComponent(focusedSat.inclination_deg)}`)
        .then((res) => setOrpi(res.data))
        .catch((err) => {
          console.error('orpi score failed', err);
          setOrpi(null);
        })
        .finally(() => {
          pendingOrpiKeyRef.current = '';
          setOrpiLoading(false);
        });
    }, 0);

    return () => {
      clearTimeout(handler);
      if (pendingOrpiKeyRef.current === key) pendingOrpiKeyRef.current = '';
    };
  }, [focusedSat, focusedAltForScore, orpiInRange]);

  return (
    <div className="app">
      <div className="viewport">
        <Canvas camera={{ position: [16, 8, 16], fov: 42 }} gl={{ antialias: true, toneMapping: THREE.ACESFilmicToneMapping }}>
          <SpaceScene
            satellites={selected}
            showShells={showShells}
            focusSatId={focusSatId}
            onTelemetry={onTelemetry}
            autoRotate={autoRotate}
          />
        </Canvas>
      </div>

      {/* HUD */}
      <div className="hud">
        <div className="quickbar panel-weak" role="toolbar" aria-label="Quick controls">
          <button
            className={clsx('qbtn', showLeftDeck ? 'qbtn-on' : null)}
            onClick={() => setShowLeftDeck((v) => !v)}
            title={showLeftDeck ? 'Masquer le panneau satellites' : 'Afficher le panneau satellites'}
            aria-label="Toggle satellites panel"
          >
            <Search size={15} />
          </button>
          <button
            className={clsx('qbtn', showRightDeck ? 'qbtn-on' : null)}
            onClick={() => setShowRightDeck((v) => !v)}
            title={showRightDeck ? 'Masquer le brief ORPI' : 'Afficher le brief ORPI'}
            aria-label="Toggle ORPI brief"
          >
            <Sparkles size={15} />
          </button>
          <a
            className="qbtn"
            href="/info.html"
            target="_blank"
            rel="noreferrer"
            title="Infos ORPI"
            aria-label="Open ORPI guide"
          >
            <Info size={15} />
          </a>
        </div>

        {!showRightDeck && (
          <button
            className="mini-brief panel-weak"
            onClick={() => setShowRightDeck(true)}
            title="Ouvrir le brief ORPI"
            aria-label="Open ORPI brief"
          >
            <div className="mini-score">{orpi ? Math.round(orpi.orpi_score) : '—'}</div>
            <div className="mini-label">{orpi?.rating || 'ORPI v0'}</div>
          </button>
        )}

        {/* Left Control Deck */}
        {showLeftDeck && (
        <div className="deck deck-left">
          <div className="panel-weak">
            <div className="panel-pad" style={{ position: 'relative' }}>
              <div className="row" style={{ marginBottom: 12 }}>
                <div className="brandmark">
                  <div className="brandmark-title">ORPI</div>
                  <div className="brandmark-sub">Orbital Pressure Index • VIS</div>
                </div>
                <Badge icon={Sparkles} label={apiOk === false ? 'API OFF' : 'LIVE'} />
              </div>

              <div className="section-title">
                <Search size={14} /> Cibler un satellite
              </div>

              <div className="seg" style={{ marginTop: 10 }}>
                <button
                  className={targetMode === 'norad' ? 'seg-on' : ''}
                  onClick={() => {
                    setTargetMode('norad');
                    setErrorMsg('');
                  }}
                >
                  NORAD / Nom
                </button>
                <button
                  className={targetMode === 'family' ? 'seg-on' : ''}
                  onClick={() => {
                    setTargetMode('family');
                    setErrorMsg('');
                    // trigger initial load
                    setFamilyQuery((v) => v);
                  }}
                >
                  Operateur (INTELSAT…)
                </button>
              </div>

              {targetMode === 'norad' && (
              <div ref={noradWrapRef} className="input-wrap" style={{ marginTop: 10 }}>
                <input
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onFocus={() => setNoradOpen(true)}
                  onKeyDown={(e) => {
                    if (e.key === 'Escape') setNoradOpen(false);
                  }}
                  placeholder="NORAD ID (ou texte si dispo)..."
                  className="input"
                />
                <span className="input-icon">
                  <Search size={16} />
                </span>
                {loading && (
                  <div className="scan">scan…</div>
                )}

                {targetMode === 'norad' && noradOpen && results.length > 0 && (
                  <div className="dropdown">
                    {results.map((sat) => {
                      const added = selectedIds.has(sat.id);
                      const pending = !!pendingAdds[sat.id];
                      return (
                        <button
                          key={sat.id}
                          onClick={() => addSatellite(sat)}
                          disabled={added || pending}
                        >
                          <span style={{ fontFamily: 'var(--mono)', fontSize: 13 }}>{sat.name}</span>
                          <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'rgba(34,211,238,0.85)' }}>
                            {added ? 'ADDED' : pending ? 'FETCH…' : '+ ADD'}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>
              )}

              {targetMode === 'family' && (
                <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <div ref={familyWrapRef} className="input-wrap">
                    <input
                      value={familyQuery}
                      onChange={(e) => setFamilyQuery(e.target.value)}
                      onFocus={() => setFamilyOpen(true)}
                      onKeyDown={(e) => {
                        if (e.key === 'Escape') setFamilyOpen(false);
                      }}
                      placeholder="Choisir un operateur (ex: INTELSAT, STARLINK, ONEWEB)…"
                      className="input"
                    />
                    <span className="input-icon">
                      <Search size={16} />
                    </span>
                    {familyLoading && <div className="scan">scan…</div>}
                    {familyOpen && familyResults.length > 0 && (
                      <div className="dropdown">
                        {familyResults.map((f) => (
                          <button
                            key={f.family}
                            onClick={() => pickFamily(f.family)}
                          >
                            <span style={{ fontFamily: 'var(--mono)', fontSize: 13 }}>
                              {f.family}
                            </span>
                            <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'rgba(34,211,238,0.65)' }}>
                              {f.count}
                            </span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>

                  <div ref={familySatWrapRef} className="input-wrap">
                    <input
                      value={familySatQuery}
                      onChange={(e) => setFamilySatQuery(e.target.value)}
                      onFocus={() => setFamilySatOpen(true)}
                      onKeyDown={(e) => {
                        if (e.key === 'Escape') setFamilySatOpen(false);
                      }}
                      placeholder={selectedFamily ? `Satellite chez ${selectedFamily} (nom ou NORAD)…` : 'Selectionne un operateur au-dessus…'}
                      className="input"
                      disabled={!selectedFamily}
                      style={!selectedFamily ? { opacity: 0.55, cursor: 'not-allowed' } : undefined}
                    />
                    <span className="input-icon">
                      <Search size={16} />
                    </span>
                    {familySatLoading && selectedFamily && <div className="scan">scan…</div>}
                    {familySatOpen && selectedFamily && familySatResults.length > 0 && (
                      <div className="dropdown">
                        {familySatResults.map((sat) => {
                          const added = selectedIds.has(sat.id);
                          const pending = !!pendingAdds[sat.id];
                          return (
                            <button
                              key={sat.id}
                              onClick={() => pickFamilySatellite(sat)}
                              disabled={added || pending}
                            >
                              <span style={{ fontFamily: 'var(--mono)', fontSize: 13 }}>{sat.name}</span>
                              <span style={{ fontFamily: 'var(--mono)', fontSize: 10, color: 'rgba(34,211,238,0.85)' }}>
                                {added ? 'ADDED' : pending ? 'FETCH…' : '+ ADD'}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </div>
              )}

              {errorMsg && (
                <div style={{ marginTop: 10, fontFamily: 'var(--mono)', fontSize: 12, color: 'rgba(251,113,133,0.95)' }}>
                  {errorMsg}
                </div>
              )}

            </div>

            <div className="divider" />

            <div className="panel-pad" style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div className="row">
                <div className="section-title">Pistes actives</div>
                <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                  <Badge icon={Grid3x3} label={`${selected.length} actifs`} />
                </div>
              </div>

              {selected.length === 0 && (
                <div className="sat-empty">Aucun satellite sélectionné</div>
              )}

              <div className="sat-list">
                {selected.map((sat) => {
                  const t = telemetry[String(sat.id)];
                  const altBase = Number.isFinite(sat.alt_mean_km) ? sat.alt_mean_km : (t && Number.isFinite(t.alt_km) ? t.alt_km : null);
                  const altText = Number.isFinite(altBase) ? `${altBase.toFixed(0)} km` : '—';
                  const incText = Number.isFinite(sat.inclination_deg) ? `${sat.inclination_deg.toFixed(1)}°` : '—';
                  return (
                  <div
                    key={sat.id}
                    className={clsx('sat-item', focusSatId === sat.id ? 'sat-item-focus' : null)}
                  >
                    <div className="sat-meta">
                      <span
                        className="dot"
                        style={{ background: sat.color, color: sat.color }}
                      />
                      <div>
                        <div className="sat-name">{sat.name}</div>
                        <div className="sat-id">ID {sat.id} • ALT {altText} • INC {incText}</div>
                      </div>
                    </div>
                    <div className="btn-group">
                      <button
                        onClick={() => toggleFocus(sat.id)}
                        className={clsx('btn', focusSatId === sat.id ? 'btn-on' : null)}
                        title="Focus camera"
                      >
                        <Crosshair size={14} />
                      </button>
                      <button
                        onClick={() => toggleSatellite(sat.id)}
                        className={clsx(
                          'btn',
                          sat.visible
                            ? 'btn-on'
                            : null
                        )}
                      >
                        {sat.visible ? <Eye size={14} /> : <EyeOff size={14} />}
                      </button>
                      <button
                        onClick={() => removeSatellite(sat.id)}
                        className="btn btn-danger"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>
                  </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
        )}

        {/* Right Risk Brief (AXA-friendly) */}
        {showRightDeck && (
        <div className="deck-right">
          <div className="panel-weak right-card">
            <div className="row">
              <div className="section-title">ORPI v0</div>
              <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
                {focusedSat && (
                  <a
                    className="badge badge-link"
                    href={`${API_BASE}/api/brief/${focusedSat.id}.pdf`}
                    target="_blank"
                    rel="noreferrer"
                    title="Exporter un underwriting brief (PDF)"
                  >
                    <Download size={13} /> PDF
                  </a>
                )}
                <Badge
                  icon={Sparkles}
                  label={
                    orpiLoading
                      ? 'CALC'
                      : (!focusedSat ? '—' : (orpiInRange ? (orpi?.rating || '—') : 'OUT'))
                  }
                />
              </div>
            </div>

            {!focusedSat && (
              <div className="sat-empty">
                Selectionne un satellite puis clique <span style={{ opacity: 0.9 }}>Focus</span> (viseur) pour afficher le score.
              </div>
            )}

            {focusedSat && (
              <>
                <div className="brief-head">
                  <div className="brief-name">{focusedSat.name}</div>
                  <div className="brief-sub">
                    NORAD {focusedSat.id} • ALT {Number.isFinite(focusedSat.alt_mean_km) ? `${focusedSat.alt_mean_km.toFixed(0)} km` : (focusedTel && Number.isFinite(focusedTel.alt_km) ? `${focusedTel.alt_km.toFixed(0)} km` : '—')} • INC {Number.isFinite(focusedSat.inclination_deg) ? `${focusedSat.inclination_deg.toFixed(1)}°` : '—'}
                  </div>
                </div>

                {!orpiInRange && Number.isFinite(focusedAltForScore) && (
                  <div className="sat-empty">
                    ORPI v0 couvre LEO/MEO/GEO (200–42000 km). Ce satellite est hors plage ({focusedAltForScore.toFixed(0)} km).
                  </div>
                )}

                <div className="score-row">
                  <div className="score-big">{orpi && orpiInRange ? Math.round(orpi.orpi_score) : 'N/A'}</div>
                  <div className="score-meta">
                    <div className="score-pill">Percentile P{orpi && orpiInRange ? Math.round(orpi.percentile) : '—'}</div>
                    <div className="score-pill">
                      Cellule alt {orpi && orpiInRange ? Math.round(orpi.cell.alt_bin_start) : '—'} / inc {orpi && orpiInRange ? Math.round(orpi.cell.inc_bin_start) : '—'}
                    </div>
                  </div>
                </div>

                {orpi && orpiInRange && (
                  <>
                    <div className="comp">
                      <div className="comp-row">
                        <div className="comp-k">Pression (N_eff x Vrel)</div>
                        <div className="comp-v">P{Math.round(orpi.components.pressure.percentile)}</div>
                        <div className="comp-bar"><div className="comp-fill" style={{ width: `${orpi.components.pressure.percentile}%` }} /></div>
                      </div>
                      <div className="comp-row">
                        <div className="comp-k">Variabilite (sigma)</div>
                        <div className="comp-v">P{Math.round(orpi.components.volatility.percentile)}</div>
                        <div className="comp-bar"><div className="comp-fill comp-fill2" style={{ width: `${orpi.components.volatility.percentile}%` }} /></div>
                      </div>
                      <div className="comp-row">
                        <div className="comp-k">Croissance (scenario annualisee)</div>
                        <div className="comp-v">P{Math.round(orpi.components.growth.percentile)}</div>
                        <div className="comp-bar"><div className="comp-fill comp-fill3" style={{ width: `${orpi.components.growth.percentile}%` }} /></div>
                      </div>
                    </div>

                    <div className="just">{orpi.justification}</div>

                    <details className="details">
                      <summary>Details & hypothese</summary>
                      <div className="details-body">
                        <div className="details-kv">
                          <div>Cellule utilisee:</div>
                          <div>alt {orpi.cell.alt_bin_start} km / inc {orpi.cell.inc_bin_start} deg</div>
                        </div>
                        <div className="details-kv">
                          <div>Composantes (0-100):</div>
                          <div>P {orpi.components.pressure.score} • S {orpi.components.volatility.score} • G {orpi.components.growth.score}</div>
                        </div>
                        <div className="details-kv">
                          <div>Features brutes:</div>
                          <div>
                            N_eff={orpi.features.n_eff_sum} • Vrel={orpi.features.vrel_mean_proxy_km_s} km/s • Pression={orpi.features.pressure_mean} • Sigma={orpi.features.risk_sigma} • Delta={orpi.features.trend_total} • Delta/an={orpi.features.trend_annual}
                          </div>
                        </div>
                        <div className="details-kv">
                          <div>Drivers:</div>
                          <div>
                            Densite P{Math.round(orpi.drivers?.density?.percentile ?? 0)} • Vrel P{Math.round(orpi.drivers?.vrel?.percentile ?? 0)}
                          </div>
                        </div>
                        {orpi.scenario && (orpi.scenario.name || orpi.scenario.target_date) && (
                          <div className="details-kv">
                            <div>Scenario:</div>
                            <div>
                              {(orpi.scenario.name || '—')}
                              {orpi.scenario.target_date ? ` (cible ${orpi.scenario.target_date})` : ''}
                              {Number.isFinite(orpi.scenario.years_to_target) ? ` • annualise sur ${orpi.scenario.years_to_target.toFixed(2)} ans` : ''}
                              {orpi.scenario.deployment_profile ? ` • ${orpi.scenario.deployment_profile}` : ''}
                            </div>
                          </div>
                        )}
                        <div className="details-note">
                          Pression = proxy de flux (N_eff x Vrel). Variabilite = sigma sur la fenetre (robuste). Croissance = delta annualise sous scenario declaratif (non garanti).
                        </div>
                        <div className="details-note">
                          Indice comparatif base sur des percentiles (robuste). Orbites = trajectoires SGP4. Point = satellite (animation).
                        </div>
                      </div>
                    </details>
                  </>
                )}
              </>
            )}
          </div>
        </div>
        )}
      </div>
    </div>
  );
}
