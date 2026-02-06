import io
import math
from datetime import datetime, timedelta, timezone


def _safe_float(x, default=None):
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def satrec_epoch_utc(satrec):
    """Convert sgp4 Satrec epoch (epochyr, epochdays) to UTC datetime."""
    try:
        yy = int(getattr(satrec, "epochyr"))
        dd = float(getattr(satrec, "epochdays"))
    except Exception:
        return None

    year = 2000 + yy if yy < 57 else 1900 + yy
    try:
        d0 = datetime(year, 1, 1, tzinfo=timezone.utc)
        return d0 + timedelta(days=dd - 1.0)
    except Exception:
        return None


def orbit_altitude_series_km(points_km, earth_radius_km=6378.137):
    """Return (altitudes_km, min_km, mean_km, max_km) from TEME points."""
    if not points_km:
        return [], None, None, None
    alts = []
    for p in points_km:
        if not p or len(p) < 3:
            continue
        x = _safe_float(p[0])
        y = _safe_float(p[1])
        z = _safe_float(p[2])
        if x is None or y is None or z is None:
            continue
        r = math.sqrt(x * x + y * y + z * z)
        alts.append(r - float(earth_radius_km))
    if not alts:
        return [], None, None, None
    mn = min(alts)
    mx = max(alts)
    mean = sum(alts) / float(len(alts))
    return alts, mn, mean, mx


def _risk_color(rating):
    r = (rating or "").upper()
    if r == "CRITICAL":
        return (0.68, 0.12, 0.12)
    if r == "HIGH":
        return (0.74, 0.40, 0.07)
    if r == "MODERATE":
        return (0.74, 0.63, 0.12)
    return (0.12, 0.47, 0.24)


def _fmt(v, nd=3, none="N/A"):
    if v is None:
        return none
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return none


def _draw_wrapped_text(c, text, x, y, width, font="Helvetica", size=10, leading=13):
    from reportlab.pdfbase.pdfmetrics import stringWidth

    c.setFont(font, size)
    words = (text or "").split()
    if not words:
        return y

    line = words[0]
    yy = y
    for w in words[1:]:
        candidate = f"{line} {w}"
        if stringWidth(candidate, font, size) <= width:
            line = candidate
            continue
        c.drawString(x, yy, line)
        yy -= leading
        line = w
    c.drawString(x, yy, line)
    return yy


def _draw_percentile_bar(c, x, y, w, h, pct, fill_rgb, label, right_label):
    p = max(0.0, min(100.0, _safe_float(pct, 0.0)))
    c.setFillColorRGB(0.95, 0.95, 0.95)
    c.rect(x, y, w, h, stroke=0, fill=1)
    c.setFillColorRGB(fill_rgb[0], fill_rgb[1], fill_rgb[2])
    c.rect(x, y, w * (p / 100.0), h, stroke=0, fill=1)
    c.setStrokeColorRGB(0.82, 0.82, 0.82)
    c.rect(x, y, w, h, stroke=1, fill=0)

    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.12, 0.12, 0.12)
    c.drawString(x, y + h + 3, label)
    c.setFont("Helvetica-Bold", 9)
    c.drawRightString(x + w, y + h + 3, right_label)


def _draw_altitude_chart(c, alts, period_min, x, y, w, h):
    c.setFillColorRGB(1, 1, 1)
    c.rect(x, y, w, h, stroke=1, fill=1)

    if not alts or len(alts) < 2:
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.5, 0.5, 0.5)
        c.drawString(x + 10, y + h / 2, "No orbit profile available")
        return

    ymin = min(alts)
    ymax = max(alts)
    if abs(ymax - ymin) < 1e-6:
        ymax = ymin + 1.0
    pad = max(5.0, (ymax - ymin) * 0.08)
    ymin -= pad
    ymax += pad

    plot_x = x + 44
    plot_y = y + 22
    plot_w = w - 60
    plot_h = h - 40

    # Grid
    c.setStrokeColorRGB(0.90, 0.90, 0.90)
    c.setLineWidth(1)
    for i in range(6):
        gy = plot_y + plot_h * i / 5.0
        c.line(plot_x, gy, plot_x + plot_w, gy)
        val = ymin + (ymax - ymin) * i / 5.0
        c.setFillColorRGB(0.35, 0.35, 0.35)
        c.setFont("Helvetica", 8)
        c.drawRightString(plot_x - 6, gy - 2, f"{val:.0f}")

    for frac in [0.0, 0.25, 0.50, 0.75, 1.0]:
        gx = plot_x + plot_w * frac
        c.setStrokeColorRGB(0.92, 0.92, 0.92)
        c.line(gx, plot_y, gx, plot_y + plot_h)
        c.setFillColorRGB(0.35, 0.35, 0.35)
        c.setFont("Helvetica", 8)
        c.drawCentredString(gx, y + 8, f"{int(round((period_min or 0) * frac))}m")

    # Line
    c.setStrokeColorRGB(0.06, 0.45, 0.76)
    c.setLineWidth(1.6)
    for i, a in enumerate(alts):
        xx = plot_x + plot_w * i / float(len(alts) - 1)
        yy = plot_y + plot_h * ((a - ymin) / (ymax - ymin))
        if i == 0:
            c.line(xx, yy, xx, yy)
        else:
            c.line(px, py, xx, yy)
        px, py = xx, yy

    c.setFillColorRGB(0.20, 0.20, 0.20)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 8, y + h - 14, "Altitude profile over one propagated orbital period")
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.drawString(x + 8, y + h - 25, "Altitude (km)")


def render_orpi_brief_pdf(ctx):
    """
    Render a one-page, decision-oriented underwriting brief in English.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except Exception as e:
        raise RuntimeError("Missing dependency: reportlab. Install with `pip install reportlab`.") from e

    W, H = A4
    margin = 26
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)

    sat = ctx.get("sat") or {}
    orpi = ctx.get("orpi") or {}
    orpi_error = ctx.get("orpi_error")
    features = orpi.get("features") or {}
    components = orpi.get("components") or {}
    sub = orpi.get("subcomponents") or {}
    scenario = orpi.get("scenario") or {}
    model = orpi.get("model") or {}

    score = _safe_float(orpi.get("orpi_score"))
    percentile = _safe_float(orpi.get("percentile"))
    confidence_score = _safe_float(orpi.get("confidence_score"))
    confidence_meta = orpi.get("confidence") or {}
    rating = orpi.get("rating") or "N/A"
    stance = orpi.get("underwriting_stance") or "No underwriting guidance available"
    if not orpi and orpi_error:
        stance = f"No ORPI score available for this orbit. Reason: {orpi_error}"

    generated_at = ctx.get("generated_at") or datetime.now(timezone.utc)
    generated_at = generated_at.astimezone(timezone.utc)

    # Page background
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    # Header
    top = H - margin
    c.setFont("Helvetica-Bold", 15)
    c.setFillColorRGB(0.08, 0.08, 0.08)
    c.drawString(margin, top, "ORPI Underwriting Brief")

    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.34, 0.34, 0.34)
    c.drawRightString(W - margin, top + 1, generated_at.strftime("Generated %Y-%m-%d %H:%M UTC"))

    c.setFont("Helvetica-Bold", 11)
    c.setFillColorRGB(0.12, 0.12, 0.12)
    c.drawString(margin, top - 20, f"Satellite: {sat.get('name', 'UNKNOWN')} (NORAD {sat.get('id', 'N/A')})")

    c.setStrokeColorRGB(0.78, 0.78, 0.78)
    c.line(margin, top - 28, W - margin, top - 28)

    # Summary cards
    y_cards = top - 178
    card_h = 136
    left_w = 200
    right_w = W - (margin * 2) - left_w - 12

    c.setFillColorRGB(0.98, 0.98, 0.98)
    c.setStrokeColorRGB(0.82, 0.82, 0.82)
    c.rect(margin, y_cards, left_w, card_h, stroke=1, fill=1)

    c.setFont("Helvetica-Bold", 10)
    c.setFillColorRGB(0.24, 0.24, 0.24)
    c.drawString(margin + 10, y_cards + card_h - 16, "ORPI SCORE")

    score_text = "N/A" if score is None else str(int(round(score)))
    c.setFont("Helvetica-Bold", 38)
    c.setFillColorRGB(0.08, 0.08, 0.08)
    c.drawString(margin + 10, y_cards + 62, score_text)

    # Risk badge
    rc = _risk_color(rating)
    c.setFillColorRGB(rc[0], rc[1], rc[2])
    c.roundRect(margin + 96, y_cards + 72, 92, 20, 4, stroke=0, fill=1)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(margin + 142, y_cards + 78, rating)

    c.setFillColorRGB(0.20, 0.20, 0.20)
    c.setFont("Helvetica", 10)
    c.drawString(margin + 10, y_cards + 40, f"Percentile: P{int(round(percentile)) if percentile is not None else 'N/A'}")
    c.drawString(margin + 10, y_cards + 28, f"Confidence: {_fmt(confidence_score, 1)} / 100")

    c.setFont("Helvetica", 9)
    cell = orpi.get("cell") or {}
    c.drawString(
        margin + 10,
        y_cards + 14,
        f"Cell: alt {int(round(_safe_float(cell.get('alt_bin_start'), 0.0)))} km / inc {int(round(_safe_float(cell.get('inc_bin_start'), 0.0)))} deg",
    )

    c.setFillColorRGB(0.98, 0.98, 0.98)
    c.setStrokeColorRGB(0.82, 0.82, 0.82)
    c.rect(margin + left_w + 12, y_cards, right_w, card_h, stroke=1, fill=1)

    sx = margin + left_w + 22
    sy = y_cards + card_h - 16
    c.setFillColorRGB(0.24, 0.24, 0.24)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(sx, sy, "ORBIT SNAPSHOT")

    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.12, 0.12, 0.12)

    lines = []
    lines.append(f"Inclination: {_fmt(sat.get('inc_deg'), 2)} deg")
    lines.append(f"RAAN: {_fmt(sat.get('raan_deg'), 2)} deg")
    lines.append(f"Period: {_fmt(sat.get('period_minutes'), 2)} min")
    lines.append(f"Mean motion: {_fmt(sat.get('mean_motion_rev_per_day'), 6)} rev/day")

    tle_epoch = sat.get("tle_epoch_utc")
    if isinstance(tle_epoch, datetime):
        lines.append(f"TLE epoch: {tle_epoch.strftime('%Y-%m-%d %H:%M UTC')}")
    else:
        lines.append("TLE epoch: N/A")

    age = _safe_float(sat.get("tle_age_days"))
    lines.append(f"TLE age: {_fmt(age, 1)} days")

    yy = sy - 16
    for line in lines:
        c.drawString(sx, yy, line)
        yy -= 14

    # Underwriting recommendation block
    y_rec = y_cards - 86
    c.setFillColorRGB(0.985, 0.985, 0.985)
    c.setStrokeColorRGB(0.82, 0.82, 0.82)
    c.rect(margin, y_rec, W - 2 * margin, 72, stroke=1, fill=1)
    c.setFillColorRGB(0.16, 0.16, 0.16)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin + 10, y_rec + 54, "UNDERWRITING STANCE")
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(0.14, 0.14, 0.14)
    _draw_wrapped_text(c, stance, margin + 10, y_rec + 38, W - 2 * margin - 20, font="Helvetica", size=9, leading=12)

    # Component bars
    y_comp = y_rec - 98
    c.setFillColorRGB(0.985, 0.985, 0.985)
    c.setStrokeColorRGB(0.82, 0.82, 0.82)
    c.rect(margin, y_comp, W - 2 * margin, 88, stroke=1, fill=1)
    c.setFillColorRGB(0.16, 0.16, 0.16)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(margin + 10, y_comp + 72, "COMPONENT PERCENTILES")

    bar_x = margin + 10
    bar_w = W - 2 * margin - 20
    p_pct = (components.get("pressure") or {}).get("percentile", 0)
    v_pct = (components.get("volatility") or {}).get("percentile", 0)
    g_pct = (components.get("growth") or {}).get("percentile", 0)

    _draw_percentile_bar(c, bar_x, y_comp + 52, bar_w, 8, p_pct, (0.07, 0.45, 0.77), "Pressure block", f"P{int(round(_safe_float(p_pct, 0.0)))}")
    _draw_percentile_bar(c, bar_x, y_comp + 32, bar_w, 8, v_pct, (0.52, 0.26, 0.75), "Volatility block", f"P{int(round(_safe_float(v_pct, 0.0)))}")
    _draw_percentile_bar(c, bar_x, y_comp + 12, bar_w, 8, g_pct, (0.17, 0.60, 0.30), "Growth block", f"P{int(round(_safe_float(g_pct, 0.0)))}")

    # Altitude chart
    y_chart = y_comp - 198
    chart_h = 188
    c.setFillColorRGB(0.985, 0.985, 0.985)
    c.setStrokeColorRGB(0.82, 0.82, 0.82)
    c.rect(margin, y_chart, W - 2 * margin, chart_h, stroke=1, fill=1)

    orbit = ctx.get("orbit") or {}
    alts = orbit.get("altitudes_km") or []
    _draw_altitude_chart(c, alts, sat.get("period_minutes"), margin + 8, y_chart + 8, W - 2 * margin - 16, chart_h - 16)

    # Bottom metrics and method snapshot
    y_bottom = margin
    bottom_h = y_chart - y_bottom - 10
    c.setFillColorRGB(0.985, 0.985, 0.985)
    c.setStrokeColorRGB(0.82, 0.82, 0.82)
    c.rect(margin, y_bottom, W - 2 * margin, bottom_h, stroke=1, fill=1)

    c.setFont("Helvetica-Bold", 10)
    c.setFillColorRGB(0.16, 0.16, 0.16)
    c.drawString(margin + 10, y_bottom + bottom_h - 14, "METRICS AND METHOD SNAPSHOT")

    fx = margin + 10
    fy = y_bottom + bottom_h - 28
    c.setFont("Helvetica", 8.6)
    c.setFillColorRGB(0.12, 0.12, 0.12)

    metrics_lines = [
        f"N_eff (effective occupancy): {_fmt(features.get('n_eff_sum'), 3)}",
        f"Vrel proxy (km/s): {_fmt(features.get('vrel_mean_proxy_km_s'), 3)}",
        f"Pressure mean: {_fmt(features.get('pressure_mean'), 3)}",
        f"Sigma: {_fmt(features.get('risk_sigma'), 3)}",
        f"Volatility ratio (sigma/pressure): {_fmt(features.get('volatility_ratio'), 6)}",
        f"Scenario total delta: {_fmt(features.get('trend_total'), 3)}",
        f"Scenario annual delta: {_fmt(features.get('trend_annual'), 3)}",
        f"Confidence score: {_fmt(confidence_score, 1)} / 100",
    ]
    for line in metrics_lines:
        c.drawString(fx, fy, line)
        fy -= 11

    sx2 = margin + 285
    sy2 = y_bottom + bottom_h - 28
    c.setFont("Helvetica", 8.6)

    model_name = model.get("version", "ORPI_v1")
    scenario_name = scenario.get("name") or "N/A"
    scenario_target = scenario.get("target_date") or "N/A"
    scenario_years = _fmt(scenario.get("years_to_target"), 2)

    method_lines = [
        f"Model: {model_name}",
        f"Architecture: percentile ensemble (cell-based)",
        f"Pressure internals: exposure + congestion + geometry",
        f"Scenario: {scenario_name}",
        f"Scenario target date: {scenario_target}",
        f"Annualization horizon: {scenario_years} years",
    ]
    if confidence_meta:
        c_f = confidence_meta.get("freshness") or {}
        c_c = confidence_meta.get("coverage") or {}
        c_s = confidence_meta.get("stability") or {}
        method_lines.extend(
            [
                f"Confidence freshness: {_fmt(c_f.get('score'), 1)} (p90 TLE age={_fmt(c_f.get('p90_tle_age_days'), 2)}d)",
                f"Confidence coverage: {_fmt(c_c.get('score'), 1)} (sample_sum={int(c_c.get('sample_count_sum') or 0)})",
                f"Confidence stability: {_fmt(c_s.get('score'), 1)} (range={_fmt(c_s.get('orpi_range'), 2)})",
            ]
        )

    psub = sub.get("pressure") or {}
    ex_pct = ((psub.get("exposure") or {}).get("percentile"))
    co_pct = ((psub.get("congestion") or {}).get("percentile"))
    ge_pct = ((psub.get("geometry") or {}).get("percentile"))
    method_lines.append(
        f"Pressure internals percentiles: exp P{int(round(_safe_float(ex_pct, 0.0)))} / cong P{int(round(_safe_float(co_pct, 0.0)))} / geom P{int(round(_safe_float(ge_pct, 0.0)))}"
    )

    for line in method_lines:
        c.drawString(sx2, sy2, line)
        sy2 -= 11

    # Footer limitation note
    c.setFont("Helvetica", 7.7)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.drawString(
        margin + 10,
        y_bottom + 8,
        "Use as a comparative underwriting index. Not an event-level collision prediction. No maneuver/covariance modeling.",
    )

    c.showPage()
    c.save()
    return buf.getvalue()
