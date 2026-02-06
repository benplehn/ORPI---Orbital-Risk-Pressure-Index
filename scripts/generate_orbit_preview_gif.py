#!/usr/bin/env python3
"""
Generate a README GIF from the real ORPI UI (React + Three.js), not a synthetic render.

Workflow:
1) Open UI page.
2) Search/select a NORAD satellite from dropdown.
3) Optionally collapse side panels for a clear Earth+orbit scene.
4) Capture a sequence of real browser frames.
5) Encode GIF + poster PNG.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image


def _require_playwright():
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency: playwright. Install with `python3 -m pip install playwright` "
            "then `python3 -m playwright install chromium`."
        ) from exc
    return sync_playwright


def _save_gif(frame_paths: list[Path], gif_out: Path, png_out: Path, duration_ms: int) -> None:
    frames = []
    for p in frame_paths:
        with Image.open(p) as im:
            frames.append(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=256))
    if not frames:
        raise RuntimeError("No frames captured, GIF cannot be created.")

    gif_out.parent.mkdir(parents=True, exist_ok=True)
    png_out.parent.mkdir(parents=True, exist_ok=True)

    frames[0].save(
        gif_out,
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=duration_ms,
        loop=0,
        disposal=2,
    )

    mid = max(0, len(frames) // 2)
    frames[mid].convert("RGBA").save(png_out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ORPI UI orbit preview GIF from real UI")
    parser.add_argument("--ui-url", default="http://127.0.0.1:5173")
    parser.add_argument("--norad", type=int, default=25544)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--step-ms", type=int, default=90)
    parser.add_argument("--settle-ms", type=int, default=2200)
    parser.add_argument("--zoom-out-steps", type=int, default=16)
    parser.add_argument("--zoom-out-delta", type=int, default=260)
    parser.add_argument(
        "--hide-panels",
        action="store_true",
        help="Hide side panels before capture (disabled by default).",
    )
    parser.add_argument("--gif-out", default="docs/assets/ui-orbit-preview.gif")
    parser.add_argument("--png-out", default="docs/assets/ui-orbit-preview.png")
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()

    sync_playwright = _require_playwright()

    repo_root = Path(__file__).resolve().parents[1]
    tmp_dir = repo_root / "docs" / "assets" / "_ui_gif_frames"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": int(args.width), "height": int(args.height)},
            device_scale_factor=1,
        )
        page = context.new_page()
        page.goto(args.ui_url, wait_until="networkidle")
        page.wait_for_timeout(1200)

        # 1) Select NORAD via the real UI flow.
        input_box = page.locator("input.input").first
        input_box.click()
        input_box.fill(str(args.norad))
        page.wait_for_timeout(700)

        dd = page.locator(".dropdown button")
        if dd.count() == 0:
            raise RuntimeError(
                "No dropdown result found in UI search. Check backend/API connection and satellite availability."
            )

        preferred = page.locator(".dropdown button", has_text=str(args.norad))
        if preferred.count() > 0:
            preferred.first.click()
        else:
            dd.first.click()

        page.wait_for_timeout(int(args.settle_ms))

        # Ensure satellite has been added to real UI list.
        sat_items = page.locator(".sat-item")
        sat_items.first.wait_for(timeout=20000)

        # 2) Optional: hide side panels.
        qbtn = page.locator(".quickbar .qbtn")
        if args.hide_panels and qbtn.count() >= 2:
            qbtn.nth(0).click()
            page.wait_for_timeout(180)
            qbtn.nth(1).click()
            page.wait_for_timeout(400)

        # 3) Zoom out in the real WebGL canvas so Earth + orbit are visible.
        # Keep panels visible by default for a full-window product preview.
        cx = int(args.width * 0.5)
        cy = int(args.height * 0.52)
        page.mouse.move(cx, cy)
        for _ in range(max(0, int(args.zoom_out_steps))):
            page.mouse.wheel(0, int(args.zoom_out_delta))
            page.wait_for_timeout(45)
        page.wait_for_timeout(300)

        # 4) Capture live UI frames.
        for i in range(max(12, int(args.frames))):
            out = tmp_dir / f"frame_{i:04d}.png"
            page.screenshot(path=str(out))
            frame_paths.append(out)
            page.wait_for_timeout(max(20, int(args.step_ms)))

        context.close()
        browser.close()

    gif_out = (repo_root / args.gif_out).resolve()
    png_out = (repo_root / args.png_out).resolve()
    _save_gif(frame_paths, gif_out, png_out, duration_ms=max(20, int(args.step_ms)))

    if not args.keep_frames and tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print(f"Wrote GIF: {gif_out}")
    print(f"Wrote PNG: {png_out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
