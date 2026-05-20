"""Live MJPEG preview of the robot head color + depth streams.

Pulls frames from the GDK in a background thread and serves them over HTTP
as a pair of MJPEG endpoints, plus a small landing page that shows both
side-by-side. Browse to ``http://localhost:8080`` from any browser on the
host (host networking is on in the compose file).

Run from inside the container:

    cd /workspace
    python3 tools/preview.py            # listens on 0.0.0.0:8080

Stop with Ctrl-C. The preview holds GDK exclusively while running — Ctrl-C
it before launching the recorder (two GDK clients in one container don't
share well).

Endpoints:

    /                landing page (color + depth side by side)
    /stream/color    multipart MJPEG of the head color camera
    /stream/depth    multipart MJPEG of head depth, colorised 0..6 m
    /healthz         tiny JSON with frame ages
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2  # type: ignore
import numpy as np
from flask import Flask, Response, jsonify


# Match the recorder's clip range so visual scale lines up with the saved
# .npy files. Anything outside this band is shown saturated.
DEPTH_VIS_MIN_M = 0.3
DEPTH_VIS_MAX_M = 6.0


@dataclass
class FrameBuffer:
    """Latest (color, depth) pair shared between the GDK thread and Flask.

    A single mutex around the whole struct is fine at 30 fps — the critical
    section is just a couple of attribute writes.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    color: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    color_t_monotonic: float = 0.0
    depth_t_monotonic: float = 0.0
    stop: bool = False


_color_format_logged = False


def color_to_bgr(img) -> np.ndarray:
    """Convert a GDK ``Image`` to BGR uint8 HxWx3.

    GDK exposes different pixel formats per camera; we've seen NV12 / YUV
    from the head color stream on G2. This function inspects shape +
    ``color_format`` / ``encoding`` and converts to BGR so OpenCV /
    cv2.imencode / downstream code can treat it as a normal BGR frame.

    The first time it runs it logs what it saw — keeps the diagnostic
    surface small but loud enough to debug "black frames" issues.
    """

    global _color_format_logged
    arr = np.asarray(img.data)
    fmt = str(getattr(img, "color_format", "")).lower()
    enc = str(getattr(img, "encoding", "")).lower()

    if not _color_format_logged:
        _color_format_logged = True
        print(
            f"[gdk] color: shape={arr.shape} dtype={arr.dtype} "
            f"encoding={enc!r} color_format={fmt!r}",
            flush=True,
        )

    # Already 3-channel HWC.
    if arr.ndim == 3 and arr.shape[2] == 3:
        if "rgb" in fmt or "rgb" in enc:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr  # assume BGR

    # Single-channel 2D: most likely NV12 (height = H*3/2, width = W).
    if arr.ndim == 2:
        h, w = arr.shape
        if "nv12" in fmt or "nv12" in enc or (h % 3 == 0 and (h // 3) * 2 + h // 3 == h):
            return cv2.cvtColor(arr, cv2.COLOR_YUV2BGR_NV12)
        # Some cameras output YUYV (packed) with shape (H, W*2) uint8.
        if "yuyv" in fmt or "yuyv" in enc:
            return cv2.cvtColor(arr.reshape(img.height, img.width, 2), cv2.COLOR_YUV2BGR_YUYV)
        # Last resort: treat as grayscale.
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    # Anything else — return zeros at the camera resolution so it's obvious
    # something is wrong rather than crashing the stream.
    print(
        f"[gdk] WARN: unsupported color shape {arr.shape}; emitting black frame",
        file=sys.stderr,
    )
    return np.zeros((img.height, img.width, 3), dtype=np.uint8)


def colorize_depth(depth_m: np.ndarray) -> np.ndarray:
    """Float32 depth in metres → uint8 BGR with a colormap.

    NaN / out-of-band pixels show as black so they're visually distinct
    from valid returns.
    """

    d = depth_m.copy()
    mask_invalid = ~np.isfinite(d) | (d <= 0)
    d = np.where(mask_invalid, DEPTH_VIS_MAX_M, d)
    d = np.clip(d, DEPTH_VIS_MIN_M, DEPTH_VIS_MAX_M)
    norm = ((d - DEPTH_VIS_MIN_M) / (DEPTH_VIS_MAX_M - DEPTH_VIS_MIN_M) * 255).astype(np.uint8)
    color = cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)  # near=red, far=blue
    color[mask_invalid] = (0, 0, 0)
    return color


def gdk_worker(buf: FrameBuffer, rate_hz: float = 30.0) -> None:
    """Background thread: GDK init, then loop pulling frames into ``buf``.

    Any GDK exception is fatal — the thread sets buf.stop and exits so the
    Flask side can return a meaningful error instead of stale frames.
    """

    import agibot_gdk as g  # type: ignore

    print("[gdk] gdk_init...", flush=True)
    if g.gdk_init() != g.GDKRes.kSuccess:
        print("[gdk] gdk_init failed", file=sys.stderr)
        buf.stop = True
        return

    cam = None
    try:
        cam = g.Camera()
        time.sleep(3)  # camera DDS settle
        period = 1.0 / rate_hz
        color_t = g.CameraType.kHeadColor
        depth_t = g.CameraType.kHeadDepth
        print("[gdk] streaming", flush=True)

        next_tick = time.monotonic()
        while not buf.stop:
            color = cam.get_latest_image(color_t, 200.0)
            depth = cam.get_latest_image(depth_t, 200.0)
            now = time.monotonic()

            if color is not None:
                bgr = color_to_bgr(color)
                with buf.lock:
                    buf.color = bgr
                    buf.color_t_monotonic = now
            if depth is not None:
                d = np.asarray(depth.data, dtype=np.float32)
                if d.max() > 50:  # mm → m heuristic, same as the recorder
                    d = d / 1000.0
                d[d <= 0] = np.nan
                with buf.lock:
                    buf.depth = d
                    buf.depth_t_monotonic = now

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
    except Exception as e:
        print(f"[gdk] worker crashed: {e}", file=sys.stderr)
    finally:
        try:
            if cam is not None:
                cam.close_camera()
        except Exception:
            pass
        try:
            g.gdk_release()
        except Exception:
            pass
        buf.stop = True
        print("[gdk] released", flush=True)


def mjpeg_generator(buf: FrameBuffer, kind: str, rate_hz: float = 30.0):
    """Yields multipart JPEG chunks until the client disconnects."""

    boundary = b"--frame"
    period = 1.0 / rate_hz
    while not buf.stop:
        with buf.lock:
            frame = buf.color if kind == "color" else buf.depth
            frame = None if frame is None else frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        vis = frame if kind == "color" else colorize_depth(frame)
        ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            time.sleep(period)
            continue

        yield (
            boundary
            + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpg)).encode()
            + b"\r\n\r\n"
            + jpg.tobytes()
            + b"\r\n"
        )
        time.sleep(period)


def make_app(buf: FrameBuffer, rate_hz: float) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index():
        return (
            "<html><head><title>safety_backend preview</title>"
            "<style>body{font-family:sans-serif;background:#111;color:#eee;"
            "margin:0;padding:1em}h2{margin:0.2em 0}"
            "img{max-width:48vw;border:1px solid #444}"
            ".row{display:flex;gap:1em;flex-wrap:wrap}"
            "</style></head><body>"
            "<h1>safety_backend live preview</h1>"
            "<div class=row>"
            "<div><h2>head_color</h2>"
            '<img src="/stream/color"></div>'
            "<div><h2>head_depth (0.3 m red → 6.0 m blue)</h2>"
            '<img src="/stream/depth"></div>'
            "</div></body></html>"
        )

    @app.route("/stream/color")
    def stream_color():
        return Response(
            mjpeg_generator(buf, "color", rate_hz),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/stream/depth")
    def stream_depth():
        return Response(
            mjpeg_generator(buf, "depth", rate_hz),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/healthz")
    def healthz():
        now = time.monotonic()
        with buf.lock:
            color_age = None if buf.color is None else round(now - buf.color_t_monotonic, 3)
            depth_age = None if buf.depth is None else round(now - buf.depth_t_monotonic, 3)
        return jsonify(
            ok=not buf.stop and color_age is not None and depth_age is not None,
            color_age_s=color_age,
            depth_age_s=depth_age,
            stopped=buf.stop,
        )

    return app


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Live MJPEG preview of head color+depth.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--rate", type=float, default=30.0, help="capture rate (Hz)")
    args = p.parse_args(argv)

    buf = FrameBuffer()
    worker = threading.Thread(target=gdk_worker, args=(buf, args.rate), daemon=True)
    worker.start()

    app = make_app(buf, args.rate)
    try:
        # threaded=True so MJPEG streams don't block other requests.
        app.run(host=args.host, port=args.port, threaded=True, debug=False)
    finally:
        buf.stop = True
        worker.join(timeout=5.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
