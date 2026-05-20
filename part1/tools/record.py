"""Record a short clip of head color + depth (+ intrinsics + TF) for offline
testing of the safety backend.

Pulls frames from the live Agibot GDK and writes them to disk in a layout the
replay tool consumes. Requires the robot to be reachable (wired ethernet
preferred; see docs/robot-connection.md).

On-disk layout:

    <out>/
    ├── metadata.json       static: intrinsics, TF base_link\u2192head_link3, sensor list
    ├── timestamps.csv      per-frame: index, t_s, color_file, depth_file
    ├── color/000000.png    BGR uint8, lossless PNG
    ├── color/000001.png
    ├── depth/000000.npy    float32 metres (NaN where invalid)
    └── depth/000001.npy

Example:

    python3 tools/record.py --name walk_in_out_1 --duration 30
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


# ── Frame + metadata types ───────────────────────────────────────────────────


@dataclass
class Frame:
    """One synchronized (color, depth) sample with a capture timestamp.

    * ``color`` is BGR uint8 (matches OpenCV convention so cv2.imwrite stores
      colors correctly).
    * ``depth`` is float32 in metres. NaN marks invalid pixels.
    * ``t_s`` is monotonic seconds since the recording started.
    """

    t_s: float
    color: np.ndarray
    depth: np.ndarray


@dataclass
class RecordingMetadata:
    """Static (non-per-frame) info written once to ``metadata.json``."""

    started_at_iso: str
    duration_s_requested: float
    rate_hz: float
    color_shape_hwc: Tuple[int, int, int]
    depth_shape_hw: Tuple[int, int]
    color_intrinsic_fxfycxcy: Optional[Tuple[float, float, float, float]] = None
    color_distortion: Optional[List[float]] = None
    depth_intrinsic_fxfycxcy: Optional[Tuple[float, float, float, float]] = None
    depth_distortion: Optional[List[float]] = None
    base_link_to_camera_xyzqxqyqzqw: Optional[Tuple[float, ...]] = None
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            d[k] = list(v) if isinstance(v, tuple) else v
        return json.dumps(d, indent=2)


# ── GDK frame source ─────────────────────────────────────────────────────────


class GdkFrameSource:
    """Pulls head color + head depth from the live robot via Agibot GDK.

    Lazy-imports ``agibot_gdk`` so the recorder remains importable on hosts
    where GDK is not installed (e.g. CI). All robot calls are wrapped in
    ``try``/``finally`` so a crash mid-recording still releases the camera
    and the GDK runtime.
    """

    def __init__(self) -> None:
        import agibot_gdk  # type: ignore

        self._gdk = agibot_gdk
        self._initialized = False
        self._camera = None
        self._tf = None

        ok = self._gdk.gdk_init()
        if ok != self._gdk.GDKRes.kSuccess:
            raise RuntimeError(f"gdk_init failed: {ok!r}")
        self._initialized = True

        try:
            self._camera = self._gdk.Camera()
            time.sleep(3)  # camera DDS settle (per GDK quickstart)
            self._tf = self._gdk.TF()
            time.sleep(2)  # TF settle
        except Exception:
            self.close()
            raise

    def _intrinsic_tuple(
        self, cam_type
    ) -> Tuple[Tuple[float, float, float, float], List[float]]:
        intr = self._camera.get_camera_intrinsic(cam_type)
        fx, fy, cx, cy = intr.intrinsic
        return (fx, fy, cx, cy), list(intr.distortion)

    def _base_link_to_camera(self) -> Optional[Tuple[float, ...]]:
        """Lookup base_link \u2192 head_link3 via TF, returned as (x,y,z,qx,qy,qz,qw)."""

        try:
            for ts in self._tf.get_all_tf_from_base_link():
                if ts.child_frame_id == "head_link3":
                    t = ts.transform.translation
                    r = ts.transform.rotation
                    return (t.x, t.y, t.z, r.x, r.y, r.z, r.w)
        except Exception as e:
            print(f"warn: TF lookup failed: {e}", file=sys.stderr)
        return None

    def metadata(self) -> RecordingMetadata:
        color_cam = self._gdk.CameraType.kHeadColor
        depth_cam = self._gdk.CameraType.kHeadDepth

        # One frame to learn the resolution.
        color_probe = self._camera.get_latest_image(color_cam, 2000.0)
        depth_probe = self._camera.get_latest_image(depth_cam, 2000.0)
        if color_probe is None or depth_probe is None:
            raise RuntimeError(
                "couldn't fetch a probe frame from the robot. "
                "Is the wired ethernet plugged in and the robot reachable?"
            )

        ch, cw = color_probe.height, color_probe.width
        dh, dw = depth_probe.height, depth_probe.width

        color_intr, color_dist = self._intrinsic_tuple(color_cam)
        depth_intr, depth_dist = self._intrinsic_tuple(depth_cam)

        return RecordingMetadata(
            started_at_iso=time.strftime("%Y-%m-%dT%H:%M:%S"),
            duration_s_requested=0.0,  # filled in by run_recording
            rate_hz=0.0,
            color_shape_hwc=(ch, cw, 3),
            depth_shape_hw=(dh, dw),
            color_intrinsic_fxfycxcy=color_intr,
            color_distortion=color_dist,
            depth_intrinsic_fxfycxcy=depth_intr,
            depth_distortion=depth_dist,
            base_link_to_camera_xyzqxqyqzqw=self._base_link_to_camera(),
            notes={"color_type": "kHeadColor", "depth_type": "kHeadDepth"},
        )

    def frames(self, duration_s: float, rate_hz: float) -> Iterator[Frame]:
        color_cam = self._gdk.CameraType.kHeadColor
        depth_cam = self._gdk.CameraType.kHeadDepth
        period = 1.0 / rate_hz
        n_frames = int(round(duration_s * rate_hz))

        t0 = time.monotonic()
        for i in range(n_frames):
            # 200 ms is generous: cameras run at 30 fps so a frame should be
            # available within ~33 ms. A miss likely means the cable/DDS is
            # unhealthy — surface that with a warning so the operator can
            # decide whether to keep going.
            color = self._camera.get_latest_image(color_cam, 200.0)
            depth = self._camera.get_latest_image(depth_cam, 200.0)
            if color is None or depth is None:
                print(
                    f"warn: missed frame at i={i} (color={color is not None}, "
                    f"depth={depth is not None})",
                    file=sys.stderr,
                )
                _sleep_until(t0 + (i + 1) * period)
                continue

            color_arr = np.asarray(color.data)
            depth_arr = np.asarray(depth.data, dtype=np.float32)

            # Depth from this camera is millimetres on G2; convert to metres
            # and mark zeros (no return) as NaN. The >50 heuristic is
            # paranoia in case a future firmware switches units.
            if depth_arr.max() > 50:
                depth_arr = depth_arr / 1000.0
            depth_arr[depth_arr <= 0] = np.nan

            yield Frame(t_s=time.monotonic() - t0, color=color_arr, depth=depth_arr)
            _sleep_until(t0 + (i + 1) * period)

    def close(self) -> None:
        try:
            if self._camera is not None:
                self._camera.close_camera()
        except Exception:
            pass
        try:
            if self._initialized:
                self._gdk.gdk_release()
        except Exception:
            pass


def _sleep_until(deadline_monotonic: float) -> None:
    remaining = deadline_monotonic - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)


# ── Recording driver ─────────────────────────────────────────────────────────


def _countdown(seconds: int) -> None:
    """Visible countdown — gives the operator time to walk into position
    after GDK init has completed.
    """

    if seconds <= 0:
        return
    print(
        f"\nGDK ready. Recording starts in {seconds} s — position yourself.",
        flush=True,
    )
    for remaining in range(seconds, 0, -1):
        sys.stdout.write(f"  {remaining}...\n")
        sys.stdout.flush()
        time.sleep(1.0)
    print("  RECORDING NOW", flush=True)


def run_recording(
    source: GdkFrameSource,
    out_dir: Path,
    duration_s: float,
    rate_hz: float,
    countdown_s: int = 0,
) -> int:
    color_dir = out_dir / "color"
    depth_dir = out_dir / "depth"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    import cv2  # lazy so this file imports on hosts without OpenCV

    meta = source.metadata()
    meta.duration_s_requested = duration_s
    meta.rate_hz = rate_hz
    (out_dir / "metadata.json").write_text(meta.to_json())

    _countdown(countdown_s)

    n_written = 0
    print(
        f"recording \u2192 {out_dir}  duration={duration_s:.1f}s  rate={rate_hz:.0f}Hz",
        flush=True,
    )

    with (out_dir / "timestamps.csv").open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["index", "t_s", "color_file", "depth_file"])

        try:
            for i, frame in enumerate(source.frames(duration_s, rate_hz)):
                color_rel = f"color/{i:06d}.png"
                depth_rel = f"depth/{i:06d}.npy"

                cv2.imwrite(str(out_dir / color_rel), frame.color)
                np.save(out_dir / depth_rel, frame.depth.astype(np.float32))

                writer.writerow([i, f"{frame.t_s:.4f}", color_rel, depth_rel])
                n_written += 1
                if i % 30 == 0:
                    print(
                        f"  i={i:4d}  t={frame.t_s:5.2f}s  "
                        f"color={frame.color.shape}  depth={frame.depth.shape}",
                        flush=True,
                    )
        except KeyboardInterrupt:
            print("\ninterrupted — finishing up.", flush=True)
        finally:
            source.close()

    print(f"wrote {n_written} frames to {out_dir}", flush=True)
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record head color+depth (+TF/intrinsics).")
    p.add_argument(
        "--name",
        required=True,
        help="recording name; data goes to part1/data/recordings/<name>/",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="recording duration in seconds (default: 10)",
    )
    p.add_argument(
        "--rate",
        type=float,
        default=30.0,
        help="capture rate in Hz (default: 30)",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/recordings"),
        help="root directory for recordings (default: data/recordings)",
    )
    p.add_argument(
        "--countdown",
        type=int,
        default=10,
        help="seconds to wait after GDK init before recording starts; gives "
        "the operator time to walk into position (default: 10, set 0 to skip)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = args.out_root / args.name
    if out_dir.exists() and any(out_dir.iterdir()):
        print(
            f"refusing to overwrite non-empty {out_dir}. "
            f"Pick a new --name or delete the directory first.",
            file=sys.stderr,
        )
        return 2

    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    source = GdkFrameSource()
    return run_recording(
        source, out_dir, args.duration, args.rate, countdown_s=args.countdown
    )


if __name__ == "__main__":
    sys.exit(main())
