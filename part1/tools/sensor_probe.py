"""Quick probe: verify LiDAR and fisheye data from GDK and validate the
projection pipeline end-to-end.

Run inside the Docker container (LiDAR + TF only):
    python3 tools/sensor_probe.py

Run fisheye probe after activating develop mode from the HOST first:
    ./scripts/activate_fisheye.sh          # host terminal
    docker compose run --rm safety_backend python3 tools/sensor_probe.py --fisheye

Output: per-sensor summary with data shape, value ranges, and a sample
        LiDAR→fisheye pixel projection to confirm the math is sane.
"""

from __future__ import annotations

import argparse
import sys
import time
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def _banner(title: str) -> None:
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def probe_lidar(lidar, g) -> np.ndarray | None:
    """Pull one LiDAR scan and report statistics."""
    _banner("LiDAR front")
    try:
        pc = lidar.get_latest_pointcloud(g.LidarType.kLidarFront, 2000.0)
        if pc is None:
            print("  ✗ get_latest_pointcloud returned None")
            return None
        print(f"  point_step    : {pc.point_step}")
        print(f"  raw bytes     : {len(np.asarray(pc.data)):,}")
        print(f"  fields        : {[(f.name, f.offset, f.datatype) for f in pc.fields]}")

        from safety_perception import _parse_pointcloud
        xyz = _parse_pointcloud(pc)
        if xyz is None or len(xyz) == 0:
            print(f"  ✗ _parse_pointcloud returned empty")
            return None

        print(f"  Points parsed : {len(xyz):,}")
        print(f"  X range       : {xyz[:,0].min():.2f} … {xyz[:,0].max():.2f} m")
        print(f"  Y range       : {xyz[:,1].min():.2f} … {xyz[:,1].max():.2f} m")
        print(f"  Z range       : {xyz[:,2].min():.2f} … {xyz[:,2].max():.2f} m")
        dists = np.linalg.norm(xyz, axis=1)
        print(f"  Dist min/med/max: {dists.min():.2f} / {np.median(dists):.2f} / {dists.max():.2f} m")
        print(f"  ✓ LiDAR OK")
        return xyz
    except Exception as e:
        print(f"  ✗ LiDAR error: {e}")
        import traceback; traceback.print_exc()
        return None


def probe_tf(tf, g) -> dict:
    """Dump all TF frames and check for livox_front + head_link3."""
    _banner("TF frames")
    frames = {}
    try:
        all_tf = tf.get_all_tf_from_base_link()
        for ts in all_tf:
            fid = ts.child_frame_id
            t = ts.transform.translation
            frames[fid] = np.array([t.x, t.y, t.z])
            print(f"  {fid:<35}  t=[{t.x:+.3f} {t.y:+.3f} {t.z:+.3f}]")
        for key in ("livox_front", "head_link3"):
            status = "✓" if key in frames else "✗ MISSING"
            print(f"  {status} {key}")
    except Exception as e:
        print(f"  ✗ TF error: {e}")
    return frames


def probe_fisheye(cam, tf_mod, g, xyz_lidar: np.ndarray | None) -> None:
    """Pull fisheye frames, report shape/encoding, and test LiDAR projection."""
    from safety_perception import (
        _color_to_bgr, _get_fisheye_K_D, _get_extrinsic_quat_trans,
        _get_head_link3_tf, _project_lidar_to_fisheye_pixels,
        _FISHEYE_CAMERA_ATTR, _FISHEYE_EXTRINSIC_ATTR,
    )
    from scipy.spatial.transform import Rotation

    cam_names = ["left", "right", "back"]

    for name in cam_names:
        _banner(f"Fisheye {name}")
        cam_attr = _FISHEYE_CAMERA_ATTR.get(name)
        ext_attr = _FISHEYE_EXTRINSIC_ATTR.get(name)

        try:
            cam_type = getattr(g.CameraType, cam_attr)
        except AttributeError:
            print(f"  ✗ CameraType.{cam_attr} not in GDK enum")
            continue

        try:
            img = cam.get_latest_image(cam_type, 2000.0)
            if img is None:
                print(f"  ✗ get_latest_image returned None")
                continue
            bgr = _color_to_bgr(img)
            print(f"  Resolution    : {img.width}×{img.height}")
            print(f"  Encoding      : {getattr(img, 'encoding', '?')}")
            print(f"  BGr shape     : {bgr.shape}  dtype={bgr.dtype}")
            print(f"  Pixel mean    : {bgr.mean():.1f}  (0=black, 255=white)")
        except Exception as e:
            print(f"  ✗ Image error: {e}")
            continue

        # Intrinsics
        try:
            K, D, w, h = _get_fisheye_K_D(cam, cam_type)
            print(f"  K (fx,fy,cx,cy): {K[0,0]:.1f} {K[1,1]:.1f} {K[0,2]:.1f} {K[1,2]:.1f}")
            print(f"  D distortion  : {D.flatten()}")
        except Exception as e:
            print(f"  ✗ Intrinsics error: {e}")
            K, D, w, h = None, None, img.width, img.height

        # Extrinsic
        rot_c2h = trans_c2h = None
        if ext_attr:
            try:
                ext_type = getattr(g.SensorExtrinsicType, ext_attr)
                rot_c2h, trans_c2h = _get_extrinsic_quat_trans(tf_mod, ext_type)
                if rot_c2h is not None:
                    print(f"  Extrinsic t   : {trans_c2h}")
                    print(f"  Extrinsic R   : {rot_c2h.as_euler('xyz', degrees=True)} deg")
                else:
                    print(f"  ✗ Extrinsic unavailable (SensorExtrinsicType.{ext_attr})")
            except AttributeError:
                print(f"  ✗ SensorExtrinsicType.{ext_attr} not in GDK enum")

        # Test LiDAR→fisheye projection
        if xyz_lidar is not None and K is not None and rot_c2h is not None:
            rot_h2b, trans_h2b = _get_head_link3_tf(tf_mod)
            if rot_h2b is not None:
                # Transform LiDAR to base_link first (using identity if no livox TF)
                uvz = _project_lidar_to_fisheye_pixels(
                    xyz_lidar, K, D, rot_c2h, trans_c2h,
                    rot_h2b, trans_h2b, w, h,
                )
                total = len(xyz_lidar)
                in_img = len(uvz)
                print(f"  LiDAR→pixel   : {in_img:,}/{total:,} points project into image")
                if in_img > 0:
                    print(f"  Z range (cam) : {uvz[:,2].min():.2f} … {uvz[:,2].max():.2f} m")
                    print(f"  ✓ Projection OK")
                else:
                    print(f"  ✗ No LiDAR points project into image — check TF/extrinsic")
            else:
                print(f"  ✗ head_link3 TF unavailable — skipping projection test")
        elif K is None:
            print(f"  ✗ Skipping projection (no intrinsics)")
        elif xyz_lidar is None:
            print(f"  ✗ Skipping projection (no LiDAR data)")
        elif rot_c2h is None:
            print(f"  ✗ Skipping projection (no extrinsic)")

        print(f"  ✓ Fisheye {name} OK")


def main() -> int:
    p = argparse.ArgumentParser(description="GDK sensor probe — LiDAR + fisheye")
    p.add_argument("--fisheye", action="store_true",
                   help="Probe fisheye cameras (assumes robot already in develop mode "
                        "OR use with --activate to SSH-switch mode from the container)")
    p.add_argument("--activate", action="store_true",
                   help="SSH to robot to activate develop mode (requires SSH access)")
    p.add_argument("--robot-ip", default="10.42.1.101")
    args = p.parse_args()

    import agibot_gdk as g  # type: ignore

    print("GDK init...", flush=True)
    assert g.gdk_init() == g.GDKRes.kSuccess, "gdk_init failed"

    cam = tf = lidar = None
    fisheye_activated = False
    try:
        cam   = g.Camera()
        tf    = g.TF()
        lidar = g.Lidar()
        print("Waiting for DDS settle (4 s)...", flush=True)
        time.sleep(4)

        # Activate fisheye if requested.
        if args.fisheye and args.activate:
            from safety_perception import _activate_fisheye
            fisheye_activated = _activate_fisheye(cam, args.robot_ip)
        elif args.fisheye:
            print("[fisheye] --activate not set; assuming robot is already in develop mode")
            fisheye_activated = True

        xyz_lidar = probe_lidar(lidar, g)
        frames    = probe_tf(tf, g)

        # If we have the livox_front TF, transform LiDAR points to base_link.
        xyz_base = None
        if xyz_lidar is not None:
            _banner("LiDAR → base_link TF")
            from safety_perception import _get_extrinsic_quat_trans
            rot_l2b, trans_l2b = _get_extrinsic_quat_trans(
                tf, g.SensorExtrinsicType.kChassisFrontLidarToBaseLink
            )
            if rot_l2b is not None:
                xyz_base = rot_l2b.apply(xyz_lidar) + trans_l2b
                print(f"  Extrinsic t   : {trans_l2b}")
                print(f"  Sample raw    : {xyz_lidar[0]}")
                print(f"  Sample base   : {xyz_base[0]}")
                print(f"  ✓ LiDAR TF OK")
            else:
                print(f"  ✗ kChassisFrontLidarToBaseLink unavailable — using raw frame")
                xyz_base = xyz_lidar

        if args.fisheye:
            probe_fisheye(cam, tf, g, xyz_base)
        else:
            _banner("Fisheye")
            print("  Skipped (run with --fisheye to activate)")

        _banner("Summary")
        print(f"  LiDAR    : {'✓' if xyz_lidar is not None else '✗'}")
        print(f"  TF       : {'✓' if 'livox_front' in frames and 'head_link3' in frames else '✗ missing frames'}")
        print(f"  Fisheye  : {'✓ activated' if fisheye_activated else 'skipped (--fisheye to test)'}")
        print()

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
        if fisheye_activated:
            from safety_perception import _deactivate_fisheye
            _deactivate_fisheye(args.robot_ip)

    return 0


if __name__ == "__main__":
    sys.exit(main())
