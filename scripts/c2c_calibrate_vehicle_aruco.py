#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c2c_calibrate_vehicle_aruco.py

FAST-Calib vehicle-level camera-to-camera extrinsic calibration helper.

Typical usage:
  rosrun fast_calib c2c_calibrate_vehicle_aruco.py --vehicle 221 -g right --save-debug

Vehicle-level config:
  config/vehicles/<vehicle_id>/vehicle.yaml
  config/vehicles/<vehicle_id>/cameras/*.yaml

The script reads:
  1) vehicle.yaml: which side/group to calibrate, which cameras belong to the group,
     and where the collected image pair directory is.
  2) cameras/*.yaml: all camera intrinsics for this vehicle.

Image data must be collected by collect_two_camera_aruco_single_once.py:
  pair_0001_cam0.jpg
  pair_0001_cam1.jpg
  pair_0002_cam0.jpg
  pair_0002_cam1.jpg

Transform convention:
  T_cam1_cam0 = T_<cam1>_from_<cam0>
  p_cam1 = R_cam1_cam0 * p_cam0 + t_cam1_cam0

For vehicle group right by convention:
  cam0 = rear_right
  cam1 = front_right

For vehicle group left by convention:
  cam0 = rear_left
  cam1 = front_left
"""

import argparse
import csv
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_VEHICLE = "221"
DEFAULT_CONFIG_ROOT = "$(find fast_calib)/config/vehicles"
DEFAULT_DICTIONARY = "DICT_6X6_250"
DEFAULT_MARKER_ID = 1
DEFAULT_MARKER_SIZE_M = 0.80

MAX_PNP_REPROJ_PX = 0.80
MAX_ROT_DEV_DEG = 1.20
MAX_TRANS_DEV_M = 0.20
MAX_CROSS_0_TO_1_MEAN_PX = 0.50
MAX_CROSS_1_TO_0_MEAN_PX = 5.00
MIN_CROSS_FILTER_PAIRS = 3


def resolve_ros_path(path_text):
    """Resolve $(find pkg) and return an absolute local path."""
    text = str(path_text)
    pattern = re.compile(r"\$\(find\s+([^)]+)\)")

    def repl(match):
        pkg = match.group(1).strip()
        try:
            import rospkg
            return rospkg.RosPack().get_path(pkg)
        except Exception:
            here = Path(__file__).resolve()
            candidates = [here.parent] + list(here.parents)
            for parent in candidates:
                if parent.name == pkg and (parent / "package.xml").exists():
                    return str(parent)
                if (parent / "package.xml").exists() and parent.name == pkg:
                    return str(parent)
            raise RuntimeError("Cannot resolve $(find {}). Source your catkin workspace.".format(pkg))

    return str(Path(pattern.sub(repl, text)).expanduser().resolve())


def load_yaml_file(path):
    path = Path(resolve_ros_path(path))
    if not path.exists():
        raise RuntimeError("YAML file does not exist: {}".format(path))
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return path, data if isinstance(data, dict) else {}


def as_abs_path(value, base_dir=None):
    if value is None:
        return None
    text = str(value)
    if "$(find" in text:
        return Path(resolve_ros_path(text))
    p = Path(text).expanduser()
    if p.is_absolute():
        return p.resolve()
    if base_dir is not None:
        return (Path(base_dir) / p).resolve()
    return p.resolve()


def get_nested(d, keys, default=None):
    cur = d
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def normalize_group(group):
    group = str(group).lower()
    if group in ("r", "right", "right_pair"):
        return "right"
    if group in ("l", "left", "left_pair"):
        return "left"
    return group


def load_vehicle_config(args):
    vehicle_id = str(args.vehicle)
    config_root = Path(resolve_ros_path(args.config_root))

    if args.vehicle_config:
        vehicle_yaml = Path(resolve_ros_path(args.vehicle_config))
        vehicle_dir = vehicle_yaml.parent
    else:
        vehicle_dir = config_root / vehicle_id
        vehicle_yaml = vehicle_dir / "vehicle.yaml"

    vehicle_yaml, vehicle_cfg = load_yaml_file(vehicle_yaml)

    if "vehicle_id" not in vehicle_cfg:
        vehicle_cfg["vehicle_id"] = vehicle_id

    return vehicle_id, vehicle_dir, vehicle_yaml, vehicle_cfg


def _as_float_list(value):
    if value is None:
        return None
    if isinstance(value, dict):
        for k in ("data", "D", "K"):
            if k in value:
                return _as_float_list(value[k])
        return None
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return None


def parse_intrinsic_node(node):
    if not isinstance(node, dict):
        return None

    # Many camera yaml files put parameters under intrinsics/camera.
    for key in ("intrinsics", "camera", "camera_intrinsics", "params"):
        if key in node and isinstance(node[key], dict):
            parsed = parse_intrinsic_node(node[key])
            if parsed is not None:
                return parsed

    lower_to_key = {str(k).lower(): k for k in node.keys()}

    if all(k in lower_to_key for k in ("fx", "fy", "cx", "cy")):
        fx = float(node[lower_to_key["fx"]])
        fy = float(node[lower_to_key["fy"]])
        cx = float(node[lower_to_key["cx"]])
        cy = float(node[lower_to_key["cy"]])
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

        def v(name, default=0.0):
            return float(node[lower_to_key[name]]) if name in lower_to_key else float(default)

        D = np.array([v("k1"), v("k2"), v("p1"), v("p2"), v("k3")], dtype=np.float64)
        return K, D

    K_list = None
    for key in ("K", "k", "camera_matrix", "intrinsic_matrix"):
        if key in node:
            K_list = _as_float_list(node[key])
            break

    if K_list is not None and len(K_list) == 9:
        K = np.asarray(K_list, dtype=np.float64).reshape(3, 3)
        D_list = None
        for key in ("D", "d", "distortion_coefficients", "dist_coeffs", "distortion"):
            if key in node:
                D_list = _as_float_list(node[key])
                break
        if D_list is None:
            D_list = [
                float(node.get("k1", 0.0)),
                float(node.get("k2", 0.0)),
                float(node.get("p1", 0.0)),
                float(node.get("p2", 0.0)),
                float(node.get("k3", 0.0)),
            ]
        while len(D_list) < 5:
            D_list.append(0.0)
        return K, np.asarray(D_list[:5], dtype=np.float64)

    return None


def load_vehicle_cameras(vehicle_dir, vehicle_cfg, camera_dir_override=""):
    cam_dir_value = camera_dir_override or vehicle_cfg.get("camera_intrinsics_dir") or vehicle_cfg.get("camera_dir") or "cameras"
    camera_dir = as_abs_path(cam_dir_value, base_dir=vehicle_dir)
    if not camera_dir.exists():
        raise RuntimeError("Camera intrinsics directory does not exist: {}".format(camera_dir))

    cameras = {}
    for path in sorted(camera_dir.glob("*.yaml")) + sorted(camera_dir.glob("*.yml")):
        _, data = load_yaml_file(path)
        parsed = parse_intrinsic_node(data)
        if parsed is None:
            print("[WARN] skip camera yaml without intrinsics:", path)
            continue

        name = data.get("camera_name") or data.get("name") or data.get("frame_id") or path.stem
        K, D = parsed
        cameras[str(name)] = {"K": K, "D": D, "path": path}

    if not cameras:
        raise RuntimeError("No valid camera intrinsics found under {}".format(camera_dir))

    return camera_dir, cameras


def resolve_group_from_vehicle(vehicle_cfg, group_arg, cam0_override="", cam1_override="", allow_disabled=False):
    c2c = vehicle_cfg.get("c2c_calibration") or vehicle_cfg.get("camera_to_camera_calibration") or {}
    if not isinstance(c2c, dict):
        c2c = {}

    default_group = c2c.get("default_group", "right")
    group = normalize_group(group_arg or default_group)

    groups = c2c.get("groups") or vehicle_cfg.get("camera_groups") or {}
    if not isinstance(groups, dict):
        groups = {}

    aliases = {
        "right": ("right", "r", "right_pair"),
        "left": ("left", "l", "left_pair"),
    }.get(group, (group,))

    group_node = None
    for alias in aliases:
        if alias in groups and isinstance(groups[alias], dict):
            group_node = groups[alias]
            break

    if group_node is None:
        # Convention fallback.
        if group == "right":
            group_node = {"enabled": True, "cam0": "rear_right", "cam1": "front_right", "data_dir": "right_pair"}
        elif group == "left":
            group_node = {"enabled": True, "cam0": "rear_left", "cam1": "front_left", "data_dir": "left_pair"}
        else:
            raise RuntimeError("Cannot find group '{}' in vehicle config".format(group))

    enabled = bool(group_node.get("enabled", True))
    if not enabled and not allow_disabled:
        raise RuntimeError(
            "Group '{}' is disabled in vehicle.yaml. Set enabled: true or pass --allow-disabled.".format(group)
        )

    cam0 = cam0_override or group_node.get("cam0") or group_node.get("camera0")
    cam1 = cam1_override or group_node.get("cam1") or group_node.get("camera1")
    if not cam0 or not cam1:
        raise RuntimeError("Group '{}' must define cam0 and cam1".format(group))

    return group, group_node, str(cam0), str(cam1)


def resolve_data_dir(vehicle_dir, vehicle_cfg, group_node, data_dir_override=""):
    if data_dir_override:
        return as_abs_path(data_dir_override)

    data_root = vehicle_cfg.get("data_root") or vehicle_cfg.get("calib_data_root") or ""
    base = as_abs_path(data_root, base_dir=vehicle_dir) if data_root else vehicle_dir

    data_dir_value = (
        group_node.get("data_dir")
        or group_node.get("image_pair_dir")
        or group_node.get("pair_dir")
        or group_node.get("path")
    )
    if not data_dir_value:
        raise RuntimeError("Group config must define data_dir, or pass --data-dir")

    return as_abs_path(data_dir_value, base_dir=base)


def resolve_aruco_params(vehicle_cfg, args):
    c2c = vehicle_cfg.get("c2c_calibration") or vehicle_cfg.get("camera_to_camera_calibration") or {}
    if not isinstance(c2c, dict):
        c2c = {}

    dictionary = args.dictionary or c2c.get("dictionary") or c2c.get("aruco_dictionary") or DEFAULT_DICTIONARY

    marker_id = args.marker_id
    if marker_id is None:
        marker_id = int(c2c.get("marker_id", DEFAULT_MARKER_ID))

    marker_size = args.marker_size
    if marker_size is None:
        marker_size = (
            c2c.get("marker_size_m")
            or c2c.get("marker_size")
            or c2c.get("aruco_marker_size_m")
            or DEFAULT_MARKER_SIZE_M
        )
    return str(dictionary), int(marker_id), float(marker_size)


def make_dictionary(name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV has no cv2.aruco. Install opencv-contrib-python or use OpenCV with aruco.")
    if not hasattr(cv2.aruco, name):
        raise RuntimeError("cv2.aruco has no dictionary: {}".format(name))
    dict_id = getattr(cv2.aruco, name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def make_detector_params():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        p = cv2.aruco.DetectorParameters_create()
    else:
        p = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(p, "cornerRefinementWinSize"):
        p.cornerRefinementWinSize = 5
    if hasattr(p, "cornerRefinementMaxIterations"):
        p.cornerRefinementMaxIterations = 80
    if hasattr(p, "cornerRefinementMinAccuracy"):
        p.cornerRefinementMinAccuracy = 0.003
    return p


def detect_markers(gray, dictionary, params):
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        return detector.detectMarkers(gray)
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def find_pairs(data_dir):
    pat = re.compile(r"pair_(\d+)_cam0\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)
    pairs = []
    for p0 in sorted(Path(data_dir).glob("pair_*_cam0.*")):
        m = pat.match(p0.name)
        if not m:
            continue
        idx = int(m.group(1))
        p1 = None
        for ext in ("jpg", "jpeg", "png", "bmp"):
            cand = Path(data_dir) / "pair_{:04d}_cam1.{}".format(idx, ext)
            if cand.exists():
                p1 = cand
                break
        if p1 is not None:
            pairs.append((idx, p0, p1))
    return pairs


def marker_object_points(marker_size):
    half = float(marker_size) / 2.0
    # ArUco corner order: top-left, top-right, bottom-right, bottom-left.
    return np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)


def project_points(obj_pts, rvec, tvec, K, D):
    proj, _ = cv2.projectPoints(
        np.asarray(obj_pts, dtype=np.float64),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        K,
        D,
    )
    return proj.reshape(-1, 2)


def reprojection_error(obj_pts, img_pts, rvec, tvec, K, D):
    proj = project_points(obj_pts, rvec, tvec, K, D)
    err = np.linalg.norm(proj - np.asarray(img_pts, dtype=np.float64).reshape(-1, 2), axis=1)
    return float(np.mean(err))


def solve_pnp_square(obj_pts, img_pts, K, D):
    img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    best = None

    if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE") and hasattr(cv2, "solvePnPGeneric"):
        try:
            ret = cv2.solvePnPGeneric(obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            ok, rvecs, tvecs = ret[0], ret[1], ret[2]
            if ok:
                for rv, tv in zip(rvecs, tvecs):
                    err = reprojection_error(obj_pts, img_pts, rv, tv, K, D)
                    if best is None or err < best[2]:
                        best = (np.asarray(rv).reshape(3, 1), np.asarray(tv).reshape(3, 1), err)
        except Exception:
            best = None

    if best is None:
        ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP failed")
        best = (
            np.asarray(rvec).reshape(3, 1),
            np.asarray(tvec).reshape(3, 1),
            reprojection_error(obj_pts, img_pts, rvec, tvec, K, D),
        )

    return best


def rt_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def T_to_rt(T):
    rvec, _ = cv2.Rodrigues(np.asarray(T[:3, :3], dtype=np.float64))
    return rvec.reshape(3, 1), np.asarray(T[:3, 3], dtype=np.float64).reshape(3, 1)


def invert_T(T):
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rotmat_to_quat_xyzw(R):
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q


def quat_xyzw_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q /= np.linalg.norm(q)
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def average_quaternions_xyzw(quats):
    A = np.zeros((4, 4), dtype=np.float64)
    ref = np.asarray(quats[0], dtype=np.float64).reshape(4)
    if ref[3] < 0:
        ref = -ref
    for q in quats:
        q = np.asarray(q, dtype=np.float64).reshape(4)
        q /= np.linalg.norm(q)
        if np.dot(q, ref) < 0:
            q = -q
        A += np.outer(q, q)
    vals, vecs = np.linalg.eigh(A / len(quats))
    q = vecs[:, np.argmax(vals)]
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q


def average_T(T_list):
    q = average_quaternions_xyzw([rotmat_to_quat_xyzw(T[:3, :3]) for T in T_list])
    t = np.median(np.array([T[:3, 3] for T in T_list]), axis=0)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rotmat(q)
    T[:3, 3] = t
    return T


def rotation_angle_deg(R):
    val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(val)))


def rpy_from_R_zyx(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def detect_one_image(image_path, dictionary, params, target_id):
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Cannot read image: {}".format(image_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detect_markers(gray, dictionary, params)
    if ids is None:
        return img, None, []
    flat_ids = [int(x) for x in ids.flatten().tolist()]
    for marker_id, corner in zip(flat_ids, corners):
        if marker_id == target_id:
            return img, corner.reshape(4, 2).astype(np.float64), flat_ids
    return img, None, flat_ids


def run_global_optimization(records, T_init, obj_pts, K0, D0, K1, D1, no_global_opt=False):
    if no_global_opt:
        return T_init, None

    try:
        from scipy.optimize import least_squares
    except Exception as e:
        print("[WARN] scipy is not available, skip global optimization: {}".format(e))
        return T_init, None

    rvec_e, tvec_e = T_to_rt(T_init)
    x0 = []
    x0.extend(rvec_e.reshape(3))
    x0.extend(tvec_e.reshape(3))
    for rec in records:
        x0.extend(rec["rvec0"].reshape(3))
        x0.extend(rec["tvec0"].reshape(3))
    x0 = np.asarray(x0, dtype=np.float64)

    def residual(x):
        T_ext = rt_to_T(x[0:3].reshape(3, 1), x[3:6].reshape(3, 1))
        out = []
        off = 6
        for i, rec in enumerate(records):
            rvec0 = x[off + 6*i: off + 6*i + 3].reshape(3, 1)
            tvec0 = x[off + 6*i + 3: off + 6*i + 6].reshape(3, 1)

            proj0 = project_points(obj_pts, rvec0, tvec0, K0, D0)
            out.extend((proj0 - rec["corners0"]).reshape(-1))

            T1 = T_ext @ rt_to_T(rvec0, tvec0)
            rvec1, tvec1 = T_to_rt(T1)
            proj1 = project_points(obj_pts, rvec1, tvec1, K1, D1)
            out.extend((proj1 - rec["corners1"]).reshape(-1))
        return np.asarray(out, dtype=np.float64)

    print("[INFO] global optimization: frames={}, variables={}, residuals={}".format(
        len(records), len(x0), len(residual(x0))
    ))

    res = least_squares(
        residual,
        x0,
        method="trf",
        loss="huber",
        f_scale=2.0,
        max_nfev=300,
        verbose=1,
    )

    T_opt = rt_to_T(res.x[0:3].reshape(3, 1), res.x[3:6].reshape(3, 1))

    off = 6
    for i, rec in enumerate(records):
        rec["rvec0_opt"] = res.x[off + 6*i: off + 6*i + 3].reshape(3, 1)
        rec["tvec0_opt"] = res.x[off + 6*i + 3: off + 6*i + 6].reshape(3, 1)

    return T_opt, res


def compute_cross_errors(records, T_ext, obj_pts, K0, D0, K1, D1, use_opt_pose=False):
    T_inv = invert_T(T_ext)
    for rec in records:
        rvec0 = rec.get("rvec0_opt", rec["rvec0"]) if use_opt_pose else rec["rvec0"]
        tvec0 = rec.get("tvec0_opt", rec["tvec0"]) if use_opt_pose else rec["tvec0"]

        T1_pred = T_ext @ rt_to_T(rvec0, tvec0)
        rvec1_pred, tvec1_pred = T_to_rt(T1_pred)
        proj1 = project_points(obj_pts, rvec1_pred, tvec1_pred, K1, D1)
        e01 = np.linalg.norm(proj1 - rec["corners1"], axis=1)

        T0_pred = T_inv @ rt_to_T(rec["rvec1"], rec["tvec1"])
        rvec0_pred, tvec0_pred = T_to_rt(T0_pred)
        proj0 = project_points(obj_pts, rvec0_pred, tvec0_pred, K0, D0)
        e10 = np.linalg.norm(proj0 - rec["corners0"], axis=1)

        rec["cross_0_to_1_mean_px"] = float(np.mean(e01))
        rec["cross_0_to_1_max_px"] = float(np.max(e01))
        rec["cross_1_to_0_mean_px"] = float(np.mean(e10))
        rec["cross_1_to_0_max_px"] = float(np.max(e10))


def summarize(records, key):
    vals = np.array([r[key] for r in records], dtype=np.float64)
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "max": float(np.max(vals)),
    }


def format_pair_indices(records):
    if not records:
        return "none"
    return ",".join("pair_{:04d}".format(r["idx"]) for r in records)


def format_cross_filter_rejection(r):
    return "pair_{:04d}(0to1={:.3f}px,1to0={:.3f}px)".format(
        r["idx"], r["cross_0_to_1_mean_px"], r["cross_1_to_0_mean_px"]
    )


def threshold_enabled(value):
    return value is not None and float(value) > 0.0


def apply_cross_filter(used, args):
    kept = []
    rejected = []
    for r in used:
        ok = True
        if threshold_enabled(args.max_cross_0_to_1_mean):
            ok = ok and r["cross_0_to_1_mean_px"] <= args.max_cross_0_to_1_mean
        if threshold_enabled(args.max_cross_1_to_0_mean):
            ok = ok and r["cross_1_to_0_mean_px"] <= args.max_cross_1_to_0_mean
        (kept if ok else rejected).append(r)
    return kept, rejected


def draw_overlay(img, detected, projected, title):
    vis = img.copy()
    for i, (d, p) in enumerate(zip(detected, projected)):
        d = tuple(np.round(d).astype(int))
        p = tuple(np.round(p).astype(int))
        cv2.circle(vis, d, 7, (0, 255, 0), -1)
        cv2.circle(vis, p, 7, (0, 0, 255), 2)
        cv2.line(vis, d, p, (0, 255, 255), 2)
        cv2.putText(vis, str(i), (d[0] + 8, d[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(vis, title, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (0, 255, 255), 2)
    return vis


def save_debug(records, T_ext, obj_pts, K1, D1, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rec in records:
        rvec0 = rec.get("rvec0_opt", rec["rvec0"])
        tvec0 = rec.get("tvec0_opt", rec["tvec0"])
        T1 = T_ext @ rt_to_T(rvec0, tvec0)
        rvec1, tvec1 = T_to_rt(T1)
        proj1 = project_points(obj_pts, rvec1, tvec1, K1, D1)
        title = "pair_{:04d}: green=detected red=projected mean={:.2f}px".format(
            rec["idx"], rec["cross_0_to_1_mean_px"]
        )
        vis = draw_overlay(rec["img1"], rec["corners1"], proj1, title)
        cv2.imwrite(str(out_dir / "pair_{:04d}_cam1_cross_reproj.jpg".format(rec["idx"])),
                    vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def sample_marker_plane_points(marker_size, grid_steps):
    steps = max(2, int(grid_steps))
    half = float(marker_size) / 2.0
    xs = np.linspace(-half, half, steps)
    ys = np.linspace(-half, half, steps)
    pts = []
    ij = []
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            pts.append([x, y, 0.0])
            ij.append((xi, yi))
    return np.asarray(pts, dtype=np.float64), ij, steps


def as_int_points(points):
    return np.round(np.asarray(points, dtype=np.float64).reshape(-1, 2)).astype(np.int32)


def point_in_image(pt, width, height, margin=30):
    x, y = float(pt[0]), float(pt[1])
    return -margin <= x < width + margin and -margin <= y < height + margin


def draw_marker_projection_validation(rec, T_ext, obj_pts, marker_size, K1, D1, grid_steps):
    vis = rec["img1"].copy()
    h, w = vis.shape[:2]

    rvec0 = rec.get("rvec0_opt", rec["rvec0"])
    tvec0 = rec.get("tvec0_opt", rec["tvec0"])
    T1 = T_ext @ rt_to_T(rvec0, tvec0)
    rvec1_pred, tvec1_pred = T_to_rt(T1)

    grid_pts, grid_ij, steps = sample_marker_plane_points(marker_size, grid_steps)
    grid_proj = project_points(grid_pts, rvec1_pred, tvec1_pred, K1, D1)
    for p, (xi, yi) in zip(grid_proj, grid_ij):
        if point_in_image(p, w, h):
            hue = int(255.0 * yi / max(1, steps - 1))
            color = cv2.applyColorMap(np.array([[hue]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0].tolist()
            cv2.circle(vis, tuple(np.round(p).astype(int)), 2, color, -1, lineType=cv2.LINE_AA)

    detected = as_int_points(rec["corners1"])
    projected = as_int_points(project_points(obj_pts, rvec1_pred, tvec1_pred, K1, D1))
    cv2.polylines(vis, [detected], True, (0, 255, 0), 3, lineType=cv2.LINE_AA)
    cv2.polylines(vis, [projected], True, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    for i, (d, p) in enumerate(zip(detected, projected)):
        cv2.circle(vis, tuple(d), 7, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, tuple(p), 7, (0, 0, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(vis, tuple(d), tuple(p), (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(vis, str(i), (int(d[0]) + 8, int(d[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    axis_len = float(marker_size) * 0.35
    axis_pts = np.array([
        [0.0, 0.0, 0.0],
        [axis_len, 0.0, 0.0],
        [0.0, axis_len, 0.0],
        [0.0, 0.0, axis_len],
    ], dtype=np.float64)
    axis_proj = as_int_points(project_points(axis_pts, rvec1_pred, tvec1_pred, K1, D1))
    origin = tuple(axis_proj[0])
    cv2.arrowedLine(vis, origin, tuple(axis_proj[1]), (0, 0, 255), 3, cv2.LINE_AA, 0, 0.18)
    cv2.arrowedLine(vis, origin, tuple(axis_proj[2]), (0, 255, 0), 3, cv2.LINE_AA, 0, 0.18)
    cv2.arrowedLine(vis, origin, tuple(axis_proj[3]), (255, 0, 0), 3, cv2.LINE_AA, 0, 0.18)

    state = "USED" if rec.get("used_final", False) else "REJECTED"
    reason = rec.get("filter_reason", "")
    title = "pair_{:04d} {} rear(cam0)->front(cam1) grid; green=front detect red=rear projected".format(
        rec["idx"], state
    )
    cv2.putText(vis, title, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (0, 255, 255), 2, cv2.LINE_AA)
    if reason:
        cv2.putText(vis, "filter: {}".format(reason), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, "mean px 0to1={:.3f} 1to0={:.3f}".format(
        rec.get("cross_0_to_1_mean_px", float("nan")),
        rec.get("cross_1_to_0_mean_px", float("nan"))),
        (30, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 255), 2, cv2.LINE_AA)
    return vis


def make_validation_panel(rec, front_vis):
    rear = rec["img0"].copy()
    corners = as_int_points(rec["corners0"])
    cv2.polylines(rear, [corners], True, (0, 255, 0), 3, lineType=cv2.LINE_AA)
    for i, p in enumerate(corners):
        cv2.circle(rear, tuple(p), 7, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.putText(rear, str(i), (int(p[0]) + 8, int(p[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(rear, "rear cam0 detected marker", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                (0, 255, 255), 2, cv2.LINE_AA)

    target_h = 720
    scale0 = target_h / float(rear.shape[0])
    scale1 = target_h / float(front_vis.shape[0])
    rear_small = cv2.resize(rear, (int(rear.shape[1] * scale0), target_h), interpolation=cv2.INTER_AREA)
    front_small = cv2.resize(front_vis, (int(front_vis.shape[1] * scale1), target_h), interpolation=cv2.INTER_AREA)
    return np.hstack([rear_small, front_small])


def save_contact_sheet(images, out_path, cols=3, thumb_w=640):
    if not images:
        return
    thumbs = []
    for img in images:
        scale = thumb_w / float(img.shape[1])
        thumb_h = max(1, int(img.shape[0] * scale))
        thumbs.append(cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    thumb_h = max(t.shape[0] for t in thumbs)
    rows = []
    for start in range(0, len(thumbs), cols):
        row_imgs = thumbs[start:start + cols]
        padded = []
        for img in row_imgs:
            if img.shape[0] < thumb_h:
                pad = np.zeros((thumb_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                img = np.vstack([img, pad])
            padded.append(img)
        while len(padded) < cols:
            padded.append(np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8))
        rows.append(np.hstack(padded))
    sheet = np.vstack(rows)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def save_validation(records, T_ext, obj_pts, marker_size, K1, D1, out_dir, grid_steps, include_all=False):
    out_dir = Path(out_dir)
    front_dir = out_dir / "front_projection"
    panel_dir = out_dir / "panels"
    front_dir.mkdir(parents=True, exist_ok=True)
    panel_dir.mkdir(parents=True, exist_ok=True)

    chosen = records if include_all else [r for r in records if r.get("used_final", False)]
    contact_images = []
    for rec in chosen:
        front_vis = draw_marker_projection_validation(rec, T_ext, obj_pts, marker_size, K1, D1, grid_steps)
        panel = make_validation_panel(rec, front_vis)
        suffix = "used" if rec.get("used_final", False) else "rejected"
        front_path = front_dir / "pair_{:04d}_{}_rear_to_front_grid.jpg".format(rec["idx"], suffix)
        panel_path = panel_dir / "pair_{:04d}_{}_panel.jpg".format(rec["idx"], suffix)
        cv2.imwrite(str(front_path), front_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(panel_path), panel, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if rec.get("used_final", False) and len(contact_images) < 12:
            contact_images.append(front_vis)

    save_contact_sheet(contact_images, out_dir / "used_projection_contact_sheet.jpg")


def yaml_matrix_block(name, M):
    lines = [name + ":"]
    for row in np.asarray(M):
        lines.append("  - [{}]".format(", ".join("{:.10f}".format(float(x)) for x in row)))
    return "\n".join(lines)


def yaml_vector_line(name, v):
    return "{}: [{}]".format(name, ", ".join("{:.10f}".format(float(x)) for x in np.asarray(v).reshape(-1)))


def write_outputs(out_dir, result_name, meta, cam0_name, cam1_name, K0, D0, K1, D1,
                  T_ext, records_all, records_used, opt_result):
    out_dir = Path(out_dir)
    yaml_path = out_dir / "{}.yaml".format(result_name)
    csv_path = out_dir / "{}_per_pair_errors.csv".format(result_name)
    txt_path = out_dir / "{}_summary.txt".format(result_name)

    R = T_ext[:3, :3]
    t = T_ext[:3, 3]
    q = rotmat_to_quat_xyzw(R)
    rpy = rpy_from_R_zyx(R)
    T_inv = invert_T(T_ext)

    e0 = summarize(records_used, "pnp0_mean_px")
    e1 = summarize(records_used, "pnp1_mean_px")
    e01 = summarize(records_used, "cross_0_to_1_mean_px")
    e10 = summarize(records_used, "cross_1_to_0_mean_px")

    used_idx = set(r["idx"] for r in records_used)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "pair_index", "used", "pnp0_mean_px", "pnp1_mean_px",
            "rot_dev_deg", "trans_dev_m",
            "cross_0_to_1_mean_px", "cross_0_to_1_max_px",
            "cross_1_to_0_mean_px", "cross_1_to_0_max_px",
            "cam0_ids", "cam1_ids", "filter_reason",
        ])
        for r in records_all:
            w.writerow([
                r["idx"], int(r["idx"] in used_idx),
                "{:.6f}".format(r.get("pnp0_mean_px", float("nan"))),
                "{:.6f}".format(r.get("pnp1_mean_px", float("nan"))),
                "{:.6f}".format(r.get("rot_dev_deg", float("nan"))),
                "{:.6f}".format(r.get("trans_dev_m", float("nan"))),
                "{:.6f}".format(r.get("cross_0_to_1_mean_px", float("nan"))),
                "{:.6f}".format(r.get("cross_0_to_1_max_px", float("nan"))),
                "{:.6f}".format(r.get("cross_1_to_0_mean_px", float("nan"))),
                "{:.6f}".format(r.get("cross_1_to_0_max_px", float("nan"))),
                " ".join(map(str, r.get("ids0", []))),
                " ".join(map(str, r.get("ids1", []))),
                r.get("filter_reason", ""),
            ])

    static_tf = (
        "rosrun tf static_transform_publisher "
        "{:.10f} {:.10f} {:.10f} "
        "{:.10f} {:.10f} {:.10f} {:.10f} "
        "{}_camera_frame {}_camera_frame 100"
    ).format(t[0], t[1], t[2], q[0], q[1], q[2], q[3], cam0_name, cam1_name)

    with yaml_path.open("w") as f:
        f.write("# Generated by scripts/c2c_calibrate_vehicle_aruco.py\n")
        for key, value in meta.items():
            f.write("{}: '{}'\n".format(key, value))
        f.write("cam0_name: '{}'\n".format(cam0_name))
        f.write("cam1_name: '{}'\n".format(cam1_name))
        f.write("num_pairs_all: {}\n".format(len(records_all)))
        f.write("num_pairs_used: {}\n".format(len(records_used)))
        f.write("transform_convention: 'p_cam1 = R_cam1_cam0 * p_cam0 + t_cam1_cam0'\n")
        f.write("T_name: 'T_{}_from_{}'\n".format(cam1_name, cam0_name))
        f.write(yaml_matrix_block("K_cam0", K0) + "\n")
        f.write(yaml_vector_line("D_cam0", D0) + "\n")
        f.write(yaml_matrix_block("K_cam1", K1) + "\n")
        f.write(yaml_vector_line("D_cam1", D1) + "\n")
        f.write(yaml_matrix_block("T_cam1_cam0", T_ext) + "\n")
        f.write(yaml_matrix_block("R_cam1_cam0", R) + "\n")
        f.write(yaml_vector_line("t_cam1_cam0_m", t) + "\n")
        f.write(yaml_vector_line("quaternion_xyzw_cam1_cam0", q) + "\n")
        f.write(yaml_vector_line("rpy_deg_zyx_roll_pitch_yaw", rpy) + "\n")
        f.write(yaml_matrix_block("T_cam0_cam1", T_inv) + "\n")
        f.write("baseline_m: {:.10f}\n".format(float(np.linalg.norm(t))))
        f.write("errors_used:\n")
        f.write("  pnp_cam0_mean_px: {:.6f}\n".format(e0["mean"]))
        f.write("  pnp_cam1_mean_px: {:.6f}\n".format(e1["mean"]))
        f.write("  cross_cam0_to_cam1_mean_px: {:.6f}\n".format(e01["mean"]))
        f.write("  cross_cam0_to_cam1_median_px: {:.6f}\n".format(e01["median"]))
        f.write("  cross_cam0_to_cam1_p90_px: {:.6f}\n".format(e01["p90"]))
        f.write("  cross_cam0_to_cam1_max_px: {:.6f}\n".format(e01["max"]))
        f.write("  cross_cam1_to_cam0_mean_px: {:.6f}\n".format(e10["mean"]))
        f.write("  cross_cam1_to_cam0_median_px: {:.6f}\n".format(e10["median"]))
        f.write("  cross_cam1_to_cam0_p90_px: {:.6f}\n".format(e10["p90"]))
        f.write("  cross_cam1_to_cam0_max_px: {:.6f}\n".format(e10["max"]))
        if opt_result is not None:
            f.write("global_optimization:\n")
            f.write("  success: {}\n".format(bool(opt_result.success)))
            f.write("  cost: {:.10f}\n".format(float(opt_result.cost)))
            f.write("  optimality: {:.10f}\n".format(float(opt_result.optimality)))
            f.write("  nfev: {}\n".format(int(opt_result.nfev)))
        f.write("static_transform_publisher: '{}'\n".format(static_tf))

    with txt_path.open("w") as f:
        f.write("T_cam1_cam0 = T_{}_from_{}\n".format(cam1_name, cam0_name))
        f.write("p_cam1 = R * p_cam0 + t\n\n")
        f.write("R =\n{}\n\n".format(np.array2string(R, precision=10, suppress_small=False)))
        f.write("t [m] = {}\n".format(np.array2string(t, precision=10)))
        f.write("baseline [m] = {:.6f}\n".format(float(np.linalg.norm(t))))
        f.write("quaternion xyzw = {}\n".format(np.array2string(q, precision=10)))
        f.write("RPY deg [roll pitch yaw] = {}\n\n".format(np.array2string(rpy, precision=6)))
        f.write("PnP cam0 mean/median/max = {:.4f}/{:.4f}/{:.4f} px\n".format(e0["mean"], e0["median"], e0["max"]))
        f.write("PnP cam1 mean/median/max = {:.4f}/{:.4f}/{:.4f} px\n".format(e1["mean"], e1["median"], e1["max"]))
        f.write("Cross cam0->cam1 mean/median/p90/max = {:.4f}/{:.4f}/{:.4f}/{:.4f} px\n".format(
            e01["mean"], e01["median"], e01["p90"], e01["max"]))
        f.write("Cross cam1->cam0 mean/median/p90/max = {:.4f}/{:.4f}/{:.4f}/{:.4f} px\n\n".format(
            e10["mean"], e10["median"], e10["p90"], e10["max"]))
        f.write(static_tf + "\n")

    return yaml_path, csv_path, txt_path, static_tf


def main():
    parser = argparse.ArgumentParser(
        description="FAST-Calib vehicle camera-to-camera extrinsic calibration from vehicle.yaml and cameras/*.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vehicle", default=DEFAULT_VEHICLE, help="Vehicle id, e.g. 221 or 227")
    parser.add_argument("--config-root", default=DEFAULT_CONFIG_ROOT,
                        help="Root containing config/vehicles/<vehicle>/")
    parser.add_argument("--vehicle-config", default="",
                        help="Override vehicle.yaml path")
    parser.add_argument("--camera-dir", default="",
                        help="Override camera intrinsic directory")
    parser.add_argument("-g", "--group", default="",
                        help="Calibration group: right/r or left/l. Empty means vehicle.yaml c2c_calibration.default_group.")
    parser.add_argument("--data-dir", default="",
                        help="Override image pair directory. Normally read from vehicle.yaml group data_dir.")
    parser.add_argument("--cam0-name", default="", help="Override cam0 name")
    parser.add_argument("--cam1-name", default="", help="Override cam1 name")
    parser.add_argument("--dictionary", default="", help="Override ArUco dictionary")
    parser.add_argument("--marker-id", type=int, default=None, help="Override ArUco marker id")
    parser.add_argument("--marker-size", type=float, default=None,
                        help="Override marker outer black square size in meters")
    parser.add_argument("--result-name", default="c2c_extrinsic_result", help="Output file prefix")
    parser.add_argument("--allow-disabled", action="store_true",
                        help="Allow running a group with enabled: false")
    parser.add_argument("--no-filter", action="store_true", help="Disable automatic outlier filtering")
    parser.add_argument("--no-cross-filter", action="store_true",
                        help="Disable second-pass cross-reprojection outlier filtering")
    parser.add_argument("--max-cross-0-to-1-mean", type=float, default=MAX_CROSS_0_TO_1_MEAN_PX,
                        help="Second-pass max mean cross reprojection from cam0 pose into cam1; <=0 disables this check")
    parser.add_argument("--max-cross-1-to-0-mean", type=float, default=MAX_CROSS_1_TO_0_MEAN_PX,
                        help="Second-pass max mean cross reprojection from cam1 pose into cam0; <=0 disables this check")
    parser.add_argument("--min-cross-filter-pairs", type=int, default=MIN_CROSS_FILTER_PAIRS,
                        help="Minimum remaining pairs required to accept second-pass cross filtering")
    parser.add_argument("--no-global-opt", action="store_true", help="Disable scipy global nonlinear optimization")
    parser.add_argument("--save-debug", action="store_true", help="Save cross-reprojection overlay images")
    parser.add_argument("--save-validation", action="store_true",
                        help="Save rear/cam0 marker-plane grid projected into front/cam1 images")
    parser.add_argument("--validation-grid", type=int, default=25,
                        help="Grid resolution for validation marker-plane point projection")
    parser.add_argument("--validation-all-pairs", action="store_true",
                        help="Save validation overlays for all valid pairs, not only final used pairs")
    args = parser.parse_args()

    vehicle_id, vehicle_dir, vehicle_yaml, vehicle_cfg = load_vehicle_config(args)
    camera_dir, cameras = load_vehicle_cameras(vehicle_dir, vehicle_cfg, args.camera_dir)
    group, group_node, cam0_name, cam1_name = resolve_group_from_vehicle(
        vehicle_cfg, args.group, args.cam0_name, args.cam1_name, args.allow_disabled
    )
    data_dir = resolve_data_dir(vehicle_dir, vehicle_cfg, group_node, args.data_dir)

    if cam0_name not in cameras:
        raise RuntimeError("cam0 '{}' not found. Available cameras: {}".format(cam0_name, sorted(cameras.keys())))
    if cam1_name not in cameras:
        raise RuntimeError("cam1 '{}' not found. Available cameras: {}".format(cam1_name, sorted(cameras.keys())))

    K0, D0 = cameras[cam0_name]["K"], cameras[cam0_name]["D"]
    K1, D1 = cameras[cam1_name]["K"], cameras[cam1_name]["D"]
    dictionary_name, marker_id, marker_size = resolve_aruco_params(vehicle_cfg, args)

    if not data_dir.exists():
        raise RuntimeError("data_dir does not exist: {}".format(data_dir))

    print("============================================================")
    print("FAST-Calib vehicle C2C ArUco calibration")
    print("vehicle_id     :", vehicle_id)
    print("vehicle_yaml   :", vehicle_yaml)
    print("camera_dir     :", camera_dir)
    print("data_dir       :", data_dir)
    print("group          :", group)
    print("cam0           :", cam0_name, "from", cameras[cam0_name]["path"])
    print("cam1           :", cam1_name, "from", cameras[cam1_name]["path"])
    print("dictionary     :", dictionary_name)
    print("marker_id      :", marker_id)
    print("marker_size_m  :", marker_size)
    print("============================================================")

    dictionary = make_dictionary(dictionary_name)
    params = make_detector_params()
    obj_pts = marker_object_points(marker_size)

    pairs = find_pairs(data_dir)
    if not pairs:
        raise RuntimeError("No pair_XXXX_cam0.* / pair_XXXX_cam1.* image pairs found in {}".format(data_dir))
    print("[INFO] found {} image pairs".format(len(pairs)))

    records = []
    for idx, p0, p1 in pairs:
        try:
            img0, corners0, ids0 = detect_one_image(p0, dictionary, params, marker_id)
            img1, corners1, ids1 = detect_one_image(p1, dictionary, params, marker_id)
            if corners0 is None or corners1 is None:
                print("[SKIP] pair_{:04d}: target id missing, cam0 ids={}, cam1 ids={}".format(idx, ids0, ids1))
                continue

            rvec0, tvec0, err0 = solve_pnp_square(obj_pts, corners0, K0, D0)
            rvec1, tvec1, err1 = solve_pnp_square(obj_pts, corners1, K1, D1)

            T0 = rt_to_T(rvec0, tvec0)
            T1 = rt_to_T(rvec1, tvec1)
            T10 = T1 @ invert_T(T0)

            records.append({
                "idx": idx,
                "img0": img0,
                "img1": img1,
                "corners0": corners0,
                "corners1": corners1,
                "ids0": ids0,
                "ids1": ids1,
                "rvec0": rvec0,
                "tvec0": tvec0,
                "rvec1": rvec1,
                "tvec1": tvec1,
                "pnp0_mean_px": float(err0),
                "pnp1_mean_px": float(err1),
                "T_pair": T10,
            })
            print("[OK] pair_{:04d}: pnp0={:.3f}px pnp1={:.3f}px".format(idx, err0, err1))
        except Exception as e:
            print("[SKIP] pair_{:04d}: {}".format(idx, e))

    if len(records) < 2:
        raise RuntimeError("Valid image pairs are too few. Need at least 2, recommend 15+.")

    T_all = average_T([r["T_pair"] for r in records])
    for r in records:
        dT = r["T_pair"] @ invert_T(T_all)
        r["rot_dev_deg"] = rotation_angle_deg(dT[:3, :3])
        r["trans_dev_m"] = float(np.linalg.norm(r["T_pair"][:3, 3] - T_all[:3, 3]))

    for r in records:
        r["filter_reason"] = ""

    if args.no_filter:
        used, rejected = list(records), []
    else:
        used, rejected = [], []
        for r in records:
            ok = (
                r["pnp0_mean_px"] <= MAX_PNP_REPROJ_PX
                and r["pnp1_mean_px"] <= MAX_PNP_REPROJ_PX
                and r["rot_dev_deg"] <= MAX_ROT_DEV_DEG
                and r["trans_dev_m"] <= MAX_TRANS_DEV_M
            )
            (used if ok else rejected).append(r)
        if len(used) < 3:
            print("[WARN] filtered frames are too few, use all valid frames")
            used, rejected = list(records), []
        else:
            for r in rejected:
                r["filter_reason"] = "first_pass_filter"

    first_pass_used = list(used)
    first_pass_rejected = list(rejected)
    cross_rejected = []

    print("[INFO] first-pass filter: valid pairs: {}, used: {}, rejected: {}".format(
        len(records), len(first_pass_used), format_pair_indices(first_pass_rejected)
    ))

    T_init = average_T([r["T_pair"] for r in used])
    T_final, opt_result = run_global_optimization(
        used, T_init, obj_pts, K0, D0, K1, D1, no_global_opt=args.no_global_opt
    )

    compute_cross_errors(records, T_final, obj_pts, K0, D0, K1, D1, use_opt_pose=False)
    compute_cross_errors(used, T_final, obj_pts, K0, D0, K1, D1, use_opt_pose=True)

    cross_filter_enabled = not args.no_filter and not args.no_cross_filter
    if cross_filter_enabled:
        kept, cross_rejected = apply_cross_filter(used, args)
        if cross_rejected:
            min_pairs = max(3, int(args.min_cross_filter_pairs))
            if len(kept) >= min_pairs:
                print("[INFO] cross filter thresholds: cam0->cam1 mean <= {:.3f}px, cam1->cam0 mean <= {:.3f}px".format(
                    args.max_cross_0_to_1_mean, args.max_cross_1_to_0_mean
                ))
                print("[INFO] cross filter rejected: {}".format(
                    ",".join(format_cross_filter_rejection(r) for r in cross_rejected)
                ))
                for r in cross_rejected:
                    r["filter_reason"] = "cross_filter"
                used = kept
                T_init = average_T([r["T_pair"] for r in used])
                T_final, opt_result = run_global_optimization(
                    used, T_init, obj_pts, K0, D0, K1, D1, no_global_opt=args.no_global_opt
                )
                compute_cross_errors(records, T_final, obj_pts, K0, D0, K1, D1, use_opt_pose=False)
                compute_cross_errors(used, T_final, obj_pts, K0, D0, K1, D1, use_opt_pose=True)
            else:
                print("[WARN] cross filter would leave only {} pairs (< {}), keep first-pass result".format(
                    len(kept), min_pairs
                ))
                cross_rejected = []
        else:
            print("[INFO] cross filter rejected: none")
    elif args.no_cross_filter:
        print("[INFO] cross filter disabled by --no-cross-filter")
    elif args.no_filter:
        print("[INFO] cross filter disabled by --no-filter")

    print("[INFO] final used pairs: {}, rejected: {}".format(
        len(used), format_pair_indices([r for r in records if r["idx"] not in set(x["idx"] for x in used)])
    ))

    used_idx_final = set(r["idx"] for r in used)
    for r in records:
        r["used_final"] = r["idx"] in used_idx_final

    if args.save_debug:
        debug_dir = data_dir / "{}_debug_reprojection".format(args.result_name)
        save_debug(used, T_final, obj_pts, K1, D1, debug_dir)
        print("[SAVE] debug images:", debug_dir)

    if args.save_validation:
        validation_dir = data_dir / "{}_validation_projection".format(args.result_name)
        save_validation(
            records,
            T_final,
            obj_pts,
            marker_size,
            K1,
            D1,
            validation_dir,
            args.validation_grid,
            include_all=args.validation_all_pairs,
        )
        print("[SAVE] validation images:", validation_dir)

    meta = {
        "vehicle_id": vehicle_id,
        "vehicle_yaml": str(vehicle_yaml),
        "camera_dir": str(camera_dir),
        "data_dir": str(data_dir),
        "group": group,
        "dictionary": dictionary_name,
        "marker_id": marker_id,
        "marker_size_m": marker_size,
        "first_pass_pairs_used": len(first_pass_used),
        "first_pass_rejected": format_pair_indices(first_pass_rejected),
        "cross_filter_enabled": cross_filter_enabled,
        "cross_filter_max_0_to_1_mean_px": args.max_cross_0_to_1_mean,
        "cross_filter_max_1_to_0_mean_px": args.max_cross_1_to_0_mean,
        "cross_filter_rejected": format_pair_indices(cross_rejected),
        "validation_saved": args.save_validation,
        "validation_grid": args.validation_grid,
        "validation_all_pairs": args.validation_all_pairs,
    }

    yaml_path, csv_path, txt_path, static_tf = write_outputs(
        data_dir,
        args.result_name,
        meta,
        cam0_name,
        cam1_name,
        K0,
        D0,
        K1,
        D1,
        T_final,
        records,
        used,
        opt_result,
    )

    R = T_final[:3, :3]
    t = T_final[:3, 3]
    q = rotmat_to_quat_xyzw(R)
    rpy = rpy_from_R_zyx(R)
    e0 = summarize(used, "pnp0_mean_px")
    e1 = summarize(used, "pnp1_mean_px")
    e01 = summarize(used, "cross_0_to_1_mean_px")
    e10 = summarize(used, "cross_1_to_0_mean_px")

    print("============================================================")
    print("FINAL: T_{}_from_{}".format(cam1_name, cam0_name))
    print("R =")
    print(np.array2string(R, precision=10, suppress_small=False))
    print("t [m] =", np.array2string(t, precision=10, suppress_small=False))
    print("baseline [m] = {:.6f}".format(float(np.linalg.norm(t))))
    print("quaternion xyzw =", np.array2string(q, precision=10, suppress_small=False))
    print("RPY deg [roll pitch yaw] =", np.array2string(rpy, precision=6, suppress_small=False))
    print("PnP cam0 mean/median/max = {:.4f}/{:.4f}/{:.4f} px".format(e0["mean"], e0["median"], e0["max"]))
    print("PnP cam1 mean/median/max = {:.4f}/{:.4f}/{:.4f} px".format(e1["mean"], e1["median"], e1["max"]))
    print("Cross cam0->cam1 mean/median/p90/max = {:.4f}/{:.4f}/{:.4f}/{:.4f} px".format(
        e01["mean"], e01["median"], e01["p90"], e01["max"]))
    print("Cross cam1->cam0 mean/median/p90/max = {:.4f}/{:.4f}/{:.4f}/{:.4f} px".format(
        e10["mean"], e10["median"], e10["p90"], e10["max"]))
    print("static_transform_publisher:")
    print(static_tf)
    print("[SAVE] YAML:", yaml_path)
    print("[SAVE] CSV :", csv_path)
    print("[SAVE] TXT :", txt_path)
    print("============================================================")


if __name__ == "__main__":
    main()
