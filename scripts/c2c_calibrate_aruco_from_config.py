#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c2c_calibrate_aruco_from_config.py

FAST-Calib helper:
  Use paired images collected by collect_two_camera_aruco_single_once.py to
  estimate camera-to-camera extrinsics. Camera intrinsics are read from a
  FAST-Calib yaml config, so normally you only need to provide --data-dir.

Default convention for -g r:
  cam0 = rear_right
  cam1 = front_right
  T_cam1_cam0 = T_front_right_from_rear_right
  p_cam1 = R_cam1_cam0 * p_cam0 + t_cam1_cam0

Default convention for -g l:
  cam0 = rear_left
  cam1 = front_left

Expected image naming:
  pair_0001_cam0.jpg
  pair_0001_cam1.jpg
  pair_0002_cam0.jpg
  pair_0002_cam1.jpg
  ...

Supported intrinsics yaml layouts:

1) Recommended FAST-Calib vehicle-camera layout:

camera_intrinsics:
  rear_right:
    fx: 1019.539983
    fy: 1020.371160
    cx: 934.132180
    cy: 545.606007
    k1: -0.313954
    k2: 0.070619
    p1: -0.000869
    p2: 0.002769
    k3: 0.0
  front_right:
    fx: 1045.618822
    fy: 1046.422188
    cx: 971.136335
    cy: 547.311407
    k1: -0.305536
    k2: 0.064082
    p1: -0.000154
    p2: -0.001684
    k3: 0.0

camera_groups:
  right:
    cam0: rear_right
    cam1: front_right
  left:
    cam0: rear_left
    cam1: front_left

c2c_calibration:
  dictionary: DICT_6X6_250
  marker_id: 1
  marker_size_m: 0.80

2) ROS camera_info-like layout:

cameras:
  rear_right:
    camera_matrix:
      data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    distortion_coefficients:
      data: [k1, k2, p1, p2, k3]

Dependencies:
  ROS Noetic python environment
  OpenCV with aruco module
  PyYAML
  scipy is optional; if installed, the script performs global nonlinear
  optimization after robust initialization.
"""

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_CONFIG = "$(find fast_calib)/config/qr_params.yaml"
DEFAULT_GROUP = "r"
DEFAULT_DICTIONARY = "DICT_6X6_250"
DEFAULT_MARKER_ID = 1
DEFAULT_MARKER_SIZE_M = 0.80

MAX_PNP_REPROJ_PX = 0.80
MAX_ROT_DEV_DEG = 1.20
MAX_TRANS_DEV_M = 0.20


def resolve_ros_path(path_text):
    """Resolve $(find pkg) in a path without requiring a running roscore."""
    path_text = str(path_text)
    pattern = re.compile(r"\$\(find\s+([^)]+)\)")

    def repl(match):
        pkg = match.group(1).strip()
        try:
            import rospkg
            return rospkg.RosPack().get_path(pkg)
        except Exception:
            # Fallback for direct execution inside the package.
            here = Path(__file__).resolve()
            for parent in [here.parent] + list(here.parents):
                if parent.name == pkg:
                    return str(parent)
                if (parent / "package.xml").exists() and parent.name == pkg:
                    return str(parent)
            raise RuntimeError(
                "Cannot resolve $(find {}). Source your catkin workspace or install rospkg.".format(pkg)
            )

    return str(Path(pattern.sub(repl, path_text)).expanduser().resolve())


def load_yaml(config_path):
    config_path = resolve_ros_path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise RuntimeError("Config yaml root must be a dictionary: {}".format(config_path))
    return config_path, data


def _as_float_list(value):
    if value is None:
        return None
    if isinstance(value, dict):
        if "data" in value:
            return _as_float_list(value["data"])
        if "D" in value:
            return _as_float_list(value["D"])
        if "K" in value:
            return _as_float_list(value["K"])
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return None


def parse_intrinsic_node(node):
    if not isinstance(node, dict):
        return None

    # fx/fy/cx/cy style
    keys = {str(k).lower(): k for k in node.keys()}
    if all(k in keys for k in ("fx", "fy", "cx", "cy")):
        fx = float(node[keys["fx"]])
        fy = float(node[keys["fy"]])
        cx = float(node[keys["cx"]])
        cy = float(node[keys["cy"]])
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float64)

        k1 = float(node.get(keys.get("k1", "k1"), 0.0))
        k2 = float(node.get(keys.get("k2", "k2"), 0.0))
        p1 = float(node.get(keys.get("p1", "p1"), 0.0))
        p2 = float(node.get(keys.get("p2", "p2"), 0.0))
        k3 = float(node.get(keys.get("k3", "k3"), 0.0))
        D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        return K, D

    # K/D style
    K_list = None
    for k in ("K", "k", "camera_matrix", "intrinsic_matrix"):
        if k in node:
            K_list = _as_float_list(node[k])
            break

    if K_list is not None and len(K_list) == 9:
        K = np.asarray(K_list, dtype=np.float64).reshape(3, 3)

        D_list = None
        for k in ("D", "d", "distortion_coefficients", "dist_coeffs", "distortion"):
            if k in node:
                D_list = _as_float_list(node[k])
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
        D = np.asarray(D_list[:5], dtype=np.float64)
        return K, D

    return None


def load_camera_intrinsics(config):
    """
    Return:
      cameras[name] = {"K": K, "D": D}
    """
    cameras = {}

    container_keys = (
        "camera_intrinsics",
        "cameras",
        "camera_params",
        "camera_config",
        "vehicle_cameras",
    )

    candidate_maps = []
    for key in container_keys:
        if key in config and isinstance(config[key], dict):
            candidate_maps.append(config[key])

    # Also allow camera names directly at yaml root.
    candidate_maps.append(config)

    for cmap in candidate_maps:
        for name, node in cmap.items():
            parsed = parse_intrinsic_node(node)
            if parsed is None:
                continue
            K, D = parsed
            cameras[str(name)] = {"K": K, "D": D}

    if not cameras:
        raise RuntimeError(
            "No camera intrinsics found in config. Add camera_intrinsics/cameras "
            "with fx, fy, cx, cy, k1, k2, p1, p2 for each camera."
        )

    return cameras


def normalize_group_name(group):
    if group in ("r", "right", "right_pair"):
        return "right"
    if group in ("l", "left", "left_pair"):
        return "left"
    return group


def resolve_group_cameras(config, group, cam0_override="", cam1_override=""):
    if cam0_override and cam1_override:
        return cam0_override, cam1_override

    group = normalize_group_name(group)

    group_maps = []
    for key in ("camera_groups", "c2c_groups", "stereo_groups", "camera_pairs"):
        if key in config and isinstance(config[key], dict):
            group_maps.append(config[key])

    aliases = {
        "right": ("right", "r", "right_pair"),
        "left": ("left", "l", "left_pair"),
    }.get(group, (group,))

    for gmap in group_maps:
        for alias in aliases:
            if alias not in gmap or not isinstance(gmap[alias], dict):
                continue
            node = gmap[alias]
            cam0 = cam0_override or node.get("cam0") or node.get("camera0") or node.get("rear") or node.get("rear_camera")
            cam1 = cam1_override or node.get("cam1") or node.get("camera1") or node.get("front") or node.get("front_camera")
            if cam0 and cam1:
                return str(cam0), str(cam1)

    # Default vehicle naming used by the current collection scripts.
    fallback = {
        "right": ("rear_right", "front_right"),
        "left": ("rear_left", "front_left"),
    }
    if group in fallback:
        cam0, cam1 = fallback[group]
        return cam0_override or cam0, cam1_override or cam1

    raise RuntimeError("Cannot resolve group cameras for group={}".format(group))


def read_c2c_params(config, args):
    c2c = {}
    for key in ("c2c_calibration", "camera_to_camera_calibration", "aruco_c2c"):
        if key in config and isinstance(config[key], dict):
            c2c.update(config[key])

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
    marker_size = float(marker_size)

    return str(dictionary), int(marker_id), marker_size


def make_dictionary(name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("OpenCV has no cv2.aruco. Install opencv-contrib-python or use ROS OpenCV with aruco.")
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
    data_dir = Path(data_dir)
    pat = re.compile(r"pair_(\d+)_cam0\.(jpg|jpeg|png|bmp)$", re.IGNORECASE)
    pairs = []
    for p0 in sorted(data_dir.glob("pair_*_cam0.*")):
        m = pat.match(p0.name)
        if not m:
            continue
        idx = int(m.group(1))
        p1 = None
        for ext in ("jpg", "jpeg", "png", "bmp"):
            cand = data_dir / "pair_{:04d}_cam1.{}".format(idx, ext)
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
        err = reprojection_error(obj_pts, img_pts, rvec, tvec, K, D)
        best = (np.asarray(rvec).reshape(3, 1), np.asarray(tvec).reshape(3, 1), err)

    return best


def rt_to_T(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def T_to_rt(T):
    rvec, _ = cv2.Rodrigues(np.asarray(T[:3, :3], dtype=np.float64))
    tvec = np.asarray(T[:3, 3], dtype=np.float64).reshape(3, 1)
    return rvec.reshape(3, 1), tvec


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
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return q


def quat_xyzw_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    q = q / np.linalg.norm(q)
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
        q = q / np.linalg.norm(q)
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
        rvec_ext = x[0:3].reshape(3, 1)
        tvec_ext = x[3:6].reshape(3, 1)
        T_ext = rt_to_T(rvec_ext, tvec_ext)
        out = []
        off = 6
        for i, rec in enumerate(records):
            rvec0 = x[off + 6*i: off + 6*i + 3].reshape(3, 1)
            tvec0 = x[off + 6*i + 3: off + 6*i + 6].reshape(3, 1)

            proj0 = project_points(obj_pts, rvec0, tvec0, K0, D0)
            out.extend((proj0 - rec["corners0"]).reshape(-1))

            T0 = rt_to_T(rvec0, tvec0)
            T1 = T_ext @ T0
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

        T0 = rt_to_T(rvec0, tvec0)
        T1_pred = T_ext @ T0
        rvec1_pred, tvec1_pred = T_to_rt(T1_pred)
        proj1 = project_points(obj_pts, rvec1_pred, tvec1_pred, K1, D1)
        e01 = np.linalg.norm(proj1 - rec["corners1"], axis=1)

        T1_meas = rt_to_T(rec["rvec1"], rec["tvec1"])
        T0_pred = T_inv @ T1_meas
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


def yaml_matrix_block(name, M):
    lines = [name + ":"]
    for row in np.asarray(M):
        lines.append("  - [{}]".format(", ".join("{:.10f}".format(float(x)) for x in row)))
    return "\n".join(lines)


def yaml_vector_line(name, v):
    return "{}: [{}]".format(name, ", ".join("{:.10f}".format(float(x)) for x in np.asarray(v).reshape(-1)))


def write_outputs(out_dir, result_name, args, config_path, cam0_name, cam1_name, K0, D0, K1, D1,
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

    e_pnp0 = summarize(records_used, "pnp0_mean_px")
    e_pnp1 = summarize(records_used, "pnp1_mean_px")
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
            "cam0_ids", "cam1_ids",
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
            ])

    static_tf = (
        "rosrun tf static_transform_publisher "
        "{:.10f} {:.10f} {:.10f} "
        "{:.10f} {:.10f} {:.10f} {:.10f} "
        "{}_camera_frame {}_camera_frame 100"
    ).format(t[0], t[1], t[2], q[0], q[1], q[2], q[3], cam0_name, cam1_name)

    with yaml_path.open("w") as f:
        f.write("# Generated by scripts/c2c_calibrate_aruco_from_config.py\n")
        f.write("config_path: '{}'\n".format(config_path))
        f.write("data_dir: '{}'\n".format(str(Path(args.data_dir).resolve())))
        f.write("group: '{}'\n".format(args.group))
        f.write("cam0_name: '{}'\n".format(cam0_name))
        f.write("cam1_name: '{}'\n".format(cam1_name))
        f.write("dictionary: '{}'\n".format(args.dictionary_resolved))
        f.write("marker_id: {}\n".format(args.marker_id_resolved))
        f.write("marker_size_m: {:.10f}\n".format(args.marker_size_resolved))
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
        f.write("  pnp_cam0_mean_px: {:.6f}\n".format(e_pnp0["mean"]))
        f.write("  pnp_cam1_mean_px: {:.6f}\n".format(e_pnp1["mean"]))
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
        f.write("PnP cam0 mean/median/max = {:.4f}/{:.4f}/{:.4f} px\n".format(
            e_pnp0["mean"], e_pnp0["median"], e_pnp0["max"]))
        f.write("PnP cam1 mean/median/max = {:.4f}/{:.4f}/{:.4f} px\n".format(
            e_pnp1["mean"], e_pnp1["median"], e_pnp1["max"]))
        f.write("Cross cam0->cam1 mean/median/p90/max = {:.4f}/{:.4f}/{:.4f}/{:.4f} px\n".format(
            e01["mean"], e01["median"], e01["p90"], e01["max"]))
        f.write("Cross cam1->cam0 mean/median/p90/max = {:.4f}/{:.4f}/{:.4f}/{:.4f} px\n\n".format(
            e10["mean"], e10["median"], e10["p90"], e10["max"]))
        f.write(static_tf + "\n")

    return yaml_path, csv_path, txt_path, static_tf


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate camera-to-camera extrinsic from FAST-Calib config and paired ArUco images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", "-i", required=True,
                        help="Directory containing pair_XXXX_cam0.* and pair_XXXX_cam1.*")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="FAST-Calib yaml config containing all vehicle camera intrinsics")
    parser.add_argument("-g", "--group", default=DEFAULT_GROUP,
                        choices=["l", "r", "left", "right"],
                        help="Camera group")
    parser.add_argument("--cam0-name", default="",
                        help="Override cam0 camera name in config")
    parser.add_argument("--cam1-name", default="",
                        help="Override cam1 camera name in config")
    parser.add_argument("--dictionary", default="",
                        help="Aruco dictionary, e.g. DICT_6X6_250. Empty means read config or use default.")
    parser.add_argument("--marker-id", type=int, default=None,
                        help="Aruco marker id. Empty means read config or use default.")
    parser.add_argument("--marker-size", type=float, default=None,
                        help="Marker outer black square size in meters. Empty means read config or default 0.80.")
    parser.add_argument("--result-name", default="c2c_extrinsic_result",
                        help="Output file prefix")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable automatic outlier filtering")
    parser.add_argument("--no-global-opt", action="store_true",
                        help="Disable scipy global nonlinear optimization")
    parser.add_argument("--save-debug", action="store_true",
                        help="Save cam1 cross-reprojection overlay images")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise RuntimeError("data_dir does not exist: {}".format(data_dir))

    config_path, config = load_yaml(args.config)
    cameras = load_camera_intrinsics(config)
    cam0_name, cam1_name = resolve_group_cameras(config, args.group, args.cam0_name, args.cam1_name)

    if cam0_name not in cameras:
        raise RuntimeError("cam0 '{}' not found in config cameras: {}".format(cam0_name, sorted(cameras.keys())))
    if cam1_name not in cameras:
        raise RuntimeError("cam1 '{}' not found in config cameras: {}".format(cam1_name, sorted(cameras.keys())))

    K0, D0 = cameras[cam0_name]["K"], cameras[cam0_name]["D"]
    K1, D1 = cameras[cam1_name]["K"], cameras[cam1_name]["D"]

    dictionary_name, marker_id, marker_size = read_c2c_params(config, args)
    args.dictionary_resolved = dictionary_name
    args.marker_id_resolved = marker_id
    args.marker_size_resolved = marker_size

    print("============================================================")
    print("FAST-Calib camera-to-camera ArUco calibration")
    print("config       :", config_path)
    print("data_dir     :", data_dir)
    print("group        :", args.group)
    print("cam0         :", cam0_name)
    print("cam1         :", cam1_name)
    print("dictionary   :", dictionary_name)
    print("marker_id    :", marker_id)
    print("marker_size  :", marker_size, "m")
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

    if args.no_filter:
        used = records
        rejected = []
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
            used, rejected = records, []

    print("[INFO] valid pairs: {}, used: {}, rejected: {}".format(
        len(records), len(used), ",".join("pair_{:04d}".format(r["idx"]) for r in rejected) or "none"
    ))

    T_init = average_T([r["T_pair"] for r in used])
    T_final, opt_result = run_global_optimization(
        used, T_init, obj_pts, K0, D0, K1, D1, no_global_opt=args.no_global_opt
    )

    compute_cross_errors(records, T_final, obj_pts, K0, D0, K1, D1, use_opt_pose=False)
    compute_cross_errors(used, T_final, obj_pts, K0, D0, K1, D1, use_opt_pose=True)

    if args.save_debug:
        debug_dir = data_dir / "{}_debug_reprojection".format(args.result_name)
        save_debug(used, T_final, obj_pts, K1, D1, debug_dir)
        print("[SAVE] debug images:", debug_dir)

    yaml_path, csv_path, txt_path, static_tf = write_outputs(
        data_dir,
        args.result_name,
        args,
        config_path,
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
