#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Summarize vehicle camera-to-LiDAR extrinsics, including cameras derived by C2C.

Input conventions:
  - Direct camera/LiDAR calibration uses T_cam_lidar:
        p_cam = T_cam_lidar * p_lidar
  - C2C calibration uses T_cam1_cam0:
        p_cam1 = T_cam1_cam0 * p_cam0

If one camera in a C2C pair has direct T_cam_lidar, the other camera is derived:
  T_cam1_lidar = T_cam1_cam0 * T_cam0_lidar
  T_cam0_lidar = inv(T_cam1_cam0) * T_cam1_lidar
"""

import argparse
import csv
import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml


DEFAULT_VEHICLE = "221"
DEFAULT_CONFIG_ROOT = "$(find fast_calib)/config/vehicles"
DEFAULT_REPORT_ROOT = Path("/home/glf/dataDisk/calib/vehicle_calib_report")
DEFAULT_C2C_ROOT = Path("/home/glf/dataDisk/calib/c2c")
DEFAULT_C2C_RESULT_NAME = "auto"
DEFAULT_LIDAR_TOPIC_FRAME_MAP = "velodyne_first:lidar_first,velodyne_second:lidar_second"
DEFAULT_BASE_FRAME = "base_link"
DEFAULT_CAMERA_FRAME_SUFFIX = "_camera_frame"
DEFAULT_BASE_LIDAR_TF_BY_VEHICLE = {
    "221": {
        "lidar_first": [7.38763, 1.3081, 1.6, 0.00559065, 0.00440275, 0.18853, 0.982042],
        "lidar_second": [-7.41613, -1.38264, 1.6, 0.00169926, -0.00675155, 0.983151, -0.182665],
    },
}


def resolve_ros_path(path_text: Any) -> str:
    text = str(path_text)
    pattern = re.compile(r"\$\(find\s+([^)]+)\)")

    def repl(match):
        pkg = match.group(1).strip()
        try:
            import rospkg
            return rospkg.RosPack().get_path(pkg)
        except Exception:
            here = Path(__file__).resolve()
            for parent in [here.parent] + list(here.parents):
                package_xml = parent / "package.xml"
                if not package_xml.exists():
                    continue
                if parent.name == pkg:
                    return str(parent)
                try:
                    root = ET.parse(str(package_xml)).getroot()
                    package_name = root.findtext("name", default="").strip()
                    if package_name == pkg:
                        return str(parent)
                except Exception:
                    pass
            raise RuntimeError("Cannot resolve $(find {}). Source your catkin workspace.".format(pkg))

    return str(Path(pattern.sub(repl, text)).expanduser().resolve())


def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(resolve_ros_path(path))
    if not path.exists():
        raise FileNotFoundError(str(path))
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def as_abs_path(value: Any, base_dir: Optional[Path] = None) -> Path:
    text = str(value)
    if "$(find" in text:
        return Path(resolve_ros_path(text))
    p = Path(text).expanduser()
    if p.is_absolute():
        return p.resolve()
    if base_dir is not None:
        return (base_dir / p).resolve()
    return p.resolve()


def load_vehicle_config(args) -> Tuple[str, Path, Path, Dict[str, Any]]:
    vehicle_id = str(args.vehicle)
    if args.vehicle_config:
        vehicle_yaml = Path(resolve_ros_path(args.vehicle_config))
        vehicle_dir = vehicle_yaml.parent
    else:
        config_root = Path(resolve_ros_path(args.config_root))
        vehicle_dir = config_root / vehicle_id
        vehicle_yaml = vehicle_dir / "vehicle.yaml"
    return vehicle_id, vehicle_dir, vehicle_yaml, load_yaml(vehicle_yaml)


def mat4(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError("{} must be 4x4, got {}".format(name, arr.shape))
    return arr


def invert_T(T: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rotation_angle_deg(R: np.ndarray) -> float:
    val = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(float(val)))


def rotmat_to_quat_xyzw(R: np.ndarray) -> List[float]:
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
    return rounded_list(q)


def quat_xyzw_to_rotmat(q: Iterable[Any]) -> np.ndarray:
    q = np.asarray(list(q), dtype=np.float64).reshape(4)
    n = float(np.dot(q, q))
    if n <= 0.0:
        raise ValueError("Quaternion norm must be positive")
    q = q / math.sqrt(n)
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def transform_from_xyz_quat(args: Iterable[Any], name: str) -> np.ndarray:
    values = np.asarray(list(args), dtype=np.float64).reshape(-1)
    if values.size != 7:
        raise ValueError("{} must contain 7 values: x y z qx qy qz qw".format(name))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rotmat(values[3:7])
    T[:3, 3] = values[:3]
    return T


def rpy_from_R_zyx(R: np.ndarray) -> List[float]:
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-8:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return rounded_list(np.degrees([roll, pitch, yaw]))


def rounded(x: Any, ndigits: int = 10) -> float:
    return float(round(float(x), ndigits))


def rounded_list(values: Iterable[Any], ndigits: int = 10) -> List[float]:
    return [rounded(v, ndigits) for v in np.asarray(list(values), dtype=np.float64).reshape(-1)]


def rounded_matrix(M: np.ndarray, ndigits: int = 10) -> List[List[float]]:
    return [[rounded(v, ndigits) for v in row] for row in np.asarray(M, dtype=np.float64)]


def flat_matrix(M: np.ndarray, ndigits: int = 10) -> List[float]:
    return rounded_list(np.asarray(M, dtype=np.float64).reshape(-1), ndigits)


def fmt(v: Any) -> str:
    if v is None:
        return ""
    try:
        return "{:.6f}".format(float(v))
    except Exception:
        return str(v)


def fmt_tf(v: Any) -> str:
    return "{:.6f}".format(float(v))


def sanitize_frame_piece(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text)).strip("_")


def parse_lidar_topic_frame_map(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Invalid lidar topic/frame map item '{}'; expected topic:frame".format(item))
        topic, frame = item.split(":", 1)
        out[topic.strip()] = frame.strip()
    return out


def parse_xyz_quat_text(text: str, name: str) -> List[float]:
    values = [float(v) for v in str(text).replace(",", " ").split()]
    if len(values) != 7:
        raise ValueError("{} must contain 7 numbers: x y z qx qy qz qw".format(name))
    return values


def load_base_lidar_tf_yaml(path: Path) -> Tuple[str, Dict[str, List[float]]]:
    data = load_yaml(path)
    base_frame = str(data.get("base_frame") or DEFAULT_BASE_FRAME)
    lidars = data.get("lidars") if isinstance(data.get("lidars"), dict) else data
    out: Dict[str, List[float]] = {}
    if isinstance(lidars, dict):
        for frame, entry in lidars.items():
            if isinstance(entry, dict):
                xyz = entry.get("xyz") or entry.get("translation")
                quat = entry.get("quaternion_xyzw") or entry.get("quat_xyzw") or entry.get("q")
                if xyz is not None and quat is not None:
                    out[str(frame)] = [float(v) for v in list(xyz) + list(quat)]
                    continue
                args = entry.get("xyz_quat") or entry.get("args")
                if args is not None:
                    out[str(frame)] = parse_xyz_quat_text(" ".join(str(v) for v in args), str(frame))
                    continue
            elif isinstance(entry, (list, tuple)):
                out[str(frame)] = [float(v) for v in entry]
    return base_frame, out


def resolve_base_lidar_transforms(args, vehicle_id: str) -> Tuple[str, Dict[str, np.ndarray]]:
    base_frame = args.base_frame
    raw: Dict[str, List[float]] = {}
    if args.base_lidar_tf_yaml:
        base_frame, raw = load_base_lidar_tf_yaml(Path(args.base_lidar_tf_yaml).expanduser().resolve())
    elif vehicle_id in DEFAULT_BASE_LIDAR_TF_BY_VEHICLE:
        raw = dict(DEFAULT_BASE_LIDAR_TF_BY_VEHICLE[vehicle_id])

    if args.base_lidar_first:
        raw["lidar_first"] = parse_xyz_quat_text(args.base_lidar_first, "--base-lidar-first")
    if args.base_lidar_second:
        raw["lidar_second"] = parse_xyz_quat_text(args.base_lidar_second, "--base-lidar-second")

    transforms = {
        frame: transform_from_xyz_quat(values, "T_{}_{}".format(base_frame, frame))
        for frame, values in raw.items()
    }
    return base_frame, transforms


def matrix_entry(T: np.ndarray) -> Dict[str, Any]:
    T = np.asarray(T, dtype=np.float64)
    T_inv = invert_T(T)
    return {
        "Rcl": rounded_matrix(T[:3, :3]),
        "Rcl_flat": flat_matrix(T[:3, :3]),
        "Pcl": rounded_list(T[:3, 3]),
        "Pcl_xyz": rounded_list(T[:3, 3]),
        "T_cam_lidar": rounded_matrix(T),
        "T_cam_lidar_flat": flat_matrix(T),
        "T_lidar_cam": rounded_matrix(T_inv),
        "quaternion_xyzw_cam_lidar": rotmat_to_quat_xyzw(T[:3, :3]),
        "rpy_deg_zyx_roll_pitch_yaw_cam_lidar": rpy_from_R_zyx(T[:3, :3]),
    }


def read_direct_extrinsics(path: Path) -> Dict[str, Dict[str, Any]]:
    data = load_yaml(path)
    out: Dict[str, Dict[str, Any]] = {}

    if isinstance(data.get("sensors"), dict):
        for name, entry in data["sensors"].items():
            if isinstance(entry, dict) and entry.get("T_cam_lidar") is not None:
                T = mat4(entry["T_cam_lidar"], "{}.T_cam_lidar".format(name))
                out[str(name)] = {
                    "camera": str(name),
                    "T": T,
                    "status": entry.get("status", ""),
                    "rmse": entry.get("rmse"),
                    "selected_groups": entry.get("selected_groups", []),
                    "final_group_residuals": entry.get("final_group_residuals", {}),
                    "source_files": entry.get("source_files", {}),
                    "source_summary": str(path),
                    "raw": entry,
                }
        return out

    if isinstance(data.get("extrinsics"), dict):
        for name, entry in data["extrinsics"].items():
            if isinstance(entry, dict) and entry.get("T_cam_lidar") is not None:
                T = mat4(entry["T_cam_lidar"], "{}.T_cam_lidar".format(name))
                out[str(name)] = {
                    "camera": str(name),
                    "T": T,
                    "status": entry.get("status", ""),
                    "rmse": entry.get("rmse"),
                    "selected_groups": entry.get("selected_groups", []),
                    "final_group_residuals": entry.get("final_group_residuals", {}),
                    "source_files": {"direct_summary": str(path)},
                    "source_summary": str(path),
                    "raw": entry,
                }
        return out

    raise RuntimeError("Cannot find sensors/extrinsics with T_cam_lidar in {}".format(path))


def vehicle_project_root(vehicle_dir: Path) -> Path:
    # vehicle_dir is normally <pkg>/config/vehicles/<vehicle_id>.
    if vehicle_dir.parent.name == "vehicles" and vehicle_dir.parent.parent.name == "config":
        return vehicle_dir.parent.parent.parent
    return Path(__file__).resolve().parents[1]


def lidar_topic_from_generated_job(vehicle_dir: Path, vehicle_id: str, camera: str) -> str:
    project_root = vehicle_project_root(vehicle_dir)
    path = project_root / "config" / "generated_jobs" / vehicle_id / "{}.pipeline.yaml".format(camera)
    if not path.exists():
        return ""
    try:
        job = load_yaml(path)
    except Exception:
        return ""
    fast_calib = job.get("fast_calib", {}) if isinstance(job.get("fast_calib"), dict) else {}
    return str(fast_calib.get("lidar_topic") or "")


def build_lidar_topic_lookup(vehicle_id: str, vehicle_dir: Path, vehicle_cfg: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    sensors = vehicle_cfg.get("sensors", {}) if isinstance(vehicle_cfg.get("sensors"), dict) else {}
    for camera, cfg in sensors.items():
        topic = ""
        if isinstance(cfg, dict):
            raw = str(cfg.get("lidar_topic") or "")
            if raw.lower() not in ("", "auto", "unique"):
                topic = raw
        if not topic:
            topic = lidar_topic_from_generated_job(vehicle_dir, vehicle_id, str(camera))
        if topic:
            out[str(camera)] = topic
    return out


def resolve_data_dir(vehicle_dir: Path, vehicle_cfg: Dict[str, Any], group_node: Dict[str, Any],
                     c2c_root: Path, vehicle_id: str, group_name: str) -> Path:
    data_dir_value = (
        group_node.get("data_dir")
        or group_node.get("image_pair_dir")
        or group_node.get("pair_dir")
        or group_node.get("path")
    )
    if data_dir_value:
        data_root = vehicle_cfg.get("data_root") or vehicle_cfg.get("calib_data_root") or ""
        base = as_abs_path(data_root, vehicle_dir) if data_root else vehicle_dir
        return as_abs_path(data_dir_value, base)

    suffix = "left_pair" if group_name == "left" else "right_pair" if group_name == "right" else "{}_pair".format(group_name)
    return (c2c_root / vehicle_id / suffix).resolve()


def resolve_c2c_result(data_dir: Path, result_name: str) -> Path:
    if result_name and result_name != "auto":
        p = data_dir / "{}.yaml".format(result_name)
        if not p.exists():
            raise FileNotFoundError(str(p))
        return p

    candidates = [
        data_dir / "c2c_extrinsic_result.yaml",
        data_dir / "extrinsic_result.yaml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("No C2C result under {}. Tried: {}".format(
        data_dir, ", ".join(str(p.name) for p in candidates)
    ))


def load_c2c_groups(vehicle_id: str, vehicle_dir: Path, vehicle_cfg: Dict[str, Any],
                    c2c_root: Path, result_name: str) -> List[Dict[str, Any]]:
    c2c = vehicle_cfg.get("c2c_calibration") or vehicle_cfg.get("camera_to_camera_calibration") or {}
    groups = c2c.get("groups") if isinstance(c2c, dict) else None
    if not isinstance(groups, dict):
        return []

    out = []
    for group_name, group_node in groups.items():
        if not isinstance(group_node, dict):
            continue
        if not bool(group_node.get("enabled", True)):
            continue
        cam0 = group_node.get("cam0") or group_node.get("camera0")
        cam1 = group_node.get("cam1") or group_node.get("camera1")
        if not cam0 or not cam1:
            continue
        data_dir = resolve_data_dir(vehicle_dir, vehicle_cfg, group_node, c2c_root, vehicle_id, str(group_name))
        result_path = resolve_c2c_result(data_dir, result_name)
        result = load_yaml(result_path)
        T10 = mat4(result.get("T_cam1_cam0"), "{}.T_cam1_cam0".format(result_path))
        out.append({
            "group": str(group_name),
            "cam0": str(result.get("cam0_name") or cam0),
            "cam1": str(result.get("cam1_name") or cam1),
            "data_dir": str(data_dir),
            "result_path": str(result_path),
            "T_cam1_cam0": T10,
            "T_cam0_cam1": invert_T(T10),
            "num_pairs_used": result.get("num_pairs_used"),
            "num_pairs_all": result.get("num_pairs_all"),
            "errors_used": result.get("errors_used", {}),
            "raw": result,
        })
    return out


def direct_entry(camera: str, direct: Dict[str, Any]) -> Dict[str, Any]:
    entry = {
        "status": direct.get("status") or "ok",
        "source": "direct_lidar_camera_calibration",
        "source_camera": camera,
        "lidar_topic": direct.get("lidar_topic", ""),
        "rmse": direct.get("rmse"),
        "selected_groups": direct.get("selected_groups", []),
        "final_group_residuals": direct.get("final_group_residuals", {}),
        "source_files": direct.get("source_files", {}),
    }
    entry.update(matrix_entry(direct["T"]))
    return entry


def derived_entry(camera: str, T: np.ndarray, base_camera: str, c2c: Dict[str, Any],
                  direction: str, direct_base: Dict[str, Any]) -> Dict[str, Any]:
    if direction == "cam0_to_cam1":
        c2c_transform_name = "T_{}_from_{}".format(c2c["cam1"], c2c["cam0"])
        formula = "T_{}_lidar = T_{}_from_{} * T_{}_lidar".format(
            c2c["cam1"], c2c["cam1"], c2c["cam0"], c2c["cam0"]
        )
    else:
        c2c_transform_name = "T_{}_from_{}".format(c2c["cam0"], c2c["cam1"])
        formula = "T_{}_lidar = T_{}_from_{} * T_{}_lidar".format(
            c2c["cam0"], c2c["cam0"], c2c["cam1"], c2c["cam1"]
        )

    entry = {
        "status": "ok",
        "source": "derived_from_c2c_and_lidar_camera_calibration",
        "source_camera": camera,
        "base_camera": base_camera,
        "lidar_topic": direct_base.get("lidar_topic", ""),
        "base_camera_rmse": direct_base.get("rmse"),
        "derivation": {
            "formula": formula,
            "c2c_group": c2c["group"],
            "c2c_transform": c2c_transform_name,
            "base_transform": "T_{}_lidar".format(base_camera),
        },
        "source_files": {
            "base_direct_summary": direct_base.get("source_summary", ""),
            "base_final_extrinsic": direct_base.get("source_files", {}).get("final_extrinsic", ""),
            "c2c_result": c2c["result_path"],
        },
        "c2c_quality": {
            "num_pairs_used": c2c.get("num_pairs_used"),
            "num_pairs_all": c2c.get("num_pairs_all"),
            "errors_used": c2c.get("errors_used", {}),
        },
    }
    entry.update(matrix_entry(T))
    return entry


def consistency_entry(c2c: Dict[str, Any], direct: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cam0, cam1 = c2c["cam0"], c2c["cam1"]
    pred_cam1 = c2c["T_cam1_cam0"] @ direct[cam0]["T"]
    pred_cam0 = c2c["T_cam0_cam1"] @ direct[cam1]["T"]
    d1 = pred_cam1 @ invert_T(direct[cam1]["T"])
    d0 = pred_cam0 @ invert_T(direct[cam0]["T"])
    return {
        "group": c2c["group"],
        "cam0": cam0,
        "cam1": cam1,
        "c2c_result": c2c["result_path"],
        "cam1_predicted_from_cam0_vs_direct": {
            "rotation_delta_deg": rounded(rotation_angle_deg(d1[:3, :3]), 6),
            "translation_delta_m": rounded(np.linalg.norm(pred_cam1[:3, 3] - direct[cam1]["T"][:3, 3]), 6),
        },
        "cam0_predicted_from_cam1_vs_direct": {
            "rotation_delta_deg": rounded(rotation_angle_deg(d0[:3, :3]), 6),
            "translation_delta_m": rounded(np.linalg.norm(pred_cam0[:3, 3] - direct[cam0]["T"][:3, 3]), 6),
        },
    }


def merge_extrinsics(direct: Dict[str, Dict[str, Any]], c2c_groups: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
    merged = {camera: direct_entry(camera, info) for camera, info in sorted(direct.items())}
    checks = []
    warnings = []

    for c2c in c2c_groups:
        cam0, cam1 = c2c["cam0"], c2c["cam1"]
        has0 = cam0 in direct
        has1 = cam1 in direct

        if has0 and not has1:
            T = c2c["T_cam1_cam0"] @ direct[cam0]["T"]
            merged[cam1] = derived_entry(cam1, T, cam0, c2c, "cam0_to_cam1", direct[cam0])
            print("[DERIVE] {} from {} using {}".format(cam1, cam0, c2c["result_path"]))
        elif has1 and not has0:
            T = c2c["T_cam0_cam1"] @ direct[cam1]["T"]
            merged[cam0] = derived_entry(cam0, T, cam1, c2c, "cam1_to_cam0", direct[cam1])
            print("[DERIVE] {} from {} using {}".format(cam0, cam1, c2c["result_path"]))
        elif has0 and has1:
            checks.append(consistency_entry(c2c, direct))
            print("[CHECK] {} and {} both have direct extrinsics; wrote consistency check".format(cam0, cam1))
        else:
            warnings.append("C2C group {} skipped: neither {} nor {} has direct T_cam_lidar".format(
                c2c["group"], cam0, cam1
            ))

    return merged, checks, warnings


def write_readable(path: Path, vehicle_id: str, extrinsics: Dict[str, Any]) -> None:
    lines = [
        "vehicle_id: {}".format(vehicle_id),
        "frame_convention:",
        "  transform: T_cam_lidar",
        "  description: \"p_cam = Rcl * p_lidar + Pcl; T_cam_lidar maps LiDAR points into the camera frame.\"",
        "extrinsics:",
    ]
    for name, ext in extrinsics.items():
        lines.append("  {}:".format(name))
        lines.append("    status: {}".format(ext.get("status", "")))
        lines.append("    source: {}".format(ext.get("source", "")))
        if ext.get("base_camera"):
            lines.append("    base_camera: {}".format(ext.get("base_camera")))
        if ext.get("lidar_topic"):
            lines.append("    lidar_topic: {}".format(ext.get("lidar_topic")))
        if ext.get("rmse") is not None:
            lines.append("    rmse: {}".format(fmt(ext.get("rmse"))))
        if ext.get("base_camera_rmse") is not None:
            lines.append("    base_camera_rmse: {}".format(fmt(ext.get("base_camera_rmse"))))
        lines.append("    Pcl: [{}]".format(", ".join(fmt(v) for v in ext.get("Pcl", []))))
        lines.append("    quaternion_xyzw_cam_lidar: [{}]".format(
            ", ".join(fmt(v) for v in ext.get("quaternion_xyzw_cam_lidar", []))
        ))
        lines.append("    T_cam_lidar:")
        for row in ext.get("T_cam_lidar", []):
            lines.append("      - [{}]".format(", ".join(fmt(v) for v in row)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, extrinsics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "camera", "status", "source", "base_camera", "lidar_topic", "rmse", "base_camera_rmse",
        "tx", "ty", "tz", "qx", "qy", "qz", "qw", "roll_deg", "pitch_deg", "yaw_deg",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for camera, ext in extrinsics.items():
            q = ext.get("quaternion_xyzw_cam_lidar", ["", "", "", ""])
            rpy = ext.get("rpy_deg_zyx_roll_pitch_yaw_cam_lidar", ["", "", ""])
            t = ext.get("Pcl", ["", "", ""])
            w.writerow({
                "camera": camera,
                "status": ext.get("status", ""),
                "source": ext.get("source", ""),
                "base_camera": ext.get("base_camera", ""),
                "lidar_topic": ext.get("lidar_topic", ""),
                "rmse": ext.get("rmse", ""),
                "base_camera_rmse": ext.get("base_camera_rmse", ""),
                "tx": t[0], "ty": t[1], "tz": t[2],
                "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3],
                "roll_deg": rpy[0], "pitch_deg": rpy[1], "yaw_deg": rpy[2],
            })


def camera_frame_name(camera: str, suffix: str) -> str:
    return "{}{}".format(sanitize_frame_piece(camera), suffix)


def lidar_frame_for_extrinsic(ext: Dict[str, Any], topic_frame_map: Dict[str, str]) -> str:
    topic = str(ext.get("lidar_topic") or "")
    return topic_frame_map.get(topic, topic)


def transform_entry(T: np.ndarray) -> Dict[str, Any]:
    T = np.asarray(T, dtype=np.float64)
    T_inv = invert_T(T)
    return {
        "translation_xyz": rounded_list(T[:3, 3]),
        "quaternion_xyzw": rotmat_to_quat_xyzw(T[:3, :3]),
        "T": rounded_matrix(T),
        "T_inverse": rounded_matrix(T_inv),
    }


def static_node_xml(name: str, T_parent_child: np.ndarray, parent: str, child: str) -> str:
    t = T_parent_child[:3, 3]
    q = rotmat_to_quat_xyzw(T_parent_child[:3, :3])
    args = " ".join(
        [fmt_tf(v) for v in list(t) + list(q)] + [parent, child]
    )
    return (
        '    <node pkg="tf2_ros" type="static_transform_publisher" '
        'name="{}" args="{}" />'
    ).format(sanitize_frame_piece(name), args)


def build_tf_outputs(extrinsics: Dict[str, Any], args, vehicle_id: str) -> Tuple[Dict[str, Any], List[str], List[str]]:
    topic_frame_map = parse_lidar_topic_frame_map(args.lidar_topic_frame_map)
    base_frame, base_lidar_T = resolve_base_lidar_transforms(args, vehicle_id)
    camera_suffix = args.camera_frame_suffix
    warnings = []

    base_entries: Dict[str, Any] = {}
    tree_nodes: List[str] = [
        '<launch>',
        '    <!-- base_frame -> lidar_frame -->',
    ]
    direct_base_nodes: List[str] = [
        '<launch>',
        '    <!-- Direct base_frame -> camera_frame transforms for engineering conversion.',
        '         Do not launch together with the chained base->lidar->camera TF tree. -->',
    ]

    for lidar_frame, T_base_lidar in sorted(base_lidar_T.items()):
        tree_nodes.append(static_node_xml(
            "{}_{}_broadcaster".format(base_frame, lidar_frame),
            T_base_lidar,
            base_frame,
            lidar_frame,
        ))
    tree_nodes.append('')
    tree_nodes.append('    <!-- lidar_frame -> camera_frame -->')

    for camera, ext in extrinsics.items():
        lidar_frame = lidar_frame_for_extrinsic(ext, topic_frame_map)
        child_frame = camera_frame_name(camera, camera_suffix)
        if not lidar_frame:
            warnings.append("No lidar frame for camera {}; missing lidar_topic".format(camera))
            continue

        T_cam_lidar = mat4(ext["T_cam_lidar"], "{}.T_cam_lidar".format(camera))
        T_lidar_cam = invert_T(T_cam_lidar)
        tree_nodes.append(static_node_xml(
            "{}_{}_broadcaster".format(lidar_frame, child_frame),
            T_lidar_cam,
            lidar_frame,
            child_frame,
        ))

        entry = {
            "source": ext.get("source"),
            "lidar_topic": ext.get("lidar_topic"),
            "lidar_frame": lidar_frame,
            "camera_frame": child_frame,
            "T_lidar_camera": transform_entry(T_lidar_cam),
        }

        if lidar_frame in base_lidar_T:
            T_base_cam = base_lidar_T[lidar_frame] @ T_lidar_cam
            direct_base_nodes.append(static_node_xml(
                "{}_{}_broadcaster".format(base_frame, child_frame),
                T_base_cam,
                base_frame,
                child_frame,
            ))
            entry["base_frame"] = base_frame
            entry["T_base_camera"] = transform_entry(T_base_cam)
        else:
            warnings.append("No {} -> {} transform; skipped base output for {}".format(
                base_frame, lidar_frame, camera
            ))

        base_entries[camera] = entry

    tree_nodes.append('</launch>')
    direct_base_nodes.append('</launch>')

    obj = {
        "vehicle_id": vehicle_id,
        "base_frame": base_frame,
        "frame_convention": {
            "T_lidar_camera": "ROS TF parent=lidar_frame child=camera_frame; maps camera-frame points into lidar frame.",
            "T_base_camera": "ROS TF parent=base_frame child=camera_frame; maps camera-frame points into base frame.",
        },
        "lidar_topic_frame_map": topic_frame_map,
        "base_lidar_transforms": {
            frame: transform_entry(T)
            for frame, T in sorted(base_lidar_T.items())
        },
        "cameras": base_entries,
        "warnings": warnings,
    }
    return obj, tree_nodes, direct_base_nodes


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize all vehicle camera-to-LiDAR extrinsics, deriving missing cameras from C2C results."
    )
    parser.add_argument("--vehicle", default=DEFAULT_VEHICLE, help="Vehicle id")
    parser.add_argument("--config-root", default=DEFAULT_CONFIG_ROOT,
                        help="Root containing config/vehicles/<vehicle>/")
    parser.add_argument("--vehicle-config", default="", help="Override vehicle.yaml path")
    parser.add_argument("--direct-extrinsics", default="",
                        help="Existing direct camera/LiDAR summary YAML. Defaults to vehicle_calib_report/<vehicle>/<vehicle>_extrinsics.yaml")
    parser.add_argument("--c2c-root", default=str(DEFAULT_C2C_ROOT),
                        help="Root containing c2c/<vehicle>/<pair>/ results")
    parser.add_argument("--c2c-result-name", default=DEFAULT_C2C_RESULT_NAME,
                        help="C2C result prefix, or 'auto' to prefer c2c_extrinsic_result.yaml then extrinsic_result.yaml")
    parser.add_argument("--output-dir", default="",
                        help="Output directory. Defaults to vehicle_calib_report/<vehicle>")
    parser.add_argument("--output-prefix", default="",
                        help="Output prefix. Defaults to <vehicle>_all_camera_extrinsics")
    parser.add_argument("--emit-tf-launch", action="store_true",
                        help="Also write lidar->camera and base_link->camera static TF launch files.")
    parser.add_argument("--base-frame", default=DEFAULT_BASE_FRAME,
                        help="Base frame name used for base->lidar and base->camera outputs.")
    parser.add_argument("--base-lidar-tf-yaml", default="",
                        help="Optional YAML with base_frame and lidars mapping to xyz/quaternion_xyzw.")
    parser.add_argument("--base-lidar-first", default="",
                        help="Override base->lidar_first as: x y z qx qy qz qw.")
    parser.add_argument("--base-lidar-second", default="",
                        help="Override base->lidar_second as: x y z qx qy qz qw.")
    parser.add_argument("--lidar-topic-frame-map", default=DEFAULT_LIDAR_TOPIC_FRAME_MAP,
                        help="Comma-separated lidar_topic:tf_frame map, e.g. velodyne_first:lidar_first.")
    parser.add_argument("--camera-frame-suffix", default=DEFAULT_CAMERA_FRAME_SUFFIX,
                        help="Suffix used to build camera TF child frame names.")
    args = parser.parse_args()

    vehicle_id, vehicle_dir, vehicle_yaml, vehicle_cfg = load_vehicle_config(args)
    report_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (DEFAULT_REPORT_ROOT / vehicle_id)
    direct_path = Path(args.direct_extrinsics).expanduser().resolve() if args.direct_extrinsics else (report_dir / "{}_extrinsics.yaml".format(vehicle_id))
    c2c_root = Path(args.c2c_root).expanduser().resolve()
    prefix = args.output_prefix or "{}_all_camera_extrinsics".format(vehicle_id)

    print("[INFO] vehicle_yaml:", vehicle_yaml)
    print("[INFO] direct_extrinsics:", direct_path)
    print("[INFO] c2c_root:", c2c_root)

    direct = read_direct_extrinsics(direct_path)
    lidar_topics = build_lidar_topic_lookup(vehicle_id, vehicle_dir, vehicle_cfg)
    for camera, topic in lidar_topics.items():
        if camera in direct:
            direct[camera]["lidar_topic"] = topic
    c2c_groups = load_c2c_groups(vehicle_id, vehicle_dir, vehicle_cfg, c2c_root, args.c2c_result_name)
    extrinsics, consistency_checks, warnings = merge_extrinsics(direct, c2c_groups)
    extrinsics = dict(sorted(extrinsics.items()))

    obj = {
        "vehicle_id": vehicle_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frame_convention": {
            "transform": "T_cam_lidar",
            "description": "p_cam = Rcl * p_lidar + Pcl; T_cam_lidar maps LiDAR points into the camera frame.",
            "inverse_transform": "T_lidar_cam is also provided for consumers that need camera-to-LiDAR mapping.",
        },
        "inputs": {
            "vehicle_yaml": str(vehicle_yaml),
            "direct_extrinsics": str(direct_path),
            "c2c_result_name": args.c2c_result_name,
        },
        "extrinsics": extrinsics,
        "consistency_checks": consistency_checks,
        "warnings": warnings,
    }

    compact = {
        "vehicle_id": vehicle_id,
        "frame_convention": obj["frame_convention"],
        "extrinsics": {
            name: {
                "status": ext.get("status"),
                "source": ext.get("source"),
                "base_camera": ext.get("base_camera"),
                "lidar_topic": ext.get("lidar_topic"),
                "Rcl": ext.get("Rcl"),
                "Pcl": ext.get("Pcl"),
                "T_cam_lidar": ext.get("T_cam_lidar"),
                "T_lidar_cam": ext.get("T_lidar_cam"),
            }
            for name, ext in extrinsics.items()
        },
    }

    full_path = report_dir / "{}.yaml".format(prefix)
    compact_path = report_dir / "{}_compact.yaml".format(prefix)
    readable_path = report_dir / "{}_readable.yaml".format(prefix)
    csv_path = report_dir / "{}_summary.csv".format(prefix)
    write_yaml(full_path, obj)
    write_yaml(compact_path, compact)
    write_readable(readable_path, vehicle_id, extrinsics)
    write_csv(csv_path, extrinsics)

    print("[SAVE] full     :", full_path)
    print("[SAVE] compact  :", compact_path)
    print("[SAVE] readable :", readable_path)
    print("[SAVE] csv      :", csv_path)
    if args.emit_tf_launch:
        tf_obj, tree_launch_lines, base_launch_lines = build_tf_outputs(extrinsics, args, vehicle_id)
        tf_yaml_path = report_dir / "{}_base_camera_extrinsics.yaml".format(prefix)
        tf_launch_path = report_dir / "{}_tf.launch".format(prefix)
        base_tf_launch_path = report_dir / "{}_base_camera_tf.launch".format(prefix)
        write_yaml(tf_yaml_path, tf_obj)
        write_lines(tf_launch_path, tree_launch_lines)
        write_lines(base_tf_launch_path, base_launch_lines)
        print("[SAVE] base/tf  :", tf_yaml_path)
        print("[SAVE] launch   :", tf_launch_path)
        print("[SAVE] base lnch:", base_tf_launch_path)
        for w in tf_obj.get("warnings", []):
            print("[WARN]", w)
    if warnings:
        for w in warnings:
            print("[WARN]", w)
    print("[INFO] cameras:", ", ".join(extrinsics.keys()))


if __name__ == "__main__":
    main()
