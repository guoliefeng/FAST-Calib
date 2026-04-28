#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompute FAST-Calib extrinsic from manually adjusted LiDAR circle centers.

Workflow:
  1. Run FAST-Calib normally.
  2. Run scripts/inspect_lidar_centers.py to create manual_centers.yaml files.
  3. Edit lidar_centers in manual_centers.yaml.
  4. Run this script to recompute the rigid transform.

This script writes a separate manual result and does not overwrite final_extrinsic.yaml.
"""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import yaml


def natural_key(v: str) -> List[object]:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(v))]


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> Dict:
    with path.expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, obj: Dict) -> None:
    mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def ff(x: float) -> float:
    return float(f"{float(x):.9f}")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def infer_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        return Path(args.data_dir).expanduser().resolve()

    if not args.vehicle or not args.sensor:
        raise RuntimeError("Provide either --data-dir, or both --vehicle and --sensor.")

    vehicle_file = project_root() / "config" / "vehicles" / str(args.vehicle) / "vehicle.yaml"
    if not vehicle_file.exists():
        raise FileNotFoundError(f"vehicle config not found: {vehicle_file}")

    vehicle_cfg = load_yaml(vehicle_file)
    sensor_cfg = (vehicle_cfg.get("sensors") or {}).get(args.sensor)
    if not sensor_cfg:
        raise RuntimeError(f"sensor '{args.sensor}' not found in {vehicle_file}")

    data_dir = sensor_cfg.get("data_dir")
    if not data_dir:
        raise RuntimeError(
            f"{args.vehicle}/{args.sensor} has empty data_dir. "
            "Set it in vehicle.yaml or pass --data-dir explicitly."
        )
    return Path(str(data_dir)).expanduser().resolve()


def infer_output_dir(data_dir: Path, args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser().resolve()
    return data_dir / "_calib_output"


def list_manual_files(output_dir: Path, groups_arg: str, manual_root: Path) -> List[Path]:
    if groups_arg and groups_arg.lower() not in ("all", "*"):
        out = []
        for group in [x.strip() for x in groups_arg.split(",") if x.strip()]:
            p = manual_root / group / "manual_centers.yaml"
            if not p.exists():
                raise FileNotFoundError(f"manual file not found: {p}")
            out.append(p)
        return out

    files = sorted(manual_root.glob("*/manual_centers.yaml"), key=lambda p: natural_key(p.parent.name))
    if not files:
        raise RuntimeError(f"No manual_centers.yaml files found under {manual_root}. Run inspect_lidar_centers.py first.")
    return files


def as_points(value, name: str, path: Path) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (4, 3):
        raise RuntimeError(f"{name} in {path} must be a 4x3 array, got shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name} in {path} contains non-finite values")
    return arr


def load_manual_record(path: Path) -> Tuple[str, np.ndarray, np.ndarray]:
    obj = load_yaml(path)
    if obj.get("enabled", True) is False:
        raise RuntimeError(f"manual file disabled: {path}")
    group = str(obj.get("group") or path.parent.name)
    lidar = as_points(obj.get("lidar_centers"), "lidar_centers", path)
    qr = as_points(obj.get("qr_centers"), "qr_centers", path)
    return group, lidar, qr


def svd_solve(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    ma = a.mean(axis=0)
    mb = b.mean(axis=0)
    x = a - ma
    y = b - mb
    u, _, vt = np.linalg.svd(x.T @ y)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        d = np.eye(3)
        d[2, 2] = -1
        r = vt.T @ d @ u.T
    t = mb - r @ ma
    pred = (r @ a.T).T + t
    err = np.linalg.norm(pred - b, axis=1)
    return r, t, float(np.sqrt(np.mean(err * err)))


def eval_group(lidar: np.ndarray, qr: np.ndarray, r: np.ndarray, t: np.ndarray) -> float:
    pred = (r @ lidar.T).T + t
    err = np.linalg.norm(pred - qr, axis=1)
    return float(np.sqrt(np.mean(err * err)))


def make_transform(r: np.ndarray, t: np.ndarray) -> List[List[float]]:
    mat = np.eye(4)
    mat[:3, :3] = r
    mat[:3, 3] = t
    return [[ff(x) for x in row] for row in mat.tolist()]


def write_xyz_pcd(path: Path, points: Sequence[Sequence[float]]) -> None:
    mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        n = len(points)
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {n}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")
        for p in points:
            f.write(f"{float(p[0]):.9g} {float(p[1]):.9g} {float(p[2]):.9g}\n")


def make_sphere_points(center: Sequence[float], radius: float, rings: int = 9, sectors: int = 18) -> List[List[float]]:
    cx, cy, cz = [float(x) for x in center]
    pts: List[List[float]] = []
    for i in range(rings + 1):
        theta = math.pi * i / rings
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(sectors):
            phi = 2.0 * math.pi * j / sectors
            pts.append([
                cx + radius * st * math.cos(phi),
                cy + radius * st * math.sin(phi),
                cz + radius * ct,
            ])
    return pts


def write_manual_marker_pc_ds(result_dir: Path, records: Dict[str, Tuple[np.ndarray, np.ndarray]], radius: float) -> None:
    for group, (lidar, _) in records.items():
        pts: List[List[float]] = []
        for c in lidar:
            pts.extend(make_sphere_points(c, radius))
        write_xyz_pcd(result_dir / group / "manual_lidar_center_spheres.pcd", pts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Recompute lidar-to-camera extrinsic from manually edited circle centers.")
    ap.add_argument("--vehicle", help="vehicle id, e.g. 221")
    ap.add_argument("--sensor", help="sensor name, e.g. rear_left")
    ap.add_argument("--data-dir", help="override data directory")
    ap.add_argument("--output-dir", help="override calibration output directory; default: <data_dir>/_calib_output")
    ap.add_argument("--groups", default="all", help="comma-separated groups or 'all'; default: all")
    ap.add_argument("--manual-root", help="default: <output_dir>/05_center_inspect")
    ap.add_argument("--result-dir", help="default: <output_dir>/05_manual_centers")
    ap.add_argument("--sphere-radius", type=float, default=0.035)
    args = ap.parse_args()

    data_dir = infer_data_dir(args)
    output_dir = infer_output_dir(data_dir, args)
    manual_root = Path(args.manual_root).expanduser().resolve() if args.manual_root else output_dir / "05_center_inspect"
    result_dir = mkdir(Path(args.result_dir).expanduser().resolve() if args.result_dir else output_dir / "05_manual_centers")

    records: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    source_files: Dict[str, str] = {}
    for path in list_manual_files(output_dir, args.groups, manual_root):
        group, lidar, qr = load_manual_record(path)
        records[group] = (lidar, qr)
        source_files[group] = str(path)

    if not records:
        raise RuntimeError("No valid manual records loaded.")

    all_lidar = np.vstack([records[g][0] for g in sorted(records.keys(), key=natural_key)])
    all_qr = np.vstack([records[g][1] for g in sorted(records.keys(), key=natural_key)])

    r, t, rmse = svd_solve(all_lidar, all_qr)
    group_rmse = {g: eval_group(records[g][0], records[g][1], r, t) for g in sorted(records.keys(), key=natural_key)}

    result = {
        "status": "ok",
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "manual_root": str(manual_root),
        "group_count": len(records),
        "groups": sorted(records.keys(), key=natural_key),
        "rmse": ff(rmse),
        "group_rmse": {g: ff(v) for g, v in group_rmse.items()},
        "rotation_lidar_to_camera": [[ff(x) for x in row] for row in r.tolist()],
        "translation_lidar_to_camera": [ff(x) for x in t.tolist()],
        "T_lidar_to_camera": make_transform(r, t),
        "source_manual_files": source_files,
    }

    write_yaml(result_dir / "manual_multi_result.yaml", result)
    write_yaml(result_dir / "manual_final_extrinsic.yaml", {
        "T_lidar_to_camera": result["T_lidar_to_camera"],
        "rotation_lidar_to_camera": result["rotation_lidar_to_camera"],
        "translation_lidar_to_camera": result["translation_lidar_to_camera"],
        "rmse": result["rmse"],
        "group_count": result["group_count"],
    })

    with (result_dir / "manual_group_residuals.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "manual_group_rmse", "manual_file"])
        w.writeheader()
        for g in sorted(records.keys(), key=natural_key):
            w.writerow({"group": g, "manual_group_rmse": f"{group_rmse[g]:.9f}", "manual_file": source_files[g]})

    write_manual_marker_pc_ds(result_dir, records, args.sphere_radius)

    print(f"[manual] groups={len(records)} rmse={rmse:.9f}")
    print(f"[manual] result: {result_dir / 'manual_multi_result.yaml'}")
    print(f"[manual] final : {result_dir / 'manual_final_extrinsic.yaml'}")
    print(f"[manual] csv   : {result_dir / 'manual_group_residuals.csv'}")


if __name__ == "__main__":
    main()
