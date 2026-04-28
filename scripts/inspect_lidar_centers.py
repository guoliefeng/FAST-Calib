#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inspect LiDAR circle centers produced by FAST-Calib.

This tool does not change calibration results. It reads FAST-Calib outputs:
  <data_dir>/_calib_output/03_calibrate/<group>/circle_center_record.txt
and exports visual marker point clouds:
  <data_dir>/_calib_output/05_center_inspect/<group>/*.pcd

Typical usage:
  python3 scripts/inspect_lidar_centers.py --vehicle 221 --sensor rear_left
  python3 scripts/inspect_lidar_centers.py --data-dir /path/to/data_221_rear_left --groups save_data_001
"""

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

CENTER_RE = re.compile(r"\{([^}]*)\}")


def natural_key(v: str) -> List[object]:
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(v))]


def load_yaml(path: Path) -> Dict:
    with path.expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_centers_from_line(line: str) -> List[List[float]]:
    centers: List[List[float]] = []
    for raw in CENTER_RE.findall(line):
        vals = [float(x.strip()) for x in raw.split(",")]
        if len(vals) == 3:
            centers.append(vals)
    return centers


def read_circle_record(path: Path) -> Tuple[List[List[float]], List[List[float]]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lidar_line = next((l for l in reversed(lines) if l.startswith("lidar_centers:")), "")
    qr_line = next((l for l in reversed(lines) if l.startswith("qr_centers:")), "")
    lidar = parse_centers_from_line(lidar_line)
    qr = parse_centers_from_line(qr_line)
    if len(lidar) != 4 or len(qr) != 4:
        raise RuntimeError(f"Expected 4 lidar centers and 4 qr centers in {path}, got lidar={len(lidar)}, qr={len(qr)}")
    return lidar, qr


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


def make_cross_points(center: Sequence[float], half_size: float, samples_per_axis: int = 25) -> List[List[float]]:
    cx, cy, cz = [float(x) for x in center]
    pts: List[List[float]] = []
    denom = max(samples_per_axis - 1, 1)
    for axis in range(3):
        for i in range(samples_per_axis):
            t = -half_size + 2.0 * half_size * i / denom
            p = [cx, cy, cz]
            p[axis] += t
            pts.append(p)
    return pts


def write_centers_csv(path: Path, centers: Sequence[Sequence[float]]) -> None:
    mkdir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["center_index", "x", "y", "z"])
        w.writeheader()
        for i, c in enumerate(centers):
            w.writerow({
                "center_index": i,
                "x": f"{float(c[0]):.9f}",
                "y": f"{float(c[1]):.9f}",
                "z": f"{float(c[2]):.9f}",
            })


def write_manual_template(path: Path,
                          group: str,
                          source_record: Path,
                          source_cloud: Optional[Path],
                          lidar: Sequence[Sequence[float]],
                          qr: Sequence[Sequence[float]]) -> None:
    mkdir(path.parent)
    lines: List[str] = []
    lines.append("# Manual LiDAR circle center override for FAST-Calib.")
    lines.append("# Edit only lidar_centers unless you are certain the image-side qr_centers are wrong.")
    lines.append("# Keep the order 0..3 consistent with the original FAST-Calib record.")
    lines.append("# Units are meters in the LiDAR coordinate frame.")
    lines.append(f"group: {group}")
    lines.append(f"source_record: {source_record}")
    if source_cloud:
        lines.append(f"source_cloud: {source_cloud}")
    lines.append("enabled: true")
    lines.append("lidar_centers:")
    for i, c in enumerate(lidar):
        lines.append(f"  - [{float(c[0]):.9f}, {float(c[1]):.9f}, {float(c[2]):.9f}]  # center_{i}")
    lines.append("qr_centers:")
    for i, c in enumerate(qr):
        lines.append(f"  - [{float(c[0]):.9f}, {float(c[1]):.9f}, {float(c[2]):.9f}]  # keep from original record, center_{i}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


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


def list_groups(data_dir: Path, output_dir: Path, groups_arg: str) -> List[str]:
    if groups_arg and groups_arg.lower() not in ("all", "*"):
        return [x.strip() for x in groups_arg.split(",") if x.strip()]

    candidates = set()
    cal_dir = output_dir / "03_calibrate"
    if cal_dir.exists():
        for p in cal_dir.glob("*/circle_center_record.txt"):
            candidates.add(p.parent.name)

    if not candidates:
        for d in data_dir.glob("save_data_*"):
            if d.is_dir():
                candidates.add(d.name)

    if not candidates:
        raise RuntimeError(f"No groups found under {data_dir} or {cal_dir}")

    return sorted(candidates, key=natural_key)


def find_record(output_dir: Path, group: str) -> Path:
    direct = output_dir / "03_calibrate" / group / "circle_center_record.txt"
    if direct.exists():
        return direct

    matches = sorted(output_dir.glob(f"**/{group}/circle_center_record.txt"), key=lambda p: len(str(p)))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"circle_center_record.txt not found for group={group} under {output_dir}")


def find_cloud(data_dir: Path, output_dir: Path, group: str) -> Optional[Path]:
    candidates = [
        data_dir / group / "board_candidate.pcd",
        output_dir / "02_roi" / "board_candidates" / f"{group}_board_candidate.pcd",
        data_dir / group / "1.pcd",
        output_dir / "02_roi" / "debug" / f"{group}_roi_box.pcd",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def write_cloudcompare_script(path: Path, cloud: Optional[Path], sphere_pcd: Path, cross_pcd: Path) -> None:
    lines = ["#!/usr/bin/env bash", "set -e"]
    cmd = ["CloudCompare"]
    if cloud:
        cmd += ["-O", f"\"{cloud}\""]
    cmd += ["-O", f"\"{sphere_pcd}\"", "-O", f"\"{cross_pcd}\""]
    lines.append(" ".join(cmd))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def inspect_group(data_dir: Path, output_dir: Path, group: str, args: argparse.Namespace) -> Dict[str, str]:
    record = find_record(output_dir, group)
    cloud = find_cloud(data_dir, output_dir, group)
    lidar, qr = read_circle_record(record)

    group_out = mkdir(output_dir / "05_center_inspect" / group)
    centers_pcd = group_out / "lidar_centers.pcd"
    spheres_pcd = group_out / "lidar_center_spheres.pcd"
    crosses_pcd = group_out / "lidar_center_crosses.pcd"
    centers_csv = group_out / "lidar_centers.csv"
    manual_yaml = group_out / "manual_centers.yaml"
    cc_script = group_out / "open_in_cloudcompare.sh"

    sphere_pts: List[List[float]] = []
    cross_pts: List[List[float]] = []
    for c in lidar:
        sphere_pts.extend(make_sphere_points(c, args.sphere_radius, args.sphere_rings, args.sphere_sectors))
        cross_pts.extend(make_cross_points(c, args.cross_half_size, args.cross_samples))

    write_xyz_pcd(centers_pcd, lidar)
    write_xyz_pcd(spheres_pcd, sphere_pts)
    write_xyz_pcd(crosses_pcd, cross_pts)
    write_centers_csv(centers_csv, lidar)
    write_manual_template(manual_yaml, group, record, cloud, lidar, qr)
    write_cloudcompare_script(cc_script, cloud, spheres_pcd, crosses_pcd)

    return {
        "group": group,
        "record": str(record),
        "cloud": str(cloud) if cloud else "",
        "centers_pcd": str(centers_pcd),
        "spheres_pcd": str(spheres_pcd),
        "crosses_pcd": str(crosses_pcd),
        "manual_yaml": str(manual_yaml),
        "cloudcompare_script": str(cc_script),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export visual LiDAR circle-center markers for FAST-Calib results.")
    ap.add_argument("--vehicle", help="vehicle id, e.g. 221")
    ap.add_argument("--sensor", help="sensor name, e.g. rear_left")
    ap.add_argument("--data-dir", help="override data directory; useful when vehicle.yaml has empty data_dir")
    ap.add_argument("--output-dir", help="override calibration output directory; default: <data_dir>/_calib_output")
    ap.add_argument("--groups", default="all", help="comma-separated groups or 'all'; default: all")
    ap.add_argument("--sphere-radius", type=float, default=0.035, help="visual marker sphere radius in meters")
    ap.add_argument("--sphere-rings", type=int, default=9)
    ap.add_argument("--sphere-sectors", type=int, default=18)
    ap.add_argument("--cross-half-size", type=float, default=0.10, help="visual cross half size in meters")
    ap.add_argument("--cross-samples", type=int, default=25)
    args = ap.parse_args()

    data_dir = infer_data_dir(args)
    output_dir = infer_output_dir(data_dir, args)
    groups = list_groups(data_dir, output_dir, args.groups)

    rows: List[Dict[str, str]] = []
    for group in groups:
        print(f"[inspect] group={group}")
        try:
            row = inspect_group(data_dir, output_dir, group, args)
            rows.append(row)
            print(f"  centers: {row['spheres_pcd']}")
            print(f"  manual : {row['manual_yaml']}")
            print(f"  open   : {row['cloudcompare_script']}")
        except Exception as e:
            print(f"  [FAILED] {e}")
            rows.append({"group": group, "error": str(e)})

    summary = mkdir(output_dir / "05_center_inspect") / "inspect_summary.csv"
    keys = sorted({k for row in rows for k in row.keys()})
    with summary.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[summary] {summary}")


if __name__ == "__main__":
    main()
