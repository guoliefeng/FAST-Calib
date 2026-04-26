#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run four FAST-Calib pipeline jobs in one command.

Before running:
  source /opt/ros/noetic/setup.bash
  source /home/glf/dataDisk/calib/FAST-Calib_ws/devel/setup.bash

Usage:
  cd /home/glf/dataDisk/calib/FAST-Calib_ws/src/FAST-Calib
  python3 scripts/run_four_calib_jobs.py --stage all

Stage-by-stage:
  python3 scripts/run_four_calib_jobs.py --stage extract_pcd
  python3 scripts/run_four_calib_jobs.py --stage roi
  python3 scripts/run_four_calib_jobs.py --stage calibrate
  python3 scripts/run_four_calib_jobs.py --stage verify
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

import yaml

PROJECT_ROOT = Path("/home/glf/dataDisk/calib/FAST-Calib_ws/src/FAST-Calib").resolve()
PIPELINE = PROJECT_ROOT / "scripts" / "calib_pipeline.py"
JOB_DIR = PROJECT_ROOT / "config" / "jobs"
REPORT_DIR = Path("/home/glf/dataDisk/calib/four_calib_batch_report").resolve()


def first_existing(paths: List[str]) -> str:
    for p in paths:
        pp = Path(p).expanduser()
        if pp.exists():
            return str(pp.resolve())
    return str(Path(paths[0]).expanduser().resolve())


SENSORS: Dict[str, Dict[str, Any]] = {
    "front": {
        "data_dir": "/home/glf/dataDisk/calib/0425/data_221",
        "config_candidates": [
            "/home/glf/dataDisk/calib/FAST-Calib_ws/src/FAST-Calib/config/hesai0425_rearr.yaml",
            "/home/glf/dataDisk/calib/FAST-Calib_ws/src/FAST-Calib/config/hesai0425.yaml",
            "/home/glf/dataDisk/calib/FAST-Calib_ws/src/FAST-Calib/config/hesai0426_front.yaml",
        ],
        "roi": {"x_min": 0.45, "x_max": 3.40, "y_min": -4.25, "y_max": 0.10, "z_min": -1.35, "z_max": 0.45},
        "detector": {"cluster_grid": 0.28, "min_component_points": 50, "min_coarse_points": 50, "expected_height": 1.0,
                     "board_span": {"x": [0.30, 3.20], "y": [0.05, 3.00], "z": [0.30, 1.80]}},
    },
    "rear": {
        "data_dir": "/home/glf/dataDisk/calib/0426/data_221_rear",
        "config_candidates": [
            "/home/glf/dataDisk/calib/FAST-Calib_ws/src/FAST-Calib/config/hesai0426_rear.yaml",
            "/home/glf/dataDisk/calib/FAST-Calib_ws/src/FAST-Calib/config/hesai0425_rear.yaml",
        ],
        "roi": {"x_min": 0.80, "x_max": 4.60, "y_min": -5.10, "y_max": 0.20, "z_min": -1.40, "z_max": 0.55},
        "detector": {"cluster_grid": 0.28, "min_component_points": 50, "min_coarse_points": 50, "expected_height": 1.0,
                     "board_span": {"x": [0.30, 5.00], "y": [0.05, 3.50], "z": [0.30, 1.80]}},
    },
    "front_left": {
        "data_dir": "/home/glf/dataDisk/calib/0426/data_221_front_left",
        "config_candidates": ["/home/glf/dataDisk/calib/FAST-Calib_ws/src/FAST-Calib/config/hesai0426_front_left.yaml"],
        "roi": {"x_min": -2.30, "x_max": 2.40, "y_min": 1.70, "y_max": 5.90, "z_min": -1.35, "z_max": 0.45},
        "detector": {"cluster_grid": 0.28, "min_component_points": 50, "min_coarse_points": 50, "expected_height": 1.0,
                     "board_span": {"x": [0.30, 2.30], "y": [0.01, 1.70], "z": [0.30, 1.40]}},
    },
    "rear_right": {
        "data_dir": "/home/glf/dataDisk/calib/0426/data_221_rear_right",
        "config_candidates": ["/home/glf/dataDisk/calib/FAST-Calib_ws/src/FAST-Calib/config/hesai0426_rear_right.yaml"],
        "roi": {"x_min": -3.50, "x_max": 1.80, "y_min": 2.00, "y_max": 6.20, "z_min": -1.50, "z_max": 0.50},
        "detector": {"cluster_grid": 0.28, "min_component_points": 80, "min_coarse_points": 80, "expected_height": 1.0,
                     "board_span": {"x": [0.30, 5.00], "y": [0.05, 3.00], "z": [0.30, 1.80]}},
    },
}


def make_job(sensor_name: str, sensor: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_name": sensor_name,
        "data_dir": sensor["data_dir"],
        "output_dir": "${data_dir}/_calib_output",
        "layout": {
            "group_prefix": "save_data_",
            "bag_name": "1.bag",
            "image_name": "img_0001.jpg",
            "pcd_name": "1.pcd",
            "board_pcd_name": "board_candidate.pcd",
        },
        "extract_pcd": {"enabled": True, "topic_policy": "unique", "frame": "middle", "overwrite": False},
        "roi": {
            "enabled": True,
            "profile": sensor_name,
            "coarse": sensor["roi"],
            "padding": {"x": 0.15, "y": 0.15, "z": 0.15},
            "robust_quantile": [0.5, 99.5],
            "detector": sensor["detector"],
        },
        "fast_calib": {
            "package": "fast_calib",
            "calib_launch": "calib.launch",
            "config_file": first_existing(sensor["config_candidates"]),
            "pointcloud_source": "pcd",
            "max_single_rmse": 0.030,
            "max_multi_rmse": 0.030,
            "multi_min_groups": 3,
            "multi_mode": "all_centers",
        },
        "verify": {"enabled": True, "all_groups": False, "point_radius": 2},
    }


def write_job(sensor_name: str, sensor: Dict[str, Any]) -> Path:
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    path = JOB_DIR / f"auto_{sensor_name}_pipeline.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(make_job(sensor_name, sensor), f, allow_unicode=True, sort_keys=False)
    return path


def run_cmd(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def truthy_text(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "ok")


def parse_sensor_summary(sensor_name: str, sensor: Dict[str, Any]) -> Dict[str, Any]:
    data_dir = Path(sensor["data_dir"])
    output_dir = data_dir / "_calib_output"
    rows = read_csv(output_dir / "batch_summary.csv")

    multi = {}
    multi_path = output_dir / "04_multi" / "multi_result.yaml"
    if multi_path.exists():
        with multi_path.open("r", encoding="utf-8") as f:
            multi = yaml.safe_load(f) or {}

    four_center_groups = []
    failed_groups = []
    for row in rows:
        group = row.get("group", "")
        four_center_ok = truthy_text(row.get("four_center_ok", ""))
        if not four_center_ok:
            four_center_ok = row.get("lidar_centers") == "4" and row.get("qr_centers") == "4"
        if four_center_ok:
            four_center_groups.append(group)
        else:
            failed_groups.append({
                "sensor": sensor_name,
                "group": group,
                "reason": row.get("reason", ""),
                "lidar_centers": row.get("lidar_centers", ""),
                "qr_centers": row.get("qr_centers", ""),
                "rmse": row.get("rmse", ""),
                "log_path": row.get("log_path", ""),
            })

    selected_groups = multi.get("selected_groups", [])
    if not isinstance(selected_groups, list):
        selected_groups = []

    return {
        "sensor": sensor_name,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "status": multi.get("status", ""),
        "rmse": multi.get("rmse", ""),
        "four_center_count": len(four_center_groups),
        "selected_count": len(selected_groups),
        "four_center_groups": four_center_groups,
        "selected_groups": selected_groups,
        "failed_groups": failed_groups,
        "final_extrinsic": str(output_dir / "final_extrinsic.yaml"),
        "batch_summary": str(output_dir / "batch_summary.csv"),
        "multi_result": str(multi_path),
    }


def write_report(summaries: List[Dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    summary_csv = REPORT_DIR / "four_sensor_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["sensor", "data_dir", "status", "rmse", "four_center_count", "selected_count", "final_extrinsic", "batch_summary", "multi_result"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k, "") for k in fields})

    failed_csv = REPORT_DIR / "non_four_center_groups.csv"
    failed_rows = []
    for s in summaries:
        failed_rows.extend(s["failed_groups"])
    with failed_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["sensor", "group", "reason", "lidar_centers", "qr_centers", "rmse", "log_path"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failed_rows)

    report_md = REPORT_DIR / "four_sensor_report.md"
    lines = ["# Four LiDAR-Camera Calibration Report\n\n"]
    for s in summaries:
        lines.append(f"## {s['sensor']}\n\n")
        lines.append(f"- data_dir: `{s['data_dir']}`\n")
        lines.append(f"- status: `{s['status']}`\n")
        lines.append(f"- multi RMSE: `{s['rmse']}`\n")
        lines.append(f"- four-center groups: {s['four_center_count']}\n")
        lines.append(f"- selected groups: {s['selected_count']}\n")
        lines.append(f"- final extrinsic: `{s['final_extrinsic']}`\n")
        lines.append(f"- selected groups: `{', '.join(s['selected_groups'])}`\n")
        bad = [r["group"] for r in s["failed_groups"]]
        lines.append(f"- non-four-center groups: `{', '.join(bad) if bad else 'None'}`\n\n")
    lines.append(f"Failure details: `{failed_csv}`\n")
    report_md.write_text("".join(lines), encoding="utf-8")

    print("\n" + "=" * 90)
    print(f"[REPORT] {summary_csv}")
    print(f"[FAILED GROUPS] {failed_csv}")
    print(f"[REPORT MD] {report_md}")
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four FAST-Calib pipeline jobs")
    parser.add_argument("--stage", default="all", choices=["all", "scan", "extract_pcd", "roi", "calibrate", "verify"])
    parser.add_argument("--sensors", default="front,rear,front_left,rear_right")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    if not PIPELINE.exists():
        raise FileNotFoundError(f"pipeline script not found: {PIPELINE}")

    selected = [x.strip() for x in args.sensors.split(",") if x.strip()]
    for name in selected:
        if name not in SENSORS:
            raise ValueError(f"unknown sensor {name}; available={list(SENSORS.keys())}")

    job_paths = {}
    for name in selected:
        job_paths[name] = write_job(name, SENSORS[name])
        print(f"[JOB] {name}: {job_paths[name]}")

    if args.dry_run:
        print("[DRY RUN] generated job yaml files only.")
        return

    run_logs = REPORT_DIR / "run_logs"
    run_logs.mkdir(parents=True, exist_ok=True)
    exit_codes = {}

    for name in selected:
        print("\n" + "=" * 90)
        print(f"[RUN] sensor={name}, stage={args.stage}")
        print("=" * 90)
        cmd = [sys.executable, str(PIPELINE), "--job", str(job_paths[name]), "--stage", args.stage]
        code = run_cmd(cmd, run_logs / f"{name}_{args.stage}.log")
        exit_codes[name] = code
        if code != 0:
            print(f"[ERROR] {name} failed; log={run_logs / f'{name}_{args.stage}.log'}")
            if args.stop_on_error:
                sys.exit(code)

    summaries = [parse_sensor_summary(name, SENSORS[name]) for name in selected]
    write_report(summaries)

    failed = {k: v for k, v in exit_codes.items() if v != 0}
    if failed:
        print(f"[WARN] some sensors failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
