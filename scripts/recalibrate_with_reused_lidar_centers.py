#!/usr/bin/env python3
"""Recalibrate camera-LiDAR extrinsics with new intrinsics and saved LiDAR centers.

This script intentionally does not read bags, PCDs, or invoke the LiDAR detector.
For every valid prior ``circle_center_record.txt`` it retains ``lidar_centers``
exactly, recomputes only image-side QR circle centers using the current camera
intrinsics, then performs the same four-center permutation search and SVD merge
used by ``calib_pipeline.py``.
"""

import argparse
import csv
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

from calibrate_ariy_batch import detect_qr_centers, sort_pattern_centers
from calib_pipeline import (
    choose_permutation_search,
    load_pcd_xyz,
    natural_key,
    read_record,
    write_yaml,
)


def ff(value):
    return float(f"{float(value):.9f}")


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_groups(value):
    if value.lower() in ("all", "*"):
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def center_text(points):
    return "".join(" {{{:.9f},{:.9f},{:.9f}}}".format(*point) for point in points)


def camera_matrix(config):
    k = np.array(
        [[config["fx"], 0.0, config["cx"]], [0.0, config["fy"], config["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.array(
        [
            config.get("k1", 0.0), config.get("k2", 0.0), config.get("p1", 0.0), config.get("p2", 0.0),
            config.get("k3", 0.0), config.get("k4", 0.0), config.get("k5", 0.0), config.get("k6", 0.0),
        ],
        dtype=np.float64,
    )
    if not np.any(np.abs(distortion[4:]) > 1e-12):
        distortion = distortion[:5]
    return k, distortion


def board_pcd(group_dir, previous_output, group):
    candidates = [
        group_dir / "board_candidate.pcd",
        previous_output / "02_roi" / "board_candidates" / f"{group}_board_candidate.pcd",
        previous_output / "03_single" / group / "colored_cloud.pcd",
    ]
    return next((path for path in candidates if path.exists()), None)


def write_projection(image_path, points_path, transform, k, distortion, output_path, radius):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return 0, "image_not_found"
    if points_path is None:
        return 0, "board_pcd_not_found"
    points = load_pcd_xyz(points_path)
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    camera_points = (transform @ points_h.T).T[:, :3]
    camera_points = camera_points[camera_points[:, 2] > 0.05]
    if not len(camera_points):
        return 0, "all_points_behind_camera"
    uv, _ = cv2.projectPoints(camera_points.reshape(-1, 1, 3), np.zeros((3, 1)), np.zeros((3, 1)), k, distortion)
    uv = uv.reshape(-1, 2)
    height, width = image.shape[:2]
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    uv = uv[mask]
    for u, v in uv:
        cv2.circle(image, (int(round(u)), int(round(v))), radius, (0, 0, 255), -1)
    cv2.imwrite(str(output_path), image)
    return int(len(uv)), "ok" if len(uv) else "all_projected_points_outside_image"


def main():
    parser = argparse.ArgumentParser(
        description="Recompute image-side centers with current intrinsics while reusing saved LiDAR centers."
    )
    parser.add_argument("--data-dir", required=True, help="Dataset directory containing calibration groups.")
    parser.add_argument("--camera-config", required=True, help="Current camera intrinsics YAML.")
    parser.add_argument("--previous-output-dir", default="", help="Prior output; default: <data-dir>/_calib_output.")
    parser.add_argument("--output-dir", default="", help="New output; default: <data-dir>/_calib_output_intrinsics_fixed_lidar.")
    parser.add_argument("--groups", default="all", help="Comma-separated calibration groups or all.")
    parser.add_argument("--max-multi-rmse", type=float, default=None)
    parser.add_argument("--max-group-rmse", type=float, default=None)
    parser.add_argument("--min-groups", type=int, default=None)
    parser.add_argument("--point-radius", type=int, default=2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    previous_output = Path(args.previous_output_dir).expanduser().resolve() if args.previous_output_dir else data_dir / "_calib_output"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "_calib_output_intrinsics_fixed_lidar"
    camera_config = Path(args.camera_config).expanduser().resolve()
    if not previous_output.exists():
        raise FileNotFoundError(f"Previous output does not exist: {previous_output}")
    if not camera_config.exists():
        raise FileNotFoundError(f"Camera config does not exist: {camera_config}")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")

    old_final = load_yaml(previous_output / "final_extrinsic.yaml")
    max_multi = float(args.max_multi_rmse if args.max_multi_rmse is not None else old_final.get("max_multi_rmse", 0.03))
    max_group = float(args.max_group_rmse if args.max_group_rmse is not None else old_final.get("max_group_rmse", max(2.0 * max_multi, 0.06)))
    min_groups = int(args.min_groups if args.min_groups is not None else 3)
    config = load_yaml(camera_config)
    wanted = parse_groups(args.groups)
    previous_single = previous_output / "03_single"
    recorded_group_names = {
        record.parent.name
        for record in previous_single.glob("*/circle_center_record.txt")
        if record.is_file()
    }
    group_dirs = sorted(
        (data_dir / name for name in recorded_group_names if (data_dir / name).is_dir()),
        key=lambda path: natural_key(path.name),
    )
    if not group_dirs:
        # Compatibility fallback for older outputs without a 03_single index.
        group_dirs = sorted(
            (path for path in data_dir.glob("save_data_*") if path.is_dir()),
            key=lambda path: natural_key(path.name),
        )
    if wanted is not None:
        available = {path.name for path in group_dirs}
        missing = sorted(wanted - available, key=natural_key)
        if missing:
            raise FileNotFoundError("Dataset groups not found: " + ", ".join(missing))
        group_dirs = [path for path in group_dirs if path.name in wanted]
    if not group_dirs:
        raise RuntimeError(
            f"No calibration groups with saved circle_center_record.txt found in {data_dir}"
        )

    output_dir.mkdir(parents=True)
    old_roi = previous_output / "02_roi"
    if old_roi.exists():
        shutil.copytree(old_roi, output_dir / "02_roi")
    single_dir = output_dir / "03_single"
    verify_dir = output_dir / "05_verify"
    multi_dir = output_dir / "04_multi"
    single_dir.mkdir(parents=True)
    verify_dir.mkdir(parents=True)
    multi_dir.mkdir(parents=True)

    records = {}
    source_records = {}
    rows = []
    for group_dir in group_dirs:
        group = group_dir.name
        source_record = previous_output / "03_single" / group / "circle_center_record.txt"
        lidar, _ = read_record(source_record)
        row = {"group": group, "source_record": str(source_record), "status": "", "reason": "", "lidar_centers_reused": 0}
        if lidar is None:
            row.update(status="skipped", reason="previous_four_lidar_centers_not_found")
            rows.append(row)
            continue
        image_path = group_dir / "img_0001.jpg"
        group_output = single_dir / group
        group_output.mkdir(parents=True)
        try:
            qr = sort_pattern_centers(detect_qr_centers(str(image_path), config, str(group_output / "qr_detect.png")), "camera")
        except Exception as error:
            row.update(status="failed", reason=f"qr_detection_failed: {error}")
            rows.append(row)
            continue
        record_path = group_output / "circle_center_record.txt"
        record_path.write_text(
            "# LiDAR centers reused exactly from previous calibration; QR centers extracted with current camera intrinsics.\n"
            f"source_record: {source_record}\n"
            f"lidar_centers:{center_text(lidar)}\n"
            f"qr_centers:{center_text(qr)}\n",
            encoding="utf-8",
        )
        records[group] = (lidar, qr)
        source_records[group] = str(source_record)
        row.update(status="ok", reason="", lidar_centers_reused=4, output_record=str(record_path), image=str(image_path))
        rows.append(row)

    if len(records) < min_groups:
        raise RuntimeError(f"Only {len(records)} valid groups; need at least {min_groups}.")
    selected, candidates, _ = choose_permutation_search(records, min_groups, max_multi, max_group)
    if selected is None:
        raise RuntimeError("Permutation search did not return a calibration candidate.")
    transform = np.eye(4)
    transform[:3, :3] = selected["R"]
    transform[:3, 3] = selected["t"]
    per_all = selected.get("per_group_all", {})
    selected_groups = selected["groups"]

    final = {
        "status": "ok" if selected["rmse"] <= max_multi and selected["worst_rmse"] <= max_group else "warn",
        "rmse": ff(selected["rmse"]),
        "max_multi_rmse": ff(max_multi),
        "max_group_rmse": ff(max_group),
        "multi_mode": "permutation_search",
        "selection_policy": "reused_lidar_centers_permutation_search",
        "pointcloud_processing": "reused_previous_circle_centers",
        "camera_intrinsics": str(camera_config),
        "source_output": str(previous_output),
        "four_center_groups": sorted(records, key=natural_key),
        "usable_four_center_groups": sorted(records, key=natural_key),
        "selected_groups": selected_groups,
        "selected_group_residuals": {group: ff(selected["per_group"][group]) for group in selected_groups},
        "final_group_residuals": {group: ff(per_all[group]) for group in sorted(per_all, key=natural_key)},
        "T_cam_lidar": [[ff(value) for value in row] for row in transform.tolist()],
        "Rcl": [[ff(value) for value in row] for row in selected["R"].tolist()],
        "Pcl": [ff(value) for value in selected["t"].tolist()],
        "permutation_note": "qr_centers are reordered per group before joint SVD; tuple means reordered_qr = original_qr[tuple].",
        "selected_qr_permutations": {group: [int(value) for value in selected["permutations"][group]] for group in selected_groups},
        "best_qr_permutations_all_groups": {
            group: [int(value) for value in selected["best_permutations_all"][group]]
            for group in sorted(selected["best_permutations_all"], key=natural_key)
        },
        "source_circle_center_records": source_records,
    }
    write_yaml(output_dir / "final_extrinsic.yaml", final)
    write_yaml(multi_dir / "multi_result.yaml", final)
    (multi_dir / "selected_groups.txt").write_text("\n".join(selected_groups) + "\n", encoding="utf-8")

    with (output_dir / "batch_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (multi_dir / "group_residuals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "selected", "rmse", "source_record"])
        writer.writeheader()
        for group in sorted(records, key=natural_key):
            writer.writerow({"group": group, "selected": group in selected_groups, "rmse": f"{per_all[group]:.9f}", "source_record": source_records[group]})

    k, distortion = camera_matrix(config)
    verify_rows = []
    for group in selected_groups:
        group_dir = data_dir / group
        points_path = board_pcd(group_dir, previous_output, group)
        point_count, reason = write_projection(
            group_dir / "img_0001.jpg", points_path, transform, k, distortion,
            verify_dir / f"{group}_board_overlay.png", args.point_radius,
        )
        verify_rows.append({"group": group, "status": "ok" if reason == "ok" else "warn", "reason": reason, "points_projected": point_count, "board_pcd": str(points_path or "")})
    with (verify_dir / "verify_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["group", "status", "reason", "points_projected", "board_pcd"])
        writer.writeheader()
        writer.writerows(verify_rows)

    print(f"[reused-lidar] valid_groups={len(records)} selected={selected_groups} rmse={selected['rmse']:.6f}")
    print(f"[reused-lidar] final={output_dir / 'final_extrinsic.yaml'}")
    print(f"[reused-lidar] verification={verify_dir / 'verify_summary.csv'}")


if __name__ == "__main__":
    main()
