#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the complete six-camera vehicle calibration workflow.

Workflow:
  1. Direct camera-LiDAR calibration for enabled sensors.
  2. Enabled left/right camera-to-camera (C2C) calibration.
  3. Six-camera extrinsic summary.
  4. Six individual ROS TF YAML files in the /home/guoli/data/227 layout.

All vehicle-specific values come from config/vehicles/<vehicle>/vehicle.yaml
and its cameras/*.yaml files.
"""

import argparse
import math
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


def find_project_root() -> Path:
    source_root = Path(__file__).resolve().parents[1]
    if (source_root / "config" / "vehicles").is_dir():
        return source_root
    try:
        import rospkg
        return Path(rospkg.RosPack().get_path("fast_calib")).resolve()
    except Exception:
        return source_root


PROJECT_ROOT = find_project_root()
CONFIG_ROOT = PROJECT_ROOT / "config" / "vehicles"
RUN_DIRECT = PROJECT_ROOT / "scripts" / "run_vehicle_calib.py"
RUN_C2C = PROJECT_ROOT / "scripts" / "c2c_calibrate_vehicle_aruco.py"
RUN_SUMMARY = PROJECT_ROOT / "scripts" / "summarize_vehicle_extrinsics.py"
RUN_TF_EXPORT = PROJECT_ROOT / "scripts" / "export_camera_extrinsics_ros_tf.py"
EXPECTED_CAMERAS = {
    "front",
    "front_left",
    "front_right",
    "rear",
    "rear_left",
    "rear_right",
}
DIRECT_CAMERAS = {"front", "front_left", "rear", "rear_right"}
EXPECTED_C2C_PAIRS = {
    "left": ("rear_left", "front_left"),
    "right": ("rear_right", "front_right"),
}
PAIR_RE = re.compile(
    r"^pair_(\d+)_cam([01])\.(?:jpg|jpeg|png|bmp|tif|tiff)$",
    re.IGNORECASE,
)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("{} must contain a YAML mapping".format(path))
    return value


def resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def sensor_output_dir(sensor: Dict[str, Any]) -> Path:
    data_dir = Path(str(sensor["data_dir"])).expanduser().resolve()
    value = str(sensor.get("output_dir", "${data_dir}/_calib_output"))
    return Path(value.replace("${data_dir}", str(data_dir))).expanduser().resolve()


def group_data_dir(vehicle_dir: Path, vehicle_cfg: Dict[str, Any],
                   group: Dict[str, Any]) -> Path:
    value = (
        group.get("data_dir")
        or group.get("image_pair_dir")
        or group.get("pair_dir")
        or group.get("path")
    )
    if not value:
        raise ValueError("C2C group must define data_dir")
    data_root = vehicle_cfg.get("data_root") or vehicle_cfg.get("calib_data_root")
    base_dir = resolve_path(data_root, vehicle_dir) if data_root else vehicle_dir
    return resolve_path(value, base_dir)


def paired_image_count(data_dir: Path) -> int:
    ids = {0: set(), 1: set()}
    for path in data_dir.iterdir():
        match = PAIR_RE.match(path.name)
        if match:
            ids[int(match.group(2))].add(match.group(1))
    return len(ids[0] & ids[1])


def enabled_c2c_groups(vehicle_cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    c2c = vehicle_cfg.get("c2c_calibration")
    if not isinstance(c2c, dict):
        raise ValueError("vehicle.yaml must contain c2c_calibration")
    groups = c2c.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("c2c_calibration.groups must be a mapping")
    order = {"left": 0, "right": 1}
    selected = [
        (str(name), value)
        for name, value in groups.items()
        if isinstance(value, dict) and bool(value.get("enabled", True))
    ]
    return sorted(selected, key=lambda item: (order.get(item[0], 99), item[0]))


def preflight(vehicle: str, vehicle_dir: Path, vehicle_cfg: Dict[str, Any],
              skip_direct: bool, skip_c2c: bool) -> List[Tuple[str, Dict[str, Any], Path]]:
    configured_id = str(vehicle_cfg.get("vehicle_id", ""))
    if configured_id and configured_id != vehicle:
        raise ValueError(
            "vehicle_id mismatch: directory={} but vehicle.yaml={}".format(vehicle, configured_id)
        )

    camera_dir = vehicle_dir / "cameras"
    missing_camera_files = sorted(
        name for name in EXPECTED_CAMERAS if not (camera_dir / "{}.yaml".format(name)).is_file()
    )
    if missing_camera_files:
        raise FileNotFoundError(
            "Missing camera YAML files: {}".format(", ".join(missing_camera_files))
        )

    sensors = vehicle_cfg.get("sensors")
    if not isinstance(sensors, dict):
        raise ValueError("vehicle.yaml must contain sensors")
    enabled = [
        (str(name), value)
        for name, value in sensors.items()
        if isinstance(value, dict) and bool(value.get("enabled", False))
    ]
    enabled_names = {name for name, _ in enabled}
    missing_direct = DIRECT_CAMERAS - enabled_names
    if missing_direct:
        raise ValueError(
            "The standard six-camera workflow requires enabled direct sensors: {}; missing={}".format(
                ", ".join(sorted(DIRECT_CAMERAS)), ", ".join(sorted(missing_direct))
            )
        )

    if not skip_direct:
        output_dirs = {}
        for name, sensor in enabled:
            data_dir = Path(str(sensor.get("data_dir", ""))).expanduser().resolve()
            if not data_dir.is_dir():
                raise FileNotFoundError("{} data_dir does not exist: {}".format(name, data_dir))
            camera_config = resolve_path(sensor.get("camera_config", ""), PROJECT_ROOT)
            if not camera_config.is_file():
                raise FileNotFoundError(
                    "{} camera_config does not exist: {}".format(name, camera_config)
                )
            roi_profile = resolve_path(sensor.get("roi_profile", ""), PROJECT_ROOT)
            if not roi_profile.is_file():
                raise FileNotFoundError(
                    "{} roi_profile does not exist: {}".format(name, roi_profile)
                )

            output_dir = sensor_output_dir(sensor)
            if output_dir in output_dirs:
                raise ValueError(
                    "{} and {} use the same output_dir: {}".format(
                        output_dirs[output_dir], name, output_dir
                    )
                )
            output_dirs[output_dir] = name

            layout = sensor.get("layout") if isinstance(sensor.get("layout"), dict) else {}
            prefix = str(layout.get("group_prefix", "save_data_"))
            bag_name = str(layout.get("bag_name", "1.bag"))
            image_name = str(layout.get("image_name", "img_0001.jpg"))
            groups = [
                path for path in sorted(data_dir.glob("{}*".format(prefix)))
                if path.is_dir() and (path / bag_name).is_file() and (path / image_name).is_file()
            ]
            if not groups:
                raise FileNotFoundError(
                    "{}: no {}*/ containing {} and {} under {}".format(
                        name, prefix, bag_name, image_name, data_dir
                    )
                )
            print("[PREFLIGHT] direct {:>11}: {} complete groups".format(name, len(groups)))

    c2c_groups = []
    c2c_cfg = vehicle_cfg.get("c2c_calibration")
    if not isinstance(c2c_cfg, dict):
        raise ValueError("vehicle.yaml must contain c2c_calibration")
    if not str(c2c_cfg.get("dictionary", "")).strip():
        raise ValueError("c2c_calibration.dictionary must be configured")
    if c2c_cfg.get("marker_id") is None:
        raise ValueError("c2c_calibration.marker_id must be configured")
    int(c2c_cfg["marker_id"])
    marker_size = float(c2c_cfg.get("marker_size_m", 0.0))
    if not math.isfinite(marker_size) or marker_size <= 0.0:
        raise ValueError("c2c_calibration.marker_size_m must be positive")

    configured_groups = enabled_c2c_groups(vehicle_cfg)
    configured_group_names = {name for name, _ in configured_groups}
    missing_c2c = set(EXPECTED_C2C_PAIRS) - configured_group_names
    if missing_c2c:
        raise ValueError(
            "The standard six-camera workflow requires enabled C2C groups left and right; missing={}".format(
                ", ".join(sorted(missing_c2c))
            )
        )

    for name, group in configured_groups:
        if name in EXPECTED_C2C_PAIRS:
            expected_cam0, expected_cam1 = EXPECTED_C2C_PAIRS[name]
            cam0 = str(group.get("cam0", ""))
            cam1 = str(group.get("cam1", ""))
            if (cam0, cam1) != (expected_cam0, expected_cam1):
                raise ValueError(
                    "C2C {} must use cam0={} and cam1={}; got cam0={}, cam1={}".format(
                        name, expected_cam0, expected_cam1, cam0, cam1
                    )
                )
        data_dir = group_data_dir(vehicle_dir, vehicle_cfg, group)
        if not data_dir.is_dir():
            raise FileNotFoundError("C2C {} data_dir does not exist: {}".format(name, data_dir))
        pair_count = paired_image_count(data_dir)
        if not skip_c2c and pair_count < 3:
            raise ValueError(
                "C2C {} has only {} complete image pairs under {}; need at least 3".format(
                    name, pair_count, data_dir
                )
            )
        print(
            "[PREFLIGHT] C2C   {:>11}: {} pairs, marker_size_m={}".format(
                name, pair_count, marker_size
            )
        )
        c2c_groups.append((name, group, data_dir))

    return c2c_groups


def command_text(command: Iterable[str]) -> str:
    return shlex.join([str(item) for item in command])


def run(command: List[str], dry_run: bool, env: Dict[str, str]) -> None:
    print("\n[COMMAND] {}".format(command_text(command)))
    if dry_run:
        return
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env)
    if result.returncode:
        raise RuntimeError(
            "Command failed with exit code {}: {}".format(
                result.returncode, command_text(command)
            )
        )


def check_direct_results(path: Path) -> None:
    data = load_yaml(path)
    sensors = data.get("sensors") or data.get("extrinsics")
    if not isinstance(sensors, dict) or not sensors:
        raise ValueError("{} contains no direct extrinsics".format(path))
    for camera, entry in sensors.items():
        if not isinstance(entry, dict) or entry.get("T_cam_lidar") is None:
            raise ValueError("{} has no T_cam_lidar for {}".format(path, camera))
        status = str(entry.get("status", "")).lower()
        if status in {"failed", "error", "missing_result", ""}:
            raise ValueError("{} has invalid status for {}: {}".format(path, camera, status))
        if status != "ok":
            print(
                "[WARN] direct {} status={}, rmse={}; inspect projection before use".format(
                    camera, status, entry.get("rmse")
                )
            )


def check_c2c_result(path: Path, vehicle: str, marker_size: float) -> None:
    data = load_yaml(path)
    result_vehicle = str(data.get("vehicle_id", ""))
    if result_vehicle and result_vehicle != vehicle:
        raise ValueError("{} vehicle_id is {}, expected {}".format(path, result_vehicle, vehicle))
    result_size = float(data.get("marker_size_m", 0.0))
    if abs(result_size - marker_size) > 1e-9:
        raise ValueError(
            "{} marker_size_m is {}, expected {}".format(path, result_size, marker_size)
        )
    used = int(data.get("num_pairs_used", 0))
    if used < 3:
        raise ValueError("{} uses only {} image pairs".format(path, used))


def check_full_summary(path: Path) -> None:
    data = load_yaml(path)
    extrinsics = data.get("extrinsics")
    if not isinstance(extrinsics, dict):
        raise ValueError("{} contains no extrinsics".format(path))
    actual = set(extrinsics)
    if actual != EXPECTED_CAMERAS:
        raise ValueError(
            "{} cameras mismatch; missing={}, extra={}".format(
                path,
                sorted(EXPECTED_CAMERAS - actual),
                sorted(actual - EXPECTED_CAMERAS),
            )
        )
    warnings = data.get("warnings")
    if warnings:
        raise ValueError("{} contains warnings: {}".format(path, warnings))


def check_tf_export(output_dir: Path) -> None:
    expected = {
        "camera-extrinsic-{}.yaml".format(camera.replace("_", "-"))
        for camera in EXPECTED_CAMERAS
    }
    actual = {path.name for path in output_dir.glob("camera-extrinsic-*.yaml")}
    if actual != expected:
        raise ValueError(
            "{} TF files mismatch; missing={}, extra={}".format(
                output_dir, sorted(expected - actual), sorted(actual - expected)
            )
        )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description="Run direct LiDAR-camera, left/right C2C, six-camera summary and TF export."
    )
    parser.add_argument("--vehicle", required=True, help="Vehicle id")
    parser.add_argument(
        "--profile",
        default="hesai_fast",
        choices=["default", "fast", "hesai_fast"],
        help="Direct calibration profile; use default for sparse LiDAR accumulation.",
    )
    parser.add_argument(
        "--report-root",
        default=os.environ.get(
            "FAST_CALIB_REPORT_ROOT", str(PROJECT_ROOT / "vehicle_calib_report")
        ),
        help="Vehicle report root",
    )
    parser.add_argument(
        "--export-dir",
        default="",
        help="227-style six-file output directory; defaults to ~/data/<vehicle>",
    )
    parser.add_argument(
        "--skip-direct",
        action="store_true",
        help="Reuse existing direct camera-LiDAR results.",
    )
    parser.add_argument(
        "--skip-c2c",
        action="store_true",
        help="Reuse existing left/right C2C results.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Generate the six-camera summary but not individual ROS TF YAML files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configured inputs and print commands without running them.",
    )
    args = parser.parse_args()

    vehicle = str(args.vehicle)
    vehicle_dir = CONFIG_ROOT / vehicle
    vehicle_yaml = vehicle_dir / "vehicle.yaml"
    if not vehicle_yaml.is_file():
        raise FileNotFoundError("Vehicle config does not exist: {}".format(vehicle_yaml))
    vehicle_cfg = load_yaml(vehicle_yaml)
    c2c_groups = preflight(
        vehicle, vehicle_dir, vehicle_cfg, args.skip_direct, args.skip_c2c
    )

    report_root = Path(args.report_root).expanduser().resolve()
    report_dir = report_root / vehicle
    direct_summary = report_dir / "{}_extrinsics.yaml".format(vehicle)
    full_summary = report_dir / "{}_all_camera_extrinsics.yaml".format(vehicle)
    export_dir = (
        Path(args.export_dir).expanduser().resolve()
        if args.export_dir
        else (Path.home() / "data" / vehicle).resolve()
    )
    child_env = dict(os.environ)
    child_env["FAST_CALIB_REPORT_ROOT"] = str(report_root)

    if not args.skip_direct:
        run(
            [
                sys.executable,
                str(RUN_DIRECT),
                "--vehicle",
                vehicle,
                "--profile",
                args.profile,
                "--stage",
                "all",
                "--stop-on-error",
            ],
            args.dry_run,
            child_env,
        )
    elif not args.dry_run:
        print("[REUSE] direct camera-LiDAR results:", direct_summary)

    if not args.dry_run:
        check_direct_results(direct_summary)

    c2c_cfg = vehicle_cfg["c2c_calibration"]
    marker_size = float(c2c_cfg["marker_size_m"])
    result_name = str(c2c_cfg.get("result_name", "c2c_extrinsic_result"))
    for group_name, group, data_dir in c2c_groups:
        result_path = data_dir / "{}.yaml".format(result_name)
        if not args.skip_c2c:
            max_01 = float(group.get("max_cross_0_to_1_mean_px", 1.5))
            max_10 = float(group.get("max_cross_1_to_0_mean_px", 5.0))
            min_pairs = int(group.get("min_cross_filter_pairs", 3))
            run(
                [
                    sys.executable,
                    str(RUN_C2C),
                    "--vehicle",
                    vehicle,
                    "--config-root",
                    str(CONFIG_ROOT),
                    "--group",
                    group_name,
                    "--result-name",
                    result_name,
                    "--max-cross-0-to-1-mean",
                    str(max_01),
                    "--max-cross-1-to-0-mean",
                    str(max_10),
                    "--min-cross-filter-pairs",
                    str(min_pairs),
                    "--save-debug",
                    "--save-validation",
                ],
                args.dry_run,
                child_env,
            )
        elif not args.dry_run:
            print("[REUSE] C2C {} result: {}".format(group_name, result_path))
        if not args.dry_run:
            check_c2c_result(result_path, vehicle, marker_size)

    run(
        [
            sys.executable,
            str(RUN_SUMMARY),
            "--vehicle",
            vehicle,
            "--config-root",
            str(CONFIG_ROOT),
            "--direct-extrinsics",
            str(direct_summary),
            "--c2c-root",
            str(vehicle_dir),
            "--c2c-result-name",
            result_name,
            "--output-dir",
            str(report_dir),
            "--output-prefix",
            "{}_all_camera_extrinsics".format(vehicle),
        ],
        args.dry_run,
        child_env,
    )

    if not args.dry_run:
        check_full_summary(full_summary)

    if not args.skip_export:
        run(
            [
                sys.executable,
                str(RUN_TF_EXPORT),
                str(full_summary),
                str(export_dir),
            ],
            args.dry_run,
            child_env,
        )
        if not args.dry_run:
            check_tf_export(export_dir)

    if args.dry_run:
        print("\n[DRY RUN] Inputs passed preflight; no calibration command was executed.")
        return

    print("\n[DONE] vehicle:", vehicle)
    print("[DONE] direct summary:", direct_summary)
    print("[DONE] six-camera summary:", full_summary)
    if not args.skip_export:
        print("[DONE] ROS TF YAML directory:", export_dir)


if __name__ == "__main__":
    main()
