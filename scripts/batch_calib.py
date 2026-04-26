#!/usr/bin/env python3
import os
import csv
import itertools
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import rospy
import rospkg
import yaml


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RMSE_RE = re.compile(r"\[Result\]\s+RMSE:\s+([0-9.+\-eE]+)")
POINTS_RE = re.compile(r"Loaded\s+([0-9]+)\s+points")
FILTERED_RE = re.compile(r"(?:Depth filtered cloud size|Filtered cloud size):\s+([0-9]+)")
PLANE_RE = re.compile(r"Plane cloud size:\s+([0-9]+)")
EDGE_RE = re.compile(r"Extracted\s+([0-9]+)\s+edge points")
CANDIDATES_RE = re.compile(r"Circle candidates found:\s+([0-9]+)")
MAIN_COUNT_RE = re.compile(r"got lidar=([0-9]+), qr=([0-9]+)")
NEED_COUNT_RE = re.compile(r"Need 4 LiDAR centers and 4 QR centers.*got lidar=([0-9]+), qr=([0-9]+)")
SELECTED_RE = re.compile(
    r"Selected circle group .*rmse=([0-9.+\-eE]+), geom_score=([0-9.+\-eE]+), geom_valid=(true|false), support=([0-9]+)"
)
CENTER_RE = re.compile(r"\{([^}]*)\}")


def strip_ansi(text):
    return ANSI_RE.sub("", text)


def expand_ros_find(value):
    if not isinstance(value, str):
        return value
    rospack = rospkg.RosPack()
    pattern = re.compile(r"\$\(find\s+([^)]+)\)")

    def repl(match):
        return rospack.get_path(match.group(1).strip())

    return pattern.sub(repl, value)


def parse_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [item.strip() for item in str(value).split(",") if item.strip()]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def parse_float(value, default):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_int(value, default):
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_sources(pointcloud_source, pointcloud_sources):
    if pointcloud_source not in (None, ""):
        raw = str(pointcloud_source).strip().lower()
        if raw in ("both", "all"):
            return ["bag", "pcd"]
        return [item.lower() for item in parse_list(raw)]

    sources = parse_list(pointcloud_sources)
    if not sources:
        sources = ["bag"]
    out = []
    for source in sources:
        raw = str(source).strip().lower()
        if raw in ("both", "all"):
            out.extend(["bag", "pcd"])
        elif raw:
            out.append(raw)
    return list(dict.fromkeys(out))


def natural_key(value):
    parts = re.split(r"(\d+)", str(value))
    return [int(p) if p.isdigit() else p for p in parts]


def resolve_path(value, base_dir=None):
    value = expand_ros_find(value)
    if not isinstance(value, str) or not value:
        return value
    path = Path(value).expanduser()
    if base_dir and not path.is_absolute():
        path = Path(base_dir).expanduser() / path
    return str(path)


def load_batch_yaml(path):
    if not path:
        return {}
    yaml_path = Path(path).expanduser()
    if not yaml_path.exists():
        rospy.logwarn("Batch ROI yaml does not exist: %s", yaml_path)
        return {}
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        rospy.logwarn("Batch ROI yaml is not a dictionary: %s", yaml_path)
        return {}
    batch = data.get("batch_calib", data)
    if not isinstance(batch, dict):
        rospy.logwarn("Batch ROI yaml has no dictionary batch_calib block: %s", yaml_path)
        return {}
    rospy.loginfo("Loaded batch ROI yaml: %s", yaml_path)
    return batch


def get_batch_config():
    cfg = rospy.get_param("/batch_calib", {})
    if not isinstance(cfg, dict):
        cfg = {}

    def private_or_cfg(name, default=None):
        private_value = rospy.get_param("~" + name, None)
        if private_value not in (None, ""):
            return private_value
        return cfg.get(name, default)

    config_file = expand_ros_find(private_or_cfg("config_file", ""))
    data_dir = resolve_path(private_or_cfg("data_dir", cfg.get("data_dir", "")))
    output_path = resolve_path(private_or_cfg("output_path", cfg.get("output_path", "")))
    output_dir_name = str(private_or_cfg("output_dir_name", cfg.get("output_dir_name", "fast_calib_output")))
    roi_file = private_or_cfg("roi_file", cfg.get("roi_file", ""))
    roi_file_path = resolve_path(roi_file, data_dir) if roi_file else ""
    roi_cfg = load_batch_yaml(roi_file_path) if roi_file_path else {}
    use_config_groups = parse_bool(private_or_cfg("use_config_groups", cfg.get("use_config_groups", True)))
    private_groups = rospy.get_param("~groups", None)
    if private_groups not in (None, ""):
        groups = parse_list(private_groups)
    elif use_config_groups:
        groups = parse_list(cfg.get("groups", []))
    else:
        groups = []
    private_source = rospy.get_param("~pointcloud_source", None)
    private_sources = rospy.get_param("~pointcloud_sources", None)
    if private_source not in (None, ""):
        pointcloud_source = private_source
        pointcloud_sources = []
    elif private_sources not in (None, ""):
        pointcloud_source = ""
        pointcloud_sources = private_sources
    else:
        pointcloud_source = cfg.get("pointcloud_source", "")
        pointcloud_sources = cfg.get("pointcloud_sources", ["bag"])
    sources = parse_sources(pointcloud_source, pointcloud_sources)

    run_multi = parse_bool(private_or_cfg("run_multi", cfg.get("run_multi", False)))
    continue_on_error = parse_bool(private_or_cfg("continue_on_error", cfg.get("continue_on_error", True)))
    pcd_prefer_group_dir = parse_bool(private_or_cfg("pcd_prefer_group_dir", cfg.get("pcd_prefer_group_dir", True)))
    max_single_rmse = parse_float(private_or_cfg("max_single_rmse", cfg.get("max_single_rmse", 0.05)), 0.05)
    multi_min_groups = parse_int(private_or_cfg("multi_min_groups", cfg.get("multi_min_groups", 3)), 3)
    multi_mode = str(private_or_cfg("multi_mode", cfg.get("multi_mode", "best"))).lower()
    lidar_topic = private_or_cfg("lidar_topic", rospy.get_param("/lidar_topic", "/velodyne_first"))
    use_config_rois = parse_bool(private_or_cfg("use_config_rois", cfg.get("use_config_rois", True)))
    group_layout = str(private_or_cfg("group_layout", cfg.get("group_layout", "flat"))).strip().lower()
    group_dir_prefix = str(private_or_cfg("group_dir_prefix", cfg.get("group_dir_prefix", "save_data_")))
    bag_name = str(private_or_cfg("bag_name", cfg.get("bag_name", "1.bag")))
    image_name = str(private_or_cfg("image_name", cfg.get("image_name", "img_0001.jpg")))
    pcd_name = str(private_or_cfg("pcd_name", cfg.get("pcd_name", "")))
    rois = roi_cfg.get("rois", cfg.get("rois", {}))
    if not isinstance(rois, dict):
        rois = {}
    if not use_config_rois:
        rois = {}
    default_roi = roi_cfg.get("default_roi", cfg.get("default_roi", {}))
    if not isinstance(default_roi, dict):
        default_roi = {}
    for key in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]:
        private_value = rospy.get_param("~" + key, None)
        if private_value not in (None, ""):
            default_roi[key] = private_value
        elif key not in default_roi:
            global_value = rospy.get_param("/" + key, None)
            if global_value not in (None, ""):
                default_roi[key] = global_value

    return {
        "config_file": config_file,
        "data_dir": data_dir,
        "output_path": output_path,
        "output_dir_name": output_dir_name,
        "roi_file": roi_file,
        "roi_file_path": roi_file_path,
        "groups": groups,
        "use_config_groups": use_config_groups,
        "sources": sources,
        "run_multi": run_multi,
        "continue_on_error": continue_on_error,
        "pcd_prefer_group_dir": pcd_prefer_group_dir,
        "max_single_rmse": max_single_rmse,
        "multi_min_groups": multi_min_groups,
        "multi_mode": multi_mode,
        "lidar_topic": lidar_topic,
        "use_config_rois": use_config_rois,
        "group_layout": group_layout,
        "group_dir_prefix": group_dir_prefix,
        "bag_name": bag_name,
        "image_name": image_name,
        "pcd_name": pcd_name,
        "rois": rois,
        "default_roi": default_roi,
    }


def group_directory(config, group):
    data = Path(config["data_dir"])
    layout = config.get("group_layout", "flat")
    if layout in ("save_data", "nested", "directory", "dir"):
        return data / f"{config.get('group_dir_prefix', 'save_data_')}{group}"
    if layout == "auto":
        nested = data / f"{config.get('group_dir_prefix', 'save_data_')}{group}"
        if nested.is_dir():
            return nested
    return data


def group_image_path(config, group):
    data = Path(config["data_dir"])
    layout = config.get("group_layout", "flat")
    if layout in ("save_data", "nested", "directory", "dir"):
        return group_directory(config, group) / config.get("image_name", "img_0001.jpg")
    if layout == "auto":
        flat = data / f"{group}.jpg"
        if flat.exists():
            return flat
        return group_directory(config, group) / config.get("image_name", "img_0001.jpg")
    return data / f"{group}.jpg"


def candidate_pcd_paths(config, group):
    data = Path(config["data_dir"])
    gdir = group_directory(config, group)
    pcd_name = config.get("pcd_name", "")
    if config.get("group_layout", "flat") in ("save_data", "nested", "directory", "dir", "auto") and gdir != data:
        candidates = []
        if pcd_name:
            candidates.append(gdir / pcd_name)
        candidates.extend([
            gdir / str(group),
            gdir / "pcd",
            gdir / "1.pcd",
            gdir / f"{group}.pcd",
        ])
        return candidates
    return [data / str(group), data / f"{group}.pcd"]


def discover_groups(data_dir, sources):
    data = Path(data_dir)
    config = {
        "data_dir": data_dir,
        "group_layout": rospy.get_param("~group_layout", rospy.get_param("/batch_calib/group_layout", "flat")),
        "group_dir_prefix": rospy.get_param("~group_dir_prefix", rospy.get_param("/batch_calib/group_dir_prefix", "save_data_")),
        "bag_name": rospy.get_param("~bag_name", rospy.get_param("/batch_calib/bag_name", "1.bag")),
        "image_name": rospy.get_param("~image_name", rospy.get_param("/batch_calib/image_name", "img_0001.jpg")),
        "pcd_name": rospy.get_param("~pcd_name", rospy.get_param("/batch_calib/pcd_name", "")),
    }
    layout = str(config["group_layout"]).strip().lower()
    groups = []
    if layout in ("save_data", "nested", "directory", "dir", "auto"):
        prefix = config["group_dir_prefix"]
        for group_dir in sorted(data.glob(f"{prefix}*"), key=lambda p: natural_key(p.name)):
            if not group_dir.is_dir():
                continue
            group = group_dir.name[len(prefix):] if group_dir.name.startswith(prefix) else group_dir.name
            if not group:
                continue
            ok = group_image_path(config, group).exists()
            if "bag" in sources:
                ok = ok and (group_dir / config["bag_name"]).exists()
            if "pcd" in sources:
                ok = ok and any(path.exists() for path in candidate_pcd_paths(config, group))
            if ok:
                groups.append(group)
        if groups or layout != "auto":
            return groups

    for image in sorted(data.glob("*.jpg"), key=lambda p: natural_key(p.stem)):
        group = image.stem
        ok = True
        if "bag" in sources:
            ok = ok and (data / f"{group}.bag").exists()
        if "pcd" in sources:
            ok = ok and ((data / group).is_dir() or (data / f"{group}.pcd").exists())
        if ok:
            groups.append(group)
    return groups


def get_roi(group, rois, group_dir_prefix="save_data_"):
    group_text = str(group)
    keys = [
        group_text,
        int(group_text) if group_text.isdigit() else None,
    ]
    if group_dir_prefix and not group_text.startswith(group_dir_prefix):
        keys.append(f"{group_dir_prefix}{group_text}")
    if group_dir_prefix and group_text.startswith(group_dir_prefix):
        suffix = group_text[len(group_dir_prefix):]
        keys.extend([suffix, int(suffix) if suffix.isdigit() else None])
    roi = None
    for key in keys:
        if key in rois:
            roi = rois[key]
            break
    if not isinstance(roi, dict):
        return None
    required = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    if any(k not in roi for k in required):
        return None
    return {k: roi[k] for k in required}


def get_group_roi(group, rois, default_roi, group_dir_prefix="save_data_"):
    roi = get_roi(group, rois, group_dir_prefix)
    if roi is not None:
        return roi
    if not isinstance(default_roi, dict):
        return None
    required = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    if any(k not in default_roi for k in required):
        return None
    return {k: default_roi[k] for k in required}


def source_path(config, group, source):
    data = Path(config["data_dir"])
    gdir = group_directory(config, group)
    layout = config.get("group_layout", "flat")
    if source == "bag":
        if layout in ("save_data", "nested", "directory", "dir") or (layout == "auto" and gdir != data):
            return gdir / config.get("bag_name", "1.bag")
        return data / f"{group}.bag"
    if source == "pcd":
        if layout in ("save_data", "nested", "directory", "dir") or (layout == "auto" and gdir != data):
            for path in candidate_pcd_paths(config, group):
                if path.exists():
                    return path
            return candidate_pcd_paths(config, group)[0]
        group_dir = data / str(group)
        pcd_file = data / f"{group}.pcd"
        if config["pcd_prefer_group_dir"] and group_dir.is_dir():
            return group_dir
        return pcd_file
    raise ValueError(f"unsupported source: {source}")


def run_single(config, group, source, roi, output_dir):
    image_path = group_image_path(config, group)
    cloud_path = source_path(config, group, source)
    if not image_path.exists():
        return {
            "ok": False,
            "quality": "failed",
            "reason": "missing_image",
            "message": f"missing image: {image_path}",
            "image_path": str(image_path),
            "cloud_path": str(cloud_path),
        }
    if not cloud_path.exists():
        return {
            "ok": False,
            "quality": "failed",
            "reason": f"missing_{source}",
            "message": f"missing {source}: {cloud_path}",
            "image_path": str(image_path),
            "cloud_path": str(cloud_path),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "roslaunch",
        "fast_calib",
        "calib.launch",
        "rviz:=false",
        f"config_file:={config['config_file']}",
        f"pointcloud_source:={source}",
        f"lidar_topic:={config['lidar_topic']}",
        f"image_path:={image_path}",
        f"output_path:={output_dir}",
        "exit_after_save:=true",
        f"x_min:={roi['x_min']}",
        f"x_max:={roi['x_max']}",
        f"y_min:={roi['y_min']}",
        f"y_max:={roi['y_max']}",
        f"z_min:={roi['z_min']}",
        f"z_max:={roi['z_max']}",
    ]
    if source == "bag":
        cmd.append(f"bag_path:={cloud_path}")
    else:
        cmd.append(f"pcd_path:={cloud_path}")

    rospy.loginfo("Running group %s source %s", group, source)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    clean = strip_ansi(proc.stdout)
    log_path = output_dir / "run.log"
    log_path.write_text(clean, encoding="utf-8")

    rmse_match = RMSE_RE.search(clean)
    points_match = POINTS_RE.search(clean)
    diagnostics = parse_run_diagnostics(clean, proc.returncode, config["max_single_rmse"])
    ok = proc.returncode == 0 and diagnostics["quality"] != "failed"
    message = "ok" if ok else diagnostics.get("reason", "calibration failed")
    result = {
        "ok": ok,
        "returncode": proc.returncode,
        "rmse": float(rmse_match.group(1)) if rmse_match else None,
        "points": int(points_match.group(1)) if points_match else None,
        "image_path": str(image_path),
        "cloud_path": str(cloud_path),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "message": message,
    }
    result.update(diagnostics)
    return result


def last_int_match(regex, text):
    matches = regex.findall(text)
    if not matches:
        return None
    return int(matches[-1])


def parse_run_diagnostics(text, returncode, max_single_rmse):
    result = {
        "quality": "failed",
        "reason": "calibration_failed",
        "filtered_points": last_int_match(FILTERED_RE, text),
        "plane_points": last_int_match(PLANE_RE, text),
        "edge_points": last_int_match(EDGE_RE, text),
        "circle_candidates": last_int_match(CANDIDATES_RE, text),
        "qr_centers": None,
        "lidar_centers": None,
        "selected_rmse": None,
        "geom_score": None,
        "geom_valid": None,
        "support": None,
    }

    count_match = NEED_COUNT_RE.search(text) or MAIN_COUNT_RE.search(text)
    if count_match:
        result["lidar_centers"] = int(count_match.group(1))
        result["qr_centers"] = int(count_match.group(2))

    selected_match = SELECTED_RE.search(text)
    if selected_match:
        result["selected_rmse"] = float(selected_match.group(1))
        result["geom_score"] = float(selected_match.group(2))
        result["geom_valid"] = selected_match.group(3) == "true"
        result["support"] = int(selected_match.group(4))

    rmse_match = RMSE_RE.search(text)
    rmse = float(rmse_match.group(1)) if rmse_match else None

    points_match = POINTS_RE.search(text)
    points = int(points_match.group(1)) if points_match else None

    if points == 0 or "Loaded 0 points" in text:
        result["reason"] = "cloud_empty"
        return result
    if result["qr_centers"] is not None and result["qr_centers"] < 4:
        result["reason"] = "qr_centers_failed"
        return result
    if result["lidar_centers"] is not None and result["lidar_centers"] < 4:
        result["reason"] = "lidar_centers_failed"
        return result
    if "Skip SVD calibration" in text or "process has died" in text:
        result["reason"] = "calibration_failed"
        return result
    if returncode == 0 and rmse is None:
        result["reason"] = "missing_rmse"
        return result

    if returncode == 0:
        if result["geom_valid"] is False:
            result["quality"] = "warn"
            result["reason"] = "geometry_bad"
        elif rmse is not None and rmse > max_single_rmse:
            result["quality"] = "warn"
            result["reason"] = "high_rmse"
        else:
            result["quality"] = "good"
            result["reason"] = "ok"
        result["qr_centers"] = result["qr_centers"] if result["qr_centers"] is not None else 4
        result["lidar_centers"] = result["lidar_centers"] if result["lidar_centers"] is not None else 4
        return result

    if "Loading the image" in text and "failed" in text:
        result["reason"] = "image_load_failed"
    elif "Loaded 0 points" in text:
        result["reason"] = "cloud_empty"
    elif "Filtered cloud is empty" in text:
        result["reason"] = "roi_empty"
    elif "Plane fitting failed" in text:
        result["reason"] = "plane_failed"
    elif result["qr_centers"] is not None and result["qr_centers"] < 4:
        result["reason"] = "qr_centers_failed"
    elif result["lidar_centers"] is not None and result["lidar_centers"] < 4:
        result["reason"] = "lidar_centers_failed"
    elif "Unable to find a candidate set" in text:
        result["reason"] = "lidar_circle_match_failed"
    return result


SUMMARY_FIELDS = [
    "group",
    "source",
    "quality",
    "ok",
    "reason",
    "points",
    "filtered_points",
    "plane_points",
    "edge_points",
    "circle_candidates",
    "qr_centers",
    "lidar_centers",
    "rmse",
    "selected_rmse",
    "geom_valid",
    "support",
    "cloud_path",
    "image_path",
    "output_dir",
    "log_path",
]


def write_summary(output_root, rows):
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "batch_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_field(row.get(field)) for field in SUMMARY_FIELDS})

    # Keep the old filename for compatibility with earlier workflow.
    legacy_path = output_root / "batch_summary.txt"
    legacy_path.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")

    report_path = output_root / "batch_report.md"
    report_path.write_text(format_markdown_report(rows), encoding="utf-8")
    return csv_path


def format_field(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


def format_markdown_report(rows):
    lines = [
        "# Batch Calibration Report",
        "",
        "| Group | Source | Quality | Reason | Points | Filtered | Plane | Edge | Candidates | Centers L/Q | RMSE | Output |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in rows:
        centers = f"{row.get('lidar_centers', '')}/{row.get('qr_centers', '')}"
        output = row.get("output_dir", "")
        output_text = f"`{output}`" if output else ""
        rmse = "" if row.get("rmse") is None else f"{row['rmse']:.4f}"
        lines.append(
            "| {group} | {source} | {quality} | {reason} | {points} | {filtered} | {plane} | {edge} | {candidates} | {centers} | {rmse} | {output} |".format(
                group=row.get("group", ""),
                source=row.get("source", ""),
                quality=row.get("quality", ""),
                reason=row.get("reason", ""),
                points=row.get("points", "") or "",
                filtered=row.get("filtered_points", "") or "",
                plane=row.get("plane_points", "") or "",
                edge=row.get("edge_points", "") or "",
                candidates=row.get("circle_candidates", "") or "",
                centers=centers,
                rmse=rmse,
                output=output_text,
            )
        )
    lines.append("")
    lines.append("Quality: `good` means calibration succeeded and RMSE is within the configured threshold; `warn` means calibration produced an output but needs review; `failed` means no valid extrinsic was saved.")
    return "\n".join(lines) + "\n"


def parse_centers_line(line):
    points = []
    for match in CENTER_RE.findall(line):
        try:
            vals = [float(v.strip()) for v in match.split(",")]
        except ValueError:
            return None
        if len(vals) != 3:
            return None
        points.append(vals)
    return np.asarray(points, dtype=float)


def read_center_record(record_path):
    lines = Path(record_path).read_text(encoding="utf-8").splitlines()
    lidar_line = next((line for line in lines if line.startswith("lidar_centers:")), "")
    qr_line = next((line for line in lines if line.startswith("qr_centers:")), "")
    lidar = parse_centers_line(lidar_line)
    qr = parse_centers_line(qr_line)
    if lidar is None or qr is None or lidar.shape != (4, 3) or qr.shape != (4, 3):
        return None, None
    return lidar, qr


def solve_rigid_transform(lidar_points, qr_points):
    lidar_points = np.asarray(lidar_points, dtype=float)
    qr_points = np.asarray(qr_points, dtype=float)
    if lidar_points.shape != qr_points.shape or lidar_points.shape[0] < 3:
        return None

    mu_l = lidar_points.mean(axis=0)
    mu_q = qr_points.mean(axis=0)
    x = lidar_points - mu_l
    y = qr_points - mu_q
    sigma = x.T @ y
    u, _, vt = np.linalg.svd(sigma)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0:
        d = np.eye(3)
        d[2, 2] = -1.0
        rot = vt.T @ d @ u.T
    trans = mu_q - rot @ mu_l
    return rot, trans


def solve_rigid_rmse(lidar_points, qr_points):
    transform = solve_rigid_transform(lidar_points, qr_points)
    if transform is None:
        return None
    rot, trans = transform
    pred = (rot @ np.asarray(lidar_points, dtype=float).T).T + trans
    err = np.linalg.norm(pred - qr_points, axis=1)
    return float(np.sqrt(np.mean(err * err)))


def evaluate_multi_combo(by_group, combo):
    lidar_parts = []
    qr_parts = []
    records = {}
    for group in combo:
        record_path = Path(by_group[group]["output_dir"]) / "circle_center_record.txt"
        lidar, qr = read_center_record(record_path)
        if lidar is None:
            return None
        records[group] = (lidar, qr)
        lidar_parts.append(lidar)
        qr_parts.append(qr)

    lidar_points = np.vstack(lidar_parts)
    qr_points = np.vstack(qr_parts)
    transform = solve_rigid_transform(lidar_points, qr_points)
    if transform is None:
        return None
    rot, trans = transform
    pred = (rot @ lidar_points.T).T + trans
    err = np.linalg.norm(pred - qr_points, axis=1)
    rmse = float(np.sqrt(np.mean(err * err)))

    per_group = {}
    for group, (lidar, qr) in records.items():
        group_pred = (rot @ lidar.T).T + trans
        group_err = np.linalg.norm(group_pred - qr, axis=1)
        per_group[group] = float(np.sqrt(np.mean(group_err * group_err)))
    return rmse, per_group


def greedy_multi_candidates(by_group, groups, min_groups):
    current = list(groups)
    candidates = []
    while len(current) >= min_groups:
        evaluated = evaluate_multi_combo(by_group, current)
        if evaluated is None:
            break
        rmse, per_group = evaluated
        candidates.append({"groups": tuple(current), "rmse": rmse})
        if len(current) == min_groups:
            break
        worst_group = max(per_group, key=per_group.get)
        current = [group for group in current if group != worst_group]
    return candidates


def choose_multi_rows(source_rows, mode, min_groups):
    valid_rows = [row for row in source_rows if row.get("ok") and (Path(row["output_dir"]) / "circle_center_record.txt").exists()]
    if len(valid_rows) < min_groups:
        return [], None, []

    if mode == "all":
        return sorted(valid_rows, key=lambda r: natural_key(r["group"])), None, []

    if mode == "good":
        good_rows = [row for row in valid_rows if row.get("quality") == "good"]
        if len(good_rows) >= min_groups:
            return sorted(good_rows, key=lambda r: natural_key(r["group"])), None, []
        ranked = sorted(valid_rows, key=lambda r: float("inf") if r.get("rmse") is None else r["rmse"])
        return sorted(ranked[:min_groups], key=lambda r: natural_key(r["group"])), None, []

    by_group = {row["group"]: row for row in valid_rows}
    groups = sorted(by_group.keys(), key=natural_key)
    if len(groups) > 12:
        good_groups = [group for group in groups if by_group[group].get("quality") == "good"]
        if len(good_groups) >= min_groups and len(good_groups) <= 12:
            groups = good_groups
        else:
            candidates = greedy_multi_candidates(by_group, groups, min_groups)
            if not candidates:
                return [], None, []
            best = min(candidates, key=lambda item: (item["rmse"], len(item["groups"])))
            rows = [by_group[group] for group in best["groups"]]
            return sorted(rows, key=lambda r: natural_key(r["group"])), best, candidates

    candidates = []
    for size in range(min_groups, len(groups) + 1):
        for combo in itertools.combinations(groups, size):
            evaluated = evaluate_multi_combo(by_group, combo)
            if evaluated is None:
                continue
            rmse, _ = evaluated
            candidates.append({"groups": combo, "rmse": rmse})

    if not candidates:
        return [], None, []
    best = min(candidates, key=lambda item: (item["rmse"], len(item["groups"])))
    rows = [by_group[group] for group in best["groups"]]
    return sorted(rows, key=lambda r: natural_key(r["group"])), best, candidates


def run_multi(config, output_root, rows):
    by_source = {}
    for row in rows:
        if row.get("ok"):
            by_source.setdefault(row["source"], []).append(row)

    multi_rows = []
    for source, source_rows in by_source.items():
        selected_rows, best, candidates = choose_multi_rows(
            source_rows,
            config["multi_mode"],
            config["multi_min_groups"],
        )
        if len(selected_rows) < config["multi_min_groups"]:
            rospy.logwarn("Skip multi calibration for %s: only %d usable groups", source, len(selected_rows))
            continue

        groups_label = "_".join(row["group"] for row in selected_rows)
        multi_dir = output_root / source / f"multi_{config['multi_mode']}_{groups_label}"
        multi_dir.mkdir(parents=True, exist_ok=True)
        record_path = multi_dir / "circle_center_record.txt"
        with record_path.open("w", encoding="utf-8") as out:
            for row in selected_rows:
                record = Path(row["output_dir"]) / "circle_center_record.txt"
                if record.exists():
                    out.write(record.read_text(encoding="utf-8").rstrip() + "\n")

        candidates_path = multi_dir / "multi_candidates.csv"
        with candidates_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["groups", "rmse"])
            writer.writeheader()
            for candidate in sorted(candidates, key=lambda item: item["rmse"]):
                writer.writerow({
                    "groups": " ".join(candidate["groups"]),
                    "rmse": f"{candidate['rmse']:.6f}",
                })

        cmd = [
            "roslaunch",
            "fast_calib",
            "multi_calib.launch",
            "rviz:=false",
            f"config_file:={config['config_file']}",
            f"output_path:={multi_dir}",
        ]
        selected_group_text = ",".join(row["group"] for row in selected_rows)
        rospy.loginfo("Running multi calibration for %s groups [%s]", source, selected_group_text)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        clean = strip_ansi(proc.stdout)
        (multi_dir / "run.log").write_text(clean, encoding="utf-8")
        rmse_match = RMSE_RE.search(clean)
        multi_rows.append({
            "source": source,
            "groups": selected_group_text,
            "mode": config["multi_mode"],
            "precheck_rmse": None if best is None else best["rmse"],
            "result_rmse": None if not rmse_match else float(rmse_match.group(1)),
            "output_dir": str(multi_dir),
            "returncode": proc.returncode,
        })

    if multi_rows:
        path = output_root / "multi_summary.csv"
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["source", "groups", "mode", "precheck_rmse", "result_rmse", "output_dir", "returncode"],
            )
            writer.writeheader()
            for row in multi_rows:
                writer.writerow({key: format_field(row.get(key)) for key in writer.fieldnames})
        rospy.loginfo("Multi calibration summary saved to %s", path)


def main():
    rospy.init_node("batch_fast_calib")
    config = get_batch_config()
    if not config["config_file"]:
        rospy.logerr("config_file is empty")
        return 1
    if not config["data_dir"]:
        rospy.logerr("batch_calib data_dir is empty")
        return 1

    data_dir = Path(config["data_dir"])
    if not data_dir.is_dir():
        rospy.logerr("data_dir does not exist: %s", data_dir)
        return 1

    groups = config["groups"] or discover_groups(config["data_dir"], config["sources"])
    groups = sorted([str(g) for g in groups], key=natural_key)
    if not groups:
        rospy.logerr("No groups found in %s", data_dir)
        return 1

    output_root = Path(config["output_path"]) if config["output_path"] else data_dir / config["output_dir_name"]
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for group in groups:
        roi = get_group_roi(group, config["rois"], config["default_roi"], config["group_dir_prefix"])
        if roi is None:
            row = {
                "group": group,
                "source": "",
                "ok": False,
                "message": f"missing ROI for group {group}",
                "output_dir": "",
                "log_path": "",
            }
            rows.append(row)
            rospy.logerr(row["message"])
            if not config["continue_on_error"]:
                break
            continue

        for source in config["sources"]:
            out_dir = output_root / source / str(group)
            result = run_single(config, group, source, roi, out_dir)
            result.update({"group": group, "source": source})
            rows.append(result)
            if result["ok"]:
                rospy.loginfo(
                    "Finished group %s source %s: points=%s rmse=%s",
                    group,
                    source,
                    result.get("points"),
                    result.get("rmse"),
                )
            else:
                rospy.logerr("Failed group %s source %s: %s", group, source, result.get("message"))
                if not config["continue_on_error"]:
                    summary_path = write_summary(output_root, rows)
                    rospy.loginfo("Summary saved to %s", summary_path)
                    return 1

    if config["run_multi"]:
        run_multi(config, output_root, rows)

    summary_path = write_summary(output_root, rows)
    rospy.loginfo("Batch calibration summary saved to %s", summary_path)
    failed = [r for r in rows if not r.get("ok")]
    return 1 if failed and not config["continue_on_error"] else 0


if __name__ == "__main__":
    sys.exit(main())
