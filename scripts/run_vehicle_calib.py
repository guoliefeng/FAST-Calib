#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic vehicle calibration launcher.

Examples:
  python3 scripts/run_vehicle_calib.py --vehicle 224 --stage all
  python3 scripts/run_vehicle_calib.py --vehicle 224 --sensors front,rear --stage all
  python3 scripts/run_vehicle_calib.py --vehicle 221 --dry-run
"""
import argparse, copy, csv, os, subprocess, sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = PROJECT_ROOT / "scripts" / "calib_pipeline.py"
DEFAULT_CFG = PROJECT_ROOT / "config" / "defaults" / "pipeline_default.yaml"
PROFILE_CFG = {
    "default": None,
    "hesai_fast": PROJECT_ROOT / "config" / "defaults" / "pipeline_hesai_fast.yaml",
    "fast": PROJECT_ROOT / "config" / "defaults" / "pipeline_hesai_fast.yaml",
}
VEHICLE_ROOT = PROJECT_ROOT / "config" / "vehicles"
GENERATED_ROOT = PROJECT_ROOT / "config" / "generated_jobs"
REPORT_ROOT = Path(
    os.environ.get("FAST_CALIB_REPORT_ROOT", PROJECT_ROOT / "vehicle_calib_report")
).expanduser().resolve()
POINTCLOUD_TYPES = {"sensor_msgs/PointCloud2", "livox_ros_driver/CustomMsg"}

def load_yaml(p: Path) -> Dict[str, Any]:
    with Path(p).expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def write_yaml(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)

def fmt_num(v: Any) -> str:
    if isinstance(v, int):
        return str(v)
    try:
        x = float(v)
    except Exception:
        return str(v)
    return f"{x:.6f}"

def flatten_matrix(m: Any) -> List[Any]:
    if not isinstance(m, list):
        return []
    out: List[Any] = []
    for row in m:
        if isinstance(row, list):
            out.extend(row)
        else:
            out.append(row)
    return out

def one_line_list(values: Any) -> str:
    return "[" + ", ".join(fmt_num(v) for v in (values or [])) + "]"

def matrix_rows_lines(name: str, matrix: Any, indent: str = "  ") -> List[str]:
    lines = [f"{indent}{name}:"]
    for row in (matrix or []):
        lines.append(f"{indent}  - {one_line_list(row)}")
    return lines

def write_readable_extrinsics(path: Path, compact: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"vehicle_id: {compact.get('vehicle_id', '')}")
    lines.append("frame_convention:")
    fc = compact.get("frame_convention", {})
    lines.append(f"  transform: {fc.get('transform', 'T_cam_lidar')}")
    lines.append(f"  description: \"{fc.get('description', '')}\"")
    lines.append("extrinsics:")
    for sensor, ext in (compact.get("extrinsics") or {}).items():
        lines.append(f"  {sensor}:")
        lines.append(f"    status: {ext.get('status', '')}")
        lines.append(f"    rmse: {fmt_num(ext.get('rmse'))}")
        lines.extend(matrix_rows_lines("Rcl", ext.get("Rcl"), "    "))
        lines.append(f"    Rcl_flat: {one_line_list(flatten_matrix(ext.get('Rcl')))}")
        lines.append(f"    Pcl: {one_line_list(ext.get('Pcl'))}")
        lines.append(f"    Pcl_xyz: {one_line_list(ext.get('Pcl'))}")
        lines.extend(matrix_rows_lines("T_cam_lidar", ext.get("T_cam_lidar"), "    "))
        lines.append(f"    T_cam_lidar_flat: {one_line_list(flatten_matrix(ext.get('T_cam_lidar')))}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def rpath(s: str) -> Path:
    p = Path(str(s)).expanduser()
    return (PROJECT_ROOT / p).resolve() if not p.is_absolute() else p.resolve()

def output_dir_for(cfg: Dict[str, Any]) -> Path:
    data_dir = Path(str(cfg["data_dir"])).expanduser().resolve()
    value = str(cfg.get("output_dir", "${data_dir}/_calib_output"))
    return Path(value.replace("${data_dir}", str(data_dir))).expanduser().resolve()

def detect_topic(data_dir: str, group_prefix: str = "save_data_") -> str:
    try:
        import rosbag
    except Exception as e:
        print(f"[WARN] rosbag import failed, lidar_topic stays empty: {e}")
        return ""
    bags = sorted(Path(data_dir).expanduser().glob(f"{group_prefix}*/1.bag"))
    if not bags:
        print(f"[WARN] no bag found in {data_dir}/{group_prefix}*/1.bag")
        return ""
    with rosbag.Bag(str(bags[0]), "r") as bag:
        info = bag.get_type_and_topic_info()
    pcs = [(t, ti.msg_type, ti.message_count) for t, ti in info.topics.items() if ti.msg_type in POINTCLOUD_TYPES]
    if len(pcs) == 1:
        print(f"[topic] {Path(data_dir).name}: {pcs[0][0]} ({pcs[0][1]}, count={pcs[0][2]})")
        return pcs[0][0]
    print(f"[WARN] expected one point cloud topic in {bags[0]}, found={pcs}")
    return ""

def load_pipeline_defaults(profile: str) -> Dict[str, Any]:
    defaults = load_yaml(DEFAULT_CFG)
    profile_path = PROFILE_CFG.get(str(profile or "default").strip().lower())
    if profile_path:
        if not profile_path.exists():
            raise FileNotFoundError(f"pipeline profile not found: {profile_path}")
        defaults = deep_merge(defaults, load_yaml(profile_path))
        print(f"[profile] merged {profile_path.name}")
    return defaults

def build_job(vehicle: str, sensor: str, cfg: Dict[str, Any], profile: str = "default") -> Dict[str, Any]:
    defaults = load_pipeline_defaults(profile)
    camera_cfg = rpath(cfg["camera_config"])
    roi_profile = rpath(cfg["roi_profile"])
    if not camera_cfg.exists(): raise FileNotFoundError(f"camera_config not found: {camera_cfg}")
    if not roi_profile.exists(): raise FileNotFoundError(f"roi_profile not found: {roi_profile}")
    data_dir = cfg.get("data_dir")
    if not data_dir: raise RuntimeError(f"{vehicle}/{sensor}: data_dir is empty")

    roi_obj = load_yaml(roi_profile)
    roi_block = roi_obj.get("roi", roi_obj)
    job = copy.deepcopy(defaults)
    job["job_name"] = f"{vehicle}_{sensor}"
    job["data_dir"] = data_dir
    job["output_dir"] = cfg.get("output_dir", "${data_dir}/_calib_output")
    job["roi"] = deep_merge(job.get("roi", {}), roi_block)
    job.setdefault("fast_calib", {})["config_file"] = str(camera_cfg)

    topic = cfg.get("lidar_topic", "auto")
    if str(topic).lower() in ("auto", "unique", ""):
        topic = detect_topic(
            data_dir,
            str((cfg.get("layout") or {}).get("group_prefix", "save_data_")),
        )
    job["fast_calib"]["lidar_topic"] = topic

    if roi_block.get("board_pcd_template"):
        job.setdefault("layout", {})["board_pcd_template"] = roi_block["board_pcd_template"]
    for key in ("layout", "extract_pcd", "roi", "fast_calib", "verify"):
        if key in cfg:
            job[key] = deep_merge(job.get(key, {}), cfg[key])
    return job

def read_csv(p: Path) -> List[Dict[str, str]]:
    if not p.exists(): return []
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def yes(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "ok")

def summarize(vehicle: str, sensor: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = output_dir_for(cfg)
    rows = read_csv(out / "batch_summary.csv")
    mpath = out / "04_multi" / "multi_result.yaml"
    multi = load_yaml(mpath) if mpath.exists() else {}
    four, bad = [], []
    for r in rows:
        g = r.get("group", "")
        ok = yes(r.get("four_center_ok", "")) or (r.get("lidar_centers") == "4" and r.get("qr_centers") == "4")
        (four if ok else bad).append(g)
    selected = multi.get("selected_groups", []) if isinstance(multi.get("selected_groups", []), list) else []
    ratio = len(selected)/len(four) if four else 0.0
    warns = []
    if four and len(selected) < 5: warns.append("selected_count_lt_5")
    if four and ratio < 0.4: warns.append("selected_ratio_lt_0.4")
    if str(multi.get("status", "")).lower() not in ("ok", ""): warns.append("multi_status_not_ok")
    return {"vehicle":vehicle,"sensor":sensor,"data_dir":cfg["data_dir"],"status":multi.get("status",""),"rmse":multi.get("rmse",""),"four_center_count":len(four),"selected_count":len(selected),"selected_ratio":f"{ratio:.3f}","warnings":",".join(warns),"failed_groups":",".join(bad),"final_extrinsic":str(out/"final_extrinsic.yaml"),"multi_result":str(mpath)}

def camera_info(camera_config: str) -> Dict[str, Any]:
    cfg_path = rpath(camera_config)
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}
    keys = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
    return {
        "config_file": str(cfg_path),
        "intrinsics": {k: cfg[k] for k in keys if k in cfg},
    }

def export_vehicle_extrinsics(vehicle: str, names: List[str], sensors: Dict[str, Any], report_dir: Path) -> Path:
    """Write one downstream-friendly YAML containing all selected sensor extrinsics."""
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"{vehicle}_extrinsics.yaml"
    compact_path = report_dir / f"{vehicle}_extrinsics_compact.yaml"
    readable_path = report_dir / f"{vehicle}_extrinsics_readable.yaml"

    obj: Dict[str, Any] = {
        "vehicle_id": str(vehicle),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frame_convention": {
            "transform": "T_cam_lidar",
            "description": "p_cam = Rcl * p_lidar + Pcl; T_cam_lidar maps LiDAR points into the camera frame.",
        },
        "sensors": {},
    }
    compact: Dict[str, Any] = {
        "vehicle_id": str(vehicle),
        "frame_convention": obj["frame_convention"],
        "extrinsics": {},
    }

    for sensor in names:
        cfg = sensors[sensor]
        data_dir = Path(str(cfg["data_dir"])).expanduser()
        output_dir = output_dir_for(cfg)
        final_path = output_dir / "final_extrinsic.yaml"
        multi_path = output_dir / "04_multi" / "multi_result.yaml"
        final = load_yaml(final_path) if final_path.exists() else {}

        status = final.get("status", "missing_result" if not final else "")
        entry: Dict[str, Any] = {
            "enabled": bool(cfg.get("enabled", False)),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "camera": camera_info(str(cfg["camera_config"])),
            "source_files": {
                "final_extrinsic": str(final_path),
                "multi_result": str(multi_path),
            },
            "status": status,
            "rmse": final.get("rmse"),
            "selected_groups": final.get("selected_groups", []),
            "Rcl": final.get("Rcl"),
            "Pcl": final.get("Pcl"),
            "T_cam_lidar": final.get("T_cam_lidar"),
            "final_group_residuals": final.get("final_group_residuals", {}),
        }
        if not final_path.exists():
            entry["warning"] = "final_extrinsic.yaml not found"
        elif str(status).lower() != "ok":
            entry["warning"] = "calibration status is not ok; inspect rmse and projection before downstream use"
        obj["sensors"][sensor] = entry

        if final.get("T_cam_lidar") is not None:
            compact["extrinsics"][sensor] = {
                "status": status,
                "rmse": final.get("rmse"),
                "Rcl": final.get("Rcl"),
                "Pcl": final.get("Pcl"),
                "T_cam_lidar": final.get("T_cam_lidar"),
            }

    write_yaml(out_path, obj)
    write_readable_extrinsics(compact_path, compact)
    write_readable_extrinsics(readable_path, compact)
    print(f"[EXTRINSICS] full   : {out_path}")
    print(f"[EXTRINSICS] compact: {compact_path}")
    print(f"[EXTRINSICS] readable: {readable_path}")
    return out_path

def run_cmd(cmd: List[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(str(x) for x in cmd))
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end=""); log.write(line)
        return proc.wait()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle", required=True)
    ap.add_argument("--stage", default="all", choices=["all","scan","extract_pcd","roi","calibrate","verify","export"])
    ap.add_argument("--sensors", default="", help="comma-separated positions, e.g. front,rear")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument(
        "--profile",
        default="default",
        choices=sorted(PROFILE_CFG.keys()),
        help="Pipeline preset. hesai_fast: middle-frame PCD + board_pcd calib (much faster on Hesai).",
    )
    args = ap.parse_args()

    vfile = VEHICLE_ROOT / args.vehicle / "vehicle.yaml"
    if not vfile.exists(): raise FileNotFoundError(f"vehicle config not found: {vfile}")
    vcfg = load_yaml(vfile)
    sensors = vcfg.get("sensors", {})
    names = [x.strip() for x in args.sensors.split(",") if x.strip()] or [n for n,c in sensors.items() if bool(c.get("enabled", False))]
    for n in names:
        if n not in sensors: raise ValueError(f"unknown sensor {n}; available={list(sensors.keys())}")
        if not bool(sensors[n].get("enabled", False)): print(f"[WARN] {n} is disabled; running because requested explicitly")

    gen = GENERATED_ROOT / args.vehicle; gen.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for n in names:
        job = build_job(args.vehicle, n, sensors[n], profile=args.profile)
        p = gen / f"{n}.pipeline.yaml"
        write_yaml(p, job); jobs[n] = p
        print(f"[JOB] {args.vehicle}/{n}: {p}")
    if args.dry_run:
        print("[DRY RUN] generated job yaml files only."); return

    report = REPORT_ROOT / args.vehicle; log_dir = report / "run_logs"; log_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "export":
        export_vehicle_extrinsics(args.vehicle, names, sensors, report)
        return

    codes = {}
    for n in names:
        print("\n" + "="*90); print(f"[RUN] vehicle={args.vehicle}, sensor={n}, stage={args.stage}"); print("="*90)
        codes[n] = run_cmd([sys.executable, str(PIPELINE), "--job", str(jobs[n]), "--stage", args.stage], log_dir / f"{n}_{args.stage}.log")
        if codes[n] != 0 and args.stop_on_error: sys.exit(codes[n])

    report.mkdir(parents=True, exist_ok=True)
    summary = report / f"{args.vehicle}_summary.csv"
    rows = [summarize(args.vehicle, n, sensors[n]) for n in names]
    fields = ["vehicle","sensor","data_dir","status","rmse","four_center_count","selected_count","selected_ratio","warnings","failed_groups","final_extrinsic","multi_result"]
    with summary.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    print("\n" + "="*90); print(f"[SUMMARY] {summary}"); print("="*90)
    export_vehicle_extrinsics(args.vehicle, names, sensors, report)
    failed = {k:v for k,v in codes.items() if v != 0}
    if failed: print(f"[WARN] some sensors failed: {failed}"); sys.exit(1)

if __name__ == "__main__":
    main()
