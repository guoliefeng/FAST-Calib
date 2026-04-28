#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic vehicle calibration launcher.

Examples:
  python3 scripts/run_vehicle_calib.py --vehicle 224 --stage all
  python3 scripts/run_vehicle_calib.py --vehicle 224 --sensors front,rear --stage all
  python3 scripts/run_vehicle_calib.py --vehicle 221 --dry-run
"""
import argparse, copy, csv, subprocess, sys
from pathlib import Path
from typing import Any, Dict, List
import yaml

PROJECT_ROOT = Path("/home/glf/dataDisk/calib/FAST-Calib_ws/src/FAST-Calib").resolve()
PIPELINE = PROJECT_ROOT / "scripts" / "calib_pipeline.py"
DEFAULT_CFG = PROJECT_ROOT / "config" / "defaults" / "pipeline_default.yaml"
VEHICLE_ROOT = PROJECT_ROOT / "config" / "vehicles"
GENERATED_ROOT = PROJECT_ROOT / "config" / "generated_jobs"
REPORT_ROOT = Path("/home/glf/dataDisk/calib/vehicle_calib_report").resolve()
POINTCLOUD_TYPES = {"sensor_msgs/PointCloud2", "livox_ros_driver/CustomMsg"}

def load_yaml(p: Path) -> Dict[str, Any]:
    with Path(p).expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def write_yaml(p: Path, obj: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)

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

def detect_topic(data_dir: str) -> str:
    try:
        import rosbag
    except Exception as e:
        print(f"[WARN] rosbag import failed, lidar_topic stays empty: {e}")
        return ""
    bags = sorted(Path(data_dir).expanduser().glob("save_data_*/1.bag"))
    if not bags:
        print(f"[WARN] no bag found in {data_dir}/save_data_*/1.bag")
        return ""
    with rosbag.Bag(str(bags[0]), "r") as bag:
        info = bag.get_type_and_topic_info()
    pcs = [(t, ti.msg_type, ti.message_count) for t, ti in info.topics.items() if ti.msg_type in POINTCLOUD_TYPES]
    if len(pcs) == 1:
        print(f"[topic] {Path(data_dir).name}: {pcs[0][0]} ({pcs[0][1]}, count={pcs[0][2]})")
        return pcs[0][0]
    print(f"[WARN] expected one point cloud topic in {bags[0]}, found={pcs}")
    return ""

def build_job(vehicle: str, sensor: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    defaults = load_yaml(DEFAULT_CFG)
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
    job["output_dir"] = "${data_dir}/_calib_output"
    job["roi"] = deep_merge(job.get("roi", {}), roi_block)
    job.setdefault("fast_calib", {})["config_file"] = str(camera_cfg)

    topic = cfg.get("lidar_topic", "auto")
    if str(topic).lower() in ("auto", "unique", ""):
        topic = detect_topic(data_dir)
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
    out = Path(cfg["data_dir"]) / "_calib_output"
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
    ap.add_argument("--stage", default="all", choices=["all","scan","extract_pcd","roi","calibrate","verify"])
    ap.add_argument("--sensors", default="", help="comma-separated positions, e.g. front,rear")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop-on-error", action="store_true")
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
        job = build_job(args.vehicle, n, sensors[n])
        p = gen / f"{n}.pipeline.yaml"
        write_yaml(p, job); jobs[n] = p
        print(f"[JOB] {args.vehicle}/{n}: {p}")
    if args.dry_run:
        print("[DRY RUN] generated job yaml files only."); return

    report = REPORT_ROOT / args.vehicle; log_dir = report / "run_logs"; log_dir.mkdir(parents=True, exist_ok=True)
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
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print("\n" + "="*90); print(f"[SUMMARY] {summary}"); print("="*90)
    failed = {k:v for k,v in codes.items() if v != 0}
    if failed: print(f"[WARN] some sensors failed: {failed}"); sys.exit(1)

if __name__ == "__main__":
    main()
