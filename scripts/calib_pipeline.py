#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAST-Calib one-command pipeline.
Stages: scan -> extract_pcd -> roi -> calibrate -> verify
Usage:
  source /opt/ros/noetic/setup.bash
  source /home/glf/dataDisk/calib/FAST-Calib_ws/devel/setup.bash
  python3 calib_pipeline.py --job job_rear_right.yaml --stage all
"""
import argparse, csv, itertools, math, re, shutil, subprocess
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import yaml

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RMSE_RE = re.compile(r"\[Result\]\s+RMSE:\s+([0-9.+\-eE]+)")
POINTS_RE = re.compile(r"Loaded\s+([0-9]+)\s+points")
FILTERED_RE = re.compile(r"(?:Depth filtered cloud size|Filtered cloud size):\s+([0-9]+)")
PLANE_RE = re.compile(r"Plane cloud size:\s+([0-9]+)")
EDGE_RE = re.compile(r"Extracted\s+([0-9]+)\s+edge points")
NEED_COUNT_RE = re.compile(r"Need 4 LiDAR centers and 4 QR centers.*got lidar=([0-9]+), qr=([0-9]+)")
MAIN_COUNT_RE = re.compile(r"got lidar=([0-9]+), qr=([0-9]+)")
CENTER_RE = re.compile(r"\{([^}]*)\}")
POINTCLOUD_TYPES = {"sensor_msgs/PointCloud2", "livox_ros_driver/CustomMsg"}


def natural_key(v):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(v))]

def strip_ansi(s): return ANSI_RE.sub("", s)
def mkdir(p): p.mkdir(parents=True, exist_ok=True); return p

def load_yaml(p):
    with Path(p).expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def write_yaml(p, obj):
    mkdir(Path(p).parent)
    with Path(p).open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)

def normalize_roi_yaml(data):
    if not isinstance(data, dict):
        return {"batch_calib": {"default_roi": {}, "rois": {}}}
    if "batch_calib" in data:
        batch = data.get("batch_calib") or {}
        return {"batch_calib": {
            "default_roi": batch.get("default_roi", {}),
            "rois": batch.get("rois", {}),
        }}
    return {"batch_calib": {
        "default_roi": data.get("default_roi", data.get("unified_roi", {})),
        "rois": data.get("rois", data.get("groups", {})),
    }}

def resolve_path(s, base=None, output_dir=None):
    s = str(s).replace("${data_dir}", str(base) if base else "")
    s = s.replace("${output_dir}", str(output_dir) if output_dir else "")
    p = Path(s).expanduser()
    if base and s and not p.is_absolute(): p = Path(base) / p
    return p.resolve()

def job_paths(job):
    data_dir = resolve_path(job["data_dir"])
    output_dir = resolve_path(job.get("output_dir", "_calib_output"), data_dir)
    return data_dir, output_dir

def ff(x): return float(f"{float(x):.6f}")

def boolv(v, default=False):
    if v is None: return default
    if isinstance(v, bool): return v
    return str(v).lower() in ("1", "true", "yes", "on")

# -----------------------------------------------------------------------------
# scan
# -----------------------------------------------------------------------------
def scan_groups(job):
    data_dir, out = job_paths(job)
    layout = job.get("layout", {})
    prefix = layout.get("group_prefix", "save_data_")
    bag_name = layout.get("bag_name", "1.bag")
    image_name = layout.get("image_name", "img_0001.jpg")
    pcd_name = layout.get("pcd_name", "1.pcd")
    board_name = layout.get("board_pcd_name", "board_candidate.pcd")
    board_template = layout.get("board_pcd_template", "")
    groups = []
    for d in sorted(data_dir.glob(f"{prefix}*"), key=natural_key):
        if not d.is_dir(): continue
        board_pcd = d / board_name
        if board_template:
            rel = board_template.replace("${group}", d.name).replace("{group}", d.name)
            board_pcd = resolve_path(rel, data_dir, out)
        groups.append({
            "group": d.name,
            "dir": str(d),
            "bag": str(d / bag_name),
            "image": str(d / image_name),
            "pcd": str(d / pcd_name),
            "board_pcd": str(board_pcd),
        })
    if not groups: raise RuntimeError(f"No group dirs found: {data_dir}/{prefix}*")
    path = mkdir(out) / "00_manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as f:
        fields = ["group", "dir", "bag", "image", "pcd", "board_pcd"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(groups)
    print(f"[scan] groups={len(groups)} manifest={path}")
    return groups

# -----------------------------------------------------------------------------
# bag -> pcd
# -----------------------------------------------------------------------------
def ros_imports():
    try:
        import rosbag
        import sensor_msgs.point_cloud2 as pc2
        return rosbag, pc2
    except Exception as e:
        raise RuntimeError("ROS python modules unavailable. source /opt/ros/noetic/setup.bash first. " + str(e))

def bag_topics(bag_path):
    rosbag, _ = ros_imports()
    with rosbag.Bag(str(bag_path), "r") as bag:
        info = bag.get_type_and_topic_info()
    return [{"topic": t, "type": ti.msg_type, "count": ti.message_count} for t, ti in info.topics.items()]

def choose_topic(bag_path):
    topics = bag_topics(bag_path)
    pc = [x for x in topics if x["type"] in POINTCLOUD_TYPES]
    if len(topics) == 1 and topics[0]["type"] in POINTCLOUD_TYPES: return topics[0]
    if len(pc) == 1: return pc[0]
    raise RuntimeError(f"Need exactly one point cloud topic in {bag_path}, found: {pc or topics}")

def frame_idx(sel, count):
    s = str(sel).lower()
    if s == "middle": return count // 2
    if s == "last": return count - 1
    i = int(s)
    return count + i if i < 0 else i

def selected_frame_indices(sel, count, stride=1, max_frames=0):
    """Return frame indices to export from one bag topic."""
    stride = max(1, int(stride or 1))
    max_frames = max(0, int(max_frames or 0))
    s = str(sel).lower()
    if s in ("all", "accumulate", "merged", "merge", "stack"):
        idx = list(range(0, count, stride))
        if max_frames > 0:
            idx = idx[:max_frames]
        return idx
    return [frame_idx(sel, count)]

def read_msg(bag_path, topic, count, frame):
    rosbag, _ = ros_imports()
    target = frame_idx(frame, count)
    with rosbag.Bag(str(bag_path), "r") as bag:
        for i, (_, msg, _) in enumerate(bag.read_messages(topics=[topic])):
            if i == target: return msg, i
    raise RuntimeError("target frame not found")

def read_msgs(bag_path, topic, count, frame, stride=1, max_frames=0):
    rosbag, _ = ros_imports()
    targets = set(selected_frame_indices(frame, count, stride=stride, max_frames=max_frames))
    msgs = []
    with rosbag.Bag(str(bag_path), "r") as bag:
        for i, (_, msg, _) in enumerate(bag.read_messages(topics=[topic])):
            if i in targets:
                msgs.append((msg, i))
    if not msgs:
        raise RuntimeError("target frame(s) not found")
    return msgs

def pointcloud2_fields(msg):
    names = {f.name for f in msg.fields}
    for r in ("x", "y", "z"):
        if r not in names: raise RuntimeError(f"PointCloud2 lacks {r}")
    fields = ["x", "y", "z"]
    if "intensity" in names: fields.append("intensity")
    elif "reflectivity" in names: fields.append("reflectivity")
    if "ring" in names: fields.append("ring")
    elif "line" in names: fields.append("line")
    if "time" in names: fields.append("time")
    elif "timestamp" in names: fields.append("timestamp")
    return fields

def pointcloud2_rows(msg, fields):
    _, pc2 = ros_imports(); rows = []
    for p in pc2.read_points(msg, field_names=fields, skip_nans=False):
        vals = []
        for v in p:
            try: vals.append(float(v))
            except Exception: vals.append(float("nan"))
        if all(math.isfinite(vals[i]) for i in range(3)): rows.append(vals)
    return rows

def livox_rows(msg):
    fields = ["x", "y", "z", "intensity", "line"]; rows = []
    for p in msg.points:
        x, y, z = float(p.x), float(p.y), float(p.z)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)): continue
        rows.append([x, y, z, float(getattr(p, "reflectivity", 0.0)), float(getattr(p, "line", 0.0))])
    return fields, rows

def write_ascii_pcd(path, fields, rows):
    mkdir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as f:
        n = len(rows)
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n")
        f.write("FIELDS " + " ".join(fields) + "\n")
        f.write("SIZE " + " ".join(["4"] * len(fields)) + "\n")
        f.write("TYPE " + " ".join(["F"] * len(fields)) + "\n")
        f.write("COUNT " + " ".join(["1"] * len(fields)) + "\n")
        f.write(f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA ascii\n")
        for row in rows: f.write(" ".join(f"{float(x):.9g}" for x in row) + "\n")

def stage_extract_pcd(job, groups):
    cfg = job.get("extract_pcd", {})
    if not boolv(cfg.get("enabled", True), True): return
    _, out = job_paths(job); path = out / "01_pcd_export_summary.csv"; mkdir(path.parent)
    rows = []
    for g in groups:
        bag = Path(g["bag"]); pcd = Path(g["pcd"]); print(f"[extract] {g['group']}")
        row = {"group": g["group"], "bag": str(bag), "pcd": str(pcd), "status": "", "topic": "", "type": "", "frame": "", "frame_count": "", "frame_indices": "", "count": "", "points": "", "error": ""}
        try:
            if pcd.exists() and not boolv(cfg.get("overwrite", False), False):
                row["status"] = "skipped_exists"; rows.append(row); continue
            info = choose_topic(bag)
            frame_sel = cfg.get("frame", "all")
            msgs = read_msgs(
                bag,
                info["topic"],
                info["count"],
                frame_sel,
                stride=cfg.get("frame_stride", 1),
                max_frames=cfg.get("max_frames", 0),
            )
            fields, pts = None, []
            for msg, _idx in msgs:
                if info["type"] == "sensor_msgs/PointCloud2":
                    if fields is None:
                        fields = pointcloud2_fields(msg)
                    pts.extend(pointcloud2_rows(msg, fields))
                else:
                    livox_fields, livox_pts = livox_rows(msg)
                    if fields is None:
                        fields = livox_fields
                    pts.extend(livox_pts)
            if not pts: raise RuntimeError("empty selected frame")
            write_ascii_pcd(pcd, fields, pts)
            row.update(status="ok", topic=info["topic"], type=info["type"],
                       frame=frame_sel, frame_count=len(msgs),
                       frame_indices=",".join(str(i) for _, i in msgs),
                       count=info["count"], points=len(pts))
        except Exception as e:
            row.update(status="failed", error=str(e)); print("  failed", e)
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows[0].keys()); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(f"[extract] summary={path}")

# -----------------------------------------------------------------------------
# PCD loader and ROI
# -----------------------------------------------------------------------------
def parse_pcd_header(path):
    header = {}; lines = []
    with Path(path).open("rb") as f:
        while True:
            line = f.readline()
            if not line: raise ValueError("PCD header incomplete")
            s = line.decode("utf-8", errors="ignore").strip(); lines.append(s)
            if s and not s.startswith("#"):
                parts = s.split(); header[parts[0].upper()] = parts[1:]
            if s.upper().startswith("DATA"):
                off = f.tell(); break
    return header, lines, off

def unique_names(fields):
    used = {}; out = []; first = {}
    for name in fields:
        if name in used: used[name] += 1; uname = f"{name}__{used[name]}"
        else: used[name] = 0; uname = name; first[name] = uname
        out.append(uname)
    return out, first

def build_dtype(fields, sizes, types, counts):
    uniq, first = unique_names(fields); ds = []
    for name, size, typ, count in zip(uniq, sizes, types, counts):
        size = int(size); count = int(count)
        table = {"F": {4: np.float32, 8: np.float64}, "U": {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}, "I": {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}}
        base = table.get(typ, {}).get(size)
        if base is None: raise ValueError(f"unsupported field {typ}{size}")
        ds.append((name, base) if count == 1 else (name, base, (count,)))
    return np.dtype(ds), first

def load_pcd_xyz(path):
    header, lines, off = parse_pcd_header(Path(path)); fields = header["FIELDS"]; data = header["DATA"][0].lower()
    if not all(k in fields for k in ("x", "y", "z")): raise ValueError("PCD lacks xyz")
    n = int(header.get("POINTS", [int(header["WIDTH"][0]) * int(header.get("HEIGHT", ["1"])[0])])[0])
    if data == "ascii":
        start = next(i + 1 for i, s in enumerate(lines) if s.upper().startswith("DATA"))
        arr = np.loadtxt(Path(path).read_text(errors="ignore").splitlines()[start:])
        if arr.ndim == 1: arr = arr.reshape(1, -1)
        xyz = arr[:, [fields.index("x"), fields.index("y"), fields.index("z")]].astype(float)
    elif data == "binary":
        dtype, first = build_dtype(fields, header["SIZE"], header["TYPE"], header.get("COUNT", ["1"] * len(fields)))
        with Path(path).open("rb") as f:
            f.seek(off); arr = np.fromfile(f, dtype=dtype, count=n)
        xyz = np.stack([arr[first["x"]], arr[first["y"]], arr[first["z"]]], axis=1).astype(float)
    else:
        raise ValueError(f"unsupported PCD DATA {data}")
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if xyz.shape[0] == 0: raise ValueError("empty xyz")
    return xyz

def write_xyz_pcd(path, pts):
    rows = pts.tolist(); write_ascii_pcd(path, ["x", "y", "z"], rows)

def coarse_filter(pts, roi):
    return pts[(pts[:,0]>=roi["x_min"]) & (pts[:,0]<=roi["x_max"]) & (pts[:,1]>=roi["y_min"]) & (pts[:,1]<=roi["y_max"]) & (pts[:,2]>=roi["z_min"]) & (pts[:,2]<=roi["z_max"])]

def robust_bbox(pts, qs=(0.5, 99.5)):
    mn = np.percentile(pts, qs[0], axis=0) if pts.shape[0] > 20 else pts.min(axis=0)
    mx = np.percentile(pts, qs[1], axis=0) if pts.shape[0] > 20 else pts.max(axis=0)
    return mn, mx, mx-mn

def components3d(pts, grid, min_points):
    cells = np.floor(pts / grid).astype(np.int64); cmap = defaultdict(list)
    for i, c in enumerate(cells): cmap[(int(c[0]), int(c[1]), int(c[2]))].append(i)
    visited = set(); comps = []; neigh = [(i,j,k) for i in (-1,0,1) for j in (-1,0,1) for k in (-1,0,1)]
    for cell in list(cmap.keys()):
        if cell in visited: continue
        q = deque([cell]); visited.add(cell); cc = []
        while q:
            c = q.popleft(); cc.append(c)
            for dx,dy,dz in neigh:
                nb=(c[0]+dx,c[1]+dy,c[2]+dz)
                if nb not in visited and nb in cmap: visited.add(nb); q.append(nb)
        idx=[]
        for c in cc: idx += cmap[c]
        if len(idx) >= min_points: comps.append(np.asarray(idx))
    return comps

def pick_board(coarse, roi_cfg):
    det = roi_cfg.get("detector", {}); grid=float(det.get("cluster_grid",0.28)); minp=int(det.get("min_component_points",80))
    bs = det.get("board_span", {}); sx=bs.get("x",[0.3,5.0]); sy=bs.get("y",[0.05,3.0]); sz=bs.get("z",[0.3,1.8])
    candidates=[]
    for comp in components3d(coarse, grid, minp):
        pts = coarse[comp]; mn,mx,span = robust_bbox(pts)
        if sx[0] <= span[0] <= sx[1] and sy[0] <= span[1] <= sy[1] and sz[0] <= span[2] <= sz[1]:
            score = pts.shape[0] - 60*abs(span[2]-float(det.get("expected_height",1.0)))
            candidates.append((score, pts, "component"))
    if candidates:
        return sorted(candidates, key=lambda x:x[0], reverse=True)[0][1:]
    return coarse, "fallback_coarse"

def make_box(rmin, rmax, n=80):
    xmin,ymin,zmin=rmin; xmax,ymax,zmax=rmax
    c=[np.array([xmin,ymin,zmin]),np.array([xmax,ymin,zmin]),np.array([xmax,ymax,zmin]),np.array([xmin,ymax,zmin]),np.array([xmin,ymin,zmax]),np.array([xmax,ymin,zmax]),np.array([xmax,ymax,zmax]),np.array([xmin,ymax,zmax])]
    e=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    out=[]
    for a,b in e:
        for t in np.linspace(0,1,n): out.append(c[a]*(1-t)+c[b]*t)
    return np.asarray(out)

def stage_roi(job, groups):
    roi_cfg = job.get("roi", {})
    _, out = job_paths(job); roi_dir = mkdir(out/"02_roi"); dbg=mkdir(roi_dir/"debug"); board_dir=mkdir(roi_dir/"board_candidates")
    existing = roi_cfg.get("existing_file", "")
    if not boolv(roi_cfg.get("enabled", True), True):
        if not existing:
            raise RuntimeError("ROI stage disabled but roi.existing_file is empty")
        src = resolve_path(existing, job_paths(job)[0], out)
        if not src.exists():
            raise RuntimeError(f"ROI file does not exist: {src}")
        dst = roi_dir / "roi_groups.yaml"
        write_yaml(dst, normalize_roi_yaml(load_yaml(src)))
        print(f"[roi] using existing ROI file: {src} -> {dst}")
        return
    coarse = roi_cfg["coarse"]; pad = roi_cfg.get("padding", {"x":0.15,"y":0.15,"z":0.15}); qs=roi_cfg.get("robust_quantile", [0.5,99.5])
    rows=[]; rois={}; mins=[]; maxs=[]
    for g in groups:
        print(f"[roi] {g['group']}"); row={"group":g["group"],"status":"","method":"","points_total":"","points_coarse":"","points_board":"","error":""}
        try:
            pts=load_pcd_xyz(g["pcd"]); cf=coarse_filter(pts, coarse)
            if cf.shape[0] < int(roi_cfg.get("detector",{}).get("min_coarse_points",50)): raise RuntimeError(f"too few coarse points {cf.shape[0]}")
            board, method = pick_board(cf, roi_cfg); mn,mx,span=robust_bbox(board, qs)
            rmin=mn-np.array([pad.get("x",0.15),pad.get("y",0.15),pad.get("z",0.15)]); rmax=mx+np.array([pad.get("x",0.15),pad.get("y",0.15),pad.get("z",0.15)])
            roi={"x_min":ff(rmin[0]),"x_max":ff(rmax[0]),"y_min":ff(rmin[1]),"y_max":ff(rmax[1]),"z_min":ff(rmin[2]),"z_max":ff(rmax[2])}
            rois[g["group"]]=roi; mins.append(rmin); maxs.append(rmax)
            roi_board_pcd=board_dir/f"{g['group']}_board_candidate.pcd"
            write_xyz_pcd(g["board_pcd"], board)
            write_xyz_pcd(roi_board_pcd, board)
            write_xyz_pcd(dbg/f"{g['group']}_roi_box.pcd", make_box(rmin,rmax))
            row.update(status="ok",method=method,points_total=pts.shape[0],points_coarse=cf.shape[0],points_board=board.shape[0],board_pcd=str(g["board_pcd"]),roi_board_pcd=str(roi_board_pcd),**roi)
        except Exception as e:
            row.update(status="failed",error=str(e)); print("  failed",e)
        rows.append(row)
    fields=sorted(set(k for r in rows for k in r.keys()))
    with (roi_dir/"roi_summary.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    if not mins:
        raise RuntimeError("all ROI detections failed; adjust roi.coarse or set roi.enabled=false and roi.existing_file")
    umin=np.vstack(mins).min(axis=0); umax=np.vstack(maxs).max(axis=0)
    unified={"x_min":ff(umin[0]),"x_max":ff(umax[0]),"y_min":ff(umin[1]),"y_max":ff(umax[1]),"z_min":ff(umin[2]),"z_max":ff(umax[2])}
    write_yaml(roi_dir/"roi_groups.yaml", {"batch_calib":{"default_roi":unified,"rois":rois}})
    write_yaml(roi_dir/"roi_unified.yaml", {"default_roi":unified})
    print(f"[roi] unified={unified}")
    print(f"[roi] board candidates saved in: {board_dir}")

# -----------------------------------------------------------------------------
# calibrate and multi-SVD
# -----------------------------------------------------------------------------
def roi_for_group(data, group):
    b=normalize_roi_yaml(data).get("batch_calib",{})
    return b.get("rois",{}).get(group,b.get("default_roi"))

def run_calib(job, g, roi, out_dir):
    fc=job["fast_calib"]; source=fc.get("pointcloud_source","pcd"); img=Path(g["image"]); mkdir(out_dir)
    ros_source=source
    if source=="pcd":
        cloud=Path(g["pcd"])
    elif source=="board_pcd":
        ros_source="pcd"
        cloud=Path(g["board_pcd"])
    else:
        cloud=Path(g["bag"])
    for stale in ("circle_center_record.txt", "single_calib_result.txt", "colored_cloud.pcd", "run.log"):
        p = out_dir / stale
        if p.exists():
            p.unlink()
    cmd=["roslaunch",fc.get("package","fast_calib"),fc.get("calib_launch","calib.launch"),"rviz:=false",f"config_file:={fc['config_file']}",f"pointcloud_source:={ros_source}",f"image_path:={img}",f"output_path:={out_dir}","exit_after_save:=true",f"x_min:={roi['x_min']}",f"x_max:={roi['x_max']}",f"y_min:={roi['y_min']}",f"y_max:={roi['y_max']}",f"z_min:={roi['z_min']}",f"z_max:={roi['z_max']}"]
    camera_cfg=load_yaml(fc["config_file"])
    for dist_key in ("k3","k4","k5","k6"):
        cmd.append(f"{dist_key}:={camera_cfg.get(dist_key,0)}")
    for airy_key in ("airy_hole_detector", "airy_boundary_radius", "airy_boundary_min_angular_gap", "airy_boundary_min_neighbors"):
        if airy_key in fc:
            cmd.append(f"{airy_key}:={fc[airy_key]}")
    for airy_template_key in (
        "airy_template_detector",
        "airy_template_grid",
        "airy_template_angle_step_deg",
        "airy_template_ring_band",
        "airy_template_min_score",
    ):
        if airy_template_key in fc:
            cmd.append(f"{airy_template_key}:={fc[airy_template_key]}")
    for lidar_center_key in (
        "lidar_center_extraction_mode",
        "lidar_strict_geometry",
        "lidar_geometry_side_rel_tol",
        "lidar_geometry_diag_rel_tol",
        "lidar_geometry_perimeter_rel_tol",
        "lidar_min_plane_points",
        "lidar_ransac_min_inliers",
        "lidar_ransac_radius_tolerance",
        "lidar_ransac_inlier_threshold",
        "lidar_template_min_ring_per_hole",
        "lidar_template_min_outer_per_hole",
        "lidar_template_max_inside_per_hole",
        "lidar_template_local_refine_radius",
        "lidar_template_local_refine_step",
    ):
        if lidar_center_key in fc:
            cmd.append(f"{lidar_center_key}:={fc[lidar_center_key]}")
    if ros_source == "bag" and fc.get("lidar_topic"):
        cmd.append(f"lidar_topic:={fc['lidar_topic']}")
    cmd.append(("pcd_path:=" if ros_source=="pcd" else "bag_path:=") + str(cloud))
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    log=strip_ansi(proc.stdout); (out_dir/"run.log").write_text(log,encoding="utf-8")
    rm=RMSE_RE.search(log); cnt=NEED_COUNT_RE.search(log) or MAIN_COUNT_RE.search(log)
    lidar=qr=None
    if cnt: lidar=int(cnt.group(1)); qr=int(cnt.group(2))
    rmse=float(rm.group(1)) if rm else None; ok=proc.returncode==0 and rmse is not None
    return {"group":g["group"],"ok":ok,"rmse":rmse,"reason":"ok" if ok else "failed","points":lastint(POINTS_RE,log),"filtered_points":lastint(FILTERED_RE,log),"plane_points":lastint(PLANE_RE,log),"edge_points":lastint(EDGE_RE,log),"lidar_centers":lidar if lidar is not None else (4 if ok else None),"qr_centers":qr if qr is not None else (4 if ok else None),"cloud_path":str(cloud),"image_path":str(img),"output_dir":str(out_dir),"log_path":str(out_dir/"run.log")}

def lastint(regex,text):
    m=regex.findall(text); return int(m[-1]) if m else None

def parse_centers(line):
    pts=[]
    for m in CENTER_RE.findall(line):
        vals=[float(x.strip()) for x in m.split(",")]
        if len(vals)==3: pts.append(vals)
    return np.asarray(pts,dtype=float)

def read_record(path):
    if not Path(path).exists(): return None,None
    lines=Path(path).read_text(encoding="utf-8").splitlines()
    lidar=parse_centers(next((l for l in reversed(lines) if l.startswith("lidar_centers:")),"")); cam=parse_centers(next((l for l in reversed(lines) if l.startswith("qr_centers:")),""))
    if lidar.shape!=(4,3) or cam.shape!=(4,3): return None,None
    return lidar,cam

def svd_solve(a,b):
    ma=a.mean(axis=0); mb=b.mean(axis=0); x=a-ma; y=b-mb; u,_,vt=np.linalg.svd(x.T@y); R=vt.T@u.T
    if np.linalg.det(R)<0: D=np.eye(3); D[2,2]=-1; R=vt.T@D@u.T
    t=mb-R@ma; pred=(R@a.T).T+t; err=np.linalg.norm(pred-b,axis=1); return R,t,float(np.sqrt(np.mean(err*err)))

def eval_groups(records, gs):
    a=np.vstack([records[g][0] for g in gs]); b=np.vstack([records[g][1] for g in gs]); R,t,rmse=svd_solve(a,b); per={}
    for g in gs:
        aa,bb=records[g]; pred=(R@aa.T).T+t; per[g]=float(np.sqrt(np.mean(np.linalg.norm(pred-bb,axis=1)**2)))
    return R,t,rmse,per

def residuals_for_records(records, R, t):
    per={}
    for g,(aa,bb) in records.items():
        pred=(R@aa.T).T+t
        per[g]=float(np.sqrt(np.mean(np.linalg.norm(pred-bb,axis=1)**2)))
    return per

def center_array_to_list(a):
    return [[ff(x) for x in p] for p in a.tolist()]

def row_lookup(rows):
    out={}
    for r in rows:
        g=r.get("group","")
        if g:
            out[g]=r
    return out

def write_center_records(multi_dir, records, rows, selected_groups=None, per_group=None):
    selected=set(selected_groups or [])
    per_group=per_group or {}
    rows_by_group=row_lookup(rows)
    yaml_obj={"groups":{}}
    csv_rows=[]
    for group in sorted(records.keys(), key=natural_key):
        lidar,qr=records[group]
        info=rows_by_group.get(group,{})
        yaml_obj["groups"][group]={
            "selected": group in selected,
            "single_rmse": info.get("rmse"),
            "multi_group_rmse": ff(per_group[group]) if group in per_group else None,
            "reason": info.get("reason",""),
            "log_path": info.get("log_path",""),
            "output_dir": info.get("output_dir",""),
            "lidar_centers": center_array_to_list(lidar),
            "qr_centers": center_array_to_list(qr),
        }
        for i in range(4):
            csv_rows.append({
                "group": group,
                "selected": group in selected,
                "center_index": i,
                "single_rmse": info.get("rmse",""),
                "multi_group_rmse": ff(per_group[group]) if group in per_group else "",
                "lidar_x": ff(lidar[i,0]), "lidar_y": ff(lidar[i,1]), "lidar_z": ff(lidar[i,2]),
                "qr_x": ff(qr[i,0]), "qr_y": ff(qr[i,1]), "qr_z": ff(qr[i,2]),
                "log_path": info.get("log_path",""),
            })
    write_yaml(multi_dir/"all_circle_centers.yaml", yaml_obj)
    with (multi_dir/"all_circle_centers.csv").open("w",encoding="utf-8",newline="") as f:
        fields=["group","selected","center_index","single_rmse","multi_group_rmse",
                "lidar_x","lidar_y","lidar_z","qr_x","qr_y","qr_z","log_path"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(csv_rows)

def write_selection_tables(multi_dir, history, selected_groups, per_group):
    with (multi_dir/"selection_history.csv").open("w",encoding="utf-8",newline="") as f:
        fields=["iter","group_count","rmse","worst_group","worst_rmse","removed_next","groups"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for i,h in enumerate(history or []):
            w.writerow({
                "iter": i,
                "group_count": len(h.get("groups",[])),
                "rmse": ff(h.get("rmse",0.0)),
                "worst_group": h.get("worst_group",""),
                "worst_rmse": ff(h.get("worst_rmse",0.0)) if h.get("worst_rmse","") != "" else "",
                "removed_next": h.get("removed_next",""),
                "groups": ",".join(h.get("groups",[])),
            })
    with (multi_dir/"group_residuals.csv").open("w",encoding="utf-8",newline="") as f:
        fields=["group","selected","multi_group_rmse"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        selected=set(selected_groups or [])
        for g in sorted(per_group.keys(), key=natural_key):
            w.writerow({"group":g,"selected":g in selected,"multi_group_rmse":ff(per_group[g])})

def write_permutation_tables(multi_dir, records, selected_groups, per_group, permutations):
    if not permutations:
        return
    selected=set(selected_groups or [])
    with (multi_dir/"group_permutations.csv").open("w",encoding="utf-8",newline="") as f:
        fields=["group","selected","multi_group_rmse","qr_permutation"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for g in sorted(permutations.keys(), key=natural_key):
            w.writerow({
                "group":g,
                "selected":g in selected,
                "multi_group_rmse":ff(per_group[g]) if g in per_group else "",
                "qr_permutation":" ".join(str(int(i)) for i in permutations[g]),
            })
    with (multi_dir/"all_circle_centers_permuted.csv").open("w",encoding="utf-8",newline="") as f:
        fields=["group","selected","center_index","qr_original_index",
                "lidar_x","lidar_y","lidar_z","qr_x","qr_y","qr_z"]
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for g in sorted(records.keys(), key=natural_key):
            if g not in permutations:
                continue
            lidar,qr=records[g]
            perm=permutations[g]
            for i, orig_i in enumerate(perm):
                w.writerow({
                    "group":g,
                    "selected":g in selected,
                    "center_index":i,
                    "qr_original_index":int(orig_i),
                    "lidar_x":ff(lidar[i,0]), "lidar_y":ff(lidar[i,1]), "lidar_z":ff(lidar[i,2]),
                    "qr_x":ff(qr[orig_i,0]), "qr_y":ff(qr[orig_i,1]), "qr_z":ff(qr[orig_i,2]),
                })

def choose_all_center_records(records, min_groups):
    groups=list(records.keys())
    if len(groups)<min_groups:
        return None
    R,t,rmse,per=eval_groups(records,groups)
    return {"groups":groups,"rmse":rmse,"per_group":per,"R":R,"t":t,"policy":"all_center_records"}

CENTER_PERMS_4 = list(itertools.permutations(range(4)))

def best_qr_permutation(lidar, qr, R, t):
    pred=(R@lidar.T).T+t
    best_err=None; best_perm=None
    for perm in CENTER_PERMS_4:
        target=qr[list(perm)]
        err=float(np.sqrt(np.mean(np.linalg.norm(pred-target,axis=1)**2)))
        if best_err is None or err<best_err:
            best_err=err; best_perm=perm
    return best_err,best_perm

def best_permutations_for_transform(records, R, t):
    rows=[]; perms={}
    for g,(lidar,qr) in records.items():
        err,perm=best_qr_permutation(lidar,qr,R,t)
        rows.append((err,g,perm)); perms[g]=perm
    rows.sort(key=lambda x:(x[0], natural_key(x[1])))
    return rows,perms

def eval_groups_with_qr_permutations(records, groups, permutations):
    a=[]; b=[]
    for g in groups:
        lidar,qr=records[g]
        a.append(lidar)
        b.append(qr[list(permutations[g])])
    a=np.vstack(a); b=np.vstack(b)
    R,t,rmse=svd_solve(a,b)
    per={}
    for g in groups:
        lidar,qr=records[g]
        target=qr[list(permutations[g])]
        pred=(R@lidar.T).T+t
        per[g]=float(np.sqrt(np.mean(np.linalg.norm(pred-target,axis=1)**2)))
    return R,t,rmse,per

def permutation_candidate(records, seed_R, seed_t, min_groups, max_group_rmse, max_iters=8):
    R=seed_R; t=seed_t; last_sig=None; groups=None; perms=None
    for _ in range(max_iters):
        rows,perms=best_permutations_for_transform(records,R,t)
        inliers=[g for err,g,_ in rows if err<=max_group_rmse]
        groups=inliers if len(inliers)>=min_groups else [g for _,g,_ in rows[:min_groups]]
        sig=(tuple(groups), tuple(tuple(perms[g]) for g in groups))
        R2,t2,_,_=eval_groups_with_qr_permutations(records,groups,perms)
        if sig==last_sig:
            R,t=R2,t2
            break
        R,t=R2,t2; last_sig=sig

    rows,perms=best_permutations_for_transform(records,R,t)
    inliers=[g for err,g,_ in rows if err<=max_group_rmse]
    groups=inliers if len(inliers)>=min_groups else [g for _,g,_ in rows[:min_groups]]
    R,t,rmse,per=eval_groups_with_qr_permutations(records,groups,perms)
    rows,all_perms=best_permutations_for_transform(records,R,t)
    per_all={g:err for err,g,_ in rows}
    return {
        "groups":list(groups),
        "rmse":rmse,
        "per_group":per,
        "per_group_all":per_all,
        "R":R,
        "t":t,
        "policy":"permutation_search",
        "permutations":{g:perms[g] for g in groups},
        "best_permutations_all":all_perms,
        "worst_group":max(per,key=per.get) if per else "",
        "worst_rmse":max(per.values()) if per else 0.0,
    }

def choose_permutation_search(records, min_groups, max_multi, max_group_rmse):
    groups=sorted(records.keys(), key=natural_key)
    if len(groups)<min_groups:
        return None, [], []
    candidates=[]
    for seed_group in groups:
        lidar,qr=records[seed_group]
        for perm in CENTER_PERMS_4:
            R,t,_=svd_solve(lidar,qr[list(perm)])
            item=permutation_candidate(records,R,t,min_groups,max_group_rmse)
            item["seed_group"]=seed_group
            item["seed_permutation"]=perm
            candidates.append(item)

    dedup={}
    for c in candidates:
        key=(tuple(c["groups"]), tuple((g, tuple(c["permutations"][g])) for g in c["groups"]))
        old=dedup.get(key)
        if old is None or c["rmse"]<old["rmse"]:
            dedup[key]=c
    candidates=list(dedup.values())
    valid=[c for c in candidates if c["rmse"]<=max_multi and c["worst_rmse"]<=max_group_rmse]
    if valid:
        selected=max(valid, key=lambda c:(len(c["groups"]), -c["rmse"], -c["worst_rmse"]))
    else:
        selected=min(candidates, key=lambda c:(c["rmse"], c["worst_rmse"], -len(c["groups"]))) if candidates else None
    ranked=sorted(candidates, key=lambda c:(-(len(c["groups"])), c["rmse"], c["worst_rmse"]))[:80]
    return selected, ranked, []

def choose_robust_max_groups(records, min_groups, max_multi, max_group_rmse):
    groups=list(records.keys())
    if len(groups)<min_groups:
        return None, [], []
    cur=list(groups); history=[]
    while len(cur)>=min_groups:
        R,t,rmse,per=eval_groups(records,cur)
        worst=max(per,key=per.get)
        item={
            "groups":list(cur),
            "rmse":rmse,
            "per_group":per,
            "R":R,
            "t":t,
            "policy":"robust_max_groups",
            "worst_group":worst,
            "worst_rmse":per[worst],
            "removed_next": worst if len(cur)>min_groups else "",
        }
        history.append(item)
        if rmse<=max_multi and per[worst]<=max_group_rmse:
            return item, [], history
        if len(cur)==min_groups:
            break
        cur=[g for g in cur if g!=worst]
    valid=[h for h in history if h["rmse"]<=max_multi and h["worst_rmse"]<=max_group_rmse]
    if valid:
        return max(valid, key=lambda h:(len(h["groups"]), -h["rmse"])), [], history
    fallback=min(history, key=lambda h:(h["rmse"], len(h["groups"]))) if history else None
    return fallback, [], history

def choose_multi(records, min_groups, max_multi, mode, max_group_rmse=None):
    groups=list(records.keys())
    if max_group_rmse is None:
        max_group_rmse=max(2.0*max_multi, 0.06)
    if len(groups)<min_groups:
        return None, [], []
    candidates=[]
    if mode in ("permutation_search", "perm_search", "permutation", "permute_centers", "permutation_robust"):
        return choose_permutation_search(records, min_groups, max_multi, max_group_rmse)
    if mode in ("robust_max_groups", "robust", "max_consensus", "largest_consensus"):
        return choose_robust_max_groups(records, min_groups, max_multi, max_group_rmse)
    if mode in ("max_groups", "best", "exhaustive"):
        for size in range(min_groups, len(groups)+1):
            best=None
            for combo in itertools.combinations(groups, size):
                R,t,rmse,per=eval_groups(records, combo)
                item={"groups":list(combo),"rmse":rmse,"per_group":per,"R":R,"t":t}
                if best is None or rmse < best["rmse"]:
                    best=item
            if best is not None:
                candidates.append(best)
        valid=[c for c in candidates if c["rmse"] <= max_multi]
        if valid:
            if mode == "best":
                selected=min(valid, key=lambda c: (c["rmse"], -len(c["groups"])))
            else:
                selected=max(valid, key=lambda c: (len(c["groups"]), -c["rmse"]))
            return selected, candidates, []
        fallback=min(candidates, key=lambda c: (c["rmse"], -len(c["groups"]))) if candidates else None
        return fallback, candidates, []

    cur=list(groups); history=[]
    while len(cur)>=min_groups:
        R,t,rmse,per=eval_groups(records,cur); history.append({"groups":list(cur),"rmse":rmse,"per_group":per,"R":R,"t":t})
        if rmse<=max_multi:
            return history[-1], [], history
        cur=[g for g in cur if g!=max(per,key=per.get)]
    return None, [], history

def stage_calibrate(job, groups):
    _, out=job_paths(job); single=mkdir(out/"03_single"); multi=mkdir(out/"04_multi"); roi_data=load_yaml(out/"02_roi"/"roi_groups.yaml"); fc=job["fast_calib"]
    max_single=float(fc.get("max_single_rmse",0.03)); max_multi=float(fc.get("max_multi_rmse",0.03)); max_group_rmse=float(fc.get("max_group_rmse",max(2.0*max_multi,0.06))); min_groups=int(fc.get("multi_min_groups",3)); multi_mode=str(fc.get("multi_mode","max_groups")); filter_high_single=boolv(fc.get("filter_high_single_for_multi",False),False)
    rows=[]; records={}; records_all={}; four_center_groups=[]; usable_four_center_groups=[]; non_four_center_rows=[]
    for g in groups:
        roi=roi_for_group(roi_data,g["group"])
        if not roi: rows.append({"group":g["group"],"ok":False,"reason":"missing_roi"}); continue
        print(f"[calib] {g['group']}"); r=run_calib(job,g,roi,single/g["group"])
        if r.get("ok") and r.get("rmse",999)>max_single: r["ok"]=False; r["reason"]="high_single_rmse"
        a,b=read_record(Path(r.get("output_dir",""))/"circle_center_record.txt")
        four_center_ok = (a is not None and b is not None)
        r["four_center_ok"]=four_center_ok
        if four_center_ok:
            records_all[g["group"]]=(a,b); four_center_groups.append(g["group"])
            if r.get("ok") or not filter_high_single:
                records[g["group"]]=(a,b); usable_four_center_groups.append(g["group"])
        else:
            non_four_center_rows.append({
                "group": g["group"],
                "reason": r.get("reason",""),
                "lidar_centers": r.get("lidar_centers",""),
                "qr_centers": r.get("qr_centers",""),
                "rmse": r.get("rmse",""),
                "log_path": r.get("log_path",""),
            })
        rows.append(r)
    fields=sorted(set(k for r in rows for k in r.keys()))
    with (out/"batch_summary.csv").open("w",encoding="utf-8",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    with (multi/"non_four_center_groups.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["group","reason","lidar_centers","qr_centers","rmse","log_path"])
        w.writeheader(); w.writerows(non_four_center_rows)
    write_center_records(multi, records_all, rows)
    if multi_mode in ("all_centers","all_4centers","all_detected","all_center_records"):
        selected=choose_all_center_records(records,min_groups); candidates=[]; history_raw=[]
    else:
        selected,candidates,history_raw=choose_multi(records,min_groups,max_multi,multi_mode,max_group_rmse)
    if selected is None:
        failed_result={"status":"failed","reason":"not_enough_usable_4center_groups","four_center_groups":four_center_groups,"usable_four_center_groups":usable_four_center_groups,"non_four_center_groups":non_four_center_rows}
        write_yaml(multi/"multi_result.yaml",failed_result)
        write_yaml(out/"final_extrinsic.yaml",failed_result)
        return
    cur=selected["groups"]; R=selected["R"]; t=selected["t"]; rmse=selected["rmse"]; per=selected["per_group"]; per_all=selected.get("per_group_all") or residuals_for_records(records,R,t)
    if records_all.keys() != records.keys():
        per_all=residuals_for_records(records_all,R,t)
    write_center_records(multi, records_all, rows, cur, per_all)
    write_selection_tables(multi, (candidates if candidates else history_raw), cur, per_all)
    write_permutation_tables(multi, records, cur, per_all, selected.get("best_permutations_all"))
    history=[
        {"groups":c["groups"],"rmse":ff(c["rmse"]),"per_group":{k:ff(v) for k,v in c["per_group"].items()}}
        for c in (candidates if candidates else history_raw)
    ]
    if candidates:
        with (multi/"multi_candidates.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["group_count","rmse","groups"]); w.writeheader()
            for c in candidates:
                w.writerow({"group_count":len(c["groups"]),"rmse":f"{c['rmse']:.6f}","groups":" ".join(c["groups"])})
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=t
    result={"status":"ok" if rmse<=max_multi and (not per or max(per.values())<=max_group_rmse) else "warn","rmse":ff(rmse),"max_multi_rmse":ff(max_multi),"max_group_rmse":ff(max_group_rmse),"multi_mode":multi_mode,"selection_policy":selected.get("policy",multi_mode),"four_center_groups":four_center_groups,"usable_four_center_groups":usable_four_center_groups,"non_four_center_groups":non_four_center_rows,"selected_groups":cur,"final_group_residuals":{k:ff(v) for k,v in sorted(per_all.items(), key=lambda kv:natural_key(kv[0]))},"T_cam_lidar":[[ff(x) for x in row] for row in T.tolist()],"Rcl":[[ff(x) for x in row] for row in R.tolist()],"Pcl":[ff(x) for x in t.tolist()],"history":history}
    if selected.get("best_permutations_all"):
        result["permutation_note"]="qr_centers are reordered per group before joint SVD; tuple means reordered_qr = original_qr[tuple]."
        result["selected_qr_permutations"]={g:[int(i) for i in selected.get("permutations",{}).get(g,())] for g in cur}
        result["best_qr_permutations_all_groups"]={g:[int(i) for i in selected["best_permutations_all"][g]] for g in sorted(selected["best_permutations_all"].keys(), key=natural_key)}
    write_yaml(multi/"multi_result.yaml",result); write_yaml(out/"final_extrinsic.yaml",result); (multi/"selected_groups.txt").write_text("\n".join(cur)+"\n",encoding="utf-8")
    with (multi/"multi_calib_result.txt").open("w", encoding="utf-8") as f:
        f.write("# FAST-LIVO2 calibration format\n")
        f.write("Rcl: [ " + ", ".join(f"{x: .6f}" for x in R[0]) + ",\n")
        f.write("       " + ", ".join(f"{x: .6f}" for x in R[1]) + ",\n")
        f.write("       " + ", ".join(f"{x: .6f}" for x in R[2]) + "]\n")
        f.write("Pcl: [ " + ", ".join(f"{x: .6f}" for x in t) + "]\n")
    print(f"[multi] selected={cur} rmse={rmse:.6f}")

# -----------------------------------------------------------------------------
# verify projection
# -----------------------------------------------------------------------------
def camera_from_config(path):
    c=load_yaml(path)
    return float(c["fx"]),float(c["fy"]),float(c["cx"]),float(c["cy"]),float(c.get("k1",0)),float(c.get("k2",0)),float(c.get("p1",0)),float(c.get("p2",0)),float(c.get("k3",0)),float(c.get("k4",0)),float(c.get("k5",0)),float(c.get("k6",0))

def stage_verify(job, groups):
    if not boolv(job.get("verify",{}).get("enabled",True),True): return
    import cv2
    _,out=job_paths(job); verify=mkdir(out/"05_verify"); final=load_yaml(out/"final_extrinsic.yaml")
    if "T_cam_lidar" not in final:
        rows=[{"group":g["group"],"status":"skipped","reason":"no_valid_extrinsic","points_projected":0,"image":g["image"],"board_pcd":g["board_pcd"],"overlay":""} for g in groups]
        with (verify/"verify_summary.csv").open("w",encoding="utf-8",newline="") as f:
            w=csv.DictWriter(f,fieldnames=["group","status","reason","points_projected","image","board_pcd","overlay"])
            w.writeheader(); w.writerows(rows)
        print("[verify] skipped: no valid T_cam_lidar in final_extrinsic.yaml")
        return
    T=np.asarray(final["T_cam_lidar"],dtype=float); selected=set(final.get("selected_groups",[]))
    fx,fy,cx,cy,k1,k2,p1,p2,k3,k4,k5,k6=camera_from_config(job["fast_calib"]["config_file"]); K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=float)
    D=np.array([k1,k2,p1,p2,k3,k4,k5,k6] if any(abs(x)>1e-12 for x in (k4,k5,k6)) else [k1,k2,p1,p2,k3],dtype=float)
    rows=[]; written=[]
    for g in groups:
        if selected and g["group"] not in selected and not boolv(job.get("verify",{}).get("all_groups",False),False): continue
        img=cv2.imread(g["image"]); bp=Path(g["board_pcd"])
        if img is None:
            rows.append({"group":g["group"],"status":"skipped","reason":"image_not_found_or_unreadable","points_projected":0,"image":g["image"],"board_pcd":str(bp),"overlay":""})
            print(f"[verify] {g['group']} skipped: image not found or unreadable: {g['image']}")
            continue
        if not bp.exists():
            rows.append({"group":g["group"],"status":"skipped","reason":"board_pcd_not_found","points_projected":0,"image":g["image"],"board_pcd":str(bp),"overlay":""})
            print(f"[verify] {g['group']} skipped: board pcd not found: {bp}")
            continue
        pts=load_pcd_xyz(bp); pts_h=np.hstack([pts,np.ones((pts.shape[0],1))]); pc=(T@pts_h.T).T[:,:3]; pc=pc[pc[:,2]>0.05]
        if pc.shape[0]==0:
            rows.append({"group":g["group"],"status":"skipped","reason":"all_points_behind_camera","points_projected":0,"image":g["image"],"board_pcd":str(bp),"overlay":""})
            print(f"[verify] {g['group']} skipped: all transformed points are behind the camera")
            continue
        uv,_=cv2.projectPoints(pc.reshape(-1,1,3),np.zeros((3,1)),np.zeros((3,1)),K,D); uv=uv.reshape(-1,2); h,w=img.shape[:2]; m=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h); uv=uv[m]
        if uv.shape[0]==0:
            rows.append({"group":g["group"],"status":"warn","reason":"all_projected_points_outside_image","points_projected":0,"image":g["image"],"board_pcd":str(bp),"overlay":""})
            print(f"[verify] {g['group']} warning: all projected points are outside the image")
            continue
        for u,v in uv: cv2.circle(img,(int(round(u)),int(round(v))),int(job.get("verify",{}).get("point_radius",2)),(0,0,255),-1)
        outimg=verify/f"{g['group']}_board_overlay.png"; cv2.imwrite(str(outimg),img); written.append(outimg); rows.append({"group":g["group"],"status":"ok","reason":"","points_projected":uv.shape[0],"image":g["image"],"board_pcd":str(bp),"overlay":str(outimg)})
        print(f"[verify] {g['group']} projected={uv.shape[0]}")
    with (verify/"verify_summary.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["group","status","reason","points_projected","image","board_pcd","overlay"])
        w.writeheader(); w.writerows(rows)
    if written:
        thumbs=[]; cols=int(job.get("verify",{}).get("grid_cols",5))
        for p in written:
            im=cv2.imread(str(p))
            if im is None: continue
            tw=480; scale=tw/im.shape[1]
            thumbs.append(cv2.resize(im,(tw,int(im.shape[0]*scale)),interpolation=cv2.INTER_AREA))
        if thumbs:
            h=max(t.shape[0] for t in thumbs); w=max(t.shape[1] for t in thumbs); blank=np.zeros((h,w,3),dtype=np.uint8); rows_img=[]
            for i in range(0,len(thumbs),cols):
                cells=[]
                for t in thumbs[i:i+cols]:
                    cell=blank.copy(); cell[:t.shape[0],:t.shape[1]]=t; cells.append(cell)
                while len(cells)<cols: cells.append(blank.copy())
                rows_img.append(np.hstack(cells))
            grid=np.vstack(rows_img); grid_path=verify/"image_grid_verify.jpg"; cv2.imwrite(str(grid_path),grid,[int(cv2.IMWRITE_JPEG_QUALITY),92])
            print(f"[verify] grid={grid_path}")

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--job",required=True); ap.add_argument("--stage",default="all",choices=["all","scan","extract_pcd","roi","calibrate","verify"]); args=ap.parse_args()
    job=load_yaml(args.job); data,out=job_paths(job); mkdir(out); print(f"[job] data={data} output={out} stage={args.stage}")
    groups=scan_groups(job)
    if args.stage in ("all","extract_pcd"): stage_extract_pcd(job,groups)
    groups=scan_groups(job)
    if args.stage in ("all","roi"): stage_roi(job,groups)
    if args.stage in ("all","calibrate"): stage_calibrate(job,groups)
    if args.stage in ("all","verify"): stage_verify(job,groups)
if __name__=="__main__": main()
