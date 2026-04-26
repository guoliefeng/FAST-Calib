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
import argparse, csv, math, re, subprocess
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

def resolve_path(s, base=None):
    s = str(s).replace("${data_dir}", str(base) if base else "")
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
    groups = []
    for d in sorted(data_dir.glob(f"{prefix}*"), key=natural_key):
        if not d.is_dir(): continue
        groups.append({
            "group": d.name,
            "dir": str(d),
            "bag": str(d / bag_name),
            "image": str(d / image_name),
            "pcd": str(d / pcd_name),
            "board_pcd": str(d / board_name),
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

def read_msg(bag_path, topic, count, frame):
    rosbag, _ = ros_imports()
    target = frame_idx(frame, count)
    with rosbag.Bag(str(bag_path), "r") as bag:
        for i, (_, msg, _) in enumerate(bag.read_messages(topics=[topic])):
            if i == target: return msg, i
    raise RuntimeError("target frame not found")

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
        row = {"group": g["group"], "bag": str(bag), "pcd": str(pcd), "status": "", "topic": "", "type": "", "frame": "", "count": "", "points": "", "error": ""}
        try:
            if pcd.exists() and not boolv(cfg.get("overwrite", False), False):
                row["status"] = "skipped_exists"; rows.append(row); continue
            info = choose_topic(bag); msg, idx = read_msg(bag, info["topic"], info["count"], cfg.get("frame", "middle"))
            if info["type"] == "sensor_msgs/PointCloud2": fields = pointcloud2_fields(msg); pts = pointcloud2_rows(msg, fields)
            else: fields, pts = livox_rows(msg)
            if not pts: raise RuntimeError("empty selected frame")
            write_ascii_pcd(pcd, fields, pts)
            row.update(status="ok", topic=info["topic"], type=info["type"], frame=idx, count=info["count"], points=len(pts))
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
    if not boolv(roi_cfg.get("enabled", True), True): return
    _, out = job_paths(job); roi_dir = mkdir(out/"02_roi"); dbg=mkdir(roi_dir/"debug")
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
            write_xyz_pcd(g["board_pcd"], board); write_xyz_pcd(dbg/f"{g['group']}_roi_box.pcd", make_box(rmin,rmax))
            row.update(status="ok",method=method,points_total=pts.shape[0],points_coarse=cf.shape[0],points_board=board.shape[0],**roi)
        except Exception as e:
            row.update(status="failed",error=str(e)); print("  failed",e)
        rows.append(row)
    fields=sorted(set(k for r in rows for k in r.keys()))
    with (roi_dir/"roi_summary.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    if not mins: raise RuntimeError("all ROI detections failed")
    umin=np.vstack(mins).min(axis=0); umax=np.vstack(maxs).max(axis=0)
    unified={"x_min":ff(umin[0]),"x_max":ff(umax[0]),"y_min":ff(umin[1]),"y_max":ff(umax[1]),"z_min":ff(umin[2]),"z_max":ff(umax[2])}
    write_yaml(roi_dir/"roi_groups.yaml", {"batch_calib":{"default_roi":unified,"rois":rois}})
    write_yaml(roi_dir/"roi_unified.yaml", {"default_roi":unified})
    print(f"[roi] unified={unified}")

# -----------------------------------------------------------------------------
# calibrate and multi-SVD
# -----------------------------------------------------------------------------
def roi_for_group(data, group):
    b=data.get("batch_calib",data); return b.get("rois",{}).get(group,b.get("default_roi"))

def run_calib(job, g, roi, out_dir):
    fc=job["fast_calib"]; source=fc.get("pointcloud_source","pcd"); cloud=Path(g["pcd"] if source=="pcd" else g["bag"]); img=Path(g["image"]); mkdir(out_dir)
    cmd=["roslaunch",fc.get("package","fast_calib"),fc.get("calib_launch","calib.launch"),"rviz:=false",f"config_file:={fc['config_file']}",f"pointcloud_source:={source}",f"image_path:={img}",f"output_path:={out_dir}","exit_after_save:=true",f"x_min:={roi['x_min']}",f"x_max:={roi['x_max']}",f"y_min:={roi['y_min']}",f"y_max:={roi['y_max']}",f"z_min:={roi['z_min']}",f"z_max:={roi['z_max']}"]
    cmd.append(("pcd_path:=" if source=="pcd" else "bag_path:=") + str(cloud))
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
    lidar=parse_centers(next((l for l in lines if l.startswith("lidar_centers:")),"")); cam=parse_centers(next((l for l in lines if l.startswith("qr_centers:")),""))
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

def stage_calibrate(job, groups):
    _, out=job_paths(job); single=mkdir(out/"03_single"); multi=mkdir(out/"04_multi"); roi_data=load_yaml(out/"02_roi"/"roi_groups.yaml"); fc=job["fast_calib"]
    max_single=float(fc.get("max_single_rmse",0.03)); max_multi=float(fc.get("max_multi_rmse",0.03)); min_groups=int(fc.get("multi_min_groups",3))
    rows=[]; records={}
    for g in groups:
        roi=roi_for_group(roi_data,g["group"])
        if not roi: rows.append({"group":g["group"],"ok":False,"reason":"missing_roi"}); continue
        print(f"[calib] {g['group']}"); r=run_calib(job,g,roi,single/g["group"])
        if r.get("ok") and r.get("rmse",999)>max_single: r["ok"]=False; r["reason"]="high_single_rmse"
        rows.append(r); a,b=read_record(Path(r.get("output_dir",""))/"circle_center_record.txt")
        if r.get("ok") and a is not None: records[g["group"]]=(a,b)
    fields=sorted(set(k for r in rows for k in r.keys()))
    with (out/"batch_summary.csv").open("w",encoding="utf-8",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    cur=list(records.keys()); history=[]
    while len(cur)>=min_groups:
        R,t,rmse,per=eval_groups(records,cur); history.append({"groups":list(cur),"rmse":ff(rmse),"per_group":{k:ff(v) for k,v in per.items()}})
        if rmse<=max_multi: break
        cur=[g for g in cur if g!=max(per,key=per.get)]
    if len(cur)<min_groups: write_yaml(multi/"multi_result.yaml",{"status":"failed","reason":"not_enough_groups","good_groups":list(records.keys())}); return
    R,t,rmse,per=eval_groups(records,cur); T=np.eye(4); T[:3,:3]=R; T[:3,3]=t
    result={"status":"ok" if rmse<=max_multi else "warn","rmse":ff(rmse),"selected_groups":cur,"T_cam_lidar":[[ff(x) for x in row] for row in T.tolist()],"Rcl":[[ff(x) for x in row] for row in R.tolist()],"Pcl":[ff(x) for x in t.tolist()],"history":history}
    write_yaml(multi/"multi_result.yaml",result); write_yaml(out/"final_extrinsic.yaml",result); (multi/"selected_groups.txt").write_text("\n".join(cur)+"\n",encoding="utf-8")
    print(f"[multi] selected={cur} rmse={rmse:.6f}")

# -----------------------------------------------------------------------------
# verify projection
# -----------------------------------------------------------------------------
def camera_from_config(path):
    c=load_yaml(path)
    return float(c["fx"]),float(c["fy"]),float(c["cx"]),float(c["cy"]),float(c.get("k1",0)),float(c.get("k2",0)),float(c.get("p1",0)),float(c.get("p2",0)),float(c.get("k3",0))

def stage_verify(job, groups):
    if not boolv(job.get("verify",{}).get("enabled",True),True): return
    import cv2
    _,out=job_paths(job); verify=mkdir(out/"05_verify"); final=load_yaml(out/"final_extrinsic.yaml"); T=np.asarray(final["T_cam_lidar"],dtype=float); selected=set(final.get("selected_groups",[]))
    fx,fy,cx,cy,k1,k2,p1,p2,k3=camera_from_config(job["fast_calib"]["config_file"]); K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=float); D=np.array([k1,k2,p1,p2,k3],dtype=float)
    rows=[]
    for g in groups:
        if selected and g["group"] not in selected and not boolv(job.get("verify",{}).get("all_groups",False),False): continue
        img=cv2.imread(g["image"]); bp=Path(g["board_pcd"])
        if img is None or not bp.exists(): continue
        pts=load_pcd_xyz(bp); pts_h=np.hstack([pts,np.ones((pts.shape[0],1))]); pc=(T@pts_h.T).T[:,:3]; pc=pc[pc[:,2]>0.05]
        if pc.shape[0]==0: continue
        uv,_=cv2.projectPoints(pc.reshape(-1,1,3),np.zeros((3,1)),np.zeros((3,1)),K,D); uv=uv.reshape(-1,2); h,w=img.shape[:2]; m=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h); uv=uv[m]
        for u,v in uv: cv2.circle(img,(int(round(u)),int(round(v))),int(job.get("verify",{}).get("point_radius",2)),(0,0,255),-1)
        outimg=verify/f"{g['group']}_board_overlay.png"; cv2.imwrite(str(outimg),img); rows.append({"group":g["group"],"points_projected":uv.shape[0],"overlay":str(outimg)})
        print(f"[verify] {g['group']} projected={uv.shape[0]}")
    with (verify/"verify_summary.csv").open("w",encoding="utf-8",newline="") as f: w=csv.DictWriter(f,fieldnames=["group","points_projected","overlay"]); w.writeheader(); w.writerows(rows)

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
