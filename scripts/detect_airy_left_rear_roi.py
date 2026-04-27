#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Airy left-rear calibration-board ROI detector, version 2.

Dataset root:
  /home/glf/dataDisk/calib/0427/224_left_rear

Reason for this version:
  The Airy left-rear board is not well extracted by a pure axis-aligned ROI.
  In the coarse ROI there are usually two components:
    1) large background/wall/ground component with large Z span
    2) calibration-board component with about 1.3~1.6 m XY span and smaller Z span

This script:
  1. reads pcd_all/save_data_*.pcd or save_data_*/1.pcd
  2. applies a coarse spatial ROI
  3. splits points into 3D grid-connected components
  4. selects the board-like component by size/span
  5. writes FAST-Calib-compatible roi_groups.yaml
  6. saves board_candidates/*.pcd for CloudCompare inspection

Outputs:
  ROOT_DIR/_calib_output/02_roi/
    ├── roi_groups.yaml
    ├── roi_unified.yaml
    ├── roi_summary.csv
    ├── board_candidates/save_data_*_board_candidate.pcd
    └── debug/save_data_*_roi_box.pcd

Run:
  python3 detect_airy_left_rear_roi_v2.py
"""

import csv
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml


# =============================================================================
# User configuration
# =============================================================================

ROOT_DIR = "/home/glf/dataDisk/calib/0427/224_left_rear"

GROUP_PREFIX = "save_data_"
PCD_NAME = "1.pcd"

OUTPUT_DIR_NAME = "_calib_output"
ROI_DIR_NAME = "02_roi"

# -------------------------------------------------------------------------
# Coarse range derived from the board manually cropped in CloudCompare.
#
# Manual board crop union from the uploaded samples is approximately:
#   x: -4.56 ~ -1.88
#   y: -5.08 ~ -2.24
#   z:  1.51 ~  3.37
#
# This coarse ROI is intentionally larger.  It normally contains the board
# plus one large background component.  The component selector below removes
# the background.
# -------------------------------------------------------------------------
COARSE_ROI = {
    "x_min": -5.05,
    "x_max": -1.55,
    "y_min": -5.35,
    "y_max": -1.95,
    "z_min":  1.15,
    "z_max":  3.65,
}

# Component clustering.
# 0.20 worked well on the uploaded Airy PCDs.  Candidate grids are tried in order.
GRID_SIZES = [0.20, 0.16, 0.24]
MIN_COMPONENT_POINTS = 40

# Board-like component constraints.
# Keep these loose enough for sparse Airy scan lines.
BOARD_SPAN_X = (0.70, 2.00)
BOARD_SPAN_Y = (0.70, 2.20)
BOARD_SPAN_Z = (0.15, 1.20)

# The board tends to be about 1.4~1.6 m in x/y span in the cropped examples.
EXPECTED_XY_SPAN = 1.45

# Padding added to each detected component bbox.
PAD_X = 0.08
PAD_Y = 0.08
PAD_Z = 0.08

# If True, the unified ROI is the fixed coarse ROI.
# If False, the unified ROI is the union of detected per-group board bboxes.
USE_COARSE_AS_UNIFIED_ROI = False

# Save debug outputs.
SAVE_BOARD_CANDIDATE = True
SAVE_ROI_BOX = True

# Stop if all groups fail.
REQUIRE_AT_LEAST_ONE_SUCCESS = True

# =============================================================================


def natural_key(path):
    parts = re.split(r"(\d+)", str(path))
    return [int(p) if p.isdigit() else p for p in parts]


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ff(x: float) -> float:
    return float(f"{float(x):.6f}")


# =============================================================================
# PCD loader, supports repeated "_" fields
# =============================================================================

def parse_pcd_header(path: Path):
    header = {}
    lines = []
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"PCD header incomplete: {path}")
            s = line.decode("utf-8", errors="ignore").strip()
            lines.append(s)
            if s and not s.startswith("#"):
                parts = s.split()
                header[parts[0].upper()] = parts[1:]
            if s.upper().startswith("DATA"):
                offset = f.tell()
                break
    return header, lines, offset


def unique_field_names(fields):
    used = {}
    unique = []
    first = {}
    for name in fields:
        if name in used:
            used[name] += 1
            uname = f"{name}__{used[name]}"
        else:
            used[name] = 0
            uname = name
            first[name] = uname
        unique.append(uname)
    return unique, first


def build_dtype(fields, sizes, types, counts):
    unique, first = unique_field_names(fields)
    dtype_fields = []
    for name, size, typ, count in zip(unique, sizes, types, counts):
        size = int(size)
        count = int(count)

        if typ == "F":
            base = {4: np.float32, 8: np.float64}.get(size)
        elif typ == "U":
            base = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}.get(size)
        elif typ == "I":
            base = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}.get(size)
        else:
            base = None

        if base is None:
            raise RuntimeError(f"Unsupported PCD field TYPE={typ}, SIZE={size}")

        if count == 1:
            dtype_fields.append((name, base))
        else:
            dtype_fields.append((name, base, (count,)))

    return np.dtype(dtype_fields), first


def load_pcd_xyz_intensity_ring(path: Path):
    header, lines, offset = parse_pcd_header(path)

    fields = header["FIELDS"]
    sizes = header["SIZE"]
    types = header["TYPE"]
    counts = header.get("COUNT", ["1"] * len(fields))
    data_type = header["DATA"][0].lower()

    for required in ("x", "y", "z"):
        if required not in fields:
            raise RuntimeError(f"PCD lacks {required}: {path}")

    if "POINTS" in header:
        n = int(header["POINTS"][0])
    else:
        n = int(header["WIDTH"][0]) * int(header.get("HEIGHT", ["1"])[0])

    intensity = None
    ring = None

    if data_type == "binary":
        dtype, first = build_dtype(fields, sizes, types, counts)
        with path.open("rb") as f:
            f.seek(offset)
            arr = np.fromfile(f, dtype=dtype, count=n)

        xyz = np.stack(
            [arr[first["x"]], arr[first["y"]], arr[first["z"]]],
            axis=1
        ).astype(np.float64)

        if "intensity" in first:
            intensity = arr[first["intensity"]].astype(np.float64)
        elif "reflectivity" in first:
            intensity = arr[first["reflectivity"]].astype(np.float64)

        if "ring" in first:
            ring = arr[first["ring"]].astype(np.float64)
        elif "line" in first:
            ring = arr[first["line"]].astype(np.float64)

    elif data_type == "ascii":
        start = next(i + 1 for i, s in enumerate(lines) if s.upper().startswith("DATA"))
        arr = np.loadtxt(path.read_text(encoding="utf-8", errors="ignore").splitlines()[start:])
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        xyz = arr[:, [fields.index("x"), fields.index("y"), fields.index("z")]].astype(np.float64)

        if "intensity" in fields:
            intensity = arr[:, fields.index("intensity")].astype(np.float64)
        elif "reflectivity" in fields:
            intensity = arr[:, fields.index("reflectivity")].astype(np.float64)

        if "ring" in fields:
            ring = arr[:, fields.index("ring")].astype(np.float64)
        elif "line" in fields:
            ring = arr[:, fields.index("line")].astype(np.float64)

    else:
        raise RuntimeError(f"Unsupported PCD DATA type: {data_type}. Use ascii/binary PCD.")

    valid = np.isfinite(xyz).all(axis=1)
    xyz = xyz[valid]
    if intensity is not None:
        intensity = intensity[valid]
    if ring is not None:
        ring = ring[valid]

    if xyz.shape[0] == 0:
        raise RuntimeError(f"No valid xyz points in {path}")

    return xyz, intensity, ring


def write_xyz_pcd(path: Path, pts: np.ndarray):
    mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {pts.shape[0]}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {pts.shape[0]}\n")
        f.write("DATA ascii\n")
        for p in pts:
            f.write(f"{p[0]:.9f} {p[1]:.9f} {p[2]:.9f}\n")


# =============================================================================
# ROI and component selection
# =============================================================================

def crop_roi(xyz: np.ndarray, roi: Dict[str, float]):
    return xyz[
        (xyz[:, 0] >= roi["x_min"]) & (xyz[:, 0] <= roi["x_max"]) &
        (xyz[:, 1] >= roi["y_min"]) & (xyz[:, 1] <= roi["y_max"]) &
        (xyz[:, 2] >= roi["z_min"]) & (xyz[:, 2] <= roi["z_max"])
    ]


def bbox(pts: np.ndarray):
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    span = mx - mn
    return mn, mx, span


def grid_components(points: np.ndarray, grid_size: float, min_points: int):
    if points.shape[0] == 0:
        return []

    cells = np.floor(points / grid_size).astype(np.int64)
    cell_map = defaultdict(list)

    for i, c in enumerate(cells):
        cell_map[(int(c[0]), int(c[1]), int(c[2]))].append(i)

    visited = set()
    components = []

    neigh = [(i, j, k) for i in (-1, 0, 1)
                      for j in (-1, 0, 1)
                      for k in (-1, 0, 1)]

    for cell in list(cell_map.keys()):
        if cell in visited:
            continue

        q = deque([cell])
        visited.add(cell)
        comp_cells = []

        while q:
            c = q.popleft()
            comp_cells.append(c)

            for dx, dy, dz in neigh:
                nb = (c[0] + dx, c[1] + dy, c[2] + dz)
                if nb in visited:
                    continue
                if nb in cell_map:
                    visited.add(nb)
                    q.append(nb)

        indices = []
        for c in comp_cells:
            indices.extend(cell_map[c])

        if len(indices) >= min_points:
            components.append(np.asarray(indices, dtype=np.int64))

    return components


def is_board_like(span: np.ndarray) -> bool:
    return (
        BOARD_SPAN_X[0] <= span[0] <= BOARD_SPAN_X[1] and
        BOARD_SPAN_Y[0] <= span[1] <= BOARD_SPAN_Y[1] and
        BOARD_SPAN_Z[0] <= span[2] <= BOARD_SPAN_Z[1]
    )


def score_candidate(points: np.ndarray, span: np.ndarray) -> float:
    # Reject background-like components by span constraints first.
    # Then prefer:
    #   - enough points
    #   - x/y span close to the known board size
    #   - z span not too large
    n = points.shape[0]
    score = float(n)
    score -= 120.0 * abs(span[0] - EXPECTED_XY_SPAN)
    score -= 120.0 * abs(span[1] - EXPECTED_XY_SPAN)
    score -= 150.0 * max(0.0, span[2] - 0.60)
    return score


def select_board_component(coarse_pts: np.ndarray):
    diagnostics = []

    for grid in GRID_SIZES:
        comps = grid_components(coarse_pts, grid, MIN_COMPONENT_POINTS)
        candidates = []

        for comp_id, comp in enumerate(comps):
            pts = coarse_pts[comp]
            mn, mx, span = bbox(pts)

            info = {
                "grid": grid,
                "component_id": comp_id,
                "points": int(pts.shape[0]),
                "x_span": ff(span[0]),
                "y_span": ff(span[1]),
                "z_span": ff(span[2]),
                "x_min": ff(mn[0]),
                "x_max": ff(mx[0]),
                "y_min": ff(mn[1]),
                "y_max": ff(mx[1]),
                "z_min": ff(mn[2]),
                "z_max": ff(mx[2]),
                "board_like": is_board_like(span),
            }
            diagnostics.append(info)

            if not info["board_like"]:
                continue

            score = score_candidate(pts, span)
            candidates.append((score, grid, comp_id, pts, mn, mx, span))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            score, grid, comp_id, pts, mn, mx, span = candidates[0]
            return {
                "ok": True,
                "method": f"component_grid_{grid:.2f}",
                "score": float(score),
                "component_id": int(comp_id),
                "points": pts,
                "min": mn,
                "max": mx,
                "span": span,
                "diagnostics": diagnostics,
            }

    return {
        "ok": False,
        "method": "no_board_like_component",
        "diagnostics": diagnostics,
    }


def expand_bbox(mn: np.ndarray, mx: np.ndarray):
    pad = np.array([PAD_X, PAD_Y, PAD_Z], dtype=np.float64)
    return mn - pad, mx + pad


def roi_dict(rmin: np.ndarray, rmax: np.ndarray) -> Dict[str, float]:
    return {
        "x_min": ff(rmin[0]),
        "x_max": ff(rmax[0]),
        "y_min": ff(rmin[1]),
        "y_max": ff(rmax[1]),
        "z_min": ff(rmin[2]),
        "z_max": ff(rmax[2]),
    }


def make_roi_box_points(rmin, rmax, n_per_edge=80):
    xmin, ymin, zmin = rmin
    xmax, ymax, zmax = rmax

    corners = [
        np.array([xmin, ymin, zmin]), np.array([xmax, ymin, zmin]),
        np.array([xmax, ymax, zmin]), np.array([xmin, ymax, zmin]),
        np.array([xmin, ymin, zmax]), np.array([xmax, ymin, zmax]),
        np.array([xmax, ymax, zmax]), np.array([xmin, ymax, zmax]),
    ]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    pts = []
    for a, b in edges:
        for t in np.linspace(0.0, 1.0, n_per_edge):
            pts.append(corners[a] * (1.0 - t) + corners[b] * t)

    return np.asarray(pts, dtype=np.float64)


# =============================================================================
# File search and outputs
# =============================================================================

def find_pcds(root: Path) -> List[Tuple[str, Path]]:
    pcds = []

    # Layout 1: pcd_all/save_data_*.pcd
    pcd_all = root / "pcd_all"
    if pcd_all.exists():
        for p in sorted(pcd_all.glob(f"{GROUP_PREFIX}*.pcd"), key=natural_key):
            pcds.append((p.stem, p))

    # Layout 2: save_data_*/1.pcd
    for d in sorted(root.glob(f"{GROUP_PREFIX}*"), key=natural_key):
        if d.is_dir():
            p = d / PCD_NAME
            if p.exists():
                pcds.append((d.name, p))

    # Layout 3: root/save_data_*.pcd
    for p in sorted(root.glob(f"{GROUP_PREFIX}*.pcd"), key=natural_key):
        pcds.append((p.stem, p))

    # Fallback.
    if not pcds:
        for p in sorted(root.rglob("*.pcd"), key=natural_key):
            if OUTPUT_DIR_NAME in p.parts:
                continue
            pcds.append((p.stem, p))

    seen = set()
    out = []
    for group, path in pcds:
        rp = str(path.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        out.append((group, path))
    return out


def write_yaml(path: Path, obj):
    mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def main():
    root = Path(ROOT_DIR).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {root}")

    output_root = root / OUTPUT_DIR_NAME
    roi_dir = mkdir(output_root / ROI_DIR_NAME)
    board_dir = mkdir(roi_dir / "board_candidates")
    debug_dir = mkdir(roi_dir / "debug")

    pcds = find_pcds(root)
    if not pcds:
        raise RuntimeError(f"No PCD files found under: {root}")

    print("=" * 90)
    print(f"[Root]        {root}")
    print(f"[PCD count]   {len(pcds)}")
    print(f"[Coarse ROI]  {COARSE_ROI}")
    print(f"[Grid sizes]  {GRID_SIZES}")
    print("=" * 90)

    rows = []
    component_rows = []
    rois = {}
    roi_mins = []
    roi_maxs = []

    for group, pcd_path in pcds:
        print(f"\n[Processing] {group}: {pcd_path}")

        row = {
            "group": group,
            "pcd": str(pcd_path),
            "status": "",
            "method": "",
            "score": "",
            "component_id": "",
            "points_total": "",
            "points_coarse": "",
            "points_board": "",
            "x_min": "", "x_max": "",
            "y_min": "", "y_max": "",
            "z_min": "", "z_max": "",
            "span_x": "", "span_y": "", "span_z": "",
            "board_candidate": "",
            "roi_box": "",
            "error": "",
        }

        try:
            xyz, intensity, ring = load_pcd_xyz_intensity_ring(pcd_path)
            coarse = crop_roi(xyz, COARSE_ROI)

            if coarse.shape[0] < MIN_COMPONENT_POINTS:
                raise RuntimeError(f"Too few points in coarse ROI: {coarse.shape[0]}")

            result = select_board_component(coarse)

            # Save diagnostics for all components.
            for d in result.get("diagnostics", []):
                component_rows.append({"group": group, **d})

            if not result["ok"]:
                raise RuntimeError("No board-like component found. Check component_diagnostics.csv and adjust constraints.")

            board = result["points"]
            mn = result["min"]
            mx = result["max"]
            span = result["span"]
            rmin, rmax = expand_bbox(mn, mx)
            roi = roi_dict(rmin, rmax)

            rois[group] = roi
            roi_mins.append(rmin)
            roi_maxs.append(rmax)

            board_path = board_dir / f"{group}_board_candidate.pcd"
            box_path = debug_dir / f"{group}_roi_box.pcd"

            if SAVE_BOARD_CANDIDATE:
                write_xyz_pcd(board_path, board)
            if SAVE_ROI_BOX:
                write_xyz_pcd(box_path, make_roi_box_points(rmin, rmax))

            row.update({
                "status": "ok",
                "method": result["method"],
                "score": ff(result["score"]),
                "component_id": result["component_id"],
                "points_total": int(xyz.shape[0]),
                "points_coarse": int(coarse.shape[0]),
                "points_board": int(board.shape[0]),
                **roi,
                "span_x": ff(span[0]),
                "span_y": ff(span[1]),
                "span_z": ff(span[2]),
                "board_candidate": str(board_path),
                "roi_box": str(box_path),
            })

            print(f"  ok: method={row['method']}, board_points={row['points_board']}, span=({row['span_x']}, {row['span_y']}, {row['span_z']})")
            print(f"  roi: {roi}")
            print(f"  board_candidate: {board_path}")

        except Exception as e:
            row.update({"status": "failed", "error": str(e)})
            print(f"  failed: {e}")

        rows.append(row)

    # Write summaries.
    summary_path = roi_dir / "roi_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    diag_path = roi_dir / "component_diagnostics.csv"
    if component_rows:
        with diag_path.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(component_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(component_rows)

    if REQUIRE_AT_LEAST_ONE_SUCCESS and not roi_mins:
        raise RuntimeError("All ROI detections failed.")

    if USE_COARSE_AS_UNIFIED_ROI:
        unified = {k: ff(v) for k, v in COARSE_ROI.items()}
    else:
        umin = np.vstack(roi_mins).min(axis=0)
        umax = np.vstack(roi_maxs).max(axis=0)
        unified = roi_dict(umin, umax)

    roi_yaml = {
        "batch_calib": {
            "default_roi": unified,
            "rois": rois,
        }
    }

    write_yaml(roi_dir / "roi_groups.yaml", roi_yaml)
    write_yaml(roi_dir / "roi_unified.yaml", {"default_roi": unified})

    print("\n" + "=" * 90)
    print("[Unified ROI]")
    for k, v in unified.items():
        print(f"{k}: {v}")
    print(f"\n[Summary]       {summary_path}")
    print(f"[Diagnostics]   {diag_path}")
    print(f"[ROI groups]    {roi_dir / 'roi_groups.yaml'}")
    print(f"[ROI unified]   {roi_dir / 'roi_unified.yaml'}")
    print(f"[Candidates]    {board_dir}")
    print(f"[Debug boxes]    {debug_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
