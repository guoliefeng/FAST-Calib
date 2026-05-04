#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Render FAST-Calib LiDAR circle-center order images.

For each group, this script reads circle_center_record.txt and an optional PCD
board cloud, projects them onto the board plane, and saves a PNG with visible
center indices 0/1/2/3. This is intended for fast engineering review of whether
LiDAR center numbering is stable.

Inputs:
  <data_dir>/_calib_output/03_calibrate/<group>/circle_center_record.txt
  optional PCDs from common FAST-Calib output paths
  optional <output_dir>/05_center_order_audit/center_order_audit.csv

Outputs:
  <output_dir>/05_center_images/<group>_center_order.png
  <output_dir>/05_center_images/index.html
"""

import argparse
import csv
import html
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(
        "[ERROR] matplotlib is required to render center images.\n"
        "Install it with: pip3 install matplotlib\n"
        f"Original error: {exc}"
    )


CENTER_RE = re.compile(r"\{([^}]*)\}")


def natural_key(v: str) -> List[object]:
    parts: List[object] = [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(v))]
    parts.append(str(v))
    return parts


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


def read_circle_record(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lidar_line = next((l for l in reversed(lines) if l.startswith("lidar_centers:")), "")
    qr_line = next((l for l in reversed(lines) if l.startswith("qr_centers:")), "")
    lidar = parse_centers_from_line(lidar_line)
    qr = parse_centers_from_line(qr_line)
    if len(lidar) != 4 or len(qr) != 4:
        raise RuntimeError(f"Expected 4 lidar centers and 4 qr centers in {path}, got lidar={len(lidar)}, qr={len(qr)}")
    return np.asarray(lidar, dtype=float), np.asarray(qr, dtype=float)


def resolve_records(output_dir: Path, groups_arg: str) -> List[Path]:
    primary_roots = [output_dir / "03_calibrate", output_dir / "03_single"]
    if groups_arg.lower() not in ("all", "*"):
        records: List[Path] = []
        missing: List[str] = []
        for group in [x.strip() for x in groups_arg.split(",") if x.strip()]:
            found: Optional[Path] = None
            for root in primary_roots:
                p = root / group / "circle_center_record.txt"
                if p.exists():
                    found = p
                    break
            if found is None:
                matches = sorted(output_dir.glob(f"**/{group}/circle_center_record.txt"), key=lambda p: len(str(p)))
                found = matches[0] if matches else None
            if found is None:
                missing.append(group)
            else:
                records.append(found)
        if missing:
            raise FileNotFoundError("Missing circle_center_record.txt for group(s): " + ", ".join(missing))
        return records

    for root in primary_roots:
        if root.exists():
            records = sorted(root.glob("*/circle_center_record.txt"), key=lambda p: natural_key(p.parent.name))
            if records:
                return records
    records = sorted(output_dir.glob("**/circle_center_record.txt"), key=lambda p: natural_key(p.parent.name))
    if not records:
        raise RuntimeError(f"No circle_center_record.txt found under {output_dir}")
    return records


def find_cloud(data_dir: Path, output_dir: Path, group: str) -> Optional[Path]:
    candidates = [
        data_dir / group / "board_candidate.pcd",
        output_dir / "02_roi" / "board_candidates" / f"{group}_board_candidate.pcd",
        data_dir / group / "1.pcd",
        data_dir / group / "cloud.pcd",
        output_dir / "02_roi" / "debug" / f"{group}_roi_box.pcd",
        output_dir / "03_calibrate" / group / "colored_cloud.pcd",
        output_dir / "03_single" / group / "colored_cloud.pcd",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def read_pcd_xyz(path: Path, max_points: int) -> np.ndarray:
    fields: List[str] = []
    points_declared: Optional[int] = None
    data_ascii = False
    header_lines = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            header_lines += 1
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            key = s.split()[0].upper()
            if key == "FIELDS":
                fields = s.split()[1:]
            elif key == "POINTS":
                try:
                    points_declared = int(s.split()[1])
                except Exception:
                    points_declared = None
            elif key == "DATA":
                if "ascii" not in s.lower():
                    raise RuntimeError(f"Only ASCII PCD is supported: {path}")
                data_ascii = True
                break

    if not data_ascii:
        raise RuntimeError(f"Invalid PCD or missing DATA ascii: {path}")
    if not fields:
        raise RuntimeError(f"Missing FIELDS line in PCD: {path}")
    try:
        xi, yi, zi = fields.index("x"), fields.index("y"), fields.index("z")
    except ValueError as exc:
        raise RuntimeError(f"PCD must contain x y z fields: {path}") from exc

    step = 1
    if points_declared and max_points > 0 and points_declared > max_points:
        step = int(math.ceil(points_declared / max_points))

    pts: List[List[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(header_lines):
            next(f, None)
        for line_idx, line in enumerate(f):
            if step > 1 and (line_idx % step) != 0:
                continue
            parts = line.strip().split()
            if len(parts) <= max(xi, yi, zi):
                continue
            try:
                pts.append([float(parts[xi]), float(parts[yi]), float(parts[zi])])
            except ValueError:
                continue
            if max_points > 0 and len(pts) >= max_points:
                break
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 3), dtype=float)


def normalize(v: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 1e-12 and np.isfinite(n):
        return v / n
    if fallback is not None:
        return normalize(fallback)
    return np.array([1.0, 0.0, 0.0], dtype=float)


def board_projection_axes(centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = centers.mean(axis=0)
    centered = centers - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=True)
    normal = normalize(vh[-1], np.array([0.0, 0.0, 1.0]))
    axis_u = centers[1] - centers[0]
    axis_u = axis_u - normal * float(np.dot(axis_u, normal))
    if np.linalg.norm(axis_u) < 1e-9:
        axis_u = vh[0]
    axis_u = normalize(axis_u, vh[0])
    axis_v = normalize(np.cross(normal, axis_u), vh[1])
    if float(np.dot(centers[3] - centers[0], axis_v)) < 0:
        axis_v = -axis_v
    return centroid, axis_u, axis_v


def project_to_board(points: np.ndarray, origin: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 2), dtype=float)
    rel = points - origin.reshape(1, 3)
    return np.column_stack((rel @ axis_u, rel @ axis_v))


def crop_points(raw_2d: np.ndarray, center_2d: np.ndarray, margin_ratio: float = 1.1) -> np.ndarray:
    if raw_2d.size == 0:
        return raw_2d
    mn = center_2d.min(axis=0)
    mx = center_2d.max(axis=0)
    span = np.maximum(mx - mn, 0.2)
    margin = np.maximum(span * margin_ratio, 0.25)
    lo = mn - margin
    hi = mx + margin
    mask = (raw_2d[:, 0] >= lo[0]) & (raw_2d[:, 0] <= hi[0]) & (raw_2d[:, 1] >= lo[1]) & (raw_2d[:, 1] <= hi[1])
    return raw_2d[mask]


def load_audit_rows(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if not path or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {str(row.get("group", "")): dict(row) for row in csv.DictReader(f) if row.get("group")}


def render_group_image(group: str,
                       record: Path,
                       cloud: Optional[Path],
                       audit_row: Optional[Dict[str, str]],
                       out_png: Path,
                       max_points: int,
                       dpi: int) -> None:
    lidar, _ = read_circle_record(record)
    origin, axis_u, axis_v = board_projection_axes(lidar)
    center_2d = project_to_board(lidar, origin, axis_u, axis_v)

    raw_2d = np.empty((0, 2), dtype=float)
    raw_count = 0
    if cloud and cloud.exists():
        try:
            raw_pts = read_pcd_xyz(cloud, max_points=max_points)
            raw_count = int(raw_pts.shape[0])
            raw_2d = crop_points(project_to_board(raw_pts, origin, axis_u, axis_v), center_2d)
        except Exception as exc:
            print(f"[render][WARN] Cannot read cloud for {group}: {cloud}: {exc}")

    mkdir(out_png.parent)
    fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=dpi)
    ax.set_facecolor("#f7f7f5")
    if raw_2d.size:
        ax.scatter(
            raw_2d[:, 0],
            raw_2d[:, 1],
            s=1.15,
            c="#4f5965",
            alpha=0.58,
            linewidths=0,
            marker=".",
            label=f"board point cloud ({raw_2d.shape[0]}/{raw_count} pts)",
        )
    else:
        ax.text(0.02, 0.02, "No board PCD found; showing centers only.", transform=ax.transAxes, fontsize=9, color="0.35", va="bottom")

    order = [0, 1, 2, 3, 0]
    ax.plot(center_2d[order, 0], center_2d[order, 1], "k--", linewidth=1.4, alpha=0.75, label="current order 0→1→2→3→0")
    ax.annotate("", xy=(center_2d[1, 0], center_2d[1, 1]), xytext=(center_2d[0, 0], center_2d[0, 1]),
                arrowprops=dict(arrowstyle="->", linewidth=2.0, color="black"))

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
    for idx, xy in enumerate(center_2d):
        ax.scatter([xy[0]], [xy[1]], s=260, c=colors[idx], edgecolors="black", linewidths=1.2, zorder=5)
        ax.text(xy[0], xy[1], str(idx), ha="center", va="center", color="white", fontsize=15, fontweight="bold", zorder=6)
        ax.text(xy[0], xy[1] - 0.055, f"center_{idx}", ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.75", alpha=0.85), zorder=7)

    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        mid = 0.5 * (center_2d[a] + center_2d[b])
        d = float(np.linalg.norm(lidar[a] - lidar[b]))
        ax.text(mid[0], mid[1], f"{d:.3f} m", fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.72))

    title = f"{group} - LiDAR circle center order"
    if cloud:
        title += f"\ncloud={cloud.name}, displayed_points={int(raw_2d.shape[0])}"
    if audit_row:
        suspicious = audit_row.get("suspicious", "")
        ambiguous = audit_row.get("ambiguous", "")
        title += (f"\nbest_perm={audit_row.get('best_perm', '')}, current_rmse={audit_row.get('current_rmse', '')}, "
                  f"best_rmse={audit_row.get('best_rmse', '')}, improvement={audit_row.get('improvement', '')}")
        if suspicious == "YES" or ambiguous == "YES":
            title += f"\nSUSPICIOUS={suspicious}, AMBIGUOUS={ambiguous}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("board-plane u axis (m), aligned with current center_0 → center_1")
    ax.set_ylabel("board-plane v axis (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.65)
    ax.legend(loc="best", fontsize=8)
    mn = center_2d.min(axis=0)
    mx = center_2d.max(axis=0)
    span = np.maximum(mx - mn, 0.2)
    pad = np.maximum(span * 0.8, 0.25)
    ax.set_xlim(mn[0] - pad[0], mx[0] + pad[0])
    ax.set_ylim(mn[1] - pad[1], mx[1] + pad[1])
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def write_html_index(out_root: Path, rows: List[Dict[str, str]]) -> None:
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>FAST-Calib LiDAR center order images</title>",
        "<style>body{font-family:Arial,sans-serif;margin:20px;background:#fafafa}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px}.card{background:white;border:1px solid #ddd;border-radius:8px;padding:10px;box-shadow:0 1px 3px #ddd}img{width:100%;height:auto;border:1px solid #eee}code{font-size:12px}.bad{color:#b00020;font-weight:bold}</style>",
        "</head><body><h1>FAST-Calib LiDAR center order images</h1>",
        "<p>Each image projects the board point cloud and four LiDAR centers onto the fitted board plane. The black arrow shows current center_0 → center_1.</p>",
        "<div class='grid'>",
    ]
    for row in rows:
        group = html.escape(row["group"])
        rel = html.escape(row["image_rel"])
        suspicious = html.escape(row.get("suspicious", ""))
        cls = "bad" if suspicious == "YES" else ""
        lines.extend([
            "<div class='card'>",
            f"<h3 class='{cls}'>{group}</h3>",
            f"<p>best_perm=<code>{html.escape(row.get('best_perm', ''))}</code><br>current_rmse=<code>{html.escape(row.get('current_rmse', ''))}</code><br>best_rmse=<code>{html.escape(row.get('best_rmse', ''))}</code><br>suspicious=<code>{suspicious}</code></p>",
            f"<a href='{rel}'><img src='{rel}' alt='{group}'></a>",
            "</div>",
        ])
    lines.extend(["</div></body></html>"])
    (out_root / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Render LiDAR circle-center order PNGs for quick visual inspection.")
    ap.add_argument("--data-dir", required=True, help="Example: /home/glf/dataDisk/calib/c2l/223/front")
    ap.add_argument("--output-dir", help="Default: <data-dir>/_calib_output")
    ap.add_argument("--groups", default="all", help="Comma-separated groups or all. Default: all")
    ap.add_argument("--audit-csv", help="Default: <output_dir>/05_center_order_audit/center_order_audit.csv")
    ap.add_argument("--max-points", type=int, default=80000, help="Maximum PCD points to draw per group. Default: 80000")
    ap.add_argument("--dpi", type=int, default=160, help="PNG DPI. Default: 160")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "_calib_output"
    audit_csv = Path(args.audit_csv).expanduser().resolve() if args.audit_csv else output_dir / "05_center_order_audit" / "center_order_audit.csv"
    audit_rows = load_audit_rows(audit_csv)
    records = resolve_records(output_dir, args.groups)
    out_root = mkdir(output_dir / "05_center_images")

    html_rows: List[Dict[str, str]] = []
    for record in records:
        group = record.parent.name
        cloud = find_cloud(data_dir, output_dir, group)
        out_png = out_root / f"{group}_center_order.png"
        audit_row = audit_rows.get(group)
        print(f"[render] group={group}")
        print(f"  record: {record}")
        print(f"  cloud : {cloud if cloud else '(not found; centers only)'}")
        print(f"  image : {out_png}")
        render_group_image(group, record, cloud, audit_row, out_png, args.max_points, args.dpi)
        html_rows.append({
            "group": group,
            "image_rel": str(out_png.relative_to(out_root)),
            "suspicious": audit_row.get("suspicious", "") if audit_row else "",
            "best_perm": audit_row.get("best_perm", "") if audit_row else "",
            "current_rmse": audit_row.get("current_rmse", "") if audit_row else "",
            "best_rmse": audit_row.get("best_rmse", "") if audit_row else "",
        })

    write_html_index(out_root, html_rows)
    open_script = out_root / "open_center_images.sh"
    open_script.write_text(f"#!/usr/bin/env bash\nset -e\nxdg-open \"{out_root / 'index.html'}\"\n", encoding="utf-8")
    open_script.chmod(0o755)
    print(f"[render] index: {out_root / 'index.html'}")
    print(f"[render] open : {open_script}")


if __name__ == "__main__":
    main()
