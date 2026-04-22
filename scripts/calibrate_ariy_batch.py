#!/usr/bin/python3
import argparse
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2


TARGET_NUM_CIRCLES = 4


def make_aruco_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        return cv2.aruco.DetectorParameters_create()
    return cv2.aruco.DetectorParameters()


def root_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def expand_find_fast_calib(value: str) -> str:
    return value.replace("$(find fast_calib)", str(root_dir()))


def load_config(path: str) -> dict:
    cfg = {}
    scalar_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*):\s*([^#]+)")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = scalar_re.match(line)
            if not m:
                continue
            key, raw = m.group(1), m.group(2).strip().strip("\"'")
            try:
                cfg[key] = float(raw)
            except ValueError:
                cfg[key] = expand_find_fast_calib(raw)
    return cfg


def detect_qr_centers(image_path: str, cfg: dict, debug_path: str = None):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    K = np.array(
        [[cfg["fx"], 0.0, cfg["cx"]], [0.0, cfg["fy"], cfg["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.array([cfg["k1"], cfg["k2"], cfg["p1"], cfg["p2"], 0.0], dtype=np.float64)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = make_aruco_detector_parameters()
    corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=params)
    if ids is None or len(ids) < int(cfg.get("min_detected_markers", 3)):
        raise RuntimeError(f"Only detected {0 if ids is None else len(ids)} markers in {image_path}")

    marker_size = float(cfg["marker_size"])
    width = float(cfg["delta_width_qr_center"])
    height = float(cfg["delta_height_qr_center"])
    circle_w = float(cfg["delta_width_circles"]) / 2.0
    circle_h = float(cfg["delta_height_circles"]) / 2.0

    board_corners = []
    board_circle_centers = []
    for i in range(4):
        x_qr_center = -1 if (i % 3) == 0 else 1
        y_qr_center = 1 if i < 2 else -1
        x_center = x_qr_center * width
        y_center = y_qr_center * height
        board_circle_centers.append([x_qr_center * circle_w, y_qr_center * circle_h, 0.0])

        marker_corners = []
        for j in range(4):
            x_qr = -1 if (j % 3) == 0 else 1
            y_qr = 1 if j < 2 else -1
            marker_corners.append(
                [x_center + x_qr * marker_size / 2.0, y_center + y_qr * marker_size / 2.0, 0.0]
            )
        board_corners.append(np.asarray(marker_corners, dtype=np.float32))

    board_ids = np.asarray([1, 2, 4, 3], dtype=np.int32)
    board = cv2.aruco.Board_create(board_corners, dictionary, board_ids)

    ok, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, K, dist, None, None)
    if ok <= 0:
        raise RuntimeError(f"estimatePoseBoard failed for {image_path}")

    R, _ = cv2.Rodrigues(rvec)
    board_circle_centers = np.asarray(board_circle_centers, dtype=np.float64)
    centers_cam = board_circle_centers @ R.T + tvec.reshape(1, 3)

    dbg = image.copy()
    cv2.aruco.drawDetectedMarkers(dbg, corners, ids)
    for p in centers_cam:
        uv, _ = cv2.projectPoints(p.reshape(1, 3), np.zeros((3, 1)), np.zeros((3, 1)), K, dist)
        u, v = np.round(uv.reshape(2)).astype(int)
        cv2.circle(dbg, (u, v), 5, (0, 255, 0), -1)
    if debug_path:
        cv2.imwrite(debug_path, dbg)

    return centers_cam


def read_roi_points(bag_path: str, topic: str, roi):
    pts = []
    rings = []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            for p in pc2.read_points(msg, field_names=("x", "y", "z", "ring"), skip_nans=True):
                x, y, z, ring = p
                if roi[0] <= x <= roi[1] and roi[2] <= y <= roi[3] and roi[4] <= z <= roi[5]:
                    pts.append((x, y, z))
                    rings.append(int(ring))
    if not pts:
        raise RuntimeError(f"No points in ROI for {bag_path}")
    return np.asarray(pts, dtype=np.float64), np.asarray(rings, dtype=np.int32)


def fit_plane(points, seed=0):
    rng = np.random.default_rng(seed)
    sample = points
    if len(points) > 250000:
        sample = points[rng.choice(len(points), 250000, replace=False)]

    best_count = -1
    best_n = None
    best_d = None
    for _ in range(1200):
        idx = rng.choice(len(sample), 3, replace=False)
        p1, p2, p3 = sample[idx]
        n = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = -float(np.dot(n, p1))
        count = int((np.abs(sample @ n + d) < 0.01).sum())
        if count > best_count:
            best_count = count
            best_n = n
            best_d = d

    dist = np.abs(points @ best_n + best_d)
    inliers = points[dist < 0.01]
    if len(inliers) >= 3:
        centroid = inliers.mean(axis=0)
        _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
        best_n = vh[-1]
        best_n /= np.linalg.norm(best_n)
        best_d = -float(np.dot(best_n, centroid))
    return best_n, best_d


def extract_ring_edges(points, rings, normal, d):
    edges = []
    norm_n = np.linalg.norm(normal)
    for ring in np.unique(rings):
        idxs = np.nonzero(rings == ring)[0]
        if len(idxs) < 10:
            continue
        for k in range(1, len(idxs) - 1):
            i = idxs[k]
            p = points[i]
            if abs(float(np.dot(normal, p) + d)) / norm_n >= 0.03:
                continue
            prev_p = points[idxs[k - 1]]
            next_p = points[idxs[k + 1]]
            if np.linalg.norm(p - prev_p) > 0.10 or np.linalg.norm(p - next_p) > 0.10:
                edges.append(p)
    if not edges:
        raise RuntimeError("No edge points extracted")
    return np.asarray(edges, dtype=np.float64)


def align_edges_to_plane(edges, normal):
    normal = normal / np.linalg.norm(normal)
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z_axis)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-9:
        R = np.eye(3)
    else:
        axis /= axis_norm
        angle = math.acos(float(np.clip(np.dot(normal, z_axis), -1.0, 1.0)))
        K = np.array(
            [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
        )
        R = np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)
    aligned = (R @ edges.T).T
    return aligned, R


def fit_circle_from_three(p1, p2, p3):
    A = np.array(
        [[2.0 * (p2[0] - p1[0]), 2.0 * (p2[1] - p1[1])],
         [2.0 * (p3[0] - p1[0]), 2.0 * (p3[1] - p1[1])]]
    )
    b = np.array(
        [p2[0] ** 2 + p2[1] ** 2 - p1[0] ** 2 - p1[1] ** 2,
         p3[0] ** 2 + p3[1] ** 2 - p1[0] ** 2 - p1[1] ** 2]
    )
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    c = np.linalg.solve(A, b)
    r = float(np.linalg.norm(p1 - c))
    return c, r


def detect_circle_candidates(xy, circle_radius, seed=0, max_candidates=40):
    rng = np.random.default_rng(seed)
    work = xy.copy()
    candidates = []
    for _ in range(max_candidates):
        if len(work) <= 3:
            break
        best_count = 0
        best_center = None
        best_radius = None
        best_inliers = None
        for _ in range(6000):
            ids = rng.choice(len(work), 3, replace=False)
            result = fit_circle_from_three(work[ids[0]], work[ids[1]], work[ids[2]])
            if result is None:
                continue
            center, radius = result
            if not (circle_radius - 0.03 <= radius <= circle_radius + 0.03):
                continue
            dist = np.abs(np.linalg.norm(work - center.reshape(1, 2), axis=1) - radius)
            inliers = np.nonzero(dist < 0.02)[0]
            if len(inliers) > best_count:
                best_count = len(inliers)
                best_center = center
                best_radius = radius
                best_inliers = inliers
        if best_inliers is None or best_count < 5:
            break
        candidates.append((float(best_center[0]), float(best_center[1]), float(best_radius), int(best_count)))
        keep = np.ones(len(work), dtype=bool)
        keep[best_inliers] = False
        work = work[keep]
    return candidates


def sorted_square_score(points_xy, width, height, tol=0.16):
    center = points_xy.mean(axis=0)
    order = np.argsort(np.arctan2(points_xy[:, 1] - center[1], points_xy[:, 0] - center[0]))
    sorted_pts = points_xy[order]
    sides = np.array([np.linalg.norm(sorted_pts[i] - sorted_pts[(i + 1) % 4]) for i in range(4)])
    patterns = [np.array([width, height, width, height]), np.array([height, width, height, width])]
    errors = [float(np.sum(((sides - p) / p) ** 2)) for p in patterns]
    perim_err = abs(float(sides.sum()) - 2.0 * (width + height)) / (2.0 * (width + height))
    radius_err = np.mean(np.abs(np.linalg.norm(points_xy - center.reshape(1, 2), axis=1) -
                                math.sqrt(width * width + height * height) / 2.0))
    ok = any(np.all(np.abs(sides - p) / p < tol) for p in patterns) and perim_err < tol
    return min(errors) + perim_err + radius_err, ok, order, sides


def select_circle_set(candidates, width, height):
    if len(candidates) < 4:
        raise RuntimeError(f"Only {len(candidates)} circle candidates")
    best = None
    for combo in __import__("itertools").combinations(range(len(candidates)), 4):
        pts = np.asarray([[candidates[i][0], candidates[i][1]] for i in combo], dtype=np.float64)
        score, ok, order, sides = sorted_square_score(pts, width, height)
        inliers = sum(candidates[i][3] for i in combo)
        # Prefer geometrically valid sets, then lower geometry error, then stronger circle support.
        rank = (0 if ok else 1, score, -inliers)
        if best is None or rank < best[0]:
            best = (rank, combo, order, sides, score, ok)
    _, combo, order, sides, score, ok = best
    return list(combo), order, sides, score, ok


def select_circle_set_with_qr(candidates, width, height, R_align, avg_z, qr_sorted):
    if len(candidates) < 4:
        raise RuntimeError(f"Only {len(candidates)} circle candidates")

    R_inv = np.linalg.inv(R_align)
    best = None
    for combo in __import__("itertools").combinations(range(len(candidates)), 4):
        pts_xy = np.asarray([[candidates[i][0], candidates[i][1]] for i in combo], dtype=np.float64)
        geom_score, geom_ok, _, sides = sorted_square_score(pts_xy, width, height)
        selected_aligned = np.asarray(
            [[candidates[i][0], candidates[i][1], avg_z] for i in combo], dtype=np.float64
        )
        lidar_centers = (R_inv @ selected_aligned.T).T
        lidar_sorted = sort_pattern_centers(lidar_centers, "lidar")
        _, _, rmse = solve_rigid(lidar_sorted, qr_sorted)
        inliers = sum(candidates[i][3] for i in combo)
        # RMSE is the strongest final check because it validates correspondence
        # against the camera-side target geometry. Geometry remains a tie-breaker.
        rank = (rmse, 0 if geom_ok else 1, geom_score, -inliers)
        if best is None or rank < best[0]:
            best = (rank, combo, sides, geom_score, geom_ok, lidar_sorted, rmse)

    _, combo, sides, geom_score, geom_ok, lidar_sorted, rmse = best
    return list(combo), sides, geom_score, geom_ok, lidar_sorted, rmse


def sort_pattern_centers(points, mode):
    points = np.asarray(points, dtype=np.float64)
    if mode == "lidar":
        work = np.column_stack([-points[:, 1], -points[:, 2], points[:, 0]])
    else:
        work = points.copy()
    center = work.mean(axis=0)
    angles = np.arctan2(work[:, 1] - center[1], work[:, 0] - center[0])
    order = np.argsort(angles)
    sorted_work = work[order].copy()
    v01 = sorted_work[1, :2] - sorted_work[0, :2]
    v12 = sorted_work[2, :2] - sorted_work[1, :2]
    if np.cross(np.r_[v01, 0.0], np.r_[v12, 0.0])[2] > 0:
        sorted_work[[1, 3]] = sorted_work[[3, 1]]
    if mode == "lidar":
        return np.column_stack([sorted_work[:, 2], -sorted_work[:, 0], -sorted_work[:, 1]])
    return sorted_work


def solve_rigid(lidar_pts, cam_pts):
    L = np.asarray(lidar_pts, dtype=np.float64)
    C = np.asarray(cam_pts, dtype=np.float64)
    mu_l = L.mean(axis=0)
    mu_c = C.mean(axis=0)
    sigma = (L - mu_l).T @ (C - mu_c)
    U, _, Vt = np.linalg.svd(sigma)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        D = np.eye(3)
        D[2, 2] = -1
        R = Vt.T @ D @ U.T
    t = mu_c - R @ mu_l
    residuals = (L @ R.T + t.reshape(1, 3)) - C
    rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    return R, t, rmse


def write_single_result(out_dir: Path, cfg: dict, R, t):
    with open(out_dir / "single_calib_result.txt", "w", encoding="utf-8") as f:
        f.write("# FAST-LIVO2 calibration format\n")
        f.write("cam_model: Pinhole\n")
        f.write("cam_width: 1920\n")
        f.write("cam_height: 1080\n")
        f.write("scale: 1.0\n")
        f.write(f"cam_fx: {cfg['fx']:.6g}\n")
        f.write(f"cam_fy: {cfg['fy']:.6g}\n")
        f.write(f"cam_cx: {cfg['cx']:.6g}\n")
        f.write(f"cam_cy: {cfg['cy']:.6g}\n")
        f.write(f"cam_d0: {cfg['k1']:.6g}\n")
        f.write(f"cam_d1: {cfg['k2']:.6g}\n")
        f.write(f"cam_d2: {cfg['p1']:.6g}\n")
        f.write(f"cam_d3: {cfg['p2']:.6g}\n\n")
        f.write("Rcl: [")
        f.write(f"{R[0,0]:10.6f}, {R[0,1]:10.6f}, {R[0,2]:10.6f},\n")
        f.write(f"      {R[1,0]:10.6f}, {R[1,1]:10.6f}, {R[1,2]:10.6f},\n")
        f.write(f"      {R[2,0]:10.6f}, {R[2,1]:10.6f}, {R[2,2]:10.6f}]\n")
        f.write(f"Pcl: [{t[0]:10.6f}, {t[1]:10.6f}, {t[2]:10.6f}]\n")


def fmt_pts(points):
    return "".join(f" {{{p[0]:.6g},{p[1]:.6g},{p[2]:.6g}}}" for p in points)


def calibrate_group(group: int, cfg: dict, args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) / str(group)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_path = str(data_dir / f"{group}.jpg")
    bag_path = str(data_dir / f"{group}.bag")
    qr_centers = detect_qr_centers(image_path, cfg, str(out_dir / "qr_detect.png"))

    roi = tuple(args.roi)
    points, rings = read_roi_points(bag_path, args.topic, roi)
    normal, d = fit_plane(points, seed=group)
    edges = extract_ring_edges(points, rings, normal, d)
    aligned, R_align = align_edges_to_plane(edges, normal)
    xy = aligned[:, :2]
    avg_z = float(aligned[:, 2].mean())
    candidates = detect_circle_candidates(xy, float(cfg["circle_radius"]), seed=100 + group)
    qr_sorted = sort_pattern_centers(qr_centers, "camera")
    combo, sides, score, ok, lidar_sorted, rmse_hint = select_circle_set_with_qr(
        candidates,
        float(cfg["delta_width_circles"]),
        float(cfg["delta_height_circles"]),
        R_align,
        avg_z,
        qr_sorted,
    )
    R, t, rmse = solve_rigid(lidar_sorted, qr_sorted)

    with open(out_dir / "circle_center_record.txt", "w", encoding="utf-8") as f:
        f.write(f"time: ariy_group_{group}\n")
        f.write("lidar_centers:" + fmt_pts(lidar_sorted) + "\n")
        f.write("qr_centers:" + fmt_pts(qr_sorted) + "\n")
    write_single_result(out_dir, cfg, R, t)

    print(
        f"[Group {group}] roi_points={len(points)} edges={len(edges)} candidates={len(candidates)} "
        f"combo={combo} sides={np.round(sides, 4).tolist()} geom_ok={ok} rmse={rmse:.4f}"
    )
    return rmse


def main():
    root = root_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(root / "config" / "qr_params.yaml"))
    parser.add_argument("--data-dir", default=str(root / "calib_data" / "0421" / "ariy"))
    parser.add_argument("--output-dir", default=str(root / "output" / "0421" / "ariy"))
    parser.add_argument("--topic", default="/velodyne_front")
    parser.add_argument("--groups", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument(
        "--roi",
        nargs=6,
        type=float,
        default=[-1.30, 1.00, -0.90, 1.50, 1.45, 2.50],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    rmses = []
    for group in args.groups:
        rmses.append(calibrate_group(group, cfg, args))
    print(f"[Summary] groups={args.groups} mean_rmse={np.mean(rmses):.4f} max_rmse={np.max(rmses):.4f}")


if __name__ == "__main__":
    main()
