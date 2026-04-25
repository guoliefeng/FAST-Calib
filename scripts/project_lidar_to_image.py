#!/usr/bin/python3
import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2


def fast_calib_root() -> Path:
    return Path(__file__).resolve().parents[1]


def expand_ros_find(path: str) -> str:
    root = str(fast_calib_root())
    return path.replace("$(find fast_calib)", root)


def load_config(path: str) -> dict:
    values = {}
    scalar_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*):\s*([^#]+)")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = scalar_re.match(line)
            if not m:
                continue
            key, raw = m.group(1), m.group(2).strip()
            raw = raw.strip("\"'")
            try:
                values[key] = float(raw)
            except ValueError:
                values[key] = expand_ros_find(raw)
    return values


def load_extrinsic(path: str):
    text = Path(path).read_text(encoding="utf-8")
    r_match = re.search(r"Rcl:\s*\[([^\]]+)\]", text, re.S)
    p_match = re.search(r"Pcl:\s*\[([^\]]+)\]", text, re.S)
    if not r_match or not p_match:
        raise ValueError(f"Cannot parse Rcl/Pcl from {path}")

    nums_r = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", r_match.group(1))]
    nums_p = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", p_match.group(1))]
    if len(nums_r) != 9 or len(nums_p) != 3:
        raise ValueError(f"Invalid Rcl/Pcl dimensions in {path}")
    return np.array(nums_r, dtype=np.float64).reshape(3, 3), np.array(nums_p, dtype=np.float64)


def read_bag_points(bag_path: str, topic: str):
    pts = []
    with rosbag.Bag(bag_path, "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[topic]):
            for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                pts.append((p[0], p[1], p[2]))
    if not pts:
        raise RuntimeError(f"No points read from {bag_path} topic {topic}")
    return np.asarray(pts, dtype=np.float64)


def project_points(points_lidar, Rcl, Pcl, K, dist, image_shape, min_depth, max_depth):
    points_cam = points_lidar @ Rcl.T + Pcl.reshape(1, 3)
    z = points_cam[:, 2]
    valid = (z > min_depth) & (z < max_depth)
    points_cam = points_cam[valid]
    z = z[valid]

    x = points_cam[:, 0] / z
    y = points_cam[:, 1] / z
    k1, k2, p1, p2, k3 = dist
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_dist = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_dist = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    u = K[0, 0] * x_dist + K[0, 2]
    v = K[1, 1] * y_dist + K[1, 2]

    finite = np.isfinite(u) & np.isfinite(v)
    # Very oblique points can become finite but far outside int32 pixel range
    # after radial distortion. Drop them before integer conversion.
    finite &= (np.abs(u) < 1.0e7) & (np.abs(v) < 1.0e7)
    u = u[finite]
    v = v[finite]
    z = z[finite]

    h, w = image_shape[:2]
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    return ui[inside], vi[inside], z[inside]


def undistort_image_for_projection(image, K, dist, alpha, camera_matrix_mode):
    h, w = image.shape[:2]
    dist_cv = np.asarray(dist, dtype=np.float64).reshape(1, -1)
    if camera_matrix_mode == "original":
        new_K = K.copy()
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_cv, (w, h), alpha, (w, h))
    undistorted = cv2.undistort(image, K, dist_cv, None, new_K)
    zero_dist = np.zeros_like(dist, dtype=np.float64)
    return undistorted, new_K, zero_dist


def depth_colors(depth):
    if depth.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    # Near points become red/yellow, far points blue/purple.
    values = (255.0 * (1.0 - norm)).astype(np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(values, cv2.COLORMAP_TURBO).reshape(-1, 3)


def draw_projection(image, u, v, depth, alpha=0.82, radius=1, max_points=350000):
    if u.size == 0:
        return image.copy()

    if max_points > 0 and u.size > max_points:
        rng = np.random.default_rng(7)
        keep = rng.choice(u.size, size=max_points, replace=False)
        u, v, depth = u[keep], v[keep], depth[keep]

    colors = depth_colors(depth)
    overlay = image.copy()
    mask = np.zeros(image.shape[:2], dtype=bool)
    h, w = image.shape[:2]

    offsets = [(0, 0)]
    if radius > 0:
        offsets = [
            (dx, dy)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
            if dx * dx + dy * dy <= radius * radius
        ]

    for dx, dy in offsets:
        uu = u + dx
        vv = v + dy
        ok = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
        overlay[vv[ok], uu[ok]] = colors[ok]
        mask[vv[ok], uu[ok]] = True

    out = image.copy()
    out[mask] = (image[mask].astype(np.float32) * (1.0 - alpha) +
                 overlay[mask].astype(np.float32) * alpha).astype(np.uint8)
    return out


def make_grid(image_paths, output_path):
    images = []
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        images.append(img)
    if not images:
        return

    thumb_w = 960
    thumbs = []
    for img in images:
        scale = thumb_w / img.shape[1]
        thumbs.append(cv2.resize(img, (thumb_w, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA))

    h = max(t.shape[0] for t in thumbs)
    padded = []
    for t in thumbs:
        pad = np.zeros((h, thumb_w, 3), dtype=np.uint8)
        pad[:t.shape[0], :t.shape[1]] = t
        padded.append(pad)

    while len(padded) < 4:
        padded.append(np.zeros_like(padded[0]))
    grid = np.vstack([np.hstack(padded[:2]), np.hstack(padded[2:4])])
    cv2.imwrite(str(output_path), grid)


def target_crop_bbox(image):
    if not hasattr(cv2, "aruco"):
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(dictionary)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None

    pts = np.concatenate([c.reshape(-1, 2) for c in corners], axis=0)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    w = x1 - x0
    h = y1 - y0
    margin = 0.18 * max(w, h)
    ih, iw = image.shape[:2]
    x0 = max(0, int(x0 - margin))
    y0 = max(0, int(y0 - margin))
    x1 = min(iw, int(x1 + margin))
    y1 = min(ih, int(y1 + margin))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def main():
    root = fast_calib_root()
    parser = argparse.ArgumentParser(description="Project LiDAR point clouds onto camera images.")
    parser.add_argument("--config", default=str(root / "config" / "hesai.yaml"))
    parser.add_argument("--extrinsic", default=str(root / "output" / "hesai" / "multi" / "multi_calib_result.txt"))
    parser.add_argument("--data-dir", default=str(root / "calib_data" / "hesai"))
    parser.add_argument("--output-dir", default=str(root / "output" / "hesai" / "projections"))
    parser.add_argument("--groups", nargs="+", default=["1", "2", "3", "4"])
    parser.add_argument("--topic", default=None)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=80.0)
    parser.add_argument("--radius", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.82)
    parser.add_argument("--max-points", type=int, default=350000)
    parser.add_argument(
        "--undistort",
        action="store_true",
        help="Undistort each image first, then project points with the new camera matrix and zero distortion.",
    )
    parser.add_argument(
        "--undistort-alpha",
        type=float,
        default=0.0,
        help="Alpha passed to cv2.getOptimalNewCameraMatrix when --undistort is used. 0 crops invalid pixels, 1 keeps full FOV.",
    )
    parser.add_argument(
        "--undistort-camera-matrix",
        choices=["optimal", "original"],
        default="optimal",
        help="Camera matrix used for the undistorted image. 'original' keeps fx/fy/cx/cy from the config.",
    )
    parser.add_argument("--no-crops", action="store_true")
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    topic = args.topic or cfg.get("lidar_topic", "/velodyne_first")
    K = np.array([
        [cfg["fx"], 0.0, cfg["cx"]],
        [0.0, cfg["fy"], cfg["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist = np.array([cfg["k1"], cfg["k2"], cfg["p1"], cfg["p2"], 0.0], dtype=np.float64)
    Rcl, Pcl = load_extrinsic(args.extrinsic)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    written = []
    written_crops = []

    for group in args.groups:
        bag_path = data_dir / f"{group}.bag"
        image_path = data_dir / f"{group}.jpg"
        if not bag_path.exists() or not image_path.exists():
            raise FileNotFoundError(f"Missing data for group {group}: {bag_path}, {image_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        project_image = image
        project_K = K
        project_dist = dist
        if args.undistort:
            project_image, project_K, project_dist = undistort_image_for_projection(
                image, K, dist, args.undistort_alpha, args.undistort_camera_matrix
            )

        print(f"[Group {group}] reading {bag_path}")
        points = read_bag_points(str(bag_path), topic)
        u, v, depth = project_points(
            points, Rcl, Pcl, project_K, project_dist, project_image.shape, args.min_depth, args.max_depth
        )
        projected = draw_projection(project_image, u, v, depth, alpha=args.alpha, radius=args.radius, max_points=args.max_points)

        suffix = "_undistorted" if args.undistort else ""
        out_path = out_dir / f"hesai_{group}_projected_multi{suffix}.png"
        cv2.imwrite(str(out_path), projected)
        written.append(out_path)
        if not args.no_crops:
            bbox = target_crop_bbox(project_image)
            if bbox:
                x0, y0, x1, y1 = bbox
                crop_path = out_dir / f"hesai_{group}_projected_multi{suffix}_crop.png"
                cv2.imwrite(str(crop_path), projected[y0:y1, x0:x1])
                written_crops.append(crop_path)
        print(f"[Group {group}] points={points.shape[0]} visible={u.size} saved={out_path}")

    if not args.no_grid:
        grid_path = out_dir / "hesai_1_2_3_4_projected_grid.png"
        make_grid(written, grid_path)
        print(f"[Grid] saved={grid_path}")
        if written_crops:
            crop_grid_path = out_dir / "hesai_1_2_3_4_projected_crop_grid.png"
            make_grid(written_crops, crop_grid_path)
            print(f"[Crop grid] saved={crop_grid_path}")


if __name__ == "__main__":
    main()
