#!/usr/bin/python3
import argparse
import itertools
import re
from pathlib import Path

import numpy as np

from calibrate_ariy_batch import (
    detect_circle_candidates,
    detect_qr_centers,
    extract_ring_edges,
    fit_plane,
    fmt_pts,
    load_config,
    read_roi_points,
    root_dir,
    solve_rigid,
    sort_pattern_centers,
    sorted_square_score,
    align_edges_to_plane,
    write_single_result,
)


def parse_lidar_centers(record_path: Path) -> np.ndarray:
    text = record_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("lidar_centers:"):
            pts = []
            for item in re.findall(r"\{([^}]*)\}", line):
                pts.append([float(v) for v in item.split(",")])
            if len(pts) == 4:
                return np.asarray(pts, dtype=np.float64)
    raise RuntimeError(f"Cannot parse four lidar centers from {record_path}")


def predicted_roi(pred_lidar: np.ndarray, margin: np.ndarray):
    mins = pred_lidar.min(axis=0) - margin
    maxs = pred_lidar.max(axis=0) + margin
    return (mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2])


def select_group(group, cfg, args, qr_sorted, R, t):
    data_dir = Path(args.data_dir)
    pred = (qr_sorted - t.reshape(1, 3)) @ R
    roi = predicted_roi(pred, np.asarray(args.roi_margin, dtype=np.float64))

    points, rings = read_roi_points(str(data_dir / f"{group}.bag"), args.topic, roi)
    normal, d = fit_plane(points, seed=1000 + group)
    edges = extract_ring_edges(points, rings, normal, d)
    aligned, R_align = align_edges_to_plane(edges, normal)
    avg_z = float(aligned[:, 2].mean())
    candidates = detect_circle_candidates(
        aligned[:, :2],
        float(cfg["circle_radius"]),
        seed=200 + group,
        max_candidates=args.max_candidates,
    )
    if len(candidates) < 4:
        raise RuntimeError(f"group {group}: only {len(candidates)} circle candidates in roi={roi}")

    lidar_candidates = (
        np.linalg.inv(R_align)
        @ np.asarray([[c[0], c[1], avg_z] for c in candidates], dtype=np.float64).T
    ).T

    nearest = [
        np.argsort(np.linalg.norm(lidar_candidates - p.reshape(1, 3), axis=1))[
            : min(args.nearest_candidates, len(lidar_candidates))
        ]
        for p in pred
    ]

    best = None
    for combo in itertools.product(*nearest):
        if len(set(combo)) < 4:
            continue
        lidar = lidar_candidates[list(combo)]
        pred_rmse = float(np.sqrt(np.mean(np.sum((lidar - pred) ** 2, axis=1))))
        geom_score, geom_ok, _, sides = sorted_square_score(
            lidar[:, :2],
            float(cfg["delta_width_circles"]),
            float(cfg["delta_height_circles"]),
            tol=args.geometry_tolerance,
        )
        support = sum(candidates[i][3] for i in combo)
        rank = (0 if geom_ok else 1, pred_rmse, geom_score, -support)
        if best is None or rank < best["rank"]:
            best = {
                "rank": rank,
                "combo": combo,
                "lidar": lidar,
                "pred_rmse": pred_rmse,
                "geom_ok": geom_ok,
                "geom_score": geom_score,
                "sides": sides,
                "roi": roi,
                "roi_points": len(points),
                "edge_points": len(edges),
                "candidate_count": len(candidates),
            }

    if best is None:
        raise RuntimeError(f"group {group}: no unique assignment found")
    return best


def write_records(group_data, selected, cfg, output_dir):
    output_dir = Path(output_dir)
    aggregate_path = output_dir / "multi" / "circle_center_record.txt"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)

    with aggregate_path.open("w", encoding="utf-8") as agg:
        for data, item in zip(group_data, selected):
            group = data["group"]
            group_dir = output_dir / str(group)
            group_dir.mkdir(parents=True, exist_ok=True)

            if "preserve_record_path" in item:
                agg.write(Path(item["preserve_record_path"]).read_text(encoding="utf-8"))
                continue

            R_single, t_single, _ = solve_rigid(item["lidar"], data["qr"])
            write_single_result(group_dir, cfg, R_single, t_single)

            record = (
                f"time: ariy_group_{group}\n"
                f"lidar_centers:{fmt_pts(item['lidar'])}\n"
                f"qr_centers:{fmt_pts(data['qr'])}\n"
            )
            (group_dir / "circle_center_record.txt").write_text(record, encoding="utf-8")
            (group_dir / "roi_used.txt").write_text(
                "roi: " + " ".join(f"{v:.6f}" for v in item["roi"]) + "\n"
                f"roi_points: {item['roi_points']}\n"
                f"edge_points: {item['edge_points']}\n"
                f"circle_candidates: {item['candidate_count']}\n"
                f"prediction_rmse: {item['pred_rmse']:.6f}\n"
                f"geometry_ok: {item['geom_ok']}\n"
                "sides: " + " ".join(f"{v:.6f}" for v in item["sides"]) + "\n",
                encoding="utf-8",
            )
            agg.write(record)
    return aggregate_path


def main():
    root = root_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(root / "config" / "qr_params.yaml"))
    parser.add_argument("--data-dir", default=str(root / "calib_data" / "0421" / "ariy"))
    parser.add_argument("--output-dir", default=str(root / "output" / "0421" / "ariy"))
    parser.add_argument(
        "--anchor-record",
        default=str(root / "output" / "0421" / "ariy" / "1" / "circle_center_record.txt"),
    )
    parser.add_argument("--anchor-group", type=int, default=1)
    parser.add_argument("--preserve-anchor", action="store_true")
    parser.add_argument("--topic", default="/velodyne_front")
    parser.add_argument("--groups", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--nearest-candidates", type=int, default=14)
    parser.add_argument("--max-candidates", type=int, default=60)
    parser.add_argument("--geometry-tolerance", type=float, default=0.30)
    parser.add_argument("--roi-margin", nargs=3, type=float, default=[0.55, 0.55, 0.50])
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    qr_by_group = {}
    for group in args.groups:
        group_dir = output_dir / str(group)
        group_dir.mkdir(parents=True, exist_ok=True)
        qr_by_group[group] = sort_pattern_centers(
            detect_qr_centers(
                str(data_dir / f"{group}.jpg"),
                cfg,
                str(group_dir / "qr_detect.png"),
            ),
            "camera",
        )

    anchor_lidar = parse_lidar_centers(Path(args.anchor_record))
    if args.anchor_group not in qr_by_group:
        raise RuntimeError(f"anchor group {args.anchor_group} must be included in --groups")
    R, t, anchor_rmse = solve_rigid(anchor_lidar, qr_by_group[args.anchor_group])
    print(f"[Init] anchor={args.anchor_record} rmse={anchor_rmse:.4f}")

    best = None
    for iteration in range(args.iterations):
        selected = []
        print(f"[Iter {iteration}]")
        for group in args.groups:
            if args.preserve_anchor and group == args.anchor_group:
                item = {
                    "lidar": anchor_lidar,
                    "preserve_record_path": args.anchor_record,
                    "pred_rmse": anchor_rmse,
                    "geom_ok": True,
                    "geom_score": 0.0,
                    "sides": np.zeros(4),
                    "roi": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                    "roi_points": 0,
                    "edge_points": 0,
                    "candidate_count": 0,
                }
            else:
                item = select_group(group, cfg, args, qr_by_group[group], R, t)
            selected.append(item)
            if "preserve_record_path" in item:
                print(f"  group {group}: preserved anchor record={item['preserve_record_path']}")
            else:
                roi_txt = tuple(round(v, 2) for v in item["roi"])
                sides_txt = np.round(item["sides"], 3).tolist()
                print(
                    f"  group {group}: pred_rmse={item['pred_rmse']:.4f} "
                    f"geom={item['geom_ok']}/{item['geom_score']:.3f} "
                    f"sides={sides_txt} roi={roi_txt} "
                    f"pts={item['roi_points']} edges={item['edge_points']} cand={item['candidate_count']}"
                )

        lidar_all = np.vstack([item["lidar"] for item in selected])
        qr_all = np.vstack([qr_by_group[group] for group in args.groups])
        R, t, global_rmse = solve_rigid(lidar_all, qr_all)
        per_group = []
        for group, item in zip(args.groups, selected):
            residual = (item["lidar"] @ R.T + t.reshape(1, 3)) - qr_by_group[group]
            per_group.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))
        print(
            f"[Iter {iteration}] global_rmse={global_rmse:.4f} "
            f"per_group={np.round(per_group, 4).tolist()}"
        )
        if best is None or global_rmse < best["global_rmse"]:
            best = {
                "global_rmse": global_rmse,
                "per_group": per_group,
                "selected": selected,
                "R": R.copy(),
                "t": t.copy(),
                "iteration": iteration,
            }

    group_data = [{"group": group, "qr": qr_by_group[group]} for group in args.groups]
    aggregate_path = write_records(group_data, best["selected"], cfg, output_dir)

    summary_path = output_dir / "multi" / "auto_roi_summary.txt"
    T = np.eye(4)
    T[:3, :3] = best["R"]
    T[:3, 3] = best["t"]
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"best_iteration: {best['iteration']}\n")
        f.write(f"global_rmse: {best['global_rmse']:.6f}\n")
        f.write("per_group: " + " ".join(f"{v:.6f}" for v in best["per_group"]) + "\n")
        f.write("T_cam_lidar:\n")
        for row in T:
            f.write(" ".join(f"{v:.9f}" for v in row) + "\n")

    print(f"[Best] iteration={best['iteration']} global_rmse={best['global_rmse']:.4f}")
    print(f"[Best] per_group={np.round(best['per_group'], 4).tolist()}")
    print(f"[Output] aggregate={aggregate_path}")
    print(f"[Output] summary={summary_path}")


if __name__ == "__main__":
    main()
