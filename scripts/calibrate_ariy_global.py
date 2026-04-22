#!/usr/bin/python3
import argparse
import os
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


def enumerate_sets(candidates, R_align, avg_z, qr_sorted, width, height, keep=240):
    import itertools

    R_inv = np.linalg.inv(R_align)
    combo_entries = []
    for combo in itertools.combinations(range(len(candidates)), 4):
        pts_xy = np.asarray([[candidates[i][0], candidates[i][1]] for i in combo], dtype=np.float64)
        geom_score, geom_ok, _, sides = sorted_square_score(pts_xy, width, height, tol=0.20)
        if geom_score > 1.2:
            continue
        selected_aligned = np.asarray(
            [[candidates[i][0], candidates[i][1], avg_z] for i in combo], dtype=np.float64
        )
        lidar = (R_inv @ selected_aligned.T).T
        inliers = sum(candidates[i][3] for i in combo)
        combo_entries.append(
            {
                "combo": combo,
                "lidar_raw": lidar,
                "geom_score": geom_score,
                "geom_ok": geom_ok,
                "sides": sides,
                "inliers": inliers,
                "rank": (0 if geom_ok else 1, geom_score, -inliers),
            }
        )
    combo_entries.sort(key=lambda s: s["rank"])

    sets = []
    # Expand the best geometric candidates into all point correspondences. The
    # target is symmetric enough that angle sorting alone can be wrong for side
    # views; global consistency will choose the correct permutation.
    for entry in combo_entries[: max(20, keep // 6)]:
        for perm in itertools.permutations(range(4)):
            lidar_ordered = entry["lidar_raw"][list(perm)]
            _, _, single_rmse = solve_rigid(lidar_ordered, qr_sorted)
            sets.append(
                {
                    "combo": entry["combo"],
                    "perm": perm,
                    "lidar": lidar_ordered,
                    "geom_score": entry["geom_score"],
                    "geom_ok": entry["geom_ok"],
                    "sides": entry["sides"],
                    "single_rmse": single_rmse,
                    "inliers": entry["inliers"],
                    "rank": (entry["geom_score"], single_rmse, -entry["inliers"]),
                }
            )
    sets.sort(key=lambda s: s["rank"])
    return sets[:keep]


def transform_rmse(R, t, lidar, cam):
    residual = (lidar @ R.T + t.reshape(1, 3)) - cam
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def evaluate_transform(R, t, group_data):
    selected = []
    rmses = []
    for data in group_data:
        best = None
        for s in data["sets"]:
            rmse = transform_rmse(R, t, s["lidar"], data["qr"])
            # Keep geometry as a light tie-breaker; global transform residual dominates.
            rank = (rmse, 0 if s["geom_ok"] else 1, s["geom_score"])
            if best is None or rank < best[0]:
                best = (rank, s, rmse)
        selected.append(best[1])
        rmses.append(best[2])
    rmses = np.asarray(rmses)
    return float(np.mean(rmses) + 0.5 * np.max(rmses)), selected, rmses


def refine_selection(group_data, selected, iterations=8):
    R = t = None
    rmses = None
    for _ in range(iterations):
        L = np.vstack([s["lidar"] for s in selected])
        C = np.vstack([d["qr"] for d in group_data])
        R, t, _ = solve_rigid(L, C)
        _, selected, rmses = evaluate_transform(R, t, group_data)
    L = np.vstack([s["lidar"] for s in selected])
    C = np.vstack([d["qr"] for d in group_data])
    R, t, global_rmse = solve_rigid(L, C)
    final_rmses = np.asarray([transform_rmse(R, t, s["lidar"], d["qr"]) for s, d in zip(selected, group_data)])
    return R, t, global_rmse, selected, final_rmses


def build_group_data(group, cfg, args):
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) / str(group)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = str(data_dir / f"{group}.jpg")
    bag_path = str(data_dir / f"{group}.bag")

    qr = sort_pattern_centers(
        detect_qr_centers(image_path, cfg, str(output_dir / "qr_detect.png")),
        "camera",
    )
    points, rings = read_roi_points(bag_path, args.topic, tuple(args.roi))
    normal, d = fit_plane(points, seed=group)
    edges = extract_ring_edges(points, rings, normal, d)
    aligned, R_align = align_edges_to_plane(edges, normal)
    avg_z = float(aligned[:, 2].mean())
    candidates = detect_circle_candidates(aligned[:, :2], float(cfg["circle_radius"]), seed=500 + group)
    sets = enumerate_sets(
        candidates,
        R_align,
        avg_z,
        qr,
        float(cfg["delta_width_circles"]),
        float(cfg["delta_height_circles"]),
        keep=args.keep_sets,
    )
    if not sets:
        raise RuntimeError(f"group {group}: no candidate sets")
    print(
        f"[Group {group}] roi={len(points)} edges={len(edges)} candidates={len(candidates)} "
        f"sets={len(sets)} best_geom={sets[0]['geom_score']:.4f} best_single={sets[0]['single_rmse']:.4f}"
    )
    return {"group": group, "qr": qr, "sets": sets}


def write_outputs(group_data, selected, cfg, output_dir):
    output_dir = Path(output_dir)
    aggregate = output_dir / "multi" / "circle_center_record.txt"
    aggregate.parent.mkdir(parents=True, exist_ok=True)
    with open(aggregate, "w", encoding="utf-8") as agg:
        for data, s in zip(group_data, selected):
            group = data["group"]
            group_dir = output_dir / str(group)
            group_dir.mkdir(parents=True, exist_ok=True)
            R, t, _ = solve_rigid(s["lidar"], data["qr"])
            write_single_result(group_dir, cfg, R, t)
            record = (
                f"time: ariy_group_{group}\n"
                f"lidar_centers:{fmt_pts(s['lidar'])}\n"
                f"qr_centers:{fmt_pts(data['qr'])}\n"
            )
            with open(group_dir / "circle_center_record.txt", "w", encoding="utf-8") as f:
                f.write(record)
            agg.write(record)
    return aggregate


def main():
    root = root_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(root / "config" / "qr_params.yaml"))
    parser.add_argument("--data-dir", default=str(root / "calib_data" / "0421" / "ariy"))
    parser.add_argument("--output-dir", default=str(root / "output" / "0421" / "ariy"))
    parser.add_argument("--topic", default="/velodyne_front")
    parser.add_argument("--groups", nargs="+", type=int, default=list(range(1, 10)))
    parser.add_argument("--keep-sets", type=int, default=240)
    parser.add_argument(
        "--roi",
        nargs=6,
        type=float,
        default=[-1.30, 1.00, -0.90, 1.50, 1.45, 2.50],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    group_data = [build_group_data(g, cfg, args) for g in args.groups]

    best = None
    trial_count = 0
    for data in group_data:
        # Top 80 per group is enough to seed the global search while keeping runtime bounded.
        for s in data["sets"][:80]:
            R, t, _ = solve_rigid(s["lidar"], data["qr"])
            score, selected, rmses = evaluate_transform(R, t, group_data)
            trial_count += 1
            if best is None or score < best[0]:
                best = (score, selected, rmses, R, t)

    R, t, global_rmse, selected, final_rmses = refine_selection(group_data, best[1])
    aggregate = write_outputs(group_data, selected, cfg, args.output_dir)

    print(f"[Global] trials={trial_count} aggregate={aggregate}")
    print(f"[Global] RMSE={global_rmse:.4f} per_group={np.round(final_rmses, 4).tolist()}")
    print("[Global] T_cam_lidar:")
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    print(np.array2string(T, formatter={"float_kind": lambda x: f"{x: .6f}"}))
    for data, s, r in zip(group_data, selected, final_rmses):
        print(
            f"  group {data['group']}: rmse={r:.4f} combo={s['combo']} perm={s.get('perm')} "
            f"geom_ok={s['geom_ok']} geom={s['geom_score']:.4f} sides={np.round(s['sides'],4).tolist()}"
        )


if __name__ == "__main__":
    main()
