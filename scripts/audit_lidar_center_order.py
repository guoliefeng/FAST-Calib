#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit LiDAR circle-center ordering for FAST-Calib.

This script reads:
  <data_dir>/_calib_output/03_calibrate/<group>/circle_center_record.txt

For each group it tries all 24 permutations of the 4 LiDAR centers and compares
them against the QR centers using SVD rigid alignment. It reports whether the
current order [0,1,2,3] is suspicious.

Outputs:
  <data_dir>/_calib_output/05_center_order_audit/center_order_audit.csv
  <data_dir>/_calib_output/05_center_order_audit/suspicious_groups.txt
  <data_dir>/_calib_output/05_center_order_audit/recommended_manual_permutation.txt
"""

import argparse
import csv
import itertools
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

CENTER_RE = re.compile(r"\{([^}]*)\}")
CURRENT_PERM = (0, 1, 2, 3)


def natural_key(v: str) -> List[object]:
    parts: List[object] = [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", str(v))]
    parts.append(str(v))
    return parts


def format_perm(perm: Sequence[int]) -> str:
    return "[" + ",".join(str(x) for x in perm) + "]"


def parse_perm_text(text: str) -> List[int]:
    return [int(x) for x in re.findall(r"\d+", text)]


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
        raise RuntimeError(
            f"Expected 4 lidar centers and 4 qr centers, got "
            f"lidar={len(lidar)}, qr={len(qr)} in {path}"
        )

    return np.asarray(lidar, dtype=float), np.asarray(qr, dtype=float)


def svd_solve(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray]:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    x = src - src_mean
    y = dst - dst_mean

    u, _, vt = np.linalg.svd(x.T @ y)
    r = vt.T @ u.T

    if np.linalg.det(r) < 0:
        d = np.eye(3)
        d[2, 2] = -1
        r = vt.T @ d @ u.T

    t = dst_mean - r @ src_mean
    pred = (r @ src.T).T + t
    err = np.linalg.norm(pred - dst, axis=1)
    rmse = float(np.sqrt(np.mean(err * err)))
    return rmse, err


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def geometry_report(points: np.ndarray) -> Dict[str, float]:
    return {
        "d01": dist(points[0], points[1]),
        "d12": dist(points[1], points[2]),
        "d23": dist(points[2], points[3]),
        "d30": dist(points[3], points[0]),
        "d02": dist(points[0], points[2]),
        "d13": dist(points[1], points[3]),
    }


def audit_record(record_path: Path) -> Dict[str, object]:
    lidar, qr = read_circle_record(record_path)

    current_rmse, current_err = svd_solve(lidar, qr)

    rows = []
    for perm in itertools.permutations(range(4)):
        lidar_perm = lidar[list(perm), :]
        rmse, err = svd_solve(lidar_perm, qr)
        rows.append((rmse, perm, err))

    rows.sort(key=lambda x: (x[0], x[1]))
    best_rmse, best_perm, best_err = rows[0]
    second_rmse, second_perm, _ = rows[1]

    geom = geometry_report(lidar)

    out: Dict[str, object] = {
        "group": record_path.parent.name,
        "record": str(record_path),
        "current_perm": format_perm(CURRENT_PERM),
        "best_perm": format_perm(best_perm),
        "second_perm": format_perm(second_perm),
        "current_rmse": current_rmse,
        "best_rmse": best_rmse,
        "second_rmse": second_rmse,
        "improvement": current_rmse - best_rmse,
        "ambiguous_gap": second_rmse - best_rmse,
        "current_point_err": ";".join(f"{x:.9f}" for x in current_err),
        "best_point_err": ";".join(f"{x:.9f}" for x in best_err),
    }
    out.update(geom)
    return out


def make_manual_hint(row: Dict[str, object]) -> str:
    group = str(row["group"])
    best_perm = parse_perm_text(str(row["best_perm"]))
    lines = []
    lines.append(f"{group}: best_perm = {format_perm(best_perm)}")
    lines.append("  In manual_centers.yaml, reorder lidar_centers as:")
    for new_idx, old_idx in enumerate(best_perm):
        lines.append(f"    new center_{new_idx} = old center_{old_idx}")
    return "\n".join(lines)


def resolve_records(output_dir: Path, groups_arg: str) -> List[Path]:
    primary_roots = [output_dir / "03_calibrate", output_dir / "03_single"]

    if groups_arg.lower() not in ("all", "*"):
        group_names = [x.strip() for x in groups_arg.split(",") if x.strip()]
        if not group_names:
            raise RuntimeError("--groups is empty; use all or a comma-separated group list.")

        records: List[Path] = []
        missing: List[str] = []
        for group in group_names:
            record = next(
                (root / group / "circle_center_record.txt" for root in primary_roots
                 if (root / group / "circle_center_record.txt").exists()),
                None,
            )
            if record is None:
                matches = sorted(output_dir.glob(f"**/{group}/circle_center_record.txt"), key=lambda p: len(str(p)))
                record = matches[0] if matches else None

            if record is not None:
                records.append(record)
                continue

            expected = " or ".join(str(root / group / "circle_center_record.txt") for root in primary_roots)
            missing.append(f"{group}: {expected}")

        if missing:
            raise FileNotFoundError(
                "Missing circle_center_record.txt for requested group(s):\n  "
                + "\n  ".join(missing)
            )
        return records

    for root in primary_roots:
        if root.exists():
            records = sorted(root.glob("*/circle_center_record.txt"), key=lambda p: natural_key(p.parent.name))
            if records:
                return records

    records = sorted(output_dir.glob("**/circle_center_record.txt"), key=lambda p: natural_key(p.parent.name))
    if not records:
        raise RuntimeError(
            "No circle_center_record.txt found under "
            f"{primary_roots[0]}, {primary_roots[1]}, or {output_dir}"
        )
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit LiDAR center ordering by trying all 24 permutations.")
    ap.add_argument("--data-dir", required=True, help="Example: /home/glf/dataDisk/calib/c2l/223/front")
    ap.add_argument("--output-dir", help="Default: <data-dir>/_calib_output")
    ap.add_argument("--groups", default="all", help="Comma-separated groups or all. Default: all")
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="If best order improves current RMSE by more than this value in meters, mark suspicious. Default: 0.01",
    )
    ap.add_argument(
        "--ambiguous-threshold",
        type=float,
        default=0.003,
        help="If second-best and best are closer than this value, mark as ambiguous. Default: 0.003",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else data_dir / "_calib_output"
    records = resolve_records(output_dir, args.groups)

    audit_dir = output_dir / "05_center_order_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    out_csv = audit_dir / "center_order_audit.csv"
    suspicious_txt = audit_dir / "suspicious_groups.txt"
    hint_txt = audit_dir / "recommended_manual_permutation.txt"

    fieldnames = [
        "group", "record",
        "current_perm", "best_perm", "second_perm",
        "current_rmse", "best_rmse", "second_rmse",
        "improvement", "ambiguous_gap",
        "suspicious", "ambiguous",
        "current_point_err", "best_point_err",
        "d01", "d12", "d23", "d30", "d02", "d13",
    ]

    rows: List[Dict[str, object]] = []
    suspicious_rows: List[Dict[str, object]] = []

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            try:
                row = audit_record(record)
                suspicious = (
                    str(row["best_perm"]) != format_perm(CURRENT_PERM)
                    and float(row["improvement"]) > args.threshold
                )
                ambiguous = float(row["ambiguous_gap"]) < args.ambiguous_threshold

                row["suspicious"] = "YES" if suspicious else "NO"
                row["ambiguous"] = "YES" if ambiguous else "NO"

                for k in [
                    "current_rmse", "best_rmse", "second_rmse",
                    "improvement", "ambiguous_gap",
                    "d01", "d12", "d23", "d30", "d02", "d13",
                ]:
                    row[k] = f"{float(row[k]):.9f}"

                rows.append(row)
                if suspicious:
                    suspicious_rows.append(row)

                writer.writerow(row)

                flag = "SUSPICIOUS" if suspicious else "ok"
                amb = " ambiguous" if ambiguous else ""
                print(
                    f"[{flag}{amb}] {row['group']}: "
                    f"current={row['current_rmse']} best={row['best_rmse']} "
                    f"best_perm={row['best_perm']} improvement={row['improvement']}"
                )

            except Exception as e:
                raise RuntimeError(f"Failed to audit {record}: {e}") from e

    suspicious_txt.write_text(
        "\n".join(str(r["group"]) for r in suspicious_rows) + ("\n" if suspicious_rows else ""),
        encoding="utf-8",
    )

    hint_txt.write_text(
        "\n\n".join(make_manual_hint(r) for r in suspicious_rows) + ("\n" if suspicious_rows else ""),
        encoding="utf-8",
    )

    print(f"\n[audit] csv        : {out_csv}")
    print(f"[audit] suspicious : {suspicious_txt}")
    print(f"[audit] hints      : {hint_txt}")
    print(f"[audit] suspicious groups: {len(suspicious_rows)} / {len(rows)}")


if __name__ == "__main__":
    main()
