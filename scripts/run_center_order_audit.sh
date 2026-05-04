#!/usr/bin/env bash
set -euo pipefail

# One-command LiDAR center order inspection and audit.
#
# Usage:
#   bash scripts/run_center_order_audit.sh /home/glf/dataDisk/calib/c2l/223/front
#   bash scripts/run_center_order_audit.sh /home/glf/dataDisk/calib/c2l/223/front save_data_2,save_data_3,save_data_6
#   bash scripts/run_center_order_audit.sh /home/glf/dataDisk/calib/c2l/223/front all 0.01
#
# Outputs:
#   <data_dir>/_calib_output/05_center_inspect/<group>/open_in_cloudcompare.sh
#   <data_dir>/_calib_output/05_center_order_audit/center_order_audit.csv

DATA_DIR="${1:-}"
GROUP_SPEC="${2:-all}"
THRESHOLD="${3:-0.01}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -z "$DATA_DIR" ]; then
  echo "Usage:"
  echo "  bash scripts/run_center_order_audit.sh <data_dir> [groups] [threshold_m]"
  echo
  echo "Example:"
  echo "  bash scripts/run_center_order_audit.sh /home/glf/dataDisk/calib/c2l/223/front"
  echo "  bash scripts/run_center_order_audit.sh /home/glf/dataDisk/calib/c2l/223/front save_data_2,save_data_3 0.01"
  echo "  bash scripts/run_center_order_audit.sh /home/glf/dataDisk/calib/c2l/223/front all 0.01"
  exit 1
fi

echo "[run] data_dir : $DATA_DIR"
echo "[run] groups   : $GROUP_SPEC"
echo "[run] threshold: $THRESHOLD m"

python3 "$SCRIPT_DIR/inspect_lidar_centers.py" \
  --data-dir "$DATA_DIR" \
  --groups "$GROUP_SPEC"

python3 "$SCRIPT_DIR/audit_lidar_center_order.py" \
  --data-dir "$DATA_DIR" \
  --groups "$GROUP_SPEC" \
  --threshold "$THRESHOLD"

AUDIT_DIR="$DATA_DIR/_calib_output/05_center_order_audit"

echo
echo "[done] Audit result:"
echo "  center_order_audit.csv: $AUDIT_DIR/center_order_audit.csv"
echo "  suspicious_groups.txt: $AUDIT_DIR/suspicious_groups.txt"
echo "  recommended_manual_permutation.txt: $AUDIT_DIR/recommended_manual_permutation.txt"
echo
INSPECT_DIR="$DATA_DIR/_calib_output/05_center_inspect"
if [ "$GROUP_SPEC" = "all" ] || [ "$GROUP_SPEC" = "*" ]; then
  if [ -d "$INSPECT_DIR" ]; then
    OPEN_SCRIPT="$(find "$INSPECT_DIR" -mindepth 2 -maxdepth 2 -name open_in_cloudcompare.sh | sort -V | sed -n '1p' || true)"
  else
    OPEN_SCRIPT=""
  fi
else
  FIRST_GROUP="${GROUP_SPEC%%,*}"
  OPEN_SCRIPT="$INSPECT_DIR/$FIRST_GROUP/open_in_cloudcompare.sh"
fi

echo "[view] To visually check one group:"
if [ -n "$OPEN_SCRIPT" ]; then
  echo "  bash \"$OPEN_SCRIPT\""
else
  echo "  No open_in_cloudcompare.sh found under $INSPECT_DIR"
fi
echo
echo "[tip] suspicious_groups.txt lists groups whose current [0,1,2,3] order is probably wrong."
