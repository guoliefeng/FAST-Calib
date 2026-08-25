# 相机外参标定说明

本文说明本项目的车辆六相机外参标定方案、已知风险和标准操作。详细配置示例见 [vehicle_camera_calibration_workflow.md](vehicle_camera_calibration_workflow.md)。

## 设计方案

### 标定拓扑

优先直接标定四路相机，另外两路由相机间外参（C2C）推导：

| 相机 | 所属 LiDAR | 来源 |
|---|---|---|
| `front`、`front_left` | `lidar_left_front` | 相机-LiDAR 直接标定 |
| `rear`、`rear_right` | `lidar_right_rear` | 相机-LiDAR 直接标定 |
| `rear_left` | `lidar_left_front` | `front_left` + 左侧 C2C |
| `front_right` | `lidar_right_rear` | `rear_right` + 右侧 C2C |

坐标约定：

```text
p_cam  = T_cam_lidar * p_lidar
p_cam1 = T_cam1_cam0 * p_cam0
```

左侧 C2C 固定为 `rear_left -> front_left`，右侧固定为 `rear_right -> front_right`。汇总脚本按矩阵方向自动推导缺失相机。

### 输出与验收

- 直接标定输出：`<data_dir>/_calib_output/final_extrinsic.yaml`，核心矩阵为 `T_cam_lidar`。
- 组内验证：`05_verify/*_board_overlay.png`，红色为 LiDAR 标定板点云投影。
- C2C 输出：`c2c_extrinsic_result*.yaml` 和 `*_validation_projection/used_projection_contact_sheet.jpg`。
- 全车汇总：`scripts/summarize_vehicle_extrinsics.py` 生成六路 YAML/CSV。
- ROS TF 默认使用 `T_lidar_camera = inverse(T_cam_lidar)`；只有驱动坐标约定明确要求时，才导出
  `T_export = Rz(-90 deg) * inverse(T_cam_lidar)`。

## 当前问题

1. **四孔平面对称**：四个圆心可产生 180 度对称解。单看 RMSE 可能更小，但相机位置会落到错误一侧；必须同时检查圆心编号、板点投影和 `base_link` 物理位置。
2. **内参变更会改变外参**：更新内参后，图像 QR/ArUco 中心会变化。不能继续复用旧 `T_cam_lidar`；可以复用已确认的 LiDAR 圆心，重新计算图像侧和联合 SVD。
3. **LiDAR 驱动 yaw 变化**：驱动坐标绕 Z 轴变化时，先更新 ROI；不得在标定算法内硬编码旋转。导出时只能选择一种 TF 约定，不能与原 TF 同时发布。
4. **C2C 是长基线间接结果**：平面标记存在 IPPE 分支歧义，反向 PnP 误差只能作诊断。C2C 至少检查有效组数、前向投影、留一法稳定性和车辆物理位置；数据少的 C2C 应标记为 `candidate`/`hold`。

## 操作说明

### 1. 准备配置

检查 `config/vehicles/<vehicle>/`：

- 六份 `cameras/*.yaml` 的内参和畸变模型；
- `vehicle.yaml` 的数据目录、相机-LiDAR对应关系和 C2C 相机顺序；
- ROI 与当前 LiDAR 坐标系一致；
- `marker_size`（四孔板小码）与 `c2c_calibration.marker_size_m`（C2C 单码）分别按实物填写。

### 2. 直接标定四路相机

```bash
cd /home/glf/dataDisk/calib/FAST-Calib_ws/src/FAST-Calib
source /opt/ros/noetic/setup.bash
source /home/glf/dataDisk/calib/FAST-Calib_ws/devel/setup.bash

python3 scripts/run_vehicle_calib.py --vehicle <vehicle> --stage all
```

先检查每路 `final_extrinsic.yaml` 的状态、RMSE、`selected_groups` 和 `05_verify` 投影图。若圆心顺序有疑问：

```bash
bash scripts/run_center_order_audit.sh <data_dir>
```

### 3. 内参更新、复用 LiDAR 圆心

确认旧 LiDAR 圆心正确后，不必重新跑 bag：

```bash
python3 scripts/recalibrate_with_reused_lidar_centers.py \
  --data-dir <data_dir> \
  --camera-config config/vehicles/<vehicle>/cameras/<camera>.yaml \
  --previous-output-dir <old_output_dir> \
  --output-dir <new_output_dir>
```

新输出目录必须与旧目录不同；完成后重新检查 `05_verify`。

### 4. 标定 C2C 并汇总

```bash
python3 scripts/c2c_calibrate_vehicle_aruco.py --vehicle <vehicle> --group left --save-validation
python3 scripts/c2c_calibrate_vehicle_aruco.py --vehicle <vehicle> --group right --save-validation

python3 scripts/summarize_vehicle_extrinsics.py --vehicle <vehicle> \
  --direct-extrinsics <direct_extrinsics.yaml> \
  --c2c-root <c2c_root> \
  --output-dir <report_dir>
```

发布前至少满足：直接四路板点投影正常、C2C 接触表正常、六路在 `base_link` 下位置符合前/后/左/右安装关系。仅在这些检查都通过后，才发布对应的一套 TF YAML。
