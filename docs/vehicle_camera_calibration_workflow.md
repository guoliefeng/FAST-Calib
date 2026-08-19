# 六相机车辆标定流程

本文记录 5960、5971 使用的完整流程，适用于下一辆同类车辆。完成新车配置和数据整理后，可以用一个命令依次完成：

1. 四路相机—雷达直接标定；
2. 左、右两组相机—相机（C2C）标定；
3. 生成六路相机外参汇总；
4. 导出与 `/home/guoli/data/227` 相同格式的六个 ROS TF YAML。

本文中的路径使用 Linux 正斜杠 `/`。

## 1. 六相机标定拓扑

正常情况下只直接标定四路相机，另外两路通过 C2C 推导：

| 相机 | 对应雷达 | 获得方式 |
|---|---|---|
| `front` | `lidar_left_front` | 相机—雷达直接标定 |
| `front_left` | `lidar_left_front` | 相机—雷达直接标定 |
| `rear_left` | `lidar_left_front` | 由 `front_left` 和左侧 C2C 推导 |
| `rear` | `lidar_right_rear` | 相机—雷达直接标定 |
| `rear_right` | `lidar_right_rear` | 相机—雷达直接标定 |
| `front_right` | `lidar_right_rear` | 由 `rear_right` 和右侧 C2C 推导 |

C2C 的相机顺序固定为：

```text
left:  cam0=rear_left  -> cam1=front_left
right: cam0=rear_right -> cam1=front_right
```

涉及三种外参方向：

```text
直接标定:
p_camera = T_cam_lidar * p_lidar

C2C:
p_cam1 = T_cam1_cam0 * p_cam0

227 格式 ROS TF:
p_lidar = T_lidar_camera * p_camera
T_lidar_camera = inverse(T_cam_lidar)
```

不要把直接标定结果中的 `Pcl` 当作 227 格式 YAML 的 translation。导出程序会自动完成逆变换。

## 2. 环境要求

当前流程用于 Ubuntu 20.04、ROS Noetic。先进入工程并加载 ROS 环境：

```bash
cd /home/guoli/proj/fast_calib_ws/src/FAST-Calib
source /opt/ros/noetic/setup.bash
source /home/guoli/proj/fast_calib_ws/devel/setup.bash
```

Python 环境需要能够导入：

```text
yaml
numpy
scipy
cv2
cv2.aruco
```

如果 `cv2` 中没有 `aruco`，需要使用带 ArUco 模块的 OpenCV。

## 3. 新车辆配置

每辆车需要一个独立配置目录：

```text
config/vehicles/<车辆号>/
├── vehicle.yaml
└── cameras/
    ├── front.yaml
    ├── front_left.yaml
    ├── front_right.yaml
    ├── rear.yaml
    ├── rear_left.yaml
    └── rear_right.yaml
```

可以复制 5971 作为模板，但目标目录必须不存在，并且复制后必须逐项替换车辆号、路径、内参和 ROI：

```bash
VEHICLE_ID=下一辆车编号
cp -a config/vehicles/5971 "config/vehicles/${VEHICLE_ID}"
```

不要只修改目录名。模板中可能保留旧车辆的注释或相同内参，必须按相机位置或相机序列号核对六份文件。

### 3.1 六份相机内参

每个 `cameras/<camera>.yaml` 至少要正确填写：

```yaml
fx: 0.0
fy: 0.0
cx: 0.0
cy: 0.0

k1: 0.0
k2: 0.0
p1: 0.0
p2: 0.0
k3: 0.0
k4: 0.0
k5: 0.0
k6: 0.0
```

如果内参模型只有 5 个畸变参数，可以不写 `k4`、`k5`、`k6`；只要其中一个存在，流程会按 8 参数 rational distortion 使用。

四孔大标定板的参数也必须与实物一致：

```yaml
marker_size: 0.250
delta_width_qr_center: 0.725
delta_height_qr_center: 0.500
delta_width_circles: 0.650
delta_height_circles: 0.650
circle_radius: 0.225
min_detected_markers: 3
```

两个 marker 参数含义不同：

| 配置位置 | 含义 |
|---|---|
| `cameras/*.yaml: marker_size` | 四孔大板上单个小 ArUco 的边长 |
| `vehicle.yaml: c2c_calibration.marker_size_m` | C2C 单 ArUco 黑色外框的实测边长 |

5960、5971 的 C2C 使用 `marker_size_m: 1.05`。不能因为两个字段都叫 marker size 就写成相同数值。

### 3.2 四路直接标定配置

`vehicle.yaml` 中启用四路直接标定：

```yaml
vehicle_id: '下一辆车编号'

sensors:
  front:
    enabled: true
    data_dir: /home/guoli/data/lianyun/下一辆车编号
    output_dir: ${data_dir}/_calib_output/front
    camera_config: config/vehicles/下一辆车编号/cameras/front.yaml
    roi_profile: config/roi_profiles/hesai_new/front.yaml
    lidar_topic: auto
    layout:
      group_prefix: mode_1_

  front_left:
    enabled: true
    data_dir: /home/guoli/data/lianyun/下一辆车编号
    output_dir: ${data_dir}/_calib_output/front_left
    camera_config: config/vehicles/下一辆车编号/cameras/front_left.yaml
    roi_profile: config/roi_profiles/hesai_new/front_left.yaml
    lidar_topic: auto
    layout:
      group_prefix: mode_2_

  rear:
    enabled: true
    data_dir: /home/guoli/data/lianyun/下一辆车编号
    output_dir: ${data_dir}/_calib_output/rear
    camera_config: config/vehicles/下一辆车编号/cameras/rear.yaml
    roi_profile: config/roi_profiles/hesai_new/rear.yaml
    lidar_topic: auto
    layout:
      group_prefix: mode_3_

  rear_right:
    enabled: true
    data_dir: /home/guoli/data/lianyun/下一辆车编号
    output_dir: ${data_dir}/_calib_output/rear_right
    camera_config: config/vehicles/下一辆车编号/cameras/rear_right.yaml
    roi_profile: config/roi_profiles/hesai_new/rear_right.yaml
    lidar_topic: auto
    layout:
      group_prefix: mode_4_
```

四路数据放在同一个根目录时，必须同时满足：

- `group_prefix` 不同；
- `output_dir` 不同。

否则四路会读取错误的数据或互相覆盖结果。

当前约定为：

| 数据前缀 | 相机 |
|---|---|
| `mode_1_` | `front` |
| `mode_2_` | `front_left` |
| `mode_3_` | `rear` |
| `mode_4_` | `rear_right` |

`lidar_topic: auto` 只适用于每个 bag 中只有一个 PointCloud2/CustomMsg 点云话题。如果 bag 中有多个点云话题，必须写明实际话题。

最终 227 格式要求父 frame 为 `lidar_left_front` 或 `lidar_right_rear`。如果 bag 话题名与 TF frame 名不同，不能直接把话题名当 frame，需要先确认并调整导出映射。

### 3.3 ROI 和点云参数

新车辆必须检查每一路的：

```yaml
roi:
  coarse:
    x_min: 0.0
    x_max: 0.0
    y_min: 0.0
    y_max: 0.0
    z_min: 0.0
    z_max: 0.0
  detector:
    board_span:
      x: [0.0, 0.0]
      y: [0.0, 0.0]
      z: [0.0, 0.0]

fast_calib:
  lidar_center_extraction_mode: template
  lidar_min_plane_points: 250
  multi_mode: permutation_search
  max_group_rmse: 0.03
```

ROI 与雷达安装坐标系有关，不能直接认为 5960/5971 的范围适用于下一辆车。应先查看 PCD 中标定板位置，再确定 `coarse` 范围。

`lidar_min_plane_points` 应低于正常标定板 ROI 的有效点数。过高会导致稀疏数据全部失败，过低可能把噪声当作标定板。

### 3.4 C2C 配置

推荐的初始配置如下：

```yaml
c2c_calibration:
  default_group: left
  dictionary: DICT_6X6_250
  marker_id: 1
  marker_size_m: 1.05
  result_name: c2c_extrinsic_result
  groups:
    left:
      enabled: true
      cam0: rear_left
      cam1: front_left
      data_dir: /home/guoli/data/lianyun/下一辆车编号/aruco_single_id1_data/left_pair
      max_cross_0_to_1_mean_px: 1.5
      max_cross_1_to_0_mean_px: 5.0
      min_cross_filter_pairs: 3

    right:
      enabled: true
      cam0: rear_right
      cam1: front_right
      data_dir: /home/guoli/data/lianyun/下一辆车编号/aruco_single_id1_data/right_pair
      max_cross_0_to_1_mean_px: 1.5
      max_cross_1_to_0_mean_px: 5.0
      min_cross_filter_pairs: 3
```

5960 左右两组实际使用 `1.5 / 5.0 px`。5971 右侧使用 `1.5 / 5.0 px`，左侧因数据情况使用 `1.5 / 6.0 px`。

下一辆车建议先使用 `1.5 / 5.0 px`。只有在确认检测、内参、marker 尺寸和投影图都正常后，才考虑小幅放宽阈值；不能为了让程序通过而持续增大阈值。

## 4. 数据目录格式

### 4.1 四路相机—雷达数据

四路数据可以放在同一个车辆目录，通过 `mode_1_`～`mode_4_` 区分：

```text
/home/guoli/data/lianyun/<车辆号>/
├── mode_1_001/
│   ├── 1.bag
│   └── img_0001.jpg
├── mode_1_002/
│   ├── 1.bag
│   └── img_0001.jpg
├── mode_2_001/
│   ├── 1.bag
│   └── img_0001.jpg
├── mode_3_001/
│   ├── 1.bag
│   └── img_0001.jpg
├── mode_4_001/
│   ├── 1.bag
│   └── img_0001.jpg
└── ...
```

每个场景至少需要：

```text
1.bag
img_0001.jpg
```

`1.pcd` 和 `board_candidate.pcd` 由流水线生成，可以在后续重跑中复用。默认流程使用 `img_0001.jpg`；同目录中的 `img_0002.jpg`、`img_0003.jpg` 不会被默认任务使用。

建议每一路采集 15～25 个距离、角度和位置有变化的有效场景。多组求解默认至少需要 6 个有效场景。

### 4.2 C2C 成对图像

```text
/home/guoli/data/lianyun/<车辆号>/aruco_single_id1_data/
├── left_pair/
│   ├── pair_0001_cam0.jpg
│   ├── pair_0001_cam1.jpg
│   ├── pair_0002_cam0.jpg
│   ├── pair_0002_cam1.jpg
│   └── ...
└── right_pair/
    ├── pair_0001_cam0.jpg
    ├── pair_0001_cam1.jpg
    ├── pair_0002_cam0.jpg
    ├── pair_0002_cam1.jpg
    └── ...
```

编号必须成对。推荐采集至少 15 对不同距离、横向位置和姿态的清晰图像。

相机含义不能写反：

| 目录 | cam0 | cam1 |
|---|---|---|
| `left_pair` | `rear_left` | `front_left` |
| `right_pair` | `rear_right` | `front_right` |

## 5. 一条命令执行完整标定

### 5.1 先做预检查

```bash
VEHICLE_ID=下一辆车编号

python3 scripts/run_full_vehicle_calib.py \
  --vehicle "${VEHICLE_ID}" \
  --profile hesai_fast \
  --export-dir "/home/guoli/data/${VEHICLE_ID}" \
  --dry-run
```

预检查会验证：

- 六份相机 YAML 是否存在；
- 四路直接标定的数据、相机配置和 ROI 配置是否存在；
- 四路 `output_dir` 是否互不冲突；
- 每路有多少个完整的 bag/图像场景；
- 左右 C2C 图像是否成对；
- `marker_size_m` 是否有效；
- 实际准备执行的所有命令和路径。

`--dry-run` 不会执行标定。

### 5.2 执行完整流程

Hesai 等稠密机械雷达推荐：

```bash
VEHICLE_ID=下一辆车编号

python3 scripts/run_full_vehicle_calib.py \
  --vehicle "${VEHICLE_ID}" \
  --profile hesai_fast \
  --export-dir "/home/guoli/data/${VEHICLE_ID}"
```

Airy/Livox 等需要累积多帧的稀疏点云使用：

```bash
python3 scripts/run_full_vehicle_calib.py \
  --vehicle "${VEHICLE_ID}" \
  --profile default \
  --export-dir "/home/guoli/data/${VEHICLE_ID}"
```

总入口会按以下顺序运行：

```text
run_vehicle_calib.py --stage all
  -> c2c left
  -> c2c right
  -> summarize_vehicle_extrinsics.py
  -> export_camera_extrinsics_ros_tf.py
```

总入口会显式传入报告目录，不使用汇总脚本中旧机器的默认路径。

## 6. 修改配置后的重跑方式

### 6.1 修改相机内参

直接重新执行完整命令。`--stage all` 会重新做 ROI、标定和投影验证；已有 `1.pcd` 默认会复用：

```bash
python3 scripts/run_full_vehicle_calib.py \
  --vehicle "${VEHICLE_ID}" \
  --profile hesai_fast \
  --export-dir "/home/guoli/data/${VEHICLE_ID}"
```

这也是最不容易遗漏 C2C、汇总或导出的方式。

### 6.2 只修改 C2C marker 尺寸或 C2C 内参

复用已有四路相机—雷达结果：

```bash
python3 scripts/run_full_vehicle_calib.py \
  --vehicle "${VEHICLE_ID}" \
  --profile hesai_fast \
  --skip-direct \
  --export-dir "/home/guoli/data/${VEHICLE_ID}"
```

### 6.3 只重新生成汇总和 227 格式文件

```bash
python3 scripts/run_full_vehicle_calib.py \
  --vehicle "${VEHICLE_ID}" \
  --skip-direct \
  --skip-c2c \
  --export-dir "/home/guoli/data/${VEHICLE_ID}"
```

### 6.4 替换了 bag，但目录名不变

默认配置为：

```yaml
extract_pcd:
  overwrite: false
```

这时旧 `1.pcd` 会被复用。应在受影响的 sensor 配置中临时加入：

```yaml
extract_pcd:
  overwrite: true
```

重新运行完整流程，确认新 PCD 已生成后再改回 `false`。不要在不确认目标目录的情况下批量删除数据。

### 6.5 单路失败后的分阶段恢复

以下命令只处理单路：

```bash
python3 scripts/run_vehicle_calib.py \
  --vehicle "${VEHICLE_ID}" \
  --sensors front \
  --profile hesai_fast \
  --stage roi \
  --stop-on-error

python3 scripts/run_vehicle_calib.py \
  --vehicle "${VEHICLE_ID}" \
  --sensors front \
  --profile hesai_fast \
  --stage calibrate \
  --stop-on-error

python3 scripts/run_vehicle_calib.py \
  --vehicle "${VEHICLE_ID}" \
  --sensors front \
  --profile hesai_fast \
  --stage verify \
  --stop-on-error
```

`--stage calibrate` 只执行 calibrate，不会自动继续 verify。

局部运行会把车辆级直接外参报告覆盖成当前 `--sensors` 子集。修复完成后必须重新导出全部启用的直接相机：

```bash
python3 scripts/run_vehicle_calib.py \
  --vehicle "${VEHICLE_ID}" \
  --profile hesai_fast \
  --stage export
```

随后再执行总入口的汇总阶段：

```bash
python3 scripts/run_full_vehicle_calib.py \
  --vehicle "${VEHICLE_ID}" \
  --skip-direct \
  --skip-c2c \
  --export-dir "/home/guoli/data/${VEHICLE_ID}"
```

分阶段重跑时必须保持同一个 `--profile`。否则生成的 pipeline job 可能从 `board_pcd/middle` 变成 `bag/all`，得到不同的输入数据。

## 7. 输出文件

### 7.1 四路直接标定

```text
/home/guoli/data/lianyun/<车辆号>/_calib_output/<sensor>/
├── 00_manifest.csv
├── 01_pcd_export_summary.csv
├── 02_roi/
├── 03_single/
├── 04_multi/
├── 05_verify/
│   ├── image_grid_verify.jpg
│   └── verify_summary.csv
├── batch_summary.csv
└── final_extrinsic.yaml
```

车辆级直接外参报告：

```text
vehicle_calib_report/<车辆号>/
├── <车辆号>_summary.csv
├── <车辆号>_extrinsics.yaml
├── <车辆号>_extrinsics_compact.yaml
├── <车辆号>_extrinsics_readable.yaml
└── run_logs/
```

### 7.2 C2C

每个 `left_pair`、`right_pair` 下生成：

```text
c2c_extrinsic_result.yaml
c2c_extrinsic_result_per_pair_errors.csv
c2c_extrinsic_result_summary.txt
c2c_extrinsic_result_debug_reprojection/
c2c_extrinsic_result_validation_projection/
└── used_projection_contact_sheet.jpg
```

### 7.3 六相机汇总

```text
vehicle_calib_report/<车辆号>/
├── <车辆号>_all_camera_extrinsics.yaml
├── <车辆号>_all_camera_extrinsics_compact.yaml
├── <车辆号>_all_camera_extrinsics_readable.yaml
└── <车辆号>_all_camera_extrinsics_summary.csv
```

`*_all_camera_extrinsics.yaml` 是后续导出的正式输入，必须包含：

```text
front
front_left
front_right
rear
rear_left
rear_right
```

### 7.4 227 格式 ROS TF

默认示例输出目录：

```text
/home/guoli/data/<车辆号>/
├── camera-extrinsic-front.yaml
├── camera-extrinsic-front-left.yaml
├── camera-extrinsic-front-right.yaml
├── camera-extrinsic-rear.yaml
├── camera-extrinsic-rear-left.yaml
└── camera-extrinsic-rear-right.yaml
```

这些文件表达 `T_lidar_camera`，格式与 `/home/guoli/data/227` 一致。

## 8. 标定质量验收

命令退出码为 0 不代表结果一定合格。必须检查数值和投影图。

### 8.1 四路直接标定

检查每一路：

```text
_calib_output/<sensor>/final_extrinsic.yaml
_calib_output/<sensor>/batch_summary.csv
_calib_output/<sensor>/05_verify/image_grid_verify.jpg
vehicle_calib_report/<车辆号>/<车辆号>_summary.csv
```

建议验收条件：

- `final_extrinsic.yaml` 包含完整 `T_cam_lidar`；
- `status: ok` 优先；
- 默认多组 RMSE 目标为不超过 `0.03 m`；
- `selected_groups` 建议至少 5～6 组；
- `selected_count / four_center_count` 不应明显低于 0.4；
- `final_group_residuals` 不应存在大量离群场景；
- 投影图中红色投影应整体落在真实标定板/圆孔位置上；
- 不应出现系统性偏移、镜像、旋转方向错误或只在个别场景对齐。

`status: warn` 不能自动当作合格，也不一定代表结果必须作废。必须结合 RMSE、有效组数和所有验证投影人工确认。

### 8.2 左右 C2C

检查：

```text
left_pair/c2c_extrinsic_result.yaml
right_pair/c2c_extrinsic_result.yaml
*/c2c_extrinsic_result_validation_projection/used_projection_contact_sheet.jpg
```

建议验收条件：

- `vehicle_id` 与当前车辆一致；
- `marker_size_m` 与 `vehicle.yaml` 一致；
- `num_pairs_used >= 3`，建议有更多有效对；
- 第一遍过滤要求单对图像两侧 PnP 均值不超过 `0.8 px`；
- cross reprojection 均值不超过配置阈值；
- 联系表中绿色检测点与红色投影点应基本重合；
- 左右 `baseline_m` 应符合车辆物理尺寸，并且两侧不能出现明显不合理差异。

如果第一遍过滤后少于 3 对，脚本会回退使用全部有效对，因此最终 PnP 均值仍可能超过 `0.8 px`。出现回退时要重点检查每对误差和投影图，不能把“命令成功”当作质量合格。

不要只看平均误差。少量严重错位、所有图像姿态过于相似、marker 尺寸写错，都可能使最终外参不可靠。

### 8.3 六相机汇总和导出

检查：

- `*_all_camera_extrinsics.yaml` 中恰好有六个相机；
- `warnings` 为空；
- `front_right` 的来源为 `rear_right + right C2C`；
- `rear_left` 的来源为 `front_left + inverse(left C2C)`；
- 每个相机都有非空 `lidar_topic`、`T_cam_lidar` 和 `T_lidar_cam`；
- `/home/guoli/data/<车辆号>` 中存在六个 `camera-extrinsic-*.yaml`。

## 9. 常见问题

### 所有相机数据放在同一个目录

这是支持的。必须用不同 `layout.group_prefix` 选择数据，并用不同 `output_dir` 保存结果。

### C2C marker 尺寸改了但结果没有变化

确认修改的是：

```yaml
c2c_calibration:
  marker_size_m: ...
```

然后重跑 C2C。相机 YAML 中的 `marker_size` 是另一块标定板的参数。

### 修改内参后只重跑了 summary

summary 只组合已有外参，不会重新计算外参。内参变化后必须重新执行直接标定和 C2C。

### `lidar_topic: auto` 识别失败

检查 bag 中是否没有点云话题或存在多个点云话题。存在多个时，在 `vehicle.yaml` 中明确填写正确话题。

### ROI 没有覆盖标定板

单独重跑 `scan/extract_pcd/roi`，查看 PCD 后修改 `roi.coarse`。不要通过降低所有几何约束来掩盖错误 ROI。

### C2C 有很多照片但 `num_pairs_all` 很少

`num_pairs_all` 是检测和 PnP 成功的数量，不是磁盘文件数量。检查：

- cam0/cam1 是否写反；
- marker id 和 dictionary；
- marker 是否完整、清晰、没有过曝；
- 六份内参是否对应正确相机；
- 图像是否成对编号。

### 旧 debug 图混入新结果

C2C 会重写正式 YAML 和联系表，但调试目录可能保留旧单帧图片。需要比较两次结果时，建议先把旧结果目录整体归档，再运行新标定。

## 10. 手动执行完整流程

总入口失败时，可以按下面顺序定位问题。路径必须显式给出，避免使用旧机器的默认报告目录。

```bash
CALIB_ROOT=/home/guoli/proj/fast_calib_ws/src/FAST-Calib
VEHICLE_ID=下一辆车编号
DATA_ROOT="/home/guoli/data/lianyun/${VEHICLE_ID}"
REPORT_ROOT="${CALIB_ROOT}/vehicle_calib_report"
REPORT_DIR="${REPORT_ROOT}/${VEHICLE_ID}"
EXPORT_DIR="/home/guoli/data/${VEHICLE_ID}"

cd "${CALIB_ROOT}"
source /opt/ros/noetic/setup.bash
source /home/guoli/proj/fast_calib_ws/devel/setup.bash
export FAST_CALIB_REPORT_ROOT="${REPORT_ROOT}"
```

四路直接标定：

```bash
python3 scripts/run_vehicle_calib.py \
  --vehicle "${VEHICLE_ID}" \
  --profile hesai_fast \
  --stage all \
  --stop-on-error
```

左侧 C2C：

```bash
python3 scripts/c2c_calibrate_vehicle_aruco.py \
  --vehicle "${VEHICLE_ID}" \
  --config-root "${CALIB_ROOT}/config/vehicles" \
  --group left \
  --max-cross-0-to-1-mean 1.5 \
  --max-cross-1-to-0-mean 5.0 \
  --min-cross-filter-pairs 3 \
  --save-debug \
  --save-validation
```

右侧 C2C：

```bash
python3 scripts/c2c_calibrate_vehicle_aruco.py \
  --vehicle "${VEHICLE_ID}" \
  --config-root "${CALIB_ROOT}/config/vehicles" \
  --group right \
  --max-cross-0-to-1-mean 1.5 \
  --max-cross-1-to-0-mean 5.0 \
  --min-cross-filter-pairs 3 \
  --save-debug \
  --save-validation
```

六相机汇总：

```bash
python3 scripts/summarize_vehicle_extrinsics.py \
  --vehicle "${VEHICLE_ID}" \
  --config-root "${CALIB_ROOT}/config/vehicles" \
  --direct-extrinsics "${REPORT_DIR}/${VEHICLE_ID}_extrinsics.yaml" \
  --c2c-root "${DATA_ROOT}/aruco_single_id1_data" \
  --c2c-result-name c2c_extrinsic_result \
  --output-dir "${REPORT_DIR}" \
  --output-prefix "${VEHICLE_ID}_all_camera_extrinsics"
```

导出 227 格式：

```bash
python3 scripts/export_camera_extrinsics_ros_tf.py \
  "${REPORT_DIR}/${VEHICLE_ID}_all_camera_extrinsics.yaml" \
  "${EXPORT_DIR}"
```
