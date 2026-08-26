# 相机外参标定说明


## 设计方案

### 标定拓扑

优先直接标定四路相机，另外两路由相机间外参（C2C）推导：

| 相机 | 所属 LiDAR | 来源 |
|---|---|---|
| `front`、`front_left` | `lidar_left_front` | 相机-LiDAR 直接标定 |
| `rear`、`rear_right` | `lidar_right_rear` | 相机-LiDAR 直接标定 |
| `rear_left` | `lidar_left_front` | `front_left` + 左侧 C2C |
| `front_right` | `lidar_right_rear` | `rear_right` + 右侧 C2C |


左侧 C2C 固定为 `rear_left -> front_left`，右侧固定为 `rear_right -> front_right`。

整体流程：

**数据采集 → 标定板检测 → Camera–LiDAR 外参计算 → C2C 外参计算  → 投影验证** 


### 输出与验收

- 直接标定输出：`<data_dir>/_calib_output/final_extrinsic.yaml`，核心矩阵为 `T_cam_lidar`。
- 组内验证：`05_verify/*_board_overlay.png`，红色为 LiDAR 标定板点云投影。
- C2C 输出：`c2c_extrinsic_result*.yaml` 和 `*_validation_projection/used_projection_contact_sheet.jpg`。

## 当前问题

1. **标定板点云ROI**：目前标定板点云ROI 提取不一定准确， 有时需要人为在得到ROI范围。
2. **标定板点云圆心提取困难**：有时标定板的点云过于稀疏以及噪声较大，使得圆心提取不好。
3. **当前操作不方便**：当前标定流程过于繁琐，应该是车上计算得到外参，例如车上启动程序，然后在标定板放好，便可得到外参。
4. **标定方案可靠性存疑**：当前标定方案仅作为临时方案处理，后期都是靠相机与雷达直接的关系得到外参，其中有两个相机只能通过c2c的方式得到外参，这样存在累计误差，比如相机到base_link 的误差存在i相机和雷达间的误差以及雷达与base_link 的误差，两层误差传播，而车上两个用c2c标定的相机存在三层误差，如果感知仅使用相机和雷达间的投影关系那么仅存在相机和雷达的误差，但如果感知使用相机和base_link 的关系，那么还会雷达到base_link这一层的误差，如果后期想要提高标定精度，这一层是一定要考虑到的，如果是想要相机和base_link 的外参，可考虑相机与 INS标定，如此一来，相机和base_link 的传播误差会少一层，但是相机与雷达的误差会多一层，这种资源的协调的分配需要考虑，哪个需要分配多些，哪些需要分配少些，需商榷。
5. **方案需调研充分**：当前标定方案用FAST-Calib ，确实可以得到外参，但是可靠性不一定好。后期可以参考其他论文，比如 General, Single-shot, Target-less, and Automatic LiDAR-Camera Extrinsic Calibration Toolbox ，此论文是最近几年 Targetless LiDAR-Camera 标定的标杆项目之一，以及Direct, Targetless and Automatic Joint Calibration of LiDAR-Camera Intrinsic and Extrinsic \ PLK-Calib: Single-shot and Target-less LiDAR-Camera Extrinsic Calibration using Plücker Lines\ RAVES-Calib:Robust, Accurate and Versatile Extrinsic Self Calibration Using Optimal Geometric Features . 如果是相机与INS 的标定，基本都是手眼标定，可以参考：zxl19/Hand_Eye_Extrinsic_Calibration ，此项目支持lidar -ins , camera- ins ,lidar - camera,甚至三传感器标定。还有GNSS-Aided Online Camera Calibration ，还可以把camera-imu-GNSS 一起标定；总而言之，后面的同事要开展此工作，需先调研充分。

## 操作说明

### 1. 准备配置

检查 `config/vehicles/<vehicle>/`：

- 六份 `cameras/*.yaml` 的内参；
- `vehicle.yaml` 的数据目录、相机-LiDAR对应关系和 C2C 相机顺序；
- ROI 与当前 LiDAR 坐标系一致；
- `marker_size`（四孔板小码）与 `c2c_calibration.marker_size_m`（C2C 单码）分别按实物填写。

### 2. 直接标定四路相机

```bash


python3 scripts/run_vehicle_calib.py --vehicle <vehicle> --stage all

# 例如：
python3 scripts/run_vehicle_calib.py --vehicle 221  --stage all
```

### 3 C2C 标定
python3 scripts/c2c_calibrate_vehicle_aruco.py \
    --vehicle <vehicle> --group left --save-validation

python3 scripts/c2c_calibrate_vehicle_aruco.py \
    --vehicle <vehicle> --group right --save-validation

完成左右两组 Camera–Camera 外参计算。

### 3. 结果检查

主要检查：

  Camera–LiDAR 标定板点云投影是否与图像目标重合；
  C2C 投影结果是否一致；
##  当前结论

现有方案已经具备六路相机外参标定和统一输出能力，可以满足当前车辆标定使用。

下一阶段重点不是继续增加人工操作，而是提高：

自动化程度、特征提取稳定性、标定精度和结果自检能力。

最终目标是形成一套可在车端运行的标准化、一键式相机外参标定工具。