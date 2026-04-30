# Two-camera calibration fixed executable

This patch adds an isolated executable:

```bash
rosrun fast_calib two_camera_calib_fixed /path/to/two_camera_left.yaml
```

or:

```bash
roslaunch fast_calib two_camera_calib_fixed.launch config:=/path/to/two_camera_left.yaml
```

The YAML should use absolute paths for `data_dir` and `output_dir`, for example:

```yaml
data_dir: "/root/calib_ws/two_camera_data/left_pair"
output_dir: "/root/calib_ws/output/two_camera/left_pair_rear_to_front"
```

Convention:

```text
X_cam1 = T_cam1_cam0 * X_cam0
```

For your left config:

```text
cam0 = rear_left
cam1 = front_left
X_front_left = T_front_left_rear_left * X_rear_left
```
