# Two-camera solvePnP calibration

Run:

```bash
roslaunch fast_calib two_camera_calib_solvepnp.launch \
  config:=/home/glf/dataDisk/calib/two_camera_left.yaml
```

The executable uses explicit ArUco corner 3D-2D correspondences and `cv::solvePnP(..., SOLVEPNP_ITERATIVE)`.

Convention:

```text
X_cam1 = T_cam1_cam0 * X_cam0
```

For the left-pair YAML:

```text
cam0 = rear_left
cam1 = front_left
T_cam1_cam0 = T_front_left_rear_left
```
