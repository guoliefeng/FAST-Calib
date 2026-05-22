# Vehicle camera extrinsics and TF launch

This note explains how to regenerate vehicle camera extrinsics, ROS static TF launch files, and camera poses in `base_link`.

## Quick run

For vehicle 221:

```bash
cd /home/glf/dataDisk/calib/FAST-Calib_ws/src/FAST-Calib
python3 scripts/summarize_vehicle_extrinsics.py --vehicle 221 --emit-tf-launch
```

Main outputs are written under:

```text
/home/glf/dataDisk/calib/vehicle_calib_report/221/
```

Important files:

```text
221_all_camera_extrinsics_readable.yaml
221_all_camera_extrinsics_tf.launch
221_all_camera_extrinsics_base_camera_tf.launch
221_all_camera_extrinsics_base_camera_extrinsics.yaml
```

Use `221_all_camera_extrinsics_tf.launch` when you want a normal TF tree:

```text
base_link -> lidar_first  -> front_camera_frame
base_link -> lidar_first  -> front_left_camera_frame
base_link -> lidar_first  -> rear_left_camera_frame
base_link -> lidar_second -> rear_camera_frame
base_link -> lidar_second -> rear_right_camera_frame
base_link -> lidar_second -> front_right_camera_frame
```

Use `221_all_camera_extrinsics_base_camera_tf.launch` only when an engineer wants direct `base_link -> camera_frame` transforms. Do not launch it together with `221_all_camera_extrinsics_tf.launch`, because the same camera frame would then have two TF parents.

## Input sources

The script reads direct camera-LiDAR calibration from:

```text
/home/glf/dataDisk/calib/vehicle_calib_report/<vehicle>/<vehicle>_extrinsics.yaml
```

It reads camera-camera calibration from the vehicle config:

```text
config/vehicles/<vehicle>/vehicle.yaml
```

For 221, the camera-camera groups are:

```text
left:  rear_left  -> front_left
right: rear_right -> front_right
```

If one camera in a pair has direct LiDAR calibration and the other does not, the script derives the missing camera with the C2C result.

## Coordinate conventions

Direct camera-LiDAR calibration is stored as `T_cam_lidar`:

```text
p_cam = T_cam_lidar * p_lidar
```

ROS static TF uses parent-to-child transforms. For a LiDAR parent and camera child, the script publishes:

```text
T_lidar_camera = inverse(T_cam_lidar)
```

For direct camera poses in `base_link`, the script computes:

```text
T_base_camera = T_base_lidar * T_lidar_camera
```

The `*_base_camera_extrinsics.yaml` file stores both matrix and quaternion forms:

```text
cameras.<camera>.T_lidar_camera
cameras.<camera>.T_base_camera
```

## Vehicle 221 defaults

Vehicle 221 has built-in defaults for the two LiDAR poses:

```text
base_link -> lidar_first:
  xyz  = 7.38763 1.3081 1.6
  qxyzw = 0.00559065 0.00440275 0.18853 0.982042

base_link -> lidar_second:
  xyz  = -7.41613 -1.38264 1.6
  qxyzw = 0.00169926 -0.00675155 0.983151 -0.182665
```

The generated launch may show the equivalent quaternion sign for `lidar_second`. Quaternion `q` and `-q` represent the same rotation.

## Override LiDAR poses

Override the 221 defaults from the command line:

```bash
python3 scripts/summarize_vehicle_extrinsics.py --vehicle 221 --emit-tf-launch \
  --base-lidar-first "7.38763 1.3081 1.6 0.00559065 0.00440275 0.18853 0.982042" \
  --base-lidar-second "-7.41613 -1.38264 1.6 0.00169926 -0.00675155 0.983151 -0.182665"
```

Or provide a YAML file:

```yaml
base_frame: base_link
lidars:
  lidar_first:
    xyz: [7.38763, 1.3081, 1.6]
    quaternion_xyzw: [0.00559065, 0.00440275, 0.18853, 0.982042]
  lidar_second:
    xyz: [-7.41613, -1.38264, 1.6]
    quaternion_xyzw: [0.00169926, -0.00675155, 0.983151, -0.182665]
```

Run with:

```bash
python3 scripts/summarize_vehicle_extrinsics.py --vehicle 221 --emit-tf-launch \
  --base-lidar-tf-yaml /path/to/base_lidar_tf.yaml
```

## Output file meaning

`*_all_camera_extrinsics_readable.yaml` is the human-readable summary of all cameras. Check this first to confirm the selected calibration version and RMSE.

`*_all_camera_extrinsics_tf.launch` is the recommended TF launch. It publishes `base_link -> lidar` and `lidar -> camera`.

`*_all_camera_extrinsics_base_camera_tf.launch` publishes direct `base_link -> camera` transforms for engineering conversion. Do not use it together with the recommended TF launch.

`*_all_camera_extrinsics_base_camera_extrinsics.yaml` stores the same direct base-camera results as structured data, including 4x4 matrices.

## Common checks

For 221 after the latest regeneration, `front_left` should show:

```text
rmse: 0.008104
```

If it still shows about `0.010245`, the all-camera summary is stale. Rerun:

```bash
python3 scripts/summarize_vehicle_extrinsics.py --vehicle 221 --emit-tf-launch
```

If `$(find fast_calib)` cannot be resolved, run the command from this repository root or source the catkin workspace.

