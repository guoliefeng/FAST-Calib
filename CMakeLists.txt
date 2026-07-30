cmake_minimum_required(VERSION 3.0.2)
project(fast_calib)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_ros
  pcl_conversions
  geometry_msgs
)

find_package(PCL 1.10 REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs pcl_conversions pcl_ros
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_BUILD_TYPE Release)

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

catkin_install_python(PROGRAMS
  scripts/c2c_calibrate_aruco_from_config.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_executable(fast_calib src/main.cpp)

target_link_libraries(fast_calib
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(multi_fast_calib src/multi_scene.cpp)
target_link_libraries(multi_fast_calib
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(two_camera_calib src/two_camera_calib.cpp)
target_link_libraries(two_camera_calib
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(two_camera_calib_fixed src/two_camera_calib_fixed.cpp)
target_link_libraries(two_camera_calib_fixed
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

add_executable(two_camera_calib_solvepnp src/two_camera_calib_solvepnp.cpp)
target_link_libraries(two_camera_calib_solvepnp
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

catkin_install_python(PROGRAMS
  scripts/batch_calib.py
  scripts/calib_pipeline.py
  scripts/run_four_calib_jobs.py
  scripts/detect_airy_left_rear_roi.py
  scripts/run_vehicle_calib.py
  scripts/inspect_lidar_centers.py
  scripts/recompute_extrinsic_from_manual_centers.py
  scripts/c2c_calibrate_vehicle_aruco.py
  scripts/summarize_vehicle_extrinsics.py
  scripts/export_camera_extrinsics_ros_tf.py
  scripts/run_full_vehicle_calib.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  docs/vehicle_camera_calibration_workflow.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/docs
)
