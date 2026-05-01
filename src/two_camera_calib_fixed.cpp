/*
 * two_camera_calib.cpp
 *
 * ROS1 node for camera-camera extrinsic calibration using a 4-ArUco FAST-Calib board.
 *
 * Input:
 *   YAML config, for example:
 *
 *   pair_name: "left_pair_rear_to_front"
 *   data_dir: "/root/calib_ws/two_camera_data/left_pair"
 *   output_dir: "/root/calib_ws/output/two_camera/left_pair_rear_to_front"
 *
 *   cam0:
 *     name: "rear_left"
 *     fx: 1020.321275
 *     fy: 1026.490039
 *     cx: 958.172619
 *     cy: 522.228574
 *     k1: -0.289258
 *     k2: 0.056707
 *     p1: 0.001357
 *     p2: -0.000796
 *
 *   cam1:
 *     name: "front_left"
 *     fx: 1012.229557
 *     fy: 1013.831352
 *     cx: 925.288190
 *     cy: 514.768269
 *     k1: -0.303981
 *     k2: 0.067020
 *     p1: 0.000418
 *     p2: 0.000216
 *
 *   target:
 *     dictionary: "DICT_6X6_250"
 *     marker_size_m: 0.20
 *     delta_width_qr_center_m: 0.55
 *     delta_height_qr_center_m: 0.35
 *     delta_width_circles_m: 0.50
 *     delta_height_circles_m: 0.40
 *
 *   runtime:
 *     min_detected_markers: 3
 *     refine_markers: false
 *     reproj_rmse_thresh_px: 6.0
 *     outlier_rot_thresh_deg: 1.0
 *     outlier_trans_thresh_m: 0.10
 *     min_valid_pairs: 2
 *
 * Output convention:
 *   X_cam1 = T_cam1_cam0 * X_cam0
 *
 * For flat data layout:
 *   data_dir/
 *     pair_0001_cam0.jpg
 *     pair_0001_cam1.jpg
 *     pair_0002_cam0.jpg
 *     pair_0002_cam1.jpg
 *
 * Notes:
 *   - This file intentionally does not depend on FAST-Calib's LiDAR code.
 *   - ArUco board geometry follows FAST-Calib's QRDetect:
 *       board marker ids: 1, 2, 4, 3
 *       marker 1: top-left
 *       marker 2: top-right
 *       marker 4: bottom-right
 *       marker 3: bottom-left
 */

#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <dirent.h>

namespace {

constexpr double kPi = 3.14159265358979323846;

double rad2deg(double x) { return x * 180.0 / kPi; }
double deg2rad(double x) { return x * kPi / 180.0; }

bool pathExists(const std::string& path) {
  struct stat buffer;
  return stat(path.c_str(), &buffer) == 0;
}

bool makeDirRecursive(const std::string& dir) {
  if (dir.empty()) return false;
  if (pathExists(dir)) return true;

  std::string cmd = "mkdir -p \"" + dir + "\"";
  int ret = std::system(cmd.c_str());
  return ret == 0 && pathExists(dir);
}

std::string joinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

std::string extLower(const std::string& path) {
  const auto pos = path.find_last_of('.');
  if (pos == std::string::npos) return "";
  std::string e = path.substr(pos + 1);
  std::transform(e.begin(), e.end(), e.begin(), ::tolower);
  return e;
}

bool isImageExt(const std::string& e) {
  return e == "jpg" || e == "jpeg" || e == "png" || e == "bmp";
}

std::vector<std::string> listFiles(const std::string& dir) {
  std::vector<std::string> files;
  DIR* dp = opendir(dir.c_str());
  if (!dp) return files;

  struct dirent* ep = nullptr;
  while ((ep = readdir(dp)) != nullptr) {
    std::string name(ep->d_name);
    if (name == "." || name == "..") continue;
    files.push_back(name);
  }
  closedir(dp);
  std::sort(files.begin(), files.end());
  return files;
}

struct CameraConfig {
  std::string name;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  double k1 = 0.0;
  double k2 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double k3 = 0.0;

  cv::Mat K() const {
    return (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
                                      0.0, fy, cy,
                                      0.0, 0.0, 1.0);
  }

  cv::Mat D() const {
    return (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, k3);
  }
};

struct TargetConfig {
  std::string dictionary = "DICT_6X6_250";
  double marker_size_m = 0.20;
  double delta_width_qr_center_m = 0.55;
  double delta_height_qr_center_m = 0.35;
  double delta_width_circles_m = 0.50;
  double delta_height_circles_m = 0.40;
};

struct RuntimeConfig {
  int min_detected_markers = 3;
  bool refine_markers = false;
  double reproj_rmse_thresh_px = 6.0;
  double outlier_rot_thresh_deg = 1.0;
  double outlier_trans_thresh_m = 0.10;
  int min_valid_pairs = 2;
  bool save_debug = true;
};

struct AppConfig {
  std::string pair_name = "two_camera_pair";
  std::string data_dir;
  std::string output_dir;

  CameraConfig cam0;
  CameraConfig cam1;
  TargetConfig target;
  RuntimeConfig runtime;
};

struct ImagePair {
  std::string pair_id;
  std::string cam0_path;
  std::string cam1_path;
};

struct PoseResult {
  bool ok = false;
  int markers = 0;
  double reproj_rmse_px = -1.0;
  std::vector<int> ids;
  cv::Mat debug_image;
  Eigen::Matrix4d T_cam_board = Eigen::Matrix4d::Identity();
};

struct PairResult {
  std::string pair_id;
  bool ok0 = false;
  bool ok1 = false;
  bool hard_valid = false;
  bool inlier = false;

  int markers0 = 0;
  int markers1 = 0;
  double reproj0 = -1.0;
  double reproj1 = -1.0;

  double rot_err_deg = -1.0;
  double trans_err_m = -1.0;

  Eigen::Matrix4d T_cam1_cam0 = Eigen::Matrix4d::Identity();
};

double readDouble(const cv::FileNode& n, const std::string& key, double default_value) {
  cv::FileNode v = n[key];
  if (v.empty()) return default_value;
  return static_cast<double>(v);
}

int readInt(const cv::FileNode& n, const std::string& key, int default_value) {
  cv::FileNode v = n[key];
  if (v.empty()) return default_value;
  return static_cast<int>(v);
}

bool readBool(const cv::FileNode& n, const std::string& key, bool default_value) {
  cv::FileNode v = n[key];
  if (v.empty()) return default_value;
  if (v.isInt()) return static_cast<int>(v) != 0;
  if (v.isString()) {
    std::string s = static_cast<std::string>(v);
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s == "true" || s == "1" || s == "yes";
  }
  return default_value;
}

std::string readString(const cv::FileNode& n, const std::string& key, const std::string& default_value) {
  cv::FileNode v = n[key];
  if (v.empty()) return default_value;
  return static_cast<std::string>(v);
}

CameraConfig readCamera(const cv::FileNode& n, const std::string& key) {
  cv::FileNode c = n[key];
  if (c.empty()) {
    throw std::runtime_error("Missing camera block: " + key);
  }

  CameraConfig cam;
  cam.name = readString(c, "name", key);
  cam.fx = readDouble(c, "fx", 0.0);
  cam.fy = readDouble(c, "fy", 0.0);
  cam.cx = readDouble(c, "cx", 0.0);
  cam.cy = readDouble(c, "cy", 0.0);
  cam.k1 = readDouble(c, "k1", 0.0);
  cam.k2 = readDouble(c, "k2", 0.0);
  cam.p1 = readDouble(c, "p1", 0.0);
  cam.p2 = readDouble(c, "p2", 0.0);
  cam.k3 = readDouble(c, "k3", 0.0);

  if (cam.fx <= 0.0 || cam.fy <= 0.0) {
    throw std::runtime_error("Invalid camera intrinsics for " + key);
  }
  return cam;
}

AppConfig readConfigFromYaml(const std::string& config_path) {
  cv::FileStorage fs(config_path, cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw std::runtime_error("Cannot open config file: " + config_path);
  }

  AppConfig cfg;
  cfg.pair_name = readString(fs.root(), "pair_name", cfg.pair_name);
  cfg.data_dir = readString(fs.root(), "data_dir", "");
  cfg.output_dir = readString(fs.root(), "output_dir", "");

  if (cfg.data_dir.empty()) throw std::runtime_error("data_dir is empty.");
  if (cfg.output_dir.empty()) throw std::runtime_error("output_dir is empty.");

  cfg.cam0 = readCamera(fs.root(), "cam0");
  cfg.cam1 = readCamera(fs.root(), "cam1");

  cv::FileNode target = fs["target"];
  if (!target.empty()) {
    cfg.target.dictionary = readString(target, "dictionary", cfg.target.dictionary);
    cfg.target.marker_size_m = readDouble(target, "marker_size_m", cfg.target.marker_size_m);
    cfg.target.delta_width_qr_center_m = readDouble(target, "delta_width_qr_center_m", cfg.target.delta_width_qr_center_m);
    cfg.target.delta_height_qr_center_m = readDouble(target, "delta_height_qr_center_m", cfg.target.delta_height_qr_center_m);
    cfg.target.delta_width_circles_m = readDouble(target, "delta_width_circles_m", cfg.target.delta_width_circles_m);
    cfg.target.delta_height_circles_m = readDouble(target, "delta_height_circles_m", cfg.target.delta_height_circles_m);
  }

  cv::FileNode runtime = fs["runtime"];
  if (!runtime.empty()) {
    cfg.runtime.min_detected_markers = readInt(runtime, "min_detected_markers", cfg.runtime.min_detected_markers);
    cfg.runtime.refine_markers = readBool(runtime, "refine_markers", cfg.runtime.refine_markers);
    cfg.runtime.reproj_rmse_thresh_px = readDouble(runtime, "reproj_rmse_thresh_px", cfg.runtime.reproj_rmse_thresh_px);
    cfg.runtime.outlier_rot_thresh_deg = readDouble(runtime, "outlier_rot_thresh_deg", cfg.runtime.outlier_rot_thresh_deg);
    cfg.runtime.outlier_trans_thresh_m = readDouble(runtime, "outlier_trans_thresh_m", cfg.runtime.outlier_trans_thresh_m);
    cfg.runtime.min_valid_pairs = readInt(runtime, "min_valid_pairs", cfg.runtime.min_valid_pairs);
    cfg.runtime.save_debug = readBool(runtime, "save_debug", cfg.runtime.save_debug);
  }

  return cfg;
}

template <typename T>
bool getRequiredRosParam(const ros::NodeHandle& nh, const std::string& key, T& value) {
  if (!nh.getParam(key, value)) {
    ROS_ERROR_STREAM("Missing required private param: " << nh.getNamespace() << "/" << key);
    return false;
  }
  return true;
}

template <typename T>
void getOptionalRosParam(const ros::NodeHandle& nh, const std::string& key, T& value) {
  nh.param(key, value, value);
}

bool readCameraFromRosParams(const ros::NodeHandle& nh,
                             const std::string& prefix,
                             CameraConfig& cam) {
  if (!getRequiredRosParam(nh, prefix + "/name", cam.name) ||
      !getRequiredRosParam(nh, prefix + "/fx", cam.fx) ||
      !getRequiredRosParam(nh, prefix + "/fy", cam.fy) ||
      !getRequiredRosParam(nh, prefix + "/cx", cam.cx) ||
      !getRequiredRosParam(nh, prefix + "/cy", cam.cy) ||
      !getRequiredRosParam(nh, prefix + "/k1", cam.k1) ||
      !getRequiredRosParam(nh, prefix + "/k2", cam.k2) ||
      !getRequiredRosParam(nh, prefix + "/p1", cam.p1) ||
      !getRequiredRosParam(nh, prefix + "/p2", cam.p2)) {
    return false;
  }

  getOptionalRosParam(nh, prefix + "/k3", cam.k3);
  return true;
}

AppConfig readConfigFromRosParams(const ros::NodeHandle& nh) {
  AppConfig cfg;
  if (!getRequiredRosParam(nh, "pair_name", cfg.pair_name) ||
      !getRequiredRosParam(nh, "data_dir", cfg.data_dir) ||
      !getRequiredRosParam(nh, "output_dir", cfg.output_dir)) {
    throw std::runtime_error("Missing required ROS parameters.");
  }

  if (!readCameraFromRosParams(nh, "cam0", cfg.cam0) ||
      !readCameraFromRosParams(nh, "cam1", cfg.cam1)) {
    throw std::runtime_error("Missing required camera ROS parameters.");
  }

  getOptionalRosParam(nh, "target/dictionary", cfg.target.dictionary);
  getOptionalRosParam(nh, "target/marker_size_m", cfg.target.marker_size_m);
  getOptionalRosParam(nh, "target/delta_width_qr_center_m", cfg.target.delta_width_qr_center_m);
  getOptionalRosParam(nh, "target/delta_height_qr_center_m", cfg.target.delta_height_qr_center_m);
  getOptionalRosParam(nh, "target/delta_width_circles_m", cfg.target.delta_width_circles_m);
  getOptionalRosParam(nh, "target/delta_height_circles_m", cfg.target.delta_height_circles_m);

  getOptionalRosParam(nh, "runtime/min_detected_markers", cfg.runtime.min_detected_markers);
  getOptionalRosParam(nh, "runtime/refine_markers", cfg.runtime.refine_markers);
  getOptionalRosParam(nh, "runtime/reproj_rmse_thresh_px", cfg.runtime.reproj_rmse_thresh_px);
  getOptionalRosParam(nh, "runtime/outlier_rot_thresh_deg", cfg.runtime.outlier_rot_thresh_deg);
  getOptionalRosParam(nh, "runtime/outlier_trans_thresh_m", cfg.runtime.outlier_trans_thresh_m);
  getOptionalRosParam(nh, "runtime/min_valid_pairs", cfg.runtime.min_valid_pairs);
  getOptionalRosParam(nh, "runtime/save_debug", cfg.runtime.save_debug);

  return cfg;
}

int arucoDictionaryId(const std::string& name) {
  // FAST-Calib default.
  if (name == "DICT_6X6_250") return cv::aruco::DICT_6X6_250;

  if (name == "DICT_4X4_50") return cv::aruco::DICT_4X4_50;
  if (name == "DICT_4X4_100") return cv::aruco::DICT_4X4_100;
  if (name == "DICT_4X4_250") return cv::aruco::DICT_4X4_250;
  if (name == "DICT_4X4_1000") return cv::aruco::DICT_4X4_1000;

  if (name == "DICT_5X5_50") return cv::aruco::DICT_5X5_50;
  if (name == "DICT_5X5_100") return cv::aruco::DICT_5X5_100;
  if (name == "DICT_5X5_250") return cv::aruco::DICT_5X5_250;
  if (name == "DICT_5X5_1000") return cv::aruco::DICT_5X5_1000;

  if (name == "DICT_6X6_50") return cv::aruco::DICT_6X6_50;
  if (name == "DICT_6X6_100") return cv::aruco::DICT_6X6_100;
  if (name == "DICT_6X6_1000") return cv::aruco::DICT_6X6_1000;

  if (name == "DICT_7X7_50") return cv::aruco::DICT_7X7_50;
  if (name == "DICT_7X7_100") return cv::aruco::DICT_7X7_100;
  if (name == "DICT_7X7_250") return cv::aruco::DICT_7X7_250;
  if (name == "DICT_7X7_1000") return cv::aruco::DICT_7X7_1000;

  throw std::runtime_error("Unsupported ArUco dictionary: " + name);
}

void buildFastCalibBoard(
    const TargetConfig& target,
    std::vector<std::vector<cv::Point3f>>& board_corners,
    std::vector<int>& board_ids) {
  board_corners.clear();
  board_corners.resize(4);

  const float width = static_cast<float>(target.delta_width_qr_center_m);
  const float height = static_cast<float>(target.delta_height_qr_center_m);
  const float marker_size = static_cast<float>(target.marker_size_m);

  for (int i = 0; i < 4; ++i) {
    const int x_qr_center = (i % 3) == 0 ? -1 : 1;  // i=0,3 left; i=1,2 right
    const int y_qr_center = (i < 2) ? 1 : -1;       // i=0,1 top;  i=2,3 bottom

    const float x_center = x_qr_center * width;
    const float y_center = y_qr_center * height;

    for (int j = 0; j < 4; ++j) {
      const int x_qr = (j % 3) == 0 ? -1 : 1;
      const int y_qr = (j < 2) ? 1 : -1;

      board_corners[i].push_back(
          cv::Point3f(x_center + x_qr * marker_size / 2.0f,
                      y_center + y_qr * marker_size / 2.0f,
                      0.0f));
    }
  }

  // Same warning/ordering as FAST-Calib:
  // Marker 0 -> ArUco ID 1
  // Marker 1 -> ArUco ID 2
  // Marker 2 -> ArUco ID 4
  // Marker 3 -> ArUco ID 3
  board_ids = {1, 2, 4, 3};
}

std::vector<ImagePair> findFlatImagePairs(const std::string& data_dir) {
  std::vector<ImagePair> pairs;

  const std::regex cam0_regex(R"((pair_[0-9]+)_cam0\.(jpg|jpeg|png|bmp))",
                              std::regex::icase);

  for (const auto& name : listFiles(data_dir)) {
    std::smatch match;
    if (!std::regex_match(name, match, cam0_regex)) continue;

    const std::string pair_id = match[1].str();
    const std::string ext = match[2].str();

    const std::string cam0_path = joinPath(data_dir, name);
    const std::string cam1_name = pair_id + "_cam1." + ext;
    const std::string cam1_path = joinPath(data_dir, cam1_name);

    if (!pathExists(cam1_path)) {
      ROS_WARN_STREAM("Skip " << pair_id << ": missing " << cam1_name);
      continue;
    }

    ImagePair p;
    p.pair_id = pair_id;
    p.cam0_path = cam0_path;
    p.cam1_path = cam1_path;
    pairs.push_back(p);
  }

  std::sort(pairs.begin(), pairs.end(),
            [](const ImagePair& a, const ImagePair& b) {
              return a.pair_id < b.pair_id;
            });

  return pairs;
}

Eigen::Matrix4d makeTransformFromRvecTvec(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
  cv::Mat Rcv;
  cv::Rodrigues(rvec, Rcv);

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      T(r, c) = Rcv.at<double>(r, c);
    }
  }
  T(0, 3) = tvec[0];
  T(1, 3) = tvec[1];
  T(2, 3) = tvec[2];
  return T;
}

cv::Vec3d rvecFromMatrix(const Eigen::Matrix3d& R) {
  cv::Mat Rcv(3, 3, CV_64F);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      Rcv.at<double>(r, c) = R(r, c);
    }
  }
  cv::Mat rvec;
  cv::Rodrigues(Rcv, rvec);
  return cv::Vec3d(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2));
}

double computeBoardReprojectionRmse(
    const std::vector<std::vector<cv::Point2f>>& detected_corners,
    const std::vector<int>& detected_ids,
    const std::vector<std::vector<cv::Point3f>>& board_corners,
    const std::vector<int>& board_ids,
    const cv::Vec3d& rvec,
    const cv::Vec3d& tvec,
    const cv::Mat& K,
    const cv::Mat& D) {
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;

  for (size_t i = 0; i < detected_ids.size(); ++i) {
    auto it = std::find(board_ids.begin(), board_ids.end(), detected_ids[i]);
    if (it == board_ids.end()) continue;

    const size_t board_idx = std::distance(board_ids.begin(), it);
    for (int k = 0; k < 4; ++k) {
      object_points.push_back(board_corners[board_idx][k]);
      image_points.push_back(detected_corners[i][k]);
    }
  }

  if (object_points.empty() || object_points.size() != image_points.size()) {
    return -1.0;
  }

  std::vector<cv::Point2f> projected;
  cv::projectPoints(object_points, rvec, tvec, K, D, projected);

  double sum2 = 0.0;
  for (size_t i = 0; i < projected.size(); ++i) {
    const double dx = projected[i].x - image_points[i].x;
    const double dy = projected[i].y - image_points[i].y;
    sum2 += dx * dx + dy * dy;
  }

  return std::sqrt(sum2 / static_cast<double>(projected.size()));
}

bool buildBoardCorrespondences(
    const std::vector<std::vector<cv::Point2f>>& detected_corners,
    const std::vector<int>& detected_ids,
    const std::vector<std::vector<cv::Point3f>>& board_corners,
    const std::vector<int>& board_ids,
    std::vector<cv::Point3f>& object_points,
    std::vector<cv::Point2f>& image_points) {
  object_points.clear();
  image_points.clear();

  for (size_t i = 0; i < detected_ids.size(); ++i) {
    auto it = std::find(board_ids.begin(), board_ids.end(), detected_ids[i]);
    if (it == board_ids.end()) continue;
    const size_t board_idx = std::distance(board_ids.begin(), it);
    if (detected_corners[i].size() != 4 || board_corners[board_idx].size() != 4) continue;

    for (int k = 0; k < 4; ++k) {
      object_points.push_back(board_corners[board_idx][k]);
      image_points.push_back(detected_corners[i][k]);
    }
  }

  return object_points.size() >= 4 && object_points.size() == image_points.size();
}

PoseResult detectBoardPose(
    const cv::Mat& image_bgr,
    const CameraConfig& cam,
    const TargetConfig& target,
    const RuntimeConfig& runtime,
    const cv::Ptr<cv::aruco::Dictionary>& dictionary,
    const cv::Ptr<cv::aruco::Board>& board,
    const std::vector<std::vector<cv::Point3f>>& board_corners,
    const std::vector<int>& board_ids) {
  PoseResult result;
  image_bgr.copyTo(result.debug_image);

  cv::Ptr<cv::aruco::DetectorParameters> detector_params =
      cv::aruco::DetectorParameters::create();

#if (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2) || CV_MAJOR_VERSION < 3
  detector_params->doCornerRefinement = true;
#else
  detector_params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
#endif

  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<std::vector<cv::Point2f>> rejected;

  cv::aruco::detectMarkers(image_bgr, dictionary, corners, ids, detector_params, rejected);

  // For portability across ROS OpenCV versions, we do not force refineDetectedMarkers here.
  // Some OpenCV 4.x builds expose different overloads. The direct detection is enough when
  // all 4 markers are visible, which is the intended capture condition.
  (void)runtime;
  (void)target;

  if (!ids.empty()) {
    cv::aruco::drawDetectedMarkers(result.debug_image, corners, ids);
  }

  result.ids = ids;
  result.markers = static_cast<int>(ids.size());

  if (static_cast<int>(ids.size()) < runtime.min_detected_markers) {
    return result;
  }

  cv::Mat K = cam.K();
  cv::Mat D = cam.D();

  cv::Vec3d rvec(0, 0, 0);
  cv::Vec3d tvec(0, 0, 0);
  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  if (!buildBoardCorrespondences(corners, ids, board_corners, board_ids,
                                 object_points, image_points)) {
    return result;
  }

  const bool pnp_ok = cv::solvePnP(object_points,
                                   image_points,
                                   K,
                                   D,
                                   rvec,
                                   tvec,
                                   false,
                                   cv::SOLVEPNP_ITERATIVE);

  if (!pnp_ok) {
    return result;
  }

  result.reproj_rmse_px = computeBoardReprojectionRmse(
      corners, ids, board_corners, board_ids, rvec, tvec, K, D);

  result.T_cam_board = makeTransformFromRvecTvec(rvec, tvec);
  result.ok = result.reproj_rmse_px >= 0.0 && result.reproj_rmse_px <= runtime.reproj_rmse_thresh_px;

  try {
    cv::aruco::drawAxis(result.debug_image, K, D, rvec, tvec, 0.2);
  } catch (...) {
    // drawAxis may not exist or may throw in some OpenCV builds. Ignore visualization failure.
  }

  cv::Scalar color = result.ok ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
  std::ostringstream oss;
  oss << cam.name << " markers=" << result.markers << " rmse=" << std::fixed << std::setprecision(3)
      << result.reproj_rmse_px << " " << (result.ok ? "OK" : "BAD");
  cv::putText(result.debug_image, oss.str(), cv::Point(20, 40),
              cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv::LINE_AA);

  return result;
}

double rotationAngleDeg(const Eigen::Matrix3d& Ra, const Eigen::Matrix3d& Rb) {
  Eigen::Matrix3d dR = Ra.transpose() * Rb;
  double c = (dR.trace() - 1.0) * 0.5;
  c = std::max(-1.0, std::min(1.0, c));
  return rad2deg(std::acos(c));
}

Eigen::Quaterniond averageQuaternions(const std::vector<Eigen::Quaterniond>& qs,
                                      const std::vector<double>& weights) {
  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();

  Eigen::Quaterniond q_ref = qs.front();
  for (size_t i = 0; i < qs.size(); ++i) {
    Eigen::Quaterniond q = qs[i].normalized();
    if (q_ref.dot(q) < 0.0) {
      q.coeffs() *= -1.0;
    }

    Eigen::Vector4d v;
    v << q.w(), q.x(), q.y(), q.z();
    A += weights[i] * (v * v.transpose());
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(A);
  Eigen::Vector4d v = solver.eigenvectors().col(3);

  Eigen::Quaterniond q(v(0), v(1), v(2), v(3));
  q.normalize();
  return q;
}

Eigen::Matrix4d averageTransforms(const std::vector<PairResult>& pairs,
                                  const std::vector<int>& indices) {
  if (indices.empty()) {
    throw std::runtime_error("averageTransforms got empty indices.");
  }

  std::vector<Eigen::Quaterniond> qs;
  std::vector<double> weights;

  Eigen::Vector3d t_sum = Eigen::Vector3d::Zero();
  double w_sum = 0.0;

  for (const int idx : indices) {
    const Eigen::Matrix4d& T = pairs[idx].T_cam1_cam0;
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);

    // Equal weights. This is intentionally simple and stable.
    double w = 1.0;

    qs.emplace_back(R);
    weights.push_back(w);
    t_sum += w * t;
    w_sum += w;
  }

  Eigen::Quaterniond q_avg = averageQuaternions(qs, weights);
  Eigen::Vector3d t_avg = t_sum / w_sum;

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = q_avg.toRotationMatrix();
  T.block<3, 1>(0, 3) = t_avg;
  return T;
}

Eigen::Vector3d rpyFromR(const Eigen::Matrix3d& R) {
  // Roll-pitch-yaw for R = Rz(yaw) * Ry(pitch) * Rx(roll).
  double roll = std::atan2(R(2, 1), R(2, 2));
  double pitch = std::asin(-R(2, 0));
  double yaw = std::atan2(R(1, 0), R(0, 0));
  return Eigen::Vector3d(roll, pitch, yaw);
}

void writeMatrixYaml(std::ofstream& ofs, const std::string& name, const Eigen::Matrix4d& T) {
  ofs << name << ":\n";
  for (int r = 0; r < 4; ++r) {
    ofs << "  - [";
    for (int c = 0; c < 4; ++c) {
      ofs << std::fixed << std::setprecision(10) << T(r, c);
      if (c != 3) ofs << ", ";
    }
    ofs << "]\n";
  }
}

void saveResultYaml(const AppConfig& cfg,
                    const Eigen::Matrix4d& T_cam1_cam0,
                    const std::vector<PairResult>& pair_results,
                    const std::vector<int>& used_indices,
                    const std::string& path) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    throw std::runtime_error("Cannot write result yaml: " + path);
  }

  Eigen::Matrix4d T_cam0_cam1 = T_cam1_cam0.inverse();
  Eigen::Vector3d t = T_cam1_cam0.block<3, 1>(0, 3);
  Eigen::Vector3d rpy = rpyFromR(T_cam1_cam0.block<3, 3>(0, 0));

  ofs << "pair_name: \"" << cfg.pair_name << "\"\n";
  ofs << "convention: \"X_cam1 = T_cam1_cam0 * X_cam0\"\n";
  ofs << "method: \"aruco_corners_solvepnp_iterative_no_initial_guess\"\n";
  ofs << "cam0_name: \"" << cfg.cam0.name << "\"\n";
  ofs << "cam1_name: \"" << cfg.cam1.name << "\"\n";
  ofs << "used_pairs: " << used_indices.size() << "\n";
  ofs << "total_hard_valid_pairs: " << pair_results.size() << "\n";
  ofs << "\n";

  writeMatrixYaml(ofs, "T_cam1_cam0", T_cam1_cam0);
  ofs << "\n";
  writeMatrixYaml(ofs, "T_cam0_cam1", T_cam0_cam1);
  ofs << "\n";

  ofs << "translation_xyz_m: ["
      << std::fixed << std::setprecision(10) << t.x() << ", "
      << t.y() << ", " << t.z() << "]\n";
  ofs << "translation_norm_m: " << std::fixed << std::setprecision(10) << t.norm() << "\n";
  ofs << "rpy_deg: ["
      << std::fixed << std::setprecision(10)
      << rad2deg(rpy.x()) << ", " << rad2deg(rpy.y()) << ", " << rad2deg(rpy.z()) << "]\n";

  ofs << "\nused_pair_ids:\n";
  for (int idx : used_indices) {
    ofs << "  - \"" << pair_results[idx].pair_id << "\"\n";
  }

  ofs.close();
}

void saveMetricsCsv(const std::vector<PairResult>& results, const std::string& path) {
  std::ofstream ofs(path);
  if (!ofs.is_open()) {
    throw std::runtime_error("Cannot write metrics csv: " + path);
  }

  ofs << "pair_id,ok0,ok1,hard_valid,inlier,markers0,markers1,reproj0_px,reproj1_px,"
         "rot_err_deg,trans_err_m,tx,ty,tz\n";

  for (const auto& r : results) {
    ofs << r.pair_id << ","
        << r.ok0 << ","
        << r.ok1 << ","
        << r.hard_valid << ","
        << r.inlier << ","
        << r.markers0 << ","
        << r.markers1 << ","
        << r.reproj0 << ","
        << r.reproj1 << ","
        << r.rot_err_deg << ","
        << r.trans_err_m << ","
        << r.T_cam1_cam0(0, 3) << ","
        << r.T_cam1_cam0(1, 3) << ","
        << r.T_cam1_cam0(2, 3) << "\n";
  }

  ofs.close();
}

void printTransformToRos(const Eigen::Matrix4d& T) {
  std::ostringstream oss;
  oss << "\n";
  oss << std::fixed << std::setprecision(10);
  for (int r = 0; r < 4; ++r) {
    oss << "  ";
    for (int c = 0; c < 4; ++c) {
      oss << std::setw(15) << T(r, c);
    }
    if (r != 3) oss << "\n";
  }
  ROS_INFO_STREAM("T_cam1_cam0:" << oss.str());

  Eigen::Vector3d t = T.block<3, 1>(0, 3);
  Eigen::Vector3d rpy = rpyFromR(T.block<3, 3>(0, 0));
  ROS_INFO_STREAM("translation_xyz_m = [" << t.x() << ", " << t.y() << ", " << t.z() << "]");
  ROS_INFO_STREAM("translation_norm_m = " << t.norm());
  ROS_INFO_STREAM("rpy_deg = [" << rad2deg(rpy.x()) << ", " << rad2deg(rpy.y()) << ", " << rad2deg(rpy.z()) << "]");
}

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "two_camera_calib");
  ros::NodeHandle nh("~");

  try {
    std::string config_path;
    nh.param<std::string>("config", config_path, "");

    if (config_path.empty() && argc >= 2) {
      config_path = argv[1];
    }

    AppConfig cfg;
    if (nh.hasParam("pair_name")) {
      cfg = readConfigFromRosParams(nh);
    } else if (!config_path.empty()) {
      try {
        cfg = readConfigFromYaml(config_path);
      } catch (const std::exception& e) {
        ROS_ERROR_STREAM("OpenCV FileStorage could not read config file: " << e.what());
        return 1;
      }
    } else {
      ROS_ERROR("Missing config. Load ROS params or pass an OpenCV YAML path.");
      return 1;
    }

    makeDirRecursive(cfg.output_dir);
    const std::string vis_dir = joinPath(cfg.output_dir, "vis");
    if (cfg.runtime.save_debug) {
      makeDirRecursive(vis_dir);
    }

    ROS_INFO_STREAM("Two-camera calibration for pair: " << cfg.pair_name);
    ROS_INFO_STREAM("config: " << config_path);
    ROS_INFO_STREAM("data_dir: " << cfg.data_dir);
    ROS_INFO_STREAM("output_dir: " << cfg.output_dir);
    ROS_INFO_STREAM("cam0: " << cfg.cam0.name);
    ROS_INFO_STREAM("cam1: " << cfg.cam1.name);
    ROS_INFO("Convention: X_cam1 = T_cam1_cam0 * X_cam0");

    std::vector<ImagePair> image_pairs = findFlatImagePairs(cfg.data_dir);
    ROS_INFO_STREAM("Found " << image_pairs.size() << " image pairs.");

    if (image_pairs.empty()) {
      ROS_ERROR_STREAM("No image pairs found in: " << cfg.data_dir);
      return 1;
    }

    std::vector<std::vector<cv::Point3f>> board_corners;
    std::vector<int> board_ids;
    buildFastCalibBoard(cfg.target, board_corners, board_ids);

    cv::Ptr<cv::aruco::Dictionary> dictionary =
        cv::aruco::getPredefinedDictionary(arucoDictionaryId(cfg.target.dictionary));
    cv::Ptr<cv::aruco::Board> board =
        cv::aruco::Board::create(board_corners, dictionary, board_ids);

    std::vector<PairResult> hard_valid_results;

    for (const auto& p : image_pairs) {
      cv::Mat img0 = cv::imread(p.cam0_path, cv::IMREAD_COLOR);
      cv::Mat img1 = cv::imread(p.cam1_path, cv::IMREAD_COLOR);

      if (img0.empty() || img1.empty()) {
        ROS_WARN_STREAM("Skip " << p.pair_id << ": image read failed.");
        continue;
      }

      PoseResult pose0 = detectBoardPose(img0, cfg.cam0, cfg.target, cfg.runtime,
                                         dictionary, board, board_corners, board_ids);
      PoseResult pose1 = detectBoardPose(img1, cfg.cam1, cfg.target, cfg.runtime,
                                         dictionary, board, board_corners, board_ids);

      PairResult pr;
      pr.pair_id = p.pair_id;
      pr.ok0 = pose0.ok;
      pr.ok1 = pose1.ok;
      pr.markers0 = pose0.markers;
      pr.markers1 = pose1.markers;
      pr.reproj0 = pose0.reproj_rmse_px;
      pr.reproj1 = pose1.reproj_rmse_px;
      pr.hard_valid = pose0.ok && pose1.ok;

      if (pr.hard_valid) {
        // T_cam_board maps board coordinates to camera coordinates.
        // Therefore:
        //   X_cam0 = T_cam0_board * X_board
        //   X_cam1 = T_cam1_board * X_board
        //   X_cam1 = T_cam1_board * inv(T_cam0_board) * X_cam0
        pr.T_cam1_cam0 = pose1.T_cam_board * pose0.T_cam_board.inverse();
        hard_valid_results.push_back(pr);
      }

      ROS_INFO_STREAM("pair=" << p.pair_id
                      << " markers=(" << pose0.markers << "," << pose1.markers << ")"
                      << " reproj=(" << pose0.reproj_rmse_px << "," << pose1.reproj_rmse_px << ")"
                      << " hard_valid=" << pr.hard_valid);

      if (cfg.runtime.save_debug) {
        cv::imwrite(joinPath(vis_dir, p.pair_id + "_cam0_detect.jpg"), pose0.debug_image);
        cv::imwrite(joinPath(vis_dir, p.pair_id + "_cam1_detect.jpg"), pose1.debug_image);
      }
    }

    if (hard_valid_results.empty()) {
      ROS_ERROR("No hard-valid pairs. Check image data, intrinsics, dictionary, marker size and board layout.");
      return 1;
    }

    std::vector<int> all_indices;
    for (int i = 0; i < static_cast<int>(hard_valid_results.size()); ++i) {
      all_indices.push_back(i);
    }

    // First estimate from all hard-valid pairs.
    Eigen::Matrix4d T_initial = averageTransforms(hard_valid_results, all_indices);

    std::vector<int> inlier_indices;
    for (int i = 0; i < static_cast<int>(hard_valid_results.size()); ++i) {
      PairResult& r = hard_valid_results[i];

      r.rot_err_deg = rotationAngleDeg(T_initial.block<3, 3>(0, 0),
                                       r.T_cam1_cam0.block<3, 3>(0, 0));
      r.trans_err_m = (r.T_cam1_cam0.block<3, 1>(0, 3) -
                       T_initial.block<3, 1>(0, 3)).norm();

      r.inlier = (r.rot_err_deg <= cfg.runtime.outlier_rot_thresh_deg &&
                  r.trans_err_m <= cfg.runtime.outlier_trans_thresh_m);

      if (r.inlier) {
        inlier_indices.push_back(i);
      }
    }

    std::vector<int> used_indices;
    if (static_cast<int>(inlier_indices.size()) >= cfg.runtime.min_valid_pairs) {
      used_indices = inlier_indices;
    } else {
      double max_rot_err = 0.0;
      double max_trans_err = 0.0;
      for (int idx : all_indices) {
        max_rot_err = std::max(max_rot_err, hard_valid_results[idx].rot_err_deg);
        max_trans_err = std::max(max_trans_err, hard_valid_results[idx].trans_err_m);
      }

      const double fallback_rot_guard =
          std::max(3.0, 5.0 * cfg.runtime.outlier_rot_thresh_deg);
      const double fallback_trans_guard =
          std::max(0.5, 5.0 * cfg.runtime.outlier_trans_thresh_m);
      if (max_rot_err > fallback_rot_guard || max_trans_err > fallback_trans_guard) {
        saveMetricsCsv(hard_valid_results, joinPath(cfg.output_dir, "per_pair_metrics.csv"));
        ROS_ERROR_STREAM("Only " << inlier_indices.size()
                         << " inlier pairs after outlier rejection, less than min_valid_pairs="
                         << cfg.runtime.min_valid_pairs
                         << ". Hard-valid pairs are mutually inconsistent"
                         << " (max_rot_err_deg=" << max_rot_err
                         << ", max_trans_err_m=" << max_trans_err
                         << "), so no averaged result is written.");
        return 3;
      }

      ROS_WARN_STREAM("Only " << inlier_indices.size()
                      << " inlier pairs after outlier rejection, less than min_valid_pairs="
                      << cfg.runtime.min_valid_pairs
                      << ". Use all hard-valid candidates because they are still within the consistency guard"
                      << " (max_rot_err_deg=" << max_rot_err
                      << ", max_trans_err_m=" << max_trans_err
                      << "). Collect more data for final delivery.");
      used_indices = all_indices;

      for (int idx : used_indices) {
        hard_valid_results[idx].inlier = true;
      }
    }

    Eigen::Matrix4d T_final = averageTransforms(hard_valid_results, used_indices);

    // Recompute deviations to final transform for clearer metrics.
    for (auto& r : hard_valid_results) {
      r.rot_err_deg = rotationAngleDeg(T_final.block<3, 3>(0, 0),
                                       r.T_cam1_cam0.block<3, 3>(0, 0));
      r.trans_err_m = (r.T_cam1_cam0.block<3, 1>(0, 3) -
                       T_final.block<3, 1>(0, 3)).norm();
    }

    ROS_INFO("==== Two-camera calibration result ====");
    ROS_INFO("Convention: X_cam1 = T_cam1_cam0 * X_cam0");
    printTransformToRos(T_final);
    ROS_INFO_STREAM("used_pairs = " << used_indices.size() << " / " << hard_valid_results.size());

    const std::string result_path = joinPath(cfg.output_dir, "result.yaml");
    const std::string metrics_path = joinPath(cfg.output_dir, "per_pair_metrics.csv");

    saveResultYaml(cfg, T_final, hard_valid_results, used_indices, result_path);
    saveMetricsCsv(hard_valid_results, metrics_path);

    ROS_INFO_STREAM("Saved result to: " << result_path);
    ROS_INFO_STREAM("Saved metrics to: " << metrics_path);

    return 0;
  } catch (const std::exception& e) {
    ROS_ERROR_STREAM("two_camera_calib failed: " << e.what());
    return 1;
  }
}
