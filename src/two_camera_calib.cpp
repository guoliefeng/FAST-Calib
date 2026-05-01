/*
 * Two-camera extrinsic calibration using the FAST-Calib ArUco target.
 *
 * Coordinate convention:
 *   X_cam1 = T_cam1_cam0 * X_cam0
 *
 * Input data convention:
 *   data_dir/
 *     pair_0001_cam0.jpg
 *     pair_0001_cam1.jpg
 *     pair_0002_cam0.jpg
 *     pair_0002_cam1.jpg
 *     ...
 */

#include <ros/ros.h>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace fast_calib_two_camera {

constexpr double kRad2Deg = 180.0 / M_PI;
constexpr double kDeg2Rad = M_PI / 180.0;

struct CameraModel {
  std::string name;
  cv::Mat K;
  cv::Mat D;
};

struct TargetModel {
  std::string dictionary = "DICT_6X6_250";
  double marker_size_m = 0.20;
  double delta_width_qr_center_m = 0.55;
  double delta_height_qr_center_m = 0.35;
  double delta_width_circles_m = 0.50;
  double delta_height_circles_m = 0.40;
};

struct RuntimeConfig {
  int min_detected_markers = 3;
  bool refine_markers = true;
  double reproj_rmse_thresh_px = 1.5;
  double outlier_rot_thresh_deg = 0.5;
  double outlier_trans_thresh_m = 0.01;
  int min_valid_pairs = 5;
};

struct AppConfig {
  std::string pair_name;
  std::string data_dir;
  std::string output_dir;
  CameraModel cam0;
  CameraModel cam1;
  TargetModel target;
  RuntimeConfig runtime;
};

struct ImagePair {
  std::string pair_id;
  std::string cam0_path;
  std::string cam1_path;
};

struct BoardGeometry {
  cv::Ptr<cv::aruco::Dictionary> dictionary;
  cv::Ptr<cv::aruco::Board> board;
  std::vector<int> board_ids;
  std::vector<std::vector<cv::Point3f>> board_corners;
  std::vector<cv::Point3f> circle_centers_board;
};

struct BoardDetectionResult {
  bool ok = false;
  int used_markers = 0;
  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  cv::Vec3d rvec = cv::Vec3d(0, 0, 0);
  cv::Vec3d tvec = cv::Vec3d(0, 0, 0);
  Eigen::Matrix4d T_cam_board = Eigen::Matrix4d::Identity();
  std::vector<Eigen::Vector3d> circle_centers_cam;
  double reproj_rmse_px = -1.0;
  cv::Mat vis;
};

struct PairMetric {
  std::string pair_id;
  bool detect_ok_cam0 = false;
  bool detect_ok_cam1 = false;
  bool hard_valid = false;
  bool accepted = false;
  bool outlier = false;
  int markers_cam0 = 0;
  int markers_cam1 = 0;
  double reproj_cam0_px = -1.0;
  double reproj_cam1_px = -1.0;
  double rot_err_deg = -1.0;
  double trans_err_m = -1.0;
  Eigen::Matrix4d T_cam1_cam0 = Eigen::Matrix4d::Identity();
  std::vector<Eigen::Vector3d> circle_centers_cam0;
  std::vector<Eigen::Vector3d> circle_centers_cam1;
};

template <typename T>
bool GetRequiredParam(const ros::NodeHandle& nh, const std::string& key, T& value) {
  if (!nh.getParam(key, value)) {
    ROS_ERROR_STREAM("Missing required param: " << nh.getNamespace() << "/" << key);
    return false;
  }
  return true;
}

template <typename T>
void GetOptionalParam(const ros::NodeHandle& nh, const std::string& key, T& value) {
  nh.param(key, value, value);
}

bool LoadCameraModel(const ros::NodeHandle& nh, const std::string& prefix, CameraModel& cam) {
  double fx = 0, fy = 0, cx = 0, cy = 0;
  double k1 = 0, k2 = 0, p1 = 0, p2 = 0;
  if (!GetRequiredParam(nh, prefix + "/name", cam.name) ||
      !GetRequiredParam(nh, prefix + "/fx", fx) ||
      !GetRequiredParam(nh, prefix + "/fy", fy) ||
      !GetRequiredParam(nh, prefix + "/cx", cx) ||
      !GetRequiredParam(nh, prefix + "/cy", cy) ||
      !GetRequiredParam(nh, prefix + "/k1", k1) ||
      !GetRequiredParam(nh, prefix + "/k2", k2) ||
      !GetRequiredParam(nh, prefix + "/p1", p1) ||
      !GetRequiredParam(nh, prefix + "/p2", p2)) {
    return false;
  }

  cam.K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx,
                                      0.0, fy, cy,
                                      0.0, 0.0, 1.0);
  cam.D = (cv::Mat_<double>(1, 5) << k1, k2, p1, p2, 0.0);
  return true;
}

bool LoadConfig(const ros::NodeHandle& nh, AppConfig& cfg) {
  if (!GetRequiredParam(nh, "pair_name", cfg.pair_name) ||
      !GetRequiredParam(nh, "data_dir", cfg.data_dir) ||
      !GetRequiredParam(nh, "output_dir", cfg.output_dir)) {
    return false;
  }
  if (!LoadCameraModel(nh, "cam0", cfg.cam0) || !LoadCameraModel(nh, "cam1", cfg.cam1)) {
    return false;
  }

  GetOptionalParam(nh, "target/dictionary", cfg.target.dictionary);
  GetOptionalParam(nh, "target/marker_size_m", cfg.target.marker_size_m);
  GetOptionalParam(nh, "target/delta_width_qr_center_m", cfg.target.delta_width_qr_center_m);
  GetOptionalParam(nh, "target/delta_height_qr_center_m", cfg.target.delta_height_qr_center_m);
  GetOptionalParam(nh, "target/delta_width_circles_m", cfg.target.delta_width_circles_m);
  GetOptionalParam(nh, "target/delta_height_circles_m", cfg.target.delta_height_circles_m);

  GetOptionalParam(nh, "runtime/min_detected_markers", cfg.runtime.min_detected_markers);
  GetOptionalParam(nh, "runtime/refine_markers", cfg.runtime.refine_markers);
  GetOptionalParam(nh, "runtime/reproj_rmse_thresh_px", cfg.runtime.reproj_rmse_thresh_px);
  GetOptionalParam(nh, "runtime/outlier_rot_thresh_deg", cfg.runtime.outlier_rot_thresh_deg);
  GetOptionalParam(nh, "runtime/outlier_trans_thresh_m", cfg.runtime.outlier_trans_thresh_m);
  GetOptionalParam(nh, "runtime/min_valid_pairs", cfg.runtime.min_valid_pairs);

  return true;
}

bool IsDir(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool MakeDirIfNeeded(const std::string& path) {
  if (path.empty()) return false;
  if (IsDir(path)) return true;
  if (mkdir(path.c_str(), 0755) == 0) return true;
  return IsDir(path);
}

bool MakeDirs(const std::string& path) {
  if (path.empty()) return false;
  std::string current;
  if (path[0] == '/') current = "/";
  std::stringstream ss(path);
  std::string item;
  while (std::getline(ss, item, '/')) {
    if (item.empty()) continue;
    if (!current.empty() && current.back() != '/') current += "/";
    current += item;
    if (!MakeDirIfNeeded(current)) return false;
  }
  return true;
}

std::string JoinPath(const std::string& a, const std::string& b) {
  if (a.empty()) return b;
  if (a.back() == '/') return a + b;
  return a + "/" + b;
}

std::string BaseName(const std::string& path) {
  const size_t pos = path.find_last_of('/');
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

bool HasImageExtension(const std::string& name) {
  const size_t pos = name.find_last_of('.');
  if (pos == std::string::npos) return false;
  std::string ext = name.substr(pos + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp";
}

bool FileExists(const std::string& path) {
  struct stat st;
  return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::vector<ImagePair> CollectImagePairs(const std::string& data_dir) {
  std::vector<ImagePair> pairs;
  DIR* dir = opendir(data_dir.c_str());
  if (!dir) {
    ROS_ERROR_STREAM("Cannot open data_dir: " << data_dir);
    return pairs;
  }

  std::set<std::string> files;
  while (dirent* entry = readdir(dir)) {
    std::string name(entry->d_name);
    if (name == "." || name == "..") continue;
    if (HasImageExtension(name)) files.insert(name);
  }
  closedir(dir);

  const std::string tag = "_cam0";
  for (const auto& name : files) {
    const size_t dot = name.find_last_of('.');
    if (dot == std::string::npos) continue;
    const std::string stem = name.substr(0, dot);
    const std::string ext = name.substr(dot);
    if (stem.size() <= tag.size()) continue;
    if (stem.substr(stem.size() - tag.size()) != tag) continue;

    const std::string pair_id = stem.substr(0, stem.size() - tag.size());
    const std::string cam1_name = pair_id + "_cam1" + ext;
    if (files.count(cam1_name) == 0) {
      ROS_WARN_STREAM("Cannot find cam1 image for pair_id=" << pair_id << ", skip");
      continue;
    }
    ImagePair pair;
    pair.pair_id = pair_id;
    pair.cam0_path = JoinPath(data_dir, name);
    pair.cam1_path = JoinPath(data_dir, cam1_name);
    pairs.push_back(pair);
  }

  std::sort(pairs.begin(), pairs.end(), [](const ImagePair& a, const ImagePair& b) {
    return a.pair_id < b.pair_id;
  });
  return pairs;
}

cv::Ptr<cv::aruco::Dictionary> CreateDictionary(const std::string& name) {
  if (name == "DICT_6X6_250") {
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  }
  throw std::runtime_error("Unsupported ArUco dictionary: " + name +
                           ". Current FAST-Calib target uses DICT_6X6_250.");
}

BoardGeometry BuildBoardGeometry(const TargetModel& target) {
  BoardGeometry geom;
  geom.dictionary = CreateDictionary(target.dictionary);
  geom.board_ids = {1, 2, 4, 3};
  geom.board_corners.resize(4);

  const double width = target.delta_width_qr_center_m;
  const double height = target.delta_height_qr_center_m;
  const double circle_width = target.delta_width_circles_m / 2.0;
  const double circle_height = target.delta_height_circles_m / 2.0;

  for (int i = 0; i < 4; ++i) {
    const int x_qr_center = (i % 3) == 0 ? -1 : 1;
    const int y_qr_center = (i < 2) ? 1 : -1;
    const double x_center = x_qr_center * width;
    const double y_center = y_qr_center * height;

    geom.circle_centers_board.push_back(
        cv::Point3f(static_cast<float>(x_qr_center * circle_width),
                    static_cast<float>(y_qr_center * circle_height), 0.0f));

    for (int j = 0; j < 4; ++j) {
      const int x_qr = (j % 3) == 0 ? -1 : 1;
      const int y_qr = (j < 2) ? 1 : -1;
      geom.board_corners[i].push_back(
          cv::Point3f(static_cast<float>(x_center + x_qr * target.marker_size_m / 2.0),
                      static_cast<float>(y_center + y_qr * target.marker_size_m / 2.0),
                      0.0f));
    }
  }

  geom.board = cv::aruco::Board::create(geom.board_corners, geom.dictionary, geom.board_ids);
  return geom;
}

Eigen::Matrix4d RtToEigen44(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);
  R_cv.convertTo(R_cv, CV_64F);

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      T(r, c) = R_cv.at<double>(r, c);
    }
  }
  T(0, 3) = tvec[0];
  T(1, 3) = tvec[1];
  T(2, 3) = tvec[2];
  return T;
}

double ComputeReprojectionRmse(const BoardGeometry& geom,
                               const std::vector<int>& ids,
                               const std::vector<std::vector<cv::Point2f>>& corners,
                               const cv::Vec3d& rvec,
                               const cv::Vec3d& tvec,
                               const CameraModel& cam) {
  if (ids.empty()) return -1.0;

  double sum_sq = 0.0;
  int n = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    auto it = std::find(geom.board_ids.begin(), geom.board_ids.end(), ids[i]);
    if (it == geom.board_ids.end()) continue;
    const int board_idx = static_cast<int>(std::distance(geom.board_ids.begin(), it));

    std::vector<cv::Point2f> projected;
    cv::projectPoints(geom.board_corners[board_idx], rvec, tvec, cam.K, cam.D, projected);
    for (size_t j = 0; j < projected.size() && j < corners[i].size(); ++j) {
      const double dx = projected[j].x - corners[i][j].x;
      const double dy = projected[j].y - corners[i][j].y;
      sum_sq += dx * dx + dy * dy;
      ++n;
    }
  }
  if (n == 0) return -1.0;
  return std::sqrt(sum_sq / static_cast<double>(n));
}

bool BuildCorrespondences(const BoardGeometry& geom,
                          const std::vector<int>& ids,
                          const std::vector<std::vector<cv::Point2f>>& corners,
                          std::vector<cv::Point3f>& object_points,
                          std::vector<cv::Point2f>& image_points) {
  object_points.clear();
  image_points.clear();

  for (size_t i = 0; i < ids.size(); ++i) {
    auto it = std::find(geom.board_ids.begin(), geom.board_ids.end(), ids[i]);
    if (it == geom.board_ids.end()) continue;
    const int board_idx = static_cast<int>(std::distance(geom.board_ids.begin(), it));
    if (corners[i].size() != 4 || geom.board_corners[board_idx].size() != 4) continue;

    for (int j = 0; j < 4; ++j) {
      object_points.push_back(geom.board_corners[board_idx][j]);
      image_points.push_back(corners[i][j]);
    }
  }

  return object_points.size() >= 4 && object_points.size() == image_points.size();
}

std::vector<Eigen::Vector3d> TransformCircleCenters(const BoardGeometry& geom,
                                                    const Eigen::Matrix4d& T_cam_board) {
  std::vector<Eigen::Vector3d> centers;
  centers.reserve(geom.circle_centers_board.size());
  for (const auto& p : geom.circle_centers_board) {
    Eigen::Vector4d pb(p.x, p.y, p.z, 1.0);
    Eigen::Vector4d pc = T_cam_board * pb;
    centers.emplace_back(pc.x(), pc.y(), pc.z());
  }
  return centers;
}

bool DetectBoardPose(const cv::Mat& image,
                     const CameraModel& cam,
                     const TargetModel& target,
                     const BoardGeometry& geom,
                     const RuntimeConfig& runtime,
                     BoardDetectionResult& out) {
  out = BoardDetectionResult();
  if (image.empty()) {
    return false;
  }
  image.copyTo(out.vis);

  cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
#if (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2) || CV_MAJOR_VERSION < 3
  parameters->doCornerRefinement = true;
#else
  parameters->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
#endif

  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<std::vector<cv::Point2f>> rejected;
  cv::aruco::detectMarkers(image, geom.dictionary, corners, ids, parameters, rejected);

#if (CV_MAJOR_VERSION > 3) || (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION >= 3)
  if (runtime.refine_markers && !rejected.empty()) {
    cv::aruco::refineDetectedMarkers(image, geom.board, corners, ids, rejected, cam.K, cam.D);
  }
#endif

  if (!ids.empty()) {
    cv::aruco::drawDetectedMarkers(out.vis, corners, ids);
  }

  out.ids = ids;
  out.corners = corners;
  out.used_markers = static_cast<int>(ids.size());

  if (static_cast<int>(ids.size()) < runtime.min_detected_markers) {
    return false;
  }

  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  if (!BuildCorrespondences(geom, ids, corners, object_points, image_points)) {
    return false;
  }

  cv::Vec3d rvec(0, 0, 0), tvec(0, 0, 0);
  const bool pnp_ok = cv::solvePnP(object_points,
                                   image_points,
                                   cam.K,
                                   cam.D,
                                   rvec,
                                   tvec,
                                   false,
                                   cv::SOLVEPNP_ITERATIVE);
  if (!pnp_ok) {
    return false;
  }

  cv::aruco::drawAxis(out.vis, cam.K, cam.D, rvec, tvec, 0.2);

  out.rvec = rvec;
  out.tvec = tvec;
  out.T_cam_board = RtToEigen44(rvec, tvec);
  out.circle_centers_cam = TransformCircleCenters(geom, out.T_cam_board);
  out.reproj_rmse_px = ComputeReprojectionRmse(geom, ids, corners, rvec, tvec, cam);
  out.ok = true;

  for (const auto& center : out.circle_centers_cam) {
    std::vector<cv::Point3f> obj(1);
    obj[0] = cv::Point3f(static_cast<float>(center.x()),
                         static_cast<float>(center.y()),
                         static_cast<float>(center.z()));
    std::vector<cv::Point2f> uv;
    cv::projectPoints(obj, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), cam.K, cam.D, uv);
    if (!uv.empty()) {
      cv::circle(out.vis, uv[0], 5, cv::Scalar(0, 255, 0), -1);
    }
  }

  return true;
}

Eigen::Matrix4d ComputeRelativePose(const BoardDetectionResult& cam0_det,
                                    const BoardDetectionResult& cam1_det) {
  return cam1_det.T_cam_board * cam0_det.T_cam_board.inverse();
}

Eigen::Quaterniond RotationOf(const Eigen::Matrix4d& T) {
  Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  Eigen::Quaterniond q(R);
  q.normalize();
  return q;
}

Eigen::Vector3d TranslationOf(const Eigen::Matrix4d& T) {
  return T.block<3, 1>(0, 3);
}

double RotationAngleDeg(const Eigen::Matrix3d& R_a, const Eigen::Matrix3d& R_b) {
  Eigen::Matrix3d dR = R_a.transpose() * R_b;
  double c = (dR.trace() - 1.0) / 2.0;
  c = std::max(-1.0, std::min(1.0, c));
  return std::acos(c) * kRad2Deg;
}

Eigen::Matrix4d AverageTransforms(const std::vector<PairMetric>& metrics,
                                  const std::vector<int>& indices) {
  if (indices.empty()) return Eigen::Matrix4d::Identity();

  const Eigen::Quaterniond q_ref = RotationOf(metrics[indices.front()].T_cam1_cam0);
  Eigen::Vector4d q_sum(0, 0, 0, 0);
  Eigen::Vector3d t_sum(0, 0, 0);
  double w_sum = 0.0;

  for (int idx : indices) {
    const PairMetric& m = metrics[idx];
    double weight = 1.0;
    const double e = std::max(1e-6, 0.5 * (m.reproj_cam0_px + m.reproj_cam1_px));
    weight = 1.0 / (e * e);

    Eigen::Quaterniond q = RotationOf(m.T_cam1_cam0);
    if (q.dot(q_ref) < 0.0) q.coeffs() *= -1.0;
    q_sum += weight * q.coeffs();
    t_sum += weight * TranslationOf(m.T_cam1_cam0);
    w_sum += weight;
  }

  Eigen::Quaterniond q_mean;
  q_mean.coeffs() = q_sum / std::max(1e-12, w_sum);
  q_mean.normalize();

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = q_mean.toRotationMatrix();
  T.block<3, 1>(0, 3) = t_sum / std::max(1e-12, w_sum);
  return T;
}

Eigen::Vector3d RotationMatrixToRpyDeg(const Eigen::Matrix3d& R) {
  const double sy = std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
  const bool singular = sy < 1e-9;
  double roll = 0.0, pitch = 0.0, yaw = 0.0;
  if (!singular) {
    roll = std::atan2(R(2, 1), R(2, 2));
    pitch = std::atan2(-R(2, 0), sy);
    yaw = std::atan2(R(1, 0), R(0, 0));
  } else {
    roll = std::atan2(-R(1, 2), R(1, 1));
    pitch = std::atan2(-R(2, 0), sy);
    yaw = 0.0;
  }
  return Eigen::Vector3d(roll * kRad2Deg, pitch * kRad2Deg, yaw * kRad2Deg);
}

double CircleCrossCheckRmse(const std::vector<PairMetric>& metrics,
                            const std::vector<int>& accepted_indices,
                            const Eigen::Matrix4d& T_cam1_cam0) {
  double sum_sq = 0.0;
  int n = 0;
  for (int idx : accepted_indices) {
    const PairMetric& m = metrics[idx];
    if (m.circle_centers_cam0.size() != m.circle_centers_cam1.size()) continue;
    for (size_t i = 0; i < m.circle_centers_cam0.size(); ++i) {
      Eigen::Vector4d p0(m.circle_centers_cam0[i].x(),
                         m.circle_centers_cam0[i].y(),
                         m.circle_centers_cam0[i].z(), 1.0);
      Eigen::Vector4d p1_pred = T_cam1_cam0 * p0;
      const Eigen::Vector3d diff = p1_pred.head<3>() - m.circle_centers_cam1[i];
      sum_sq += diff.squaredNorm();
      ++n;
    }
  }
  if (n == 0) return -1.0;
  return std::sqrt(sum_sq / static_cast<double>(n));
}

double MeanValue(const std::vector<double>& values) {
  if (values.empty()) return -1.0;
  return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double StdValue(const std::vector<double>& values) {
  if (values.size() < 2) return 0.0;
  const double mean = MeanValue(values);
  double sum = 0.0;
  for (double v : values) sum += (v - mean) * (v - mean);
  return std::sqrt(sum / static_cast<double>(values.size() - 1));
}

void WriteMatrixYaml(std::ofstream& ofs, const std::string& name, const Eigen::Matrix4d& T) {
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

void SaveMetricsCsv(const std::string& path, const std::vector<PairMetric>& metrics) {
  std::ofstream ofs(path);
  ofs << "pair_id,detect_ok_cam0,detect_ok_cam1,hard_valid,accepted,outlier,"
      << "markers_cam0,markers_cam1,reproj_cam0_px,reproj_cam1_px,"
      << "rot_err_deg,trans_err_m,tx,ty,tz\n";
  for (const auto& m : metrics) {
    Eigen::Vector3d t = TranslationOf(m.T_cam1_cam0);
    ofs << m.pair_id << ","
        << m.detect_ok_cam0 << "," << m.detect_ok_cam1 << ","
        << m.hard_valid << "," << m.accepted << "," << m.outlier << ","
        << m.markers_cam0 << "," << m.markers_cam1 << ","
        << std::fixed << std::setprecision(6)
        << m.reproj_cam0_px << "," << m.reproj_cam1_px << ","
        << m.rot_err_deg << "," << m.trans_err_m << ","
        << t.x() << "," << t.y() << "," << t.z() << "\n";
  }
}

void SavePairList(const std::string& path, const std::vector<PairMetric>& metrics, bool accepted) {
  std::ofstream ofs(path);
  for (const auto& m : metrics) {
    if (m.accepted == accepted) ofs << m.pair_id << "\n";
  }
}

void SaveResultYaml(const std::string& path,
                    const AppConfig& cfg,
                    const Eigen::Matrix4d& T_cam1_cam0,
                    const std::vector<PairMetric>& metrics,
                    const std::vector<int>& accepted_indices,
                    double cross_rmse_m) {
  std::vector<double> reproj0, reproj1, rot_errs, trans_errs;
  for (int idx : accepted_indices) {
    reproj0.push_back(metrics[idx].reproj_cam0_px);
    reproj1.push_back(metrics[idx].reproj_cam1_px);
    if (metrics[idx].rot_err_deg >= 0) rot_errs.push_back(metrics[idx].rot_err_deg);
    if (metrics[idx].trans_err_m >= 0) trans_errs.push_back(metrics[idx].trans_err_m);
  }

  const Eigen::Matrix4d T_cam0_cam1 = T_cam1_cam0.inverse();
  const Eigen::Vector3d t = TranslationOf(T_cam1_cam0);
  const Eigen::Vector3d rpy = RotationMatrixToRpyDeg(T_cam1_cam0.block<3, 3>(0, 0));

  int total = static_cast<int>(metrics.size());
  int used = static_cast<int>(accepted_indices.size());
  int rejected = total - used;

  std::ofstream ofs(path);
  ofs << "pair_name: " << cfg.pair_name << "\n";
  ofs << "cam0_name: " << cfg.cam0.name << "\n";
  ofs << "cam1_name: " << cfg.cam1.name << "\n";
  ofs << "convention: \"X_cam1 = T_cam1_cam0 * X_cam0\"\n";
  ofs << "method: \"aruco_corners_solvepnp_iterative_no_initial_guess\"\n";
  ofs << "target:\n";
  ofs << "  dictionary: " << cfg.target.dictionary << "\n";
  ofs << "  marker_size_m: " << cfg.target.marker_size_m << "\n";
  ofs << "  delta_width_qr_center_m: " << cfg.target.delta_width_qr_center_m << "\n";
  ofs << "  delta_height_qr_center_m: " << cfg.target.delta_height_qr_center_m << "\n";
  WriteMatrixYaml(ofs, "T_cam1_cam0", T_cam1_cam0);
  WriteMatrixYaml(ofs, "T_cam0_cam1", T_cam0_cam1);
  ofs << "translation_xyz_m: [" << std::fixed << std::setprecision(10)
      << t.x() << ", " << t.y() << ", " << t.z() << "]\n";
  ofs << "translation_norm_m: " << t.norm() << "\n";
  ofs << "rpy_deg: [" << rpy.x() << ", " << rpy.y() << ", " << rpy.z() << "]\n";
  ofs << "rmse:\n";
  ofs << "  cam0_reproj_px_mean: " << MeanValue(reproj0) << "\n";
  ofs << "  cam1_reproj_px_mean: " << MeanValue(reproj1) << "\n";
  ofs << "  crosscheck_circle_rmse_m: " << cross_rmse_m << "\n";
  ofs << "stats:\n";
  ofs << "  total_pairs: " << total << "\n";
  ofs << "  used_pairs: " << used << "\n";
  ofs << "  rejected_pairs: " << rejected << "\n";
  ofs << "  rot_err_deg_std: " << StdValue(rot_errs) << "\n";
  ofs << "  trans_err_m_std: " << StdValue(trans_errs) << "\n";
  ofs << "files:\n";
  ofs << "  metrics_csv: " << JoinPath(cfg.output_dir, "per_pair_metrics.csv") << "\n";
  ofs << "  accepted_pairs: " << JoinPath(cfg.output_dir, "accepted_pairs.txt") << "\n";
  ofs << "  rejected_pairs: " << JoinPath(cfg.output_dir, "rejected_pairs.txt") << "\n";
  ofs << "  vis_dir: " << JoinPath(cfg.output_dir, "vis") << "\n";
}

std::vector<int> CandidateIndices(const std::vector<PairMetric>& metrics) {
  std::vector<int> idx;
  for (size_t i = 0; i < metrics.size(); ++i) {
    if (metrics[i].hard_valid) idx.push_back(static_cast<int>(i));
  }
  return idx;
}

std::vector<int> SelectInliers(std::vector<PairMetric>& metrics,
                               const std::vector<int>& candidates,
                               const RuntimeConfig& runtime,
                               const Eigen::Matrix4d& T_initial) {
  std::vector<int> accepted;
  const Eigen::Matrix3d R_initial = T_initial.block<3, 3>(0, 0);
  const Eigen::Vector3d t_initial = TranslationOf(T_initial);

  for (int idx : candidates) {
    PairMetric& m = metrics[idx];
    m.rot_err_deg = RotationAngleDeg(R_initial, m.T_cam1_cam0.block<3, 3>(0, 0));
    m.trans_err_m = (TranslationOf(m.T_cam1_cam0) - t_initial).norm();

    const bool rot_ok = m.rot_err_deg <= runtime.outlier_rot_thresh_deg;
    const bool trans_ok = m.trans_err_m <= runtime.outlier_trans_thresh_m;
    m.outlier = !(rot_ok && trans_ok);
    m.accepted = !m.outlier;
    if (m.accepted) accepted.push_back(idx);
  }
  return accepted;
}

}  // namespace fast_calib_two_camera

int main(int argc, char** argv) {
  ros::init(argc, argv, "two_camera_calib");
  ros::NodeHandle pnh("~");

  using namespace fast_calib_two_camera;

  AppConfig cfg;
  if (!LoadConfig(pnh, cfg)) {
    ROS_ERROR("Failed to load two-camera calibration config.");
    return 1;
  }

  if (!MakeDirs(cfg.output_dir) || !MakeDirs(JoinPath(cfg.output_dir, "vis"))) {
    ROS_ERROR_STREAM("Failed to create output_dir: " << cfg.output_dir);
    return 1;
  }

  ROS_INFO_STREAM("Two-camera calibration for pair: " << cfg.pair_name);
  ROS_INFO_STREAM("data_dir: " << cfg.data_dir);
  ROS_INFO_STREAM("output_dir: " << cfg.output_dir);

  BoardGeometry geom;
  try {
    geom = BuildBoardGeometry(cfg.target);
  } catch (const std::exception& e) {
    ROS_ERROR_STREAM(e.what());
    return 1;
  }

  std::vector<ImagePair> pairs = CollectImagePairs(cfg.data_dir);
  if (pairs.empty()) {
    ROS_ERROR_STREAM("No synchronized image pairs found in: " << cfg.data_dir
                     << ". Expected names like pair_0001_cam0.jpg and pair_0001_cam1.jpg");
    return 1;
  }
  ROS_INFO_STREAM("Found " << pairs.size() << " image pairs.");

  std::vector<PairMetric> metrics;
  metrics.reserve(pairs.size());

  for (const auto& pair : pairs) {
    PairMetric metric;
    metric.pair_id = pair.pair_id;

    cv::Mat img0 = cv::imread(pair.cam0_path, cv::IMREAD_COLOR);
    cv::Mat img1 = cv::imread(pair.cam1_path, cv::IMREAD_COLOR);
    if (img0.empty() || img1.empty()) {
      ROS_WARN_STREAM("Failed to read images for pair_id=" << pair.pair_id);
      metrics.push_back(metric);
      continue;
    }

    BoardDetectionResult det0, det1;
    metric.detect_ok_cam0 = DetectBoardPose(img0, cfg.cam0, cfg.target, geom, cfg.runtime, det0);
    metric.detect_ok_cam1 = DetectBoardPose(img1, cfg.cam1, cfg.target, geom, cfg.runtime, det1);
    metric.markers_cam0 = det0.used_markers;
    metric.markers_cam1 = det1.used_markers;
    metric.reproj_cam0_px = det0.reproj_rmse_px;
    metric.reproj_cam1_px = det1.reproj_rmse_px;

    const std::string vis0 = JoinPath(JoinPath(cfg.output_dir, "vis"), pair.pair_id + "_cam0_detect.png");
    const std::string vis1 = JoinPath(JoinPath(cfg.output_dir, "vis"), pair.pair_id + "_cam1_detect.png");
    if (!det0.vis.empty()) cv::imwrite(vis0, det0.vis);
    if (!det1.vis.empty()) cv::imwrite(vis1, det1.vis);

    if (metric.detect_ok_cam0 && metric.detect_ok_cam1) {
      metric.T_cam1_cam0 = ComputeRelativePose(det0, det1);
      metric.circle_centers_cam0 = det0.circle_centers_cam;
      metric.circle_centers_cam1 = det1.circle_centers_cam;
      metric.hard_valid =
          det0.reproj_rmse_px >= 0 && det1.reproj_rmse_px >= 0 &&
          det0.reproj_rmse_px <= cfg.runtime.reproj_rmse_thresh_px &&
          det1.reproj_rmse_px <= cfg.runtime.reproj_rmse_thresh_px;
    }

    ROS_INFO_STREAM("pair=" << pair.pair_id
                    << " markers=(" << metric.markers_cam0 << "," << metric.markers_cam1 << ")"
                    << " reproj=(" << metric.reproj_cam0_px << "," << metric.reproj_cam1_px << ")"
                    << " hard_valid=" << metric.hard_valid);
    metrics.push_back(metric);
  }

  std::vector<int> candidates = CandidateIndices(metrics);
  if (candidates.empty()) {
    SaveMetricsCsv(JoinPath(cfg.output_dir, "per_pair_metrics.csv"), metrics);
    ROS_ERROR("No valid image pair passed the hard reprojection threshold. Check images, intrinsics and target size.");
    return 2;
  }

  Eigen::Matrix4d T_initial = AverageTransforms(metrics, candidates);
  std::vector<int> accepted = SelectInliers(metrics, candidates, cfg.runtime, T_initial);
  if (static_cast<int>(accepted.size()) < cfg.runtime.min_valid_pairs) {
    double max_rot_err = 0.0;
    double max_trans_err = 0.0;
    for (int idx : candidates) {
      max_rot_err = std::max(max_rot_err, metrics[idx].rot_err_deg);
      max_trans_err = std::max(max_trans_err, metrics[idx].trans_err_m);
    }

    const double fallback_rot_guard =
        std::max(3.0, 5.0 * cfg.runtime.outlier_rot_thresh_deg);
    const double fallback_trans_guard =
        std::max(0.5, 5.0 * cfg.runtime.outlier_trans_thresh_m);
    if (max_rot_err > fallback_rot_guard || max_trans_err > fallback_trans_guard) {
      SaveMetricsCsv(JoinPath(cfg.output_dir, "per_pair_metrics.csv"), metrics);
      ROS_ERROR_STREAM("Only " << accepted.size()
                       << " inlier pairs after outlier rejection, less than min_valid_pairs="
                       << cfg.runtime.min_valid_pairs
                       << ". Hard-valid pairs are mutually inconsistent"
                       << " (max_rot_err_deg=" << max_rot_err
                       << ", max_trans_err_m=" << max_trans_err
                       << "), so no averaged result is written.");
      return 3;
    }

    ROS_WARN_STREAM("Only " << accepted.size() << " inlier pairs after outlier rejection, "
                    << "less than min_valid_pairs=" << cfg.runtime.min_valid_pairs
                    << ". Use all hard-valid candidates because they are still within the consistency guard"
                    << " (max_rot_err_deg=" << max_rot_err
                    << ", max_trans_err_m=" << max_trans_err
                    << "). Please collect more data for final delivery.");
    accepted = candidates;
    for (int idx : accepted) metrics[idx].accepted = true;
  }

  Eigen::Matrix4d T_final = AverageTransforms(metrics, accepted);
  const double cross_rmse = CircleCrossCheckRmse(metrics, accepted, T_final);

  const std::string result_yaml = JoinPath(cfg.output_dir, "result.yaml");
  const std::string metrics_csv = JoinPath(cfg.output_dir, "per_pair_metrics.csv");
  SaveResultYaml(result_yaml, cfg, T_final, metrics, accepted, cross_rmse);
  SaveMetricsCsv(metrics_csv, metrics);
  SavePairList(JoinPath(cfg.output_dir, "accepted_pairs.txt"), metrics, true);
  SavePairList(JoinPath(cfg.output_dir, "rejected_pairs.txt"), metrics, false);

  const Eigen::Vector3d t = TranslationOf(T_final);
  const Eigen::Vector3d rpy = RotationMatrixToRpyDeg(T_final.block<3, 3>(0, 0));
  ROS_INFO_STREAM("==== Two-camera calibration result ====");
  ROS_INFO_STREAM("Convention: X_cam1 = T_cam1_cam0 * X_cam0");
  ROS_INFO_STREAM("T_cam1_cam0:\n" << T_final);
  ROS_INFO_STREAM("translation_xyz_m = [" << t.x() << ", " << t.y() << ", " << t.z() << "]");
  ROS_INFO_STREAM("translation_norm_m = " << t.norm());
  ROS_INFO_STREAM("rpy_deg = [" << rpy.x() << ", " << rpy.y() << ", " << rpy.z() << "]");
  ROS_INFO_STREAM("used_pairs = " << accepted.size() << " / " << metrics.size());
  ROS_INFO_STREAM("circle_crosscheck_rmse_m = " << cross_rmse);
  ROS_INFO_STREAM("Saved result to: " << result_yaml);

  return 0;
}
