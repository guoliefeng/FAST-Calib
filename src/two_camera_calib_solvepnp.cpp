/*
 * two_camera_calib_solvepnp.cpp
 *
 * Camera-camera extrinsic calibration using the FAST-Calib 4-ArUco target.
 *
 * Main fix compared with estimatePoseBoard-based versions:
 *   - Do NOT average single-marker rvec/tvec as an initial guess.
 *   - Do NOT call estimatePoseBoard(..., useExtrinsicGuess=true).
 *   - Build explicit 3D-2D correspondences from all detected ArUco corners and
 *     solve board pose with cv::solvePnP(..., SOLVEPNP_ITERATIVE).
 *
 * Convention:
 *   X_cam1 = T_cam1_cam0 * X_cam0
 *
 * Flat input layout:
 *   data_dir/
 *     pair_0001_cam0.jpg
 *     pair_0001_cam1.jpg
 *     pair_0002_cam0.jpg
 *     pair_0002_cam1.jpg
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
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace two_cam_solvepnp {

constexpr double kRad2Deg = 180.0 / M_PI;

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
  bool refine_markers = false;
  double reproj_rmse_thresh_px = 6.0;
  double outlier_rot_thresh_deg = 1.0;
  double outlier_trans_thresh_m = 0.10;
  int min_valid_pairs = 2;
  bool save_debug = true;
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
};

struct PoseResult {
  bool ok = false;
  int marker_count = 0;
  double reproj_rmse_px = -1.0;
  std::vector<int> ids;
  Eigen::Matrix4d T_cam_board = Eigen::Matrix4d::Identity();
  cv::Mat vis;
};

struct PairMetric {
  std::string pair_id;
  bool ok0 = false;
  bool ok1 = false;
  bool hard_valid = false;
  bool accepted = false;
  bool outlier = false;
  int markers0 = 0;
  int markers1 = 0;
  double reproj0_px = -1.0;
  double reproj1_px = -1.0;
  double rot_err_deg = -1.0;
  double trans_err_m = -1.0;
  Eigen::Matrix4d T_cam1_cam0 = Eigen::Matrix4d::Identity();
};

template <typename T>
bool GetRequiredParam(const ros::NodeHandle& nh, const std::string& key, T& value) {
  if (!nh.getParam(key, value)) {
    ROS_ERROR_STREAM("Missing required private param: " << nh.getNamespace() << "/" << key);
    return false;
  }
  return true;
}

template <typename T>
void GetOptionalParam(const ros::NodeHandle& nh, const std::string& key, T& value) {
  nh.param(key, value, value);
}

bool LoadCameraModel(const ros::NodeHandle& nh, const std::string& prefix, CameraModel& cam) {
  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
  double k1 = 0.0, k2 = 0.0, p1 = 0.0, p2 = 0.0;

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

  if (!LoadCameraModel(nh, "cam0", cfg.cam0) ||
      !LoadCameraModel(nh, "cam1", cfg.cam1)) {
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
  GetOptionalParam(nh, "runtime/save_debug", cfg.runtime.save_debug);

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

std::vector<ImagePair> CollectFlatImagePairs(const std::string& data_dir) {
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

    ImagePair p;
    p.pair_id = pair_id;
    p.cam0_path = JoinPath(data_dir, name);
    p.cam1_path = JoinPath(data_dir, cam1_name);
    pairs.push_back(p);
  }

  std::sort(pairs.begin(), pairs.end(),
            [](const ImagePair& a, const ImagePair& b) {
              return a.pair_id < b.pair_id;
            });
  return pairs;
}

cv::Ptr<cv::aruco::Dictionary> CreateDictionary(const std::string& name) {
  if (name == "DICT_6X6_250") {
    return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  }
  throw std::runtime_error("Unsupported ArUco dictionary: " + name +
                           ". FAST-Calib target normally uses DICT_6X6_250.");
}

BoardGeometry BuildFastCalibBoard(const TargetModel& target) {
  BoardGeometry geom;
  geom.dictionary = CreateDictionary(target.dictionary);
  geom.board_ids = {1, 2, 4, 3};
  geom.board_corners.resize(4);

  const double width = target.delta_width_qr_center_m;
  const double height = target.delta_height_qr_center_m;

  for (int i = 0; i < 4; ++i) {
    const int x_qr_center = (i % 3) == 0 ? -1 : 1;  // 0/3 left, 1/2 right
    const int y_qr_center = (i < 2) ? 1 : -1;       // 0/1 top,  2/3 bottom
    const double x_center = x_qr_center * width;
    const double y_center = y_qr_center * height;

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

Eigen::Matrix4d RtToMatrix(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
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

    for (int k = 0; k < 4; ++k) {
      object_points.push_back(geom.board_corners[board_idx][k]);
      image_points.push_back(corners[i][k]);
    }
  }

  return object_points.size() >= 4 && object_points.size() == image_points.size();
}

double ReprojectionRmse(const std::vector<cv::Point3f>& object_points,
                        const std::vector<cv::Point2f>& image_points,
                        const cv::Vec3d& rvec,
                        const cv::Vec3d& tvec,
                        const CameraModel& cam) {
  if (object_points.empty()) return -1.0;

  std::vector<cv::Point2f> projected;
  cv::projectPoints(object_points, rvec, tvec, cam.K, cam.D, projected);

  double sum_sq = 0.0;
  int n = 0;
  for (size_t i = 0; i < projected.size() && i < image_points.size(); ++i) {
    const double dx = projected[i].x - image_points[i].x;
    const double dy = projected[i].y - image_points[i].y;
    sum_sq += dx * dx + dy * dy;
    ++n;
  }

  return n > 0 ? std::sqrt(sum_sq / static_cast<double>(n)) : -1.0;
}

bool DetectPoseSolvePnP(const cv::Mat& image,
                        const CameraModel& cam,
                        const BoardGeometry& geom,
                        const RuntimeConfig& runtime,
                        PoseResult& out) {
  out = PoseResult();
  if (image.empty()) return false;
  image.copyTo(out.vis);

  cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
#if (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2) || CV_MAJOR_VERSION < 3
  params->doCornerRefinement = true;
#else
  params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
#endif

  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<std::vector<cv::Point2f>> rejected;
  cv::aruco::detectMarkers(image, geom.dictionary, corners, ids, params, rejected);

#if (CV_MAJOR_VERSION > 3) || (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION >= 3)
  if (runtime.refine_markers && !rejected.empty()) {
    cv::aruco::refineDetectedMarkers(image, geom.board, corners, ids, rejected, cam.K, cam.D);
  }
#endif

  out.ids = ids;
  out.marker_count = static_cast<int>(ids.size());

  if (!ids.empty()) {
    cv::aruco::drawDetectedMarkers(out.vis, corners, ids);
  }

  if (static_cast<int>(ids.size()) < runtime.min_detected_markers) {
    return false;
  }

  std::vector<cv::Point3f> object_points;
  std::vector<cv::Point2f> image_points;
  if (!BuildCorrespondences(geom, ids, corners, object_points, image_points)) {
    return false;
  }

  cv::Vec3d rvec(0, 0, 0), tvec(0, 0, 0);

  // Important: no extrinsic guess here.
  // This avoids the previous wrong local solution caused by averaging individual marker poses.
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

  out.reproj_rmse_px = ReprojectionRmse(object_points, image_points, rvec, tvec, cam);
  out.T_cam_board = RtToMatrix(rvec, tvec);
  out.ok = out.reproj_rmse_px >= 0.0 && out.reproj_rmse_px <= runtime.reproj_rmse_thresh_px;

  try {
    cv::aruco::drawAxis(out.vis, cam.K, cam.D, rvec, tvec, 0.2);
  } catch (...) {
    // Visualization only.
  }

  cv::Scalar color = out.ok ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
  std::ostringstream oss;
  oss << cam.name << " markers=" << out.marker_count
      << " rmse=" << std::fixed << std::setprecision(3) << out.reproj_rmse_px
      << " " << (out.ok ? "OK" : "BAD");
  cv::putText(out.vis, oss.str(), cv::Point(20, 40),
              cv::FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv::LINE_AA);

  return out.ok;
}

double RotationAngleDeg(const Eigen::Matrix3d& Ra, const Eigen::Matrix3d& Rb) {
  Eigen::Matrix3d dR = Ra.transpose() * Rb;
  double c = (dR.trace() - 1.0) * 0.5;
  c = std::max(-1.0, std::min(1.0, c));
  return std::acos(c) * kRad2Deg;
}

Eigen::Quaterniond RotationOf(const Eigen::Matrix4d& T) {
  Eigen::Quaterniond q(T.block<3, 3>(0, 0));
  q.normalize();
  return q;
}

Eigen::Vector3d TranslationOf(const Eigen::Matrix4d& T) {
  return T.block<3, 1>(0, 3);
}

Eigen::Matrix4d AverageTransforms(const std::vector<PairMetric>& metrics,
                                  const std::vector<int>& indices) {
  if (indices.empty()) return Eigen::Matrix4d::Identity();

  const Eigen::Quaterniond q_ref = RotationOf(metrics[indices.front()].T_cam1_cam0);
  Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
  Eigen::Vector3d t_sum = Eigen::Vector3d::Zero();
  double w_sum = 0.0;

  for (int idx : indices) {
    const PairMetric& m = metrics[idx];
    const double e = std::max(1e-6, 0.5 * (m.reproj0_px + m.reproj1_px));
    const double w = 1.0 / (e * e);

    Eigen::Quaterniond q = RotationOf(m.T_cam1_cam0);
    if (q.dot(q_ref) < 0.0) q.coeffs() *= -1.0;

    Eigen::Vector4d v(q.w(), q.x(), q.y(), q.z());
    A += w * (v * v.transpose());

    t_sum += w * TranslationOf(m.T_cam1_cam0);
    w_sum += w;
  }

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(A);
  Eigen::Vector4d v = solver.eigenvectors().col(3);

  Eigen::Quaterniond q_mean(v(0), v(1), v(2), v(3));
  q_mean.normalize();

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = q_mean.toRotationMatrix();
  T.block<3, 1>(0, 3) = t_sum / std::max(1e-12, w_sum);
  return T;
}

Eigen::Vector3d RotationMatrixToRpyDeg(const Eigen::Matrix3d& R) {
  const double sy = std::sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
  const bool singular = sy < 1e-9;

  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;

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
  ofs << "pair_id,ok0,ok1,hard_valid,accepted,outlier,markers0,markers1,"
      << "reproj0_px,reproj1_px,rot_err_deg,trans_err_m,tx,ty,tz\n";

  for (const auto& m : metrics) {
    Eigen::Vector3d t = TranslationOf(m.T_cam1_cam0);
    ofs << m.pair_id << ","
        << m.ok0 << "," << m.ok1 << ","
        << m.hard_valid << "," << m.accepted << "," << m.outlier << ","
        << m.markers0 << "," << m.markers1 << ","
        << std::fixed << std::setprecision(6)
        << m.reproj0_px << "," << m.reproj1_px << ","
        << m.rot_err_deg << "," << m.trans_err_m << ","
        << t.x() << "," << t.y() << "," << t.z() << "\n";
  }
}

void SaveResultYaml(const std::string& path,
                    const AppConfig& cfg,
                    const Eigen::Matrix4d& T_cam1_cam0,
                    const std::vector<PairMetric>& metrics,
                    const std::vector<int>& used_indices) {
  const Eigen::Matrix4d T_cam0_cam1 = T_cam1_cam0.inverse();
  const Eigen::Vector3d t = TranslationOf(T_cam1_cam0);
  const Eigen::Vector3d rpy = RotationMatrixToRpyDeg(T_cam1_cam0.block<3, 3>(0, 0));

  std::vector<double> reproj0;
  std::vector<double> reproj1;
  std::vector<double> rot_errs;
  std::vector<double> trans_errs;

  for (int idx : used_indices) {
    reproj0.push_back(metrics[idx].reproj0_px);
    reproj1.push_back(metrics[idx].reproj1_px);
    if (metrics[idx].rot_err_deg >= 0.0) rot_errs.push_back(metrics[idx].rot_err_deg);
    if (metrics[idx].trans_err_m >= 0.0) trans_errs.push_back(metrics[idx].trans_err_m);
  }

  std::ofstream ofs(path);
  ofs << "pair_name: \"" << cfg.pair_name << "\"\n";
  ofs << "cam0_name: \"" << cfg.cam0.name << "\"\n";
  ofs << "cam1_name: \"" << cfg.cam1.name << "\"\n";
  ofs << "convention: \"X_cam1 = T_cam1_cam0 * X_cam0\"\n";
  ofs << "method: \"aruco_corners_solvepnp_iterative_no_initial_guess\"\n";
  ofs << "used_pairs: " << used_indices.size() << "\n";
  ofs << "total_hard_valid_pairs: " << metrics.size() << "\n\n";

  WriteMatrixYaml(ofs, "T_cam1_cam0", T_cam1_cam0);
  ofs << "\n";
  WriteMatrixYaml(ofs, "T_cam0_cam1", T_cam0_cam1);
  ofs << "\n";

  ofs << "translation_xyz_m: ["
      << std::fixed << std::setprecision(10)
      << t.x() << ", " << t.y() << ", " << t.z() << "]\n";
  ofs << "translation_norm_m: " << std::fixed << std::setprecision(10) << t.norm() << "\n";
  ofs << "rpy_deg: ["
      << std::fixed << std::setprecision(10)
      << rpy.x() << ", " << rpy.y() << ", " << rpy.z() << "]\n";

  ofs << "quality:\n";
  ofs << "  cam0_reproj_px_mean: " << MeanValue(reproj0) << "\n";
  ofs << "  cam1_reproj_px_mean: " << MeanValue(reproj1) << "\n";
  ofs << "  rot_err_deg_std: " << StdValue(rot_errs) << "\n";
  ofs << "  trans_err_m_std: " << StdValue(trans_errs) << "\n";

  ofs << "used_pair_ids:\n";
  for (int idx : used_indices) {
    ofs << "  - \"" << metrics[idx].pair_id << "\"\n";
  }
}

void PrintTransform(const Eigen::Matrix4d& T) {
  std::ostringstream oss;
  oss << "\n" << std::fixed << std::setprecision(10);
  for (int r = 0; r < 4; ++r) {
    oss << "  ";
    for (int c = 0; c < 4; ++c) {
      oss << std::setw(15) << T(r, c);
    }
    if (r != 3) oss << "\n";
  }
  ROS_INFO_STREAM("T_cam1_cam0:" << oss.str());

  const Eigen::Vector3d t = TranslationOf(T);
  const Eigen::Vector3d rpy = RotationMatrixToRpyDeg(T.block<3, 3>(0, 0));
  ROS_INFO_STREAM("translation_xyz_m = [" << t.x() << ", " << t.y() << ", " << t.z() << "]");
  ROS_INFO_STREAM("translation_norm_m = " << t.norm());
  ROS_INFO_STREAM("rpy_deg = [" << rpy.x() << ", " << rpy.y() << ", " << rpy.z() << "]");
}

}  // namespace two_cam_solvepnp

int main(int argc, char** argv) {
  ros::init(argc, argv, "two_camera_calib_solvepnp");
  ros::NodeHandle nh("~");

  using namespace two_cam_solvepnp;

  AppConfig cfg;
  if (!LoadConfig(nh, cfg)) {
    ROS_ERROR("Failed to load two-camera calibration config from private rosparams.");
    return 1;
  }

  try {
    MakeDirs(cfg.output_dir);
    const std::string vis_dir = JoinPath(cfg.output_dir, "vis");
    if (cfg.runtime.save_debug) MakeDirs(vis_dir);

    ROS_INFO_STREAM("Two-camera calibration solvePnP for pair: " << cfg.pair_name);
    ROS_INFO_STREAM("data_dir: " << cfg.data_dir);
    ROS_INFO_STREAM("output_dir: " << cfg.output_dir);
    ROS_INFO_STREAM("cam0: " << cfg.cam0.name);
    ROS_INFO_STREAM("cam1: " << cfg.cam1.name);
    ROS_INFO("Convention: X_cam1 = T_cam1_cam0 * X_cam0");

    const std::vector<ImagePair> image_pairs = CollectFlatImagePairs(cfg.data_dir);
    ROS_INFO_STREAM("Found " << image_pairs.size() << " image pairs.");
    if (image_pairs.empty()) {
      ROS_ERROR("No image pairs found.");
      return 1;
    }

    BoardGeometry geom = BuildFastCalibBoard(cfg.target);

    std::vector<PairMetric> metrics;
    for (const auto& pair : image_pairs) {
      cv::Mat img0 = cv::imread(pair.cam0_path, cv::IMREAD_COLOR);
      cv::Mat img1 = cv::imread(pair.cam1_path, cv::IMREAD_COLOR);

      PairMetric m;
      m.pair_id = pair.pair_id;

      PoseResult pose0;
      PoseResult pose1;
      m.ok0 = DetectPoseSolvePnP(img0, cfg.cam0, geom, cfg.runtime, pose0);
      m.ok1 = DetectPoseSolvePnP(img1, cfg.cam1, geom, cfg.runtime, pose1);

      m.markers0 = pose0.marker_count;
      m.markers1 = pose1.marker_count;
      m.reproj0_px = pose0.reproj_rmse_px;
      m.reproj1_px = pose1.reproj_rmse_px;
      m.hard_valid = m.ok0 && m.ok1;

      if (m.hard_valid) {
        m.T_cam1_cam0 = pose1.T_cam_board * pose0.T_cam_board.inverse();
        metrics.push_back(m);
      }

      ROS_INFO_STREAM("pair=" << pair.pair_id
                      << " markers=(" << pose0.marker_count << "," << pose1.marker_count << ")"
                      << " reproj=(" << pose0.reproj_rmse_px << "," << pose1.reproj_rmse_px << ")"
                      << " hard_valid=" << m.hard_valid);

      if (cfg.runtime.save_debug) {
        if (!pose0.vis.empty()) cv::imwrite(JoinPath(vis_dir, pair.pair_id + "_cam0_detect.jpg"), pose0.vis);
        if (!pose1.vis.empty()) cv::imwrite(JoinPath(vis_dir, pair.pair_id + "_cam1_detect.jpg"), pose1.vis);
      }
    }

    if (metrics.empty()) {
      ROS_ERROR("No hard-valid pairs. Check marker detection, intrinsics, and board geometry.");
      return 1;
    }

    std::vector<int> all_indices;
    for (int i = 0; i < static_cast<int>(metrics.size()); ++i) all_indices.push_back(i);

    Eigen::Matrix4d T_initial = AverageTransforms(metrics, all_indices);

    std::vector<int> inlier_indices;
    for (int i = 0; i < static_cast<int>(metrics.size()); ++i) {
      metrics[i].rot_err_deg =
          RotationAngleDeg(T_initial.block<3, 3>(0, 0),
                           metrics[i].T_cam1_cam0.block<3, 3>(0, 0));
      metrics[i].trans_err_m =
          (TranslationOf(metrics[i].T_cam1_cam0) - TranslationOf(T_initial)).norm();

      metrics[i].outlier = !(metrics[i].rot_err_deg <= cfg.runtime.outlier_rot_thresh_deg &&
                             metrics[i].trans_err_m <= cfg.runtime.outlier_trans_thresh_m);
      metrics[i].accepted = !metrics[i].outlier;

      if (metrics[i].accepted) inlier_indices.push_back(i);
    }

    std::vector<int> used_indices;
    if (static_cast<int>(inlier_indices.size()) >= cfg.runtime.min_valid_pairs) {
      used_indices = inlier_indices;
    } else {
      ROS_WARN_STREAM("Only " << inlier_indices.size()
                      << " inlier pairs after outlier rejection, less than min_valid_pairs="
                      << cfg.runtime.min_valid_pairs
                      << ". Use all hard-valid pairs for now.");
      used_indices = all_indices;
      for (int idx : used_indices) {
        metrics[idx].accepted = true;
        metrics[idx].outlier = false;
      }
    }

    Eigen::Matrix4d T_final = AverageTransforms(metrics, used_indices);

    for (auto& m : metrics) {
      m.rot_err_deg = RotationAngleDeg(T_final.block<3, 3>(0, 0),
                                       m.T_cam1_cam0.block<3, 3>(0, 0));
      m.trans_err_m = (TranslationOf(m.T_cam1_cam0) - TranslationOf(T_final)).norm();
    }

    ROS_INFO("==== Two-camera calibration result ====");
    PrintTransform(T_final);
    ROS_INFO_STREAM("used_pairs = " << used_indices.size() << " / " << metrics.size());

    const std::string result_path = JoinPath(cfg.output_dir, "result.yaml");
    const std::string metrics_path = JoinPath(cfg.output_dir, "per_pair_metrics.csv");

    SaveResultYaml(result_path, cfg, T_final, metrics, used_indices);
    SaveMetricsCsv(metrics_path, metrics);

    ROS_INFO_STREAM("Saved result to: " << result_path);
    ROS_INFO_STREAM("Saved metrics to: " << metrics_path);
  } catch (const std::exception& e) {
    ROS_ERROR_STREAM("two_camera_calib_solvepnp failed: " << e.what());
    return 1;
  }

  return 0;
}
