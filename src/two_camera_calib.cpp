/*
Developer: guoliefeng / FAST-Calib extension

Two-camera extrinsic calibration using the same ArUco board definition as FAST-Calib.
The estimated transform is:
    X_cam1 = T_cam1_cam0 * X_cam0
where cam0 is the reference camera of one pair, and cam1 is the target camera.
*/

#include <ros/ros.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core/version.hpp>

#include <XmlRpcValue.h>
#include <cerrno>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

namespace fast_calib
{

struct CameraConfig
{
  std::string name;
  std::string image_path;
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  double k1 = 0.0;
  double k2 = 0.0;
  double p1 = 0.0;
  double p2 = 0.0;
  double k3 = 0.0;
};

struct TargetConfig
{
  double marker_size = 0.20;
  double delta_width_qr_center = 0.55;
  double delta_height_qr_center = 0.35;
  int min_detected_markers = 3;
  std::string dictionary = "DICT_6X6_250";
};

struct PairConfig
{
  std::string pair_name;
  std::string output_path;
  CameraConfig cam0;
  CameraConfig cam1;
};

struct PoseResult
{
  bool ok = false;
  std::vector<int> detected_ids;
  cv::Mat T_cam_board = cv::Mat::eye(4, 4, CV_64F);
  cv::Vec3d rvec = cv::Vec3d(0, 0, 0);
  cv::Vec3d tvec = cv::Vec3d(0, 0, 0);
  double reprojection_rmse_px = -1.0;
  cv::Mat debug_image;
};

template <typename T>
bool getRequiredParam(const ros::NodeHandle& nh, const std::string& key, T& value)
{
  if (!nh.getParam(key, value))
  {
    ROS_ERROR_STREAM("[two_camera_calib] Missing required parameter: " << key);
    return false;
  }
  return true;
}

bool makeDirectoryRecursive(const std::string& path)
{
  if (path.empty()) return false;

  std::string current;
  for (size_t i = 0; i < path.size(); ++i)
  {
    current.push_back(path[i]);
    if (path[i] != '/' && i + 1 != path.size()) continue;

    if (current.empty() || current == "/") continue;
    if (::mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
    {
      ROS_ERROR_STREAM("[two_camera_calib] Failed to create directory: " << current
                       << ", errno=" << errno);
      return false;
    }
  }
  return true;
}

cv::Mat cameraMatrix(const CameraConfig& cam)
{
  return (cv::Mat_<double>(3, 3) << cam.fx, 0.0, cam.cx,
                                      0.0, cam.fy, cam.cy,
                                      0.0, 0.0, 1.0);
}

cv::Mat distCoeffs(const CameraConfig& cam)
{
  return (cv::Mat_<double>(1, 5) << cam.k1, cam.k2, cam.p1, cam.p2, cam.k3);
}

cv::Ptr<cv::aruco::Dictionary> createDictionary(const std::string& name)
{
  if (name == "DICT_4X4_50") return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
  if (name == "DICT_5X5_100") return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_100);
  if (name == "DICT_6X6_250") return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  if (name == "DICT_7X7_250") return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_250);

  ROS_WARN_STREAM("[two_camera_calib] Unsupported dictionary '" << name
                  << "', fallback to DICT_6X6_250.");
  return cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
}

cv::Ptr<cv::aruco::Board> createFastCalibBoard(const TargetConfig& target,
                                                const cv::Ptr<cv::aruco::Dictionary>& dictionary)
{
  // Same board layout as src/qr_detect.hpp in FAST-Calib.
  // Marker order in board coordinates:
  // 0-------1
  // |       |
  // |   C   |
  // |       |
  // 3-------2
  // ArUco IDs: Marker 0 -> 1, Marker 1 -> 2, Marker 2 -> 4, Marker 3 -> 3.
  std::vector<std::vector<cv::Point3f>> board_corners;
  board_corners.resize(4);

  const float half_qr_width = static_cast<float>(target.delta_width_qr_center);
  const float half_qr_height = static_cast<float>(target.delta_height_qr_center);
  const float half_marker = static_cast<float>(target.marker_size / 2.0);

  for (int i = 0; i < 4; ++i)
  {
    const int x_qr_center = (i % 3) == 0 ? -1 : 1;
    const int y_qr_center = (i < 2) ? 1 : -1;
    const float x_center = x_qr_center * half_qr_width;
    const float y_center = y_qr_center * half_qr_height;

    for (int j = 0; j < 4; ++j)
    {
      const int x_qr = (j % 3) == 0 ? -1 : 1;
      const int y_qr = (j < 2) ? 1 : -1;
      board_corners[i].push_back(cv::Point3f(x_center + x_qr * half_marker,
                                             y_center + y_qr * half_marker,
                                             0.0f));
    }
  }

  const std::vector<int> board_ids{1, 2, 4, 3};
  return cv::aruco::Board::create(board_corners, dictionary, board_ids);
}

cv::Mat poseToMatrix(const cv::Vec3d& rvec, const cv::Vec3d& tvec)
{
  cv::Mat R;
  cv::Rodrigues(rvec, R);

  cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
  R.copyTo(T(cv::Rect(0, 0, 3, 3)));
  T.at<double>(0, 3) = tvec[0];
  T.at<double>(1, 3) = tvec[1];
  T.at<double>(2, 3) = tvec[2];
  return T;
}

std::vector<double> rotationMatrixToRpyDeg(const cv::Mat& R)
{
  const double r00 = R.at<double>(0, 0);
  const double r10 = R.at<double>(1, 0);
  const double r20 = R.at<double>(2, 0);
  const double r21 = R.at<double>(2, 1);
  const double r22 = R.at<double>(2, 2);

  const double roll = std::atan2(r21, r22);
  const double pitch = std::atan2(-r20, std::sqrt(r00 * r00 + r10 * r10));
  const double yaw = std::atan2(r10, r00);
  const double rad_to_deg = 180.0 / M_PI;
  return {roll * rad_to_deg, pitch * rad_to_deg, yaw * rad_to_deg};
}

double computeBoardReprojectionRmse(const cv::Ptr<cv::aruco::Board>& board,
                                    const std::vector<int>& detected_ids,
                                    const std::vector<std::vector<cv::Point2f>>& detected_corners,
                                    const cv::Vec3d& rvec,
                                    const cv::Vec3d& tvec,
                                    const cv::Mat& K,
                                    const cv::Mat& D)
{
  double sum_square_error = 0.0;
  int point_count = 0;

  for (size_t detected_idx = 0; detected_idx < detected_ids.size(); ++detected_idx)
  {
    const int marker_id = detected_ids[detected_idx];
    int board_idx = -1;
    for (size_t i = 0; i < board->ids.size(); ++i)
    {
      if (board->ids[i] == marker_id)
      {
        board_idx = static_cast<int>(i);
        break;
      }
    }
    if (board_idx < 0) continue;

    std::vector<cv::Point2f> projected;
    cv::projectPoints(board->objPoints[board_idx], rvec, tvec, K, D, projected);

    for (size_t j = 0; j < projected.size() && j < detected_corners[detected_idx].size(); ++j)
    {
      const cv::Point2f diff = projected[j] - detected_corners[detected_idx][j];
      sum_square_error += diff.x * diff.x + diff.y * diff.y;
      ++point_count;
    }
  }

  if (point_count == 0) return -1.0;
  return std::sqrt(sum_square_error / static_cast<double>(point_count));
}

void drawAxes(cv::Mat& image, const cv::Mat& K, const cv::Mat& D,
              const cv::Vec3d& rvec, const cv::Vec3d& tvec, double axis_length)
{
#if CV_MAJOR_VERSION >= 4
  cv::drawFrameAxes(image, K, D, rvec, tvec, static_cast<float>(axis_length));
#else
  cv::aruco::drawAxis(image, K, D, rvec, tvec, static_cast<float>(axis_length));
#endif
}

PoseResult detectBoardPose(const CameraConfig& cam,
                           const TargetConfig& target,
                           const cv::Ptr<cv::aruco::Dictionary>& dictionary,
                           const cv::Ptr<cv::aruco::Board>& board)
{
  PoseResult result;
  cv::Mat image = cv::imread(cam.image_path, cv::IMREAD_COLOR);
  if (image.empty())
  {
    ROS_ERROR_STREAM("[two_camera_calib] Failed to load image for " << cam.name
                     << ": " << cam.image_path);
    return result;
  }

  result.debug_image = image.clone();
  const cv::Mat K = cameraMatrix(cam);
  const cv::Mat D = distCoeffs(cam);

  cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
#if (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2) || CV_MAJOR_VERSION < 3
  parameters->doCornerRefinement = true;
#else
  parameters->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
#endif

  std::vector<int> ids;
  std::vector<std::vector<cv::Point2f>> corners;
  std::vector<std::vector<cv::Point2f>> rejected;
  cv::aruco::detectMarkers(image, dictionary, corners, ids, parameters, rejected);

#if (CV_MAJOR_VERSION > 3) || (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION >= 2)
  if (!ids.empty() && !rejected.empty())
  {
    cv::aruco::refineDetectedMarkers(image, board, corners, ids, rejected, K, D);
  }
#endif

  result.detected_ids = ids;
  if (!ids.empty())
  {
    cv::aruco::drawDetectedMarkers(result.debug_image, corners, ids);
  }

  std::ostringstream id_stream;
  for (size_t i = 0; i < ids.size(); ++i)
  {
    if (i != 0) id_stream << ",";
    id_stream << ids[i];
  }
  ROS_INFO_STREAM("[two_camera_calib] " << cam.name << " detected marker ids: ["
                  << id_stream.str() << "]");

  if (static_cast<int>(ids.size()) < target.min_detected_markers)
  {
    ROS_ERROR_STREAM("[two_camera_calib] " << cam.name << " detected " << ids.size()
                     << " markers, but at least " << target.min_detected_markers
                     << " markers are required.");
    return result;
  }

  cv::Vec3d rvec(0, 0, 0), tvec(0, 0, 0);
#if (CV_MAJOR_VERSION == 3 && CV_MINOR_VERSION <= 2) || CV_MAJOR_VERSION < 3
  const int valid = cv::aruco::estimatePoseBoard(corners, ids, board, K, D, rvec, tvec);
#else
  const int valid = cv::aruco::estimatePoseBoard(corners, ids, board, K, D, rvec, tvec, false);
#endif

  if (valid <= 0)
  {
    ROS_ERROR_STREAM("[two_camera_calib] estimatePoseBoard failed for " << cam.name);
    return result;
  }

  result.ok = true;
  result.rvec = rvec;
  result.tvec = tvec;
  result.T_cam_board = poseToMatrix(rvec, tvec);
  result.reprojection_rmse_px = computeBoardReprojectionRmse(board, ids, corners, rvec, tvec, K, D);

  drawAxes(result.debug_image, K, D, rvec, tvec, target.marker_size);
  ROS_INFO_STREAM("[two_camera_calib] " << cam.name
                  << " board reprojection RMSE: " << std::fixed << std::setprecision(3)
                  << result.reprojection_rmse_px << " px");

  return result;
}

bool readCameraConfig(const ros::NodeHandle& nh, const std::string& ns, CameraConfig& cam)
{
  bool ok = true;
  ok &= getRequiredParam(nh, ns + "/name", cam.name);
  ok &= getRequiredParam(nh, ns + "/image_path", cam.image_path);
  ok &= getRequiredParam(nh, ns + "/fx", cam.fx);
  ok &= getRequiredParam(nh, ns + "/fy", cam.fy);
  ok &= getRequiredParam(nh, ns + "/cx", cam.cx);
  ok &= getRequiredParam(nh, ns + "/cy", cam.cy);

  nh.param(ns + "/k1", cam.k1, 0.0);
  nh.param(ns + "/k2", cam.k2, 0.0);
  nh.param(ns + "/p1", cam.p1, 0.0);
  nh.param(ns + "/p2", cam.p2, 0.0);
  nh.param(ns + "/k3", cam.k3, 0.0);
  return ok;
}

bool readTargetConfig(const ros::NodeHandle& nh, TargetConfig& target)
{
  nh.param("calib_target/marker_size", target.marker_size, 0.20);
  nh.param("calib_target/delta_width_qr_center", target.delta_width_qr_center, 0.55);
  nh.param("calib_target/delta_height_qr_center", target.delta_height_qr_center, 0.35);
  nh.param("calib_target/min_detected_markers", target.min_detected_markers, 3);
  nh.param("calib_target/dictionary", target.dictionary, std::string("DICT_6X6_250"));
  return true;
}

bool readPairConfig(const ros::NodeHandle& nh, const std::string& pair_name, PairConfig& pair)
{
  pair.pair_name = pair_name;
  bool ok = true;
  ok &= getRequiredParam(nh, pair_name + "/output_path", pair.output_path);
  ok &= readCameraConfig(nh, pair_name + "/cam0", pair.cam0);
  ok &= readCameraConfig(nh, pair_name + "/cam1", pair.cam1);
  return ok;
}

std::vector<std::string> readPairNames(const ros::NodeHandle& nh)
{
  std::vector<std::string> pair_names;
  XmlRpc::XmlRpcValue names;
  if (!nh.getParam("pair_names", names))
  {
    ROS_ERROR_STREAM("[two_camera_calib] Missing required parameter: pair_names");
    return pair_names;
  }

  if (names.getType() != XmlRpc::XmlRpcValue::TypeArray)
  {
    ROS_ERROR_STREAM("[two_camera_calib] pair_names must be a YAML list, for example: [left_pair, right_pair]");
    return pair_names;
  }

  for (int i = 0; i < names.size(); ++i)
  {
    if (names[i].getType() != XmlRpc::XmlRpcValue::TypeString)
    {
      ROS_ERROR_STREAM("[two_camera_calib] pair_names[" << i << "] is not a string.");
      continue;
    }
    pair_names.push_back(static_cast<std::string>(names[i]));
  }
  return pair_names;
}

void writeMatrixToConsole(const std::string& name, const cv::Mat& T)
{
  std::cout << name << " =" << std::endl;
  std::cout << std::fixed << std::setprecision(6);
  for (int r = 0; r < T.rows; ++r)
  {
    std::cout << "[ ";
    for (int c = 0; c < T.cols; ++c)
    {
      std::cout << std::setw(11) << T.at<double>(r, c);
      if (c + 1 != T.cols) std::cout << " ";
    }
    std::cout << " ]" << std::endl;
  }
}

void saveCalibrationYaml(const PairConfig& pair,
                         const TargetConfig& target,
                         const PoseResult& cam0_pose,
                         const PoseResult& cam1_pose,
                         const cv::Mat& T_cam1_cam0,
                         const cv::Mat& T_cam0_cam1)
{
  const std::string yaml_path = pair.output_path + "/camera_camera_extrinsic.yaml";
  cv::FileStorage fs(yaml_path, cv::FileStorage::WRITE);
  if (!fs.isOpened())
  {
    ROS_ERROR_STREAM("[two_camera_calib] Failed to write result file: " << yaml_path);
    return;
  }

  const std::vector<double> rpy_cam1_cam0 = rotationMatrixToRpyDeg(T_cam1_cam0(cv::Rect(0, 0, 3, 3)));
  const std::vector<double> rpy_cam0_cam1 = rotationMatrixToRpyDeg(T_cam0_cam1(cv::Rect(0, 0, 3, 3)));

  fs << "pair_name" << pair.pair_name;
  fs << "definition" << "X_cam1 = T_cam1_cam0 * X_cam0";
  fs << "cam0_name" << pair.cam0.name;
  fs << "cam1_name" << pair.cam1.name;

  fs << "calib_target" << "{";
  fs << "dictionary" << target.dictionary;
  fs << "marker_size" << target.marker_size;
  fs << "delta_width_qr_center" << target.delta_width_qr_center;
  fs << "delta_height_qr_center" << target.delta_height_qr_center;
  fs << "min_detected_markers" << target.min_detected_markers;
  fs << "}";

  fs << "cam0_detected_ids" << "[";
  for (int id : cam0_pose.detected_ids) fs << id;
  fs << "]";
  fs << "cam1_detected_ids" << "[";
  for (int id : cam1_pose.detected_ids) fs << id;
  fs << "]";
  fs << "cam0_reprojection_rmse_px" << cam0_pose.reprojection_rmse_px;
  fs << "cam1_reprojection_rmse_px" << cam1_pose.reprojection_rmse_px;

  fs << "T_cam0_board" << cam0_pose.T_cam_board;
  fs << "T_cam1_board" << cam1_pose.T_cam_board;
  fs << "T_cam1_cam0" << T_cam1_cam0;
  fs << "T_cam0_cam1" << T_cam0_cam1;

  fs << "T_cam1_cam0_translation_xyz_m" << "["
     << T_cam1_cam0.at<double>(0, 3)
     << T_cam1_cam0.at<double>(1, 3)
     << T_cam1_cam0.at<double>(2, 3) << "]";
  fs << "T_cam1_cam0_rpy_deg" << "[" << rpy_cam1_cam0[0] << rpy_cam1_cam0[1] << rpy_cam1_cam0[2] << "]";

  fs << "T_cam0_cam1_translation_xyz_m" << "["
     << T_cam0_cam1.at<double>(0, 3)
     << T_cam0_cam1.at<double>(1, 3)
     << T_cam0_cam1.at<double>(2, 3) << "]";
  fs << "T_cam0_cam1_rpy_deg" << "[" << rpy_cam0_cam1[0] << rpy_cam0_cam1[1] << rpy_cam0_cam1[2] << "]";
  fs.release();

  ROS_INFO_STREAM("[two_camera_calib] Saved result: " << yaml_path);
}

bool calibratePair(const PairConfig& pair,
                   const TargetConfig& target,
                   const cv::Ptr<cv::aruco::Dictionary>& dictionary,
                   const cv::Ptr<cv::aruco::Board>& board)
{
  ROS_INFO_STREAM("================ Two-camera pair: " << pair.pair_name << " ================");
  ROS_INFO_STREAM("[two_camera_calib] cam0(reference): " << pair.cam0.name);
  ROS_INFO_STREAM("[two_camera_calib] cam1(target):    " << pair.cam1.name);

  if (!makeDirectoryRecursive(pair.output_path)) return false;

  const PoseResult cam0_pose = detectBoardPose(pair.cam0, target, dictionary, board);
  const PoseResult cam1_pose = detectBoardPose(pair.cam1, target, dictionary, board);
  if (!cam0_pose.ok || !cam1_pose.ok)
  {
    ROS_ERROR_STREAM("[two_camera_calib] Failed to calibrate pair: " << pair.pair_name);
    return false;
  }

  // estimatePoseBoard returns T_cam_board. Therefore:
  // X_cam1 = T_cam1_board * inv(T_cam0_board) * X_cam0
  const cv::Mat T_cam1_cam0 = cam1_pose.T_cam_board * cam0_pose.T_cam_board.inv();
  const cv::Mat T_cam0_cam1 = T_cam1_cam0.inv();

  writeMatrixToConsole("T_" + pair.cam1.name + "_from_" + pair.cam0.name, T_cam1_cam0);
  const std::vector<double> rpy = rotationMatrixToRpyDeg(T_cam1_cam0(cv::Rect(0, 0, 3, 3)));
  ROS_INFO_STREAM("[two_camera_calib] translation xyz [m]: "
                  << T_cam1_cam0.at<double>(0, 3) << ", "
                  << T_cam1_cam0.at<double>(1, 3) << ", "
                  << T_cam1_cam0.at<double>(2, 3));
  ROS_INFO_STREAM("[two_camera_calib] rpy [deg]: " << rpy[0] << ", " << rpy[1] << ", " << rpy[2]);

  const std::string cam0_debug_path = pair.output_path + "/" + pair.cam0.name + "_aruco_detect.png";
  const std::string cam1_debug_path = pair.output_path + "/" + pair.cam1.name + "_aruco_detect.png";
  cv::imwrite(cam0_debug_path, cam0_pose.debug_image);
  cv::imwrite(cam1_debug_path, cam1_pose.debug_image);
  ROS_INFO_STREAM("[two_camera_calib] Saved debug image: " << cam0_debug_path);
  ROS_INFO_STREAM("[two_camera_calib] Saved debug image: " << cam1_debug_path);

  saveCalibrationYaml(pair, target, cam0_pose, cam1_pose, T_cam1_cam0, T_cam0_cam1);
  return true;
}

}  // namespace fast_calib

int main(int argc, char** argv)
{
  ros::init(argc, argv, "two_camera_calib");
  ros::NodeHandle nh;

  fast_calib::TargetConfig target;
  fast_calib::readTargetConfig(nh, target);

  const std::vector<std::string> pair_names = fast_calib::readPairNames(nh);
  if (pair_names.empty()) return 1;

  const cv::Ptr<cv::aruco::Dictionary> dictionary = fast_calib::createDictionary(target.dictionary);
  const cv::Ptr<cv::aruco::Board> board = fast_calib::createFastCalibBoard(target, dictionary);

  int success_count = 0;
  for (const std::string& pair_name : pair_names)
  {
    fast_calib::PairConfig pair;
    if (!fast_calib::readPairConfig(nh, pair_name, pair))
    {
      ROS_ERROR_STREAM("[two_camera_calib] Invalid config for pair: " << pair_name);
      continue;
    }

    if (fast_calib::calibratePair(pair, target, dictionary, board))
    {
      ++success_count;
    }
  }

  ROS_INFO_STREAM("[two_camera_calib] Done. Successful pairs: " << success_count
                  << " / " << pair_names.size());
  return success_count == static_cast<int>(pair_names.size()) ? 0 : 2;
}
