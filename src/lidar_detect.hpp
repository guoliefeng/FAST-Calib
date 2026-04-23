/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIDAR_DETECT_HPP
#define LIDAR_DETECT_HPP
#define PCL_NO_PRECOMPILE

#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PointStamped.h>
#include <Eigen/Dense>
#include <ros/ros.h>
#include <pcl/filters/voxel_grid.h>
#include <algorithm>
#include <limits>
#include <random>
#include "common_lib.h"

class LidarDetect
{
private:
    double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
    double circle_radius_, delta_width_circles_, delta_height_circles_;

    // 存储中间结果的点云
    pcl::PointCloud<Common::Point>::Ptr filtered_cloud_;
    pcl::PointCloud<Common::Point>::Ptr plane_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr center_z0_cloud_;

    struct CircleCandidate
    {
        pcl::PointXYZ center;
        double radius = 0.0;
        int inliers = 0;
    };

    struct CandidateScore
    {
        bool found = false;
        bool geom_valid = false;
        double geom_score = std::numeric_limits<double>::infinity();
        double rmse = std::numeric_limits<double>::infinity();
        int support = 0;
        std::vector<int> group;
    };

    bool fitCircleFromThree(const pcl::PointXYZ &p1,
                            const pcl::PointXYZ &p2,
                            const pcl::PointXYZ &p3,
                            double &cx,
                            double &cy,
                            double &radius) const
    {
        const double a11 = 2.0 * (p2.x - p1.x);
        const double a12 = 2.0 * (p2.y - p1.y);
        const double a21 = 2.0 * (p3.x - p1.x);
        const double a22 = 2.0 * (p3.y - p1.y);
        const double b1 = p2.x * p2.x + p2.y * p2.y - p1.x * p1.x - p1.y * p1.y;
        const double b2 = p3.x * p3.x + p3.y * p3.y - p1.x * p1.x - p1.y * p1.y;
        const double det = a11 * a22 - a12 * a21;

        if (std::fabs(det) < 1e-9)
        {
            return false;
        }

        cx = (b1 * a22 - a12 * b2) / det;
        cy = (a11 * b2 - b1 * a21) / det;
        radius = std::hypot(static_cast<double>(p1.x) - cx, static_cast<double>(p1.y) - cy);
        return std::isfinite(radius);
    }

    bool refineCircleLeastSquares(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                                  const std::vector<int> &indices,
                                  CircleCandidate &candidate) const
    {
        if (indices.size() < 3)
        {
            return false;
        }

        Eigen::MatrixXd A(indices.size(), 3);
        Eigen::VectorXd b(indices.size());
        for (size_t i = 0; i < indices.size(); ++i)
        {
            const auto &p = cloud->points[indices[i]];
            A(static_cast<int>(i), 0) = 2.0 * p.x;
            A(static_cast<int>(i), 1) = 2.0 * p.y;
            A(static_cast<int>(i), 2) = 1.0;
            b(static_cast<int>(i)) = p.x * p.x + p.y * p.y;
        }

        Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);
        const double cx = x(0);
        const double cy = x(1);
        const double c = x(2);
        const double r2 = c + cx * cx + cy * cy;
        if (!std::isfinite(r2) || r2 <= 0.0)
        {
            return false;
        }

        candidate.center.x = static_cast<float>(cx);
        candidate.center.y = static_cast<float>(cy);
        candidate.center.z = 0.0f;
        candidate.radius = std::sqrt(r2);
        candidate.inliers = static_cast<int>(indices.size());
        return std::isfinite(candidate.radius);
    }

    std::vector<CircleCandidate> detectCircleCandidates(const pcl::PointCloud<pcl::PointXYZ>::Ptr &xy_cloud) const
    {
        constexpr int kMaxCandidates = 40;
        constexpr int kMaxIterations = 6000;
        constexpr int kMinInliers = 5;
        const double radius_min = circle_radius_ - 0.03;
        const double radius_max = circle_radius_ + 0.03;
        const double inlier_threshold = 0.02;

        std::vector<CircleCandidate> candidates;
        pcl::PointCloud<pcl::PointXYZ>::Ptr work(new pcl::PointCloud<pcl::PointXYZ>(*xy_cloud));
        std::mt19937 rng(101);

        for (int candidate_idx = 0; candidate_idx < kMaxCandidates && work->points.size() > 3; ++candidate_idx)
        {
            std::uniform_int_distribution<int> sample_dist(0, static_cast<int>(work->points.size()) - 1);
            std::vector<int> best_inliers;
            double best_cx = 0.0;
            double best_cy = 0.0;
            double best_radius = 0.0;

            for (int iter = 0; iter < kMaxIterations; ++iter)
            {
                const int i1 = sample_dist(rng);
                const int i2 = sample_dist(rng);
                const int i3 = sample_dist(rng);
                if (i1 == i2 || i1 == i3 || i2 == i3)
                {
                    continue;
                }

                double cx = 0.0;
                double cy = 0.0;
                double radius = 0.0;
                if (!fitCircleFromThree(work->points[i1], work->points[i2], work->points[i3], cx, cy, radius))
                {
                    continue;
                }
                if (radius < radius_min || radius > radius_max)
                {
                    continue;
                }

                std::vector<int> inliers;
                inliers.reserve(work->points.size());
                for (int i = 0; i < static_cast<int>(work->points.size()); ++i)
                {
                    const auto &p = work->points[i];
                    const double residual = std::fabs(std::hypot(static_cast<double>(p.x) - cx,
                                                                  static_cast<double>(p.y) - cy) -
                                                       radius);
                    if (residual < inlier_threshold)
                    {
                        inliers.push_back(i);
                    }
                }

                if (inliers.size() > best_inliers.size())
                {
                    best_inliers.swap(inliers);
                    best_cx = cx;
                    best_cy = cy;
                    best_radius = radius;
                }
            }

            if (static_cast<int>(best_inliers.size()) < kMinInliers)
            {
                break;
            }

            CircleCandidate candidate;
            candidate.center.x = static_cast<float>(best_cx);
            candidate.center.y = static_cast<float>(best_cy);
            candidate.center.z = 0.0f;
            candidate.radius = best_radius;
            candidate.inliers = static_cast<int>(best_inliers.size());
            refineCircleLeastSquares(work, best_inliers, candidate);

            if (candidate.radius >= radius_min && candidate.radius <= radius_max)
            {
                candidates.push_back(candidate);
            }

            std::vector<bool> remove(work->points.size(), false);
            for (const int idx : best_inliers)
            {
                remove[idx] = true;
            }

            pcl::PointCloud<pcl::PointXYZ>::Ptr remaining(new pcl::PointCloud<pcl::PointXYZ>);
            remaining->reserve(work->points.size() - best_inliers.size());
            for (int i = 0; i < static_cast<int>(work->points.size()); ++i)
            {
                if (!remove[i])
                {
                    remaining->push_back(work->points[i]);
                }
            }
            work.swap(remaining);
        }

        return candidates;
    }

    double geometryScore(const std::vector<pcl::PointXYZ> &points, bool &geom_valid) const
    {
        geom_valid = false;
        if (points.size() != TARGET_NUM_CIRCLES)
        {
            return std::numeric_limits<double>::infinity();
        }

        pcl::PointXYZ center;
        center.x = center.y = center.z = 0.0f;
        for (const auto &p : points)
        {
            center.x += p.x;
            center.y += p.y;
            center.z += p.z;
        }
        center.x /= TARGET_NUM_CIRCLES;
        center.y /= TARGET_NUM_CIRCLES;
        center.z /= TARGET_NUM_CIRCLES;

        std::vector<int> order(TARGET_NUM_CIRCLES);
        for (int i = 0; i < TARGET_NUM_CIRCLES; ++i)
        {
            order[i] = i;
        }
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return std::atan2(points[a].y - center.y, points[a].x - center.x) <
                   std::atan2(points[b].y - center.y, points[b].x - center.x);
        });

        double sides[TARGET_NUM_CIRCLES];
        for (int i = 0; i < TARGET_NUM_CIRCLES; ++i)
        {
            const auto &p1 = points[order[i]];
            const auto &p2 = points[order[(i + 1) % TARGET_NUM_CIRCLES]];
            sides[i] = std::hypot(static_cast<double>(p1.x) - static_cast<double>(p2.x),
                                  static_cast<double>(p1.y) - static_cast<double>(p2.y));
        }

        const double pattern1[TARGET_NUM_CIRCLES] = {delta_width_circles_, delta_height_circles_,
                                                     delta_width_circles_, delta_height_circles_};
        const double pattern2[TARGET_NUM_CIRCLES] = {delta_height_circles_, delta_width_circles_,
                                                     delta_height_circles_, delta_width_circles_};

        auto pattern_score = [&](const double pattern[TARGET_NUM_CIRCLES], bool &pattern_valid) {
            double score = 0.0;
            pattern_valid = true;
            for (int i = 0; i < TARGET_NUM_CIRCLES; ++i)
            {
                const double rel = std::fabs(sides[i] - pattern[i]) / pattern[i];
                score += rel * rel;
                if (rel > 0.35)
                {
                    pattern_valid = false;
                }
            }
            return score;
        };

        bool valid1 = false;
        bool valid2 = false;
        const double score1 = pattern_score(pattern1, valid1);
        const double score2 = pattern_score(pattern2, valid2);
        const double perimeter = sides[0] + sides[1] + sides[2] + sides[3];
        const double ideal_perimeter = 2.0 * (delta_width_circles_ + delta_height_circles_);
        const double perimeter_error = std::fabs(perimeter - ideal_perimeter) / ideal_perimeter;
        const double target_radius = std::sqrt(delta_width_circles_ * delta_width_circles_ +
                                               delta_height_circles_ * delta_height_circles_) /
                                     2.0;

        double center_radius_error = 0.0;
        for (const auto &p : points)
        {
            const double d = std::hypot(static_cast<double>(p.x) - center.x,
                                        static_cast<double>(p.y) - center.y);
            center_radius_error += std::fabs(d - target_radius);
        }
        center_radius_error /= TARGET_NUM_CIRCLES;

        geom_valid = (valid1 || valid2) && perimeter_error < 0.35;
        return std::min(score1, score2) + perimeter_error + center_radius_error;
    }

    CandidateScore selectBestCandidateGroup(const std::vector<CircleCandidate> &candidates,
                                            const Eigen::Matrix3d &R_inv,
                                            double average_z,
                                            const pcl::PointCloud<pcl::PointXYZ>::Ptr &qr_reference) const
    {
        CandidateScore best;
        if (candidates.size() < TARGET_NUM_CIRCLES)
        {
            return best;
        }

        std::vector<std::vector<int>> groups;
        comb(static_cast<int>(candidates.size()), TARGET_NUM_CIRCLES, groups);
        const bool use_qr = qr_reference && qr_reference->size() == TARGET_NUM_CIRCLES;

        pcl::PointCloud<pcl::PointXYZ>::Ptr sorted_qr(new pcl::PointCloud<pcl::PointXYZ>);
        if (use_qr)
        {
            sortPatternCenters(qr_reference, sorted_qr, "camera");
        }

        for (const auto &group : groups)
        {
            std::vector<pcl::PointXYZ> candidate_points;
            candidate_points.reserve(TARGET_NUM_CIRCLES);
            int support = 0;
            for (const int idx : group)
            {
                candidate_points.push_back(candidates[idx].center);
                support += candidates[idx].inliers;
            }

            bool geom_valid = false;
            const double geom_score = geometryScore(candidate_points, geom_valid);
            double rmse = std::numeric_limits<double>::infinity();

            if (use_qr)
            {
                pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_candidates(new pcl::PointCloud<pcl::PointXYZ>);
                lidar_candidates->reserve(TARGET_NUM_CIRCLES);
                for (const auto &p : candidate_points)
                {
                    Eigen::Vector3d aligned_point(p.x, p.y, average_z);
                    Eigen::Vector3d original_point = R_inv * aligned_point;
                    lidar_candidates->push_back(
                        pcl::PointXYZ(original_point.x(), original_point.y(), original_point.z()));
                }

                pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_sorted(new pcl::PointCloud<pcl::PointXYZ>);
                sortPatternCenters(lidar_candidates, lidar_sorted, "lidar");

                Eigen::Matrix4f transformation;
                pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
                svd.estimateRigidTransformation(*lidar_sorted, *sorted_qr, transformation);

                pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_lidar(new pcl::PointCloud<pcl::PointXYZ>);
                aligned_lidar->reserve(lidar_sorted->size());
                alignPointCloud(lidar_sorted, aligned_lidar, transformation);
                rmse = computeRMSE(sorted_qr, aligned_lidar);
            }

            bool better = false;
            if (!best.found)
            {
                better = true;
            }
            else if (use_qr)
            {
                if (rmse < best.rmse - 1e-6)
                {
                    better = true;
                }
                else if (std::fabs(rmse - best.rmse) <= 1e-6)
                {
                    if (geom_valid != best.geom_valid)
                    {
                        better = geom_valid;
                    }
                    else if (geom_score < best.geom_score - 1e-6)
                    {
                        better = true;
                    }
                    else if (std::fabs(geom_score - best.geom_score) <= 1e-6 && support > best.support)
                    {
                        better = true;
                    }
                }
            }
            else
            {
                if (geom_valid != best.geom_valid)
                {
                    better = geom_valid;
                }
                else if (geom_score < best.geom_score - 1e-6)
                {
                    better = true;
                }
                else if (std::fabs(geom_score - best.geom_score) <= 1e-6 && support > best.support)
                {
                    better = true;
                }
            }

            if (better)
            {
                best.found = true;
                best.geom_valid = geom_valid;
                best.geom_score = geom_score;
                best.rmse = rmse;
                best.support = support;
                best.group = group;
            }
        }

        return best;
    }

public:
    ros::Publisher filtered_pub_;
    ros::Publisher plane_pub_;
    ros::Publisher aligned_pub_;
    ros::Publisher edge_pub_;
    ros::Publisher center_z0_pub_;
    ros::Publisher center_pub_;

    LidarDetect(ros::NodeHandle &nh, Params &params)
        : filtered_cloud_(new pcl::PointCloud<Common::Point>),
          plane_cloud_(new pcl::PointCloud<Common::Point>),
          aligned_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          edge_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          center_z0_cloud_(new pcl::PointCloud<pcl::PointXYZ>)
    {
        x_min_ = params.x_min;
        x_max_ = params.x_max;
        y_min_ = params.y_min;
        y_max_ = params.y_max;
        z_min_ = params.z_min;
        z_max_ = params.z_max;
        circle_radius_ = params.circle_radius;
        delta_width_circles_ = params.delta_width_circles;
        delta_height_circles_ = params.delta_height_circles;

        filtered_pub_ = nh.advertise<sensor_msgs::PointCloud2>("filtered_cloud", 1);
        plane_pub_ = nh.advertise<sensor_msgs::PointCloud2>("plane_cloud", 1);
        aligned_pub_ = nh.advertise<sensor_msgs::PointCloud2>("aligned_cloud", 1);
        edge_pub_ = nh.advertise<sensor_msgs::PointCloud2>("edge_cloud", 1);
        center_z0_pub_ = nh.advertise<sensor_msgs::PointCloud2>("center_z0_cloud", 10);
        center_pub_ = nh.advertise<sensor_msgs::PointCloud2>("center_cloud", 10);
    }

    void detect_mech_lidar(pcl::PointCloud<Common::Point>::Ptr cloud,
                           pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud,
                           const pcl::PointCloud<pcl::PointXYZ>::Ptr &qr_reference = pcl::PointCloud<pcl::PointXYZ>::Ptr())
    {
        filtered_cloud_->clear();
        plane_cloud_->clear();
        aligned_cloud_->clear();
        edge_cloud_->clear();
        center_z0_cloud_->clear();
        center_cloud->clear();

        // 1. X、Y、Z方向滤波
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<Common::Point> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);
        pass_x.filter(*filtered_cloud_);

        pcl::PassThrough<Common::Point> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);
        pass_y.filter(*filtered_cloud_);

        pcl::PassThrough<Common::Point> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);
        pass_z.filter(*filtered_cloud_);

        ROS_INFO("Depth filtered cloud size: %zu", filtered_cloud_->size());

        // 2. 拟合平面，提取法向量
        plane_cloud_->reserve(filtered_cloud_->size());

        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<Common::Point> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(0.01);
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);

        pcl::ExtractIndices<Common::Point> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        ROS_INFO("Plane cloud size: %zu", plane_cloud_->size());

        if (plane_coefficients->values.size() < 4 || plane_cloud_->empty())
        {
            ROS_WARN("[LiDAR] Plane fitting failed, skip mechanical LiDAR detection.");
            return;
        }

        // 3. 根据每条 ring 相邻点距离提取边缘点
        edge_cloud_->reserve(filtered_cloud_->size());
        std::unordered_map<unsigned int, std::vector<int>> ring2indices;
        ring2indices.reserve(64);
        for (int i = 0; i < static_cast<int>(filtered_cloud_->size()); ++i)
        {
            const auto &pt = filtered_cloud_->points[i];
            ring2indices[pt.ring].push_back(i);
        }

        const auto &c = plane_coefficients->values;
        Eigen::Vector3d n(c[0], c[1], c[2]);
        const double norm_n = n.norm();
        if (norm_n < 1e-9)
        {
            ROS_WARN("[LiDAR] Invalid plane normal, skip mechanical LiDAR detection.");
            return;
        }
        Eigen::Vector3d normal = n / norm_n;

        const double neighbor_gap_threshold = 0.10;
        const int min_points_per_ring = 10;
        for (auto &kv : ring2indices)
        {
            auto &idx_vec = kv.second;
            if (static_cast<int>(idx_vec.size()) < min_points_per_ring)
            {
                continue;
            }

            for (size_t k = 1; k + 1 < idx_vec.size(); ++k)
            {
                const auto &p_prev = filtered_cloud_->points[idx_vec[k - 1]];
                const auto &p_cur = filtered_cloud_->points[idx_vec[k]];
                const auto &p_next = filtered_cloud_->points[idx_vec[k + 1]];

                const double dist_plane = std::fabs(c[0] * p_cur.x + c[1] * p_cur.y + c[2] * p_cur.z + c[3]) / norm_n;
                if (dist_plane >= 0.03)
                {
                    continue;
                }

                const double dx1 = static_cast<double>(p_cur.x) - static_cast<double>(p_prev.x);
                const double dy1 = static_cast<double>(p_cur.y) - static_cast<double>(p_prev.y);
                const double dz1 = static_cast<double>(p_cur.z) - static_cast<double>(p_prev.z);
                const double dist_prev = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);

                const double dx2 = static_cast<double>(p_cur.x) - static_cast<double>(p_next.x);
                const double dy2 = static_cast<double>(p_cur.y) - static_cast<double>(p_next.y);
                const double dz2 = static_cast<double>(p_cur.z) - static_cast<double>(p_next.z);
                const double dist_next = std::sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

                if (dist_prev > neighbor_gap_threshold || dist_next > neighbor_gap_threshold)
                {
                    edge_cloud_->push_back(pcl::PointXYZ(p_cur.x, p_cur.y, p_cur.z));
                }
            }
        }

        ROS_INFO("Extracted %zu edge points (mechanical LiDAR by neighbor distance).", edge_cloud_->size());
        if (edge_cloud_->empty())
        {
            ROS_WARN("[LiDAR] No edge points found, skip.");
            return;
        }

        // 4. 将边缘点对齐到 Z=0 平面
        aligned_cloud_->reserve(edge_cloud_->size());
        Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
        Eigen::Vector3d axis = normal.cross(z_axis);
        const double dot_nz = std::max(-1.0, std::min(1.0, normal.dot(z_axis)));
        Eigen::Matrix3d R_align = Eigen::Matrix3d::Identity();
        if (axis.norm() < 1e-9)
        {
            if (dot_nz < 0.0)
            {
                R_align = Eigen::AngleAxisd(M_PI, Eigen::Vector3d(1.0, 0.0, 0.0)).toRotationMatrix();
            }
        }
        else
        {
            axis.normalize();
            const double angle = std::acos(dot_nz);
            Eigen::AngleAxisd rotation(angle, axis);
            R_align = rotation.toRotationMatrix();
        }

        float average_z = 0.0f;
        int cnt = 0;
        for (const auto &pt : *edge_cloud_)
        {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R_align * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += static_cast<float>(aligned_point.z());
            ++cnt;
        }
        average_z /= std::max(cnt, 1);

        // 5. 在对齐后的点云中搜索圆候选
        pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud(new pcl::PointCloud<pcl::PointXYZ>(*aligned_cloud_));
        std::vector<CircleCandidate> circle_candidates = detectCircleCandidates(xy_cloud);
        for (const auto &candidate : circle_candidates)
        {
            center_z0_cloud_->push_back(candidate.center);
        }

        ROS_INFO("[LiDAR] Circle candidates found: %zu", circle_candidates.size());
        for (size_t i = 0; i < circle_candidates.size(); ++i)
        {
            ROS_INFO("[LiDAR]   candidate %zu: center=(%.4f, %.4f), r=%.4f, inliers=%d",
                     i,
                     circle_candidates[i].center.x,
                     circle_candidates[i].center.y,
                     circle_candidates[i].radius,
                     circle_candidates[i].inliers);
        }

        Eigen::Matrix3d R_inv = R_align.inverse();
        CandidateScore best_candidate = selectBestCandidateGroup(circle_candidates, R_inv, average_z, qr_reference);
        if (!best_candidate.found)
        {
            ROS_WARN("[LiDAR] Unable to find a candidate set that matches target's geometry");
            return;
        }

        if (qr_reference && qr_reference->size() == TARGET_NUM_CIRCLES)
        {
            ROS_INFO("[LiDAR] Selected circle group by QR consistency: rmse=%.4f, geom_score=%.4f, geom_valid=%s, support=%d",
                     best_candidate.rmse,
                     best_candidate.geom_score,
                     best_candidate.geom_valid ? "true" : "false",
                     best_candidate.support);
        }
        else
        {
            ROS_INFO("[LiDAR] Selected circle group by geometry: geom_score=%.4f, geom_valid=%s, support=%d",
                     best_candidate.geom_score,
                     best_candidate.geom_valid ? "true" : "false",
                     best_candidate.support);
        }

        // 6. 将选中的圆心逆变换回原始坐标系
        for (const int idx : best_candidate.group)
        {
            const pcl::PointXYZ &center = circle_candidates[idx].center;
            Eigen::Vector3d aligned_point(center.x, center.y, center.z + average_z);
            Eigen::Vector3d original_point = R_inv * aligned_point;

            pcl::PointXYZ center_point_origin;
            center_point_origin.x = original_point.x();
            center_point_origin.y = original_point.y();
            center_point_origin.z = original_point.z();
            center_cloud->points.push_back(center_point_origin);
        }
    }

    void detect_solid_lidar(pcl::PointCloud<Common::Point>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        // 1. X、Y、Z方向滤波
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<Common::Point> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);  // 设置X轴范围
        pass_x.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);  // 设置Y轴范围
        pass_y.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);  // 设置Z轴范围
        pass_z.filter(*filtered_cloud_);
    
        ROS_INFO("Filtered cloud size: %zu", filtered_cloud_->size());
        if (filtered_cloud_->empty())
        {
            ROS_WARN("[LiDAR] Filtered cloud is empty, skip solid LiDAR detection.");
            return;
        }
        
        pcl::VoxelGrid<Common::Point> voxel_filter;
        voxel_filter.setInputCloud(filtered_cloud_);
        voxel_filter.setLeafSize(0.005f, 0.005f, 0.005f);
        voxel_filter.filter(*filtered_cloud_);
        ROS_INFO("Filtered cloud size: %zu", filtered_cloud_->size());
        if (filtered_cloud_->empty())
        {
            ROS_WARN("[LiDAR] Filtered cloud is empty after voxel filtering, skip solid LiDAR detection.");
            return;
        }

        // 2. 平面分割
        plane_cloud_->reserve(filtered_cloud_->size());

        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<Common::Point> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(0.01);  // 平面分割阈值
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);
    
        if (plane_coefficients->values.size() < 4 || plane_inliers->indices.empty())
        {
            ROS_WARN("[LiDAR] Plane fitting failed, skip solid LiDAR detection.");
            return;
        }

        pcl::ExtractIndices<Common::Point> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        ROS_INFO("Plane cloud size: %zu", plane_cloud_->size());
        if (plane_cloud_->empty())
        {
            ROS_WARN("[LiDAR] Plane cloud is empty, skip solid LiDAR detection.");
            return;
        }
    
        // 3. 平面点云对齐   
        aligned_cloud_->reserve(plane_cloud_->size());

        Eigen::Vector3d normal(plane_coefficients->values[0],
            plane_coefficients->values[1],
            plane_coefficients->values[2]);
        normal.normalize();
        Eigen::Vector3d z_axis(0, 0, 1);

        Eigen::Vector3d axis = normal.cross(z_axis);
        double angle = acos(normal.dot(z_axis));

        Eigen::AngleAxisd rotation(angle, axis);
        Eigen::Matrix3d R = rotation.toRotationMatrix();

        // 应用旋转矩阵，将平面对齐到 Z=0 平面
        float average_z = 0.0;
        int cnt = 0;
        for (const auto& pt : *plane_cloud_) {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }
        average_z /= cnt;

        // 4. 提取边缘点
        edge_cloud_->reserve(aligned_cloud_->size());

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        normal_estimator.setInputCloud(aligned_cloud_);
        normal_estimator.setRadiusSearch(0.03); // 设置法线估计的搜索半径
        normal_estimator.compute(*normals);
    
        pcl::PointCloud<pcl::Boundary> boundaries;
        pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimator;
        boundary_estimator.setInputCloud(aligned_cloud_);
        boundary_estimator.setInputNormals(normals);
        boundary_estimator.setRadiusSearch(0.03); // 设置边界检测的搜索半径
        boundary_estimator.setAngleThreshold(M_PI / 4); // 设置角度阈值
        boundary_estimator.compute(boundaries);
    
        for (size_t i = 0; i < aligned_cloud_->size(); ++i) {
            if (boundaries.points[i].boundary_point > 0) {
                edge_cloud_->push_back(aligned_cloud_->points[i]);
            }
        }
        ROS_INFO("Extracted %zu edge points.", edge_cloud_->size());

        // 5. 对边缘点进行聚类
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(edge_cloud_);
    
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.05); // 设置聚类距离阈值
        ec.setMinClusterSize(50);     // 最小点数
        ec.setMaxClusterSize(1000);   // 最大点数
        ec.setSearchMethod(tree);
        ec.setInputCloud(edge_cloud_);
        ec.extract(cluster_indices);
    
        ROS_INFO("Number of edge clusters: %zu", cluster_indices.size());
    
        // 6. 对每个聚类进行圆拟合
        center_z0_cloud_->reserve(4);
        Eigen::Matrix3d R_inv = R.inverse();
    
        // 对每个聚类进行圆拟合
        for (size_t i = 0; i < cluster_indices.size(); ++i) 
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster_indices[i].indices) {
                cluster->push_back(edge_cloud_->points[idx]);
            }
    
            // 圆拟合
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_CIRCLE2D);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.01); // 设置距离阈值
            seg.setMaxIterations(1000);     // 设置最大迭代次数
            seg.setInputCloud(cluster);
            seg.segment(*inliers, *coefficients);
    
            if (inliers->indices.size() > 0) 
            {
                if (coefficients->values.size() < 3)
                {
                    ROS_INFO("[LiDAR] Edge cluster %zu: size=%zu, inliers=%zu, invalid circle coefficients.",
                             i, cluster->size(), inliers->indices.size());
                    continue;
                }

                // 计算拟合误差
                double error = 0.0;
                for (const auto& idx : inliers->indices) 
                {
                    double dx = cluster->points[idx].x - coefficients->values[0];
                    double dy = cluster->points[idx].y - coefficients->values[1];
                    double distance = sqrt(dx * dx + dy * dy) - circle_radius_; // 距离误差
                    error += abs(distance);
                }
                error /= inliers->indices.size();
                ROS_INFO("[LiDAR] Edge cluster %zu: size=%zu, inliers=%zu, fitted_radius=%.4f, radius_error=%.4f",
                         i,
                         cluster->size(),
                         inliers->indices.size(),
                         coefficients->values[2],
                         error);
    
                // 如果拟合误差较小，则认为是一个圆洞
                if (error < 0.025) 
                {
                    // 将恢复后的圆心坐标添加到点云中
                    pcl::PointXYZ center_point;
                    center_point.x = coefficients->values[0];
                    center_point.y = coefficients->values[1];
                    center_point.z = 0.0;
                    center_z0_cloud_->push_back(center_point);

                    // 将圆心坐标逆变换回原始坐标系
                    Eigen::Vector3d aligned_point(center_point.x, center_point.y, center_point.z + average_z);
                    Eigen::Vector3d original_point = R_inv * aligned_point;

                    pcl::PointXYZ center_point_origin;
                    center_point_origin.x = original_point.x();
                    center_point_origin.y = original_point.y();
                    center_point_origin.z = original_point.z();
                    center_cloud->points.push_back(center_point_origin);
                }
            }
        }
    }
    // 获取中间结果的点云
    pcl::PointCloud<Common::Point>::Ptr getFilteredCloud() const { return filtered_cloud_; }
    pcl::PointCloud<Common::Point>::Ptr getPlaneCloud() const { return plane_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getAlignedCloud() const { return aligned_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getEdgeCloud() const { return edge_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getCenterZ0Cloud() const { return center_z0_cloud_; }
};

typedef std::shared_ptr<LidarDetect> LidarDetectPtr;

#endif
