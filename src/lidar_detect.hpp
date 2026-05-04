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
#include <array>
#include <algorithm>
#include <cctype>
#include <limits>
#include <random>
#include <unordered_map>
#include "common_lib.h"

class LidarDetect
{
private:
    double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
    double circle_radius_, delta_width_circles_, delta_height_circles_;
    bool airy_hole_detector_;
    double airy_boundary_radius_;
    double airy_boundary_min_angular_gap_;
    int airy_boundary_min_neighbors_;
    bool airy_template_detector_;
    double airy_template_grid_;
    double airy_template_angle_step_deg_;
    double airy_template_ring_band_;
    double airy_template_min_score_;
    std::string lidar_center_extraction_mode_;
    bool lidar_strict_geometry_;
    double lidar_geometry_side_rel_tol_;
    double lidar_geometry_diag_rel_tol_;
    double lidar_geometry_perimeter_rel_tol_;
    int lidar_min_plane_points_;
    int lidar_ransac_min_inliers_;
    double lidar_ransac_radius_tolerance_;
    double lidar_ransac_inlier_threshold_;
    int lidar_template_min_ring_per_hole_;
    int lidar_template_min_outer_per_hole_;
    int lidar_template_max_inside_per_hole_;
    double lidar_template_max_inside_support_ratio_;
    double lidar_template_min_support_per_hole_;
    double lidar_template_local_refine_radius_;
    double lidar_template_local_refine_step_;

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
        double selection_score = std::numeric_limits<double>::infinity();
    };

    struct HoleStats
    {
        int inside = 0;
        int ring = 0;
        int outer = 0;
        int far_outer = 0;
        double support = 0.0;
        double inside_support_ratio = std::numeric_limits<double>::infinity();
        double score = -std::numeric_limits<double>::infinity();
    };

    struct TemplateEval
    {
        bool feasible = false;
        double score = -std::numeric_limits<double>::infinity();
        std::vector<pcl::PointXYZ> centers;
        std::array<HoleStats, TARGET_NUM_CIRCLES> holes;
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

    static std::string lowerString(std::string value)
    {
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return value;
    }

    bool wantsTemplate() const
    {
        return lidar_center_extraction_mode_ == "auto" ||
               lidar_center_extraction_mode_ == "template";
    }

    bool wantsRansac() const
    {
        return lidar_center_extraction_mode_ == "auto" ||
               lidar_center_extraction_mode_ == "ransac" ||
               lidar_center_extraction_mode_ == "legacy";
    }

    bool allowRansacFallback() const
    {
        return lidar_center_extraction_mode_ == "auto";
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
        const int kMinInliers = std::max(3, lidar_ransac_min_inliers_);
        const double radius_tol = std::max(0.005, lidar_ransac_radius_tolerance_);
        const double radius_min = circle_radius_ - radius_tol;
        const double radius_max = circle_radius_ + radius_tol;
        const double inlier_threshold = std::max(0.003, lidar_ransac_inlier_threshold_);

        std::vector<CircleCandidate> candidates;
        pcl::PointCloud<pcl::PointXYZ>::Ptr work(new pcl::PointCloud<pcl::PointXYZ>(*xy_cloud));
        std::mt19937 rng(101);
        ROS_INFO("[LiDAR][RANSAC] params: min_inliers=%d radius_tol=%.4f inlier_threshold=%.4f input_points=%zu",
                 kMinInliers, radius_tol, inlier_threshold, xy_cloud ? xy_cloud->size() : 0);

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

    bool validateTargetGeometryPoints(const std::vector<pcl::PointXYZ> &points,
                                      const std::string &tag,
                                      bool warn = true) const
    {
        if (points.size() != TARGET_NUM_CIRCLES)
        {
            if (warn)
            {
                ROS_WARN("[LiDAR][Geometry] %s reject: expected 4 centers, got %zu",
                         tag.c_str(), points.size());
            }
            return false;
        }

        std::vector<double> dists;
        dists.reserve(6);
        for (int i = 0; i < TARGET_NUM_CIRCLES; ++i)
        {
            for (int j = i + 1; j < TARGET_NUM_CIRCLES; ++j)
            {
                const double dx = static_cast<double>(points[i].x) - static_cast<double>(points[j].x);
                const double dy = static_cast<double>(points[i].y) - static_cast<double>(points[j].y);
                const double dz = static_cast<double>(points[i].z) - static_cast<double>(points[j].z);
                dists.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
            }
        }
        std::sort(dists.begin(), dists.end());

        const double short_side = std::min(delta_width_circles_, delta_height_circles_);
        const double long_side = std::max(delta_width_circles_, delta_height_circles_);
        const double diagonal = std::sqrt(delta_width_circles_ * delta_width_circles_ +
                                          delta_height_circles_ * delta_height_circles_);
        const std::array<double, 6> expected = {
            short_side, short_side, long_side, long_side, diagonal, diagonal};

        bool ok = true;
        for (int i = 0; i < 6; ++i)
        {
            const double tol = (i < 4) ? lidar_geometry_side_rel_tol_ : lidar_geometry_diag_rel_tol_;
            const double rel = expected[i] > 1e-9 ? std::fabs(dists[i] - expected[i]) / expected[i]
                                                  : std::numeric_limits<double>::infinity();
            if (rel > tol)
            {
                if (warn)
                {
                    ROS_WARN("[LiDAR][Geometry] %s reject: sorted_dist[%d]=%.4f expected=%.4f rel=%.3f tol=%.3f",
                             tag.c_str(), i, dists[i], expected[i], rel, tol);
                }
                ok = false;
            }
        }

        const double side_sum = dists[0] + dists[1] + dists[2] + dists[3];
        const double expected_perimeter = 2.0 * (delta_width_circles_ + delta_height_circles_);
        const double perimeter_rel = expected_perimeter > 1e-9
                                         ? std::fabs(side_sum - expected_perimeter) / expected_perimeter
                                         : std::numeric_limits<double>::infinity();
        if (perimeter_rel > lidar_geometry_perimeter_rel_tol_)
        {
            if (warn)
            {
                ROS_WARN("[LiDAR][Geometry] %s reject: side_perimeter=%.4f expected=%.4f rel=%.3f tol=%.3f",
                         tag.c_str(), side_sum, expected_perimeter, perimeter_rel,
                         lidar_geometry_perimeter_rel_tol_);
            }
            ok = false;
        }

        if (ok && warn)
        {
            ROS_INFO("[LiDAR][Geometry] %s pass: distances=[%.4f %.4f %.4f %.4f %.4f %.4f]",
                     tag.c_str(), dists[0], dists[1], dists[2], dists[3], dists[4], dists[5]);
        }
        return ok;
    }

    bool validateTargetGeometry3D(const pcl::PointCloud<pcl::PointXYZ>::Ptr &centers,
                                  const std::string &tag) const
    {
        if (!centers)
        {
            ROS_WARN("[LiDAR][Geometry] %s reject: null center cloud", tag.c_str());
            return false;
        }
        std::vector<pcl::PointXYZ> pts;
        pts.reserve(centers->size());
        for (const auto &p : centers->points)
        {
            pts.push_back(p);
        }
        return validateTargetGeometryPoints(pts, tag, true);
    }

    bool acceptFinalGeometry(const pcl::PointCloud<pcl::PointXYZ>::Ptr &centers,
                             const std::string &tag) const
    {
        const bool ok = validateTargetGeometry3D(centers, tag);
        if (!ok && lidar_strict_geometry_)
        {
            ROS_WARN("[LiDAR][Geometry] %s final geometry failed; strict mode rejects centers.", tag.c_str());
            return false;
        }
        if (!ok)
        {
            ROS_WARN("[LiDAR][Geometry] %s final geometry failed; strict mode disabled, accepting centers.", tag.c_str());
        }
        return true;
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
                if (rel > lidar_geometry_side_rel_tol_)
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
        const double diag_expected = std::sqrt(delta_width_circles_ * delta_width_circles_ +
                                               delta_height_circles_ * delta_height_circles_);
        const auto &d0 = points[order[0]];
        const auto &d1 = points[order[1]];
        const auto &d2 = points[order[2]];
        const auto &d3 = points[order[3]];
        const double diag1 = std::hypot(static_cast<double>(d0.x) - static_cast<double>(d2.x),
                                        static_cast<double>(d0.y) - static_cast<double>(d2.y));
        const double diag2 = std::hypot(static_cast<double>(d1.x) - static_cast<double>(d3.x),
                                        static_cast<double>(d1.y) - static_cast<double>(d3.y));
        const double diag_error = std::max(std::fabs(diag1 - diag_expected) / diag_expected,
                                           std::fabs(diag2 - diag_expected) / diag_expected);
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

        geom_valid = (valid1 || valid2) &&
                     perimeter_error < lidar_geometry_perimeter_rel_tol_ &&
                     diag_error < lidar_geometry_diag_rel_tol_ &&
                     validateTargetGeometryPoints(points, "candidate_2d", false);
        return std::min(score1, score2) + perimeter_error + diag_error + center_radius_error;
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
            if (!geom_valid)
            {
                continue;
            }

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

            const double rmse_aux = (use_qr && std::isfinite(rmse)) ? 0.05 * rmse : 0.0;
            const double selection_score = geom_score + rmse_aux - 1e-4 * static_cast<double>(support);
            bool better = false;
            if (!best.found)
            {
                better = true;
            }
            else
            {
                if (geom_valid != best.geom_valid)
                {
                    better = geom_valid;
                }
                else if (selection_score < best.selection_score - 1e-6)
                {
                    better = true;
                }
                else if (std::fabs(selection_score - best.selection_score) <= 1e-6 &&
                         geom_score < best.geom_score - 1e-6)
                {
                    better = true;
                }
                else if (std::fabs(selection_score - best.selection_score) <= 1e-6 &&
                         std::fabs(geom_score - best.geom_score) <= 1e-6 &&
                         support > best.support)
                {
                    better = true;
                }
                else if (use_qr &&
                         std::fabs(selection_score - best.selection_score) <= 1e-6 &&
                         std::fabs(geom_score - best.geom_score) <= 1e-6 &&
                         support == best.support &&
                         rmse < best.rmse - 1e-6)
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
                best.selection_score = selection_score;
            }
        }

        return best;
    }

    std::array<Eigen::Vector2d, TARGET_NUM_CIRCLES> templateOffsets() const
    {
        const double half_w = 0.5 * delta_width_circles_;
        const double half_h = 0.5 * delta_height_circles_;
        return {
            Eigen::Vector2d(-half_w, -half_h),
            Eigen::Vector2d( half_w, -half_h),
            Eigen::Vector2d(-half_w,  half_h),
            Eigen::Vector2d( half_w,  half_h),
        };
    }

    HoleStats computeHoleStats(const pcl::PointCloud<pcl::PointXYZ>::Ptr &aligned_plane,
                               double hx,
                               double hy) const
    {
        const double r = circle_radius_;
        const double band = std::max(0.005, airy_template_ring_band_);
        const double inner = r * 0.70;
        const double outer_min = r + 0.030;
        const double outer_max = r + 0.180;
        const double far_outer_min = r + 0.180;
        const double far_outer_max = r + 0.280;

        HoleStats stats;
        for (const auto &p : aligned_plane->points)
        {
            const double dx = static_cast<double>(p.x) - hx;
            const double dy = static_cast<double>(p.y) - hy;
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d < inner)
            {
                ++stats.inside;
            }
            if (std::fabs(d - r) < band)
            {
                ++stats.ring;
            }
            if (d >= outer_min && d <= outer_max)
            {
                ++stats.outer;
            }
            if (d >= far_outer_min && d <= far_outer_max)
            {
                ++stats.far_outer;
            }
        }
        stats.support = static_cast<double>(stats.ring) +
                        0.5 * static_cast<double>(stats.outer) +
                        0.15 * static_cast<double>(stats.far_outer);
        stats.inside_support_ratio =
            static_cast<double>(stats.inside) /
            std::max(1.0, static_cast<double>(stats.ring + stats.outer));

        stats.score = 4.0 * stats.ring +
                      0.7 * stats.outer +
                      0.2 * stats.far_outer -
                      8.0 * stats.inside -
                      10.0 * stats.inside_support_ratio;
        return stats;
    }

    bool holeStatsFeasible(const HoleStats &stats) const
    {
        const bool enough_ring_or_outer =
            stats.ring >= lidar_template_min_ring_per_hole_ ||
            stats.outer >= std::max(2, 2 * lidar_template_min_outer_per_hole_);
        const bool enough_support =
            stats.support >= lidar_template_min_support_per_hole_;
        const bool empty_enough =
            stats.inside <= lidar_template_max_inside_per_hole_ &&
            stats.inside_support_ratio <= lidar_template_max_inside_support_ratio_;
        return enough_ring_or_outer && enough_support && empty_enough;
    }

    TemplateEval evaluateTemplate(const pcl::PointCloud<pcl::PointXYZ>::Ptr &aligned_plane,
                                  double cx,
                                  double cy,
                                  double theta) const
    {
        TemplateEval eval;
        eval.centers.reserve(TARGET_NUM_CIRCLES);
        const auto offsets = templateOffsets();
        const double ct = std::cos(theta);
        const double st = std::sin(theta);
        eval.score = 0.0;
        eval.feasible = true;

        for (int i = 0; i < TARGET_NUM_CIRCLES; ++i)
        {
            const auto &off = offsets[i];
            const double hx = cx + ct * off.x() - st * off.y();
            const double hy = cy + st * off.x() + ct * off.y();
            eval.centers.push_back(pcl::PointXYZ(hx, hy, 0.0f));
            eval.holes[i] = computeHoleStats(aligned_plane, hx, hy);
            eval.score += eval.holes[i].score;
            if (!holeStatsFeasible(eval.holes[i]))
            {
                eval.feasible = false;
            }
        }
        if (!validateTargetGeometryPoints(eval.centers, "template_candidate_2d", false))
        {
            eval.feasible = false;
        }
        return eval;
    }

    void logTemplateHoleStats(const std::string &prefix,
                              const std::array<HoleStats, TARGET_NUM_CIRCLES> &holes) const
    {
        for (int i = 0; i < TARGET_NUM_CIRCLES; ++i)
        {
            ROS_INFO("[Template] %s hole_%d: score=%.2f support=%.2f inside_ratio=%.3f inside=%d ring=%d outer=%d far_outer=%d",
                     prefix.c_str(), i, holes[i].score, holes[i].support,
                     holes[i].inside_support_ratio, holes[i].inside, holes[i].ring,
                     holes[i].outer, holes[i].far_outer);
        }
    }

    bool backProjectAlignedCenters(const std::vector<pcl::PointXYZ> &aligned_centers,
                                   const Eigen::Matrix3d &R_inv,
                                   double average_z,
                                   pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud) const
    {
        center_cloud->clear();
        center_z0_cloud_->clear();
        for (const auto &center : aligned_centers)
        {
            center_z0_cloud_->push_back(center);
            Eigen::Vector3d aligned_point(center.x, center.y, static_cast<double>(center.z) + average_z);
            Eigen::Vector3d original_point = R_inv * aligned_point;
            center_cloud->push_back(pcl::PointXYZ(original_point.x(),
                                                  original_point.y(),
                                                  original_point.z()));
        }
        return center_cloud->size() == TARGET_NUM_CIRCLES;
    }

    bool detectAiryTemplateCenters(const pcl::PointCloud<pcl::PointXYZ>::Ptr &aligned_plane,
                                   const Eigen::Matrix3d &R_inv,
                                   double average_z,
                                   pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud) const
    {
        if (!airy_template_detector_)
        {
            ROS_WARN("[Template] Reject: airy_template_detector=false.");
            return false;
        }
        if (!aligned_plane || aligned_plane->empty())
        {
            ROS_WARN("[Template] Reject: aligned plane is empty.");
            return false;
        }
        if (static_cast<int>(aligned_plane->size()) < lidar_min_plane_points_)
        {
            ROS_WARN("[Template] Reject: aligned plane has %zu points, need >= %d",
                     aligned_plane->size(), lidar_min_plane_points_);
            return false;
        }
        if (circle_radius_ <= 0.0 || delta_width_circles_ <= 0.0 || delta_height_circles_ <= 0.0)
        {
            ROS_WARN("[Template] Reject: invalid target geometry r=%.4f w=%.4f h=%.4f",
                     circle_radius_, delta_width_circles_, delta_height_circles_);
            return false;
        }

        double min_x = std::numeric_limits<double>::infinity();
        double min_y = std::numeric_limits<double>::infinity();
        double max_x = -std::numeric_limits<double>::infinity();
        double max_y = -std::numeric_limits<double>::infinity();
        for (const auto &p : aligned_plane->points)
        {
            min_x = std::min(min_x, static_cast<double>(p.x));
            min_y = std::min(min_y, static_cast<double>(p.y));
            max_x = std::max(max_x, static_cast<double>(p.x));
            max_y = std::max(max_y, static_cast<double>(p.y));
        }

        const double half_w = 0.5 * delta_width_circles_;
        const double half_h = 0.5 * delta_height_circles_;
        if (max_x - min_x < delta_width_circles_ * 0.8 ||
            max_y - min_y < delta_height_circles_ * 0.8)
        {
            ROS_WARN("[Template] Reject: board-plane extent too small for template search.");
            return false;
        }

        const double grid = std::max(0.005, airy_template_grid_);
        const double angle_step = std::max(0.5, airy_template_angle_step_deg_) * M_PI / 180.0;
        TemplateEval best_any;
        TemplateEval best_feasible;
        double best_cx = 0.0;
        double best_cy = 0.0;
        double best_theta = 0.0;

        auto update_best = [&](double cx, double cy, double theta) {
            TemplateEval eval = evaluateTemplate(aligned_plane, cx, cy, theta);
            if (eval.score > best_any.score)
            {
                best_any = eval;
            }
            if (eval.feasible && eval.score > best_feasible.score)
            {
                best_feasible = eval;
                best_cx = cx;
                best_cy = cy;
                best_theta = theta;
            }
        };

        for (double theta = -M_PI / 2.0; theta <= M_PI / 2.0 + 1e-9; theta += angle_step)
        {
            for (double cx = min_x + half_w; cx <= max_x - half_w + 1e-9; cx += grid)
            {
                for (double cy = min_y + half_h; cy <= max_y - half_h + 1e-9; cy += grid)
                {
                    update_best(cx, cy, theta);
                }
            }
        }

        if (!best_feasible.feasible)
        {
            ROS_WARN("[Template] Reject: no coarse feasible template. Best_any score=%.3f",
                     best_any.score);
            logTemplateHoleStats("best_any", best_any.holes);
            return false;
        }

        const double fine_grid = std::max(0.005, grid * 0.5);
        const double fine_angle = std::max(0.5 * M_PI / 180.0, angle_step * 0.5);
        const double cx0 = best_cx;
        const double cy0 = best_cy;
        const double th0 = best_theta;
        for (double theta = th0 - angle_step; theta <= th0 + angle_step + 1e-9; theta += fine_angle)
        {
            for (double cx = cx0 - grid; cx <= cx0 + grid + 1e-9; cx += fine_grid)
            {
                for (double cy = cy0 - grid; cy <= cy0 + grid + 1e-9; cy += fine_grid)
                {
                    update_best(cx, cy, theta);
                }
            }
        }

        ROS_INFO("[Template] coarse/fine best score=%.3f center=(%.4f, %.4f) theta=%.2f deg",
                 best_feasible.score, best_cx, best_cy, best_theta * 180.0 / M_PI);
        logTemplateHoleStats("best", best_feasible.holes);

        if (!std::isfinite(best_feasible.score) || best_feasible.score < airy_template_min_score_)
        {
            ROS_WARN("[Template] Reject: score %.3f < min_score %.3f",
                     best_feasible.score, airy_template_min_score_);
            return false;
        }

        std::vector<pcl::PointXYZ> refined_centers;
        refined_centers.reserve(TARGET_NUM_CIRCLES);
        std::array<HoleStats, TARGET_NUM_CIRCLES> refined_holes;
        const double refine_radius = std::max(0.0, lidar_template_local_refine_radius_);
        const double refine_step = std::max(0.002, lidar_template_local_refine_step_);
        const int refine_steps = std::max(0, static_cast<int>(std::ceil(refine_radius / refine_step)));

        for (int hole_idx = 0; hole_idx < TARGET_NUM_CIRCLES; ++hole_idx)
        {
            const pcl::PointXYZ theoretical = best_feasible.centers[hole_idx];
            HoleStats best_local_stats = best_feasible.holes[hole_idx];
            pcl::PointXYZ best_local = theoretical;
            bool found_local = holeStatsFeasible(best_local_stats);

            for (int ix = -refine_steps; ix <= refine_steps; ++ix)
            {
                for (int iy = -refine_steps; iy <= refine_steps; ++iy)
                {
                    const double dx = ix * refine_step;
                    const double dy = iy * refine_step;
                    if (std::sqrt(dx * dx + dy * dy) > refine_radius + 1e-9)
                    {
                        continue;
                    }
                    HoleStats stats = computeHoleStats(aligned_plane,
                                                       static_cast<double>(theoretical.x) + dx,
                                                       static_cast<double>(theoretical.y) + dy);
                    if (!holeStatsFeasible(stats))
                    {
                        continue;
                    }
                    if (!found_local || stats.score > best_local_stats.score)
                    {
                        found_local = true;
                        best_local_stats = stats;
                        best_local.x = static_cast<float>(static_cast<double>(theoretical.x) + dx);
                        best_local.y = static_cast<float>(static_cast<double>(theoretical.y) + dy);
                        best_local.z = 0.0f;
                    }
                }
            }

            if (!found_local)
            {
                ROS_WARN("[Template] Reject: hole_%d has no locally feasible refine candidate.", hole_idx);
                return false;
            }
            refined_centers.push_back(best_local);
            refined_holes[hole_idx] = best_local_stats;
        }

        logTemplateHoleStats("refined", refined_holes);
        if (!validateTargetGeometryPoints(refined_centers, "template_refined_2d", true))
        {
            ROS_WARN("[Template] Reject: refined centers failed 2D geometry.");
            return false;
        }
        if (!backProjectAlignedCenters(refined_centers, R_inv, average_z, center_cloud))
        {
            ROS_WARN("[Template] Reject: back-projection did not produce four centers.");
            return false;
        }
        if (!acceptFinalGeometry(center_cloud, "template_3d"))
        {
            center_cloud->clear();
            center_z0_cloud_->clear();
            return false;
        }

        ROS_INFO("[Template] Accepted four empty-region template centers.");
        return true;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr extractAiryHoleBoundaryCandidates(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr &aligned_plane) const
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
        if (!airy_hole_detector_ || !aligned_plane || aligned_plane->empty())
        {
            return out;
        }

        const double search_r = std::max(0.020, airy_boundary_radius_);
        const double search_r2 = search_r * search_r;
        const int min_neighbors = std::max(3, airy_boundary_min_neighbors_);
        const double min_gap = std::max(1.2, airy_boundary_min_angular_gap_);

        out->reserve(aligned_plane->size());
        for (int i = 0; i < static_cast<int>(aligned_plane->size()); ++i)
        {
            const auto &p = aligned_plane->points[i];
            std::vector<double> angles;
            angles.reserve(64);
            int neighbors = 0;

            for (int j = 0; j < static_cast<int>(aligned_plane->size()); ++j)
            {
                if (i == j)
                {
                    continue;
                }
                const auto &q = aligned_plane->points[j];
                const double dx = static_cast<double>(q.x) - static_cast<double>(p.x);
                const double dy = static_cast<double>(q.y) - static_cast<double>(p.y);
                const double d2 = dx * dx + dy * dy;
                if (d2 <= 1e-10 || d2 > search_r2)
                {
                    continue;
                }
                ++neighbors;
                angles.push_back(std::atan2(dy, dx));
            }

            if (neighbors < min_neighbors || angles.size() < 3)
            {
                continue;
            }

            std::sort(angles.begin(), angles.end());
            double max_gap = 0.0;
            for (size_t k = 1; k < angles.size(); ++k)
            {
                max_gap = std::max(max_gap, angles[k] - angles[k - 1]);
            }
            max_gap = std::max(max_gap, angles.front() + 2.0 * M_PI - angles.back());

            if (max_gap >= min_gap)
            {
                out->push_back(p);
            }
        }

        ROS_INFO("[Airy] Boundary candidates by angular gap: %zu / %zu",
                 out->size(), aligned_plane->size());
        return out;
    }

    bool selectAndBackProjectCircleCenters(const std::vector<CircleCandidate> &circle_candidates,
                                           const Eigen::Matrix3d &R_inv,
                                           double average_z,
                                           const pcl::PointCloud<pcl::PointXYZ>::Ptr &qr_reference,
                                           pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud) const
    {
        center_z0_cloud_->clear();
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

        CandidateScore best_candidate = selectBestCandidateGroup(
            circle_candidates, R_inv, average_z, qr_reference);
        if (!best_candidate.found)
        {
            ROS_WARN("[LiDAR] Unable to find a candidate set that matches target's geometry");
            return false;
        }

        std::string group_text;
        for (size_t i = 0; i < best_candidate.group.size(); ++i)
        {
            group_text += (i == 0 ? "" : ",");
            group_text += std::to_string(best_candidate.group[i]);
        }

        ROS_INFO("[LiDAR][RANSAC] Selected circle group=[%s]: rmse=%.4f, geom_score=%.4f, geom_valid=%s, support=%d, selection_score=%.4f",
                 group_text.c_str(),
                 best_candidate.rmse,
                 best_candidate.geom_score,
                 best_candidate.geom_valid ? "true" : "false",
                 best_candidate.support,
                 best_candidate.selection_score);

        center_cloud->clear();
        for (const int idx : best_candidate.group)
        {
            const pcl::PointXYZ &center = circle_candidates[idx].center;
            Eigen::Vector3d aligned_point(center.x, center.y, center.z + average_z);
            Eigen::Vector3d original_point = R_inv * aligned_point;
            center_cloud->push_back(pcl::PointXYZ(original_point.x(),
                                                  original_point.y(),
                                                  original_point.z()));
        }
        if (center_cloud->size() != TARGET_NUM_CIRCLES)
        {
            return false;
        }
        if (!acceptFinalGeometry(center_cloud, "ransac_3d"))
        {
            center_cloud->clear();
            return false;
        }
        return true;
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
        airy_hole_detector_ = params.airy_hole_detector;
        airy_boundary_radius_ = params.airy_boundary_radius;
        airy_boundary_min_angular_gap_ = params.airy_boundary_min_angular_gap;
        airy_boundary_min_neighbors_ = params.airy_boundary_min_neighbors;
        airy_template_detector_ = params.airy_template_detector;
        airy_template_grid_ = params.airy_template_grid;
        airy_template_angle_step_deg_ = params.airy_template_angle_step_deg;
        airy_template_ring_band_ = params.airy_template_ring_band;
        airy_template_min_score_ = params.airy_template_min_score;
        lidar_center_extraction_mode_ = lowerString(params.lidar_center_extraction_mode);
        if (lidar_center_extraction_mode_ != "auto" &&
            lidar_center_extraction_mode_ != "template" &&
            lidar_center_extraction_mode_ != "ransac" &&
            lidar_center_extraction_mode_ != "legacy")
        {
            ROS_WARN("[LiDAR] Unknown lidar_center_extraction_mode='%s', using auto.",
                     params.lidar_center_extraction_mode.c_str());
            lidar_center_extraction_mode_ = "auto";
        }
        lidar_strict_geometry_ = params.lidar_strict_geometry;
        lidar_geometry_side_rel_tol_ = params.lidar_geometry_side_rel_tol;
        lidar_geometry_diag_rel_tol_ = params.lidar_geometry_diag_rel_tol;
        lidar_geometry_perimeter_rel_tol_ = params.lidar_geometry_perimeter_rel_tol;
        lidar_min_plane_points_ = params.lidar_min_plane_points;
        lidar_ransac_min_inliers_ = params.lidar_ransac_min_inliers;
        lidar_ransac_radius_tolerance_ = params.lidar_ransac_radius_tolerance;
        lidar_ransac_inlier_threshold_ = params.lidar_ransac_inlier_threshold;
        lidar_template_min_ring_per_hole_ = params.lidar_template_min_ring_per_hole;
        lidar_template_min_outer_per_hole_ = params.lidar_template_min_outer_per_hole;
        lidar_template_max_inside_per_hole_ = params.lidar_template_max_inside_per_hole;
        lidar_template_max_inside_support_ratio_ = params.lidar_template_max_inside_support_ratio;
        lidar_template_min_support_per_hole_ = params.lidar_template_min_support_per_hole;
        lidar_template_local_refine_radius_ = params.lidar_template_local_refine_radius;
        lidar_template_local_refine_step_ = params.lidar_template_local_refine_step;

        ROS_INFO("[LiDAR] lidar_center_extraction_mode=%s, strict_geometry=%s",
                 lidar_center_extraction_mode_.c_str(),
                 lidar_strict_geometry_ ? "true" : "false");
        ROS_INFO("[Template] min_ring=%d min_outer=%d max_inside=%d max_inside_support_ratio=%.3f min_support=%.2f",
                 lidar_template_min_ring_per_hole_,
                 lidar_template_min_outer_per_hole_,
                 lidar_template_max_inside_per_hole_,
                 lidar_template_max_inside_support_ratio_,
                 lidar_template_min_support_per_hole_);

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

        // 4. 将标定板平面点云/边缘点对齐到 Z=0 平面。模板法使用完整平面点云；
        // RANSAC 保留旧逻辑，使用按 ring gap 提取的边缘点。
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

        Eigen::Matrix3d R_inv = R_align.inverse();
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_plane(new pcl::PointCloud<pcl::PointXYZ>);
        aligned_plane->reserve(plane_cloud_->size());
        double plane_average_z = 0.0;
        for (const auto &pt : *plane_cloud_)
        {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R_align * point;
            aligned_plane->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            plane_average_z += aligned_point.z();
        }
        plane_average_z /= std::max<int>(static_cast<int>(aligned_plane->size()), 1);

        aligned_cloud_->reserve(edge_cloud_->size());
        double edge_average_z = 0.0;
        for (const auto &pt : *edge_cloud_)
        {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R_align * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            edge_average_z += aligned_point.z();
        }
        edge_average_z /= std::max<int>(static_cast<int>(aligned_cloud_->size()), 1);

        ROS_INFO("[LiDAR] lidar_center_extraction_mode=%s (mechanical)", lidar_center_extraction_mode_.c_str());
        if (wantsTemplate())
        {
            if (detectAiryTemplateCenters(aligned_plane, R_inv, plane_average_z, center_cloud))
            {
                *aligned_cloud_ = *aligned_plane;
                ROS_INFO("[LiDAR] Template extraction succeeded.");
                return;
            }
            ROS_WARN("[LiDAR] Template extraction failed.");
            if (!allowRansacFallback())
            {
                return;
            }
            ROS_WARN("[LiDAR] Falling back to legacy RANSAC.");
        }

        if (!wantsRansac())
        {
            return;
        }
        if (edge_cloud_->empty())
        {
            ROS_WARN("[LiDAR] No edge points found for legacy RANSAC.");
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud(new pcl::PointCloud<pcl::PointXYZ>(*aligned_cloud_));
        std::vector<CircleCandidate> circle_candidates = detectCircleCandidates(xy_cloud);
        if (!selectAndBackProjectCircleCenters(circle_candidates, R_inv, edge_average_z, qr_reference, center_cloud))
        {
            ROS_WARN("[LiDAR] Legacy RANSAC failed.");
        }
    }

    void detect_solid_lidar(pcl::PointCloud<Common::Point>::Ptr cloud,
                            pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud,
                            const pcl::PointCloud<pcl::PointXYZ>::Ptr &qr_reference = pcl::PointCloud<pcl::PointXYZ>::Ptr())
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

        Eigen::Matrix3d R_inv = R.inverse();
        ROS_INFO("[LiDAR] lidar_center_extraction_mode=%s (solid)", lidar_center_extraction_mode_.c_str());
        if (wantsTemplate())
        {
            if (detectAiryTemplateCenters(aligned_cloud_, R_inv, average_z, center_cloud))
            {
                ROS_INFO("[LiDAR] Template extraction succeeded.");
                return;
            }
            ROS_WARN("[LiDAR] Template extraction failed.");
            if (!allowRansacFallback())
            {
                return;
            }
            ROS_WARN("[LiDAR] Falling back to legacy RANSAC.");
        }

        if (!wantsRansac())
        {
            return;
        }

        // 4. Airy/solid-LiDAR robust hole-boundary detector. It runs before
        // the legacy normal-boundary path; if it cannot produce four centers,
        // the original detector below remains the fallback.
        pcl::PointCloud<pcl::PointXYZ>::Ptr airy_edge_cloud =
            extractAiryHoleBoundaryCandidates(aligned_cloud_);
        if (airy_edge_cloud->size() >= 8)
        {
            std::vector<CircleCandidate> airy_candidates = detectCircleCandidates(airy_edge_cloud);
            if (selectAndBackProjectCircleCenters(airy_candidates, R_inv, average_z, qr_reference, center_cloud))
            {
                *edge_cloud_ = *airy_edge_cloud;
                return;
            }
        }

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

        // 5. Legacy fallback remains the same circle-candidate RANSAC selector,
        // now using the normal-boundary edge cloud if the Airy boundary cloud
        // did not produce a valid four-center target.
        std::vector<CircleCandidate> boundary_candidates = detectCircleCandidates(edge_cloud_);
        if (!selectAndBackProjectCircleCenters(boundary_candidates, R_inv, average_z, qr_reference, center_cloud))
        {
            ROS_WARN("[LiDAR] Legacy RANSAC failed on normal-boundary edge cloud.");
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
