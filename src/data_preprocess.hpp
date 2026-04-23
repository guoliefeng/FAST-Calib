/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef DATA_PREPROCESS_HPP
#define DATA_PREPROCESS_HPP

#include "CustomMsg.h"
#include <Eigen/Core>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <limits>
#include <sys/stat.h>
#include "common_lib.h"

using namespace std;

enum class LiDARType : int {
    Unknown = 0,
    Solid   = 1,   // 固态（如 Livox）
    Mech    = 2    // 机械式多线
};

class DataPreprocess
{
public:
    // 改成带线号的点云
    pcl::PointCloud<Common::Point>::Ptr cloud_input_;
    cv::Mat img_input_;
    LiDARType lidar_type_{LiDARType::Unknown};
    LiDARType lidarType() const { return lidar_type_; }

    DataPreprocess(Params &params)
        : cloud_input_(new pcl::PointCloud<Common::Point>)
    {
        string image_path = params.image_path;

        // 读图像
        img_input_ = cv::imread(image_path, cv::IMREAD_UNCHANGED);
        if (img_input_.empty())
        {
            std::string msg = "Loading the image " + image_path + " failed";
            ROS_ERROR_STREAM(msg.c_str());
            return;
        }

        const string source = normalizeSource(params.pointcloud_source);
        if (source == "bag" || source == "rosbag")
        {
            loadBagCloud(params.bag_path, params.lidar_topic);
        }
        else if (source == "pcd")
        {
            loadPcdCloud(params.pcd_path);
        }
        else
        {
            ROS_ERROR_STREAM("Unsupported pointcloud_source: " << params.pointcloud_source
                             << ". Use 'bag' or 'pcd'.");
            return;
        }
    }

private:
    static string normalizeSource(string value)
    {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    static const sensor_msgs::PointField* findField(const sensor_msgs::PointCloud2 &msg,
                                                    const string &name)
    {
        for (const auto &field : msg.fields)
        {
            if (field.name == name)
            {
                return &field;
            }
        }
        return nullptr;
    }

    static double readFieldAsDouble(const sensor_msgs::PointCloud2 &msg,
                                    const sensor_msgs::PointField &field,
                                    size_t point_index)
    {
        const uint8_t *ptr = msg.data.data() + point_index * msg.point_step + field.offset;
        switch (field.datatype)
        {
            case sensor_msgs::PointField::INT8:
            {
                int8_t v;
                std::memcpy(&v, ptr, sizeof(v));
                return static_cast<double>(v);
            }
            case sensor_msgs::PointField::UINT8:
            {
                uint8_t v;
                std::memcpy(&v, ptr, sizeof(v));
                return static_cast<double>(v);
            }
            case sensor_msgs::PointField::INT16:
            {
                int16_t v;
                std::memcpy(&v, ptr, sizeof(v));
                return static_cast<double>(v);
            }
            case sensor_msgs::PointField::UINT16:
            {
                uint16_t v;
                std::memcpy(&v, ptr, sizeof(v));
                return static_cast<double>(v);
            }
            case sensor_msgs::PointField::INT32:
            {
                int32_t v;
                std::memcpy(&v, ptr, sizeof(v));
                return static_cast<double>(v);
            }
            case sensor_msgs::PointField::UINT32:
            {
                uint32_t v;
                std::memcpy(&v, ptr, sizeof(v));
                return static_cast<double>(v);
            }
            case sensor_msgs::PointField::FLOAT32:
            {
                float v;
                std::memcpy(&v, ptr, sizeof(v));
                return static_cast<double>(v);
            }
            case sensor_msgs::PointField::FLOAT64:
            {
                double v;
                std::memcpy(&v, ptr, sizeof(v));
                return v;
            }
            default:
                return std::numeric_limits<double>::quiet_NaN();
        }
    }

    static uint16_t readRingField(const sensor_msgs::PointCloud2 &msg,
                                  const sensor_msgs::PointField &field,
                                  size_t point_index)
    {
        const double raw = readFieldAsDouble(msg, field, point_index);
        if (!std::isfinite(raw) || raw < 0.0)
        {
            return 0xFFFF;
        }
        if (raw > static_cast<double>(std::numeric_limits<uint16_t>::max()))
        {
            return std::numeric_limits<uint16_t>::max();
        }
        return static_cast<uint16_t>(raw);
    }

    static bool isDirectory(const string &path)
    {
        struct stat info;
        if (stat(path.c_str(), &info) != 0)
        {
            return false;
        }
        return S_ISDIR(info.st_mode);
    }

    static bool hasPcdSuffix(string name)
    {
        std::transform(name.begin(), name.end(), name.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return name.size() >= 4 && name.substr(name.size() - 4) == ".pcd";
    }

    static std::vector<string> listPcdFiles(const string &path)
    {
        std::vector<string> files;
        if (!isDirectory(path))
        {
            files.push_back(path);
            return files;
        }

        DIR *dir = opendir(path.c_str());
        if (!dir)
        {
            return files;
        }

        while (dirent *entry = readdir(dir))
        {
            const string name(entry->d_name);
            if (name == "." || name == ".." || !hasPcdSuffix(name))
            {
                continue;
            }
            files.push_back(path + "/" + name);
        }
        closedir(dir);
        std::sort(files.begin(), files.end());
        return files;
    }

    bool appendPointCloud2(const sensor_msgs::PointCloud2 &msg, const string &source_name)
    {
        const auto *x_field = findField(msg, "x");
        const auto *y_field = findField(msg, "y");
        const auto *z_field = findField(msg, "z");
        if (!x_field || !y_field || !z_field)
        {
            ROS_ERROR_STREAM(source_name << " point cloud is missing x/y/z fields.");
            return false;
        }

        const auto *ring_field = findField(msg, "ring");
        if (!ring_field)
        {
            ring_field = findField(msg, "line");
        }

        if (ring_field)
        {
            lidar_type_ = LiDARType::Mech;
        }
        else if (lidar_type_ == LiDARType::Unknown)
        {
            lidar_type_ = LiDARType::Solid;
        }

        const size_t n = static_cast<size_t>(msg.width) * msg.height;
        cloud_input_->reserve(cloud_input_->size() + n);

        for (size_t i = 0; i < n; ++i)
        {
            const double x = readFieldAsDouble(msg, *x_field, i);
            const double y = readFieldAsDouble(msg, *y_field, i);
            const double z = readFieldAsDouble(msg, *z_field, i);
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            {
                continue;
            }

            Common::Point p;
            p.x = static_cast<float>(x);
            p.y = static_cast<float>(y);
            p.z = static_cast<float>(z);
            p.ring = ring_field ? readRingField(msg, *ring_field, i) : 0xFFFF;
            cloud_input_->push_back(p);
        }

        return true;
    }

    void loadBagCloud(const string &bag_path, const string &lidar_topic)
    {
        // 先检查包是否存在
        std::fstream file_;
        file_.open(bag_path, ios::in);
        if (!file_)
        {
            std::string msg = "Loading the rosbag " + bag_path + " failed";
            ROS_ERROR_STREAM(msg.c_str());
            return;
        }
        ROS_INFO("Loading the rosbag %s", bag_path.c_str());

        rosbag::Bag bag;
        try {
            bag.open(bag_path, rosbag::bagmode::Read);
        } catch (rosbag::BagException &e) {
            ROS_ERROR_STREAM("LOADING BAG FAILED: " << e.what());
            return;
        }

        std::vector<string> lidar_topic_vec = {lidar_topic};
        rosbag::View view(bag, rosbag::TopicQuery(lidar_topic_vec));

        // 累计读取
        for (const rosbag::MessageInstance &m : view)
        {
            // 1) Livox 自定义消息（含 line 字段）
            if (auto livox_custom_msg = m.instantiate<livox_ros_driver::CustomMsg>())
            {
                lidar_type_ = LiDARType::Solid;
                cloud_input_->reserve(livox_custom_msg->point_num);
                for (uint32_t i = 0; i < livox_custom_msg->point_num; ++i)
                {
                    Common::Point p;
                    p.x = livox_custom_msg->points[i].x;
                    p.y = livox_custom_msg->points[i].y;
                    p.z = livox_custom_msg->points[i].z;
                    // Livox 的 CustomPoint 有 line 字段（uint8 / uint16 视版本而定）
                    p.ring = static_cast<std::uint16_t>(livox_custom_msg->points[i].line);
                    cloud_input_->push_back(p);
                }
                continue;
            }

            // 2) 机械雷达 / 通用 PointCloud2
            if (auto pcl_msg = m.instantiate<sensor_msgs::PointCloud2>())
            {
                appendPointCloud2(*pcl_msg, "rosbag PointCloud2");
                continue;
            }

            // 其他类型忽略
        }

        ROS_INFO("Loaded %zu points from the rosbag.", cloud_input_->size());
    }

    bool loadSinglePcdCloud(const string &pcd_path, bool &has_ring_or_line)
    {
        pcl::PCLPointCloud2 pcl_blob;
        if (pcl::io::loadPCDFile(pcd_path, pcl_blob) < 0)
        {
            ROS_ERROR_STREAM("LOADING PCD FAILED: " << pcd_path);
            return false;
        }

        sensor_msgs::PointCloud2 msg;
        pcl_conversions::fromPCL(pcl_blob, msg);
        const auto *ring_field = findField(msg, "ring");
        const auto *line_field = findField(msg, "line");
        has_ring_or_line = has_ring_or_line || ring_field || line_field;

        if (!appendPointCloud2(msg, "PCD"))
        {
            return false;
        }

        return true;
    }

    void loadPcdCloud(const string &pcd_path)
    {
        if (pcd_path.empty())
        {
            ROS_ERROR("pointcloud_source is 'pcd' but pcd_path is empty.");
            return;
        }

        const std::vector<string> pcd_files = listPcdFiles(pcd_path);
        if (pcd_files.empty())
        {
            ROS_ERROR_STREAM("No PCD files found at " << pcd_path);
            return;
        }

        ROS_INFO("Loading %zu PCD file(s) from %s", pcd_files.size(), pcd_path.c_str());
        bool has_ring_or_line = false;
        size_t loaded_files = 0;
        for (const auto &file : pcd_files)
        {
            if (loadSinglePcdCloud(file, has_ring_or_line))
            {
                ++loaded_files;
            }
        }

        if (!has_ring_or_line)
        {
            ROS_WARN("PCD has no ring/line field; using solid-LiDAR detection path.");
        }
        ROS_INFO("Loaded %zu points from %zu PCD file(s).", cloud_input_->size(), loaded_files);
    }
};

typedef std::shared_ptr<DataPreprocess> DataPreprocessPtr;

#endif // DATA_PREPROCESS_HPP
