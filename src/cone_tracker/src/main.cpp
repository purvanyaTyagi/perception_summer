#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <vector>
#include <cmath>
#include <unordered_map>

using pcl::PointXYZ;

class PointCloudProcessor : public rclcpp::Node {
public:
    PointCloudProcessor() : Node("pointcloud_processor") {
        declare_parameter("z_threshold", -0.1692);
        get_parameter("z_threshold", z_threshold_);

        sub_ = create_subscription<sensor_msgs::msg::PointCloud>(
            "/carmaker/pointcloud", 10,
            std::bind(&PointCloudProcessor::cloud_callback, this, std::placeholders::_1));

        pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/dbscan_centroids", 10);

        RCLCPP_INFO(get_logger(), "Subscribed to /carmaker/pointcloud");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::PointCloud>::SharedPtr sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
    float z_threshold_;

    void cloud_callback(const sensor_msgs::msg::PointCloud::SharedPtr msg) {
        pcl::PointCloud<PointXYZ>::Ptr cloud(new pcl::PointCloud<PointXYZ>);

        for (const auto& pt : msg->points) {
            if (!std::isnan(pt.x) && !std::isnan(pt.y) && !std::isnan(pt.z)) {
                if (pt.z >= z_threshold_) {  // ⬅️ keep points with z >= threshold
                    cloud->points.emplace_back(pt.x, pt.y, pt.z);
                }
            }
        }

        if (cloud->empty()) return;

        // Run DBSCAN
        auto centroids = dbscan_centroids(cloud, 0.5 /*eps*/, 5 /*minPts*/);

        // Publish centroids as markers
        visualization_msgs::msg::MarkerArray markers;
        int id = 0;
        for (const auto& pt : centroids) {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = msg->header.frame_id;
            m.header.stamp = this->now();
            m.ns = "dbscan_centroids";
            m.id = id++;
            m.type = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = pt[0];
            m.pose.position.y = pt[1];
            m.pose.position.z = pt[2];
            m.scale.x = 0.3;
            m.scale.y = 0.3;
            m.scale.z = 0.3;
            m.color.r = 1.0;
            m.color.g = 0.0;
            m.color.b = 0.0;
            m.color.a = 1.0;
            markers.markers.push_back(m);
        }

        pub_->publish(markers);
    }

    std::vector<std::array<float, 3>> dbscan_centroids(pcl::PointCloud<PointXYZ>::Ptr cloud, float eps, int minPts) {
        std::vector<int> labels(cloud->points.size(), -1);
        int cluster_id = 0;

        auto distance = [](const PointXYZ& a, const PointXYZ& b) {
            return std::sqrt((a.x - b.x)*(a.x - b.x) +
                             (a.y - b.y)*(a.y - b.y) +
                             (a.z - b.z)*(a.z - b.z));
        };

        for (size_t i = 0; i < cloud->points.size(); ++i) {
            if (labels[i] != -1) continue;

            std::vector<int> neighbors;
            for (size_t j = 0; j < cloud->points.size(); ++j) {
                if (distance(cloud->points[i], cloud->points[j]) < eps) {
                    neighbors.push_back(j);
                }
            }

            if (neighbors.size() < static_cast<size_t>(minPts)) {
                labels[i] = -2; // noise
                continue;
            }

            labels[i] = cluster_id;
            std::vector<int> seeds = neighbors;

            for (size_t k = 0; k < seeds.size(); ++k) {
                if (labels[seeds[k]] == -2) labels[seeds[k]] = cluster_id;
                if (labels[seeds[k]] != -1) continue;
                labels[seeds[k]] = cluster_id;

                std::vector<int> new_neighbors;
                for (size_t j = 0; j < cloud->points.size(); ++j) {
                    if (distance(cloud->points[seeds[k]], cloud->points[j]) < eps) {
                        new_neighbors.push_back(j);
                    }
                }
                if (new_neighbors.size() >= static_cast<size_t>(minPts)) {
                    seeds.insert(seeds.end(), new_neighbors.begin(), new_neighbors.end());
                }
            }
            cluster_id++;
        }

        // Compute centroids
        std::unordered_map<int, std::vector<PointXYZ>> clusters;
        for (size_t i = 0; i < labels.size(); ++i) {
            if (labels[i] >= 0)
                clusters[labels[i]].push_back(cloud->points[i]);
        }

        std::vector<std::array<float, 3>> centroids;
        for (const auto& [id, pts] : clusters) {
            float x = 0, y = 0, z = 0;
            for (const auto& pt : pts) {
                x += pt.x; y += pt.y; z += pt.z;
            }
            float size = pts.size();
            centroids.push_back({x / size, y / size, z / size});
        }

        return centroids;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PointCloudProcessor>());
    rclcpp::shutdown();
    return 0;
}
