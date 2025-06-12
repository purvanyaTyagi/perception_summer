import rclpy
from rclpy.node import Node
import numpy as np
from scipy.stats import mode
import random
import atexit

from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from sensor_msgs.msg import ChannelFloat32

from sensor_msgs.msg import PointCloud, PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from sklearn.cluster import DBSCAN
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

dbscan = DBSCAN(eps=0.15, min_samples=2)  # Adjust 'eps' based on your dataset

class main_node(Node):
    def __init__(self):
        super().__init__('main_node')
        self.get_logger().info('Main node has been started!')
        self.subscription = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.listener_callback, 10)
        self.subscription
        self.publisher_marker = self.create_publisher(Marker, 'visualization_marker', 10)
        self.margin = 0.0005
        self.distance_from_ground = -0.1629 + 0.05
        self.sphere_radius = 0.7
        self.fixed_length = 32
        self.intensity_samples = []
        self.labels = []
        atexit.register(self.save_array)      

    def listener_callback(self, msg: PointCloud):
        intensity_channel = next((c for c in msg.channels if c.name == "intensity"), None)
        if intensity_channel is None:
            self.get_logger().warn("No intensity channel found!")
            return
        
        raw_intensities = intensity_channel.values
        min_intensity = min(raw_intensities)
        max_intensity = max(raw_intensities)
        range_intensity = max_intensity - min_intensity if max_intensity != min_intensity else 1.0

        intensities = []
        points = []
        factor = 100

        for i, point in enumerate(msg.points):
            if((point.z > self.distance_from_ground)):
                raw = raw_intensities[i]
                scaled = (((raw - min_intensity) / range_intensity) ** 0.5) * factor
                intensities.append(scaled)
                points.append([point.x, point.y, point.z, scaled])


        xyz = np.array([[p[0], p[1], p[2]] for p in points])
        labels = dbscan.fit_predict(xyz)
        unique_clusters = set(labels) - {-1}
        centroids = []

        points = np.array(points)
        for cluster_id in unique_clusters:
            cluster_points = points[labels == cluster_id]  # Get points in this cluster
            centroid = cluster_points.mean(axis=0)    # Compute centroid
            centroids.append(Point32(x=centroid[0], y=centroid[1], z=centroid[2]))

        for i, c in enumerate(centroids):

            local_intensities = []
            for p, intensity in zip(points, intensities):
                dx = p[0] - c.x;
                dy = p[1] - c.y;
                dz = p[2] - c.z;
                dist = (dx**2 + dy**2 + dz**2) ** 0.5
                if dist <= self.sphere_radius:
                    local_intensities.append(intensity)

            if len(local_intensities) > self.fixed_length:
                local_intensities = random.sample(local_intensities, self.fixed_length)
            elif len(local_intensities) < self.fixed_length:
                local_intensities += [0.0] * (self.fixed_length - len(local_intensities))

    # Save to list
            self.intensity_samples.append(local_intensities)

            label = "left" if c.y > 0 else "right"
            self.labels.append(label)



        marker = Marker()
        marker.header.frame_id = "Lidar_F"  # Change to "base_link" if needed
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "my_markers"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST  # Can be CUBE, SPHERE, LINE_STRIP, etc.
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0  # Identity quaternion

        # Marker scale (size of spheres)
        marker.scale.x = 2 * self.sphere_radius
        marker.scale.y = 2 * self.sphere_radius
        marker.scale.z = 2 * self.sphere_radius

        # Marker color (RGBA)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully visible

        # Add points to the marker
        points_centroid = []
        for i, c in enumerate(centroids):
            point = Point(x=c.x, y=c.y, z=c.z)
            points_centroid.append(point)

        marker.points.extend(points_centroid)
        # Publish the marker
        self.publisher_marker.publish(marker)


        # for channel in msg.channels:
        #     self.get_logger().info(f'Channel name: {channel.name}, values count: {len(channel.values)}')

        #     for i, val in enumerate(channel.values[:3]):
        #         self.get_logger().info(f" value {i}: {val}")
    def save_array(self):
        import os
        intensity_array = np.array(self.intensity_samples)
        save_path = os.path.expanduser("~/projects/perception_summer/cone_intensities_3_new.csv")

        # Save with labels
        with open(save_path, "w") as f:
            for sample, label in zip(intensity_array, self.labels):
                line = ",".join(map(str, sample)) + f",{label}\n"
                f.write(line)

        print(f"✔ Saved labeled intensity data to {save_path}")


        

def main(args=None):
    rclpy.init(args=args)
    node = main_node()
    rclpy.spin(node)
    node.save_array()
    node.destroy_node()
    rclpy.shutdown()