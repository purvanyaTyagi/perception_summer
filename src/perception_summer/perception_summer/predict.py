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

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import rclpy
from rclpy.node import Node

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

import socket
import pickle

import pandas as pd

# from tensorflow.keras.models import load_model
import joblib
import os

dbscan = DBSCAN(eps=0.15, min_samples=2)  # Adjust 'eps' based on your dataset

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.fc5(x)  # No softmax, use CrossEntropyLoss

class main_node(Node):
    def __init__(self):
        super().__init__('main_node')
        self.get_logger().info('Main node has been started!')
        self.subscription = self.create_subscription(PointCloud, '/carmaker/pointcloud', self.listener_callback, 10)
        self.subscription
        self.publisher_marker = self.create_publisher(MarkerArray, '/marker_array_perception', 10)
        self.margin = 0.0005
        self.distance_from_ground = -0.1629
        self.sphere_radius = 0.7
        self.fixed_length = 32
        self.prev_marker_count = 0
        self.model = SimpleNN(input_dim=32)  
        self.model.load_state_dict(torch.load(os.path.expanduser("~/Desktop/model_weights.pth")))

        files = [
            'cone_intensities_1_new.csv',
            'cone_intensities_2_new.csv',
            'cone_intensities_3_new.csv'
        ]
        dfs = [pd.read_csv(f, header=None) for f in files]
        data = pd.concat(dfs, ignore_index=True)
        X = data.iloc[:, :-1].values.astype('float32')  # All but last column
        y = data.iloc[:, -1].values   # Last column (labels: 'left' or 'right')
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)  # e.g. 'left' -> 0, 'right' -> 1
        # self.model = load_model('/absolute/path/to/cone_classifier_model.h5')
        # self.scaler = joblib.load('/absolute/path/to/scaler.pkl')
        # self.label_encoder = joblib.load('/absolute/path/to/label_encoder.pkl')

        #atexit.register(self.save_array)      

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

        classified_centroids = []
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

            local_intensities = np.array([local_intensities])
            sample_scaled = local_intensities
            sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32)

            with torch.no_grad():
                output = self.model(sample_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                label = self.label_encoder.inverse_transform([predicted_class])[0]

            if label == "left":
                classified_centroids.append([c, "left"])
            elif label == "right":
                classified_centroids.append([c, "right"])

        delete_array = MarkerArray()

        for i in range(self.prev_marker_count):
            delete_marker = Marker()
            delete_marker.header.frame_id = "Lidar_F"
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.ns = "cylinders"
            delete_marker.id = i
            delete_marker.action = Marker.DELETE
            delete_array.markers.append(delete_marker)

        self.publisher_marker.publish(delete_array)


        marker_array = MarkerArray()

        for i, c in enumerate(classified_centroids):
            marker = Marker()
            marker.header.frame_id = "Lidar_F"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "cylinders"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            # Position each cylinder differently
            marker.pose.position.x = c[0].x
            marker.pose.position.y = c[0].y
            marker.pose.position.z = c[0].z  # half height so cylinder sits on ground
            marker.pose.orientation.w = 1.0
            
            # Set cylinder size
            marker.scale.x = 0.5  # diameter x
            marker.scale.y = 0.5  # diameter y
            marker.scale.z = 1.0  # height
            
            # Set color
            if(c[1] == "left"):
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0  # opaque
            elif(c[1] == "right"):
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0  # opaque
            else:
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 1.0  # opaque
            
            marker_array.markers.append(marker)
        self.publisher_marker.publish(marker_array)
        self.prev_marker_count = len(classified_centroids)

def main(args=None):
    rclpy.init(args=args)
    node = main_node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()