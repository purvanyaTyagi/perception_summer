# 🧠 LiDAR-Only Cone Detection & Classification Pipeline

This project implements a LiDAR-only perception pipeline to detect traffic cones and classify them (e.g., left, right) using an Artificial Neural Network (ANN) built in PyTorch. This setup is useful for applications such as autonomous racing.

---

## 🛠 Features

- 📍 **Cone Detection** from raw LiDAR point clouds
- 📊 **Clustering** (e.g., DBSCAN) to isolate objects
- 🟠 **Centroid Extraction** from clustered points
- 🧠 **Cone Classification** using a trained PyTorch ANN
---
## 🧪 Workflow Overview

1. **Preprocessing**: Filter raw LiDAR points (e.g., clip z-axis, remove ground).
2. **Clustering**: Apply DBSCAN to segment individual cones.
3. **Feature Extraction**: Extract points from individual cones and scale the intensity values.
4. **Classification**: Pass features through a PyTorch ANN to classify cone type.
---
## 📦 Dependencies
- **ROS2-Humble**: and added dependencies
- **PyTorch** : LibTorch for c++
- **NVIDIA-CUDA**: For GPU acceleration and faster results, VERSION(12.6)
- **Sklearn**: For algorithms like DBSCAN.
---
Project Report: https://docs.google.com/document/d/1kQ-blUyZ2a6Zgz1Gi7SrNZTntNLJ8KSFtZw2osUuel8/edit?usp=sharing

Testing and Demo(recording issues in some of the videos caused flickering): https://drive.google.com/drive/folders/1IbSyTzNR2szfrdPBEUEnrQnOHEm84llj?usp=sharing


