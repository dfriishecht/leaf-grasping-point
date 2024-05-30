"""
Helper functions for testing the extraction of point clouds from
depth images.
"""


import numpy as np
import open3d as o3d


def load_pointcloud(pcd_path):
    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud(pcd_path, format='pcd')
    pcd_points = np.asarray(pcd_load.points)
    return pcd_load, pcd_points

def viz_pointcloud(pcd):
    o3d.visualization.draw_geometries([pcd])