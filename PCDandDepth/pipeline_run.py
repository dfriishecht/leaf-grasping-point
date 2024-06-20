# import skfmm
import numpy as np
import open3d as o3d
import sklearn.metrics.pairwise as pdist
from scipy import signal
from skimage import measure
from paretoset import paretoset
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter

import PointCloudMask as pcm

pointcloud_path = "Pointclouds/demo_4.pcd"
mask_path = "Images/aggrigated_masks4.png"
depth_path = "Arrays/depth4.npy"
image_path = "Images/left_rect2.png"

leafs = pcm.apply_depth_mask(
    pointcloud_path, mask_path, depth_path, image_path, plot=False
)

mask = pcm.clean_mask(leafs)
leafs[:, :, 3] = mask

depth_image = leafs[:, :, 2].astype("float32")
mask_image = leafs[:, :, 3].astype("uint8")
kernels = pcm.get_kernels(depth_image, mask_image)
graspable_mask = pcm.compute_graspable_areas(kernels, mask_image)

binary_graspable_mask = graspable_mask >= 1
leafs[:, :, 3] = leafs[:, :, 3] * binary_graspable_mask
leafs_ = np.reshape(leafs[:, :, 0:3], (1555200, 3))
index = np.argwhere(leafs_ == [0, 0, 0])
inverse_index = np.nonzero(leafs_[:, 2])
leafs_ = np.delete(leafs_, index, 0)

processed_pcd = o3d.geometry.PointCloud()
processed_pcd.points = o3d.utility.Vector3dVector(leafs_)

pcm.compute_normals(processed_pcd)
sqrt_dist = np.sum((processed_pcd.normals[999]) ** 2, axis=0)
dist = np.sqrt(sqrt_dist)
normal_orientation = abs(np.asarray(processed_pcd.normals)[:, 2])
orientation_color = np.zeros((len(normal_orientation), 3))
orientation_color[:, 0] = normal_orientation
orientation_color[:, 1:] = 0
processed_pcd.colors = o3d.utility.Vector3dVector(orientation_color)

leaf_flatness = np.zeros((1555200, 1))
j = 0
for i, _ in enumerate(inverse_index[0]):
    current_index = inverse_index[0][i]
    leaf_flatness[current_index, 0] = normal_orientation[j]
    j += 1

leaf_flatness = np.reshape(leaf_flatness, (1080, 1440, 1))
leafs = np.concatenate((leafs, leaf_flatness), axis=2)

ALPHA = 0.4
smooth_section = leafs[:, :, 4]
leaf_selection_a = ALPHA * smooth_section + (1 - ALPHA) * binary_graspable_mask

leaf_selection_ab = np.where(
    leaf_selection_a, leaf_selection_a >= np.amax(leaf_selection_a) * 0.95, 0
)

leafs[:, :, 3] *= leaf_selection_ab
viable_leaf_regions = pcm.clean_mask(leafs)
