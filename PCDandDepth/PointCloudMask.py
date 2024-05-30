"""
Python script for applying and cleaning up a semantic segmentation mask to a depth map.
"""

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter

# Load pointcloud from file and save its point position and original color as two numpy arrays
pcd_path = "Pointclouds/demo_ (1).pcd"
pcd_load = o3d.io.read_point_cloud(pcd_path, format='pcd')
pcd_points = np.asarray(pcd_load.points)
pcd_colors_orig = np.asarray(pcd_load.colors)
pcd_count = len(pcd_load.points)

print(f"Point cloud with {pcd_count} points loaded!")

# Load mask to be used for leaf segmentation
mask_image = Image.open('Images/aggrigated_masks (1).png')
mask_array = np.asarray(mask_image)[:, :, 0:3]

# Format mask array to be compatible with o3d's color array setup
reshape_mask = np.reshape(mask_array, (1555200, 3))
mask_norm = reshape_mask / 255
index = np.argwhere(mask_norm==mask_norm[0]) #TODO: Replace this with a more robust mask color selection method

# Turn background portions of the mask black. This is to ensure clean image erosion.
mask = reshape_mask.copy()
mask[index] = [0,0,0]

# Erode mask to remove artifacts in pointcloud from the depth map.
mask = np.reshape(mask, (1080, 1440, 3))
mask_erode = Image.fromarray(mask).filter(ImageFilter.MinFilter(7))
mask_erode = np.asarray(mask_erode)[:, :, 0:3]

# Use leaf mask to remove all non-leaf portions of the pointcloud
mask_erode = np.reshape(mask_erode, (1555200, 3))
pcd_colors = np.asarray(pcd_load.colors)
index = np.argwhere(mask_erode==[0, 0, 0])
pcd_colors = mask_erode/255
pcd_colors = np.delete(pcd_colors, index, 0)
pcd_points = np.delete(pcd_points, index, 0)
pcd_colors_orig = np.delete(pcd_colors_orig, index, 0)

mask_pcd_count = len(pcd_points)

print(f"Mask applied with {pcd_count - mask_pcd_count} points culled!")
print(f"Current point count: {mask_pcd_count}")

# Save the cropped leaves as new point cloud.
pcd = o3d.geometry.PointCloud()
pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
pcd.points = o3d.utility.Vector3dVector(pcd_points)

# Remove final artifacts.
# Prefer this over a more aggressive erosion since eventually it degrades the overall leaf shape
cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
inlier_cloud = pcd.select_by_index(ind)
outlier_cloud = pcd.select_by_index(ind, invert=True)
outlier_cloud.paint_uniform_color([1, 0, 0])
print(f"{len(outlier_cloud.points)} outlier points culled!")
print(f"Current point count: {len(inlier_cloud.points)}")
cropped_mask = np.asarray(inlier_cloud.colors)
cropped_points = np.asarray(inlier_cloud.points)

# Generate a series of individual point clouds for each leaf and save them to a list
mask_colors = np.unique(cropped_mask, axis=0)
leaf_list = []
for color in mask_colors:
    index = np.argwhere(cropped_mask!=color)
    colors = np.delete(cropped_mask, index, 0)
    points = np.delete(cropped_points, index, 0)
    pcd_single = o3d.geometry.PointCloud()
    pcd_single.colors = o3d.utility.Vector3dVector(colors)
    pcd_single.points = o3d.utility.Vector3dVector(points)
    if len(points) > 100:
        leaf_list.append(pcd_single)
# Also store the complete set of leaves as one array
all_leafs = cropped_points


# Create a plot of the final depth mask
min_depth = np.min(all_leafs[:,2])
max_depth = np.max(all_leafs[:,2])
fig = plt.figure()
ax1 = fig.add_subplot(111)
depth_norm = (((all_leafs[:, 2]-min_depth) / (max_depth-min_depth)))
plt.scatter(all_leafs[:,0], all_leafs[:,1], c=depth_norm,cmap=plt.cm.autumn_r, s=depth_norm)
plt.gca().invert_yaxis()
plt.colorbar()
plt.title("Cropped Leaves Visualized w/ Normalized Depth")
plt.xlabel("X-Position (m)")
plt.ylabel("Y-Position (m)")
plt.show()

