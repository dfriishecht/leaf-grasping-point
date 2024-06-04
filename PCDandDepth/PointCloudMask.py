"""
Python script for applying and cleaning up a semantic segmentation mask to a depth map.
"""

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter



def apply_depth_mask(pointcloud_path, mask_path, depth_path, plot=True):
    # Load pointcloud from file and save its point position and original color as two numpy arrays
    pcd_path = "Pointclouds/demo_ (1).pcd"
    pcd_load = o3d.io.read_point_cloud(pcd_path, format='pcd')
    pcd_points = np.asarray(pcd_load.points)
    pcd_colors = np.asarray(pcd_load.colors)
    fill = np.zeros((1555200-len(pcd_colors), 3))
    pcd_colors = np.concatenate((pcd_colors, fill), axis=0)

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
    mask_erode = Image.fromarray(mask).filter(ImageFilter.MinFilter(9))
    mask_erode = np.asarray(mask_erode)[:, :, 0:3]

    # Use leaf mask to remove all non-leaf portions of the pointcloud
    mask_erode = np.reshape(mask_erode, (1555200, 3))
    pcd_colors = pcd_colors * mask_erode

    depth = np.load("Arrays/depth (1).npy")
    depth_load = o3d.io.read_point_cloud(pcd_path, format='pcd')
    depth_points = np.concatenate((np.asarray(depth_load.points), fill), axis=0)
    depth = depth_points[:,2]
    depth = np.reshape(depth, (1080, 1440, 1))

    mask_gray = np.mean(mask_erode, -1)
    index = np.argwhere(mask_gray!=0)
    mask_gray[index] = 1
    mask_gray = np.reshape(mask_gray, (1080, 1440, 1))
    depth_masked = depth * mask_gray

    xy_pos = np.asarray(depth_points)[:,0:2]
    xy_pos = np.reshape(xy_pos, (1080,1440, 2))
    xy_pos_masked = mask_gray * xy_pos

    mask_colors = np.unique(mask_erode, axis=0)
    color_index = np.zeros(shape=(1555200, 1))

    i = 0
    for color in mask_colors:
        index = np.argwhere(mask_erode==color)
        color_index[index] = i
        i+=1

    color_index = np.reshape(color_index, (1080, 1440, 1)).astype('uint8')
    masked_points = np.concatenate((xy_pos_masked,depth_masked,color_index), axis=2)

    left_image = Image.open("Images/left_rect (1).png")
    left_array = np.asarray(left_image)[:, :, 0:3]
    mask_gray_3d = np.repeat(mask_gray, 3, axis=2)
    left_array_masked = mask_gray_3d * (left_array/255)

    if plot is True:
        fig, ax = plt.subplot_mosaic([
            ['x', 'y', 'z'],[ 'mask','orig', 'crop']
        ], figsize=(15,10))

        ax['x'].imshow(masked_points[:, :, 0])
        ax['x'].set_title("X Coords (m)")
        ax['y'].imshow(masked_points[:, :, 1])
        ax['y'].set_title("Y Coords (m)")
        ax['z'].imshow(masked_points[:, :, 2])
        ax['z'].set_title("Depth (m)")
        ax['mask'].imshow(masked_points[:, :, 3])
        ax['mask'].set_title("Leaf ID")
        ax['orig'].imshow(left_image)
        ax['orig'].set_title("Original Image")
        ax['crop'].imshow(left_array_masked)
        ax['crop'].set_title("Cropped Image")
        plt.show()

    pcd = o3d.geometry.PointCloud()
    points = np.reshape(masked_points[:,:,0:3], (1555200, 3))
    points = np.delete(points, np.argwhere(points==[0,0,0]),axis=0)
    pcd.points = o3d.utility.Vector3dVector(points)



    # # Remove final artifacts.
    # # Prefer this over a more aggressive erosion since eventually it degrades the overall leaf shape
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=1.0)
    # inlier_cloud = pcd.select_by_index(ind)
    # outlier_cloud = pcd.select_by_index(ind, invert=True)
    # outlier_cloud.paint_uniform_color([1, 0, 0])
    # print(f"{len(outlier_cloud.points)} outlier points culled!")
    # print(f"Current point count: {len(inlier_cloud.points)}")
    # cropped_mask = np.asarray(inlier_cloud.colors)
    # cropped_points = np.asarray(inlier_cloud.points)

    return masked_points


def compute_normals(point_cloud):
    print("Computing Normals!")
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
    return point_cloud