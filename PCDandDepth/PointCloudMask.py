"""
Python script for applying and cleaning up a semantic segmentation mask to a depth map.
"""

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
from scipy import signal
import cv2


def apply_depth_mask(pointcloud_path, mask_path, depth_path, image_path, plot=True):
    """
    Function for applying leaf mask over depth image. Expects 1080x1440 resolution.

    Args:
        pointcloud_path (string): Filepath to pointcloud with .pcd format

        mask_path (string): Filepath to 1080x1440 image mask

        depth_path (string): Filepath to numpy array containing data on the depth of
                each pixel in the image

        image_path (string): Filepath to the original captured 1080x1440
                rectified camera image

        plot (bool): Optional input to plot this function's result

    Returns:
        masked_points (numpy array): A 1080x1440x4 numpy array containing data about the
                segmented plants. The first three 3rd dimension indices provide the
                real-world x, y, and z position of each point. The final index provides
                the leaf id that each point belongs to. If the id is 0, then the point is
                not part of a leaf.
    """

    # Load pointcloud from file and save its point position and original color as two numpy arrays
    pcd_path = pointcloud_path
    pcd_load = o3d.io.read_point_cloud(pcd_path, format="pcd")
    pcd_colors = np.asarray(pcd_load.colors)
    fill = np.zeros((1555200 - len(pcd_colors), 3))
    pcd_colors = np.concatenate((pcd_colors, fill), axis=0)

    # Load mask to be used for leaf segmentation
    mask_image = Image.open(mask_path)
    mask_array = np.asarray(mask_image)[:, :, 0:3]
    # Format mask array to be compatible with o3d's color array setup
    reshape_mask = np.reshape(mask_array, (1555200, 3))
    mask_norm = reshape_mask / 255
    index = np.argwhere(
        mask_norm == mask_norm[0]
    )  # TODO: Replace this with a more robust mask color selection method

    # Turn background portions of the mask black. This is to ensure clean image erosion.
    mask = reshape_mask.copy()
    mask[index] = [0, 0, 0]
    # Erode mask to remove artifacts in pointcloud from the depth map.
    mask = np.reshape(mask, (1080, 1440, 3))
    mask_erode = Image.fromarray(mask).filter(ImageFilter.MinFilter(9))
    mask_erode = np.asarray(mask_erode)[:, :, 0:3]
    # Use leaf mask to remove all non-leaf portions of the pointcloud
    mask_erode = np.reshape(mask_erode, (1555200, 3))
    pcd_colors = pcd_colors * mask_erode

    depth_load = o3d.io.read_point_cloud(pcd_path, format="pcd")
    depth_points = np.concatenate((np.asarray(depth_load.points), fill), axis=0)
    depth = depth_points[:, 2]
    depth = np.reshape(depth, (1080, 1440, 1))

    mask_gray = np.mean(mask_erode, -1)
    index = np.argwhere(mask_gray != 0)
    mask_gray[index] = 1
    mask_gray = np.reshape(mask_gray, (1080, 1440, 1))
    depth_masked = depth * mask_gray

    xy_pos = np.asarray(depth_points)[:, 0:2]
    xy_pos = np.reshape(xy_pos, (1080, 1440, 2))
    xy_pos_masked = mask_gray * xy_pos

    mask_colors = np.unique(reshape_mask, axis=0)
    print(f"unique mask colors: {len(mask_colors)}")
    color_index = np.zeros(shape=(1555200, 1))
    print(mask_erode.shape)
    i = 0
    for color in mask_colors:
        index = np.all(mask_erode == color, axis=-1)
        color_index[index] = i
        i += 1
    color_index = np.reshape(color_index, (1080, 1440, 1)).astype("uint8")
    print(len(np.unique(np.reshape(color_index, (1555200, 1)), axis=0)))
    masked_points = np.concatenate((xy_pos_masked, depth_masked, color_index), axis=2)

    left_image = Image.open(image_path)
    left_array = np.asarray(left_image)[:, :, 0:3]
    mask_gray_3d = np.repeat(mask_gray, 3, axis=2)
    left_array_masked = mask_gray_3d * (left_array / 255)

    if plot is True:
        _, ax = plt.subplot_mosaic(
            [["x", "y", "z"], ["mask", "orig", "crop"]], figsize=(15, 10)
        )

        ax["x"].imshow(masked_points[:, :, 0])
        ax["x"].set_title("X Coords (m)")
        ax["y"].imshow(masked_points[:, :, 1])
        ax["y"].set_title("Y Coords (m)")
        ax["z"].imshow(masked_points[:, :, 2])
        ax["z"].set_title("Depth (m)")
        ax["mask"].imshow(masked_points[:, :, 3])
        ax["mask"].set_title("Leaf ID")
        ax["orig"].imshow(left_image)
        ax["orig"].set_title("Original Image")
        ax["crop"].imshow(left_array_masked)
        ax["crop"].set_title("Cropped Image")
        plt.show()

    pcd = o3d.geometry.PointCloud()
    points = np.reshape(masked_points[:, :, 0:3], (1555200, 3))
    points = np.delete(points, np.argwhere(points == [0, 0, 0]), axis=0)
    pcd.points = o3d.utility.Vector3dVector(points)

    return masked_points


def compute_normals(point_cloud):
    """
    Estimates the normal vector of every point in a point cloud.
    The resulting normal vectors will all have a magnitude of one, and their
    coordinates saved relative to the point they correspond to. Each point
    will be its respective normal vector's "(0,0,0)".

    Agrs:
        point_cloud (open3d point cloud object): The desired point cloud to
                compute normals for

    Returns:
        point_cloud (open3d point cloud object): The input point cloud with
        normal vector data now included. Can be called with point_cloud.normals
    """
    print("Computing Normals!")
    point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    return point_cloud


def get_kernels(depth_image, mask_image):
    """
    Calculate and get all of the convolution kernels needed for each leaf.
    Masks out each leaf individually to calculate its respective kernel size.

    Args:
        depth_image (numpy array): Array of depth values for each leaf.
        mask_image (numpy array): Array of leaf id values for each leaf.

    Returns:
        kernels (list): A list of kernel sizes indexed based on which leaf
                to convolve them with.
    """
    masks_ind = np.unique(mask_image)
    kernels = []

    for i in range(1, len(masks_ind)):
        singular_leaf_mask = np.where(mask_image, mask_image == masks_ind[i], 0)
        leaf_mask = singular_leaf_mask * depth_image
        kernel_ = kernel_size(leaf_mask, plot=False)
        kernels.append(kernel_)
    print("all kernels: ", kernels)
    return kernels


def kernel_size(depth_mask, plot=False):
    """
    Helper function to calculate the size of each kernel. The size is
    originally based on the physical dimensions of the microneedle array.
    Then, the mean depth of each leaf is used to figure out how large the array is
    relative to each leaf in the camera frame, which will scale the kernel accordingly.

    args:
        depth_mask (numpy array): The masked depth data of a single leaf.
        plot (bool): Optional argument to plot each kernel.

    returns:
        kernel_ (int): Width and height of the calculated kernel in pixels.
    """
    calib_projection_matrix = np.array(
        (
            [1722.235253, 0, 584.315697, 0],
            [0, 1722.235253, 488.690098, 0],
            [
                0,
                0,
                1,
                0,
            ],
        )
    )
    P = np.reshape(calib_projection_matrix, (3, 4))
    X = np.ones((4, 1))

    leaf_x = np.nonzero(depth_mask)[0]
    leaf_y = np.nonzero(depth_mask)[1]
    mean_depth = np.mean(depth_mask[leaf_x, leaf_y])

    depth = np.array(([0, 0, mean_depth]))
    mn_dim = 0.005
    d1 = np.array(([-mn_dim / 2, -mn_dim / 2, mean_depth]))
    d2 = np.array(([mn_dim / 2, -mn_dim / 2, mean_depth]))
    d3 = np.array(([-mn_dim / 2, mn_dim / 2, mean_depth]))
    d4 = np.array(([mn_dim / 2, mn_dim / 2, mean_depth]))
    Dc = np.array([depth]).transpose()
    D1 = np.array([d1]).transpose()
    D2 = np.array([d2]).transpose()
    D3 = np.array([d3]).transpose()
    D4 = np.array([d4]).transpose()
    X[0:3, :] = np.array(Dc)
    x = np.matmul(P, X)
    xc = x / x[-1:]
    X[0:3, :] = np.array(D1)
    x = np.matmul(P, X)
    x1 = x / x[-1:]
    X[0:3, :] = np.array(D2)
    x = np.matmul(P, X)
    x2 = x / x[-1:]
    X[0:3, :] = np.array(D3)
    x = np.matmul(P, X)
    x3 = x / x[-1:]
    X[0:3, :] = np.array(D4)
    x = np.matmul(P, X)
    x4 = x / x[-1:]

    sz_1 = np.abs(np.round(x1[0]) - np.round(x2[0]))
    sz_2 = np.abs(np.round(x1[1]) - np.round(x3[1]))
    sz_3 = np.abs(np.round(x4[0]) - np.round(x3[0]))
    sz_4 = np.abs(np.round(x4[1]) - np.round(x2[1]))

    kernel_ = np.round(
        np.average(([sz_1, sz_2, sz_3, sz_4]))
    )  # this is just a length of a square

    print("average kernel size: ", kernel_)

    if plot:
        plt.imshow(depth_mask)
        plt.plot(np.round(xc[0]), np.round(xc[1]), "r.")
        plt.plot(np.round(x1[0]), np.round(x1[1]), "r.")
        plt.plot(np.round(x2[0]), np.round(x2[1]), "r.")
        plt.plot(np.round(x3[0]), np.round(x3[1]), "r.")
        plt.plot(np.round(x4[0]), np.round(x4[1]), "r.")
        plt.show()
    return kernel_


def do_convolution(kernel, mask):
    """
    Convolves each individual leaf with their respective kernel to calculate what portions
    of the leaf can actually be grasped.

    args:
        kernel (list): List of kernels. There should be one kernel per leaf.
    mask:
        mask(numpy array): Array of leaf id values for each leaf. Anywhere there
                is no leaf is 0.
    """
    kernel_ = kernel
    mask_ = mask
    index_ = np.unique(mask_)

    graspable_areas = np.zeros((1080, 1440))

    for i in range(1, len(index_)):
        mask_local_ = mask_ == index_[i]
        mask_local_ = np.where(mask_local_, mask_local_ >= 1, 0)
        graspable_area = signal.convolve2d(
            mask_local_.astype("uint8"),
            np.ones((kernel_[i - 1].astype("uint8"), kernel_[i - 1].astype("uint8"))),
            boundary="symm",
            mode="same",
        )
        graspable_area = np.where(
            graspable_area, graspable_area < np.amax(graspable_area) * 0.9, 1
        )  # remove blurry parts
        graspable_area_ = np.logical_not(graspable_area).astype(int)
        i_, j_ = np.where(graspable_area_ == np.amax(graspable_area_))
        graspable_areas[i_, j_] = i
    return graspable_areas


def get_centroids(mask):
    """
    When provided with a mask, will calculate the centroid
    for every group of ids in the mask

    Args:
        mask (numpy array): mask, where each coordinate in the mask
        has an id attributed to it

    Returns:
        centroids (list): A list of x,y coordinate tuples corresponding to the
        calculated centroids
    """
    index = np.unique(mask)
    centroids = []

    for i, _ in enumerate(index):
        if i == 0:
            continue
        mask_ = mask == index[i]
        mask_ = np.where(mask_, mask_ >= 1, 0)
        contour, _ = cv2.findContours(
            mask_.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )
        MOMENT = cv2.moments(contour[0])
        if MOMENT["m00"] > 0:
            center_x = int(MOMENT["m10"] / MOMENT["m00"])
            center_y = int(MOMENT["m01"] / MOMENT["m00"])
            centroids.append((center_x, center_y))

    return centroids
