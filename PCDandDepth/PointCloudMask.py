"""
Python script for applying and cleaning up a semantic segmentation mask to a depth map.
"""

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter
from scipy import signal
from joblib import Parallel, delayed
from skimage import measure
import cv2
import math
from concurrent.futures import ThreadPoolExecutor
import sklearn.metrics.pairwise as pdist

# import skfmm


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
    i = 1
    for color in mask_colors:
        index = np.all(mask_erode == color, axis=-1)
        color_index[index] = i
        i += 1
    color_index = np.reshape(color_index, (1080, 1440, 1)).astype("uint8")
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

    depth = np.array([0, 0, mean_depth])
    mn_dim = 0.005
    offsets = [
        [-mn_dim / 2, -mn_dim / 2, mean_depth],
        [mn_dim / 2, -mn_dim / 2, mean_depth],
        [-mn_dim / 2, mn_dim / 2, mean_depth],
        [mn_dim / 2, mn_dim / 2, mean_depth],
    ]

    # Prepare the points
    points = [depth] + offsets
    transformed_points = []

    for point in points:
        X[0:3, :] = np.array([point]).transpose()
        x = np.matmul(P, X)
        transformed_points.append(x / x[-1:])

    # Unpack the transformed points
    xc, x1, x2, x3, x4 = transformed_points

    sz_1 = np.abs(np.round(x1[0]) - np.round(x2[0]))
    sz_2 = np.abs(np.round(x1[1]) - np.round(x3[1]))
    sz_3 = np.abs(np.round(x4[0]) - np.round(x3[0]))
    sz_4 = np.abs(np.round(x4[1]) - np.round(x2[1]))

    kernel_ = np.round(np.average(([sz_1, sz_2, sz_3, sz_4])))

    if plot:
        plt.imshow(depth_mask)
        plt.plot(np.round(xc[0]), np.round(xc[1]), "r.")
        plt.plot(np.round(x1[0]), np.round(x1[1]), "r.")
        plt.plot(np.round(x2[0]), np.round(x2[1]), "r.")
        plt.plot(np.round(x3[0]), np.round(x3[1]), "r.")
        plt.plot(np.round(x4[0]), np.round(x4[1]), "r.")
        plt.show()

    return kernel_


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
    areas = []
    for idx in index:
        if idx == 0:  # 0 is background id
            continue

        mask_ = mask == idx

        contour, _ = cv2.findContours(
            mask_.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
        )

        mask_ = np.where(mask_, mask_ >= 1, 0)
        labels = measure.label(mask_)
        props = measure.regionprops(labels)

        total_area = 0
        for prop in props:
            total_area += prop.area
        areas.append(total_area)
        if contour:
            contour_max = max(contour, key=cv2.contourArea)
            MOMENT = cv2.moments(contour_max)
            if MOMENT["m00"] > 0:
                center_x = int(MOMENT["m10"] / MOMENT["m00"])
                center_y = int(MOMENT["m01"] / MOMENT["m00"])
                centroids.append((center_x, center_y))
        else:
            print("Leaf has no contour")
            mask[idx] = 0

    return centroids, mask, areas


def clean_mask(leafs):
    """
    Function for removing small clusters from image masks

    args:
        leafs (numpy array): 1080x1440x3 array with data pertaining
                to leaves and their mask data

    returns:
        claned_mask (numpy array): 1080x1440 array with the cleaned
            mask. Individual masks with an area of less than 200 pixels
            are removed
    """
    index = np.unique(leafs[:, :, 3])
    mask = leafs[:, :, 3]
    coordinates = []
    for i, _ in enumerate(index):
        if i == 0:
            continue
        mask_ = mask == index[i]
        mask_ = np.where(mask_, mask_ >= 1, 0)
        labels = measure.label(mask_)
        props = measure.regionprops(labels)
        for prop in props:
            if prop.area <= 200:
                pixels = prop.coords
                coordinates.append(pixels)
    for i, coord in enumerate(coordinates):
        coor_ = np.array(coord)
        mask[(coor_[:, 0], coor_[:, 1])] = 0

    cleaned_mask = mask
    return cleaned_mask


def process_area(i, index_, mask_, kernel_):
    """
    Find the graspable area of a given leaf

    Args:
        i (integer): id of leaf being processed
        index_ (numpy array): Leaf location indices
        mask_ (numpy array): Mask being processed
        kernel_ (list): Kernel size based on mean leaf depth

    Returns:
        i_ (list): x coordinates for graspable regions
        j_ (list): y coordinates for graspable regions
        i (integer): id of leaf being processed
    """
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
    return i_, j_, i


def compute_graspable_areas(kernel, mask):
    """
    Function for executing 'process_area()' in parallel

    Args:
        kernel (list): List of kernel sizes to be convolved with each leaf
        mask (2d Numpy Array): Leaf mask with unique id per leaf

    Returns:
        graspable_areas (2d Numpy Array): A leaf mask with the area of each leaf
            adjusted to only include their graspable area
    """
    kernel_ = kernel
    mask_ = mask
    index_ = np.unique(mask_)
    graspable_areas = np.zeros((1080, 1440))

    results = Parallel(n_jobs=-1)(
        delayed(process_area)(i, index_, mask_, kernel_) for i in range(1, len(index_))
    )
    for i_, j_, i in results:
        graspable_areas[i_, j_] = i

    return graspable_areas


def find_number_of_points_in_radius(points, center, radius):
    """
    Finds the number of points in a radius.

    Args:
      points: A list of points.
      center: The center of the circle.
      radius: The radius of the circle.

    Returns:
      The number of points in the radius.
    """
    count = 0
    for point in points:
        if (
            math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
            <= radius
        ):
            count += 1
    return count


def combine_sdf(mask, leaf_regions, bins):
    """
    Function for combining multiple SDFs of the same dimension together. The SDFs
    are added together and then averaged/normalized at the output.

    Args:
        mask (2D Numpy Array): Leaf mask detailing where leaves are in the camera
            frame.
        leaf_regions (2D Numpy Array): Leaf mask detailing viable areas on each leaf
            to grasp.

    Returns:
        SDF (2d Numpy Array): SDF array where negative regions indicate the presence of
            leaves and positive regions indicate no leaves.
    """
    i_ = 0
    sum_ = 0
    SDF = np.zeros(leaf_regions.shape)
    counter = 0
    for i in bins:
        sum = sum_ + 1
        mask_ind = np.arrange(sum_, sum)
        mask_img = np.isin(mask, mask_ind) * leaf_regions
        sdf = get_sdf(mask_img)
        SDF += sdf
        sum_ = sum
        counter += counter
    SDF = SDF / counter
    SDF = (SDF - np.amin(SDF)) / (np.amax(SDF) - np.amin(SDF))
    return SDF


def get_sdf(mask):
    """
    Finds the SDF of a given leaf mask. If there are no leaves in the mask, will
    just return an empty array instead.

    Args:
        mask (2D Numpy Array): Leaf mask detailing where leaves are in the camera
            frame.

    Returns:
        sdf_ (2D Numpy Array): SDF for the provided mask.
    """
    mask_ = np.where(mask, mask == 0, 1)
    if np.count_nonzero(mask_ == 0) == 0:
        return np.zeros(mask_.shape)
    sdf_ = skfmm.distance(mask_, dx=1)
    return sdf_


def find_maxmin_centroid_dist(centroids, min_global, max_global):
    """
    Calculates the distance of each centroid from the leaf SDF's global
    minima and maxima

    Args:
        centroids (list): A list of x,y coordinate tuples corresponding to the
        calculated leaf centroids
        min_global (Numpy Array): Index of the global minimum value for the leaf SDF
        max_global (Numpy Array): Index of the global maximum value for the leaf SDF

    Returns:
        data (2D Numpy Array): Array of distances from global minima and maxima for each
            centroid
    """
    B = np.asarray(centroids)
    B = np.insert(B, 0, values=(min_global[1], min_global[0]), axis=0)
    pdist_B = np.array(pdist.euclidean_distances(B))
    A = np.asarray(centroids)
    A = np.insert(A, 0, values=(max_global[1], max_global[0]), axis=0)
    pdist_A = np.array(pdist.euclidean_distances(A))
    data = np.vstack(([pdist_B[0, :], pdist_A[0, :]])).transpose()
    data = np.delete(data, 0, axis=0)
    return data


def mean_mask_depth(leafs, normalized=False):
    """
    Finds the mean depth for every leaf in the mask

    Args:
        leafs (3D Numpy Array): A Numpy array containing data about
            plants in the image on a per-pixel basis.
        normalized (bool): Optional input to have the functions
            output be min-max normalized.

    Returns:
        depth_list (list): List of mean depth for each leaf in the image. Will share
            the same index as running np.unique() on the leaf mask.
    """
    unique = np.unique(leafs[:, :, 3])
    depth_list = []

    for i in range(1, len(unique)):
        leaf_indv = leafs[:, :, 3] == unique[i]
        depth_indv = leafs[:, :, 2] * leaf_indv
        mean_depth = np.mean(depth_indv[np.nonzero(depth_indv)])
        depth_list.append(mean_depth)

    depth_list = np.asarray(depth_list)
    if normalized:
        depth_list_norm = depth_list - np.min(depth_list)
        depth_list_norm /= np.max(depth_list) - np.min(depth_list)
        return depth_list_norm
    return depth_list


def find_tall_leaves(depth_list, leafs):
    GRASPER_CLEARANCE = 0.02
    unique = np.unique(leafs[:, :, 3])
    depth_hist = np.histogram(depth_list, bins="auto")
    depth_hist_median = np.where(depth_hist[0] == np.max(depth_hist[0]))[0][0]
    low_depth = np.where(
        depth_list < (depth_hist[1][depth_hist_median] - GRASPER_CLEARANCE)
    )
    tall_leaves_idx = unique[low_depth]
    idx_list = []
    for i in range(1, len(tall_leaves_idx)):
        idx_list.append(np.where(unique == tall_leaves_idx[i])[0][0])

    tall_leaves_mask = np.zeros(np.shape(leafs[:, :, 3]))

    for i, _ in enumerate(unique[idx_list]):
        tall_leaves = leafs[:, :, 3] == unique[idx_list][i]
        tall_leaves_mask += tall_leaves

    return tall_leaves_mask
