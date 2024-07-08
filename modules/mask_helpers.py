"""
Helper module for performing operations on the leaf masks
"""

import numpy as np
from skimage import measure
import cv2
import math
import skfmm
import sklearn.metrics.pairwise as pdist


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
        mask_ind = np.arange(sum_, sum)
        mask_img = np.isin(mask, mask_ind) * leaf_regions
        sdf = get_sdf(mask_img)
        SDF += sdf
        sum_ = sum
        counter += 1

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
    """
    From a given list of mean depths, create a new mask with outlier leaves. These
        are leaves who's height is greater than the median by a specified clearance for the
        robot end-effector.

    Args:
        depth_list (list): List of mean depths for each leaf in the image
        leafs (3D Numpy Array): Numpy array containing information about each leaf
            on a pixel-by-pixel basis.

    Returns:
        tall_leaves_mask (2D Numpy Array): Mask containing only tall leaves.
    """
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


def compute_minmax_dist(centroids, min, max):
    """
    Computes the euclidean distance of each centroid from the image's
    minimum and maximum SDF values.

    Args:
        centroids (list): List of tuples with the xy coords for each centroid.
        min (tuple): xy coords of global minimum.
        max (tuple): xy coords of global maximum.

    Returns:
        data (numpy array): List containing the distance from both global minima and maxima
            per centroid.
    """

    B = np.asarray(centroids)
    B = np.insert(B, 0, values=(min[1], min[0]), axis=0)

    pdist_B = np.array(pdist.euclidean_distances(B))

    A = np.asarray(centroids)
    A = np.insert(A, 0, values=(max[1], max[0]), axis=0)

    pdist_A = np.array(pdist.euclidean_distances(A))

    data = np.vstack(([pdist_B[0, :], pdist_A[0, :]])).transpose()
    data = np.delete(data, 0, axis=0)
    return data
