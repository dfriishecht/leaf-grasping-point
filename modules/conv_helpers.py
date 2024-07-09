"""
Helper module for performing various convolution-focused tasks.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from joblib import Parallel, delayed


def get_kernels(depth_image, mask_image):
    """
    Calculate and get all of the convolution kernels needed for each leaf.
    Masks out each leaf individually to calculate its respective kernel size.

    Args:
        depth_image (2d numpy array): Array of depth values for each leaf.
        mask_image (2d numpy array): Array of leaf id values for each leaf.

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
        kernels.append(kernel_.astype("uint8"))
    print("all kernels: ", kernels)
    return kernels


def kernel_size(depth_mask, plot=False):
    """
    Helper function to calculate the size of each kernel. The size is
    originally based on the physical dimensions of the microneedle array.
    Then, the mean depth of each leaf is used to figure out how large the array is
    relative to each leaf in the camera frame, which will scale the kernel accordingly.

    args:
        depth_mask (2d numpy array): The masked depth data of a single leaf.
        plot (bool): Optional argument to plot each kernel.

    returns:
        kernel (int): Width and height of the calculated kernel in pixels.
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

    kernel = np.round(np.average(([sz_1, sz_2, sz_3, sz_4])))

    if plot:
        plt.imshow(depth_mask)
        plt.plot(np.round(xc[0]), np.round(xc[1]), "r.")
        plt.plot(np.round(x1[0]), np.round(x1[1]), "r.")
        plt.plot(np.round(x2[0]), np.round(x2[1]), "r.")
        plt.plot(np.round(x3[0]), np.round(x3[1]), "r.")
        plt.plot(np.round(x4[0]), np.round(x4[1]), "r.")
        plt.show()

    return kernel


def process_area(i, index_, mask_, kernel_):
    """
    Find the graspable area of a given leaf.

    Args:
        i (integer): id of leaf being processed.
        index_ (numpy array): Leaf location indices.
        mask_ (numpy array): Mask being processed.
        kernel_ (list): Kernel size based on mean leaf depth.

    Returns:
        i_ (list): x coordinates for graspable regions.
        j_ (list): y coordinates for graspable regions.
        i (integer): id of leaf being processed.
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
        graspable_area, graspable_area < np.amax(graspable_area) * 0.95, 1
    )  # remove blurry parts
    graspable_area_ = np.logical_not(graspable_area).astype(int)
    i_, j_ = np.where(graspable_area_ == np.amax(graspable_area_))
    return i_, j_, i


def compute_graspable_areas(kernel, mask):
    """
    Function for executing 'process_area()' in parallel.

    Args:
        kernel (list): List of kernel sizes to be convolved with each leaf
        mask (2d Numpy Array): Leaf mask with unique id per leaf.

    Returns:
        graspable_areas (2d Numpy Array): A leaf mask with the area of each leaf
            adjusted to only include their graspable area.
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
