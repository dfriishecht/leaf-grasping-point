import sys
sys.path.append('~')

import argparse
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from paretoset import paretoset
from PIL import Image
import time
import skfmm
import open3d as o3d
from matplotlib import pyplot as plt
import os, os.path
from pipeline_run import main

import modules.conv_helpers as cvh
import modules.mask_helpers as mh
import modules.plant_pcd_helpers as pcdh

HOME_DIR = os.path.expanduser('~')

def max_data_count():
    DIR = 'SDF_OUT/temp'
    img_count = len([name for name in os.listdir(HOME_DIR+'/'+DIR) 
               if os.path.isfile(os.path.join(HOME_DIR+'/'+DIR, name))])
    img_count /= 8
    img_count = int(img_count)
    print(f"Total datasets to process: {img_count}")
    return img_count


def main(data_num, viz, output_dir):
    tot_t = time.time()
    # Combine mask and depth data together to segment out leaves
    pcd_path = f"{HOME_DIR}/SDF_OUT/temp/{data_num}.pcd"
    mask_path = f"{HOME_DIR}/SDF_OUT/temp/aggrigated_masks{data_num}.png"
    image_path = f"{HOME_DIR}/SDF_OUT/temp/left_rect{data_num}.png"
    leafs = pcdh.apply_depth_mask(pcd_path, mask_path, image_path, plot=False)
    mask = mh.clean_mask(leafs)
    leafs[:, :, 3] = mask
    ############################################################


    # Convolve each leaf with microneedle array-scaled kernels to get graspable area
    print("Computing Graspable Area")
    t = time.time()
    depth_image = leafs[:, :, 2].astype("float32")
    mask_image = leafs[:, :, 3].astype("uint8")
    kernels = cvh.get_kernels(depth_image, mask_image)
    graspable_mask = cvh.compute_graspable_areas(kernels, mask_image)
    print(f"Computation took {time.time()-t:.3f} s")
    ############################################################


    # Compute the normal vectors of every leaf
    binary_graspable_mask = graspable_mask >= 1
    leafs[:, :, 3] = leafs[:, :, 3] * binary_graspable_mask
    leafs_ = np.reshape(leafs[:, :, 0:3], (1555200, 3))
    index = np.argwhere(leafs_ == [0, 0, 0])
    inverse_index = np.nonzero(leafs_[:, 2])
    leafs_ = np.delete(leafs_, index, 0)
    processed_pcd = o3d.geometry.PointCloud()
    processed_pcd.points = o3d.utility.Vector3dVector(leafs_)
    cl, id = processed_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    o3d.io.write_point_cloud(f"{HOME_DIR}/{output_dir}/cleaned{data_num+71}.pcd", cl, format='auto')
    pcdh.compute_normals(processed_pcd)
    sqrt_dist = np.sum((processed_pcd.normals[999]) ** 2, axis=0)
    dist = np.sqrt(sqrt_dist)
    normal_orientation = abs(np.asarray(processed_pcd.normals)[:, 2])
    orientation_color = np.zeros((len(normal_orientation), 3))
    orientation_color[:, 0] = normal_orientation
    orientation_color[:, 1:] = 0
    processed_pcd.colors = o3d.utility.Vector3dVector(orientation_color)
    #############################################################


    # Estimate leaf flatness based on normal vectors
    leaf_flatness = np.zeros((1555200, 1))
    j = 0
    for i, _ in enumerate(inverse_index[0]):
        current_index = inverse_index[0][i]
        leaf_flatness[current_index, 0] = normal_orientation[j]
        j += 1

    leaf_flatness = np.reshape(leaf_flatness, (1080, 1440, 1))
    leafs = np.concatenate((leafs, leaf_flatness), axis=2)
    #############################################################


    # Combine graspable area with flat area to determine optimal leaf grapsing locations
    ALPHA = 0.4  # Adjustable parameter to change blend between grasping area and flat area
    smooth_section = leafs[:, :, 4]
    leaf_selection_a = ALPHA * smooth_section + (1 - ALPHA) * binary_graspable_mask

    leaf_selection_ab = np.where(
        leaf_selection_a, leaf_selection_a >= np.amax(leaf_selection_a) * 0.95, 0
    )
    leafs[:, :, 3] *= leaf_selection_ab
    viable_leaf_regions = mh.clean_mask(leafs)
    ###############################################################

    # Calculate the mean depth of each leaf and identify tall leaves
    depth_list = mh.mean_mask_depth(leafs)
    depth_list_norm = mh.mean_mask_depth(leafs, normalized=True)
    tall_leaves = mh.find_tall_leaves(depth_list, leafs)
    tall_presence = False
    if sum(sum(tall_leaves)) > 0:
        tall_presence = True
    ###############################################################


    # Find the SDF of the leaves to calculate global clutter minima and maxima
    leafs[:, :, 3] = viable_leaf_regions
    viable_leaf_regions = mh.clean_mask(leafs)
    cleaned_masks = viable_leaf_regions >= 1
    cleaned_masks = np.where(cleaned_masks, cleaned_masks == 0, 1)
    SDF = skfmm.distance(cleaned_masks, dx=1)
    if tall_presence:
        cleaned_masks_tall = tall_leaves >= 1
        cleaned_masks_tall = np.where(cleaned_masks_tall, cleaned_masks_tall == 0, 1)
        SDF_X = skfmm.distance(cleaned_masks_tall, dx=1)
        min_tall = np.unravel_index(SDF_X.argmin(), SDF_X.shape)
        max_tall = np.unravel_index(SDF_X.argmax(), SDF_X.shape)

    min_global = np.unravel_index(SDF.argmin(), SDF.shape)
    max_global = np.unravel_index(SDF.argmax(), SDF.shape)
    #################################################################


    # Find the centroid of each leaf
    centroids, mask, areas = mh.get_centroids(viable_leaf_regions.astype("uint8"))
    leafs[:, :, 3] = mask

    if tall_presence:
        tall_leaves = tall_leaves * viable_leaf_regions
        centroids_tall, mask_tall, areas_tall = mh.get_centroids(
            tall_leaves.astype("uint8")
        )
    #################################################################


    # Find the distance of each centroid from the image's SDFminima and maxima
    data = mh.compute_minmax_dist(centroids, min_global, max_global)
    if tall_presence:
        data_tall = mh.compute_minmax_dist(centroids_tall, min_tall, max_tall)
    #################################################################


    # Use the distances to determine the optimal leaves to choose based on their pareto set
    mask = paretoset(data)
    paretoset_sols = data[mask]
    res = mask
    opt_leaves = np.where(res == True)[0]

    if tall_presence:
        mask_tall = paretoset(data_tall)
        paretoset_sols_tall = data_tall[mask_tall]
        res_tall = mask_tall
        opt_leaves_tall = np.where(res_tall == True)[0]
    #################################################################

    print(f"Total runtime: {time.time()-tot_t:.3f} s")

    # Optional toggle for visualizing processed leaves
    fig, ax = plt.subplot_mosaic([
            ['viable regions', 'sdf', 'points']
        ], figsize=(15,10))
    ax['points'].imshow(Image.open(image_path))
    for i in opt_leaves:
        ax['points'].plot(centroids[i][0], centroids[i][1], "r*")
    if tall_presence:
        for i in opt_leaves_tall:
            ax["points"].plot(centroids_tall[i][0], centroids_tall[i][1], "b*")

    ax["points"].set_title("Selected Points (Blue = Tall Outliers)")
    ax["viable regions"].imshow(viable_leaf_regions)
    ax["viable regions"].set_title(f"Viable Leaf Regions (blend: {ALPHA})")
    ax["sdf"].imshow(SDF)
    ax["sdf"].set_title("SDF")
    fig.savefig(f"{HOME_DIR}/{output_dir}/viz{data_num+71}.png")
    np.save(f"{HOME_DIR}/{output_dir}/centroids{data_num+71}.png", opt_leaves)
    if tall_presence:
        np.save(f"{HOME_DIR}/{output_dir}/tall_centroids{data_num+71}.png", opt_leaves_tall)
    if viz:
        plt.show()

def collect_data(args):
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)
    img_count = max_data_count()
    for data_num in tqdm(range(0, img_count)):
        main(data_num, viz=False, output_dir=output_directory)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_directory', help='directory to save output', default='data_output')
    args = parser.parse_args()
    collect_data(args)
