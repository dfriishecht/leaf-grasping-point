import skfmm
import click
import time
import os, os.path
import numpy as np
import open3d as o3d
from paretoset import paretoset
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.lines as lines
HOME_DIR = os.path.expanduser('~')


import modules.conv_helpers as cvh
import modules.mask_helpers as mh
import modules.plant_pcd_helpers as pcdh


@click.command()
@click.option('--data_num', type=int, default='5', help='Index of data to be processed')
@click.option('--viz', type=bool, default=False, help="Toggle for visualizing script outputs")
@click.option('--output_dir', type=str, default='data_output', help='Directory to save data')
def main(data_num: str, viz, output_dir):
    tot_t = time.time()
    # Combine mask and depth data together to segment out leaves
    pcd_path = "data/pointclouds/"+f"{data_num}"+".pcd"
    mask_path = "data/images/aggrigated_masks"+f"{data_num}"+".png"
    image_path = "data/images/left_rect"+f"{data_num}"+".png"
    leafs, depth = pcdh.apply_depth_mask(pcd_path, mask_path, image_path, plot=False)
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
    o3d.io.write_point_cloud(f"{HOME_DIR}/{output_dir}/cleaned{data_num}.pcd", cl, format='auto')
    pcdh.compute_normals(processed_pcd)
    sqrt_dist = np.sum((processed_pcd.normals[999]) ** 2, axis=0)
    dist = np.sqrt(sqrt_dist)
    normal_orientation = abs(np.asarray(processed_pcd.normals)[:, 2])

    normal_corrected = np.asarray(processed_pcd.normals)
    for normal in normal_corrected:
        if normal[2] < 0:
            normal *= -1
    print(normal_corrected.shape)
    orientation_color = np.zeros((len(normal_orientation), 3))
    orientation_color[:, 0] = normal_orientation
    orientation_color[:, 1:] = 0
    processed_pcd.colors = o3d.utility.Vector3dVector(orientation_color)
    #############################################################


    # Estimate leaf flatness based on normal vectors
    leaf_flatness = np.zeros((1555200, 1))
    leaf_normals = np.zeros((1555200, 3))
    j = 0
    for i, _ in enumerate(inverse_index[0]):
        current_index = inverse_index[0][i]
        leaf_flatness[current_index, 0] = normal_orientation[j]
        leaf_normals[current_index, :] = normal_corrected[j, :]
        j += 1

    leaf_flatness = np.reshape(leaf_flatness, (1080, 1440, 1))
    leaf_normals = np.reshape(leaf_normals, (1080, 1440, 3))
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


    max_leaf = 20000
    for idx, sol in enumerate(paretoset_sols):
        if sol[1] < max_leaf:
            max_leaf = idx

    opt_point = centroids[opt_leaves[max_leaf]]

    real_grasp_coord = leafs[opt_point[1], opt_point[0], 0:3]
    real_grasp_coord = np.round(real_grasp_coord, 4)
    grasp_normal = np.round(leaf_normals[opt_point[1], opt_point[0], :], 4)
    
    SDF_max = np.unravel_index(SDF.argmax(), SDF.shape)

    print(SDF_max)
    print(opt_point)
    x_dist = SDF_max[1] - opt_point[0]
    y_dist = SDF_max[0] - opt_point[1]

    tot_dist = np.sqrt((x_dist**2)+(y_dist**2))

    x_dist /= tot_dist * .005
    y_dist /= tot_dist * .005
    target_vec = (int(x_dist+opt_point[0]), int(y_dist+opt_point[1]))
    print(target_vec)

    depth = np.reshape(depth, (1080, 1440, 3))
    real_target_vec = depth[target_vec[1], target_vec[0], :]
    real_target_vec[2] = real_grasp_coord[2]
    real_target_vec = np.round(real_target_vec, 4)
    
    print(f"Normal Vector: {grasp_normal} \n Grasp Point: {real_grasp_coord} \n Approach: {real_target_vec}")
    # coord_test = np.asarray(processed_pcd.points)
    # coord_test = np.zeros((100, 3))
    # np.append(coord_test, grasp_normal)
    # np.append(coord_test, real_target_vec)
    # np.append(coord_test, real_grasp_coord)
    # coord_test = real_grasp_coord
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(coord_test)
    # o3d.visualization.draw_geometries([pcd])
    

    print(f"Total runtime: {time.time()-tot_t:.3f} s")

    # Optional toggle for visualizing processed leaves
    fig, ax = plt.subplot_mosaic([
            ['viable regions', 'sdf', 'points']
        ], figsize=(15,10))
    ax['points'].imshow(Image.open(image_path))
    x1, y1 = [opt_point[0], opt_point[1]]
    x2, y2 = [SDF_max[1], SDF_max[0]]
    ax["points"].axline((x1, y1), (x2, y2), marker='o')
    ax["points"].plot(target_vec[0], target_vec[1], marker='*', markersize=12)
    for i in opt_leaves:    
        ax['points'].plot(centroids[i][0], centroids[i][1], "r*")
    if tall_presence:
        for i in opt_leaves_tall:
            ax["points"].plot(centroids_tall[i][0], centroids_tall[i][1], "b*")
    ax["points"].plot(opt_point[0], opt_point[1], "bo", markersize=8)
    ax["points"].plot(opt_point[0],opt_point[1], "r+", markersize=8)
    ax["points"].set_title("Selected Points (Blue = Tall Outliers)")
    ax["viable regions"].imshow(viable_leaf_regions)
    ax["viable regions"].set_title(f"Viable Leaf Regions (blend: {ALPHA})")
    ax["sdf"].imshow(SDF)
    ax["sdf"].set_title("SDF")
    fig.savefig(f"{HOME_DIR}/{output_dir}/viz{data_num}.png")
    if viz:
        plt.show()



    #TODO: Select single leaf from selection of leaves
        # Option 1: Choose leaf point with the greatest maxima -> Done
        # Option 2: Evaluate this maxima as a score alongside height of the point

   

    # 
    #TODO: Create approach vector for the selected grasping point 



if __name__ == "__main__":
    main()