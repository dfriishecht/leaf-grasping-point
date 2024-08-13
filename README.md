# Find the Optimal Leaf Grapsing Point

## 1. Overview
This package implements an optimal leaf grasping location pipeline. This is done as follows:

* A semantic leaf mask is projected over a point cloud to cull all points not part of a leaf.
* The leaves are adaptivly convovled based on their depth with a kernel representing a microneedle array. The result will be graspable leaf areas.
* Each point's normal vector is calculated and used to identify flat portions of each leaf.
* Graspable area is combined with estimated flatness to determine the viable grasping regions of each leaf.
* The centroid of each viable grasping area is calculated.
* The global "leaf clutter" minima and maxima are calculated, and leaves within the paretoset are selected as candidate leaves.
* From this set of candidate leaves, the one closest to the clutter maxima is selected.
* An approach vector to this final leaf is calculated, resulting in a grasp vector, grasp normal vector, and approach vector.

## 2. Installation
Tested on Ubuntu 20.04.06

Install Conda env
```bash
conda env create -f conda_environment.yaml
```
Activate Conda env
```bash
conda activate leaf_processing
```

## 3. Running the Pipeline
Run the script in terminal with `python pipeline_run.py`. Currently there are five different sets of data to choose from. Specify which one with the optional `--data_num` argument. This primarily functions as a demo script, and works with the provided data

```bash
python pipeline_run.py
```
```bash
Usage: pipeline_run.py [OPTIONS]


Options:
    --data_num   INTEGER    Index of data to be processed (Range 1-5)
    --viz        BOOL       Toggle for visualizing script outputs
```

## 4. Running the Pipeline on a Folder of Data
Run `python collect_results.py` instead. This script has optional arguments to specify the folder to pull data from and save data to. Unlike `pipeline_run.py` this scrip will allow you to run through as many images as exist in whatever folder you provide. Just make sure that you are providing the required point cloud, mask, and rgb image.

```bash
python collect_results.py
```
```bash
Usage: collect_results.py [OPTIONS]

Options;
    --input_directory STRING     Folder path for input data. Don't include the home directory path
    --output_directory STRING    Folder path for saving results. Don't include the home directory path
```
