# image-processing-exploration

## 1. Overview
This package implements an optimal leaf grasping location pipeline. This is done as follows:

* A semantic leaf mask is projected over a point cloud to cull all points not part of a leaf.
* The leaves are adaptivly convovled based on their depth with a kernel representing a microneedle array. The result will be graspable leaf areas.
* Each point's normal vector is calculated and used to identify flat portions of each leaf.
* Graspable area is combined with estimated flatness to determine the viable grasping regions of each leaf.
* The centroid of each viable grasping area is calculated.
* The global "leaf clutter" minima and maxima are calculated, and leaves within the paretoset are selected as candidate leaves.
* The same process is done for leaves at the top of the plants.
* A final leaf is chosen as the optimal leaf, and an approach vector to the leaf is calculated. <- TODO

## 2. Installation
Tested on Windows 10 <- TODO: Test on Ubuntu 20.04

Install with Conda
```bash
conda env create -f conda_environment.yml
```

```bash
conda activate leaf_processing
```

## 3. Running the Pipeline
For now, open `pipeline_run.py` in a python interpreter and run. You can adjust the input data used at the top of the script. Soon I'll have this setup properly as a package, at which point you can just run the file in terminal with `python pipeline_run.py`.
