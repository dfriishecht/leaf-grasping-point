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
* TODO: A final leaf is chosen as the optimal leaf, and an approach vector to the leaf is calculated.

## 2. Installation
Tested on Ubuntu 20.04.06

Install Conda env
```bash
conda env create -f conda_environment.yml
```
Activate Conda env
```bash
conda activate leaf_processing
```

## 3. Running the Pipeline
Run the script in terminal with `python pipeline_run.py`. Currently there are five different sets of data to choose from. Specify which one with the optional `--data_num` argument.

```bash
python pipeline_run.py
```
```bash
Usage: pipeline_run.py [OPTIONS]
```

Options:
    --data_num   INTEGER    Index of data to be processed (Range 1-5)
    --viz        BOOL       Toggle for visualizing script outputs

## 4. Note on the Jupyter notebook `pipeline_prototyping.ipynb`
There also exists a Jupyter notebook within this repo that is less organized than just running `pipeline_run.py`. However, it does contain more visualizations at points within the notebook, so if you wish to take a more granular look at somethhing within the code, the notebook would be a good place to explore.
