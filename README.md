# CSIT5210-project
This is the group project for CSIT5210 2023 Fall HKUST. During the process of completing this project, we would like to express our special gratitude to Andrew Draganov, et al. for providing valuable resources for our work. Their open source code [GitHub]([https://www.runoob.com](https://github.com/Andrew-Draganov/dc_dist/blob/master)) has played an important role in our understanding and implementation of related algorithms.

Below are instructions about how to run the project.

## 1. Load Data
In this project, we use the "coil-100" dataset and you can use the `get_dataset()` function in the `get_data.py` file to generator the data.

## 2. Caculate dc-dist
The algorithms in this project is based on the "density-connectivity based distance". You can use the `get_dc_dist_matrix()` function in the `distance_metric.py` file to get the dc-dist matrix.

## 3. DBSCAN*
This Python script `DBSCAN*_dc.py` applies the DBSCAN* clustering algorithm on the COIL-100 dataset using libraries like numpy, PIL, sklearn, and matplotlib. It preprocesses the data, applies PCA for dimensionality reduction, calculates an optimal eps value, and assigns DBSCAN* labels. The results are visualized as a scatter plot. Before running, download the COIL-100 dataset from Kaggle and ensure the get_dc_dist_matrix function is correctly defined and imported.

## 4. Spectral Clustering
Here is the instructions about how to set up the dataset and run the spectral_clustering.

- First download the dataset by cloning from the github or in the following link : https://www.kaggle.com/datasets/jessicali9530/coil100
- Make sure to place the dataset on the right place, or change 'the path' in the code.

- You can now run the script.

## 5. k-Center

Description of source files:

- In density_tree.py, the DensityTree class embeds the data points into an ultrametric’s density-connectivity tree representation.

- In cluster_tree.py, the cluster_tree function computes the ultrametric’s density-connectivity tree representation of testing dataset into k clusters. With q-coverage settings, the copy_tree function prunes the data tree to only retain the nodes that have at least q children. And it also stores the infomation of original child nodes for each node for tree restoring process later. Finally, the finalize_clusters function restores the optimal solution for the original tree from the pruned tree. 

Instructions:

- Please replace the "cluster_tree.py", "density_tree.py" and "datagen.py" with three corresponding updated .py files from directory "k_center_implementation_only" into original directory "dc_dist" from github repository https://github.com/Andrew-Draganov/dc_dist.

- Prepare for datasets (e.g., "synth", "coil100", "circles", "mnist") referred to experiment_utils/get_data.py.
    
  - For synth dataset, please run the following command "python3 datagen.py 1 10389 2 5" or "python datagen.py 1 10389 2 5" in main directory "dc_dist" with arguments seed=1, number of points (without noise)=10389, dimensionality=2, cluster number=5. You can also generate other datasets with different parameters by specifying other argument values. Please refer to datagen.py for details.
    
  - For coil100 dataset, create a diretory named "data" under the main directory "dc_dist" and create a sub-directory named "coil-100" under "data" dir and put coil100 data from http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip.
    
  - For minst dataset, create a directory named "mnist" under the same "data" directory and go to http://yann.lecun.com/exdb/mnist/ to download the minst data by using ./bin/mnist_get_data.sh.

- Then, go to cluster_dataset.py, you can specify a testing dataset between line 48 line 57.

- Make sure your environment has installed all required packages/libraries.

- Run following command "python3 cluster_dataset.py" or "python cluster_dataset.py" in main directory "dc_dist" to see the normalized_mutual_info_score bettween each pair of clustering algorithms ('k-Center on DC-dists', 'DBSCAN*', and 'Ultrametric Spectral Clustering') and corresponding epsilons of each of three algorithms.

- For other experiments related to k-center algorithm such as k-epsilon relationship discovering or distances visualization , please refer to the "k_vs_epsilon.py", "noise_robustness.py"， "distances_plot.py" and "compare_clustering.py" in in main directory "dc_dist" for details.
