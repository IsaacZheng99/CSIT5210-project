import numpy as np
import os
import glob
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from distance_metric import get_dc_dist_matrix

def load_coil100_data(directory=None):
    """
    This is using the coil100 dataset available on Kaggle at https://www.kaggle.com/datasets/jessicali9530/coil100
    Using it requires manually unzipping it into a directory
    """
    if directory is None:
        directory = os.path.join('data', 'coil-100')
    filelist = glob.glob(os.path.join(directory, '*.png'))
    if not filelist:
        raise ValueError('Coil 100 data directory {} is empty!'.format(directory))

    points = np.zeros([7200, 128, 128, 3])
    labels = np.zeros([7200])
    for i, fname in enumerate(filelist):
        image = np.array(Image.open(fname))
        points[i] = image

        image_name = os.path.split(fname)[-1]
        class_label = [int(c) for c in image_name[:6] if c.isdigit()]
        class_label = np.array(class_label[::-1])
        digit_powers = np.power(10, np.arange(len(class_label)))
        class_label = np.sum(class_label * digit_powers)
        labels[i] = class_label

    points = np.reshape(points, [7200, -1])
    return points, labels

# load the dataset
points, labels = load_coil100_data()

scaler = StandardScaler()
points_scaled = scaler.fit_transform(points)

pca = PCA(n_components=2)
points_pca = pca.fit_transform(points_scaled)


# Fit the nearest neighbors estimator to the data
k=4
nn = NearestNeighbors(n_neighbors=k)
nn.fit(points_pca)

# Get the distances and indices of k-nearest neighbors
distances, indices = nn.kneighbors(points_pca)

# Get the distances to the kth nearest neighbor
k_dist = distances[:, -1]

# Sort the distances
k_dist_sorted = np.sort(k_dist)

# Compute the "elbow" point in the k-distance graph
dx = k_dist_sorted[-1] - k_dist_sorted[0]
dy = len(k_dist_sorted) - 0
ds = np.sqrt(dx*dx + dy*dy)

distances_from_line = (dx*(np.arange(len(k_dist_sorted))) - dy*(k_dist_sorted-k_dist_sorted[0])) / ds
elbow_index = np.argmax(distances_from_line)

# The "elbow" point is the point of maximum distance from the line
eps = k_dist_sorted[elbow_index]

dc_dist_matrix = get_dc_dist_matrix(points_pca, n_neighbors=5, min_points=5)

# perform clustering using DBSCAN
# the eps parameter is chosen based on the k-distance graph
dbscan = DBSCAN(eps=eps, min_samples=k, metric='precomputed')
dbscan.fit(dc_dist_matrix)

# Get the labels
labels = dbscan.labels_

# Get the core sample indices
core_sample_indices = dbscan.core_sample_indices_

# Initialize a new label array with all elements set to -1 (noise)
labels_star = np.full_like(labels, -1)

# For each core sample, set its label in labels_star to its label from DBSCAN
for i in core_sample_indices:
    labels_star[i] = labels[i]

# labels_star now contains the DBSCAN* labels

noise = points_pca[labels_star == -1]
non_noise = points_pca[labels_star != -1]

plt.figure(figsize=(10,8))

plt.scatter(noise[:, 0], noise[:, 1], c='gray', alpha=0.8, label="Noise")
plt.scatter(non_noise[:, 0], non_noise[:, 1], c=labels_star[labels_star != -1], cmap='viridis')

plt.title('DBSCAN* Clustering Results (COIL-100) - minPts:{}, epsilon:{}'.format(k, eps))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()