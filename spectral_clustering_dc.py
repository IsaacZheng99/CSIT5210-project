import numpy as np
import os
from PIL import Image
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from scipy.linalg import expm
import itertools
import numba
from distance_metric import get_dc_dist_matrix

def load_coil100_data(directory=None):
    """
    This is using the coil100 dataset available on Kaggle at https://www.kaggle.com/datasets/jessicali9530/coil100
    Using it requires manually unzipping it into a directory
    """
    if directory is None:
        directory = os.path.join('dataset', 'coil-100/coil-100')
    pickled_path = os.path.join(directory, 'pickled_coil.npy')
    if os.path.exists(pickled_path):
        dataset = np.load(pickled_path, allow_pickle=True)[()]
        return dataset['points'], dataset['labels']

    print('Could not find pickled dataset at {}. Loading from png files and pickling...'.format(pickled_path))
    filelist = glob.glob(os.path.join(directory, '*.png'))
    if not filelist:
        raise ValueError('Coil 100 data directory {} is empty!'.format(directory))

    points = np.zeros([7200, 128, 128, 3])
    labels = np.zeros([7200])
    for i, fname in tqdm(enumerate(filelist)):
        image = np.array(Image.open(fname))
        points[i] = image

        image_name = os.path.split(fname)[-1]
        # This assumes that your images are named objXY__i.png
        #   where XY are the class label and i is the picture angle
        class_label = [int(c) for c in image_name[:6] if c.isdigit()]
        class_label = np.array(class_label[::-1])
        digit_powers = np.power(10, np.arange(len(class_label)))
        class_label = np.sum(class_label * digit_powers)
        labels[i] = class_label

    points = np.reshape(points, [7200, -1])
    np.save(pickled_path, {'points': points, 'labels': labels})
    return points, labels

# Load COIL-100 dataset
points, labels = load_coil100_data()

# Standardize the features
scaler = StandardScaler()
points_scaled = scaler.fit_transform(points)

# Apply PCA for dimensionality reduction (optional)
pca = PCA(n_components=2)
points_pca = pca.fit_transform(points_scaled)

dc_dist_matrix = get_dc_dist_matrix(points_pca, n_neighbors=5, min_points=5)

# Perform spectral clustering using the DC-Dist matrix
n_clusters = len(np.unique(labels))  # Number of clusters based on unique labels
spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
cluster_labels = spectral.fit_predict(dc_dist_matrix)

# Visualize the results
plt.scatter(points_pca[:, 0], points_pca[:, 1], c=cluster_labels, cmap='viridis')
plt.title('Spectral Clustering Results (COIL-100)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()