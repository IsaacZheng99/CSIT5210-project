import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import itertools
import numba


class Component:
    def __init__(self, nodes, comp_id):
        self.nodes = set(nodes)  # nodes in this cluster
        self.comp_id = comp_id  # cluster ID


def merge_components(c_i, c_j):
    merged_list = c_i.nodes.union(c_j.nodes)
    return Component(merged_list, c_i.comp_id)


def get_reach_dists(D, min_points, num_points):
    """calculate the mutual reachability distance

    Args:
        D (_type_): initial matrix (eg. Euclidean distance matrix)
        min_points (_type_): value of miu
        num_points (_type_): number of total points

    Returns:
        _type_: reachability distance matrix
    """
    # for each point, get the top miu distances (including the point itself)
    reach_dists = np.sort(D, axis=1)
    topmiu_dists = reach_dists[:, min_points - 1]

    # get the row-wise and the column-wise matrices of the top miu distances
    topmiu_dists_i, topmiu_dists_j = np.meshgrid(topmiu_dists, topmiu_dists)

    # mutual reachability distance = max(D_ij, core_i, core_j)
    # core distance is the miuth shortest distance connected to the point
    D = np.stack([D, topmiu_dists_i, topmiu_dists_j], axis=-1)
    D = np.max(D, axis=-1)

    # each node has an distance of 0 to itself
    diag_mask = np.ones([num_points, num_points]) - np.eye(num_points)
    D *= diag_mask
    return D


def get_dc_dist_matrix(points, n_neighbors, min_points=5, **kwargs):
    """calculate the dc-dist

    Args:
        points (_type_): all data points
        n_neighbors (_type_): _description_
        min_points (int, optional): value of miu

    Returns:
        _type_: dc-dist matrix
    """
    """
    We define the distance from x_i to x_j as min(max(P(x_i, x_j))), where 
        - P(x_i, x_j) is any path from x_i to x_j
        - max(P(x_i, x_j)) is the largest edge weight in the path
        - min(max(P(x_i, x_j))) is the smallest largest edge weight
    """
    # calcute the Euclidean distances and @numba.njit is for acceleration
    @numba.njit(fastmath=True, parallel=True)
    def get_dist_matrix(points, D, dim, num_points):
        for i in numba.prange(num_points):
            x = points[i]
            for j in range(i+1, num_points):
                y = points[j]
                dist = 0
                for d in range(dim):
                    dist += (x[d] - y[d]) ** 2
                dist = np.sqrt(dist)
                D[i, j] = dist
                D[j, i] = dist

        return D

    num_points = int(points.shape[0])
    dim = int(points.shape[1])
    density_connections = np.zeros([num_points, num_points])
    D = np.zeros([num_points, num_points])
    D = get_dist_matrix(points, D, dim, num_points)
    if min_points > 1:
        if min_points > num_points:
            raise ValueError('Min points cannot exceed the size of the dataset')
        D = get_reach_dists(D, min_points, num_points)

    # sort the reachability distances
    flat_D = np.reshape(D, [num_points * num_points])
    argsort_inds = np.argsort(flat_D)

    component_dict = {i: Component([i], i) for i in range(num_points)}
    neighbor_dists = [[] for i in range(num_points)]
    neighbor_inds = [[] for i in range(num_points)]
    max_comp_size = 1
    for index in argsort_inds:
        i = int(index / num_points)
        j = index % num_points
        if component_dict[i].comp_id != component_dict[j].comp_id:
            epsilon = D[i, j]
            # finally the pairwise value is maximum, which corresponds to the max process of minimax
            for node_i in component_dict[i].nodes:
                for node_j in component_dict[j].nodes:
                    density_connections[node_i, node_j] = epsilon
                    density_connections[node_j, node_i] = epsilon
            merged_component = merge_components(component_dict[i], component_dict[j])
            for node in merged_component.nodes:
                component_dict[node] = merged_component
            size_of_component = len(component_dict[i].nodes)
            if size_of_component > max_comp_size:
                max_comp_size = size_of_component
        if max_comp_size == num_points:
            break
    return np.array(density_connections)
